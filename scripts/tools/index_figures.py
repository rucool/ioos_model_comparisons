#!/usr/bin/env python
"""
Index published plots into the figure catalog (MongoDB hurricanes.figures).

Plot scripts register figures as they save them (catalog.register_figure);
this walks the plot tree to backfill the archive and to catch anything a
script didn't register (no hook yet, or MongoDB was down during the run).
Only metadata goes into MongoDB — the PNGs stay where they are.

    # 1. See what would be indexed (no database writes):
    python scripts/tools/index_figures.py --survey
    # 2. One-time backfill of the whole archive:
    python scripts/tools/index_figures.py --full
    # 3. Cron (scripts/shell/index_figures.sh): every 30 min, recent months only
    python scripts/tools/index_figures.py --recent-months 2
    #    and weekly, flag catalog entries whose file is gone:
    python scripts/tools/index_figures.py --mark-missing

Also merges profile positions from the locations.json files the profile
scripts write, for figures whose hook didn't record lat/lon.
"""

import argparse
import collections
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from ioos_model_comparisons import catalog, figure_names
from ioos_model_comparisons.env import load_env
from ioos_model_comparisons.registry import load_registry

log = logging.getLogger("index_figures")

TOP_DIRS = ("maps", "profiles", "adaptive_sampling_guidance")
_YEAR = re.compile(r"^\d{4}$")
_MONTH = re.compile(r"^\d{2}$")
_MONTHDAY = re.compile(r"^(\d{2})-\d{2}$")


def _month_index(y, m):
    return int(y) * 12 + int(m) - 1


def walk(root: Path, top_dirs=TOP_DIRS, min_month=None):
    """Yield (rel, abs_path, is_locations_json) under root. Skips symlinks;
    descends into last_14_days/ only for locations.json. With min_month
    (year*12+month-1), prunes dated directories older than that."""
    def _rec(path: Path, rel: str, year=None, month_done=False):
        try:
            entries = list(os.scandir(path))
        except OSError as e:
            log.warning("cannot list %s: %s", path, e)
            return
        for e in entries:
            if e.is_symlink():
                continue
            r = f"{rel}/{e.name}" if rel else e.name
            if e.is_dir(follow_symlinks=False):
                if e.name == "last_14_days":
                    lj = os.path.join(e.path, "locations.json")
                    if os.path.isfile(lj):
                        yield f"{r}/locations.json", lj, True
                    continue
                y, done = year, month_done
                if min_month is not None:
                    if year is None and _YEAR.match(e.name):
                        if int(e.name) * 12 + 11 < min_month:
                            continue
                        y, done = e.name, False
                    elif year and not month_done and (_MONTH.match(e.name) or _MONTHDAY.match(e.name)):
                        if _month_index(year, e.name[:2]) < min_month:
                            continue
                        done = True
                yield from _rec(Path(e.path), r, y, done)
            elif e.name.endswith(".png"):
                yield r, e.path, False
            elif e.name == "locations.json":
                yield r, e.path, True

    for top in top_dirs:
        p = root / top
        if p.is_dir():
            yield from _rec(p, top)


def positions_from_locations(rel_json: str, data: dict) -> dict:
    """{figure rel path: {lat, lon}} from one locations.json."""
    out = {}
    base = rel_json.rsplit("/", 1)[0]
    in_rolling = base.endswith("/last_14_days")
    for fname, entry in (data or {}).items():
        if not isinstance(entry, dict):
            continue
        if not in_rolling:
            rel = f"{base}/{fname}"
        else:
            # Rolling dirs hold symlinks; point at the dated file instead.
            parent = base[: -len("/last_14_days")]
            m = re.search(r"-profile-(\d{4})-(\d{2})-(\d{2})T", fname)
            g = re.search(r"_(\d{4})(\d{2})(\d{2})_to_\d{8}_", fname)
            if m:
                rel = f"{parent}/{m[1]}/{m[2]}/{m[3]}/{fname}"
            elif g:
                rel = f"{parent}/{g[1]}/{g[2]}-{g[3]}/{fname}"
            else:
                continue
        if "lat" in entry and "lon" in entry:
            out[rel] = {"lat": entry["lat"], "lon": entry["lon"]}
    return out


def _stat(path):
    st = os.stat(path)
    return {"size": st.st_size,
            "file_mtime": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
            .replace(tzinfo=None, microsecond=0)}


def survey(root):
    parsed, unparsed, samples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for rel, _p, is_json in walk(root):
        if is_json:
            continue
        meta = figure_names.parse(rel)
        prefix = "/".join(rel.split("/")[:2])
        if meta:
            parsed[meta.product] += 1
        else:
            unparsed[prefix] += 1
            if len(samples[prefix]) < 3:
                samples[prefix].append(rel)
    total_p, total_u = sum(parsed.values()), sum(unparsed.values())
    print(f"parsed {total_p}, unparsed {total_u} under {root}")
    for k, v in parsed.most_common():
        print(f"  {v:9d}  {k}")
    print("unparsed by prefix (not shown on the portal unless a pattern is added):")
    for k, v in unparsed.most_common(40):
        print(f"  {v:9d}  {k}    e.g. {samples[k][0]}")


def index(root, min_month=None, batch=1000, dry_run=False):
    coll = catalog._collection()
    if coll is None and not dry_run:
        sys.exit("MongoDB unavailable (MONGODB_URI unset or unreachable)")
    if not dry_run:
        catalog.ensure_catalog_indexes()

    stats = collections.Counter()
    pending, positions = [], {}

    def flush():
        if not pending:
            return
        ids = [it["rel"] for it in pending]
        have = {} if dry_run else {
            d["_id"]: d for d in coll.find({"_id": {"$in": ids}},
                                           {"size": 1, "file_mtime": 1, "status": 1})}
        todo = [it for it in pending
                if it["rel"] not in have
                or have[it["rel"]].get("size") != it["stat"]["size"]
                or have[it["rel"]].get("file_mtime") != it["stat"]["file_mtime"]
                or have[it["rel"]].get("status") == "missing"]
        stats["unchanged"] += len(pending) - len(todo)
        if todo and not dry_run:
            stats["written"] += catalog.register_figures(todo)
        elif todo:
            stats["would_write"] += len(todo)
        pending.clear()

    for rel, path, is_json in walk(root, min_month=min_month):
        if is_json:
            try:
                with open(path) as f:
                    positions.update(positions_from_locations(rel, json.load(f)))
            except Exception as e:
                log.warning("bad %s: %s", rel, e)
            continue
        if figure_names.parse(rel) is None:
            stats["unparsed"] += 1
            continue
        try:
            pending.append({"rel": rel, "stat": _stat(path)})
        except OSError:
            continue
        stats["seen"] += 1
        if len(pending) >= batch:
            flush()
    flush()

    if positions and not dry_run:
        stats["positions"] = catalog.set_platform_positions(positions)
    else:
        stats["positions"] = len(positions)
    return stats


def mark_missing(root, batch=1000, dry_run=False):
    """Flag catalog entries whose file no longer exists (never deletes)."""
    coll = catalog._collection()
    if coll is None:
        sys.exit("MongoDB unavailable")
    gone, checked = [], 0
    for d in coll.find({"status": {"$ne": "missing"}}, {"_id": 1}):
        checked += 1
        if not (root / d["_id"]).exists():
            gone.append(d["_id"])
    if gone and not dry_run:
        for i in range(0, len(gone), batch):
            coll.update_many({"_id": {"$in": gone[i:i + batch]}},
                             {"$set": {"status": "missing", "updated_at": catalog._utcnow()}})
    return {"checked": checked, "missing": len(gone)}


def retag_rtofs(start, end, dry_run=False):
    """Recompute rtofs version tags (after editing rtofs.yml history)."""
    from pymongo import UpdateOne
    coll = catalog._collection()
    if coll is None:
        sys.exit("MongoDB unavailable")
    reg = load_registry()
    ops = []
    q = {"valid_time": {"$gte": start, "$lt": end}, "models": {"$exists": True, "$ne": []}}
    for d in coll.find(q, {"models": 1, "valid_time": 1, "rtofs": 1}):
        tag = reg.rtofs_tag_for_models(d["models"], d["valid_time"])
        if tag != d.get("rtofs"):
            ops.append(UpdateOne({"_id": d["_id"]},
                                 {"$set": {"rtofs": tag}} if tag else {"$unset": {"rtofs": ""}}))
    if ops and not dry_run:
        coll.bulk_write(ops, ordered=False)
    return {"retagged": len(ops)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="plots root (default: site.yml publish_root / $IMC_PLOTS_ROOT)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--survey", action="store_true", help="count parsed/unparsed files; no DB")
    mode.add_argument("--full", action="store_true", help="index the whole tree")
    mode.add_argument("--recent-months", type=int, metavar="N",
                      help="only dated directories from the last N months (default 2)")
    mode.add_argument("--mark-missing", action="store_true",
                      help="flag entries whose file is gone (status: missing)")
    mode.add_argument("--retag-rtofs", nargs=2, metavar=("START", "END"),
                      help="recompute RTOFS version tags for valid times in [START, END)")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env()
    root = Path(os.path.expanduser(args.root)) if args.root else load_registry().publish_root()
    if args.root:
        # register_figures derives relative paths itself; keep URLs consistent
        os.environ.setdefault("IMC_PLOTS_ROOT", str(root))
    if not root.is_dir():
        sys.exit(f"plots root not found: {root}")

    t0 = time.time()
    if args.survey:
        survey(root)
        return
    if args.mark_missing:
        result = mark_missing(root, args.batch, args.dry_run)
    elif args.retag_rtofs:
        start, end = (dt.datetime.fromisoformat(s) for s in args.retag_rtofs)
        result = retag_rtofs(start, end, args.dry_run)
    else:
        min_month = None
        if not args.full:
            n = args.recent_months or 2
            today = dt.date.today()
            min_month = _month_index(today.year, today.month) - (n - 1)
        result = dict(index(root, min_month, args.batch, args.dry_run))
    log.info("done in %.1fs: %s", time.time() - t0, result)


if __name__ == "__main__":
    main()
