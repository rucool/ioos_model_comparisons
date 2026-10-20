"""
catalog.py — the figure catalog: one small MongoDB document per published
plot (metadata + URL only; the PNG itself stays on the file server).

Write side (plot scripts):

    from ioos_model_comparisons import catalog
    catalog.set_run_context(script="rtofs-gofs-cmems-amseas.py")
    fig.savefig(path)
    catalog.register_figure(path)                      # maps: all metadata from the path
    catalog.register_figure(path, platform={"lat": 26.1, "lon": -88.2})   # profiles

register_figure() NEVER raises and never slows a run down more than once:
  - figures outside the published plots root (site.yml publish_root, or
    $IMC_PLOTS_ROOT) are ignored, so laptop runs can't pollute production;
  - MongoDB unavailable -> returns None (db.get_client() contract), and after
    one failed write the catalog stops trying for 10 minutes;
  - CATALOG_DISABLED=1 turns it off entirely.
Anything missed is picked up by scripts/tools/index_figures.py, which walks the
plot tree on a schedule.

Read side (portal): find_for_slot, latest_valid_time, platform_figures,
latest_platform_day, platform_positions, find_one, freshness.

Document (_id is the path relative to the plots root):
    family, product, region (folder), variable, depth, var_depth, valid_time,
    models, comparison, rtofs {stream: version}, platform {type, id, lat, lon},
    url, source {script, via}, has_argo, has_gliders, status (present|missing),
    size, file_mtime, created_at, updated_at
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_BREAKER_SECONDS = 600
_breaker_until = 0.0
_warned: set = set()
_ctx: dict = {}

INDEXES = [
    ([("region", 1), ("var_depth", 1), ("valid_time", -1)], "slot"),
    ([("comparison", 1), ("region", 1), ("var_depth", 1), ("valid_time", -1)], "comparison_slot"),
    ([("family", 1), ("valid_time", -1)], "family_time"),
    ([("platform.type", 1), ("valid_time", -1)], "platform_time"),
    ([("platform.type", 1), ("platform.id", 1), ("valid_time", -1)], "platform_id_time"),
    ([("created_at", -1)], "created"),
]


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _registry():
    from ioos_model_comparisons.registry import load_registry
    return load_registry()


def _collection():
    from ioos_model_comparisons.db import get_client
    client = get_client()
    if client is None:
        return None
    cfg = _registry().site.catalog
    return client[cfg.database][cfg.collection]


def _disabled() -> bool:
    return os.getenv("CATALOG_DISABLED", "").strip().lower() in ("1", "true", "yes")


def _trip_breaker(exc):
    global _breaker_until
    _breaker_until = time.monotonic() + _BREAKER_SECONDS
    _warn_once(("breaker", type(exc).__name__),
               "figure catalog write failed (%s); skipping catalog writes for %d min",
               exc, _BREAKER_SECONDS // 60)


def _breaker_open() -> bool:
    return time.monotonic() < _breaker_until


def set_run_context(*, script=None, has_argo=None, has_gliders=None):
    """Values recorded on every figure this process registers."""
    if script is not None:
        _ctx["script"] = script
    if has_argo is not None:
        _ctx["has_argo"] = bool(has_argo)
    if has_gliders is not None:
        _ctx["has_gliders"] = bool(has_gliders)


# --------------------------------------------------------------------------
# Building documents
# --------------------------------------------------------------------------
def normalize_path(path) -> Path:
    """matplotlib appends .png to a path with no extension; mirror that."""
    p = Path(path)
    if p.suffix == "":
        p = p.with_name(p.name + ".png")
    return p


def relative_to_root(path, root=None):
    """Path relative to the published plots root as a posix string, or None."""
    root = Path(root) if root is not None else _registry().publish_root()
    p = Path(path)
    for a, b in ((os.path.abspath(p), os.path.abspath(root)),
                 (os.path.realpath(p), os.path.realpath(root))):
        try:
            return Path(a).relative_to(b).as_posix()
        except ValueError:
            continue
    return None


def build_fields(rel, *, meta=None, platform=None):
    """Catalog fields for a relative path, or None if the path isn't a known
    product and no explicit meta was given."""
    from ioos_model_comparisons import figure_names

    parsed = figure_names.parse(rel)
    if parsed is None and not meta:
        return None
    fields = figure_names.to_doc(parsed) if parsed else {}
    if meta:
        fields.update(meta)
    if platform:
        plat = dict(fields.get("platform") or {})
        for k, v in platform.items():
            if v is None:
                continue
            if k == "lon":
                v = ((float(v) + 180) % 360) - 180
            elif k == "lat":
                v = float(v)
            plat[k] = v
        fields["platform"] = plat
    reg = _registry()
    if fields.get("models"):
        tag = reg.rtofs_tag_for_models(fields["models"], fields.get("valid_time"))
        if tag:
            fields["rtofs"] = tag
    fields["url"] = reg.public_url(rel)
    return fields


def _file_stat(path):
    try:
        st = os.stat(path)
        return {"size": st.st_size,
                "file_mtime": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc)
                .replace(tzinfo=None, microsecond=0)}
    except OSError:
        return {}


def register_figure(path, *, meta=None, platform=None, script=None):
    """Upsert the catalog entry for a figure that was just saved.

    Returns the stored fields, or None when skipped (outside the published
    root, unrecognized name, disabled, MongoDB unavailable). Never raises.
    """
    try:
        if _disabled() or _breaker_open():
            return None
        p = normalize_path(path)
        rel = relative_to_root(p)
        if rel is None:
            return None
        fields = build_fields(rel, meta=meta, platform=platform)
        if fields is None:
            key = ("unparsed", "/".join(rel.split("/")[:2]))
            if key not in _warned:
                _warned.add(key)
                logger.info("figure catalog: %s is not a portal product (not cataloged; "
                            "add a pattern to figure_names.py if it should be)", rel)
            return None
        coll = _collection()
        if coll is None:
            return None
        now = _utcnow()
        fields.update(_file_stat(p))
        fields.update({k: v for k, v in _ctx.items() if k in ("has_argo", "has_gliders")})
        fields["source"] = {"script": script or _ctx.get("script") or Path(sys.argv[0]).name,
                            "via": "hook"}
        fields["status"] = "present"
        fields["updated_at"] = now
        coll.update_one({"_id": rel}, {"$set": fields, "$setOnInsert": {"created_at": now}},
                        upsert=True)
        return fields
    except Exception as exc:  # never break a plotting run
        _trip_breaker(exc)
        return None


def register_figures(items, *, via="indexer"):
    """Bulk upsert for the indexer. items: iterable of dicts with
    rel (str), stat ({size, file_mtime}) and optional platform.

    Path-derived metadata goes in $setOnInsert, so fields a hook wrote with
    more context (script, platform position, has_argo) are never overwritten;
    only size/mtime/status/updated_at are refreshed. Returns docs written."""
    from pymongo import UpdateOne

    coll = _collection()
    if coll is None:
        return 0
    now = _utcnow()
    ops = []
    for it in items:
        fields = build_fields(it["rel"])
        if fields is None:
            continue
        fields["source"] = {"script": None, "via": via}
        fields["created_at"] = now
        refresh = dict(it.get("stat") or {})
        refresh.update({"status": "present", "updated_at": now})
        ops.append(UpdateOne({"_id": it["rel"]}, {"$set": refresh, "$setOnInsert": fields},
                             upsert=True))
    if not ops:
        return 0
    coll.bulk_write(ops, ordered=False)
    return len(ops)


def set_platform_positions(positions):
    """positions: {rel: {lat, lon[, time]}} — fill platform.lat/lon where the
    hook didn't already record them (locations.json backfill)."""
    from pymongo import UpdateOne

    coll = _collection()
    if coll is None or not positions:
        return 0
    ops = []
    for rel, pos in positions.items():
        try:
            lat = float(pos["lat"])
            lon = ((float(pos["lon"]) + 180) % 360) - 180
        except (KeyError, TypeError, ValueError):
            continue
        ops.append(UpdateOne({"_id": rel, "platform.lat": {"$exists": False}},
                             {"$set": {"platform.lat": lat, "platform.lon": lon}}))
    if ops:
        coll.bulk_write(ops, ordered=False)
    return len(ops)


def ensure_catalog_indexes():
    coll = _collection()
    if coll is None:
        return False
    for keys, name in INDEXES:
        coll.create_index(keys, name=name, background=True)
    return True


# --------------------------------------------------------------------------
# Read side (portal)
# --------------------------------------------------------------------------
_PRESENT = {"status": {"$ne": "missing"}}


def find_for_slot(region, var_depth, valid_time) -> dict:
    """{comparison: doc} for one region folder / var_depth / valid time."""
    coll = _collection()
    if coll is None:
        raise RuntimeError("catalog unavailable")
    q = {"region": region, "var_depth": var_depth, "valid_time": valid_time, **_PRESENT}
    out = {}
    for d in coll.find(q, {"comparison": 1, "url": 1, "rtofs": 1, "product": 1}):
        out.setdefault(d.get("comparison"), d)
    return out


def latest_valid_time(region, var_depth, comparisons=None):
    coll = _collection()
    if coll is None:
        raise RuntimeError("catalog unavailable")
    q = {"region": region, "var_depth": var_depth, **_PRESENT}
    if comparisons:
        q["comparison"] = {"$in": list(comparisons)}
    d = coll.find_one(q, {"valid_time": 1}, sort=[("valid_time", -1)])
    return d["valid_time"] if d else None


def platform_figures(ptype, start, end, regions=None, extra=None) -> list:
    """Profile figures for ptype with start <= valid_time < end."""
    coll = _collection()
    if coll is None:
        raise RuntimeError("catalog unavailable")
    q = {"platform.type": ptype, "valid_time": {"$gte": start, "$lt": end}, **_PRESENT}
    if regions:
        q["region"] = {"$in": list(regions)}
    if extra:
        q.update(extra)
    return list(coll.find(q).sort("valid_time", 1))


def latest_platform_day(ptype, regions=None, extra=None):
    coll = _collection()
    if coll is None:
        raise RuntimeError("catalog unavailable")
    q = {"platform.type": ptype, **_PRESENT}
    if regions:
        q["region"] = {"$in": list(regions)}
    if extra:
        q.update(extra)
    d = coll.find_one(q, {"valid_time": 1}, sort=[("valid_time", -1)])
    return d["valid_time"].date() if d else None


def find_one(query, sort=None):
    coll = _collection()
    if coll is None:
        raise RuntimeError("catalog unavailable")
    return coll.find_one({**query, **_PRESENT}, sort=sort)


def freshness() -> dict:
    """{"latest_created_at": datetime|None, "count_24h": int} — None if unavailable."""
    coll = _collection()
    if coll is None:
        return None
    d = coll.find_one({}, {"created_at": 1}, sort=[("created_at", -1)])
    since = _utcnow() - _dt.timedelta(hours=24)
    return {"latest_created_at": d["created_at"] if d else None,
            "count_24h": coll.count_documents({"created_at": {"$gte": since}})}


def is_stale(fresh=None) -> bool:
    fresh = fresh if fresh is not None else freshness()
    if not fresh or not fresh.get("latest_created_at"):
        return True
    hours = _registry().site.catalog.stale_after_hours
    return _utcnow() - fresh["latest_created_at"] > _dt.timedelta(hours=hours)
