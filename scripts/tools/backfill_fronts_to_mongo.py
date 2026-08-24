#!/usr/bin/env python3
"""
backfill_fronts_to_mongo.py — load existing digitized fronts into MongoDB.

MongoDB is the store of record for the wall, rings and map overlays (see
ioos_model_comparisons/fronts/store.py). This loads the GeoJSON/PNG files that
scripts/fronts/run_digitizer_goes19.py already wrote to outputs/ into it.

It is idempotent: a stamp that already has a wall version is skipped unless
--force. That means it doubles as the catch-up path for a night when the
digitizer ran but MongoDB was unreachable — just run it again afterwards.

Requires MONGODB_URI. The production Mongo host is not reachable from a
laptop; either run this on the server, or open your SSH tunnel and point
MONGODB_URI at localhost.

Usage
-----
    python scripts/tools/backfill_fronts_to_mongo.py --dry-run
    python scripts/tools/backfill_fronts_to_mongo.py
    python scripts/tools/backfill_fronts_to_mongo.py --stamp 20260816T1055 --force
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ioos_model_comparisons.fronts import DEFAULT_OUTPUT_DIR      # noqa: E402
from ioos_model_comparisons.fronts import store                    # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WALL_STEM = "gulf_stream_north_wall"
EDDY_STEM = "gulf_stream_eddies"
OVERLAY_STEM = "gulf_stream_overlay"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                   help="digitizer output directory (default: outputs/gulf_stream_fronts)")
    p.add_argument("--stamp", action="append", default=None,
                   help="only this stamp (repeatable); default is every stamp found")
    p.add_argument("--region", default=store.DEFAULT_REGION)
    p.add_argument("--force", action="store_true",
                   help="write even if the stamp already has a version in Mongo")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be written, touch nothing")
    p.add_argument("--no-overlays", action="store_true",
                   help="skip the basemap PNGs (geometry only)")
    return p.parse_args()


def _mtime(path):
    """File mtime as a naive UTC datetime.

    Used instead of utcnow() so the audit trail reflects when the wall was
    actually produced rather than when it happened to be imported.
    """
    return datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc).replace(tzinfo=None)


def _wall_feature(fc):
    """(geometry, properties) of the wall feature in a FeatureCollection."""
    for feat in fc.get("features", []):
        if feat.get("properties", {}).get("feature") == "gulf_stream_north_wall":
            return feat.get("geometry"), feat.get("properties", {})
    return None, None


def _rings_from(fc):
    """The ring list in the shape store/eddies use, from a rings FeatureCollection."""
    rings = []
    for feat in fc.get("features", []):
        p = feat.get("properties", {}) or {}
        if not str(p.get("feature", "")).endswith("_core_ring"):
            continue
        r = {k: p.get(k) for k in
             ("feature", "kind", "radius_km", "amplitude_cm", "compactness",
              "centroid_lon", "centroid_lat", "days_tracked", "edited_by_hand")}
        r["geometry"] = feat.get("geometry")
        rings.append(r)
    return rings


def main():
    args = parse_args()

    if not os.getenv("MONGODB_URI"):
        logger.error("MONGODB_URI is not set — nothing to write to. "
                     "Run on the server, or open the SSH tunnel and point "
                     "MONGODB_URI at localhost.")
        sys.exit(1)

    out_dir = args.out_dir
    if not out_dir.is_dir():
        logger.error(f"{out_dir} does not exist")
        sys.exit(1)

    if not args.dry_run:
        store.ensure_front_indexes()

    backup_dir = out_dir / "auto_backup"
    stamps = sorted({p.stem.rsplit("_", 1)[-1]
                     for p in out_dir.glob(f"{WALL_STEM}_*.geojson")
                     if store.STAMP_RE.match(p.stem.rsplit("_", 1)[-1])})
    if args.stamp:
        stamps = [s for s in stamps if s in set(args.stamp)]
    logger.info(f"{len(stamps)} stamp(s) found in {out_dir}")

    n_wall = n_rings = n_overlay = n_skip = 0
    problems = []

    for stamp in stamps:
        if not args.force and store.wall_exists(stamp, region=args.region):
            n_skip += 1
            continue

        wall_path = out_dir / f"{WALL_STEM}_{stamp}.geojson"
        try:
            geom, props = _wall_feature(json.loads(wall_path.read_text()))
        except Exception as exc:
            problems.append(f"{stamp}: unreadable wall ({exc})")
            continue
        if geom is None:
            problems.append(f"{stamp}: no wall feature in {wall_path.name}")
            continue

        # If a hand edit was made with the old file-based editor, the live file
        # is the (simplified) edit and auto_backup/ holds the full-resolution
        # original. Preserve both, in the right order, so history is honest.
        backup = backup_dir / wall_path.name
        versions = []
        if backup.is_file():
            try:
                bgeom, bprops = _wall_feature(json.loads(backup.read_text()))
                if bgeom is not None:
                    versions.append(dict(geometry=bgeom, properties=bprops,
                                         origin="auto", resolution="full",
                                         created_at=_mtime(backup), note=None))
            except Exception as exc:
                problems.append(f"{stamp}: unreadable backup ({exc})")
            versions.append(dict(
                geometry=geom, properties=props, origin="manual",
                resolution="simplified", created_at=_mtime(wall_path),
                note="backfilled hand edit; editor unknown"))
        else:
            versions.append(dict(geometry=geom, properties=props,
                                 origin="auto", resolution="full",
                                 created_at=_mtime(wall_path), note=None))

        if args.dry_run:
            logger.info(f"{stamp}: would write {len(versions)} wall version(s) "
                        f"({', '.join(v['origin'] for v in versions)})")
        else:
            parent = None
            for v in versions:
                ver = store.save_wall_version(
                    stamp, v["geometry"], v["properties"], region=args.region,
                    origin=v["origin"], resolution=v["resolution"],
                    source=(v["properties"] or {}).get("source"),
                    parent_version=parent, note=v["note"],
                    created_at=v["created_at"], extra={"backfilled": True})
                if ver is None:
                    problems.append(f"{stamp}: wall version write failed")
                    break
                parent = ver
                n_wall += 1

        # ---- rings ----
        eddy_path = out_dir / f"{EDDY_STEM}_{stamp}.geojson"
        if eddy_path.is_file():
            try:
                rings = _rings_from(json.loads(eddy_path.read_text()))
                first = next(iter(rings), {})
                if args.dry_run:
                    logger.info(f"{stamp}: would write {len(rings)} ring(s)")
                elif store.save_rings_version(
                        stamp, rings, region=args.region, origin="auto",
                        created_at=_mtime(eddy_path),
                        extra={"backfilled": True}) is not None:
                    n_rings += 1
            except Exception as exc:
                problems.append(f"{stamp}: rings failed ({exc})")

        # ---- overlays ----
        if not args.no_overlays:
            side = out_dir / f"{OVERLAY_STEM}_{stamp}.json"
            meta_all = {}
            if side.is_file():
                try:
                    meta_all = json.loads(side.read_text()).get("fields", {})
                except Exception:
                    meta_all = {}
            for field in ("sst", "sla"):
                png = out_dir / f"{OVERLAY_STEM}_{field}_{stamp}.png"
                if not png.is_file():
                    continue
                if args.dry_run:
                    logger.info(f"{stamp}: would write {field} overlay "
                                f"({png.stat().st_size/1e6:.2f} MB)")
                    continue
                meta = dict(meta_all.get(field, {}))
                meta.update(region=args.region, stamp=stamp, field=field)
                if store.save_overlay(stamp, field, png.read_bytes(), meta,
                                      region=args.region):
                    n_overlay += 1

    logger.info(f"done — {n_wall} wall version(s), {n_rings} ring set(s), "
                f"{n_overlay} overlay(s) written; {n_skip} stamp(s) already present")
    for p in problems:
        logger.warning(p)
    if problems:
        logger.warning(f"{len(problems)} problem(s) above did not stop the run")


if __name__ == "__main__":
    main()
