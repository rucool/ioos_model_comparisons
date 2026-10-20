#!/usr/bin/env python3
"""
DEPRECATED — region defaults now live in ioos_model_comparisons/config/regions/*.yml
and are read directly by region_config(). MongoDB region_configs should hold
only deliberate overrides (see scripts/tools/unseed_region_overrides.py);
copying the defaults into MongoDB, as this script did, HIDES later YAML edits
because apply_colorbar_overrides() replaces whole top-level keys.

Kept for the rare case where you really want to force a region's current YAML
values into MongoDB as overrides:

    python scripts/tools/seed_region_configs.py --force [region ...]

It still excludes variables/sea_surface_height (owned by the colorbar tuners)
and uses a partial $set.
"""
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure the package is importable when run from any working directory
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

from ioos_model_comparisons.regions import region_config  # noqa: E402
from ioos_model_comparisons.region_catalog import ALL_REGIONS  # noqa: E402


# Fields owned by update_colorbar_limits.py / colorbar_tuner.py — never
# overwritten by this script once a region has been tuned.
_TUNED_FIELDS = {"variables", "sea_surface_height"}


def _build_doc(region_key):
    """Build a partial MongoDB $set payload for *region_key* from regions.py,
    excluding the fields owned by the live/manual colorbar tuning tools."""
    cfg = dict(region_config(region_key))
    doc = {k: v for k, v in cfg.items() if k not in _TUNED_FIELDS}

    # BSON only allows string keys — currents.limits_by_depth is keyed by int
    # depth in regions.py, so stringify it for storage (db.py converts back
    # to int on read).
    cur = doc.get("currents")
    if isinstance(cur, dict) and isinstance(cur.get("limits_by_depth"), dict):
        cur["limits_by_depth"] = {str(k): v for k, v in cur["limits_by_depth"].items()}

    return doc


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="really copy YAML defaults into MongoDB")
    ap.add_argument("regions", nargs="*", help="limit to these region keys")
    args = ap.parse_args()
    if not args.force:
        print(__doc__)
        sys.exit(2)
    regions = args.regions or ALL_REGIONS

    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI environment variable is not set")
        sys.exit(1)

    import pymongo

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    collection = client["hurricanes"]["region_configs"]
    collection.create_index("region", unique=True)

    seeded = 0
    for key in regions:
        try:
            doc = _build_doc(key)
        except Exception as exc:
            logger.warning(f"Skipping '{key}': {exc}")
            continue

        result = collection.update_one({"region": key}, {"$set": doc}, upsert=True)
        action = "inserted" if result.upserted_id else "updated"
        logger.info(f"{action}: {key}")
        seeded += 1

    logger.info(
        f"Done — {seeded}/{len(regions)} regions seeded in "
        "hurricanes.region_configs"
    )


if __name__ == "__main__":
    main()
