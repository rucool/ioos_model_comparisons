#!/usr/bin/env python3
"""
Seed MongoDB region_configs collection from regions.py defaults.

This writes most of the region_config() output — extent, folder, name, eez,
figure, currents, salinity_max, ocean_heat_content — so scripts can read the
full region definition from MongoDB via apply_colorbar_overrides() /
fetch_region_config() without redeploying code to change it.

region_configs is the single source of truth, with different fields owned by
different writers: update_colorbar_limits.py (weekly live-data tuning) and
colorbar_tuner.py (manual tuning) both write variables.temperature/salinity
and sea_surface_height directly into this same collection/document. This
script deliberately EXCLUDES those two fields from what it writes (and uses
a partial $set, not a full replace) so re-running it to pick up an unrelated
regions.py edit — a new region, a currents.limits_by_depth tweak, whatever —
never clobbers live-tuned colorbar limits back to regions.py's static
defaults. A brand-new, never-tuned region simply falls back to regions.py's
values for those two fields until a tuning run sets them.

Usage:
    python scripts/tools/seed_region_configs.py

Requires MONGODB_URI environment variable to be set.
Writes one document per region to hurricanes.region_configs (upsert).
Re-run at any time to pick up new regions or edits made in regions.py.
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

ALL_REGIONS = [
    "mastr",
    "yucatan",
    "leeward",
    "loop_current",
    "gom",
    "gom_east",
    "gom_west",
    "east_coast",
    "sab",
    "mab",
    "west_florida_shelf",
    "caribbean",
    "windward",
    "amazon",
    "hurricane",
    "tropical_western_atlantic",
    "passengers",
    "mexico_pacific",
    "hawaii",
    "wmo_v_south",
    "bahamas",
    "ru29",
    "philippines",
    "guam",
    "fiji",
    "south_africa",
]


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
    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI environment variable is not set")
        sys.exit(1)

    import pymongo

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    collection = client["hurricanes"]["region_configs"]
    collection.create_index("region", unique=True)

    seeded = 0
    for key in ALL_REGIONS:
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
        f"Done — {seeded}/{len(ALL_REGIONS)} regions seeded in "
        "hurricanes.region_configs"
    )


if __name__ == "__main__":
    main()
