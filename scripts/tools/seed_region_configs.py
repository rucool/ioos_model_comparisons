#!/usr/bin/env python3
"""
Seed MongoDB region_configs collection from regions.py defaults.

Unlike seed_colorbar_configs.py (which extracts only the colorbar-limit
sub-fields), this writes the *entire* region_config() output — extent,
folder, name, eez, figure, variables, sea_surface_height, currents,
salinity_max, ocean_heat_content — so scripts can read the full region
definition from MongoDB via apply_colorbar_overrides() / fetch_region_config()
without redeploying code to change it.

hurricanes.colorbar_configs (the weekly-tuned color limits) is untouched and
continues to be applied on top of this, so tuned limits still win.

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


def _build_doc(region_key):
    """Build a full MongoDB document for *region_key* from its regions.py config."""
    cfg = dict(region_config(region_key))
    doc = {"region": region_key}
    doc.update(cfg)

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

        result = collection.replace_one({"region": key}, doc, upsert=True)
        action = "inserted" if result.upserted_id else "updated"
        logger.info(f"{action}: {key}")
        seeded += 1

    logger.info(
        f"Done — {seeded}/{len(ALL_REGIONS)} regions seeded in "
        "hurricanes.region_configs"
    )


if __name__ == "__main__":
    main()
