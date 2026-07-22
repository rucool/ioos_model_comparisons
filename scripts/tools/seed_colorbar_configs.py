#!/usr/bin/env python3
"""
Seed MongoDB colorbar_configs collection from regions.py defaults.

Usage:
    python scripts/tools/seed_colorbar_configs.py

Requires MONGODB_URI environment variable to be set.
Writes one document per region to hurricanes.colorbar_configs (upsert).
Re-run at any time to pick up new regions added to regions.py.
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
]


def _extract_colorbar_doc(region_key):
    """Build a MongoDB document for *region_key* from its regions.py config."""
    cfg = region_config(region_key)
    doc = {"region": region_key}

    # variables: temperature and salinity depth lists
    if "variables" in cfg:
        vars_doc = {}
        for var_key, var_list in cfg["variables"].items():
            entries = [
                {"depth": e["depth"], "limits": e["limits"]}
                for e in (var_list or [])
                if "limits" in e
            ]
            if entries:
                vars_doc[var_key] = entries
        if vars_doc:
            doc["variables"] = vars_doc

    # sea_surface_height depth list
    ssh = cfg.get("sea_surface_height") or []
    ssh_entries = [
        {"depth": e["depth"], "limits": e["limits"]}
        for e in ssh
        if "limits" in e
    ]
    if ssh_entries:
        doc["sea_surface_height"] = ssh_entries

    # ocean_heat_content and salinity_max — limits sub-key only
    for top_key in ("ocean_heat_content", "salinity_max"):
        val = cfg.get(top_key)
        if isinstance(val, dict) and "limits" in val:
            doc[top_key] = {"limits": val["limits"]}

    # currents — limits / limits_by_depth sub-keys only
    cur = cfg.get("currents")
    if isinstance(cur, dict) and ("limits" in cur or "limits_by_depth" in cur):
        cur_doc = {}
        if "limits" in cur:
            cur_doc["limits"] = cur["limits"]
        if "limits_by_depth" in cur:
            # BSON only allows string keys; db.py converts back to int on read.
            cur_doc["limits_by_depth"] = {str(k): v for k, v in cur["limits_by_depth"].items()}
        doc["currents"] = cur_doc

    return doc


def main():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI environment variable is not set")
        sys.exit(1)

    import pymongo

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    collection = client["hurricanes"]["colorbar_configs"]
    collection.create_index("region", unique=True)

    seeded = 0
    for key in ALL_REGIONS:
        try:
            doc = _extract_colorbar_doc(key)
        except Exception as exc:
            logger.warning(f"Skipping '{key}': {exc}")
            continue

        result = collection.replace_one({"region": key}, doc, upsert=True)
        action = "inserted" if result.upserted_id else "updated"
        logger.info(f"{action}: {key}")
        seeded += 1

    logger.info(
        f"Done — {seeded}/{len(ALL_REGIONS)} regions seeded in "
        "hurricanes.colorbar_configs"
    )


if __name__ == "__main__":
    main()
