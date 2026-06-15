import copy
import datetime
import logging
import os

logger = logging.getLogger(__name__)

_client = None
_client_tried = False


def _get_client():
    """Lazy-init a MongoDB client from MONGODB_URI. Returns None on failure."""
    global _client, _client_tried
    if _client_tried:
        return _client
    _client_tried = True

    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.warning(
            "MONGODB_URI not set — colorbar limits will use regions.py defaults"
        )
        return None

    try:
        import pymongo

        _client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")
        logger.info("MongoDB connected — colorbar overrides active")
    except Exception as exc:
        logger.warning(
            f"MongoDB connection failed ({exc}) — colorbar limits will use regions.py defaults"
        )
        _client = None

    return _client


def fetch_colorbar_config(region_name):
    """Return the colorbar config document for *region_name*, or None.

    Queries hurricanes.colorbar_configs for {region: region_name}.
    Returns the document dict (without _id) or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        return client["hurricanes"]["colorbar_configs"].find_one(
            {"region": region_name}, {"_id": 0}
        )
    except Exception as exc:
        logger.warning(f"MongoDB query failed for region '{region_name}': {exc}")
        return None


def apply_colorbar_overrides(region_name, region_dict):
    """Overlay MongoDB colorbar limits onto a region config dict.

    Keys overridden when present in the DB document:
      - variables.temperature, variables.salinity  — replace full depth list
      - sea_surface_height                          — replace full depth list
      - ocean_heat_content.limits                   — replace limits only
      - salinity_max.limits                         — replace limits only
      - currents.limits                             — replace limits only
                                                      (bool/coarsen/kwargs kept
                                                      from regions.py)

    Returns region_dict unchanged if no DB document exists for this region.
    """
    doc = fetch_colorbar_config(region_name)
    if doc is None:
        return region_dict

    region_dict = copy.deepcopy(region_dict)

    # variables (temperature / salinity depth lists)
    if "variables" in doc:
        for var_key, var_list in doc["variables"].items():
            if "variables" not in region_dict:
                region_dict["variables"] = {}
            region_dict["variables"][var_key] = var_list
            logger.debug(f"[{region_name}] overriding variables.{var_key} from MongoDB")

    # sea_surface_height depth list
    if "sea_surface_height" in doc:
        region_dict["sea_surface_height"] = doc["sea_surface_height"]
        logger.debug(f"[{region_name}] overriding sea_surface_height from MongoDB")

    # ocean_heat_content — only the limits sub-key
    if "ocean_heat_content" in doc:
        db_ohc = doc["ocean_heat_content"]
        if "limits" in db_ohc:
            if not isinstance(region_dict.get("ocean_heat_content"), dict):
                region_dict["ocean_heat_content"] = {}
            region_dict["ocean_heat_content"]["limits"] = db_ohc["limits"]
            logger.debug(
                f"[{region_name}] overriding ocean_heat_content.limits from MongoDB"
            )

    # salinity_max — only the limits sub-key
    if "salinity_max" in doc:
        db_sm = doc["salinity_max"]
        if "limits" in db_sm:
            if not isinstance(region_dict.get("salinity_max"), dict):
                region_dict["salinity_max"] = {}
            region_dict["salinity_max"]["limits"] = db_sm["limits"]
            logger.debug(f"[{region_name}] overriding salinity_max.limits from MongoDB")

    # currents — only the limits sub-key; leave bool/coarsen/kwargs intact
    if "currents" in doc:
        db_cur = doc["currents"]
        if "limits" in db_cur and isinstance(region_dict.get("currents"), dict):
            region_dict["currents"]["limits"] = db_cur["limits"]
            logger.debug(f"[{region_name}] overriding currents.limits from MongoDB")

    return region_dict


_PLOTS_DB   = "hurricanes"
_PLOTS_COLL = "model_plots"
_INDEX_KEYS = ["script", "region", "timestamp", "plot_type", "variable", "depth", "model1", "model2"]


def ensure_plot_index(db_name=_PLOTS_DB, coll_name=_PLOTS_COLL):
    """Create the unique compound index on the plots collection (idempotent)."""
    client = _get_client()
    if client is None:
        return
    try:
        import pymongo
        coll = client[db_name][coll_name]
        coll.create_index(
            [(k, pymongo.ASCENDING) for k in _INDEX_KEYS],
            unique=True,
            background=True,
        )
        logger.debug(f"Plot index ensured on {db_name}.{coll_name}")
    except Exception as exc:
        logger.warning(f"ensure_plot_index failed: {exc}")


def log_plots(script, records, db_name=_PLOTS_DB, coll_name=_PLOTS_COLL):
    """Batch-upsert a list of plot records into MongoDB.

    Each record is a dict with keys: region, timestamp, plot_type, variable,
    depth, model1, model2.  Silently skips if MongoDB is unavailable.
    """
    if not records:
        return
    client = _get_client()
    if client is None:
        return
    try:
        import pymongo
        now = datetime.datetime.utcnow()
        ops = []
        for rec in records:
            filt = {k: rec[k] for k in _INDEX_KEYS if k != "script"}
            filt["script"] = script
            filt["timestamp"] = rec["timestamp"]
            ops.append(
                pymongo.UpdateOne(filt, {"$set": {"plotted_at": now}}, upsert=True)
            )
        client[db_name][coll_name].bulk_write(ops, ordered=False)
        logger.debug(f"Logged {len(ops)} plot record(s) to MongoDB")
    except Exception as exc:
        logger.warning(f"log_plots failed: {exc}")


def fetch_completed_plot_keys(script, timestamps, db_name=_PLOTS_DB, coll_name=_PLOTS_COLL):
    """Return a frozenset of completed-plot key tuples for the given timestamps.

    Each tuple is (region, iso_ts, plot_type, variable, depth, model1, model2).
    Returns None if MongoDB is unavailable, signalling the caller to fall back
    to file-existence checks.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        ts_list = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in timestamps]
        cursor = client[db_name][coll_name].find(
            {"script": script, "timestamp": {"$in": ts_list}},
            {"_id": 0, "region": 1, "timestamp": 1, "plot_type": 1,
             "variable": 1, "depth": 1, "model1": 1, "model2": 1},
        )
        keys = frozenset(
            (
                doc["region"],
                doc["timestamp"].strftime("%Y-%m-%dT%H%M%SZ") if hasattr(doc["timestamp"], "strftime") else doc["timestamp"],
                doc["plot_type"],
                doc["variable"],
                doc["depth"],
                doc["model1"],
                doc["model2"],
            )
            for doc in cursor
        )
        logger.debug(f"fetch_completed_plot_keys: {len(keys)} done records found")
        return keys
    except Exception as exc:
        logger.warning(f"fetch_completed_plot_keys failed ({exc}) — will fall back to file check")
        return None
