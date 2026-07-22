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


def fetch_region_config(region_name):
    """Return the full region-config document for *region_name*, or None.

    Queries hurricanes.region_configs for {region: region_name}. This
    collection mirrors the complete output of regions.region_config() —
    extent, folder, name, eez, figure, variables, sea_surface_height,
    currents, salinity_max, ocean_heat_content — seeded/updated via
    scripts/tools/seed_region_configs.py. Returns the document dict
    (without _id) or None if unavailable / not yet seeded for this region.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        return client["hurricanes"]["region_configs"].find_one(
            {"region": region_name}, {"_id": 0}
        )
    except Exception as exc:
        logger.warning(f"MongoDB region_configs query failed for region '{region_name}': {exc}")
        return None


def apply_colorbar_overrides(region_name, region_dict):
    """Overlay MongoDB region config onto a regions.py config dict.

    Two layers are applied on top of *region_dict*, in order:

    1. hurricanes.region_configs — the full region definition (extent,
       folder, name, eez, figure, variables, sea_surface_height, currents,
       salinity_max, ocean_heat_content). Any top-level key present in the
       document fully replaces the corresponding key from regions.py.

    2. hurricanes.colorbar_configs — narrower, faster-moving colorbar-limit
       overrides written by the weekly update_colorbar_limits.py cron /
       colorbar_tuner.py. Applied last so tuned limits always win over the
       (possibly stale) values baked into a region_configs document:
         - variables.temperature, variables.salinity  — replace full depth list
         - sea_surface_height                          — replace full depth list
         - ocean_heat_content.limits                   — replace limits only
         - salinity_max.limits                         — replace limits only
         - currents.limits                             — replace limits only
                                                         (bool/coarsen/kwargs kept)

    Returns region_dict unchanged if MongoDB has no documents for this region.
    """
    region_dict = copy.deepcopy(region_dict)

    full_doc = fetch_region_config(region_name)
    if full_doc is not None:
        for key, value in full_doc.items():
            if key == "region":
                continue
            region_dict[key] = value
            logger.debug(f"[{region_name}] overriding {key} from MongoDB region_configs")

    doc = fetch_colorbar_config(region_name)
    if doc is None:
        return region_dict

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


def log_plots(script, records, has_argo=False, has_gliders=False,
              db_name=_PLOTS_DB, coll_name=_PLOTS_COLL):
    """Batch-upsert a list of plot records into MongoDB.

    Each record is a dict with keys: region, timestamp, plot_type, variable,
    depth, model1, model2.  has_argo / has_gliders reflect whether those
    platform data sources were available during this script run — used by the
    pre-check to decide whether to replot when a previously-missing source
    comes back online.  Silently skips if MongoDB is unavailable.
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
                pymongo.UpdateOne(
                    filt,
                    {"$set": {"plotted_at": now, "has_argo": has_argo, "has_gliders": has_gliders}},
                    upsert=True,
                )
            )
        client[db_name][coll_name].bulk_write(ops, ordered=False)
        logger.debug(f"Logged {len(ops)} plot record(s) to MongoDB (argo={has_argo}, gliders={has_gliders})")
    except Exception as exc:
        logger.warning(f"log_plots failed: {exc}")


def fetch_completed_plot_keys(script, timestamps, db_name=_PLOTS_DB, coll_name=_PLOTS_COLL):
    """Return a dict mapping completed-plot key tuples to their platform flags.

    Key tuple: (region, iso_ts, plot_type, variable, depth, model1, model2).
    Value:      {"has_argo": bool, "has_gliders": bool}

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
             "variable": 1, "depth": 1, "model1": 1, "model2": 1,
             "has_argo": 1, "has_gliders": 1},
        )
        completed = {}
        for doc in cursor:
            key = (
                doc["region"],
                doc["timestamp"].strftime("%Y-%m-%dT%H%M%SZ") if hasattr(doc["timestamp"], "strftime") else doc["timestamp"],
                doc["plot_type"],
                doc["variable"],
                doc["depth"],
                doc["model1"],
                doc["model2"],
            )
            completed[key] = {
                "has_argo":    doc.get("has_argo",    True),
                "has_gliders": doc.get("has_gliders", True),
            }
        logger.debug(f"fetch_completed_plot_keys: {len(completed)} done records found")
        return completed
    except Exception as exc:
        logger.warning(f"fetch_completed_plot_keys failed ({exc}) — will fall back to file check")
        return None


def needs_replot(key, completed, current_has_argo, current_has_gliders):
    """Return True if the plot identified by *key* should be generated this run.

    A plot needs (re)generation when:
      - It has never been recorded in MongoDB, OR
      - It was previously plotted without Argo data and Argo is now available, OR
      - It was previously plotted without glider data and gliders are now available.

    Records that pre-date the has_argo / has_gliders fields default to True
    (assume data was present) to avoid spurious replotting of old records.
    """
    if key not in completed:
        return True
    rec = completed[key]
    if current_has_argo and not rec.get("has_argo", True):
        return True
    if current_has_gliders and not rec.get("has_gliders", True):
        return True
    return False
