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


def _intify_currents_depth_keys(doc):
    """Convert doc['currents']['limits_by_depth'] string keys back to int.

    MongoDB/BSON documents only allow string keys, so depths are stored as
    strings (e.g. "1500"). regions.py and plotting.py both key/lookup
    limits_by_depth by int depth, so convert back on the way out.
    """
    if doc is None:
        return None
    cur = doc.get("currents")
    if isinstance(cur, dict) and isinstance(cur.get("limits_by_depth"), dict):
        cur["limits_by_depth"] = {
            int(k): v for k, v in cur["limits_by_depth"].items()
        }
    return doc


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
        doc = client["hurricanes"]["region_configs"].find_one(
            {"region": region_name}, {"_id": 0}
        )
        return _intify_currents_depth_keys(doc)
    except Exception as exc:
        logger.warning(f"MongoDB region_configs query failed for region '{region_name}': {exc}")
        return None


def apply_colorbar_overrides(region_name, region_dict):
    """Overlay the hurricanes.region_configs document onto a regions.py config dict.

    region_configs is the single source of truth for MongoDB-driven region
    config — extent, folder, name, eez, figure, variables, sea_surface_height,
    currents (incl. limits_by_depth), salinity_max, ocean_heat_content. It's
    seeded from regions.py via scripts/tools/seed_region_configs.py and kept
    current by targeted field updates from update_colorbar_limits.py (weekly
    live-data tuning) and colorbar_tuner.py (manual tuning) — both update this
    same document rather than a separate collection.

    Any top-level key present in the document fully replaces the corresponding
    key from regions.py. Returns region_dict unchanged if MongoDB has no
    document for this region.
    """
    region_dict = copy.deepcopy(region_dict)

    full_doc = fetch_region_config(region_name)
    if full_doc is not None:
        for key, value in full_doc.items():
            if key == "region":
                continue
            region_dict[key] = value
            logger.debug(f"[{region_name}] overriding {key} from MongoDB region_configs")

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
