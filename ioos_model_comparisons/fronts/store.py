"""
store.py — versioned MongoDB storage for digitized fronts.

MongoDB is the store of record for the wall, the rings and the map overlays;
the digitizer also keeps writing GeoJSON/PNG to outputs/ as working copies.
The website and the next day's QC read from here, which is how a hand edit
made in the browser reaches them.

Three collections in the `hurricanes` database, all APPEND-ONLY:

    front_walls     one document per VERSION of a day's wall
    front_rings     one document per VERSION of a day's whole ring SET
    front_overlays  the SST/SLA basemap PNGs (see below)

Why versioned rather than updated in place: the browser editor can only
usefully hand you ~200 draggable vertices, so a hand-edited wall is a
simplified wall. If saving overwrote the automatic geometry, the ~3000-vertex
original would be gone. Instead the automatic run writes version 1 and an edit
writes version 2 with `parent_version: 1`; version 1 is a separate document
that is never touched. The old file-based design needed an auto_backup/ folder
and a "never overwrite a backup" rule to get the same guarantee — here the
storage model enforces it.

CURRENT is defined by exactly one rule: the highest `version` for
(region, stamp). No is_current flag, nothing to drift out of sync.

Overlays live in Mongo because the digitizer and the website run on different
hosts, and Mongo is the only channel that already spans both. A ~1 MB PNG is
far under the 16 MB document limit, so no GridFS is involved.

Convention, inherited from db.py: MONGODB_URI only, pymongo imported lazily
inside functions, and EVERY failure is soft — log a warning and return
None/[]/False, never raise. Callers degrade to files. (Note that
ioos_model_comparisons/users.py deliberately does NOT follow this, because for
authentication "degrade" would mean letting people in without a database.)
"""

from __future__ import annotations

import datetime
import logging
import re

import numpy as np

from ioos_model_comparisons.db import get_client

logger = logging.getLogger(__name__)

DB_NAME = "hurricanes"
WALLS_COLL = "front_walls"
RINGS_COLL = "front_rings"
OVERLAYS_COLL = "front_overlays"
DEFAULT_REGION = "gulf_stream"

STAMP_RE = re.compile(r"^\d{8}T\d{4}$")


def _utcnow():
    """Naive UTC now.

    datetime.utcnow() is deprecated in 3.12 but naive-UTC is what db.py
    already stores, so keep the same storage semantics without the warning.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def stamp_to_time(stamp):
    """'20260816T1055' -> naive UTC datetime, or None if malformed."""
    if not STAMP_RE.match(str(stamp or "")):
        return None
    return datetime.datetime.strptime(stamp, "%Y%m%dT%H%M")


def stamp_to_day(stamp):
    """'20260816T1055' -> '2026-08-16', or None if malformed."""
    t = stamp_to_time(stamp)
    return t.strftime("%Y-%m-%d") if t else None


def lines_to_geometry(lines):
    """[(N,2) lon/lat arrays] -> a GeoJSON LineString/MultiLineString dict."""
    coords = [[[float(x), float(y)] for x, y in np.asarray(l, float)]
              for l in lines if len(np.asarray(l)) >= 2]
    if not coords:
        return None
    if len(coords) == 1:
        return {"type": "LineString", "coordinates": coords[0]}
    return {"type": "MultiLineString", "coordinates": coords}


def geometry_to_lines(geometry):
    """Inverse of lines_to_geometry(); always returns a list of (N,2) arrays."""
    if not geometry:
        return []
    g = geometry
    coords = ([g["coordinates"]] if g.get("type") == "LineString"
              else g.get("coordinates", []))
    return [np.asarray(c, float) for c in coords]


def _count_vertices(geometry):
    return int(sum(len(l) for l in geometry_to_lines(geometry)))


def _coll(name, db_name=DB_NAME):
    client = get_client()
    if client is None:
        return None
    return client[db_name][name]


# ---------------------------------------------------------------------------
# indexes
# ---------------------------------------------------------------------------
def ensure_front_indexes(db_name=DB_NAME):
    """Create the front collections' indexes (idempotent, safe to call often)."""
    client = get_client()
    if client is None:
        return
    try:
        import pymongo
        db = client[db_name]
        for coll in (WALLS_COLL, RINGS_COLL):
            # unique: turns a concurrent double-insert into a retryable error
            # rather than two documents claiming the same version
            db[coll].create_index(
                [("region", pymongo.ASCENDING), ("stamp", pymongo.ASCENDING),
                 ("version", pymongo.DESCENDING)],
                unique=True, background=True)
            # prior-day lookup, used by the digitizer's QC every night
            db[coll].create_index(
                [("region", pymongo.ASCENDING), ("day", pymongo.DESCENDING),
                 ("version", pymongo.DESCENDING)],
                background=True)
        db[OVERLAYS_COLL].create_index(
            [("region", pymongo.ASCENDING), ("stamp", pymongo.ASCENDING),
             ("field", pymongo.ASCENDING)],
            unique=True, background=True)
        logger.debug("front indexes ensured")
    except Exception as exc:
        logger.warning(f"ensure_front_indexes failed: {exc}")


# ---------------------------------------------------------------------------
# versioned insert
# ---------------------------------------------------------------------------
def _current_doc(coll, region, stamp, projection=None):
    return coll.find_one({"region": region, "stamp": stamp},
                         projection, sort=[("version", -1)])


def _insert_versioned(coll, region, stamp, doc, retries=3):
    """Insert `doc` at max(version)+1, retrying if someone races us.

    Concurrency here is one nightly cron and an occasional human, so a
    collision is near-impossible — but the unique index means the impossible
    case becomes a retry instead of a silently lost or duplicated version.
    """
    import pymongo
    for _ in range(retries):
        cur = _current_doc(coll, region, stamp, {"version": 1})
        version = int(cur["version"]) + 1 if cur else 1
        doc["version"] = version
        try:
            coll.insert_one(dict(doc))
            return version
        except pymongo.errors.DuplicateKeyError:
            continue
    logger.warning(f"could not allocate a version for {region}/{stamp} "
                   f"after {retries} attempts")
    return None


def _manual_version(coll, region, stamp):
    """Highest manual version for a stamp, or None."""
    d = coll.find_one({"region": region, "stamp": stamp, "origin": "manual"},
                      {"version": 1}, sort=[("version", -1)])
    return int(d["version"]) if d else None


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------
def save_wall_version(stamp, geometry, properties, *, region=DEFAULT_REGION,
                      origin="auto", resolution="full", source=None,
                      edited_by=None, parent_version=None, simplify=None,
                      note=None, qc_pass=None, qc_stale=False,
                      created_at=None, extra=None, skip_if_manual=False):
    """Append a new wall version. Returns the version int, or None."""
    coll = _coll(WALLS_COLL)
    if coll is None or not geometry:
        return None
    try:
        if skip_if_manual:
            mv = _manual_version(coll, region, stamp)
            if mv is not None:
                # Guard against a routine re-run silently superseding an hour
                # of hand editing under the "highest version wins" rule.
                logger.warning(
                    f"{stamp} already has a manual version (v{mv}) — not writing "
                    f"an automatic version. Pass --force-auto to override.")
                return None
        # A re-run that produces the same geometry should not manufacture a
        # new version: versions are meant to mark real changes, and the
        # nightly cron re-running a day would otherwise pile up identical
        # documents and make the history meaningless.
        cur = _current_doc(coll, region, stamp, {"geometry": 1, "origin": 1,
                                                 "version": 1})
        if (cur and origin == "auto" and cur.get("origin") == "auto"
                and cur.get("geometry") == geometry):
            logger.info(f"{stamp}: geometry identical to v{cur['version']} "
                        f"(auto) — not creating a duplicate version")
            return cur["version"]

        lines = geometry_to_lines(geometry)
        doc = {
            "region": region,
            "stamp": stamp,
            "time": stamp_to_time(stamp),
            "day": stamp_to_day(stamp),
            "origin": origin,
            "resolution": resolution,
            "source": source,
            "geometry": geometry,
            "properties": properties or {},
            "n_pieces": len(lines),
            "n_vertices": _count_vertices(geometry),
            "qc_pass": qc_pass if qc_pass is not None else (properties or {}).get("qc_pass"),
            "qc_stale": bool(qc_stale),
            "edited_by": edited_by,
            "parent_version": parent_version,
            "simplify": simplify,
            "note": note,
            "created_at": created_at or _utcnow(),
        }
        doc.update(extra or {})
        return _insert_versioned(coll, region, stamp, doc)
    except Exception as exc:
        logger.warning(f"save_wall_version failed for {stamp}: {exc}")
        return None


def save_rings_version(stamp, rings, *, region=DEFAULT_REGION, origin="auto",
                       edited_by=None, parent_version=None, source="CMEMS-SLA",
                       sla_time=None, sla_lag_days=None, created_at=None,
                       extra=None, skip_if_manual=False):
    """Append a new version of a day's whole ring set. Returns version or None.

    One document per SET, not per ring: the editor saves them atomically and
    match_eddies() consumes a set.
    """
    coll = _coll(RINGS_COLL)
    if coll is None:
        return None
    try:
        if skip_if_manual:
            mv = _manual_version(coll, region, stamp)
            if mv is not None:
                logger.warning(f"{stamp} rings already have a manual version "
                               f"(v{mv}) — not writing automatic rings.")
                return None
        doc = {
            "region": region,
            "stamp": stamp,
            "time": stamp_to_time(stamp),
            "day": stamp_to_day(stamp),
            "origin": origin,
            "source": source,
            "sla_time": sla_time,
            "sla_lag_days": sla_lag_days,
            "rings": list(rings or []),
            "n_rings": len(rings or []),
            "edited_by": edited_by,
            "parent_version": parent_version,
            "created_at": created_at or _utcnow(),
        }
        doc.update(extra or {})
        return _insert_versioned(coll, region, stamp, doc)
    except Exception as exc:
        logger.warning(f"save_rings_version failed for {stamp}: {exc}")
        return None


def save_overlay(stamp, field, png_bytes, meta=None, *, region=DEFAULT_REGION):
    """Upsert one basemap PNG. Returns True on success.

    Overlays are NOT versioned — they are a rendering of the source data for
    that scene, not an interpretation of it, so there is nothing to preserve
    across a re-run.
    """
    coll = _coll(OVERLAYS_COLL)
    if coll is None or not png_bytes:
        return False
    try:
        import bson
        doc = dict(meta or {})
        doc.update({
            "png": bson.Binary(bytes(png_bytes)),
            "bytes": len(png_bytes),
            "time": stamp_to_time(stamp),
            "day": stamp_to_day(stamp),
            "created_at": _utcnow(),
        })
        coll.update_one({"region": region, "stamp": stamp, "field": field},
                        {"$set": doc}, upsert=True)
        return True
    except Exception as exc:
        logger.warning(f"save_overlay failed for {stamp}/{field}: {exc}")
        return False


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def fetch_wall(stamp, *, region=DEFAULT_REGION, version=None, origin=None):
    """The current wall for a stamp, or a specific version/origin. None on failure."""
    coll = _coll(WALLS_COLL)
    if coll is None:
        return None
    try:
        filt = {"region": region, "stamp": stamp}
        if version is not None:
            filt["version"] = int(version)
        if origin is not None:
            filt["origin"] = origin
        return coll.find_one(filt, {"_id": 0}, sort=[("version", -1)])
    except Exception as exc:
        logger.warning(f"fetch_wall failed for {stamp}: {exc}")
        return None


def fetch_wall_lines(stamp, **kw):
    """(lines, properties) — deliberately the same shape digitizer.read_front
    returns, so a call site differs only in where the data came from."""
    doc = fetch_wall(stamp, **kw)
    if doc is None:
        return None
    return geometry_to_lines(doc.get("geometry")), doc.get("properties", {})


def fetch_wall_lines_for_day(day, *, region=DEFAULT_REGION):
    """(lines, stamp) for the CURRENT wall on this calendar day, or (None, None).

    `day` is 'YYYY-MM-DD'. Unlike fetch_wall_lines(), the caller doesn't need
    to know the exact digitize stamp (e.g. '20260828T1055') — just the date,
    which is what a map script processing a ctime actually has on hand.
    """
    coll = _coll(WALLS_COLL)
    if coll is None:
        return None, None
    try:
        doc = coll.find_one({"region": region, "day": day}, {"_id": 0},
                            sort=[("stamp", -1), ("version", -1)])
        if doc is None:
            return None, None
        return geometry_to_lines(doc.get("geometry")), doc.get("stamp")
    except Exception as exc:
        logger.warning(f"fetch_wall_lines_for_day failed for {day}: {exc}")
        return None, None


def fetch_isotherm_lines_for_day(base_region, day, *, level=15, ref_depth=200):
    """{model_name: lines} for every model with a saved 200m isotherm on this
    calendar day, for `base_region` (a regions.py folder name, e.g.
    "mid_atlantic_bight").

    plotting.py's _save_isotherm_lines() writes one region tag per (map
    region, model) pair -- "{base_region}_{model}_isotherm{level}c_{depth}m"
    -- so different models' isotherms for the same day are separate
    documents rather than versions of one. This scans for all such tags and
    returns the CURRENT line set for each. Mongo-only; there is no on-disk
    fallback for these the way the digitized wall has.
    """
    coll = _coll(WALLS_COLL)
    if coll is None:
        return {}
    try:
        suffix = f"_isotherm{level}c_{ref_depth}m"
        prefix = f"{base_region}_"
        pattern = f"^{re.escape(prefix)}.*{re.escape(suffix)}$"
        regions = coll.distinct("region", {"region": {"$regex": pattern}, "day": day})

        out = {}
        for region in regions:
            doc = coll.find_one({"region": region, "day": day}, {"_id": 0},
                                sort=[("stamp", -1), ("version", -1)])
            if doc is None:
                continue
            lines = geometry_to_lines(doc.get("geometry"))
            if not lines:
                continue
            model = (doc.get("properties") or {}).get("model") \
                or region[len(prefix):-len(suffix)]
            out[model] = lines
        return out
    except Exception as exc:
        logger.warning(f"fetch_isotherm_lines_for_day failed for {base_region}/{day}: {exc}")
        return {}


def fetch_rings(stamp, *, region=DEFAULT_REGION, version=None):
    coll = _coll(RINGS_COLL)
    if coll is None:
        return None
    try:
        filt = {"region": region, "stamp": stamp}
        if version is not None:
            filt["version"] = int(version)
        return coll.find_one(filt, {"_id": 0}, sort=[("version", -1)])
    except Exception as exc:
        logger.warning(f"fetch_rings failed for {stamp}: {exc}")
        return None


def normalize_rings(rings):
    """Return ring dicts in the shape detect_eddies/read_eddies produce.

    Stored rings carry the GeoJSON property names (centroid_lon/centroid_lat)
    because that is what eddies_to_geojson writes, but match_eddies() indexes
    r["lon"]/r["lat"]. Without this, chaining days_tracked off a Mongo prior
    raises KeyError: 'lon'. Normalising here rather than at each call site
    keeps "rings from Mongo" and "rings from a file" interchangeable.
    """
    out = []
    for r in rings or []:
        d = dict(r)
        if "lon" not in d and "centroid_lon" in d:
            d["lon"] = d["centroid_lon"]
        if "lat" not in d and "centroid_lat" in d:
            d["lat"] = d["centroid_lat"]
        if "poly" not in d and isinstance(d.get("geometry"), dict):
            coords = d["geometry"].get("coordinates") or [[]]
            d["poly"] = np.asarray(coords[0], float)
        out.append(d)
    return out


def _fetch_prior(coll_name, before_day, region):
    coll = _coll(coll_name)
    if coll is None or not before_day:
        return None
    try:
        return coll.find_one({"region": region, "day": {"$lt": before_day}},
                             {"_id": 0},
                             sort=[("day", -1), ("version", -1)])
    except Exception as exc:
        logger.warning(f"prior lookup failed in {coll_name}: {exc}")
        return None


def fetch_prior_wall(before_day, *, region=DEFAULT_REGION):
    """Current wall from the most recent day strictly before `before_day`.

    This is what makes a hand edit propagate: an edit writes a higher version
    for that day, and the (day desc, version desc) sort returns it, so the
    next morning's displacement QC is measured against what the human drew.
    """
    return _fetch_prior(WALLS_COLL, before_day, region)


def fetch_prior_rings(before_day, *, region=DEFAULT_REGION):
    return _fetch_prior(RINGS_COLL, before_day, region)


def fetch_overlay(stamp, field, *, region=DEFAULT_REGION):
    """(png_bytes, meta) or None."""
    coll = _coll(OVERLAYS_COLL)
    if coll is None:
        return None
    try:
        doc = coll.find_one({"region": region, "stamp": stamp, "field": field},
                            {"_id": 0})
        if not doc or not doc.get("png"):
            return None
        png = bytes(doc.pop("png"))
        return png, doc
    except Exception as exc:
        logger.warning(f"fetch_overlay failed for {stamp}/{field}: {exc}")
        return None


def list_wall_days(*, region=DEFAULT_REGION, limit=400):
    """Metadata for each day's CURRENT wall, newest first.

    Aggregated with $first after $sort so geometry is never materialized —
    the days list must not drag ~3000 vertices per day over the wire.
    """
    coll = _coll(WALLS_COLL)
    if coll is None:
        return []
    try:
        pipeline = [
            {"$match": {"region": region}},
            {"$sort": {"stamp": -1, "version": -1}},
            {"$group": {
                "_id": "$stamp",
                "version": {"$first": "$version"},
                "time": {"$first": "$time"},
                "day": {"$first": "$day"},
                "origin": {"$first": "$origin"},
                "resolution": {"$first": "$resolution"},
                "qc_pass": {"$first": "$qc_pass"},
                "qc_stale": {"$first": "$qc_stale"},
                "n_pieces": {"$first": "$n_pieces"},
                "n_vertices": {"$first": "$n_vertices"},
                "edited_by": {"$first": "$edited_by"},
                "created_at": {"$first": "$created_at"},
                "n_versions": {"$sum": 1},
            }},
            {"$sort": {"_id": -1}},
            {"$limit": int(limit)},
        ]
        out = []
        for d in coll.aggregate(pipeline):
            d["stamp"] = d.pop("_id")
            out.append(d)
        return out
    except Exception as exc:
        logger.warning(f"list_wall_days failed: {exc}")
        return []


def list_wall_versions(stamp, *, region=DEFAULT_REGION):
    """Version history for one day, newest first, without geometry."""
    coll = _coll(WALLS_COLL)
    if coll is None:
        return []
    try:
        cur = coll.find({"region": region, "stamp": stamp},
                        {"_id": 0, "geometry": 0, "properties": 0},
                        sort=[("version", -1)])
        return list(cur)
    except Exception as exc:
        logger.warning(f"list_wall_versions failed for {stamp}: {exc}")
        return []


def wall_exists(stamp, *, region=DEFAULT_REGION):
    """True if any version exists — used by the backfill for idempotency."""
    coll = _coll(WALLS_COLL)
    if coll is None:
        return False
    try:
        return coll.find_one({"region": region, "stamp": stamp}, {"_id": 1}) is not None
    except Exception as exc:
        logger.warning(f"wall_exists failed for {stamp}: {exc}")
        return False
