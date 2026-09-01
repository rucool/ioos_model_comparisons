"""
regions_store.py — validated, versioned CRUD for hurricanes.region_configs.

`region_configs` holds ONE live document per region and stays exactly as it is
today, because db.apply_colorbar_overrides reads it and every plotting script
depends on that. This module adds:

    hurricanes.region_config_versions   append-only history

so an edit can be diffed and undone. The live document remains authoritative
for readers; history is a parallel record, not the source of truth. (That is
the opposite of fronts/store.py, where highest-version *is* current — here the
existing readers pin the shape, so do not "unify" the two.)

Three traps this module exists to prevent, all of which corrupt configs in
ways that only surface at plot time:

1. `currents.limits_by_depth` is keyed by int depth in regions.py but MUST be
   stored with string keys (BSON allows nothing else). db.py int-ifies on
   read. Writing int keys here silently breaks that round-trip.
2. apply_colorbar_overrides replaces WHOLE top-level keys. Saving a partial
   `variables` object drops the variables you left out.
3. Nothing validates ranges today, so `vmin > vmax` or a flipped extent is
   accepted and produces blank or nonsense plots with no error.

Every failure is soft in the db.py sense EXCEPT validation, which raises
ValueError — a caller must not be able to write an invalid config by ignoring
a return value.
"""

from __future__ import annotations

import copy
import datetime
import logging

from ioos_model_comparisons.db import get_client

logger = logging.getLogger(__name__)

DB_NAME = "hurricanes"
LIVE_COLL = "region_configs"
VERSIONS_COLL = "region_config_versions"

# Top-level keys apply_colorbar_overrides understands. Anything else is
# allowed through (the schema has irregular corners) but is reported by
# validate() so a typo like "extant" is visible rather than silently inert.
KNOWN_KEYS = {
    "region", "name", "folder", "extent", "eez", "figure", "variables",
    "sea_surface_height", "currents", "salinity_max", "ocean_heat_content",
}


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _coll(name):
    client = get_client()
    return None if client is None else client[DB_NAME][name]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_limits(where, lim, errors):
    """limits are [vmin, vmax, stride]."""
    if not isinstance(lim, (list, tuple)) or len(lim) != 3:
        errors.append(f"{where}: limits must be [vmin, vmax, stride], got {lim!r}")
        return
    vmin, vmax, stride = lim
    if not all(_num(x) for x in lim):
        errors.append(f"{where}: limits must be numbers, got {lim!r}")
        return
    if vmin >= vmax:
        errors.append(f"{where}: vmin ({vmin}) must be less than vmax ({vmax})")
    if stride <= 0:
        errors.append(f"{where}: stride must be positive, got {stride}")
    elif (vmax - vmin) / stride > 500:
        # a stride this fine makes matplotlib draw thousands of contour levels
        # and the plot takes minutes or dies
        errors.append(f"{where}: {(vmax-vmin)/stride:.0f} contour levels "
                      f"(vmax-vmin)/stride — almost certainly a typo")


def _check_var_list(where, items, errors):
    if items in (None, []):
        return
    if not isinstance(items, list):
        errors.append(f"{where}: expected a list of {{depth, limits}}, got {type(items).__name__}")
        return
    for i, d in enumerate(items):
        if not isinstance(d, dict):
            errors.append(f"{where}[{i}]: expected an object, got {d!r}")
            continue
        if "depth" in d and not _num(d["depth"]):
            errors.append(f"{where}[{i}].depth: must be a number, got {d['depth']!r}")
        if "limits" in d:
            _check_limits(f"{where}[{i}].limits", d["limits"], errors)


def validate(doc):
    """Return (errors, warnings) for a region document. Never raises."""
    errors, warnings = [], []
    if not isinstance(doc, dict):
        return ["document must be an object"], []

    region = doc.get("region")
    if not region or not isinstance(region, str) or not region.strip():
        errors.append("region: a non-empty key is required")

    for k in ("name", "folder"):
        if k in doc and (not isinstance(doc[k], str) or not doc[k].strip()):
            errors.append(f"{k}: must be a non-empty string")

    ext = doc.get("extent")
    if ext is not None:
        if not isinstance(ext, (list, tuple)) or len(ext) != 4 or not all(_num(v) for v in ext):
            errors.append("extent: must be [lon_min, lon_max, lat_min, lat_max] numbers")
        else:
            lon0, lon1, lat0, lat1 = ext
            if lon0 >= lon1:
                # NOT an error: regions.py contains [145.0, -120.0, ...], a
                # real Pacific box running east from 145E across the
                # antimeridian to 120W. Longitude wraps; latitude does not.
                # Warn so an accidental flip still gets a human's attention.
                warnings.append(
                    f"extent: lon_min ({lon0}) is greater than lon_max ({lon1}) — "
                    f"read as crossing the antimeridian. If that is not what you "
                    f"meant, the two are swapped.")
            if lat0 >= lat1:
                errors.append(f"extent: lat_min ({lat0}) must be less than lat_max ({lat1})")
            # -180..360, not -180..180: several Pacific regions in regions.py
            # use the 0-360 convention and reach 190.25, so the tighter rule
            # would reject a valid production config.
            if not (-180 <= lon0 <= 360 and -180 <= lon1 <= 360):
                errors.append("extent: longitudes must be within -180..360")
            if not (-90 <= lat0 <= 90 and -90 <= lat1 <= 90):
                errors.append("extent: latitudes must be within -90..90")
            # Swapped lon/lat pairs pass every rule above — [31.75, 45.25,
            # -77.25, -49.75] is "valid" and plots the wrong hemisphere. It
            # cannot be proven wrong from the numbers alone, so warn on the
            # shapes a swap tends to produce. The extent mini-map in the
            # editor is the real defence: a swapped box is obvious on sight.
            # 60 deg, derived from the data: every extent in regions.py sits
            # within -50..45.25 lat, so this cannot fire on a real region but
            # does catch a lon/lat swap of a mid-latitude box.
            if abs(lat0) > 60 or abs(lat1) > 60:
                warnings.append(
                    f"extent: latitudes {lat0}..{lat1} are near-polar, which is "
                    f"unusual here — are lon/lat swapped? (order is "
                    f"[lon_min, lon_max, lat_min, lat_max])")
            if (lon1 - lon0) > 200 or (lat1 - lat0) > 100:
                warnings.append(
                    f"extent: spans {lon1-lon0:.1f}deg lon x {lat1-lat0:.1f}deg lat "
                    f"— larger than any current region; check the ordering")

    if "eez" in doc and not isinstance(doc["eez"], bool):
        errors.append("eez: must be true or false")

    variables = doc.get("variables")
    if variables is not None:
        if not isinstance(variables, dict):
            errors.append("variables: must be an object keyed by variable name")
        else:
            for var, items in variables.items():
                _check_var_list(f"variables.{var}", items, errors)
    _check_var_list("sea_surface_height", doc.get("sea_surface_height"), errors)

    cur = doc.get("currents")
    if cur is not None:
        if not isinstance(cur, dict):
            errors.append("currents: must be an object")
        else:
            if "bool" in cur and not isinstance(cur["bool"], bool):
                errors.append("currents.bool: must be true or false")
            lbd = cur.get("limits_by_depth")
            if lbd is not None:
                if not isinstance(lbd, dict):
                    errors.append("currents.limits_by_depth: must be an object keyed by depth")
                else:
                    for k, lim in lbd.items():
                        try:
                            int(k)
                        except (TypeError, ValueError):
                            errors.append(f"currents.limits_by_depth: key {k!r} is not a depth")
                        _check_limits(f"currents.limits_by_depth[{k}]", lim, errors)

    fig = doc.get("figure")
    if isinstance(fig, dict) and "figsize" in fig:
        fs = fig["figsize"]
        if not isinstance(fs, (list, tuple)) or len(fs) != 2 or not all(
                _num(v) and v > 0 for v in fs):
            errors.append("figure.figsize: must be [width, height], both positive")

    for k in doc:
        if k not in KNOWN_KEYS and not k.startswith("_"):
            warnings.append(f"{k}: not a key apply_colorbar_overrides knows about — "
                            f"it will be stored but ignored by plotting")
    return errors, warnings


def normalize(doc):
    """Coerce a document into what MongoDB and db.py expect.

    Chiefly: stringify currents.limits_by_depth keys, and turn tuples into
    lists (BSON has no tuple, and a round-trip would change the type anyway).
    """
    doc = copy.deepcopy(doc)

    def _listify(v):
        if isinstance(v, tuple):
            return [_listify(x) for x in v]
        if isinstance(v, list):
            return [_listify(x) for x in v]
        if isinstance(v, dict):
            return {k: _listify(x) for k, x in v.items()}
        return v

    doc = _listify(doc)
    cur = doc.get("currents")
    if isinstance(cur, dict) and isinstance(cur.get("limits_by_depth"), dict):
        # int keys are what regions.py uses; BSON needs strings and db.py
        # converts back on read
        cur["limits_by_depth"] = {str(int(k)): v
                                  for k, v in cur["limits_by_depth"].items()}
    return doc


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def list_regions():
    """[{region, name, folder, n_versions, updated_at, updated_by}] sorted."""
    live = _coll(LIVE_COLL)
    if live is None:
        return []
    try:
        out = []
        vers = _coll(VERSIONS_COLL)
        counts = {}
        if vers is not None:
            for row in vers.aggregate([{"$group": {"_id": "$region",
                                                   "n": {"$sum": 1},
                                                   "last": {"$max": "$created_at"},
                                                   }}]):
                counts[row["_id"]] = row
        for d in live.find({}, {"region": 1, "name": 1, "folder": 1, "extent": 1}):
            c = counts.get(d.get("region"), {})
            out.append({"region": d.get("region"), "name": d.get("name"),
                        "folder": d.get("folder"), "extent": d.get("extent"),
                        "n_versions": c.get("n", 0), "updated_at": c.get("last")})
        return sorted(out, key=lambda r: r["region"] or "")
    except Exception as exc:
        logger.warning(f"list_regions failed: {exc}")
        return []


def fetch(region):
    live = _coll(LIVE_COLL)
    if live is None:
        return None
    try:
        return live.find_one({"region": region}, {"_id": 0})
    except Exception as exc:
        logger.warning(f"fetch({region}) failed: {exc}")
        return None


def list_versions(region, limit=50):
    vers = _coll(VERSIONS_COLL)
    if vers is None:
        return []
    try:
        return list(vers.find({"region": region}, {"_id": 0, "doc": 0}
                              ).sort("version", -1).limit(limit))
    except Exception as exc:
        logger.warning(f"list_versions({region}) failed: {exc}")
        return []


def fetch_version(region, version):
    vers = _coll(VERSIONS_COLL)
    if vers is None:
        return None
    try:
        return vers.find_one({"region": region, "version": int(version)}, {"_id": 0})
    except Exception as exc:
        logger.warning(f"fetch_version failed: {exc}")
        return None


def diff(old, new):
    """Flat {path: [old, new]} of changed leaves, for showing what a save did."""
    out = {}

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
        elif a != b:
            out[path] = [a, b]

    walk(old or {}, new or {})
    out.pop("_id", None)
    return out


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------
def ensure_indexes():
    client = get_client()
    if client is None:
        return
    try:
        import pymongo
        client[DB_NAME][LIVE_COLL].create_index("region", unique=True, background=True)
        client[DB_NAME][VERSIONS_COLL].create_index(
            [("region", pymongo.ASCENDING), ("version", pymongo.DESCENDING)],
            unique=True, background=True)
    except Exception as exc:
        logger.warning(f"ensure_indexes failed: {exc}")


def _next_version(region):
    vers = _coll(VERSIONS_COLL)
    if vers is None:
        return None
    last = vers.find_one({"region": region}, {"version": 1}, sort=[("version", -1)])
    return (int(last["version"]) + 1) if last else 1


def save(region, doc, *, edited_by, note=None, origin="manual"):
    """Validate, snapshot, then write the live document.

    Raises ValueError if the document is invalid — writing an invalid region
    config must not be possible by ignoring a return value. Returns the new
    version number, or None if MongoDB is unavailable.
    """
    doc = normalize(dict(doc or {}))
    doc["region"] = region
    errors, _ = validate(doc)
    if errors:
        raise ValueError("; ".join(errors))

    live, vers = _coll(LIVE_COLL), _coll(VERSIONS_COLL)
    if live is None or vers is None:
        return None
    try:
        ensure_indexes()
        current = live.find_one({"region": region}, {"_id": 0})

        # Baseline: capture the pre-edit state the first time a region is
        # touched, so the very first save is still diffable and revertible.
        if current is not None and _next_version(region) == 1:
            vers.insert_one({"region": region, "version": 1, "doc": current,
                             "origin": "baseline", "edited_by": None,
                             "note": "state before the first web edit",
                             "created_at": _utcnow()})

        version = _next_version(region)
        vers.insert_one({"region": region, "version": version, "doc": doc,
                         "origin": origin, "edited_by": edited_by,
                         "note": note, "created_at": _utcnow(),
                         "changed": diff(current, doc)})
        # Whole-document replace: apply_colorbar_overrides copies every top
        # level key, so a partial $set would leave stale keys behind that the
        # editor believes it removed.
        live.replace_one({"region": region}, doc, upsert=True)
        return version
    except Exception as exc:
        logger.warning(f"save({region}) failed: {exc}")
        return None


def revert(region, version, *, edited_by):
    """Re-apply an earlier version as a NEW version (history is append-only)."""
    old = fetch_version(region, version)
    if not old:
        return None
    return save(region, old["doc"], edited_by=edited_by, origin="revert",
                note=f"reverted to v{version}")


def delete(region, *, edited_by):
    """Remove the live override so regions.py defaults apply again.

    The history is kept: this is 'stop overriding', not 'erase what happened'.
    """
    live, vers = _coll(LIVE_COLL), _coll(VERSIONS_COLL)
    if live is None or vers is None:
        return False
    try:
        current = live.find_one({"region": region}, {"_id": 0})
        if current is None:
            return False
        vers.insert_one({"region": region, "version": _next_version(region),
                         "doc": current, "origin": "deleted",
                         "edited_by": edited_by, "created_at": _utcnow(),
                         "note": "override removed; regions.py defaults now apply"})
        live.delete_one({"region": region})
        return True
    except Exception as exc:
        logger.warning(f"delete({region}) failed: {exc}")
        return False
