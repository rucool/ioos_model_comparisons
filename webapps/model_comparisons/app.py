from flask import Flask, render_template, request, jsonify, Response, abort
from flask_login import current_user
import requests
from datetime import datetime, timedelta
import re
import os
import json
import time as _time
import threading as _threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import datetime as dt
import inspect
import pandas as pd
from erddapy import ERDDAP
from requests.exceptions import HTTPError as rHTTPError
from urllib.error import HTTPError as uHTTPError
from urllib.error import URLError

from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------------------------------------------------------------------
# Argo / glider platform helpers (inlined from ioos_model_comparisons)
# ---------------------------------------------------------------------------

rename_argo = {
    "platform_number":          "argo",
    "time (UTC)":               "time",
    "longitude (degrees_east)": "lon",
    "latitude (degrees_north)": "lat",
}


def _coerce_argo_qc_variables(include_qc):
    if include_qc is True:
        return ["time_qc", "position_qc", "profile_pres_qc", "profile_temp_qc",
                "profile_psal_qc", "pres_qc", "temp_qc", "psal_qc"]
    if not include_qc:
        return []
    if isinstance(include_qc, str):
        return [include_qc]
    return list(include_qc)


def _lon360to180(lon):
    """Wrap a longitude into [-180, 180) — same formula as calc.lon360to180
    in the main package (not vendored here)."""
    return ((lon + 180) % 360) - 180


def _get_argo_floats_by_time_one(bbox, time_start, time_end, wmo_id, variables, include_qc):
    """Single ERDDAP query — the body get_argo_floats_by_time() used to be,
    before it grew antimeridian handling. bbox here is assumed to be a plain
    lonmin<=lonmax range (no wraparound)."""
    if isinstance(variables, tuple):
        variables = list(variables)

    time_start = time_start or (time_end - dt.timedelta(days=1))

    requested_variables = ['platform_number', 'time', 'longitude', 'latitude']
    constraints = {'time>=': str(time_start), 'time<=': str(time_end)}

    if bbox:
        constraints['longitude>='] = bbox[0]
        constraints['longitude<='] = bbox[1]
        constraints['latitude>=']  = bbox[2]
        constraints['latitude<=']  = bbox[3]

    if wmo_id:
        if isinstance(wmo_id, (int, float)):
            wmo_id = str(wmo_id)
        constraints['platform_number='] = wmo_id

    if variables:
        requested_variables.extend(variables)

    requested_variables.extend(_coerce_argo_qc_variables(include_qc))
    requested_variables = list(dict.fromkeys(requested_variables))

    e = ERDDAP(server='IFREMER', protocol='tabledap', response='csv')
    e.dataset_id  = 'ArgoFloats'
    e.constraints = constraints
    e.variables   = requested_variables

    try:
        df = e.to_pandas(index_col="time (UTC)", parse_dates=True).dropna().tz_localize(None)
        df = df.reset_index().rename(rename_argo, axis=1)
        df = df.set_index(["argo", "time"]).sort_index()
    except rHTTPError:
        df = pd.DataFrame()
    return df


def get_argo_floats_by_time(bbox=(-110, -45, 0, 46),
                            time_start=None, time_end=dt.date.today(),
                            wmo_id=None, variables=None, include_qc=False):
    """Wraps _get_argo_floats_by_time_one() with antimeridian handling: IFREMER's
    ArgoFloats longitude is in [-180, 180], so a plain longitude>=/<=
    constraint can't express a bbox that crosses the dateline (e.g. Fiji's
    extent, [139.75, 190.25, ...] — floats east of 180 are actually stored
    at e.g. -170, which fails both `>= 139.75` and `<= 190.25` reads as a
    no-op upper bound). When bbox[1] > 180, split into a west half
    ([lonmin, 180]) and an east half ([-180, lonmax-360]), query both, and
    shift the east half's lon back into the caller's >180 convention so the
    two halves concatenate into one contiguous range — same west/east split
    load_cmems_subset() uses for the same reason."""
    if not bbox or bbox[1] <= 180:
        return _get_argo_floats_by_time_one(bbox, time_start, time_end, wmo_id, variables, include_qc)

    lon_west = (bbox[0], 180.0, bbox[2], bbox[3])
    lon_east = (-180.0, float(_lon360to180(bbox[1])), bbox[2], bbox[3])

    west = _get_argo_floats_by_time_one(lon_west, time_start, time_end, wmo_id, variables, include_qc)
    east = _get_argo_floats_by_time_one(lon_east, time_start, time_end, wmo_id, variables, include_qc)

    if not east.empty:
        east = east.copy()
        east['lon'] = east['lon'] + 360

    parts = [d for d in (west, east) if not d.empty]
    if not parts:
        return west  # empty DataFrame with the right columns/index
    return pd.concat(parts).sort_index()


def get_active_gliders(bbox=None, t0=None, t1=dt.date.today(), variables=None,
                       timeout=5, parallel=False):
    variables = variables or ["time", "longitude", "latitude", "profile_id", "depth"]
    bbox = bbox or [-100, -40, 18, 60]
    t0   = t0 or (t1 - dt.timedelta(days=1))

    t0 = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = t1.strftime('%Y-%m-%dT%H:%M:%SZ')

    e = ERDDAP(server='NGDAC')
    e.requests_kwargs['timeout'] = timeout

    kw = {'min_time': t0, 'max_time': t1}
    if bbox:
        kw['min_lon'] = bbox[0]
        kw['max_lon'] = bbox[1]
        kw['min_lat'] = bbox[2]
        kw['max_lat'] = bbox[3]

    search_url = e.get_search_url(search_for=None, response='csv', **kw)

    try:
        return pd.read_csv(search_url)
    except uHTTPError as error:
        print(f"{inspect.currentframe().f_code.co_name} - Error: {error}")
        return pd.DataFrame()
    except URLError as error:
        print(f"{inspect.currentframe().f_code.co_name} - Error: {error}")
        return pd.DataFrame()

# In-memory store of all active glider positions for the last 30 days.
# Loaded once on startup in a background thread and refreshed every 6 hours.
# glider_id -> [{lat, lon, time}, ...] sorted by time
_glider_all_positions     = {}
_glider_positions_ts      = 0.0
_GLIDER_POSITIONS_DAYS    = 30
_GLIDER_POSITIONS_TTL     = 6 * 3600

# Per-date cache of glider IDs that have profile plots (avoids a remote HTTP call per date)
_glider_ids_cache     = {}  # date_str -> (timestamp, list[str])
_GLIDER_IDS_CACHE_TTL = 3600


def _load_glider_positions():
    """Query NGDAC for one-point-per-day positions for all active gliders over the last 30 days."""
    today   = dt.date.today()
    t0      = today - timedelta(days=_GLIDER_POSITIONS_DAYS)
    t_start = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t_end   = (today + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        search_df = get_active_gliders(t0=t0, t1=today + timedelta(days=1))
    except Exception as e:
        print(f"_load_glider_positions search error: {e}")
        return {}
    if search_df.empty or 'Dataset ID' not in search_df.columns:
        return {}
    positions = {}
    for gid in search_df['Dataset ID'].tolist():
        try:
            url = (
                f"https://gliders.ioos.us/erddap/tabledap/{gid}.csvp"
                f"?time,latitude,longitude&time>={t_start}&time<={t_end}&latitude!=NaN"
            )
            r = requests.get(url, timeout=30)
            if r.status_code != 200 or not r.text.strip():
                continue
            gdf = pd.read_csv(StringIO(r.text))
            gdf.columns = [c.split(' (')[0].strip() for c in gdf.columns]
            if gdf.empty:
                continue
            gdf['date'] = gdf['time'].str[:10]
            gdf = gdf.groupby('date').last().reset_index()
            positions[gid] = sorted([
                {'lat': float(row['latitude']), 'lon': float(row['longitude']), 'time': str(row['time'])}
                for _, row in gdf.iterrows()
            ], key=lambda p: p['time'])
        except Exception as e:
            print(f"_load_glider_positions error for {gid}: {e}")
    print(f"_load_glider_positions: loaded {len(positions)} gliders")
    return positions


def _glider_positions_loop():
    """Background daemon: load glider positions on startup then refresh every 6 hours."""
    global _glider_all_positions, _glider_positions_ts
    while True:
        loaded = _load_glider_positions()
        if loaded:
            _glider_all_positions = loaded
            _glider_positions_ts  = _time.time()
        _time.sleep(_GLIDER_POSITIONS_TTL)


def _get_glider_ids_cached(date_obj):
    """Return glider IDs with profile plots for date_obj, with a 1-hour in-memory cache."""
    date_str = date_obj.strftime('%Y-%m-%d')
    if date_str in _glider_ids_cache:
        ts, ids = _glider_ids_cache[date_str]
        if _time.time() - ts < _GLIDER_IDS_CACHE_TTL:
            return ids
    ids = get_glider_ids(date_obj)
    _glider_ids_cache[date_str] = (_time.time(), ids)
    return ids


_bg_lock    = _threading.Lock()
_bg_started = False


def start_background_tasks():
    """Start the NGDAC glider-position refresher once per process.

    Not started at import (so tests and tooling can import app.py without
    network side effects); instead the first request starts it via
    _ensure_background_tasks below, which works whichever WSGI entry point
    (wsgi:app or app:app) the server uses.
    """
    global _bg_started
    with _bg_lock:
        if _bg_started:
            return
        _threading.Thread(target=_glider_positions_loop, daemon=True).start()
        # Keeps the last week of storm tracks current in MongoDB, so visitors
        # never wait on tropycal (one worker does it; see storm_track.py).
        storm_track.start_refresher()
        _bg_started = True


app = Flask(__name__)
app.config.setdefault("BACKGROUND_TASKS", True)


@app.before_request
def _ensure_background_tasks():
    if not _bg_started and app.config.get("BACKGROUND_TASKS", True):
        start_background_tasks()

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# --- front editor: sign-in + the MongoDB-backed wall editor -----------------
# Both are additive. The dashboard and every existing /api/* route stay public
# and anonymous; only /fronts requires a session. If FLASK_SECRET_KEY is unset
# the editor self-disables (503) and the public site is unaffected.
from ioos_model_comparisons.env import load_env    # noqa: E402
load_env()
from auth import init_auth                          # noqa: E402
from fronts_bp import fronts_bp                     # noqa: E402
from regions_bp import regions_bp                   # noqa: E402
from ohc_bp import ohc_bp                           # noqa: E402
init_auth(app)
app.register_blueprint(fronts_bp)
app.register_blueprint(regions_bp)
app.register_blueprint(ohc_bp)

_ERROR_TITLES = {401: "Sign-in required", 403: "Not allowed", 503: "Temporarily unavailable"}


@app.errorhandler(401)
@app.errorhandler(403)
@app.errorhandler(503)
def _styled_http_error(e):
    code = getattr(e, "code", 500)
    return render_template(
        "error.html", code=code,
        title=_ERROR_TITLES.get(code, "Error"),
        message=getattr(e, "description", None) or e.name,
    ), code

# Optional local plots directory; set LOCAL_PLOTS_DIR env var to serve location
# data from a local mirror. Omit (or leave empty) to always fetch from the remote server.
_LOCAL_PLOTS_BASE = os.environ.get("LOCAL_PLOTS_DIR", "")

BASE_REMOTE_PROFILES = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles"

# Region lists the portal used to hand-type here now come from
# ioos_model_comparisons/config/portal.yml (validated by registry.py), so
# adding a region/box/folder is a config edit. Names kept for the code below.
from ioos_model_comparisons.registry import load_registry  # noqa: E402
import discovery  # noqa: E402

_registry = load_registry()

# [lonmin, lonmax, latmin, latmax] per Argo profile folder, and its label —
# boxes on the Profiles map (portal.yml argo_regions). The Argo/FVON
# pipelines only publish into these coarse folders, not every fine-grained
# regions.py region, so every Argo/FVON lookup below queries all of them.
ARGO_REGION_EXTENTS = {b["key"]: b["extent"] for b in _registry.argo_boxes()}
ARGO_REGION_LABELS = {b["key"]: b["label"] for b in _registry.argo_boxes()}
# Areas the glider profile scripts search (portal.yml glider_regions).
GLIDER_REGION_EXTENTS = {b["key"]: b["extent"] for b in _registry.glider_boxes()}
GLIDER_REGION_LABELS = {b["key"]: b["label"] for b in _registry.glider_boxes()}
ARGO_REGIONS = list(ARGO_REGION_EXTENTS.keys())
FVON_REGIONS = list(_registry.portal.fvon_regions)
FVON_FOLDER_MAP = dict(_registry.portal.fvon_folder_map)

# Backend position cache: (region_key, date_str) -> (timestamp, result_dict)
_pos_cache = {}
_POS_CACHE_TTL = 3600  # seconds

# Glider track cache: date_str -> (timestamp, tracks_dict)
_tracks_cache = {}
_TRACKS_CACHE_TTL = 3600  # seconds

# Storm track cache: date_str -> (timestamp, storms_dict). (See storm_track.py.)
_storm_cache = {}
_STORM_CACHE_TTL = 3600  # seconds

# ---------------------------------------------------------------------------
# Region / variable metadata
# ---------------------------------------------------------------------------
# region_info (variables + per-variable depths, since currents are frequently
# plotted at depths temperature/salinity aren't) used to be a hand-typed
# snapshot here, kept "in sync" with ioos_model_comparisons/regions.py by
# hand — which is exactly why it silently fell out of sync (gulf_stream and
# south_africa were both missing entirely). It's now built live from the same
# source every offline plotting script and the /regions web editor already
# read: regions.region_config() + db.apply_colorbar_overrides(). See
# ioos_model_comparisons/region_catalog.py. Cached for
# region_catalog.CACHE_TTL seconds, so an edit through /regions shows up here
# without a code change or restart.
from ioos_model_comparisons.region_catalog import get_region_info  # noqa: E402
from ioos_model_comparisons import storm_track  # noqa: E402

# Regions with published comparison maps (Maps Archive / Overview), in
# display order: portal.yml map_regions, as display names. A curated subset —
# regions.py has no way to say which regions have published imagery.
def get_map_regions():
    by_key = {v.get("key"): name for name, v in get_region_info().items()}
    return [by_key[k] for k in _registry.portal.map_regions if k in by_key]


def unknown_map_regions():
    """portal.yml map_regions keys with no region config (reported by /healthz)."""
    keys = {v.get("key") for v in get_region_info().values()}
    return sorted(k for k in _registry.portal.map_regions if k not in keys)


def region_folder(region_name):
    """Display name -> region folder (what's on disk and in the catalog)."""
    info = get_region_info().get(region_name) or {}
    return info.get("folder") or region_name.replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Adaptive Sampling Guidance metadata
# ---------------------------------------------------------------------------
# portal.yml adaptive_sampling: label -> (url folder, display name in file
# names); label -> (url folder, panel subfolder, file code); label -> folder.
adaptive_sampling_regions = {k: tuple(v) for k, v in _registry.portal.adaptive_sampling.regions.items()}
adaptive_sampling_variables = {k: tuple(v) for k, v in _registry.portal.adaptive_sampling.variables.items()}
adaptive_sampling_models = dict(_registry.portal.adaptive_sampling.models)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def add_zeros(n):
    return f"{n:02d}"


# HEAD-request check for one URL (legacy discovery); see discovery.py.
check_image = discovery.check_image


# Time step order for 6-hourly cycling
TIME_STEPS = ["00Z", "06Z", "12Z", "18Z"]


def build_adaptive_sampling_url(region_key, display_name, variable_folder,
                                panel_type, var_code, model_folder,
                                date_obj, time_str):
    """Build a single adaptive sampling guidance image URL."""
    year  = date_obj.year
    month = add_zeros(date_obj.month)
    day   = add_zeros(date_obj.day)
    hour  = time_str.replace("Z", "").zfill(2)
    time_code = f"{hour}0000"

    # Display name has spaces — URL-encode them
    display_encoded = display_name.replace(" ", "%20")
    suffix = "rtofs-espc-diff" if model_folder == "espc" else "rtofs-copernicus-diff"
    filename = (
        f"{display_encoded}_{year}-{month}-{day}T{time_code}Z"
        f"_{var_code}_{suffix}.png"
    )
    return (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/adaptive_sampling_guidance/maps/"
        f"{region_key}/{variable_folder}/{panel_type}/{model_folder}/{year}/{month}/{filename}"
    )


def _legacy_asg_latest_date():
    """Scrape the server to find the most recent date with ASG files.

    Uses east_coast / OHC / espc as the reference path (most reliable).
    Returns a date string 'YYYY-MM-DD' or None.
    """
    base = (
        "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/adaptive_sampling_guidance/"
        "maps/east_coast/ocean_heat_content/three_panel/espc/"
    )
    try:
        # Get available years
        r = requests.get(base, timeout=6)
        years = sorted(re.findall(r'href="(\d{4})/"', r.text), reverse=True)
        if not years:
            return None

        year = years[0]
        r = requests.get(f"{base}{year}/", timeout=6)
        months = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
        if not months:
            return None

        month = months[0]
        r = requests.get(f"{base}{year}/{month}/", timeout=6)
        # Extract all dates from filenames like: US East Coast_2026-04-14T...
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})T', r.text)
        if not dates:
            return None

        latest = sorted(set(dates), reverse=True)[0]
        return latest
    except Exception:
        return None


_asg_latest_cache = {"ts": 0.0, "date": None}


def get_asg_latest_date():
    """Most recent date with adaptive sampling figures (cached 10 min; this
    runs on every page load). Falls back to today."""
    now = _time.time()
    if now - _asg_latest_cache["ts"] > 600 or not _asg_latest_cache["date"]:
        _asg_latest_cache["date"] = discovery.asg_latest_date(_legacy_asg_latest_date)
        _asg_latest_cache["ts"] = now
    return _asg_latest_cache["date"] or datetime.now().strftime("%Y-%m-%d")

def get_latest_map_info(region, variable_depth):
    """(date_str, 'HHZ') of the newest comparison map for a region display
    name / variable_depth, among the panels shown for it."""
    folder = region_folder(region)
    comps = [c for p in _registry.map_panels(folder, variable_depth) for c in p.comparisons]
    return discovery.latest_map(folder, variable_depth, comps or None)

def get_glider_ids(date_obj):
    """Glider IDs with a 400 m profile figure on date_obj."""
    return discovery.glider_ids(date_obj)

def get_glider_profile_url(glider_id, date_obj):
    year = date_obj.year
    month_day = f"{add_zeros(date_obj.month)}-{add_zeros(date_obj.day)}"
    start_date = f"{year}{add_zeros(date_obj.month)}{add_zeros(date_obj.day)}"
    # next day for end_date (simple increment — no month-boundary handling needed for display)
    next_day = date_obj + timedelta(days=1)
    end_date = f"{next_day.year}{add_zeros(next_day.month)}{add_zeros(next_day.day)}"
    return (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"gliders/{year}/{month_day}/{glider_id}_{start_date}_to_{end_date}_400m.png"
    )


def get_argo_ids(region, date_obj):
    """{wmo: filename} of Argo profile figures in one folder on date_obj."""
    region_key = region.lower().replace(" ", "_")
    return {k: v["filename"] for k, v in discovery.platform_ids("argo", date_obj, [region_key]).items()}

def get_argo_profile_url(region, filename, date_obj):
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    region_key = region.lower().replace(" ", "_")
    return (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"argo/{region_key}/{year}/{month}/{day}/{filename}"
    )


def get_latest_argo_date(region):
    region_key = region.lower().replace(" ", "_")
    url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/argo/{region_key}/last_14_days/"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})T', r.text)
        if not dates:
            return None
        return sorted(set(dates), reverse=True)[0]
    except Exception:
        return None


_glider_latest_date_cache = {"ts": 0.0, "date": None}
_GLIDER_LATEST_DATE_TTL = 1800  # seconds


def get_latest_glider_date(lookback_days=16):
    """Most recent date with a published per-day gliders/{date}/locations.json
    — whatever it contains, same as Argo/FVON's own *-latest-date lookups
    (which likewise just take the newest date with any data, no completeness
    check). By request 2026-08-24: always reflect the actual latest day so
    this tracks the glider script's real output directly, including while
    it's still catching up on a given day, rather than lagging behind on an
    older "more complete" day.

    Scans backward day by day (same primary source fetch_gliders() reads)
    and returns as soon as it finds a day with at least one entry. Cached
    since this can be several HTTP requests.
    """
    now = _time.time()
    if now - _glider_latest_date_cache["ts"] < _GLIDER_LATEST_DATE_TTL:
        return _glider_latest_date_cache["date"]

    today = datetime.now().date()
    result = None
    for i in range(lookback_days):
        d = today - timedelta(days=i)
        month_day = f"{add_zeros(d.month)}-{add_zeros(d.day)}"
        url = f"{BASE_REMOTE_PROFILES}/gliders/{d.year}/{month_day}/locations.json"
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200 and len(r.json()) > 0:
                result = d.strftime("%Y-%m-%d")
                break
        except Exception:
            continue

    _glider_latest_date_cache["ts"] = now
    _glider_latest_date_cache["date"] = result
    return result


def get_fvon_ids(region, date_obj):
    """{platform id: filename} of FVON profile figures in one folder on date_obj."""
    region_key = region.lower().replace(" ", "_")
    region_key = FVON_FOLDER_MAP.get(region_key, region_key)
    return {k: v["filename"] for k, v in discovery.platform_ids("fvon", date_obj, [region_key]).items()}

def get_latest_fvon_date(region):
    region_key = region.lower().replace(" ", "_")
    if region_key == "fiji":
        region_key = "south_pacific"
    url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/fvon/{region_key}/last_14_days/"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})T', r.text)
        if not dates:
            return None
        return sorted(set(dates), reverse=True)[0]
    except Exception:
        return None


def get_fvon_profile_url(region, filename, date_obj):
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    region_key = region.lower().replace(" ", "_")
    if region_key == "fiji":
        region_key = "south_pacific"
        
    return (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"fvon/{region_key}/{year}/{month}/{day}/{filename}"
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    asg_latest = get_asg_latest_date()
    region_info = get_region_info()  # live YAML + MongoDB, see region_catalog.py
    map_regions = get_map_regions()
    # Which comparison panels each region/variable shows — comparisons.yml,
    # evaluated server-side so app.js has no hardcoded model rules.
    # {region display name: {variable_depth: [panel key, ...]}}
    map_panels = {}
    for name in map_regions:
        info = region_info.get(name) or {}
        folder = info.get("folder")
        var_depths = [f"{var}_{d}" for var, depths in (info.get("depths") or {}).items()
                      for d in depths] + ["ocean_heat_content"]
        map_panels[name] = {vd: [p.key for p in _registry.map_panels(folder, vd)] for vd in var_depths}
    panel_defs = _registry.panel_defs()
    order = {k: i for i, k in enumerate(_registry.overview_order)}
    overview_panel_defs = sorted(panel_defs, key=lambda p: order.get(p["key"], 99))
    return render_template(
        "index.html",
        region_info=region_info,
        argo_regions=sorted(region_info.keys()),
        map_regions=map_regions,
        map_panels=map_panels,
        panel_defs=panel_defs,
        overview_panel_defs=overview_panel_defs,
        # Boxes on the Profiles map: the folders the Argo pipelines publish
        # into, and the areas the glider scripts search (portal.yml).
        argo_region_boxes=_registry.argo_boxes(),
        glider_region_boxes=_registry.glider_boxes(),
        adaptive_sampling_regions=adaptive_sampling_regions,
        adaptive_sampling_variables=adaptive_sampling_variables,
        adaptive_sampling_models=adaptive_sampling_models,
        today=today,
        asg_latest=asg_latest,
    )

@app.route("/api/maps")
def api_maps():
    region = request.args.get("region", "Mid Atlantic Bight")
    variable_depth = request.args.get("variable_depth", "temperature_0m")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    time_str = request.args.get("time", "00Z")

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    # {panel key: {url, available, label}} for every panel in comparisons.yml
    # (keys: copernicus, espc, three_model, eccofs, nesdis, goes16, goes19, ...)
    return jsonify(discovery.map_panels(region_folder(region), region, variable_depth,
                                        date_obj, time_str))

@app.route("/api/glider-ids")
def api_glider_ids():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    ids = get_glider_ids(date_obj)
    return jsonify({"ids": ids})


@app.route("/api/glider-profile")
def api_glider_profile():
    glider_id = request.args.get("glider_id")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    if not glider_id:
        return jsonify({"error": "glider_id required"}), 400
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    url = get_glider_profile_url(glider_id, date_obj)
    return jsonify({"url": url, "available": discovery.url_available(url)})


@app.route("/api/argo-ids")
def api_argo_ids():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    # Every Argo region folder at once, merged — see ARGO_REGIONS.
    return jsonify({"ids": discovery.platform_ids("argo", date_obj, ARGO_REGIONS)})


@app.route("/api/argo-profile")
def api_argo_profile():
    region = request.args.get("region", "Mid Atlantic Bight")
    filename = request.args.get("filename")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    if not filename:
        return jsonify({"error": "filename required"}), 400
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    url = get_argo_profile_url(region, filename, date_obj)
    return jsonify({"url": url, "available": discovery.url_available(url)})


@app.route("/api/argo-latest-date")
def api_argo_latest_date():
    # Most recent date with data in ANY Argo region folder (see ARGO_REGIONS).
    def _legacy():
        with ThreadPoolExecutor(max_workers=len(ARGO_REGIONS)) as ex:
            dates = [d for d in ex.map(get_latest_argo_date, ARGO_REGIONS) if d]
        return max(dates) if dates else None
    latest_date = discovery.latest_platform_date("argo", ARGO_REGIONS, _legacy)
    if latest_date:
        return jsonify({"date": latest_date})
    else:
        return jsonify({"date": datetime.now().strftime("%Y-%m-%d")})


@app.route("/api/glider-latest-date")
def api_glider_latest_date():
    latest_date = discovery.latest_platform_date("glider", legacy_fn=get_latest_glider_date)
    if latest_date:
        return jsonify({"date": latest_date})
    else:
        return jsonify({"date": datetime.now().strftime("%Y-%m-%d")})


@app.route("/api/fvon-ids")
def api_fvon_ids():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    # Every FVON region folder at once, merged — see FVON_REGIONS.
    return jsonify({"ids": discovery.platform_ids("fvon", date_obj, FVON_REGIONS)})


@app.route("/api/fvon-latest-date")
def api_fvon_latest_date():
    # Most recent date with data in ANY FVON region folder (see FVON_REGIONS).
    def _legacy():
        with ThreadPoolExecutor(max_workers=len(FVON_REGIONS)) as ex:
            dates = [d for d in ex.map(get_latest_fvon_date, FVON_REGIONS) if d]
        return max(dates) if dates else None
    latest_date = discovery.latest_platform_date("fvon", FVON_REGIONS, _legacy)
    if latest_date:
        return jsonify({"date": latest_date})
    else:
        return jsonify({"date": datetime.now().strftime("%Y-%m-%d")})


@app.route("/api/fvon-profile")
def api_fvon_profile():
    region = request.args.get("region", "Bahamas")
    filename = request.args.get("filename")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    if not filename:
        return jsonify({"error": "filename required"}), 400
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    url = get_fvon_profile_url(region, filename, date_obj)
    return jsonify({"url": url, "available": discovery.url_available(url)})

@app.route("/api/adaptive-sampling")
def api_adaptive_sampling():
    region_label   = request.args.get("region",   "US East Coast")
    variable_label = request.args.get("variable", "Ocean Heat Content")
    model_label    = request.args.get("model",    "ESPC")
    date_str       = request.args.get("date",     datetime.now().strftime("%Y-%m-%d"))
    time_str       = request.args.get("time",     "00Z")

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    region_entry   = adaptive_sampling_regions.get(region_label)
    variable_entry = adaptive_sampling_variables.get(variable_label)
    model_folder   = adaptive_sampling_models.get(model_label, "espc")

    if not region_entry or not variable_entry:
        return jsonify({"error": "Unknown region or variable"}), 400

    region_key, display_name   = region_entry
    variable_folder, panel_type, var_code = variable_entry

    url = build_adaptive_sampling_url(
        region_key, display_name, variable_folder,
        panel_type, var_code, model_folder,
        date_obj, time_str
    )
    return jsonify({"url": url, "available": discovery.url_available(url)})


@app.route("/api/adaptive-sampling-latest")
def api_asg_latest():
    """Return the most recent date that has adaptive sampling figures."""
    latest = get_asg_latest_date()
    return jsonify({"date": latest})


@app.route("/api/overview-latest")
def api_overview_latest():
    region = request.args.get("region", "Mid Atlantic Bight")
    variable_depth = request.args.get("variable_depth", "temperature_0m")

    info = get_latest_map_info(region, variable_depth)
    if not info:
        return jsonify({"error": "No images found", "available": False}), 404

    date_str, time_str = info
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date parsed", "available": False}), 500

    # Each model comparison is generated by an independent pipeline, so the
    # newest run often has some models ready before others. Each panel says
    # whether it's there, so the frontend can fall back to whichever model
    # has actually finished rendering.
    panels = discovery.map_panels(region_folder(region), region, variable_depth, date_obj, time_str)
    return jsonify({"available": True, "date": date_str, "time": time_str, **panels})


@app.route("/api/download")
def api_download():
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Only proxy plot images from our own archive (prevents SSRF).
    if not url.startswith('https://rucool.marine.rutgers.edu/hurricane/model_comparisons/'):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        r = requests.get(url, stream=True, timeout=(5, 30))
        r.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500

    filename = url.split('/')[-1]
    if not filename:
        filename = "plot.png"

    return Response(
        r.iter_content(chunk_size=8192),
        content_type=r.headers.get('Content-Type', 'image/png'),
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route("/api/profile-locations")
def api_profile_locations():
    ptype = request.args.get("type", "argo")
    region = request.args.get("region", "Mid Atlantic Bight")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    region_key = region.lower().replace(" ", "_")
    
    def make_url(ptype, filename, region_key=None, year=None, month_day=None):
        """Build the correct dated URL from a filename.
        Filenames encode dates as: {id}-profile-{YYYY}-{MM}-{DD}T{HMS}Z.png
        We parse year/month/day from the filename to avoid last_14_days symlinks
        which Apache doesn't follow.
        """
        import re as _re
        m = _re.search(r'(\d{4})-(\d{2})-(\d{2})T', filename)
        base = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        if ptype == "glider":
            return f"{base}gliders/{year}/{month_day}/{filename}"
        elif ptype in ("argo", "fvon") and m:
            y, mo, d = m.group(1), m.group(2), m.group(3)
            folder = "argo" if ptype == "argo" else "fvon"
            return f"{base}{folder}/{region_key}/{y}/{mo}/{d}/{filename}"
        # Fallback to last_14_days if date can't be parsed
        folder = "argo" if ptype == "argo" else "fvon"
        return f"{base}{folder}/{region_key}/last_14_days/{filename}"
    
    if ptype == "glider":
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            return jsonify({"error": "Invalid date"}), 400
        year = date_obj.year
        month_day = f"{add_zeros(date_obj.month)}-{add_zeros(date_obj.day)}"
        url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/gliders/{year}/{month_day}/locations.json"
        local_path = os.path.join(_LOCAL_PLOTS_BASE, "profiles", "gliders", str(year), month_day, "locations.json") if _LOCAL_PLOTS_BASE else ""
    elif ptype == "argo":
        url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/argo/{region_key}/last_14_days/locations.json"
        local_path = os.path.join(_LOCAL_PLOTS_BASE, "profiles", "argo", region_key, "last_14_days", "locations.json") if _LOCAL_PLOTS_BASE else ""
    elif ptype == "fvon":
        url = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/fvon/{region_key}/last_14_days/locations.json"
        local_path = os.path.join(_LOCAL_PLOTS_BASE, "profiles", "fvon", region_key, "last_14_days", "locations.json") if _LOCAL_PLOTS_BASE else ""
    else:
        return jsonify({"error": "Invalid type"}), 400

    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, 'r') as f:
                data = json.load(f)
            for filename, entry in data.items():
                entry["url"] = make_url(ptype, filename, region_key, year if ptype=='glider' else None, month_day if ptype=='glider' else None)
            return jsonify(data)
        except Exception as e:
            print(f"Error reading local locations.json: {e}")
            pass

    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            for filename, entry in data.items():
                entry["url"] = make_url(ptype, filename, region_key, year if ptype=='glider' else None, month_day if ptype=='glider' else None)
            return jsonify(data)
        else:
            return jsonify({})
    except:
        return jsonify({})


@app.route("/api/all-profile-locations")
def api_all_profile_locations():
    """Fetch glider + argo + fvon locations simultaneously and return all in one payload."""
    region = request.args.get("region", "Mid Atlantic Bight")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    region_key = region.lower().replace(" ", "_")
    
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        date_obj = datetime.now().date()

    year = date_obj.year
    month_day = f"{add_zeros(date_obj.month)}-{add_zeros(date_obj.day)}"

    base_remote = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles"
    base_local  = os.path.join(_LOCAL_PLOTS_BASE, "profiles") if _LOCAL_PLOTS_BASE else ""

    # FVON folder name differs from display name for some regions
    fvon_key = FVON_FOLDER_MAP.get(region_key, region_key)

    def load_locations(ptype):
        """Load and enrich locations.json for one platform type."""
        if ptype == "glider":
            local  = f"{base_local}/gliders/last_14_days/locations.json"
            remote = f"{base_remote}/gliders/last_14_days/locations.json"
        elif ptype == "argo":
            local  = f"{base_local}/argo/{region_key}/last_14_days/locations.json"
            remote = f"{base_remote}/argo/{region_key}/last_14_days/locations.json"
        else:  # fvon
            local  = f"{base_local}/fvon/{fvon_key}/last_14_days/locations.json"
            remote = f"{base_remote}/fvon/{fvon_key}/last_14_days/locations.json"

        data = {}
        if base_local and os.path.exists(local):
            try:
                with open(local) as f:
                    data = json.load(f)
            except Exception:
                pass

        if not data:
            try:
                r = requests.get(remote, timeout=6)
                if r.status_code == 200:
                    data = r.json()
            except Exception:
                pass

        # Inject dated URL for each entry
        enriched = {}
        for filename, entry in data.items():
            if ptype == "glider":
                # Parse YYYYMMDD from {id}_{YYYYMMDD}_to_{YYYYMMDD}_{depth}m.png
                m = re.search(r'_(\d{4})(\d{2})(\d{2})_to_', filename)
                if m:
                    y2, mo2, d2 = m.group(1), m.group(2), m.group(3)
                    url = f"{base_remote}/gliders/{y2}/{mo2}-{d2}/{filename}"
                else:
                    url = f"{base_remote}/gliders/last_14_days/{filename}"
            else:
                m = re.search(r'(\d{4})-(\d{2})-(\d{2})T', filename)
                folder = "argo" if ptype == "argo" else "fvon"
                rkey   = region_key if ptype == "argo" else fvon_key
                if m:
                    y2, mo2, d2 = m.group(1), m.group(2), m.group(3)
                    url = f"{base_remote}/{folder}/{rkey}/{y2}/{mo2}/{d2}/{filename}"
                else:
                    url = f"{base_remote}/{folder}/{rkey}/last_14_days/{filename}"
            entry["url"] = url
            entry["type"] = ptype
            enriched[filename] = entry
        return ptype, enriched

    result = {"glider": {}, "argo": {}, "fvon": {}}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(load_locations, t): t for t in ["glider", "argo", "fvon"]}
        for future in futures:
            ptype, data = future.result()
            result[ptype] = data

    return jsonify(result)


@app.route("/api/profile-positions")
def api_profile_positions():
    """Return platform positions for a specific date, merging Argo and FVON
    across every region their pipelines actually publish to (ARGO_REGIONS /
    FVON_REGIONS) rather than one user-picked region — see the comment above
    ARGO_REGIONS for why a region picker here just meant clicking a name with
    no data under it while the parent region (which has it) sat unselected.
    With the catalog backend, positions come from the figure catalog (the
    position each plot script recorded). Legacy: ERDDAP (argo) and dated
    locations.json files (gliders), with a last_14_days fallback for FVON."""
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        date_obj = datetime.now().date()

    cache_key = date_str
    if cache_key in _pos_cache:
        ts, cached = _pos_cache[cache_key]
        if _time.time() - ts < _POS_CACHE_TTL:
            return jsonify(cached)

    year     = date_obj.year
    month    = add_zeros(date_obj.month)
    day      = add_zeros(date_obj.day)
    month_day = f"{month}-{day}"
    base_local = os.path.join(_LOCAL_PLOTS_BASE, "profiles") if _LOCAL_PLOTS_BASE else ""

    def fetch_gliders():
        """Dated locations.json (primary), falling back to last_14_days filtered by date."""
        local  = os.path.join(base_local, "gliders", str(year), month_day, "locations.json") if base_local else ""
        remote = f"{BASE_REMOTE_PROFILES}/gliders/{year}/{month_day}/locations.json"
        data = {}
        if local and os.path.exists(local):
            try:
                with open(local) as f:
                    data = json.load(f)
            except Exception:
                pass
        if not data:
            try:
                r = requests.get(remote, timeout=6)
                if r.status_code == 200:
                    data = r.json()
            except Exception:
                pass

        # ERDDAP fallback: search NGDAC for active gliders, query each for lat/lon
        if not data:
            glider_files = set(get_glider_ids(date_obj))  # IDs with profile plots
            t_start = (date_obj - timedelta(days=14)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_end   = (date_obj + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            try:
                search_df = get_active_gliders(
                    t0=date_obj - timedelta(days=14),
                    t1=date_obj + timedelta(days=1),
                )
                if not search_df.empty and 'Dataset ID' in search_df.columns:
                    for glider_id in search_df['Dataset ID'].tolist():
                        if glider_id not in glider_files:
                            continue  # only show gliders that have a profile plot
                        try:
                            ngdac_url = (
                                f"https://gliders.ioos.us/erddap/tabledap/{glider_id}.csvp"
                                f"?time,latitude,longitude"
                                f"&time>={t_start}&time<={t_end}"
                                f"&latitude!=NaN"
                            )
                            r = requests.get(ngdac_url, timeout=15)
                            if r.status_code != 200 or not r.text.strip():
                                continue
                            from io import StringIO
                            gdf = pd.read_csv(StringIO(r.text))
                            # Strip unit suffixes e.g. "latitude (degrees_north)" → "latitude"
                            gdf.columns = [c.split(' (')[0].strip() for c in gdf.columns]
                            if gdf.empty:
                                continue
                            row = gdf.iloc[-1]
                            plot_url = get_glider_profile_url(glider_id, date_obj)
                            filename = plot_url.split('/')[-1]
                            data[filename] = {
                                'lat':       float(row['latitude']),
                                'lon':       float(row['longitude']),
                                'glider_id': glider_id,
                                'time':      date_str,
                            }
                        except Exception as e_inner:
                            print(f"Glider ERDDAP error for {glider_id}: {e_inner}")
            except Exception as e:
                print(f"Glider NGDAC search error: {e}")

        # Final fallback: filter last_14_days/locations.json by date
        if not data:
            l14_local  = os.path.join(base_local, "gliders", "last_14_days", "locations.json") if base_local else ""
            l14_remote = f"{BASE_REMOTE_PROFILES}/gliders/last_14_days/locations.json"
            l14 = {}
            if l14_local and os.path.exists(l14_local):
                try:
                    with open(l14_local) as f:
                        l14 = json.load(f)
                except Exception:
                    pass
            if not l14:
                try:
                    r = requests.get(l14_remote, timeout=6)
                    if r.status_code == 200:
                        l14 = r.json()
                except Exception:
                    pass
            data = {k: v for k, v in l14.items() if v.get("time", "")[:10] == date_str}

        out = {}
        for filename, entry in data.items():
            entry["url"]  = f"{BASE_REMOTE_PROFILES}/gliders/{year}/{month_day}/{filename}"
            entry["type"] = "glider"
            out[filename] = entry
        return out

    def fetch_argo_region(region_key):
        """Positions for one Argo region folder. ERDDAP for lat/lon (bbox from
        ARGO_REGION_EXTENTS), matched against that folder's own directory
        listing for the plot URL. IFREMER's ArgoFloats ERDDAP has been
        observed to intermittently omit a float from one query that an
        identical retry moments later does return (confirmed by hand — not
        a bbox/time-window bug on our side), so a float with a plot already
        generated could silently vanish from the map despite still being
        selectable in the dropdown (which doesn't depend on ERDDAP at all).
        Gaps like that — not just a totally-empty ERDDAP response — are now
        filled from this folder's last_14_days/locations.json."""
        argo_files = get_argo_ids(region_key, date_obj)  # {wmo: filename} — may be empty
        out = {}

        extent = ARGO_REGION_EXTENTS.get(region_key)
        if extent:
            try:
                adf = get_argo_floats_by_time(
                    bbox=extent,
                    time_start=date_obj - timedelta(days=1),
                    time_end=date_obj + timedelta(days=1),
                )
                if not adf.empty:
                    adf_reset = adf.reset_index().drop_duplicates("argo")
                    for _, row in adf_reset.iterrows():
                        wmo = str(row["argo"])
                        filename = argo_files.get(wmo)
                        if not filename:
                            continue  # only show floats that have a profile plot
                        out[filename] = {
                            "lat":  float(row["lat"]),
                            "lon":  float(row["lon"]),
                            "wmo":  wmo,
                            "time": date_str,
                            "url":  get_argo_profile_url(region_key, filename, date_obj),
                            "type": "argo",
                        }
            except Exception as e:
                print(f"Argo ERDDAP error for {region_key}, falling back to locations.json: {e}")

        # Fill in any plotted float ERDDAP didn't return a position for above
        # (or all of them, if ERDDAP failed outright) from this region's
        # last_14_days/locations.json.
        missing = set(argo_files.values()) - set(out.keys())
        if missing:
            local  = os.path.join(base_local, "argo", region_key, "last_14_days", "locations.json") if base_local else ""
            remote = f"{BASE_REMOTE_PROFILES}/argo/{region_key}/last_14_days/locations.json"
            data = {}
            if local and os.path.exists(local):
                try:
                    with open(local) as f:
                        data = json.load(f)
                except Exception:
                    pass
            if not data:
                try:
                    r = requests.get(remote, timeout=6)
                    if r.status_code == 200:
                        data = r.json()
                except Exception:
                    pass
            for filename, entry in data.items():
                if filename not in missing:
                    continue
                if entry.get("time", "")[:10] != date_str:
                    continue
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})T", filename)
                if m:
                    y2, mo2, d2 = m.group(1), m.group(2), m.group(3)
                    entry["url"] = f"{BASE_REMOTE_PROFILES}/argo/{region_key}/{y2}/{mo2}/{d2}/{filename}"
                entry["type"] = "argo"
                out[filename] = entry

        return out

    def fetch_argo():
        merged = {}
        with ThreadPoolExecutor(max_workers=len(ARGO_REGIONS)) as ex:
            for region_out in ex.map(fetch_argo_region, ARGO_REGIONS):
                merged.update(region_out)
        return merged

    def fetch_fvon_region(region_key):
        """Positions for one FVON region folder — filters that folder's
        last_14_days/locations.json by the selected date."""
        local  = os.path.join(base_local, "fvon", region_key, "last_14_days", "locations.json") if base_local else ""
        remote = f"{BASE_REMOTE_PROFILES}/fvon/{region_key}/last_14_days/locations.json"
        data = {}
        if local and os.path.exists(local):
            try:
                with open(local) as f:
                    data = json.load(f)
            except Exception:
                pass
        if not data:
            try:
                r = requests.get(remote, timeout=6)
                if r.status_code == 200:
                    data = r.json()
            except Exception:
                pass
        out = {}
        for filename, entry in data.items():
            if entry.get("time", "")[:10] != date_str:
                continue
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})T", filename)
            if m:
                y2, mo2, d2 = m.group(1), m.group(2), m.group(3)
                entry["url"] = f"{BASE_REMOTE_PROFILES}/fvon/{region_key}/{y2}/{mo2}/{d2}/{filename}"
            entry["type"] = "fvon"
            out[filename] = entry
        return out

    def fetch_fvon():
        merged = {}
        with ThreadPoolExecutor(max_workers=len(FVON_REGIONS)) as ex:
            for region_out in ex.map(fetch_fvon_region, FVON_REGIONS):
                merged.update(region_out)
        return merged

    def from_legacy():
        result = {"glider": {}, "argo": {}, "fvon": {}}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(fetch_gliders): "glider",
                ex.submit(fetch_argo):    "argo",
                ex.submit(fetch_fvon):    "fvon",
            }
            for future in futures:
                result[futures[future]] = future.result()
        return result

    def from_catalog():
        # Positions the plot scripts recorded with each figure. Only figures
        # cataloged without one (no hook yet, no locations.json) fall back to
        # the legacy fetch for that platform type.
        result, missing = discovery.catalog_positions(date_obj, ARGO_REGIONS, FVON_REGIONS)
        fetchers = {"glider": fetch_gliders, "argo": fetch_argo, "fvon": fetch_fvon}
        for ptype, names in missing.items():
            if names:
                legacy = fetchers[ptype]()
                result[ptype].update({n: legacy[n] for n in names if n in legacy})
        return result

    result = discovery.run("profile_positions", from_catalog, from_legacy,
                           compare=lambda r: {k: sorted(v) for k, v in r.items()})
    _pos_cache[cache_key] = (_time.time(), result)
    return jsonify(result)


@app.route("/api/profile-positions/clear-cache", methods=["POST"])
def api_clear_profile_positions_cache():
    """Drop cached /api/profile-positions results so the next request re-fetches
    from ERDDAP/locations.json instead of serving up to an hour of a stale
    snapshot (see _POS_CACHE_TTL). POST-only and editor-gated — an open,
    unauthenticated version of this is a cache-busting amplification vector:
    repeatedly clearing + requesting forces the full expensive ERDDAP/
    remote-JSON fetch on every hit instead of ever getting a cache hit,
    for both this server and the third-party services it calls. Reuses the
    same `editor` role Fronts/Regions already gate writes behind, rather
    than inventing a separate secret to manage.
    Pass ?date=YYYY-MM-DD to clear just that date, otherwise every cached
    date is dropped."""
    if not (current_user.is_authenticated and current_user.has_role("editor")):
        abort(403)
    date_str = request.args.get("date")
    if date_str:
        removed = _pos_cache.pop(date_str, None) is not None
        return jsonify({"cleared": [date_str] if removed else []})
    cleared = list(_pos_cache.keys())
    _pos_cache.clear()
    return jsonify({"cleared": cleared})


@app.route("/api/glider-positions-status")
def api_glider_positions_status():
    """Diagnostic: show whether the background glider position store has loaded."""
    loaded = bool(_glider_all_positions)
    age_s  = _time.time() - _glider_positions_ts if _glider_positions_ts else None
    return jsonify({
        "loaded":        loaded,
        "glider_count":  len(_glider_all_positions),
        "gliders":       sorted(_glider_all_positions.keys()),
        "age_seconds":   round(age_s) if age_s is not None else None,
        "next_refresh_in_seconds": round(max(0, _GLIDER_POSITIONS_TTL - age_s)) if age_s is not None else None,
    })


@app.route("/api/glider-tracks")
def api_glider_tracks():
    """Return last 5 days of positions per glider, sorted by time, for track display."""
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        end_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        end_date = datetime.now().date()

    if date_str in _tracks_cache:
        ts, cached = _tracks_cache[date_str]
        if _time.time() - ts < _TRACKS_CACHE_TTL:
            return jsonify(cached)

    if discovery.use_catalog():
        try:
            tracks = discovery.catalog_glider_tracks(end_date, days=5)
            if tracks:
                _tracks_cache[date_str] = (_time.time(), tracks)
                return jsonify(tracks)
        except Exception as e:
            print(f"catalog glider tracks failed, using legacy: {e}")

    base_local  = os.path.join(_LOCAL_PLOTS_BASE, "profiles", "gliders") if _LOCAL_PLOTS_BASE else ""
    base_remote = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/gliders"

    # Pre-load last_14_days as a fallback for days where dated files are missing
    l14 = {}
    l14_local = os.path.join(base_local, "last_14_days", "locations.json") if base_local else ""
    if l14_local and os.path.exists(l14_local):
        try:
            with open(l14_local) as f:
                l14 = json.load(f)
        except Exception:
            pass
    if not l14:
        try:
            r = requests.get(f"{base_remote}/last_14_days/locations.json", timeout=5)
            if r.status_code == 200:
                l14 = r.json()
        except Exception:
            pass

    tracks = {}  # glider_id -> [{lat, lon, time}, ...]

    for days_back in range(5):
        d = end_date - timedelta(days=days_back)
        d_str = d.strftime("%Y-%m-%d")
        month_day = f"{add_zeros(d.month)}-{add_zeros(d.day)}"
        year = d.year
        local = os.path.join(base_local, str(year), month_day, "locations.json") if base_local else ""

        data = {}
        if local and os.path.exists(local):
            try:
                with open(local) as f:
                    data = json.load(f)
            except Exception:
                pass

        if not data:
            try:
                r = requests.get(f"{base_remote}/{year}/{month_day}/locations.json", timeout=5)
                if r.status_code == 200:
                    data = r.json()
            except Exception:
                pass

        # Fall back to last_14_days filtered by this specific date
        if not data and l14:
            data = {k: v for k, v in l14.items() if v.get("time", "")[:10] == d_str}

        for filename, entry in data.items():
            gid = entry.get("glider_id") or filename.split("-")[0]
            tracks.setdefault(gid, []).append({
                "lat": entry.get("lat"),
                "lon": entry.get("lon"),
                "time": entry.get("time", ""),
            })

    # Fallback when locations.json files are missing.
    # Within 30 days: slice from the pre-loaded in-memory store (no network call).
    # Older than 30 days: query ERDDAP directly per glider.
    if not tracks:
        glider_ids  = _get_glider_ids_cached(end_date)
        days_ago    = (dt.date.today() - end_date).days
        window_start = (end_date - timedelta(days=5)).strftime('%Y-%m-%d')
        window_end   = end_date.strftime('%Y-%m-%d')

        if glider_ids and days_ago <= _GLIDER_POSITIONS_DAYS and _glider_all_positions:
            for gid in glider_ids:
                pts = [
                    p for p in _glider_all_positions.get(gid, [])
                    if window_start <= p['time'][:10] <= window_end
                ]
                if pts:
                    tracks[gid] = pts

        elif glider_ids:
            t_start = (end_date - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
            t_end   = (end_date + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
            for gid in glider_ids:
                try:
                    ngdac_url = (
                        f"https://gliders.ioos.us/erddap/tabledap/{gid}.csvp"
                        f"?time,latitude,longitude"
                        f"&time>={t_start}&time<={t_end}"
                        f"&latitude!=NaN"
                    )
                    r = requests.get(ngdac_url, timeout=15)
                    if r.status_code != 200 or not r.text.strip():
                        continue
                    gdf = pd.read_csv(StringIO(r.text))
                    gdf.columns = [c.split(' (')[0].strip() for c in gdf.columns]
                    if gdf.empty:
                        continue
                    gdf['date'] = gdf['time'].str[:10]
                    gdf = gdf.groupby('date').last().reset_index()
                    for _, row in gdf.iterrows():
                        tracks.setdefault(gid, []).append({
                            'lat': float(row['latitude']),
                            'lon': float(row['longitude']),
                            'time': str(row['time']),
                        })
                except Exception as e:
                    print(f"Glider ERDDAP track error for {gid}: {e}")

    for gid in tracks:
        tracks[gid].sort(key=lambda p: p["time"])

    _tracks_cache[date_str] = (_time.time(), tracks)
    return jsonify(tracks)


@app.route("/api/storm-track")
def api_storm_track():
    """Storms active near a given date (archive: Atlantic + East Pacific; for
    the last few days also live NHC/JTWC storms in every basin — see
    storm_track.py), with track history trimmed through that date and, where
    a forecast advisory exists for it, the forecast points and cone."""
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "date is required"}), 400
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    if date_str in _storm_cache:
        ts, cached = _storm_cache[date_str]
        # Short in-process layer over the shared MongoDB cache.
        ttl = 300 if storm_track.is_recent(date_obj) else _STORM_CACHE_TTL
        if _time.time() - ts < ttl:
            return jsonify(cached)

    try:
        storms = storm_track.cached_storms_for_date(date_obj)
    except Exception as e:
        print(f"storm_track error for {date_str}: {e}")
        storms = {}

    result = {"storms": storms}
    _storm_cache[date_str] = (_time.time(), result)
    return jsonify(result)


@app.route("/api/storm-track/clear-cache", methods=["POST"])
def api_clear_storm_track_cache():
    """Same rationale and gating as /api/profile-positions/clear-cache."""
    if not (current_user.is_authenticated and current_user.has_role("editor")):
        abort(403)
    date_str = request.args.get("date")
    if date_str:
        removed = _storm_cache.pop(date_str, None) is not None
        return jsonify({"cleared": [date_str] if removed else []})
    cleared = list(_storm_cache.keys())
    _storm_cache.clear()
    return jsonify({"cleared": cleared})


# ---------------------------------------------------------------------------
# Health check (used by scripts/release/deploy.sh and monitoring)
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _app_version():
    """Release version from the VERSION file build_webapp.py writes, else 'dev'."""
    try:
        with open(os.path.join(_APP_DIR, "VERSION")) as f:
            return f.read().strip() or "dev"
    except OSError:
        return "dev"


@app.route("/healthz")
def healthz():
    """Always 200 while the process is alive (deploys check `version`);
    ?strict=1 returns 503 when degraded, for monitoring."""
    from ioos_model_comparisons.db import get_client
    checks = {"version": _app_version()}
    degraded = False

    t0 = _time.time()
    client = get_client()
    mongo_ok = False
    if client is not None:
        try:
            client.admin.command("ping")
            mongo_ok = True
        except Exception:
            pass
    checks["mongo"] = {"ok": mongo_ok, "ms": round((_time.time() - t0) * 1000)}
    degraded |= not mongo_ok

    checks["discovery"] = {"mode": discovery.mode(), "backend": discovery.backend_name()}
    try:
        from ioos_model_comparisons import catalog
        fresh = catalog.freshness() if mongo_ok else None
        checks["catalog"] = None if fresh is None else {
            "latest_created_at": fresh["latest_created_at"].isoformat() if fresh["latest_created_at"] else None,
            "count_24h": fresh["count_24h"],
            "stale": catalog.is_stale(fresh),
        }
    except Exception as e:
        checks["catalog"] = {"error": str(e)}
    catalog_fresh = (checks["catalog"] or {}).get("stale") is False
    if discovery.mode() in ("auto", "catalog") and not catalog_fresh:
        degraded = True  # the portal is (or would be) falling back to legacy discovery
    checks["config"] = {"hash": _registry.config_hash()}

    try:
        checks["unknown_map_regions"] = unknown_map_regions()
    except Exception as e:
        checks["unknown_map_regions"] = None
        checks["region_info_error"] = str(e)
        degraded = True

    checks["status"] = "degraded" if degraded else "ok"
    code = 503 if (degraded and request.args.get("strict")) else 200
    return jsonify(checks), code


if __name__ == "__main__":
    app.run(debug=True, port=5001)
