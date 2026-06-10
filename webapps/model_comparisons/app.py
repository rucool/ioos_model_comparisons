from flask import Flask, render_template, request, jsonify, Response
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


def get_argo_floats_by_time(bbox=(-110, -45, 0, 46),
                            time_start=None, time_end=dt.date.today(),
                            wmo_id=None, variables=None, include_qc=False):
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


_threading.Thread(target=_glider_positions_loop, daemon=True).start()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Optional local plots directory; set LOCAL_PLOTS_DIR env var to serve location
# data from a local mirror. Omit (or leave empty) to always fetch from the remote server.
_LOCAL_PLOTS_BASE = os.environ.get("LOCAL_PLOTS_DIR", "")

BASE_REMOTE_PROFILES = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles"

# [lonmin, lonmax, latmin, latmax] per argo profile region key
ARGO_REGION_EXTENTS = {
    "mid_atlantic_bight":        [-77,    -63,    35,    44   ],
    "south_atlantic_bight":      [-82.25, -63.75, 24.75, 42.25],
    "gulf_of_mexico":            [-99,    -78,    18,    33   ],
    "caribbean":                 [-89,    -58,     7,    23.5 ],
    "tropical_western_atlantic": [-70.25, -40.75,  0,    25   ],
    "west_florida_shelf":        [-87.5,  -80,    22.5,  30.5 ],
    "guam":                      [129.75, 160.25,  4.75, 25.25],
    "south_pacific":             [140,    180,   -30.25, -4.75],
}

FVON_FOLDER_MAP = {"fiji": "south_pacific"}

# Backend position cache: (region_key, date_str) -> (timestamp, result_dict)
_pos_cache = {}
_POS_CACHE_TTL = 3600  # seconds

# Glider track cache: date_str -> (timestamp, tracks_dict)
_tracks_cache = {}
_TRACKS_CACHE_TTL = 3600  # seconds

# ---------------------------------------------------------------------------
# Region / variable metadata
# ---------------------------------------------------------------------------
region_info = {
    "Yucatan/Mastr": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "300m", "600m", "900m"]
    },
    "Yucatan": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m"]
    },
    "Caribbean: Leeward Islands": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Loop Current Eddy": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "30m"]
    },
    "Gulf of Mexico": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Eastern Gulf of Mexico": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "US East Coast": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "South Atlantic Bight": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Mid Atlantic Bight": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "30m", "100m", "150m", "200m"]
    },
    "West Florida Shelf": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m"]
    },
    "Caribbean": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Caribbean: Windward Islands": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m"]
    },
    "Amazon Plume": {
        "variables": ["temperature", "salinity", "currents"],
        "depths": ["0m", "150m", "160m", "200m"]
    },
    "Hurricane Alley": {
        "variables": ["temperature", "salinity", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Tropical Western Atlantic": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "100m", "150m", "200m"]
    },
    "Atlantis II Seamounts": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "300m"]
    },
    "Eastern Pacific - Mexico": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "100m", "200m"]
    },
    "Hawaii": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "100m", "200m"]
    },
    "WMO V - South Pacific": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "175m"]
    },
    "RU29": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "100m", "150m", "200m"]
    },
    "East Coast": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m"]
    },
    "Philippines Sea": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "200m", "1500m"]
    },
    "Western Gulf of Mexico": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "200m"]
    },
    "Guam": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m"]
    },
    "Fiji": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "1500m"]
    },
    "Bahamas": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "150m", "1500m"]
    }
}

argo_regions = list(region_info.keys())
map_regions = [
    "Caribbean",
    "Gulf of Mexico",
    "South Atlantic Bight",
    "Mid Atlantic Bight",
    "West Florida Shelf",
    "Caribbean: Windward Islands",
    "Tropical Western Atlantic",
    "Eastern Pacific - Mexico",
    "Hawaii",
    "Guam",
    "Fiji"
]

# ---------------------------------------------------------------------------
# Adaptive Sampling Guidance metadata
# ---------------------------------------------------------------------------
# Maps display label → (url_key, display_name_for_filename)
adaptive_sampling_regions = {
    "Caribbean":                  ("caribbean",                "Caribbean"),
    "US East Coast":              ("east_coast",               "US East Coast"),
    "Eastern Gulf of Mexico":     ("gom_east",                 "Eastern Gulf of Mexico"),
    "Western Gulf of Mexico":     ("gom_west",                 "Western Gulf of Mexico"),
    "Tropical Western Atlantic":  ("tropical_western_atlantic","Tropical Western Atlantic"),
}

# variable label → (url_folder, panel_subfolder, var_code)
adaptive_sampling_variables = {
    "Ocean Heat Content":   ("ocean_heat_content",   "three_panel",      "ohc"),
    "Sea Surface Salinity": ("sea_surface_salinity", "three_panel",      "sss"),
    "Speed":               ("speed",                "three_panel_diff", "speed"),
}

# comparison model → url folder name
adaptive_sampling_models = {
    "ESPC":       "espc",
    "Copernicus": "copernicus",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Regions whose server folder name doesn't follow the default space→underscore pattern
REGION_FOLDER_MAP = {
    "Eastern Pacific - Mexico": "mexico_pacific",
}


def add_zeros(n):
    return f"{n:02d}"


def check_image(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# Time step order for 6-hourly cycling
TIME_STEPS = ["00Z", "06Z", "12Z", "18Z"]


def build_map_urls(region, variable_depth, date_obj, time_str):
    """Return (img_copernicus, img_espc, img_espc_cmems, img_goes) URLs."""
    variable_depth_mod = variable_depth.replace("_", "-")
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)

    # Map time label (e.g. "06Z") to 6-char time code (e.g. "060000")
    hour = time_str.replace("Z", "").zfill(2)
    time_code = f"{hour}0000"

    region_url_key = REGION_FOLDER_MAP.get(region, region.replace(" ", "_").lower())
    # currents files on the server use the folder name (underscores); temp/salinity use hyphens
    if variable_depth.startswith("currents"):
        region_file_slug = region_url_key
    else:
        region_file_slug = region_url_key.replace("_", "-")
    base_url = (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/maps/"
        f"{region_url_key}/{variable_depth}/{year}/{month}/"
    )

    img_goes = None

    if variable_depth == "ocean_heat_content":
        if region in ('Guam', 'Fiji'):
            # Single combined three-model image: {RegionName}_{timestamp}_ohc_rtofs-espc-cmems.png
            file_name_ohc = f"{region}_{year}-{month}-{day}T{time_code}Z_ohc_rtofs-espc-cmems.png"
            img_copernicus = None
            img_espc       = None
            img_espc_cmems = f"{base_url}{file_name_ohc}"
        else:
            file_name_ohc = (
                f"{region_url_key}"
                f"_{year}-{month}-{day}T{time_code}Z"
            )
            img_copernicus = f"{base_url}{file_name_ohc}_heat_content_rtofs-cmems.png"
            img_espc      = f"{base_url}{file_name_ohc}_heat_content_rtofs-espc.png"
            img_espc_cmems = f"{base_url}{file_name_ohc}_heat_content_espc-cmems.png"
    else:
        file_name = (
            f"{region_file_slug}"
            f"_{year}-{month}-{day}T{time_code}Z"
            f"_{variable_depth_mod}"
        )
        img_copernicus = f"{base_url}{file_name}_rtofs-vs-cmems.png"
        img_espc      = f"{base_url}{file_name}_rtofs-vs-espc.png"
        img_espc_cmems = f"{base_url}{file_name}_espc-vs-cmems.png"
        img_goes       = f"{base_url}{file_name}_rtofs-vs-GOES.png"

    return img_copernicus, img_espc, img_espc_cmems, img_goes
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


def get_asg_latest_date():
    """Scrape the server to find the most recent date with ASG files.

    Uses east_coast / OHC / espc as the reference path (most reliable).
    Returns a date string 'YYYY-MM-DD' or today's date as fallback.
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
            return datetime.now().strftime("%Y-%m-%d")

        year = years[0]
        r = requests.get(f"{base}{year}/", timeout=6)
        months = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
        if not months:
            return datetime.now().strftime("%Y-%m-%d")

        month = months[0]
        r = requests.get(f"{base}{year}/{month}/", timeout=6)
        # Extract all dates from filenames like: US East Coast_2026-04-14T...
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})T', r.text)
        if not dates:
            return datetime.now().strftime("%Y-%m-%d")

        latest = sorted(set(dates), reverse=True)[0]
        return latest
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def get_latest_map_info(region, variable_depth):
    """Scrape the server to find the most recent date and time for standard maps."""
    region_key = REGION_FOLDER_MAP.get(region, region.replace(" ", "_").lower())
    base = f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/maps/{region_key}/{variable_depth}/"
    try:
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
        matches = re.findall(r'(\d{4}-\d{2}-\d{2})T(\d{6})Z', r.text)
        if not matches:
            return None

        latest = sorted(list(set(matches)), reverse=True)[0]
        date_str = latest[0]
        hour_str = latest[1][:2] + "Z"
        return date_str, hour_str
    except Exception:
        return None

def get_glider_ids(date_obj):
    year = date_obj.year
    month_day = f"{add_zeros(date_obj.month)}-{add_zeros(date_obj.day)}"
    url = (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"gliders/{year}/{month_day}/"
    )
    try:
        response = requests.get(url, timeout=8)
    except Exception:
        return []

    if response.status_code != 200:
        return []

    glider_ids = set()
    pattern = re.compile(r'href="([^"]+\.png)"')
    for line in response.text.splitlines():
        match = pattern.search(line)
        if match:
            file_name = match.group(1)
            glider_ids.add(file_name.split("_20")[0])
    return sorted(glider_ids)


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
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    region_key = region.lower().replace(" ", "_")
    url = (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"argo/{region_key}/{year}/{month}/{day}/"
    )
    try:
        response = requests.get(url, timeout=8)
    except Exception:
        return {}

    if response.status_code != 200:
        return {}

    argo_ids = {}
    pattern = re.compile(r'href="([^"]+\.png)"')
    for line in response.text.splitlines():
        match = pattern.search(line)
        if match:
            file_name = match.group(1)
            argo_id = file_name.split("-")[0]
            argo_ids[argo_id] = file_name
    return argo_ids


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


def get_fvon_ids(region, date_obj):
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    region_key = region.lower().replace(" ", "_")
    if region_key == "fiji":
        region_key = "south_pacific"
        
    url = (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/profiles/"
        f"fvon/{region_key}/{year}/{month}/{day}/"
    )
    try:
        response = requests.get(url, timeout=8)
    except Exception:
        return {}

    if response.status_code != 200:
        return {}

    fvon_ids = {}
    pattern = re.compile(r'href="([^"]+\.png)"')
    for line in response.text.splitlines():
        match = pattern.search(line)
        if match:
            file_name = match.group(1)
            # FVON uses format like 0-22000-0-RCM3H6L-profile-2026-05-22T001249Z.png
            fvon_id = file_name.split("-profile-")[0]
            fvon_ids[fvon_id] = file_name
    return fvon_ids


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
    asg_latest = get_asg_latest_date()  # cached on first call; fast on server restart
    return render_template(
        "index.html",
        region_info=region_info,
        argo_regions=argo_regions,
        map_regions=map_regions,
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

    img_copernicus, img_espc, img_espc_cmems, img_goes = build_map_urls(region, variable_depth, date_obj, time_str)

    return jsonify({
        "copernicus": {
            "url": img_copernicus,
            "available": check_image(img_copernicus),
            "label": "RTOFS vs. Copernicus (CMEMS)",
        },
        "espc": {
            "url": img_espc,
            "available": check_image(img_espc),
            "label": "RTOFS vs. ESPC",
        },
        "espc_cmems": {
            "url": img_espc_cmems,
            "available": check_image(img_espc_cmems),
            "label": "ESPC vs. CMEMS",
        },
        "goes": {
            "url": img_goes,
            "available": check_image(img_goes) if img_goes else False,
            "label": "RTOFS vs. GOES",
        },
    })


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
    return jsonify({"url": url, "available": check_image(url)})


@app.route("/api/argo-ids")
def api_argo_ids():
    region = request.args.get("region", "Mid Atlantic Bight")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    ids = get_argo_ids(region, date_obj)
    return jsonify({"ids": ids})


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
    return jsonify({"url": url, "available": check_image(url)})


@app.route("/api/argo-latest-date")
def api_argo_latest_date():
    region = request.args.get("region", "Mid Atlantic Bight")
    latest_date = get_latest_argo_date(region)
    if latest_date:
        return jsonify({"date": latest_date})
    else:
        return jsonify({"date": datetime.now().strftime("%Y-%m-%d")})


@app.route("/api/fvon-ids")
def api_fvon_ids():
    region = request.args.get("region", "Bahamas")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date"}), 400

    ids = get_fvon_ids(region, date_obj)
    return jsonify({"ids": ids})


@app.route("/api/fvon-latest-date")
def api_fvon_latest_date():
    region = request.args.get("region", "Bahamas")
    latest_date = get_latest_fvon_date(region)
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
    return jsonify({"url": url, "available": check_image(url)})

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
    return jsonify({"url": url, "available": check_image(url)})


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

    img_copernicus, img_espc, img_espc_cmems, img_goes = build_map_urls(region, variable_depth, date_obj, time_str)

    return jsonify({
        "available": True,
        "date": date_str,
        "time": time_str,
        "copernicus": {
            "url": img_copernicus,
            "label": "RTOFS vs. Copernicus (CMEMS)"
        },
        "espc": {
            "url": img_espc,
            "label": "RTOFS vs. ESPC"
        },
        "espc_cmems": {
            "url": img_espc_cmems,
            "label": "ESPC vs. CMEMS"
        },
        "goes": {
            "url": img_goes,
            "label": "RTOFS vs. GOES"
        },
    })
@app.route("/api/download")
def api_download():
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Optional: ensure URL belongs to rucool.marine.rutgers.edu to prevent SSRF
    if not url.startswith('https://rucool.marine.rutgers.edu/'):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        r = requests.get(url, stream=True)
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
        
    import os, json
    if os.path.exists(local_path):
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
    FVON_FOLDER_MAP = {"fiji": "south_pacific"}
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
        if os.path.exists(local):
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
    """Return platform positions for a specific date by querying ERDDAP (argo)
    and dated locations.json files (gliders), with a last_14_days fallback for FVON."""
    region   = request.args.get("region", "Mid Atlantic Bight")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    region_key = region.lower().replace(" ", "_")

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        date_obj = datetime.now().date()

    cache_key = (region_key, date_str)
    if cache_key in _pos_cache:
        ts, cached = _pos_cache[cache_key]
        if _time.time() - ts < _POS_CACHE_TTL:
            return jsonify(cached)

    year     = date_obj.year
    month    = add_zeros(date_obj.month)
    day      = add_zeros(date_obj.day)
    month_day = f"{month}-{day}"
    fvon_key = FVON_FOLDER_MAP.get(region_key, region_key)
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

    def fetch_argo():
        """ERDDAP for lat/lon; plot URL from directory listing when available.
        Falls back to filtering last_14_days/locations.json if ERDDAP is unavailable."""
        extent = ARGO_REGION_EXTENTS.get(region_key)
        if extent:
            argo_files = get_argo_ids(region, date_obj)  # {wmo: filename} — may be empty
            try:
                adf = get_argo_floats_by_time(
                    bbox=extent,
                    time_start=date_obj - timedelta(days=1),
                    time_end=date_obj + timedelta(days=1),
                )
                if not adf.empty:
                    adf_reset = adf.reset_index().drop_duplicates("argo")
                    out = {}
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
                            "url":  get_argo_profile_url(region, filename, date_obj),
                            "type": "argo",
                        }
                    if out:
                        return out
            except Exception as e:
                print(f"Argo ERDDAP error, falling back to locations.json: {e}")

        # Fallback: filter last_14_days/locations.json by date
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
        out = {}
        for filename, entry in data.items():
            if entry.get("time", "")[:10] != date_str:
                continue
            m = re.search(r"(\d{4})-(\d{2})-(\d{2})T", filename)
            if m:
                y2, mo2, d2 = m.group(1), m.group(2), m.group(3)
                entry["url"] = f"{BASE_REMOTE_PROFILES}/argo/{region_key}/{y2}/{mo2}/{d2}/{filename}"
            entry["type"] = "argo"
            out[filename] = entry
        return out

    def fetch_fvon():
        """Filter last_14_days/locations.json by the selected date."""
        local  = os.path.join(base_local, "fvon", fvon_key, "last_14_days", "locations.json") if base_local else ""
        remote = f"{BASE_REMOTE_PROFILES}/fvon/{fvon_key}/last_14_days/locations.json"
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
                entry["url"] = f"{BASE_REMOTE_PROFILES}/fvon/{fvon_key}/{y2}/{mo2}/{d2}/{filename}"
            entry["type"] = "fvon"
            out[filename] = entry
        return out

    result = {"glider": {}, "argo": {}, "fvon": {}}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(fetch_gliders): "glider",
            ex.submit(fetch_argo):    "argo",
            ex.submit(fetch_fvon):    "fvon",
        }
        for future in futures:
            result[futures[future]] = future.result()

    _pos_cache[cache_key] = (_time.time(), result)
    return jsonify(result)


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


if __name__ == "__main__":
    app.run(debug=True, port=5001)
