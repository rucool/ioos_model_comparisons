from flask import Flask, render_template, request, jsonify, Response
import requests
from datetime import datetime
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor

from werkzeug.middleware.proxy_fix import ProxyFix

# This deployment is scoped to a single region for the SPICE Cruise.
REGION = "Tropical Western Atlantic"

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Satellite / glider diagnostic maps, organized as {year}/{month}/{day}/{variable}/*.png.
# The set of variable subfolders varies day to day, so it's discovered per-date rather
# than hardcoded.
SATELLITE_BASE_URL = "https://rucool.marine.rutgers.edu/gliders/spice/plots/staircase_analyses"

# Real-time glider plots (profiles, cross-sections, T-S diagrams), organized as
# {glider_id}/{profiles,xsection,TS}/{synoptic,last_24h,last_48h}/*.png. Refreshed
# in place by a cron every ~30 min — filenames don't change, only their contents.
RT_GLIDER_BASE_URL = "https://rucool.marine.rutgers.edu/gliders/spice/plots/gliders_rt"

# Satellite catalog cache: date_str -> (timestamp, {variable: [ {filename, time, url}, ... ]})
_sat_catalog_cache = {}
_SAT_CATALOG_TTL = 600  # seconds — this source updates multiple times a day

# Master variable-list cache: (timestamp, [variable, ...]) — the union of every
# variable folder ever seen across all available dates. Changes rarely (only
# when a product or glider is added/retired), so this is cached much longer.
_sat_master_variables_cache = None
_SAT_MASTER_TTL = 6 * 3600  # seconds

# ---------------------------------------------------------------------------
# Region / variable metadata (Model Comparisons)
# ---------------------------------------------------------------------------
region_info = {
    "Tropical Western Atlantic": {
        "variables": ["temperature", "salinity", "ocean_heat_content", "currents"],
        "depths": ["0m", "100m", "150m", "200m"]
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def add_zeros(n):
    return f"{n:02d}"


def check_image(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def build_map_urls(variable_depth, date_obj, time_str):
    """Return (img_copernicus, img_espc, img_espc_cmems) URLs."""
    variable_depth_mod = variable_depth.replace("_", "-")
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)

    # Map time label (e.g. "06Z") to 6-char time code (e.g. "060000")
    hour = time_str.replace("Z", "").zfill(2)
    time_code = f"{hour}0000"

    region_url_key = "tropical_western_atlantic"
    # currents files on the server use the folder name (underscores); temp/salinity use hyphens
    region_file_slug = region_url_key if variable_depth.startswith("currents") else region_url_key.replace("_", "-")
    base_url = (
        f"https://rucool.marine.rutgers.edu/hurricane/model_comparisons/maps/"
        f"{region_url_key}/{variable_depth}/{year}/{month}/"
    )

    if variable_depth == "ocean_heat_content":
        file_name_ohc = f"{region_url_key}_{year}-{month}-{day}T{time_code}Z"
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

    return img_copernicus, img_espc, img_espc_cmems


# ---------------------------------------------------------------------------
# Satellite / glider diagnostic maps (rucool.marine.rutgers.edu/gliders/spice/plots/staircase_analyses)
# ---------------------------------------------------------------------------

_SAT_SUBDIR_RE = re.compile(r'href="([^"/?]+)/"')
_SAT_FILE_RE   = re.compile(r'href="([^"]+\.png)"')
_SAT_TIME_RE   = re.compile(r'_(\d{8})_(\d{4,6})\.png$')
_SAT_TXT_RE    = re.compile(r'href="([^"]+\.txt)"')


def get_satellite_variables(date_obj):
    """List the variable subfolders present for a given date (varies day to day)."""
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    url = f"{SATELLITE_BASE_URL}/{year}/{month}/{day}/"
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    names = {m.group(1) for m in _SAT_SUBDIR_RE.finditer(r.text)} - {".", ".."}
    return sorted(names)


def get_satellite_images(date_obj, variable):
    """List available images for one variable on date_obj, sorted by embedded time."""
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    url = f"{SATELLITE_BASE_URL}/{year}/{month}/{day}/{variable}/"
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return []
    if r.status_code != 200:
        return []

    images = []
    for m in _SAT_FILE_RE.finditer(r.text):
        filename = m.group(1)
        tm = _SAT_TIME_RE.search(filename)
        if not tm:
            continue
        digits = tm.group(2)
        images.append({
            "filename": filename,
            "time": f"{digits[:2]}:{digits[2:4]}",
            "url": f"{url}{filename}",
        })
    images.sort(key=lambda e: e["filename"])
    return images


def get_satellite_text_file(date_obj, variable):
    """Find the daily summary .txt file (if any) alongside a variable's plots
    for one date — e.g. eddy_trajectory's 12N-crossing forecast table. At most
    one per variable/date, unlike the per-timestamp PNGs."""
    year = date_obj.year
    month = add_zeros(date_obj.month)
    day = add_zeros(date_obj.day)
    url = f"{SATELLITE_BASE_URL}/{year}/{month}/{day}/{variable}/"
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return None
    if r.status_code != 200:
        return None

    m = _SAT_TXT_RE.search(r.text)
    if not m:
        return None
    filename = m.group(1)
    return {"filename": filename, "url": f"{url}{filename}"}


def get_satellite_catalog(date_obj):
    """Return {variable: [ {filename, time, url}, ... ]} for every variable on date_obj,
    fetched in parallel. Cached for _SAT_CATALOG_TTL seconds."""
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in _sat_catalog_cache:
        ts, cached = _sat_catalog_cache[date_str]
        if _time.time() - ts < _SAT_CATALOG_TTL:
            return cached

    variables = get_satellite_variables(date_obj)
    result = {}
    if variables:
        with ThreadPoolExecutor(max_workers=min(8, len(variables))) as ex:
            futures = {ex.submit(get_satellite_images, date_obj, v): v for v in variables}
            for future in futures:
                images = future.result()
                if images:
                    result[futures[future]] = images

    _sat_catalog_cache[date_str] = (_time.time(), result)
    return result


# Per-variable latest-date cache: variable -> (timestamp, date_str or None).
# Individual products can lag days behind each other, so finding one
# variable's latest date means walking day folders and checking that
# variable's subfolder specifically — worth caching briefly.
_sat_variable_latest_cache = {}
_SAT_VARIABLE_LATEST_TTL = 600  # seconds
_SAT_VARIABLE_LATEST_MAX_DAYS_CHECKED = 60  # safety cap on the day-by-day scan


def _scan_for_variable_latest_date(variable, today_str):
    """Day-by-day (newest first) scan for the first date whose variable
    subfolder has images. Returns the date string, or None if nothing was
    found within the scan cap."""
    checked = 0
    try:
        r = requests.get(f"{SATELLITE_BASE_URL}/", timeout=6)
        years = sorted(re.findall(r'href="(\d{4})/"', r.text), reverse=True)
    except Exception:
        return None

    for year in years:
        if year > today_str[:4]:
            continue
        try:
            r = requests.get(f"{SATELLITE_BASE_URL}/{year}/", timeout=6)
            months = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
        except Exception:
            continue

        for month in months:
            if f"{year}-{month}" > today_str[:7]:
                continue
            try:
                r = requests.get(f"{SATELLITE_BASE_URL}/{year}/{month}/", timeout=6)
                days = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
            except Exception:
                continue

            for day in days:
                date_str = f"{year}-{month}-{day}"
                if date_str > today_str:
                    continue

                checked += 1
                var_url = f"{SATELLITE_BASE_URL}/{year}/{month}/{day}/{variable}/"
                try:
                    r2 = requests.get(var_url, timeout=6)
                    if r2.status_code == 200 and _SAT_FILE_RE.search(r2.text):
                        return date_str
                except Exception:
                    pass

                if checked >= _SAT_VARIABLE_LATEST_MAX_DAYS_CHECKED:
                    return None
    return None


def get_satellite_latest_date_for_variable(variable):
    """Find the most recent date with images for one specific variable,
    cached briefly since it's a day-by-day directory scan. Never returns a
    date beyond today."""
    if variable in _sat_variable_latest_cache:
        ts, cached = _sat_variable_latest_cache[variable]
        if _time.time() - ts < _SAT_VARIABLE_LATEST_TTL:
            return cached

    today_str = datetime.now().strftime("%Y-%m-%d")
    result = _scan_for_variable_latest_date(variable, today_str)
    _sat_variable_latest_cache[variable] = (_time.time(), result)
    return result


def get_satellite_master_variables():
    """Union of every variable subfolder ever seen across all available dates.

    Lets the variable dropdown stay stable while stepping through dates for
    decision-making: a day with no images for a product still lists it (as
    unavailable) instead of silently dropping it, so a user can keep clicking
    back through dates on the same product without it disappearing."""
    global _sat_master_variables_cache
    if _sat_master_variables_cache is not None:
        ts, variables = _sat_master_variables_cache
        if _time.time() - ts < _SAT_MASTER_TTL:
            return variables

    day_urls = []
    try:
        r = requests.get(f"{SATELLITE_BASE_URL}/", timeout=8)
        years = re.findall(r'href="(\d{4})/"', r.text)
    except Exception:
        years = []
    for year in years:
        try:
            r = requests.get(f"{SATELLITE_BASE_URL}/{year}/", timeout=8)
            months = re.findall(r'href="(\d{2})/"', r.text)
        except Exception:
            months = []
        for month in months:
            try:
                r = requests.get(f"{SATELLITE_BASE_URL}/{year}/{month}/", timeout=8)
                days = re.findall(r'href="(\d{2})/"', r.text)
            except Exception:
                days = []
            for day in days:
                day_urls.append(f"{SATELLITE_BASE_URL}/{year}/{month}/{day}/")

    def scrape_day(url):
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                return set()
            return {m.group(1) for m in _SAT_SUBDIR_RE.finditer(r.text)} - {".", ".."}
        except Exception:
            return set()

    variables = set()
    if day_urls:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for names in ex.map(scrape_day, day_urls):
                variables |= names

    result = sorted(variables)
    _sat_master_variables_cache = (_time.time(), result)
    return result


# ---------------------------------------------------------------------------
# Real-time glider plots (rucool.marine.rutgers.edu/gliders/spice/plots/gliders_rt)
# ---------------------------------------------------------------------------

# Folder name -> the infix used in that category's filenames (profiles/ holds
# "..._profile_..." files, singular, unlike the folder name).
_RT_CATEGORY_INFIX = {
    "profiles": "profile",
    "xsection": "xsection",
    "TS": "TS",
}
_RT_PERIODS = ["synoptic", "last_24h", "last_48h"]

_rt_gliders_cache = None  # (timestamp, [glider_id, ...])
_rt_catalog_cache = {}    # glider_id -> (timestamp, catalog)
_RT_CACHE_TTL = 120  # seconds — short, since the cron overwrites plots every ~30 min


def list_subdirs(url):
    """Return the immediate subfolder names at url (Apache-style autoindex)."""
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    names = {m.group(1) for m in _SAT_SUBDIR_RE.finditer(r.text)} - {".", ".."}
    return sorted(names)


def get_rt_gliders():
    """List the glider IDs currently publishing real-time plots. Cached briefly
    since new gliders are only added when a mission starts."""
    global _rt_gliders_cache
    if _rt_gliders_cache is not None:
        ts, cached = _rt_gliders_cache
        if _time.time() - ts < _RT_CACHE_TTL:
            return cached

    gliders = list_subdirs(f"{RT_GLIDER_BASE_URL}/")
    _rt_gliders_cache = (_time.time(), gliders)
    return gliders


def get_rt_category_images(glider_id, category, period):
    """List the plots for one glider/category/period, e.g. profiles/last_24h."""
    infix = _RT_CATEGORY_INFIX[category]
    url = f"{RT_GLIDER_BASE_URL}/{glider_id}/{category}/{period}/"
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return []
    if r.status_code != 200:
        return []

    prefix = f"{glider_id}_{infix}_"
    suffix = f"_{period}.png"
    images = []
    for m in _SAT_FILE_RE.finditer(r.text):
        filename = m.group(1)
        if category == "TS":
            if filename != f"{glider_id}_TS_{period}.png":
                continue
            variable = None
        else:
            if not (filename.startswith(prefix) and filename.endswith(suffix)):
                continue
            variable = filename[len(prefix):-len(suffix)]
        images.append({
            "variable": variable,
            "filename": filename,
            "url": f"{url}{filename}",
        })
    images.sort(key=lambda e: e["variable"] or "")
    return images


def get_rt_glider_catalog(glider_id):
    """Return {category: {period: [ {variable, filename, url}, ... ]}} for one
    glider, fetched in parallel across every category/period combination."""
    if glider_id in _rt_catalog_cache:
        ts, cached = _rt_catalog_cache[glider_id]
        if _time.time() - ts < _RT_CACHE_TTL:
            return cached

    jobs = [(cat, period) for cat in _RT_CATEGORY_INFIX for period in _RT_PERIODS]
    result = {cat: {} for cat in _RT_CATEGORY_INFIX}
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        futures = {ex.submit(get_rt_category_images, glider_id, cat, period): (cat, period) for cat, period in jobs}
        for future in futures:
            cat, period = futures[future]
            images = future.result()
            if images:
                result[cat][period] = images

    _rt_catalog_cache[glider_id] = (_time.time(), result)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template(
        "index.html",
        region=REGION,
        region_info=region_info,
        today=today,
    )


@app.route("/api/maps")
def api_maps():
    variable_depth = request.args.get("variable_depth", "temperature_0m")
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    time_str = request.args.get("time", "00Z")

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    img_copernicus, img_espc, img_espc_cmems = build_map_urls(variable_depth, date_obj, time_str)

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
    })


@app.route("/api/satellite-catalog")
def api_satellite_catalog():
    """All satellite / glider diagnostic images available for a given date, grouped by variable."""
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

    return jsonify(get_satellite_catalog(date_obj))


@app.route("/api/satellite-master-variables")
def api_satellite_master_variables():
    """Union of every variable folder ever seen, for a stable variable dropdown
    that doesn't lose entries when stepping across dates with missing products."""
    return jsonify({"variables": get_satellite_master_variables()})


@app.route("/api/satellite-latest-date")
def api_satellite_latest_date():
    """Most recent date with images for one specific variable — powers the
    Data Archive tab's "Latest" button. Individual products can lag days
    behind each other, so jumping to "today" wouldn't reliably land on data
    for the variable a user actually has selected."""
    variable = request.args.get("variable", "")
    if not variable:
        return jsonify({"available": False, "error": "Missing variable"}), 400

    date_str = get_satellite_latest_date_for_variable(variable)
    if not date_str:
        return jsonify({"available": False})

    return jsonify({"available": True, "date": date_str})


@app.route("/api/satellite-text")
def api_satellite_text():
    """Daily summary text file (if any) for a variable/date — e.g. eddy_trajectory's
    12N-crossing forecast table. Fetched and returned server-side (rather than
    having the browser fetch the file directly) to sidestep cross-origin
    restrictions and give the frontend plain, ready-to-render text."""
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    variable = request.args.get("variable", "")
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"available": False, "error": "Invalid date format"}), 400

    text_file = get_satellite_text_file(date_obj, variable)
    if not text_file:
        return jsonify({"available": False})

    try:
        r = requests.get(text_file["url"], timeout=8)
        r.raise_for_status()
    except Exception:
        return jsonify({"available": False})

    return jsonify({
        "available": True,
        "filename": text_file["filename"],
        "url": text_file["url"],
        "content": r.text,
    })


@app.route("/api/gliders-rt")
def api_gliders_rt():
    """List the gliders currently publishing real-time plots."""
    return jsonify({"gliders": get_rt_gliders()})


@app.route("/api/gliders-rt/<glider_id>/catalog")
def api_gliders_rt_catalog(glider_id):
    """Full profiles/xsection/TS catalog for one glider, across all time windows."""
    if glider_id not in get_rt_gliders():
        return jsonify({"error": "Unknown glider"}), 404

    catalog = get_rt_glider_catalog(glider_id)
    if not any(catalog.get(cat) for cat in catalog):
        return jsonify({"available": False}), 404

    # Cache-bust: the cron overwrites files in place every ~30 min, so append a
    # request-time token so browsers don't serve a stale cached image. Build a
    # copy rather than mutating catalog, which is a cached object shared across
    # requests within _RT_CACHE_TTL.
    cache_bust = int(_time.time())
    response_catalog = {
        cat: {
            period: [{**img, "url": f"{img['url']}?t={cache_bust}"} for img in images]
            for period, images in periods.items()
        }
        for cat, periods in catalog.items()
    }

    return jsonify({"available": True, "catalog": response_catalog})


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

    filename = url.split('/')[-1].split('?')[0]
    if not filename:
        filename = "plot.png"

    return Response(
        r.iter_content(chunk_size=8192),
        content_type=r.headers.get('Content-Type', 'image/png'),
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SPICE Cruise viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=not args.no_debug)
