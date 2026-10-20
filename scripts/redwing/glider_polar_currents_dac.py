#!/usr/bin/env python
"""
Polar visualization of surface and depth-averaged currents for all active
Slocum gliders compared against RTOFS, ESPC, and CMEMS model estimates.

Data source: IOOS National Glider DAC ERDDAP (https://gliders.ioos.us/erddap),
rather than Rutgers' own ERDDAP. The DAC exposes the same native Slocum sensors
under CF-friendly names (e.g. m_water_vx -> measured_water_vx) alongside its
own derived u/v (a straight rename of m_final_water_vx/vy, not rotated — see
magnetic-declination note below).

Plot 1 — Surface currents (GPS drift):
  Glider u/v derived from consecutive raw GPS fixes (gps_latitude/gps_longitude,
  i.e. native m_gps_lat/m_gps_lon) pulled from the DAC trajectory dataset, using
  only fixes that fall within the same surfacing event (Iridium call). The
  displacement between the first and last fix of that event divided by the
  elapsed time gives a clean surface-only drift estimate, with no contribution
  from the dive phase. Model currents are sampled at depth ≈ 0 m.

Plot 2 — Depth-averaged currents:
  Glider u/v from measured_water_vx/vy (native m_water_vx/vy — dead-reckoning
  depth-averaged current, referenced to the glider's magnetic-compass heading,
  not true north; see e.g. Merckelbach & Garau 2015, EGO glider processing
  recommendations, which flags this as an open/unaddressed issue). Rotated to
  true east/north using magnetic declination: the native measured_gps_mag_var
  sensor when available for that record, otherwise computed from position and
  time via the NOAA World Magnetic Model (pygeomag) — the same fallback used in
  scripts/tools/rotate_slocum_water_velocity.py. Model currents are vertically
  averaged over DEPTH_AVG_CONFIG depth range.
"""

import functools
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import datetime as dt
from pygeomag import GeoMag

from ioos_model_comparisons.calc import lon180to360
from ioos_model_comparisons.models import CMEMS, Doppio, espc_uv, rtofs as load_rtofs

# ==============================================================================
# Configuration
# ==============================================================================

# Run for all active Rutgers gliders on the IOOS Glider DAC (True),
# or only the single GLIDER_NAME below (False).
RUN_ALL_ACTIVE_GLIDERS = True

# Only datasets whose "institution" global attribute contains this substring
# are considered "ours" — the DAC hosts every provider's gliders, unlike
# Rutgers' own ERDDAP which only ever had Rutgers deployments.
INSTITUTION_FILTER = "Rutgers"

# Only used when RUN_ALL_ACTIVE_GLIDERS is False, or as a fallback label.
GLIDER_NAME = "ru29"

# A glider dataset is considered "active" (worth fetching at all) if its
# time_coverage_end is within this many days of now.
ACTIVE_GLIDER_THRESHOLD_DAYS = 2

# Even among active gliders, skip the plot if the most recent GPS surfacing is
# older than this many hours.  Prevents comparing stale glider positions against
# current model data when the glider has been underwater for a long time.
GLIDER_STALE_SURFACING_HOURS = 48

# Surfacing selection mode: "latest" or "range"
SURFACING_MODE = "latest"
# When using "range", set ISO-8601 timestamps for start/end (UTC recommended).
SURFACING_RANGE_START = "2026-01-01T00:00:00Z"
SURFACING_RANGE_END   = "2026-07-01T00:00:00Z"

# Model enable/disable flags
MODEL_CONFIG = {
    "RTOFS": {
        "enabled": True,
        "color": "red",
        "timeout": 60,
    },
    "ESPC": {
        "enabled": True,
        "color": "green",
        "timeout": 60,
    },
    "CMEMS": {
        "enabled": True,
        "color": "magenta",
        "timeout": 120,
    },
    "Doppio": {
        "enabled": False,
        "color": "darkorange",
        "timeout": 120,
    },
}

# Geographic footprint of the Doppio ROMS model [lon_min, lon_max, lat_min, lat_max].
# Gliders outside this box are silently skipped for Doppio comparisons.
DOPPIO_DOMAIN = [-82, -55, 25, 48]

# Depth-averaging configuration (used for Plot 2)
DEPTH_AVG_CONFIG = {
    "min_depth":  0.0,
    "max_depth":  1000.0,
    "depth_step": 1.0,
}

# Surface-drift GPS-fix clustering (used for Plot 1).  Consecutive m_gps_lat/
# m_gps_lon fixes from the ERDDAP trajectory dataset less than this many
# minutes apart are treated as belonging to the same surfacing event (i.e.
# the glider stayed on the surface between them); a larger gap means a dive
# happened in between.  Surface drift is computed strictly from the first-to-
# last fix *within* one event so it is never contaminated by dive current.
SURFACE_EVENT_GAP_MINUTES = 20.0

# How far back to search (from "now", or from the range end) for a complete
# (>=2-fix) surfacing event.
SURFACE_FIX_LOOKBACK_HOURS = 96.0

# How far back to search for the latest depth-avg current record when
# fetch_glider_depth_avg_erddap() is called with target_time=None.
LATEST_LOOKBACK_DAYS = 14.0

# save_path = '/Users/mikesmith/Documents/gliders/depth-average-dac/'   # local dev
# NOTE: deliberately a *separate* directory from glider_polar_currents.py's
# save_path — the two scripts pull from different ERDDAP servers and would
# otherwise race to overwrite each other's output if run side by side.
save_path = '/web/www/rucool_static/www/gliders/depth-average-dac'


# Sub-directory layout inside each glider's folder:
#   {save_path}/{glider}/currents/                   ← latest plots (overwritten)
#   {save_path}/{glider}/currents/archive/           ← timestamped archive
#   {save_path}/{glider}/maps/                       ← latest map aliases (from map script)
#   {save_path}/{glider}/maps/{model}/               ← timestamped map archive
CURRENTS_SUBDIR = "currents"
ARCHIVE_SUBDIR  = "archive"

# API configuration
GLIDER_API_BASE        = "https://marine.rutgers.edu/cool/data/gliders/api"
GLIDER_API_TIMEOUT     = 30
GLIDER_DEPLOYMENTS_URL = f"{GLIDER_API_BASE}/deployments/"

# ERDDAP configuration (IOOS National Glider DAC — trajectory data for
# measured_water_vx/vy). DAC dataset IDs are the bare deployment name (no
# suffix); real-time deployments and their eventual delayed-mode reprocessing
# get separate dataset IDs (the latter suffixed "-delayed"), so an explicit
# suffix filter excludes those rather than selects them.
ERDDAP_BASE               = "https://gliders.ioos.us/erddap/tabledap"
ERDDAP_DATASET_SUFFIX     = ""
ERDDAP_EXCLUDE_SUBSTRINGS = ("-delayed",)
ERDDAP_TIMEOUT            = 30

# Glider display configuration
GLIDER_COLOR = "blue"

# Depth dimension candidates for different models
DEPTH_DIM_CANDIDATES = ("depth", "Depth", "depthu", "depthv", "z")

# Retry configuration for dataset loading
DATASET_RETRY_ATTEMPTS      = 5
DATASET_RETRY_DELAY_SECONDS = 20

# Logging configuration
LOGGER_NAME     = "glider_polar_currents_dac"
LOG_LEVEL_NAME  = os.getenv("GLIDER_POLAR_DAC_LOG_LEVEL", "INFO").upper()
LOG_FILE        = os.getenv("GLIDER_POLAR_DAC_LOG_FILE")
LOG_FORMAT      = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> logging.Logger:
    """Configure a module-level logger with console and optional file output."""
    level  = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter      = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if LOG_FILE:
        file_path = Path(LOG_FILE).expanduser()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Unable to attach file handler %s: %s", file_path, exc)

    logging.captureWarnings(True)
    return logger


LOGGER = configure_logging()


# ==============================================================================
# Utility Functions
# ==============================================================================

def calc_speed_dir(u: float, v: float) -> Tuple[float, float]:
    """Return (speed m/s, direction radians) from u/v components."""
    if not np.isfinite(u) or not np.isfinite(v):
        return np.nan, np.nan
    speed     = float(np.hypot(u, v))
    direction = float(np.arctan2(v, u))
    return speed, direction


def calc_surface_drift_uv(
    lat1: float, lon1: float, t1: pd.Timestamp,
    lat2: float, lon2: float, t2: pd.Timestamp,
) -> Tuple[float, float]:
    """
    Estimate surface drift u/v from two consecutive GPS surfacing positions.

    Uses a flat-Earth approximation:
        u (east,  m/s) = Δlon_metres / Δt_seconds
        v (north, m/s) = Δlat_metres / Δt_seconds
    """
    dt_seconds = (t2 - t1).total_seconds()
    if dt_seconds <= 0:
        LOGGER.warning(
            "Non-positive time difference between surfacings (%.1f s); "
            "cannot compute surface drift.", dt_seconds,
        )
        return np.nan, np.nan

    mean_lat_rad   = np.radians((lat1 + lat2) / 2.0)
    meters_per_deg = 111_320.0

    delta_lat_m = (lat2 - lat1) * meters_per_deg
    delta_lon_m = (lon2 - lon1) * meters_per_deg * np.cos(mean_lat_rad)

    u = delta_lon_m / dt_seconds
    v = delta_lat_m / dt_seconds

    LOGGER.debug(
        "Surface drift: Δlat=%.1f m  Δlon=%.1f m  Δt=%.0f s  →  u=%.4f  v=%.4f m/s",
        delta_lat_m, delta_lon_m, dt_seconds, u, v,
    )
    return u, v


def safe_model_call(func, model_name: str, *args, **kwargs) -> Optional[Dict]:
    """Safely call a model sampling function with error handling."""
    if not MODEL_CONFIG.get(model_name, {}).get("enabled", False):
        LOGGER.info("%s disabled; skipping %s.", model_name, func.__name__)
        return None

    start_time = datetime.utcnow()
    LOGGER.info("%s: invoking %s", model_name, func.__name__)

    try:
        result  = func(*args, **kwargs)
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        if result and np.isfinite(result.get("speed", np.nan)):
            LOGGER.info(
                "%s completed in %.2fs (speed %.1f cm/s)",
                model_name, elapsed, result["speed"] * 100,
            )
        else:
            LOGGER.warning(
                "%s completed in %.2fs but returned no valid data.",
                model_name, elapsed,
            )
        return result

    except Exception:
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        LOGGER.exception("%s failed in %.2fs during %s", model_name, elapsed, func.__name__)
        return None


def load_dataset_with_retry(load_func, model_name: str):
    """Retry dataset loading a few times before giving up."""
    for attempt in range(1, DATASET_RETRY_ATTEMPTS + 1):
        try:
            return load_func()
        except Exception as exc:
            if attempt == DATASET_RETRY_ATTEMPTS:
                LOGGER.exception(
                    "%s dataset load failed after %d attempts",
                    model_name, DATASET_RETRY_ATTEMPTS,
                )
                raise
            LOGGER.warning(
                "%s load attempt %d/%d failed: %s; retrying in %ds",
                model_name, attempt, DATASET_RETRY_ATTEMPTS, exc, DATASET_RETRY_DELAY_SECONDS,
            )
            time.sleep(DATASET_RETRY_DELAY_SECONDS)


# ==============================================================================
# Deployment Resolution
# ==============================================================================

def resolve_latest_deployment(glider_name: str, timeout: int = 30) -> str:
    """Query the COOL glider deployments API and return the most recent
    deployment name for the given glider platform."""
    try:
        response = requests.get(GLIDER_DEPLOYMENTS_URL, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch deployments from {GLIDER_DEPLOYMENTS_URL}: {exc}"
        )

    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    matches = [
        r for r in records
        if r.get("glider_name", "").lower() == glider_name.lower()
    ]

    if not matches:
        raise ValueError(f"No deployments found for glider '{glider_name}'")

    latest = max(matches, key=lambda r: r.get("start_date_epoch") or 0)
    deployment_name = latest["deployment_name"]
    LOGGER.info("Resolved latest deployment for '%s': %s", glider_name, deployment_name)
    return deployment_name


def discover_active_gliders(
    active_threshold_days: int = ACTIVE_GLIDER_THRESHOLD_DAYS,
    timeout: int = GLIDER_API_TIMEOUT,
) -> List[Dict[str, str]]:
    """
    Query the IOOS Glider DAC's advanced search for active Rutgers glider
    datasets. Unlike Rutgers' own ERDDAP (which only ever hosts Rutgers
    deployments), the DAC hosts every provider's gliders in one catalog, so
    activity and institution are both filtered from a single search response
    rather than scanning + info-fetching the entire multi-hundred-dataset
    catalog.

    A dataset is considered active if its time range overlaps the window
    [now - active_threshold_days, now]. Only datasets whose "Institution"
    column contains INSTITUTION_FILTER are kept, and any dataset ID containing
    one of ERDDAP_EXCLUDE_SUBSTRINGS (e.g. "-delayed" reprocessed datasets) is
    dropped in favor of the continuously-updating real-time dataset.

    Returns a list of ``{"glider_name": ..., "deployment_id": ...}`` dicts.
    """
    erddap_root = ERDDAP_BASE.replace("/tabledap", "")
    now    = pd.Timestamp.utcnow().tz_localize(None)
    cutoff = now - pd.Timedelta(days=active_threshold_days)

    search_url = (
        f"{erddap_root}/search/advanced.csv"
        "?page=1&itemsPerPage=100000&searchFor="
        "&protocol=tabledap&cdm_data_type=(ANY)&institution=(ANY)"
        "&ioos_category=(ANY)&keywords=(ANY)&long_name=(ANY)"
        "&standard_name=(ANY)&variableName=(ANY)"
        "&minLon=(ANY)&maxLon=(ANY)&minLat=(ANY)&maxLat=(ANY)"
        f"&minTime={cutoff.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&maxTime={now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    try:
        resp = requests.get(search_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to search DAC ERDDAP catalog: {exc}") from exc

    if not resp.text.strip():
        LOGGER.warning("DAC search returned no active datasets in the last %d day(s).", active_threshold_days)
        return []

    search_df = pd.read_csv(io.StringIO(resp.text))
    search_df.columns = search_df.columns.str.strip()

    id_col   = next((c for c in search_df.columns if c.lower() == "dataset id"), None)
    inst_col = next((c for c in search_df.columns if c.lower() == "institution"), None)
    if id_col is None:
        raise RuntimeError(
            f"Cannot find 'Dataset ID' column in DAC search results. "
            f"Columns returned: {list(search_df.columns)}"
        )

    LOGGER.info("DAC search found %d active dataset(s) across all providers.", len(search_df))

    active: List[Dict[str, str]] = []
    for _, row in search_df.iterrows():
        dataset_id = str(row[id_col])

        if any(sub in dataset_id for sub in ERDDAP_EXCLUDE_SUBSTRINGS):
            LOGGER.debug("Skipping %s: matches an excluded suffix.", dataset_id)
            continue

        institution = str(row[inst_col]) if inst_col is not None else ""
        if INSTITUTION_FILTER not in institution:
            LOGGER.debug("Skipping %s: institution '%s' does not match filter.", dataset_id, institution)
            continue

        deployment_id = dataset_id[: -len(ERDDAP_DATASET_SUFFIX)] if ERDDAP_DATASET_SUFFIX else dataset_id

        # Glider name is everything before the trailing "-YYYYMMDDTHHmmSS" stamp.
        m = re.match(r"^(.+)-\d{8}T\d{4,6}$", deployment_id)
        glider_name = m.group(1) if m else deployment_id.split("-")[0]

        LOGGER.info("Active: %s (deployment=%s, institution=%s)", glider_name, deployment_id, institution)
        active.append({"glider_name": glider_name, "deployment_id": deployment_id})

    LOGGER.info("Discovered %d active Rutgers glider(s) on the DAC.", len(active))
    return active


# ==============================================================================
# Glider Data Functions
# ==============================================================================

def _dm_to_dd(dm_coord: float) -> float:
    """Convert degrees-minutes to decimal degrees."""
    sign     = 1 if dm_coord > 0 else -1
    dm_coord = abs(dm_coord)
    degrees  = int(dm_coord // 100)
    minutes  = dm_coord % 100
    return sign * (degrees + minutes / 60)


@functools.lru_cache(maxsize=64)
def _erddap_available_variables(deployment_id: str) -> frozenset:
    """
    Return the set of variable names present in a DAC dataset.

    DAC dataset schemas are not uniform across providers/processing versions:
    some expose the full raw native-sensor set (measured_water_vx/vy,
    measured_gps_mag_var, gps_latitude/longitude, ...) while others — notably
    plain real-time feeds — expose only the CF-mapped minimum (u, v, lat_uv,
    lon_uv, latitude, longitude). Every ERDDAP query below checks this set
    first and picks whichever variable names actually exist, rather than
    assuming one schema and failing on the other. Cached per deployment since
    it's queried once per fetch call within a single run.
    """
    dataset_id = f"{deployment_id}{ERDDAP_DATASET_SUFFIX}"
    erddap_root = ERDDAP_BASE.replace("/tabledap", "")
    info_url = f"{erddap_root}/info/{dataset_id}/index.csv"
    resp = requests.get(info_url, timeout=ERDDAP_TIMEOUT)
    resp.raise_for_status()
    info_df = pd.read_csv(io.StringIO(resp.text))
    info_df.columns = info_df.columns.str.strip()
    row_col = next((c for c in info_df.columns if c.lower() == "row type"), None)
    var_col = next((c for c in info_df.columns if c.lower() == "variable name"), None)
    if row_col is None or var_col is None:
        raise RuntimeError(f"Unexpected info columns for {dataset_id}: {list(info_df.columns)}")
    names = info_df.loc[info_df[row_col] == "variable", var_col].tolist()
    return frozenset(names)


def _fetch_gps_fixes_erddap(
    deployment_id: str,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Download raw-ish GPS fixes from the DAC trajectory dataset. Unlike the
    surfacings API, this returns *every* fix logged while the glider is on
    the surface — typically several per surfacing — which is what lets us
    compute drift from fixes that are all guaranteed to be surface-only.

    Prefers gps_latitude/gps_longitude (native m_gps_lat/m_gps_lon, degrees-
    decimal-minutes) when the dataset's schema exposes them. Some DAC
    schemas — notably plain real-time feeds — don't carry the raw sensor at
    all, only the CF-mapped per-profile latitude/longitude (already decimal
    degrees); those are used as a coarser fallback in that case.

    Returns DataFrame with columns: ts, lat, lon, sorted ascending by ts.
    """
    dataset_id     = f"{deployment_id}{ERDDAP_DATASET_SUFFIX}"
    available_vars = _erddap_available_variables(deployment_id)

    if {"gps_latitude", "gps_longitude"} <= available_vars:
        lat_var, lon_var, needs_dm_conversion = "gps_latitude", "gps_longitude", True
    else:
        LOGGER.info(
            "Deployment %s has no gps_latitude/gps_longitude; falling back to "
            "profile latitude/longitude for surface drift (coarser).",
            deployment_id,
        )
        lat_var, lon_var, needs_dm_conversion = "latitude", "longitude", False

    variables   = f"time,{lat_var},{lon_var}"
    constraints = f"&{lat_var}!=NaN&{lon_var}!=NaN"
    if start_time is not None:
        constraints += f"&time%3E={pd.Timestamp(start_time).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    if end_time is not None:
        constraints += f"&time%3C={pd.Timestamp(end_time).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    constraints += "&orderBy(%22time%22)"

    url = f"{ERDDAP_BASE}/{dataset_id}.csv?{variables}{constraints}"
    LOGGER.debug("Requesting GPS fixes for %s: %s", deployment_id, url)

    try:
        response = requests.get(url, timeout=ERDDAP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.exception("Failed to fetch GPS fixes for deployment %s", deployment_id)
        raise

    # ERDDAP CSV: row 0 = variable names, row 1 = units — skip the units row
    df = pd.read_csv(io.StringIO(response.text), skiprows=[1], parse_dates=["time"])
    df = df.dropna(subset=[lat_var, lon_var])

    if df.empty:
        raise ValueError(f"No GPS fixes found for deployment '{deployment_id}'")

    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    if needs_dm_conversion:
        df["lat"] = df[lat_var].apply(_dm_to_dd)
        df["lon"] = df[lon_var].apply(_dm_to_dd)
    else:
        df["lat"] = df[lat_var]
        df["lon"] = df[lon_var]
    data = (
        df.rename(columns={"time": "ts"})[["ts", "lat", "lon"]]
        .sort_values("ts")
        .reset_index(drop=True)
    )
    LOGGER.info("Retrieved %d GPS fixes for deployment %s", len(data), deployment_id)
    return data


def _cluster_surfacing_events(
    fixes: pd.DataFrame,
    gap_minutes: float = SURFACE_EVENT_GAP_MINUTES,
) -> List[pd.DataFrame]:
    """
    Split a chronologically-sorted DataFrame of GPS fixes into surfacing
    events: consecutive fixes less than gap_minutes apart belong to the same
    event; a larger gap indicates a dive happened in between.

    Returns a list of DataFrames, oldest event first.
    """
    if fixes.empty:
        return []

    gap = pd.Timedelta(minutes=gap_minutes)
    event_id = (fixes["ts"].diff() > gap).cumsum()
    return [group.reset_index(drop=True) for _, group in fixes.groupby(event_id)]


def _normalize_range_bound(value: Optional[str], label: str) -> Optional[pd.Timestamp]:
    """Normalize a range bound to a UTC timestamp for comparison."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} value: {value!r}") from exc


def _build_surface_record_from_event(event: pd.DataFrame) -> Dict:
    """Build a GPS-drift surface record from the first and last fix of a
    single surfacing event. Because both fixes are guaranteed to have been
    taken while the glider was on the surface, the resulting drift is a
    clean surface-only estimate — no dive-time correction is needed."""
    first = event.iloc[0]
    last  = event.iloc[-1]

    u, v = calc_surface_drift_uv(
        float(first["lat"]), float(first["lon"]), pd.Timestamp(first["ts"]),
        float(last["lat"]),  float(last["lon"]),  pd.Timestamp(last["ts"]),
    )
    speed, direction = calc_speed_dir(u, v)

    LOGGER.info(
        "Surface drift (%d fixes over %.0fs): (%.4f, %.4f) @ %s → (%.4f, %.4f) @ %s  "
        "=> u=%.4f v=%.4f speed=%.2f cm/s",
        len(event), (last["ts"] - first["ts"]).total_seconds(),
        first["lat"], first["lon"], first["ts"],
        last["lat"], last["lon"], last["ts"],
        u, v, speed * 100 if np.isfinite(speed) else float("nan"),
    )

    return {
        "source":        "Glider",
        "u":             u,
        "v":             v,
        "speed":         speed,
        "direction":     direction,
        "lat":           float(last["lat"]),
        "lon":           float(last["lon"]),
        "time":          pd.Timestamp(last["ts"]),
        "previous_lat":  float(first["lat"]),
        "previous_lon":  float(first["lon"]),
        "previous_time": pd.Timestamp(first["ts"]),
        "data_available": True,
    }


def get_latest_surface_record(deployment: str) -> Dict:
    """Return surface drift computed from the most recent complete (>=2-fix)
    surfacing event, using GPS-ish fixes from the DAC trajectory dataset
    (raw gps_latitude/gps_longitude when the schema has them, else the
    coarser per-profile latitude/longitude — see _fetch_gps_fixes_erddap).

    Deployments on the coarser fallback rarely produce a qualifying 2-fix
    event (consecutive profile positions are usually a full dive cycle
    apart, not consecutive fixes within one surfacing), so this position/
    time still needs to anchor the depth-avg comparison and model sampling
    even when a clean drift vector can't be formed. In that case, rather
    than raising and losing the whole surfacing (both plots), the most
    recent single fix is returned with data_available=False — Plot 1 simply
    shows no glider arrow/row for that run instead of aborting Plot 2 too.
    """
    now = pd.Timestamp.utcnow().tz_localize(None)
    fixes = _fetch_gps_fixes_erddap(
        deployment,
        start_time=now - pd.Timedelta(hours=SURFACE_FIX_LOOKBACK_HOURS),
        end_time=now,
    )
    events = _cluster_surfacing_events(fixes)
    for event in reversed(events):
        if len(event) >= 2:
            return _build_surface_record_from_event(event)

    LOGGER.warning(
        "No surfacing event with >=2 fixes found for '%s' in the last %.0fh; "
        "using the single most recent fix as the anchor (no surface-drift vector).",
        deployment, SURFACE_FIX_LOOKBACK_HOURS,
    )
    last = fixes.iloc[-1]
    record = _build_unavailable_vector_record("Glider")
    record["lat"]  = float(last["lat"])
    record["lon"]  = float(last["lon"])
    record["time"] = pd.Timestamp(last["ts"])
    return record


def get_surface_records_for_range(
    deployment: str,
    start_time: str,
    end_time: str,
) -> List[Dict]:
    """Return surface drift vectors for every complete surfacing event in a time range."""
    start_ts = _normalize_range_bound(start_time, "start_time")
    end_ts   = _normalize_range_bound(end_time,   "end_time")

    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("start_time must be earlier than end_time.")

    fixes = _fetch_gps_fixes_erddap(deployment, start_time=start_ts, end_time=end_ts)
    events = _cluster_surfacing_events(fixes)

    records: List[Dict] = []
    for event in events:
        if len(event) < 2:
            LOGGER.debug(
                "Skipping surfacing event with only %d fix(es) at %s.",
                len(event), event.iloc[0]["ts"],
            )
            continue
        records.append(_build_surface_record_from_event(event))

    LOGGER.info(
        "Built %d surface drift records between %s and %s",
        len(records), start_time, end_time,
    )
    return records


def _correct_glider_currents(u: float, v: float, mag_var_rad: float) -> Tuple[float, float]:
    """
    Rotate glider m_water_vx/vy from magnetic-north to true-north coordinates.

    Webb Research firmware convention:
        mag_heading = true_heading + m_gps_mag_var   (mag_var > 0 → westward)
        ∴  true_heading = mag_heading − m_gps_mag_var

    m_water_vx/vy are in LMC coordinates where Y points to magnetic north.
    m_gps_mag_var is stored in radians; convert to degrees for uv2spdir.
    """
    from oceans.ocfis import uv2spdir, spdir2uv

    mag_deg      = np.degrees(mag_var_rad)
    ang_mag, spd = uv2spdir(u, v)                   # direction in magnetic frame (degrees)
    ang_true     = ang_mag - mag_deg                 # TRUE = MAG − mag_var
    ul, vl       = spdir2uv(spd, ang_true, deg=True)
    return float(ul), float(vl)


@functools.lru_cache(maxsize=1)
def get_geomag_model() -> GeoMag:
    """Instantiate and cache the WMM (World Magnetic Model) calculator."""
    LOGGER.info("Instantiating pygeomag WMM model")
    return GeoMag()


def _decimal_year(when: pd.Timestamp) -> float:
    """Convert a timestamp to a decimal UTC year, as required by pygeomag."""
    ts = pd.Timestamp(when)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    year_start = datetime(ts.year, 1, 1, tzinfo=timezone.utc)
    next_start = datetime(ts.year + 1, 1, 1, tzinfo=timezone.utc)
    return ts.year + (ts.to_pydatetime() - year_start).total_seconds() / (
        next_start - year_start
    ).total_seconds()


def _resolve_mag_var_rad(
    lat: float, lon: float, when: pd.Timestamp, native_mag_var_rad: float,
) -> Tuple[float, str]:
    """
    Return (mag_var_rad, source) in the same sign convention as the glider's
    own measured_gps_mag_var sensor (positive = magnetic north west of true
    north — see _correct_glider_currents docstring).

    measured_gps_mag_var is used when present and finite for this record.
    Otherwise, since the DAC does not require it to be co-populated with
    every measured_water_vx/vy row (the two are logged on different onboard
    cycles and often fall on different rows once flattened into one table by
    ERDDAP), declination is instead computed from the record's own position
    and time via the NOAA World Magnetic Model (pygeomag) — the same fallback
    used in scripts/tools/rotate_slocum_water_velocity.py. WMM's declination
    output is in the opposite (east-positive) sign convention, so it is
    negated here before being handed to _correct_glider_currents.
    """
    if np.isfinite(native_mag_var_rad):
        return native_mag_var_rad, "measured_gps_mag_var"

    model  = get_geomag_model()
    result = model.calculate(glat=lat, glon=lon, alt=0.0, time=_decimal_year(when))
    mag_var_rad = -np.radians(result.d)
    LOGGER.info(
        "measured_gps_mag_var missing for this record; using WMM declination "
        "%.2f°E @ (%.3f, %.3f) %s instead.",
        result.d, lat, lon, when,
    )
    return mag_var_rad, "WMM (pygeomag)"


def fetch_glider_depth_avg_erddap(
    deployment_id: str,
    target_time: Optional[pd.Timestamp] = None,
    window_hours: float = 24.0,
) -> Dict:
    """
    Fetch the glider depth-averaged current from the DAC.

    Prefers the native measured_water_vx/vy (raw m_water_vx/vy) when the
    dataset's schema exposes them; otherwise falls back to the DAC's own u/v
    (a straight rename of m_final_water_vx/vy — same native LMC frame, just
    the delayed-mode/corrected variant of the sensor, and the only depth-avg
    current field some DAC schemas — notably plain real-time feeds — publish
    at all). Position/time come from the ordinary profile latitude/longitude/
    time columns rather than lat_uv/lon_uv/time_uv: on this ERDDAP install
    the *_uv coordinate variables come back with zero rows for every
    deployment tested (their cached actual_range appears to never get
    populated), even though the plain per-profile columns work fine and are
    coincident enough for a scatter-arrow plot.

    Queries the trajectory dataset for records within a time window (see
    below), then filters to non-NaN vx/vy client-side rather than with an
    ERDDAP-side "!=NaN" constraint: on this same install, "!=NaN" against
    these particular variables 404s with "the numeric variable's source
    min=, max=, and hasNaN=true" on at least some deployments — i.e. ERDDAP's
    own cached column-range stats are stale/unpopulated, independent of
    whether the column actually has data (confirmed via Rutgers' own ERDDAP,
    which shows real values for the same sensor on the same deployment).
    Filtering after download sidesteps that rather than depending on a stat
    that isn't reliably there. When target_time is given, searches within
    ±window_hours and returns the nearest record; when None, searches the
    last LATEST_LOOKBACK_DAYS and returns the most recent record.

    measured_gps_mag_var is *not* required to be non-NaN (nor does it need to
    exist in the dataset's schema at all): it's logged on a different onboard
    cycle than the current sensors, so once ERDDAP flattens the dataset into
    one row-per-timestamp table, plenty of valid water-velocity rows land on
    a timestamp with no co-located mag-var reading. Requiring it would
    silently drop usable current observations. Instead, missing declination
    is filled in per-record via WMM — see _resolve_mag_var_rad.

    Parameters
    ----------
    deployment_id : str
        Deployment name (e.g. 'ru29-20260623T2102')
    target_time : pd.Timestamp, optional
        Surfacing time to match against.  None → use the latest record.
    window_hours : float
        Half-width of the time search window around target_time.
    """
    dataset_id     = f"{deployment_id}{ERDDAP_DATASET_SUFFIX}"
    available_vars = _erddap_available_variables(deployment_id)

    if {"measured_water_vx", "measured_water_vy"} <= available_vars:
        vx_var, vy_var = "measured_water_vx", "measured_water_vy"
    elif {"u", "v"} <= available_vars:
        LOGGER.info(
            "Deployment %s has no measured_water_vx/vy; falling back to the "
            "DAC's u/v (same native sensor, delayed-mode variant).",
            deployment_id,
        )
        vx_var, vy_var = "u", "v"
    else:
        LOGGER.warning("Deployment %s exposes no depth-averaged current variables at all.", deployment_id)
        return _build_unavailable_vector_record("Glider")

    have_mag_var = "measured_gps_mag_var" in available_vars
    mag_var_field = ",measured_gps_mag_var" if have_mag_var else ""
    variables = f"time,latitude,longitude,{vx_var},{vy_var}{mag_var_field}"

    # %3E = '>', %3C = '<' — pre-encoded because requests passes the URL as-is
    # and Tomcat rejects raw < / >. No "!=NaN"/orderByMax here — see docstring.
    if target_time is None:
        t_start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=LATEST_LOOKBACK_DAYS))
        constraints = f"&time%3E={t_start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    else:
        t_start = (target_time - pd.Timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        t_end   = (target_time + pd.Timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        constraints = f"&time%3E={t_start}&time%3C={t_end}"

    url = f"{ERDDAP_BASE}/{dataset_id}.csv?{variables}{constraints}"
    LOGGER.info("Fetching glider depth-avg from DAC ERDDAP: %s", url)

    try:
        response = requests.get(url, timeout=ERDDAP_TIMEOUT)
        response.raise_for_status()
        # ERDDAP CSV: row 0 = variable names, row 1 = units — skip the units row
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1], parse_dates=["time"])
        df = df.dropna(subset=[vx_var, vy_var])
    except Exception as exc:
        LOGGER.error("DAC ERDDAP fetch failed: %s", exc)
        return _build_unavailable_vector_record("Glider")

    if df.empty:
        LOGGER.warning("No %s/%s records returned from DAC for %s.", vx_var, vy_var, dataset_id)
        return _build_unavailable_vector_record("Glider")

    # Normalise timestamps to tz-naive UTC for comparison
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)

    if target_time is not None:
        nearest_idx = (df["time"] - target_time).abs().idxmin()
        row = df.loc[nearest_idx]
    else:
        row = df.sort_values("time").iloc[-1]

    vx  = float(row[vx_var])
    vy  = float(row[vy_var])
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    t   = pd.to_datetime(row["time"])

    native_mag_rad          = float(row["measured_gps_mag_var"]) if have_mag_var else np.nan
    mag_rad, mag_var_source = _resolve_mag_var_rad(lat, lon, t, native_mag_rad)

    # The native water-velocity sensor is in LMC coordinates (Y = magnetic
    # north), whether read from measured_water_vx/vy or the DAC's u/v (which
    # is an unrotated rename of the same sensor family — see module
    # docstring). Rotate by the magnetic declination to convert to true
    # east/north.
    vx, vy = _correct_glider_currents(vx, vy, mag_rad)

    speed, direction = calc_speed_dir(vx, vy)

    LOGGER.info(
        "Glider depth-avg (DAC, mag-corrected via %s): u=%.4f m/s  v=%.4f m/s  "
        "speed=%.2f cm/s  mag_var=%.2f° @ %s",
        mag_var_source, vx, vy,
        speed * 100 if np.isfinite(speed) else float("nan"),
        np.degrees(mag_rad),
        t,
    )
    return {
        "source":        "Glider",
        "u":             vx,
        "v":             vy,
        "speed":         speed,
        "direction":     direction,
        "lat":           lat,
        "lon":           lon,
        "time":          t,
        "data_available": True,
    }


# ==============================================================================
# Model Data Functions
# ==============================================================================

@functools.lru_cache(maxsize=1)
def get_rtofs_dataset():
    """Load and cache the RTOFS dataset at the surface layer only (Plot 1)."""
    if not MODEL_CONFIG["RTOFS"]["enabled"]:
        return None

    def _load():
        return load_rtofs().isel(depth=0)

    return load_dataset_with_retry(_load, "RTOFS")


@functools.lru_cache(maxsize=1)
def get_rtofs_full_dataset():
    """Load and cache the full-depth RTOFS dataset (Plot 2 depth-averaging)."""
    if not MODEL_CONFIG["RTOFS"]["enabled"]:
        return None

    def _load():
        return load_rtofs()

    return load_dataset_with_retry(_load, "RTOFS")


@functools.lru_cache(maxsize=1)
def get_espc_dataset():
    """Load and cache the ESPC dataset (full depth; used for both plots)."""
    if not MODEL_CONFIG["ESPC"]["enabled"]:
        return None

    def _load():
        return espc_uv(rename=True)

    return load_dataset_with_retry(_load, "ESPC")


@functools.lru_cache(maxsize=1)
def get_cmems_client():
    """Instantiate and cache the CMEMS helper."""
    if not MODEL_CONFIG["CMEMS"]["enabled"]:
        return None
    LOGGER.info("Instantiating CMEMS client")
    return CMEMS()


@functools.lru_cache(maxsize=1)
def get_doppio_instance():
    """Instantiate and cache the Doppio object (reads static grid metadata only)."""
    if not MODEL_CONFIG["Doppio"]["enabled"]:
        return None
    LOGGER.info("Instantiating Doppio instance")
    return Doppio()


def _doppio_nearest_ij(lon2d, lat2d, lon: float, lat: float):
    """Return (iy, ix) of the nearest cell in a 2-D curvilinear grid."""
    dist2 = (lon2d - lon) ** 2 + (lat2d - lat) ** 2
    iy, ix = np.unravel_index(int(np.nanargmin(dist2)), dist2.shape)
    return iy, ix


def sample_doppio_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Nearest-neighbour sample of Doppio surface layer at a single point."""
    dop = get_doppio_instance()
    if dop is None:
        return None

    ds = dop.sel(time=when)
    iy, ix = _doppio_nearest_ij(dop._lon_rho, dop._lat_rho, lon, lat)
    pt = ds.isel(depth=0, y=iy, x=ix).squeeze()

    u = float(pt["u"].values)
    v = float(pt["v"].values)
    LOGGER.info("Doppio surface: u=%.4f m/s  v=%.4f m/s", u, v)
    return _build_vector_record(
        "Doppio", u, v,
        float(pt["lon"].values), float(pt["lat"].values), pt["time"].values,
    )


def sample_doppio_depth_avg_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Nearest-neighbour sample of Doppio, then depth-average the profile."""
    dop = get_doppio_instance()
    if dop is None:
        return None

    ds = dop.sel(time=when)
    iy, ix = _doppio_nearest_ij(dop._lon_rho, dop._lat_rho, lon, lat)
    profile = ds.isel(y=iy, x=ix).squeeze()   # dims: depth

    u, v = compute_depth_avg_uv(
        profile,
        min_depth=DEPTH_AVG_CONFIG["min_depth"],
        max_depth=DEPTH_AVG_CONFIG["max_depth"],
        depth_step=DEPTH_AVG_CONFIG["depth_step"],
    )
    LOGGER.info("Doppio depth-avg: u=%.4f m/s  v=%.4f m/s", u, v)
    return _build_vector_record(
        "Doppio", u, v,
        float(profile["lon"].values), float(profile["lat"].values), profile["time"].values,
    )


def _build_vector_record(
    source: str, u: float, v: float,
    lon: float, lat: float, when: pd.Timestamp,
) -> Dict:
    """Build a standardized vector record."""
    speed, direction = calc_speed_dir(u, v)
    return {
        "source":        source,
        "u":             float(u),
        "v":             float(v),
        "speed":         speed,
        "direction":     direction,
        "lat":           float(lat),
        "lon":           float(lon),
        "time":          pd.to_datetime(when),
        "data_available": True,
    }


def _build_unavailable_vector_record(source: str) -> Dict:
    return {
        "source":        source,
        "u":             np.nan,
        "v":             np.nan,
        "speed":         np.nan,
        "direction":     np.nan,
        "lat":           np.nan,
        "lon":           np.nan,
        "time":          pd.NaT,
        "data_available": False,
    }


def compute_depth_avg_uv(
    ds,
    min_depth: float = 0.0,
    max_depth: float = 1000.0,
    depth_step: float = 1.0,
) -> Tuple[float, float]:
    """
    Depth-average u and v from a single-point depth-profile dataset.

    Handles staggered grids (e.g. ESPC depthu/depthv) by averaging each
    variable independently against its own depth coordinate.  Depth values
    that are all negative are flipped to positive before processing.

    Returns:
        u, v: depth-averaged eastward and northward velocity (m/s)
    """
    def _avg_variable(da):
        depth_dim = next(
            (d for d in DEPTH_DIM_CANDIDATES if d in da.dims), None
        )
        if depth_dim is None:
            return float(da.values) if da.size == 1 else np.nan

        depth_vals = da[depth_dim].values.astype(float)
        if np.all(depth_vals <= 0):
            depth_vals = np.abs(depth_vals)
            da = da.assign_coords({depth_dim: depth_vals})

        start = max(min_depth, float(np.nanmin(depth_vals)))
        end   = min(max_depth, float(np.nanmax(depth_vals)))
        if end <= start:
            LOGGER.warning(
                "compute_depth_avg_uv: requested range [%.0f, %.0f] m outside "
                "available data [%.0f, %.0f] m.",
                min_depth, max_depth, float(np.nanmin(depth_vals)), float(np.nanmax(depth_vals)),
            )
            return np.nan

        da = da.sortby(depth_dim)
        da_trim = da.sel({depth_dim: slice(start, end)})
        target_depths = np.arange(start, end + depth_step, depth_step)
        da_interp = da_trim.interp({depth_dim: target_depths})
        return float(da_interp.mean(skipna=True).values)

    u = _avg_variable(ds["u"])
    v = _avg_variable(ds["v"])
    return u, v


# ==============================================================================
# Model Sampling - Surface Point Interpolation (Plot 1)
# ==============================================================================

def sample_rtofs_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate RTOFS surface layer to a single point in space/time."""
    ds = get_rtofs_dataset()
    if ds is None:
        LOGGER.warning("RTOFS dataset unavailable; cannot sample surface point.")
        return None

    ds = ds[["u", "v"]]

    lon_grid  = ds.lon.isel(y=0).values
    lat_grid  = ds.lat.isel(x=0).values
    x_vals    = ds.x.values
    y_vals    = ds.y.values
    lon_order = np.argsort(lon_grid)
    lat_order = np.argsort(lat_grid)

    lon_index = np.interp(lon, lon_grid[lon_order], x_vals[lon_order])
    lat_index = np.interp(lat, lat_grid[lat_order], y_vals[lat_order])

    point = ds.interp(
        time=np.datetime64(when),
        x=lon_index,
        y=lat_index,
        kwargs={"fill_value": "extrapolate"},
    ).squeeze().compute()

    u = float(point["u"].values)
    v = float(point["v"].values)
    return _build_vector_record(
        "RTOFS", u, v,
        float(point.lon.values), float(point.lat.values), point.time.values,
    )


def sample_espc_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate ESPC surface layer to a single point in space/time."""
    ds = get_espc_dataset()
    if ds is None:
        LOGGER.warning("ESPC dataset unavailable; cannot sample surface point.")
        return None

    lon_target = lon180to360(lon)
    profile = ds.interp(
        time=np.datetime64(when),
        lon=lon_target,
        lat=lat,
    ).squeeze()

    # ESPC uses staggered grids (depthu/depthv, lat_v, etc.); collapse all
    # leftover dims on each variable independently to reach a scalar.
    u_da = profile["u"]
    v_da = profile["v"]
    for dim in list(u_da.dims):
        u_da = u_da.isel({dim: 0})
    for dim in list(v_da.dims):
        v_da = v_da.isel({dim: 0})

    u = float(u_da.values)
    v = float(v_da.values)
    return _build_vector_record(
        "ESPC", u, v,
        float(profile.lon.values), float(profile.lat.values), profile.time.values,
    )


def sample_cmems_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate CMEMS surface layer to a single point in space/time."""
    client = get_cmems_client()
    if client is None:
        LOGGER.warning("CMEMS client unavailable; cannot sample surface point.")
        return None

    profile = client.get_point(
        lon, lat, when,
        interp=True,
        vars=["currents"],
    ).squeeze()

    for depth_candidate in DEPTH_DIM_CANDIDATES:
        if depth_candidate in profile.dims:
            profile = profile.isel({depth_candidate: 0})

    u = float(profile["u"].values)
    v = float(profile["v"].values)
    return _build_vector_record(
        "CMEMS", u, v,
        float(profile.lon.values), float(profile.lat.values), profile.time.values,
    )


# ==============================================================================
# Model Sampling - Depth-Averaged Point (Plot 2)
# ==============================================================================

def sample_rtofs_depth_avg_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate RTOFS to a single x/y/time point, then depth-average."""
    ds = get_rtofs_full_dataset()
    if ds is None:
        LOGGER.warning("RTOFS full dataset unavailable; cannot compute depth-avg.")
        return None

    ds = ds[["u", "v"]]

    lon_grid  = ds.lon.isel(y=0).values
    lat_grid  = ds.lat.isel(x=0).values
    x_vals    = ds.x.values
    y_vals    = ds.y.values
    lon_order = np.argsort(lon_grid)
    lat_order = np.argsort(lat_grid)

    lon_index = np.interp(lon, lon_grid[lon_order], x_vals[lon_order])
    lat_index = np.interp(lat, lat_grid[lat_order], y_vals[lat_order])

    profile = ds.interp(
        time=np.datetime64(when),
        x=lon_index,
        y=lat_index,
        kwargs={"fill_value": "extrapolate"},
    ).squeeze().compute()

    u, v = compute_depth_avg_uv(
        profile,
        min_depth=DEPTH_AVG_CONFIG["min_depth"],
        max_depth=DEPTH_AVG_CONFIG["max_depth"],
        depth_step=DEPTH_AVG_CONFIG["depth_step"],
    )
    LOGGER.info("RTOFS depth-avg: u=%.4f m/s  v=%.4f m/s", u, v)
    return _build_vector_record(
        "RTOFS", u, v,
        float(profile.lon.values), float(profile.lat.values), profile.time.values,
    )


def sample_espc_depth_avg_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate ESPC to a single lon/lat/time point, then depth-average."""
    ds = get_espc_dataset()
    if ds is None:
        LOGGER.warning("ESPC dataset unavailable; cannot compute depth-avg.")
        return None

    lon_target = lon180to360(lon)
    profile = ds.interp(
        time=np.datetime64(when),
        lon=lon_target,
        lat=lat,
    ).squeeze()

    u, v = compute_depth_avg_uv(
        profile,
        min_depth=DEPTH_AVG_CONFIG["min_depth"],
        max_depth=DEPTH_AVG_CONFIG["max_depth"],
        depth_step=DEPTH_AVG_CONFIG["depth_step"],
    )
    LOGGER.info("ESPC depth-avg: u=%.4f m/s  v=%.4f m/s", u, v)
    return _build_vector_record(
        "ESPC", u, v,
        float(profile.lon.values), float(profile.lat.values), profile.time.values,
    )


def sample_cmems_depth_avg_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Fetch full CMEMS depth profile at a point, then depth-average."""
    client = get_cmems_client()
    if client is None:
        LOGGER.warning("CMEMS client unavailable; cannot compute depth-avg.")
        return None

    profile = client.get_point(
        lon, lat, when,
        interp=True,
        vars=["currents"],
    ).squeeze()

    u, v = compute_depth_avg_uv(
        profile,
        min_depth=DEPTH_AVG_CONFIG["min_depth"],
        max_depth=DEPTH_AVG_CONFIG["max_depth"],
        depth_step=DEPTH_AVG_CONFIG["depth_step"],
    )
    LOGGER.info("CMEMS depth-avg: u=%.4f m/s  v=%.4f m/s", u, v)
    return _build_vector_record(
        "CMEMS", u, v,
        float(profile.lon.values), float(profile.lat.values), profile.time.values,
    )


# ==============================================================================
# Visualization Functions
# ==============================================================================

def build_summary_table(vector_records: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame summarizing u/v/speed/direction for plotting."""
    def fmt_heading(value, available=True):
        if not available:
            return "N/A"
        return f"{value:.1f}" if np.isfinite(value) else "--"

    def fmt_velocity_cm(value, available=True):
        if not available:
            return "N/A"
        return f"{value * 100:.1f}" if np.isfinite(value) else "--"

    rows = []
    for record in vector_records:
        avail = record.get("data_available", True)
        if avail and np.isfinite(record["direction"]):
            direction_deg = (90 - np.degrees(record["direction"])) % 360
        else:
            direction_deg = np.nan

        rows.append({
            "Source":               record["source"],
            "u\n(+east)\n(cm/s)":   fmt_velocity_cm(record["u"],     avail),
            "v\n(+north)\n(cm/s)":  fmt_velocity_cm(record["v"],     avail),
            "Speed\n(cm/s)":        fmt_velocity_cm(record["speed"], avail),
            "Heading\n(°CWN)":      fmt_heading(direction_deg,       avail),
        })

    return pd.DataFrame(rows)


def _build_output_path(
    surfacing_time: Optional[pd.Timestamp] = None,
    base_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> Path:
    """Build output path, optionally appending surfacing time for uniqueness."""
    base_dir  = Path(base_dir) if base_dir is not None else Path(save_path)
    if filename is None:
        raise ValueError("filename must be provided to _build_output_path")
    base_path = base_dir / filename

    if surfacing_time is None:
        return base_path

    timestamp = pd.to_datetime(surfacing_time)
    if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)

    suffix = timestamp.strftime("%Y%m%dT%H%M%S")
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def plot_vectors(
    vector_records: List[Dict],
    summary_df: pd.DataFrame,
    glider_info: Dict,
    deployment_id: str,
    enabled_models: List[str],
    glider_label: str = "Glider",
    plot_title: str = "Surface Currents (GPS Drift)",
    output_files: Optional[List[Path]] = None,
):
    """Create a polar vector plot with companion summary table."""
    LOGGER.info("Rendering '%s' vector plot for %d records", plot_title, len(vector_records))

    color_map = {"Glider": GLIDER_COLOR}
    for model_name in enabled_models:
        if model_name in MODEL_CONFIG:
            color_map[model_name] = MODEL_CONFIG[model_name]["color"]

    fig = plt.figure(figsize=(18, 8))
    ax  = fig.add_subplot(121, projection="polar")

    speed_values  = np.array([r["speed"] for r in vector_records], dtype=float)
    finite_speeds = speed_values[np.isfinite(speed_values)]
    max_speed_ms  = float(finite_speeds.max()) if finite_speeds.size else 1.0
    max_speed_cm  = max_speed_ms * 100

    def _plot_arrow(record, is_glider: bool):
        speed_cm = record["speed"] * 100
        ax.arrow(
            record["direction"], 0, 0, speed_cm,
            head_width=0.12,
            head_length=3,
            fc=color_map.get(record["source"], "black"),
            ec=color_map.get(record["source"], "black"),
            linewidth=3 if is_glider else 2.3,
            length_includes_head=True,
            zorder=15 if is_glider else 5,
            label=record["source"],
            alpha=0.9 if is_glider else 0.75,
        )

    non_glider  = [r for r in vector_records if r["source"] != "Glider"]
    glider_recs = [r for r in vector_records if r["source"] == "Glider"]

    for record in non_glider:
        if np.isfinite(record["speed"]):
            _plot_arrow(record, is_glider=False)
    for record in glider_recs:
        if np.isfinite(record["speed"]):
            _plot_arrow(record, is_glider=True)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["E", "", "N", "", "W", "", "S", ""], fontsize=35, fontweight="bold")
    ax.tick_params(axis="x", pad=25)

    outer_radius = max_speed_cm * 1.2 if np.isfinite(max_speed_cm) else 1.0
    ax.set_ylim(0, outer_radius)
    ax.set_ylabel("Speed Rings (cm/s)", fontsize=20, fontweight="bold", labelpad=60)

    reticle_radius = outer_radius * 1.05
    for angle_deg in (0, 90, 180, 270):
        angle_rad = np.deg2rad(angle_deg)
        ax.plot(
            [angle_rad, angle_rad], [0, reticle_radius],
            color="#1f2933", linewidth=1.4, alpha=0.8, zorder=1, clip_on=False,
        )

    ax.set_rlabel_position(180)
    ax.grid(True, color="#9ca3af", alpha=0.5, linestyle="--", linewidth=1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.1), framealpha=0.95, fontsize=15)

    # Summary table
    table_ax = fig.add_subplot(122)
    table_ax.axis("off")

    disabled_models = [m for m in MODEL_CONFIG if not MODEL_CONFIG[m]["enabled"]]

    title_text  = f"{plot_title}\n"
    title_text += f"Glider: {glider_label.upper()}  |  Deployment: {deployment_id}\n"
    title_text += f"Location: {glider_info['lat']:.3f}°N, {abs(glider_info['lon']):.3f}°W\n"
    title_text += f"Time (UTC): {glider_info['time']}"

    if disabled_models:
        title_text += f" | Disabled: {', '.join(disabled_models)}"

    table_ax.text(
        0.5, 0.94, title_text,
        transform=table_ax.transAxes,
        ha="center", fontsize=20, fontweight="bold", linespacing=1.5,
    )

    table = table_ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.06, 0.06, 0.06, 0.06, 0.06],
        bbox=[-0.10, 0.05, 1.05, 0.8],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 2.2)

    for idx in range(len(summary_df.columns)):
        cell = table[(0, idx)]
        cell.set_facecolor("#2C5282")
        cell.set_text_props(weight="bold", color="white", fontsize=18)

    for idx in range(1, len(summary_df) + 1):
        cell   = table[(idx, 0)]
        source = summary_df.iloc[idx - 1]["Source"]
        if source in color_map:
            cell.set_text_props(color=color_map[source], weight="bold")
        else:
            cell.set_text_props(weight="bold")

    generated_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    fig.text(
        0.95, 0.01, f"Image generated: {generated_time}",
        transform=fig.transFigure,
        ha="right", va="bottom", fontsize=7, fontweight="bold",
    )

    plt.tight_layout()
    if output_files is None:
        output_files = []
    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving figure to %s", output_file)
        plt.savefig(output_file, dpi=150)
    plt.close(fig)


# ==============================================================================
# Main Execution
# ==============================================================================

def print_model_status():
    LOGGER.info("Model Configuration:")
    for model_name, config in MODEL_CONFIG.items():
        LOGGER.info("%s: %s", model_name, "enabled" if config["enabled"] else "disabled")


def process_surface_record(
    glider_record: Dict,
    deployment_id: str,
    glider_name: str = GLIDER_NAME,
    surface_output_files: Optional[List[Path]] = None,
    depth_avg_output_files: Optional[List[Path]] = None,
) -> None:
    """Run model comparisons and produce both polar plots for one surfacing."""
    LOGGER.info(
        "Glider surface position: %.3f°N, %.3f°%s @ %s",
        glider_record["lat"],
        abs(glider_record["lon"]),
        "W" if glider_record["lon"] < 0 else "E",
        glider_record["time"],
    )

    glon, glat = glider_record["lon"], glider_record["lat"]
    in_doppio_domain = (
        MODEL_CONFIG["Doppio"]["enabled"]
        and DOPPIO_DOMAIN[0] <= glon <= DOPPIO_DOMAIN[1]
        and DOPPIO_DOMAIN[2] <= glat <= DOPPIO_DOMAIN[3]
    )
    if in_doppio_domain:
        LOGGER.info("Glider at (%.3f, %.3f) is within Doppio domain.", glat, glon)
    else:
        LOGGER.info("Glider at (%.3f, %.3f) is outside Doppio domain; skipping.", glat, glon)

    enabled_models = [
        m for m in MODEL_CONFIG
        if MODEL_CONFIG[m]["enabled"] and (m != "Doppio" or in_doppio_domain)
    ]

    when = glider_record["time"]

    # ------------------------------------------------------------------
    # Fire all remote calls concurrently — surface, depth-avg, and the
    # glider ERDDAP fetch are all independent network operations.
    # ------------------------------------------------------------------
    tasks: Dict[str, Callable[[], Any]] = {
        "rtofs_sfc":  lambda: safe_model_call(sample_rtofs_surface_point,    "RTOFS", glon, glat, when),
        "espc_sfc":   lambda: safe_model_call(sample_espc_surface_point,     "ESPC",  glon, glat, when),
        "cmems_sfc":  lambda: safe_model_call(sample_cmems_surface_point,    "CMEMS", glon, glat, when),
        "glider_da":  lambda: fetch_glider_depth_avg_erddap(deployment_id, target_time=when),
        "rtofs_da":   lambda: safe_model_call(sample_rtofs_depth_avg_point,  "RTOFS", glon, glat, when),
        "espc_da":    lambda: safe_model_call(sample_espc_depth_avg_point,   "ESPC",  glon, glat, when),
        "cmems_da":   lambda: safe_model_call(sample_cmems_depth_avg_point,  "CMEMS", glon, glat, when),
    }
    if in_doppio_domain:
        tasks["doppio_sfc"] = lambda: safe_model_call(sample_doppio_surface_point,   "Doppio", glon, glat, when)
        tasks["doppio_da"]  = lambda: safe_model_call(sample_doppio_depth_avg_point, "Doppio", glon, glat, when)

    res: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        future_map = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                res[key] = future.result()
            except Exception as exc:
                LOGGER.error("Concurrent task '%s' raised: %s", key, exc)
                res[key] = None

    rtofs_sfc  = res.get("rtofs_sfc")
    espc_sfc   = res.get("espc_sfc")
    cmems_sfc  = res.get("cmems_sfc")
    doppio_sfc = res.get("doppio_sfc")
    glider_da_record = res.get("glider_da")
    rtofs_da   = res.get("rtofs_da")
    espc_da    = res.get("espc_da")
    cmems_da   = res.get("cmems_da")
    doppio_da  = res.get("doppio_da")

    # ------------------------------------------------------------------
    # Plot 1 — Surface currents
    # ------------------------------------------------------------------
    surface_records = [glider_record]
    for model_name, record in [
        ("RTOFS",  rtofs_sfc),
        ("ESPC",   espc_sfc),
        ("CMEMS",  cmems_sfc),
        ("Doppio", doppio_sfc),
    ]:
        if record is not None:
            surface_records.append(record)
        elif model_name != "Doppio" and MODEL_CONFIG.get(model_name, {}).get("enabled", False):
            surface_records.append(_build_unavailable_vector_record(model_name))
        elif model_name == "Doppio" and in_doppio_domain:
            surface_records.append(_build_unavailable_vector_record(model_name))

    summary_sfc = build_summary_table(surface_records)
    LOGGER.info("=" * 60)
    LOGGER.info("SURFACE RESULTS")
    LOGGER.info("\n%s", summary_sfc.to_string(index=False))
    LOGGER.info("-" * 60)

    plot_vectors(
        surface_records, summary_sfc, glider_record, deployment_id, enabled_models,
        glider_label=glider_name,
        plot_title="Surface Currents (GPS Drift)",
        output_files=surface_output_files,
    )

    # ------------------------------------------------------------------
    # Plot 2 — Depth-averaged currents
    # ------------------------------------------------------------------
    depth_avg_records = [glider_da_record]
    for model_name, record in [
        ("RTOFS",  rtofs_da),
        ("ESPC",   espc_da),
        ("CMEMS",  cmems_da),
        ("Doppio", doppio_da),
    ]:
        if record is not None:
            depth_avg_records.append(record)
        elif model_name != "Doppio" and MODEL_CONFIG.get(model_name, {}).get("enabled", False):
            depth_avg_records.append(_build_unavailable_vector_record(model_name))
        elif model_name == "Doppio" and in_doppio_domain:
            depth_avg_records.append(_build_unavailable_vector_record(model_name))

    summary_da = build_summary_table(depth_avg_records)
    LOGGER.info("=" * 60)
    LOGGER.info("DEPTH-AVERAGED RESULTS (%.0f–%.0f m)",
                DEPTH_AVG_CONFIG["min_depth"], DEPTH_AVG_CONFIG["max_depth"])
    LOGGER.info("\n%s", summary_da.to_string(index=False))
    LOGGER.info("-" * 60)

    min_d = int(DEPTH_AVG_CONFIG["min_depth"])
    max_d = int(DEPTH_AVG_CONFIG["max_depth"])
    plot_vectors(
        depth_avg_records, summary_da, glider_da_record, deployment_id, enabled_models,
        glider_label=glider_name,
        plot_title=f"Depth-Averaged Currents ({min_d}–{max_d} m, m_water_vx/vy, true-north via DAC)",
        output_files=depth_avg_output_files,
    )


# ==============================================================================
# Website Generation
# ==============================================================================

_RUCOOL_RED   = "#CC0033"
_RUCOOL_DARK  = "#A00028"
_RUCOOL_LOGO  = "https://rucool.marine.rutgers.edu/hfradmin/static/img/RU_Cool-favicon-32x32.png"
_BS5_CSS      = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/css/bootstrap.min.css"
_BS5_JS       = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js"
_FA_CSS       = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"

_SHARED_CSS = f"""
    :root {{
      --ru: {_RUCOOL_RED}; --ru-dark: {_RUCOOL_DARK};
      --bg: #f0f0f0; --border: #e2e8f0; --dark: #1e293b;
      --sb: #ffffff; --sb-hover: #fff5f7;
      --tr: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }}
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html,body {{ height:100%; background:var(--bg); font-family:Arial,sans-serif;
                 font-size:.95rem; color:var(--dark); padding-bottom:30px; }}
    .navbar {{ background:var(--ru); box-shadow:0 4px 6px rgba(0,0,0,.1); }}
    .navbar-brand {{ font-weight:700; font-size:1.3rem; color:#fff!important; gap:.75rem; }}
    .navbar-nav .nav-link {{ color:rgba(255,255,255,.9)!important; padding:.5rem 1rem!important;
                              border-radius:6px; font-weight:500; transition:var(--tr); }}
    .navbar-nav .nav-link:hover {{ background:rgba(255,255,255,.15); color:#fff!important; }}
    .footer-bar {{ position:fixed; bottom:0; left:0; right:0; height:30px;
                   background:var(--ru-dark); display:flex; align-items:center;
                   justify-content:flex-end; padding:0 1.5rem; z-index:1030;
                   color:rgba(255,255,255,.75); font-size:.75rem; }}
    #sidebar {{ width:220px; min-height:calc(100vh - 66px - 30px); background:var(--sb);
                padding:1rem 0; border-right:1px solid var(--border);
                box-shadow:2px 0 8px rgba(0,0,0,.05); overflow-y:auto; flex-shrink:0; }}
    .sb-item {{ border:none; background:transparent; padding:.75rem 1rem;
                margin:0 .75rem; border-radius:8px; color:var(--dark); font-weight:500;
                text-decoration:none; display:flex; align-items:center; gap:.5rem;
                transition:var(--tr); }}
    .sb-item:hover {{ background:var(--sb-hover); transform:translateX(4px); color:var(--dark); }}
    .sb-item.active {{ background:var(--ru); color:#fff; }}
    .sb-label {{ font-size:.75rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:.05em; color:#94a3b8; padding:.5rem 1.75rem .25rem; }}
    #body-row {{ display:flex; min-height:calc(100vh - 66px - 30px); }}
    #main {{ flex:1; padding:1.25rem; min-width:0; }}
    .page-title {{ font-size:1.1rem; font-weight:700; color:var(--ru); margin-bottom:1rem; }}
    .nav-tabs .nav-link {{ color:var(--dark); border:none;
                           border-bottom:3px solid transparent; background:transparent;
                           font-weight:500; padding:.5rem 1rem; }}
    .nav-tabs .nav-link:hover {{ color:var(--ru); }}
    .nav-tabs .nav-link.active {{ color:var(--ru); border-bottom-color:var(--ru); background:transparent; }}
    .nav-tabs {{ border-bottom:2px solid var(--border); }}
    .card {{ border:1px solid var(--border); border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
    .card-header {{ background:#f8fafc; border-bottom:1px solid var(--border); padding:0 .75rem; }}
    .card-header-tabs .nav-link {{ font-size:.82rem; padding:.5rem .75rem; font-weight:500;
                                    color:#555; border:none; border-bottom:2px solid transparent;
                                    background:transparent; margin-bottom:-1px; }}
    .card-header-tabs .nav-link:hover {{ color:var(--ru); }}
    .card-header-tabs .nav-link.active {{ color:var(--ru); border-bottom-color:var(--ru);
                                           background:transparent; }}
    .thumb {{ cursor:pointer; border-radius:4px; border:2px solid transparent;
               transition:var(--tr); max-width:100%; height:auto; }}
    .thumb:hover {{ border-color:var(--ru); transform:scale(1.02); }}
    .fleet-card {{ transition:var(--tr); text-decoration:none; color:inherit; }}
    .fleet-card:hover {{ transform:translateY(-3px); box-shadow:0 6px 20px rgba(0,0,0,.12)!important;
                          color:inherit; }}
    .hamburger {{ display:none; background:none; border:none; color:#fff;
                  font-size:1.4rem; cursor:pointer; padding:.3rem .5rem;
                  margin-right:.5rem; line-height:1; }}
    #sidebar-overlay {{ display:none; position:fixed; inset:0;
                        background:rgba(0,0,0,.4); z-index:1020; }}
    @media (max-width:991px) {{
      .hamburger {{ display:block; }}
      #sidebar {{ position:fixed; top:66px; left:0; bottom:30px; z-index:1025;
                  transform:translateX(-100%); transition:transform .25s ease;
                  min-height:unset; }}
      #sidebar.open {{ transform:translateX(0); }}
      #sidebar-overlay.open {{ display:block; }}
      .nav-tabs {{ flex-wrap:nowrap; overflow-x:auto;
                   -webkit-overflow-scrolling:touch; scrollbar-width:none; }}
      .nav-tabs::-webkit-scrollbar {{ display:none; }}
      .nav-tabs .nav-link {{ flex-shrink:0; white-space:nowrap; }}
    }}
"""


def _slug_to_label(slug: str) -> str:
    """Convert a map filename slug like 'rtofs-depthavg-zoom' to a display label."""
    token_map = {
        "rtofs": "RTOFS", "espc": "ESPC",
        "cmems": "CMEMS", "copernicus": "CMEMS",
        "depthavg": "Depth-Avg", "surface": "Surface", "zoom": "(Zoom)",
    }
    return " ".join(token_map.get(t.lower(), t.capitalize()) for t in slug.split("-"))


def _sidebar_html(active_glider: Optional[str], processed: List[Dict]) -> str:
    # Fleet index lives at depth-average/index.html; glider pages at depth-average/<glider>/index.html.
    # Links must be relative to whichever depth we're at.
    prefix = "" if active_glider is None else "../"
    glider_links = "\n".join(
        f'<a href="{prefix}{g["glider_name"]}/index.html" '
        f'class="sb-item{"" if g["glider_name"] != active_glider else " active"}">'
        f'<i class="fas fa-satellite-dish fa-fw"></i>{g["glider_name"].upper()}</a>'
        for g in processed
    )
    return f"""
    <div class="sb-label">Active Gliders</div>
    {glider_links}
"""


def _html_page(title: str, navbar_title: str, sidebar: str, content: str) -> str:
    gen_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <link href="{_BS5_CSS}" rel="stylesheet">
  <link href="{_FA_CSS}" rel="stylesheet" crossorigin="anonymous">
  <style>{_SHARED_CSS}</style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark sticky-top p-3">
  <button class="hamburger" id="menuBtn" aria-label="Open menu">&#9776;</button>
  <a class="navbar-brand d-flex align-items-center" href="https://rucool.marine.rutgers.edu">
    <img src="{_RUCOOL_LOGO}" width="32" height="32" class="me-2 rounded" alt="RUCool">
    <span>{navbar_title}</span>
  </a>
  <ul class="navbar-nav ms-auto flex-row">
    <li class="nav-item">
      <a class="nav-link" href="https://rucool.marine.rutgers.edu" target="_blank">
        <i class="fas fa-external-link-alt me-1"></i>RUCool
      </a>
    </li>
  </ul>
</nav>
<div id="sidebar-overlay"></div>
<div id="body-row">
  <div id="sidebar" class="d-flex flex-column">{sidebar}</div>
  <div id="main">{content}</div>
</div>
<div class="footer-bar">
  Rutgers University Center for Ocean Observing Leadership &nbsp;|&nbsp; Generated: {gen_time}
</div>
<script src="{_BS5_JS}"></script>
<script>
(function(){{
  var btn=document.getElementById('menuBtn'),
      sb=document.getElementById('sidebar'),
      ov=document.getElementById('sidebar-overlay');
  function open(){{sb.classList.add('open');ov.classList.add('open');}}
  function close(){{sb.classList.remove('open');ov.classList.remove('open');}}
  btn.addEventListener('click',function(){{sb.classList.contains('open')?close():open();}});
  ov.addEventListener('click',close);
}})();
</script>
</body>
</html>"""


def generate_glider_page(
    glider_name: str,
    deployment_id: str,
    glider_dir: Path,
    glider_record: Optional[Dict],
    processed: Optional[List[Dict]] = None,
) -> None:
    """Write {glider_dir}/index.html for one glider."""
    if processed is None:
        processed = [{"glider_name": glider_name, "deployment_id": deployment_id}]

    currents_dir = glider_dir / CURRENTS_SUBDIR
    archive_dir  = currents_dir / ARCHIVE_SUBDIR
    maps_dir     = glider_dir / "maps"

    sfc_filename = f"{glider_name}_surface_current_comparison.png"
    da_filename  = f"{glider_name}_depth_avg_current_comparison.png"
    sfc_latest   = currents_dir / sfc_filename
    da_latest    = currents_dir / da_filename

    # ── Location / time metadata ──────────────────────────────────────────────
    if glider_record:
        t   = pd.to_datetime(glider_record.get("time", pd.NaT))
        lat = glider_record.get("lat", float("nan"))
        lon = glider_record.get("lon", float("nan"))
        meta_line = (
            f"{lat:.3f}°N, {abs(lon):.3f}°{'W' if lon < 0 else 'E'} &nbsp;|&nbsp; "
            f"{t.strftime('%Y-%m-%d %H:%MZ') if pd.notna(t) else 'N/A'}"
        )
    else:
        meta_line = "Location unavailable"

    # ── Currents section — time-navigable archive viewer ─────────────────────
    # index.html lives at glider_dir; all image hrefs must be relative to that.

    def _build_entries(latest_path: Path, base_stem: str) -> List[Dict]:
        """Return [{src, label}, ...] newest-first; index 0 = latest live plot."""
        entries = []
        if latest_path.exists():
            rel = "/".join(latest_path.relative_to(glider_dir).parts)
            entries.append({"src": rel, "label": "Latest"})
        if archive_dir.exists():
            for p in sorted(
                archive_dir.glob(f"{base_stem}_????????T??????.png"), reverse=True
            ):
                rel = "/".join(p.relative_to(glider_dir).parts)
                suffix = p.stem[len(base_stem) + 1:]
                try:
                    ts    = pd.to_datetime(suffix, format="%Y%m%dT%H%M%S")
                    label = ts.strftime("%Y-%m-%d %H:%MZ")
                except Exception:
                    label = suffix
                entries.append({"src": rel, "label": label})
        return entries

    def _nav_pane(nav_id: str, entries: List[Dict]) -> str:
        if not entries:
            return "<p class='text-muted p-2'>No plots available yet.</p>"
        first = entries[0]
        multi = len(entries) > 1
        older_dis = "" if multi else " disabled"
        nav_bar = (
            f'<div class="d-flex align-items-center justify-content-between mt-2 px-1">'
            f'<button class="btn btn-sm btn-outline-secondary" id="nav-{nav_id}-older"'
            f' onclick="navStep(\'{nav_id}\',1)"{older_dis}>'
            f'<i class="fas fa-chevron-left me-1"></i>Older</button>'
            f'<span id="nav-{nav_id}-label" class="text-muted small"></span>'
            f'<button class="btn btn-sm btn-outline-secondary" id="nav-{nav_id}-newer"'
            f' onclick="navStep(\'{nav_id}\',-1)" disabled>'
            f'Newer<i class="fas fa-chevron-right ms-1"></i></button>'
            f'</div>'
        ) if multi else (
            f'<div class="text-center mt-1">'
            f'<span id="nav-{nav_id}-label" class="text-muted small"></span>'
            f'</div>'
        )
        return (
            f'<div id="nav-{nav_id}">'
            f'<a id="nav-{nav_id}-link" href="{first["src"]}" target="_blank">'
            f'<img id="nav-{nav_id}-img" src="{first["src"]}"'
            f' class="w-100" style="border-radius:4px" alt="{first["label"]}">'
            f'</a>'
            f'{nav_bar}'
            f'</div>'
        )

    da_stem  = da_filename[:-4]
    sfc_stem = sfc_filename[:-4]
    da_entries  = _build_entries(da_latest,  da_stem)
    sfc_entries = _build_entries(sfc_latest, sfc_stem)

    da_pane_html  = _nav_pane("da",  da_entries)
    sfc_pane_html = _nav_pane("sfc", sfc_entries)
    da_js  = json.dumps(da_entries)
    sfc_js = json.dumps(sfc_entries)

    nav_script = (
        "<script>\n(function(){\n"
        f"  var _nd = {{ da: {{ imgs: {da_js}, idx: 0 }}, sfc: {{ imgs: {sfc_js}, idx: 0 }} }};\n"
        "  function render(k) {\n"
        "    var d = _nd[k], e = d.imgs[d.idx];\n"
        "    var img = document.getElementById('nav-'+k+'-img'); if (!img) return;\n"
        "    img.src = e.src;\n"
        "    document.getElementById('nav-'+k+'-link').href = e.src;\n"
        "    document.getElementById('nav-'+k+'-label').textContent =\n"
        "      e.label + (d.imgs.length > 1 ? ' (' + (d.idx+1) + '/' + d.imgs.length + ')' : '');\n"
        "    var ol = document.getElementById('nav-'+k+'-older');\n"
        "    var nw = document.getElementById('nav-'+k+'-newer');\n"
        "    if (ol) ol.disabled = d.idx >= d.imgs.length - 1;\n"
        "    if (nw) nw.disabled = d.idx <= 0;\n"
        "  }\n"
        "  window.navStep = function(k, dir) {\n"
        "    var d = _nd[k];\n"
        "    d.idx = Math.max(0, Math.min(d.imgs.length-1, d.idx+dir));\n"
        "    render(k);\n"
        "  };\n"
        "  ['da','sfc'].forEach(render);\n"
        "})();\n</script>"
    )

    currents_content = f"""
<ul class="nav nav-tabs mb-3" id="curTabs" role="tablist">
  <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab"
      data-bs-target="#da-pane" type="button">
    <i class="fas fa-layer-group me-1"></i>Depth-Averaged</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
      data-bs-target="#sfc-pane" type="button">
    <i class="fas fa-water me-1"></i>Surface Currents</button></li>
</ul>
<div class="tab-content">
  <div class="tab-pane fade show active" id="da-pane">
    <div class="card"><div class="card-body p-2">{da_pane_html}</div></div>
  </div>
  <div class="tab-pane fade" id="sfc-pane">
    <p class="text-muted small px-1 pt-2 mb-1">
      <i class="fas fa-info-circle me-1"></i>
      Surface currents are derived from GPS drift between consecutive fixes taken within a
      single surfacing event (ERDDAP trajectory data), giving a direct Lagrangian estimate of
      the near-surface current vector with no contribution from dive-phase currents.
    </p>
    <div class="card"><div class="card-body p-2">{sfc_pane_html}</div></div>
  </div>
</div>
{nav_script}
"""

    # ── Maps section ──────────────────────────────────────────────────────────
    # Glob for any region alias: {glider}_*_latest_*.png
    map_items: List[Tuple[str, str, str]] = []   # (label, relative path, slug)
    if maps_dir.exists():
        for p in sorted(maps_dir.glob(f"{glider_name}_*_latest_*.png")):
            # filename pattern: {glider}_{region}_latest_{slug}.png
            parts = p.stem.split("_latest_", 1)
            slug  = parts[1] if len(parts) == 2 else p.stem
            label = _slug_to_label(slug)
            map_items.append((label, f"maps/{p.name}", slug))

    if map_items:
        def _depthavg_first(item):
            # (label, path, slug) — depth-avg slugs sort before surface slugs
            return (0 if "depthavg" in item[2] else 1, item[2])

        zoom_items = [
            (label.replace(" (Zoom)", ""), path)
            for label, path, slug in sorted(
                (i for i in map_items if i[2].endswith("-zoom")), key=_depthavg_first
            )
        ]
        wide_items = [
            (label, path)
            for label, path, slug in sorted(
                (i for i in map_items if not i[2].endswith("-zoom")), key=_depthavg_first
            )
        ]

        def _tab_group(group_id: str, items: List[Tuple[str, str]]) -> str:
            if not items:
                return "<p class='text-muted small'>No images available.</p>"
            btns = "\n".join(
                f'<li class="nav-item">'
                f'<button class="nav-link{" active" if i == 0 else ""}" '
                f'data-bs-toggle="tab" data-bs-target="#{group_id}-pane-{i}" type="button">'
                f'{lbl}</button></li>'
                for i, (lbl, _) in enumerate(items)
            )
            panes = "\n".join(
                f'<div class="tab-pane fade{"  show active" if i == 0 else ""}" id="{group_id}-pane-{i}">'
                f'<div class="card"><div class="card-body text-center p-2">'
                f'<a href="{pth}" target="_blank">'
                f'<img src="{pth}" class="w-100"'
                f' style="border-radius:4px;max-height:70vh;object-fit:contain">'
                f'</a></div></div></div>'
                for i, (_, pth) in enumerate(items)
            )
            return (
                f'<ul class="nav nav-tabs mb-2" id="{group_id}">{btns}</ul>'
                f'<div class="tab-content">{panes}</div>'
            )

        maps_content = f"""
<div class="mb-4">
  <h6 class="fw-bold mb-2" style="color:var(--dark)">
    <i class="fas fa-search-plus me-1"></i>Zoomed Region
  </h6>
  {_tab_group("zoom", zoom_items)}
</div>
<div>
  <h6 class="fw-bold mb-2" style="color:var(--dark)">
    <i class="fas fa-globe me-1"></i>Wider Region
  </h6>
  {_tab_group("wide", wide_items)}
</div>
"""
    else:
        maps_content = "<p class='text-muted'>No map images found. Run ru29_map_tropical_western_atlantic.py first.</p>"

    # ── Assemble ──────────────────────────────────────────────────────────────
    content = f"""
<p class="page-title">
  <i class="fas fa-satellite-dish me-2"></i>{glider_name.upper()}
  &nbsp;<span style="font-size:.85rem;font-weight:400;color:#64748b">
    {deployment_id} &nbsp;|&nbsp; {meta_line}</span>
</p>

<ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist">
  <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab"
      data-bs-target="#maps-section" type="button">
    <i class="fas fa-map me-1"></i>Maps</button></li>
  <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
      data-bs-target="#currents-section" type="button">
    <i class="fas fa-compass me-1"></i>Current Comparisons</button></li>
</ul>
<div class="tab-content">
  <div class="tab-pane fade show active" id="maps-section">{maps_content}</div>
  <div class="tab-pane fade" id="currents-section">{currents_content}</div>
</div>
"""
    sidebar = _sidebar_html(glider_name, processed)
    html    = _html_page(
        title        = f"{glider_name.upper()} – Depth-Averaged Current Comparisons",
        navbar_title = f"Active Gliders – Depth-Averaged Current Comparisons &rsaquo; {glider_name.upper()}",
        sidebar      = sidebar,
        content      = content,
    )

    out = glider_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    LOGGER.info("Wrote glider page: %s", out)


def generate_fleet_index(processed: List[Dict], base_dir: Path) -> None:
    """Write {base_dir}/index.html — redirects immediately to the first glider alphabetically."""
    first = sorted(processed, key=lambda g: g["glider_name"])[0]["glider_name"]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url={first}/index.html">
  <title>Redirecting…</title>
</head>
<body>
  <script>window.location.replace("{first}/index.html");</script>
</body>
</html>
"""

    out = base_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    LOGGER.info("Wrote fleet index: %s", out)


def _process_glider(glider_name: str, deployment_id: str) -> Optional[Dict]:
    """Run surface + depth-avg comparison for one glider deployment.

    Returns glider metadata dict on success, None if skipped.
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Glider: %s  |  Deployment: %s", glider_name, deployment_id)
    LOGGER.info("=" * 60)

    glider_dir   = Path(save_path) / glider_name
    currents_dir = glider_dir / CURRENTS_SUBDIR
    archive_dir  = currents_dir / ARCHIVE_SUBDIR
    sfc_filename = f"{glider_name}_surface_current_comparison.png"
    da_filename  = f"{glider_name}_depth_avg_current_comparison.png"

    # Pre-warm all dataset caches sequentially before threads are spawned.
    # HDF5/OPeNDAP concurrent opens are not thread-safe and produce temp-file
    # conflicts when multiple lru_cache functions hit the same server URL at once.
    LOGGER.info("Pre-warming model dataset caches...")
    for _fn in (get_rtofs_dataset, get_rtofs_full_dataset, get_espc_dataset, get_cmems_client):
        try:
            _fn()
        except Exception as _exc:
            LOGGER.warning("Cache warm-up failed for %s: %s", _fn.__name__, _exc)

    if SURFACING_MODE == "range":
        if not SURFACING_RANGE_START or not SURFACING_RANGE_END:
            raise ValueError(
                "SURFACING_RANGE_START and SURFACING_RANGE_END must be set "
                "when SURFACING_MODE is 'range'."
            )
        glider_records = get_surface_records_for_range(
            deployment_id, SURFACING_RANGE_START, SURFACING_RANGE_END,
        )
        if not glider_records:
            LOGGER.warning("No surfacings in range for %s; skipping.", glider_name)
            return None

        last_record = None
        for idx, glider_record in enumerate(glider_records, start=1):
            LOGGER.info(
                "Processing surfacing %d/%d (time=%s)",
                idx, len(glider_records), glider_record["time"],
            )
            process_surface_record(
                glider_record, deployment_id, glider_name,
                surface_output_files=[
                    _build_output_path(glider_record["time"], base_dir=archive_dir, filename=sfc_filename),
                ],
                depth_avg_output_files=[
                    _build_output_path(glider_record["time"], base_dir=archive_dir, filename=da_filename),
                ],
            )
            last_record = glider_record
        glider_record = last_record

    else:
        glider_record = get_latest_surface_record(deployment_id)
        age_hours = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(glider_record["time"])).total_seconds() / 3600
        if age_hours > GLIDER_STALE_SURFACING_HOURS:
            LOGGER.warning(
                "Skipping %s: last surfacing is %.1f h old (threshold %d h).",
                glider_name, age_hours, GLIDER_STALE_SURFACING_HOURS,
            )
            return None
        process_surface_record(
            glider_record, deployment_id, glider_name,
            surface_output_files=[
                currents_dir / sfc_filename,
                _build_output_path(glider_record["time"], base_dir=archive_dir, filename=sfc_filename),
            ],
            depth_avg_output_files=[
                currents_dir / da_filename,
                _build_output_path(glider_record["time"], base_dir=archive_dir, filename=da_filename),
            ],
        )

    # generate_glider_page is called from main() after all gliders are processed
    # so the sidebar can include the full fleet list.
    return {"glider_name": glider_name, "deployment_id": deployment_id, "record": glider_record}


def main():
    print_model_status()

    if RUN_ALL_ACTIVE_GLIDERS:
        gliders = discover_active_gliders()
        if not gliders:
            LOGGER.warning("No active gliders found on ERDDAP; nothing to process.")
            return
        LOGGER.info(
            "Processing %d active glider(s): %s",
            len(gliders), [g["glider_name"] for g in gliders],
        )
    else:
        deployment_id = resolve_latest_deployment(GLIDER_NAME, timeout=GLIDER_API_TIMEOUT)
        gliders = [{"glider_name": GLIDER_NAME, "deployment_id": deployment_id}]
        LOGGER.info("Single-glider mode: %s (%s)", GLIDER_NAME, deployment_id)

    processed: List[Dict] = []
    for glider_info in gliders:
        try:
            result = _process_glider(glider_info["glider_name"], glider_info["deployment_id"])
            if result is not None:
                processed.append(result)
        except Exception:
            LOGGER.exception(
                "Failed to process glider %s; continuing to next.",
                glider_info["glider_name"],
            )

    if processed:
        # Regenerate all glider pages now that we have the complete fleet list
        # (so every sidebar shows all active gliders).
        for g in processed:
            generate_glider_page(
                g["glider_name"], g["deployment_id"],
                Path(save_path) / g["glider_name"],
                g.get("record"),
                processed,
            )
        generate_fleet_index(processed, Path(save_path))


if __name__ == "__main__":
    main()
