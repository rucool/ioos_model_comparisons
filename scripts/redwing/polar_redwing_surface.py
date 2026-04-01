#!/usr/bin/env python
"""
Polar visualization of surface currents for the Redwing Slocum glider and
collocated model estimates from RTOFS, ESPC, CMEMS, and Lusitania.

The glider's surface drift velocity is derived from consecutive GPS surfacing
positions: u/v are computed as the displacement between the current and
previous surfacing divided by the elapsed time. This is used instead of
m_water_vx/vy (depth-averaged current sensors) because at the surface the
glider is drifting with the surface current rather than dead reckoning.

Model currents are sampled at the surface layer (depth ≈ 0 m).
"""

import functools
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
import datetime as dt

from ioos_model_comparisons.calc import lon180to360
from ioos_model_comparisons.models import CMEMS, espc_uv, rtofs as load_rtofs

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

# ==============================================================================
# Configuration
# ==============================================================================

# Glider deployment identifier
DEPLOYMENT_ID = "redwing-20251011T1511"

# Surfacing selection mode: "latest" or "range"
SURFACING_MODE = "latest"
# When using "range", set ISO-8601 timestamps for start/end (UTC recommended).
SURFACING_RANGE_START = "2025-11-15T00:00:00Z"
SURFACING_RANGE_END   = "2026-01-01T00:00:00Z"

# Model enable/disable flags
MODEL_CONFIG = {
    "RTOFS": {
        "enabled": False,
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
    "Lusitania": {
        "enabled": True,
        "color": "darkorange",
        "timeout": 60,
    },
}

# save_path = '/www/web/rucool/media/sentinel/'
save_path = '/Users/mikesmith/Documents/'
OUTPUT_FILENAME = "surface_current_comparison.png"
COMPASS_SUBDIR = "compass_surface"

# API configuration
GLIDER_API_BASE    = "https://marine.rutgers.edu/cool/data/gliders/api"
GLIDER_API_TIMEOUT = 30

# Minimum time separation required between the two positions used to compute
# surface drift.  Pairs closer than this are skipped (e.g. back-to-back
# Iridium dials within a few minutes).  Expected reporting interval is ~12 h.
MIN_DRIFT_INTERVAL_HOURS = 6

# Glider display configuration
GLIDER_COLOR = "blue"

# Depth dimension candidates for different models
DEPTH_DIM_CANDIDATES = ("depth", "Depth", "depthu", "depthv", "z")

# Retry configuration for dataset loading
DATASET_RETRY_ATTEMPTS      = 5
DATASET_RETRY_DELAY_SECONDS = 20

# Lusitania model configuration (THREDDS)
LUSITANIA_CONFIG = {
    "base_url":     "https://thredds.atlanticsense.com/thredds",
    "catalog_path": "/catalog/atlDatasets/Lusitania/catalog.html",
    "opendap_base": "/dodsC/atlDatasets/Lusitania/",
    "file_pattern": r"(\d{10})\.nc",  # YYYYMMDDHH.nc
    "timeout":      30,
}

# Logging configuration
LOGGER_NAME    = "polar_redwing_surface"
LOG_LEVEL_NAME = os.getenv("POLAR_REDWING_LOG_LEVEL", "INFO").upper()
LOG_FILE       = os.getenv("POLAR_REDWING_LOG_FILE")
LOG_FORMAT     = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
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

    formatter     = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
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
    """
    Calculate current speed and direction from u,v components.

    Returns:
        speed:     Current speed (m/s)
        direction: Direction in radians (0 = east, π/2 = north)
    """
    if not np.isfinite(u) or not np.isfinite(v):
        return np.nan, np.nan
    speed     = float(np.hypot(u, v))
    direction = float(np.arctan2(v, u))
    return speed, direction


def calc_surface_drift_uv(
    lat1: float,
    lon1: float,
    t1: pd.Timestamp,
    lat2: float,
    lon2: float,
    t2: pd.Timestamp,
) -> Tuple[float, float]:
    """
    Estimate surface drift u/v from two consecutive GPS surfacing positions.

    Uses a flat-Earth approximation valid for small displacements:
        u (east,  m/s) = Δlon_metres / Δt_seconds
        v (north, m/s) = Δlat_metres / Δt_seconds

    Parameters:
        lat1, lon1, t1: Earlier surfacing position / time
        lat2, lon2, t2: Later   surfacing position / time

    Returns:
        u, v: Eastward and northward velocity components (m/s)
    """
    dt_seconds = (t2 - t1).total_seconds()
    if dt_seconds <= 0:
        LOGGER.warning(
            "Non-positive time difference between surfacings (%.1f s); "
            "cannot compute surface drift.",
            dt_seconds,
        )
        return np.nan, np.nan

    mean_lat_rad      = np.radians((lat1 + lat2) / 2.0)
    meters_per_deg    = 111_320.0  # approximate metres per degree of latitude

    delta_lat_m = (lat2 - lat1) * meters_per_deg
    delta_lon_m = (lon2 - lon1) * meters_per_deg * np.cos(mean_lat_rad)

    u = delta_lon_m / dt_seconds
    v = delta_lat_m / dt_seconds

    LOGGER.debug(
        "Surface drift: Δlat=%.1f m  Δlon=%.1f m  Δt=%.0f s  →  u=%.4f  v=%.4f m/s",
        delta_lat_m,
        delta_lon_m,
        dt_seconds,
        u,
        v,
    )
    return u, v


def safe_model_call(func, model_name: str, *args, **kwargs) -> Optional[Dict]:
    """Safely call a model function with error handling."""
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
                model_name,
                elapsed,
                result["speed"] * 100,
            )
        else:
            LOGGER.warning(
                "%s completed in %.2fs but returned no valid data.",
                model_name,
                elapsed,
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
                    model_name,
                    DATASET_RETRY_ATTEMPTS,
                )
                raise
            LOGGER.warning(
                "%s load attempt %d/%d failed: %s; retrying in %ds",
                model_name,
                attempt,
                DATASET_RETRY_ATTEMPTS,
                exc,
                DATASET_RETRY_DELAY_SECONDS,
            )
            time.sleep(DATASET_RETRY_DELAY_SECONDS)


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


def _fetch_positions(deployment: str) -> pd.DataFrame:
    """
    Download surfacing GPS positions from the Rutgers COOL glider surfacings API.

    Uses /api/surfacings/ which works even when the glider is not profiling
    (e.g. drifting at the surface).  The API returns records newest-first;
    this function reverses them to chronological order.

    Returns:
        DataFrame with columns: ts, lat, lon  (sorted ascending by ts)
    """
    url = f"{GLIDER_API_BASE}/surfacings/?deployment={deployment}"
    LOGGER.debug("Requesting surfacing positions for %s", deployment)

    try:
        response = requests.get(url, timeout=GLIDER_API_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.exception("Failed to fetch surfacings for deployment %s", deployment)
        raise

    payload = response.json()
    records = payload.get("data", [])

    if not records:
        raise ValueError(f"No surfacing data found for deployment '{deployment}'")

    # API returns newest-first; reverse for chronological order
    records = records[::-1]

    rows = []
    skipped = 0
    for s in records:
        epoch = s.get("gps_timestamp_epoch")
        if epoch is None:
            skipped += 1
            continue
        ts = pd.Timestamp(epoch, unit="s")  # tz-naive UTC, consistent with model dataset time axes

        if s.get("gps_lat_degrees") is not None:
            lat = float(s["gps_lat_degrees"])
            lon = float(s["gps_lon_degrees"])
        elif s.get("gps_lat") is not None:
            lat = _dm_to_dd(s["gps_lat"])
            lon = _dm_to_dd(s["gps_lon"])
        else:
            # No GPS fix recorded for this surfacing — skip
            skipped += 1
            continue

        rows.append({"ts": ts, "lat": lat, "lon": lon})

    if skipped:
        LOGGER.info("Skipped %d surfacing records with no GPS fix or timestamp", skipped)

    if not rows:
        raise ValueError(f"No valid position records parsed for deployment '{deployment}'")

    data = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    LOGGER.info("Retrieved %d surfacing positions for deployment %s", len(data), deployment)
    return data


def _normalize_range_bound(value: Optional[str], label: str) -> Optional[pd.Timestamp]:
    """Normalize a range bound to a UTC timestamp for comparison."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} value: {value!r}") from exc


def _build_surface_record(latest: pd.Series, previous: pd.Series) -> Dict:
    """
    Build a surface-drift record from two adjacent surfacing rows.

    The u/v are derived from the displacement between the previous and current
    GPS fix divided by the elapsed time between fixes.
    """
    t2   = pd.to_datetime(latest["ts"])
    t1   = pd.to_datetime(previous["ts"])
    lat2, lon2 = float(latest["lat"]),   float(latest["lon"])
    lat1, lon1 = float(previous["lat"]), float(previous["lon"])

    LOGGER.info(
        "Surface drift: prev=(%.4f, %.4f) @ %s  →  curr=(%.4f, %.4f) @ %s  (Δt=%.1f h)",
        lat1, lon1, t1,
        lat2, lon2, t2,
        (t2 - t1).total_seconds() / 3600,
    )
    u, v = calc_surface_drift_uv(lat1, lon1, t1, lat2, lon2, t2)
    speed, direction = calc_speed_dir(u, v)
    LOGGER.info("Surface drift result: u=%.4f m/s  v=%.4f m/s  speed=%.2f cm/s", u, v, speed * 100 if np.isfinite(speed) else float("nan"))

    record = {
        "source":        "Glider",
        "u":             u,
        "v":             v,
        "speed":         speed,
        "direction":     direction,
        "lat":           lat2,
        "lon":           lon2,
        "time":          t2,
        "previous_lat":  lat1,
        "previous_lon":  lon1,
        "previous_time": t1,
        "data_available": True,
    }

    LOGGER.debug(
        "Surface record: t=%s  pos=(%.3f, %.3f)  speed=%.2f cm/s",
        t2,
        lat2,
        lon2,
        speed * 100 if np.isfinite(speed) else float("nan"),
    )
    return record


def _find_previous_position(positions: pd.DataFrame, latest_idx: int) -> Optional[pd.Series]:
    """
    Search backwards from latest_idx for a position that is at least
    MIN_DRIFT_INTERVAL_HOURS older than the position at latest_idx.
    Returns None if no suitable previous position is found.
    """
    latest_time = pd.to_datetime(positions.iloc[latest_idx]["ts"])
    for i in range(latest_idx - 1, -1, -1):
        dt_hours = (latest_time - pd.to_datetime(positions.iloc[i]["ts"])).total_seconds() / 3600
        if dt_hours >= MIN_DRIFT_INTERVAL_HOURS:
            LOGGER.debug(
                "Using position at index %d (Δt=%.1f h) as previous fix.",
                i, dt_hours,
            )
            return positions.iloc[i]
    return None


def get_latest_surface_record(deployment: str) -> Dict:
    """Return surface drift vector using the most recent GPS fix and the
    nearest previous fix that is at least MIN_DRIFT_INTERVAL_HOURS older."""
    positions = _fetch_positions(deployment)
    if len(positions) < 2:
        raise ValueError(f"Need at least 2 position records; got {len(positions)}")

    previous = _find_previous_position(positions, len(positions) - 1)
    if previous is None:
        raise ValueError(
            f"No previous position found at least {MIN_DRIFT_INTERVAL_HOURS}h "
            f"before the latest fix. Positions may be too closely spaced."
        )
    return _build_surface_record(positions.iloc[-1], previous)


def get_surface_records_for_range(
    deployment: str,
    start_time: str,
    end_time: str,
) -> List[Dict]:
    """Return surface drift vectors for all surfacings in a time range."""
    positions = _fetch_positions(deployment)
    if len(positions) < 2:
        LOGGER.warning("Not enough position records to build surface drift.")
        return []

    times_utc = pd.to_datetime(positions["ts"], utc=True, errors="coerce")
    start_ts  = _normalize_range_bound(start_time, "start_time")
    end_ts    = _normalize_range_bound(end_time,   "end_time")

    if start_ts and end_ts and start_ts > end_ts:
        raise ValueError("start_time must be earlier than end_time.")

    mask = pd.Series(True, index=positions.index)
    if start_ts is not None:
        mask &= times_utc >= start_ts
    if end_ts is not None:
        mask &= times_utc <= end_ts

    indices = positions.index[mask].tolist()
    if not indices:
        LOGGER.warning("No surfacings found between %s and %s.", start_time, end_time)
        return []

    records: List[Dict] = []
    for idx in indices:
        previous = _find_previous_position(positions, idx)
        if previous is None:
            LOGGER.debug(
                "Skipping surfacing at %s — no previous position at least %dh earlier.",
                positions.iloc[idx]["ts"],
                MIN_DRIFT_INTERVAL_HOURS,
            )
            continue
        records.append(_build_surface_record(positions.iloc[idx], previous))

    LOGGER.info(
        "Built %d surface drift records between %s and %s",
        len(records),
        start_time,
        end_time,
    )
    return records


# ==============================================================================
# Model Data Functions
# ==============================================================================

def _parse_lusitania_catalog(timeout: int = 30) -> List[Tuple[dt.datetime, str]]:
    """Parse the Lusitania THREDDS catalog to get available files."""
    catalog_url = (
        f"{LUSITANIA_CONFIG['base_url']}{LUSITANIA_CONFIG['catalog_path']}"
    )
    try:
        response = requests.get(catalog_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Failed to fetch Lusitania catalog: %s", exc)
        return []

    pattern         = re.compile(LUSITANIA_CONFIG["file_pattern"])
    available_files: List[Tuple[dt.datetime, str]] = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            text  = link.get_text(strip=True)
            match = pattern.search(text)
            if not match:
                continue
            date_str = match.group(1)
            try:
                file_date = dt.datetime.strptime(date_str, "%Y%m%d%H")
                available_files.append((file_date, text))
            except ValueError as exc:
                LOGGER.warning("Could not parse date from %s: %s", text, exc)
    else:
        for match in pattern.finditer(response.text):
            date_str = match.group(1)
            filename = f"{date_str}.nc"
            try:
                file_date = dt.datetime.strptime(date_str, "%Y%m%d%H")
                if (file_date, filename) not in available_files:
                    available_files.append((file_date, filename))
            except ValueError as exc:
                LOGGER.warning("Could not parse date from %s: %s", filename, exc)

    available_files.sort(key=lambda x: x[0])
    if available_files:
        LOGGER.info(
            "Lusitania catalog range: %s to %s",
            available_files[0][0],
            available_files[-1][0],
        )
    else:
        LOGGER.warning("No Lusitania files found in catalog response.")
    return available_files


@functools.lru_cache(maxsize=1)
def _get_lusitania_catalog() -> List[Tuple[dt.datetime, str]]:
    return _parse_lusitania_catalog(timeout=LUSITANIA_CONFIG["timeout"])


def _find_lusitania_file(
    target_time: dt.datetime,
    available_files: Optional[List[Tuple[dt.datetime, str]]] = None,
    method: str = "nearest",
) -> Optional[Tuple[dt.datetime, str]]:
    """Find the Lusitania file nearest to the target time."""
    if available_files is None:
        available_files = _get_lusitania_catalog()
    if not available_files:
        LOGGER.warning("No Lusitania files available in catalog.")
        return None

    if target_time.tzinfo is not None:
        target_time = target_time.replace(tzinfo=None)

    if method == "before":
        candidates = [f for f in available_files if f[0] <= target_time]
        return candidates[-1] if candidates else None
    if method == "after":
        candidates = [f for f in available_files if f[0] >= target_time]
        return candidates[0] if candidates else None

    best_match, min_diff = None, None
    for file_date, filename in available_files:
        diff = abs((file_date - target_time).total_seconds())
        if min_diff is None or diff < min_diff:
            min_diff  = diff
            best_match = (file_date, filename)
    return best_match


def _standardize_lusitania_coords(ds: xr.Dataset) -> xr.Dataset:
    """Normalize Lusitania coordinate names to lon/lat when possible."""
    rename_map = {}
    lon_key = next(
        (k for k in ("lon", "longitude", "LONGITUDE", "Longitude") if k in ds.coords), None
    )
    lat_key = next(
        (k for k in ("lat", "latitude", "LATITUDE", "Latitude") if k in ds.coords), None
    )
    if lon_key and lon_key != "lon":
        rename_map[lon_key] = "lon"
    if lat_key and lat_key != "lat":
        rename_map[lat_key] = "lat"
    if rename_map:
        ds = ds.rename(rename_map)
    ds = ds.isel(depth=slice(None, None, -1))
    return ds


def lusitania_uv(
    target_time: Optional[dt.datetime] = None,
    rename: bool = True,
    method: str = "nearest",
) -> Optional[xr.Dataset]:
    """Load Lusitania model data from the AtlanticSense THREDDS server."""
    if target_time is not None:
        target_time = pd.to_datetime(target_time).to_pydatetime()
        if target_time.tzinfo is not None:
            target_time = target_time.replace(tzinfo=None)

    if target_time is None:
        available = _get_lusitania_catalog()
        if not available:
            LOGGER.error("No Lusitania files available.")
            return None
        file_date, filename = available[-1]
    else:
        result = _find_lusitania_file(target_time, method=method)
        if result is None:
            return None
        file_date, filename = result

    opendap_url = (
        f"{LUSITANIA_CONFIG['base_url']}"
        f"{LUSITANIA_CONFIG['opendap_base']}{filename}"
    )
    LOGGER.info("Loading Lusitania data from: %s", opendap_url)

    try:
        ds = xr.open_dataset(opendap_url)
        ds.attrs["model"]      = "Lusitania"
        ds.attrs["source_file"] = filename
        ds.attrs["source_url"]  = opendap_url

        if rename:
            rename_map  = {}
            u_candidates = ["uo", "water_u", "u_velocity", "U", "ucur"]
            v_candidates = ["vo", "water_v", "v_velocity", "V", "vcur"]
            for u_name in u_candidates:
                if u_name in ds.data_vars:
                    rename_map[u_name] = "u"
                    break
            for v_name in v_candidates:
                if v_name in ds.data_vars:
                    rename_map[v_name] = "v"
                    break
            if rename_map:
                LOGGER.info("Renaming Lusitania variables: %s", rename_map)
                ds = ds.rename(rename_map)

        ds = _standardize_lusitania_coords(ds)
        return ds
    except Exception as exc:
        LOGGER.error("Failed to load Lusitania data: %s", exc)
        return None


@functools.lru_cache(maxsize=4)
def get_lusitania_dataset(target_time: Optional[pd.Timestamp]):
    """Load and cache the Lusitania dataset for a target time."""
    if not MODEL_CONFIG["Lusitania"]["enabled"]:
        return None

    def _load():
        ds = lusitania_uv(target_time=target_time, rename=True, method="before")
        if ds is None:
            raise ValueError("Lusitania dataset unavailable.")
        return ds

    return load_dataset_with_retry(_load, "Lusitania")


@functools.lru_cache(maxsize=1)
def get_rtofs_dataset():
    """Load and cache the RTOFS dataset (surface layer only)."""
    if not MODEL_CONFIG["RTOFS"]["enabled"]:
        return None

    def _load():
        return load_rtofs().isel(depth=0)

    return load_dataset_with_retry(_load, "RTOFS")


@functools.lru_cache(maxsize=1)
@functools.lru_cache(maxsize=1)
def get_espc_dataset():
    """Load and cache the ESPC dataset."""
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


def _build_vector_record(
    source: str,
    u: float,
    v: float,
    lon: float,
    lat: float,
    when: pd.Timestamp,
) -> Dict:
    """Build a standardized vector record for comparison."""
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


# ==============================================================================
# Model Sampling - Surface Point Interpolation
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
    return _build_vector_record("RTOFS", u, v, float(point.lon.values), float(point.lat.values), point.time.values)


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

    # Collapse all remaining dimensions on each variable independently.
    # The ESPC FMRC dataset uses staggered grids (depthu/depthv, lat_v, etc.)
    # so u and v may have different leftover dims after space/time interpolation.
    u_da = profile["u"]
    v_da = profile["v"]
    for dim in list(u_da.dims):
        u_da = u_da.isel({dim: 0})
    for dim in list(v_da.dims):
        v_da = v_da.isel({dim: 0})

    u = float(u_da.values)
    v = float(v_da.values)
    return _build_vector_record("ESPC", u, v, float(profile.lon.values), float(profile.lat.values), profile.time.values)


def sample_cmems_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate CMEMS surface layer to a single point in space/time."""
    client = get_cmems_client()
    if client is None:
        LOGGER.warning("CMEMS client unavailable; cannot sample surface point.")
        return None

    # Request surface only by fetching a shallow depth range then taking level 0
    profile = client.get_point(
        lon, lat, when,
        interp=True,
        vars=["currents"],
    ).squeeze()

    # Take the shallowest depth level — handle staggered grids (depthu / depthv)
    for depth_candidate in DEPTH_DIM_CANDIDATES:
        if depth_candidate in profile.dims:
            profile = profile.isel({depth_candidate: 0})

    u = float(profile["u"].values)
    v = float(profile["v"].values)
    return _build_vector_record("CMEMS", u, v, float(profile.lon.values), float(profile.lat.values), profile.time.values)


def sample_lusitania_surface_point(lon: float, lat: float, when: pd.Timestamp) -> Dict:
    """Interpolate Lusitania surface layer to a single point in space/time."""
    ds = get_lusitania_dataset(pd.to_datetime(when))
    if ds is None:
        LOGGER.warning("Lusitania dataset unavailable; cannot sample surface point.")
        return None
    if "lon" not in ds.coords or "lat" not in ds.coords:
        LOGGER.warning("Lusitania dataset missing lon/lat coordinates.")
        return None

    lon_target = lon
    try:
        if float(ds["lon"].max()) > 180:
            lon_target = lon180to360(lon)
    except Exception:
        pass

    point = ds.interp(
        time=np.datetime64(when),
        lon=lon_target,
        lat=lat,
    ).squeeze()

    # Take the shallowest depth level
    depth_dim = next(
        (d for d in DEPTH_DIM_CANDIDATES if d in point.dims or d in point.coords),
        None,
    )
    if depth_dim is not None:
        point = point.isel({depth_dim: 0})

    u = float(point["u"].values)
    v = float(point["v"].values)
    return _build_vector_record(
        "Lusitania", u, v,
        float(point.lon.values), float(point.lat.values),
        point.time.values,
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
    base_dir = Path(base_dir) if base_dir is not None else Path(save_path)
    filename = filename if filename is not None else OUTPUT_FILENAME
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
    enabled_models: List[str],
    output_files: Optional[List[Path]] = None,
):
    """Create the polar vector plot and companion summary table."""
    LOGGER.info("Rendering surface vector plot for %d records", len(vector_records))

    color_map = {"Glider": GLIDER_COLOR}
    for model_name in enabled_models:
        if model_name in MODEL_CONFIG:
            color_map[model_name] = MODEL_CONFIG[model_name]["color"]

    fig = plt.figure(figsize=(18, 8))
    ax  = fig.add_subplot(121, projection="polar")

    speed_values   = np.array([r["speed"] for r in vector_records], dtype=float)
    finite_speeds  = speed_values[np.isfinite(speed_values)]
    max_speed_ms   = float(finite_speeds.max()) if finite_speeds.size else 1.0
    max_speed_cm   = max_speed_ms * 100

    def _plot_arrow(record, is_glider: bool):
        speed_cm = record["speed"] * 100
        ax.arrow(
            record["direction"],
            0,
            0,
            speed_cm,
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

    non_glider = [r for r in vector_records if r["source"] != "Glider"]
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
            [angle_rad, angle_rad],
            [0, reticle_radius],
            color="#1f2933",
            linewidth=1.4,
            alpha=0.8,
            zorder=1,
            clip_on=False,
        )

    ax.set_rlabel_position(180)
    ax.grid(True, color="#9ca3af", alpha=0.5, linestyle="--", linewidth=1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.1), framealpha=0.95, fontsize=15)

    # Summary table
    table_ax = fig.add_subplot(122)
    table_ax.axis("off")

    disabled_models = [m for m in MODEL_CONFIG if not MODEL_CONFIG[m]["enabled"]]

    dt_hours = (
        (glider_info["time"] - glider_info["previous_time"]).total_seconds() / 3600
        if pd.notna(glider_info["time"]) and pd.notna(glider_info["previous_time"])
        else float("nan")
    )

    title_text  = "Surface Currents (GPS Drift)\n"
    title_text += f"Glider Deployment ID: {DEPLOYMENT_ID}\n"
    title_text += f"WMO ID: 8901157\n"
    title_text += f"Location: {glider_info['lat']:.3f}°N, {abs(glider_info['lon']):.3f}°W\n"
    title_text += f"Time (UTC): {glider_info['time']}\n"
    title_text += f"Drift interval: {dt_hours:.1f} h"

    if disabled_models:
        title_text += f" | Disabled: {', '.join(disabled_models)}"

    table_ax.text(
        0.5, 0.94,
        title_text,
        transform=table_ax.transAxes,
        ha="center",
        fontsize=20,
        fontweight="bold",
        linespacing=1.5,
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
        0.95, 0.01,
        f"Image generated: {generated_time}",
        transform=fig.transFigure,
        ha="right", va="bottom",
        fontsize=7, fontweight="bold",
    )

    plt.tight_layout()
    if output_files is None:
        output_files = [Path(save_path) / OUTPUT_FILENAME]
    for output_file in output_files:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving surface current comparison figure to %s", output_file)
        plt.savefig(output_file, dpi=150)


# ==============================================================================
# Main Execution
# ==============================================================================

def print_model_status():
    LOGGER.info("Model Configuration:")
    for model_name, config in MODEL_CONFIG.items():
        LOGGER.info("%s: %s", model_name, "enabled" if config["enabled"] else "disabled")


def process_surface_record(
    glider_record: Dict,
    output_files: Optional[List[Path]] = None,
) -> None:
    """Run model comparisons and plotting for a single surface drift record."""
    LOGGER.info(
        "Glider surface position: %.3f°N, %.3f°%s",
        glider_record["lat"],
        abs(glider_record["lon"]),
        "W" if glider_record["lon"] < 0 else "E",
    )
    LOGGER.info("Surface time: %s", glider_record["time"])
    LOGGER.info(
        "Previous position: %.3f°N, %.3f°%s at %s",
        glider_record["previous_lat"],
        abs(glider_record["previous_lon"]),
        "W" if glider_record["previous_lon"] < 0 else "E",
        glider_record["previous_time"],
    )

    enabled_models = [m for m in MODEL_CONFIG if MODEL_CONFIG[m]["enabled"]]

    rtofs_record = safe_model_call(
        sample_rtofs_surface_point, "RTOFS",
        glider_record["lon"], glider_record["lat"], glider_record["time"],
    )
    espc_record = safe_model_call(
        sample_espc_surface_point, "ESPC",
        glider_record["lon"], glider_record["lat"], glider_record["time"],
    )
    cmems_record = safe_model_call(
        sample_cmems_surface_point, "CMEMS",
        glider_record["lon"], glider_record["lat"], glider_record["time"],
    )
    lusitania_record = safe_model_call(
        sample_lusitania_surface_point, "Lusitania",
        glider_record["lon"], glider_record["lat"], glider_record["time"],
    )

    vector_records = [glider_record]
    for model_name, record in [
        ("RTOFS",     rtofs_record),
        ("ESPC",      espc_record),
        ("CMEMS",     cmems_record),
        ("Lusitania", lusitania_record),
    ]:
        if record is not None:
            vector_records.append(record)
        elif MODEL_CONFIG.get(model_name, {}).get("enabled", False):
            vector_records.append(_build_unavailable_vector_record(model_name))

    if len(vector_records) == 1:
        LOGGER.warning("No model datasets returned valid data")

    summary_df = build_summary_table(vector_records)
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("RESULTS SUMMARY")
    LOGGER.info("%s", "\n" + summary_df.to_string(index=False))
    LOGGER.info("%s", "-" * 60)

    plot_vectors(
        vector_records,
        summary_df,
        glider_record,
        enabled_models,
        output_files=output_files,
    )


def main():
    LOGGER.info(
        "Running surface current comparison (mode=%s, deployment=%s)",
        SURFACING_MODE,
        DEPLOYMENT_ID,
    )
    print_model_status()
    LOGGER.info("%s", "=" * 60)

    if SURFACING_MODE == "range":
        if not SURFACING_RANGE_START or not SURFACING_RANGE_END:
            raise ValueError(
                "SURFACING_RANGE_START and SURFACING_RANGE_END must be set "
                "when SURFACING_MODE is 'range'."
            )
        glider_records = get_surface_records_for_range(
            DEPLOYMENT_ID, SURFACING_RANGE_START, SURFACING_RANGE_END,
        )
        if not glider_records:
            LOGGER.warning("No surfacing records found in the requested range.")
            return

        for idx, glider_record in enumerate(glider_records, start=1):
            LOGGER.info(
                "Processing surfacing %d of %d (time=%s)",
                idx,
                len(glider_records),
                glider_record["time"],
            )
            compass_dir  = Path(save_path) / COMPASS_SUBDIR
            output_files = [
                _build_output_path(glider_record["time"]),
                _build_output_path(glider_record["time"], base_dir=compass_dir),
            ]
            process_surface_record(glider_record, output_files=output_files)

    else:
        glider_record = get_latest_surface_record(DEPLOYMENT_ID)
        compass_dir   = Path(save_path) / COMPASS_SUBDIR
        output_files  = [
            _build_output_path(),
            _build_output_path(glider_record["time"], base_dir=compass_dir),
        ]
        process_surface_record(glider_record, output_files=output_files)


if __name__ == "__main__":
    main()
