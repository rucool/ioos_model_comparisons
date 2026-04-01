"""
Extended Forecast Script: Redwing Glider Projection
----------------------------------------------------
This script extends ocean model forecasts beyond their availability by
holding the last u/v velocities constant ("persistence forecast").

Method:
1. Dynamic Reader: Uses the live 7-10 day forecast.
2. Static Reader: A snapshot of the last time step (frozen field) used as fallback.

Projection lengths: 30, 60, 90, 120 days
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re
import sys
import logging
import platform
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import shutil
import requests
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import copernicusmarine as cm
import cartopy.mpl.ticker as cticker
import cmocean.cm as cmo
from matplotlib.colors import Normalize

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BUFFER_DEG = 30.0  # Spatial buffer in degrees (~300km)

# Projection lengths in days
PROJECTION_DAYS = [90]

# Horizontal diffusivity values to test (m^2/s)
DIFFUSIVITY_VALUES = [100]

# Number of particles
NUM_PARTICLES = 20

# Model enable/disable flags
ENABLE_ESPC = True       # HYCOM/ESPC model
ENABLE_CMEMS = True      # Copernicus Marine CMEMS model
ENABLE_LUSITANIA = True  # Lusitania regional model

# Current visualization options
# When True, creates 3 animations: no currents, CMEMS currents, ESPC currents
PLOT_CURRENTS = True
CURRENT_STYLE = 'quiver'  # 'quiver' or 'streamplot'
QUIVER_SUBSAMPLE_CMEMS = 6  # Subsample factor for CMEMS quiver arrows (higher = fewer arrows)
QUIVER_SUBSAMPLE_ESPC = 8  # Subsample factor for ESPC quiver arrows (higher = fewer arrows)
QUIVER_SUBSAMPLE_LUSITANIA = 6  # Subsample factor for Lusitania quiver arrows (higher = fewer arrows)
STREAMPLOT_DENSITY = 3  # Density of streamlines (higher = more lines)

# Bathymetry configuration
if platform.system() == "Darwin":  # macOS
    BATHY_FILE = '/Users/mikesmith/Documents/data/SRTM15_V2.4.nc'
else:  # Linux server
    BATHY_FILE = '/home/hurricaneadm/data/srtm15/SRTM15_V2.4.nc'
BATHY_LEVELS = [-8000, -1000, -100, 0]
BATHY_COLORS = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']

# Output configuration (platform-dependent)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if platform.system() == "Darwin":  # macOS
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "redwing")
    LATEST_DIR = OUTPUT_ROOT
else:  # Linux server
    OUTPUT_ROOT = "/www/web/rucool/media/sentinel/drift/redwing"
    LATEST_DIR = "/www/web/rucool/media/sentinel"

# ============================================================================
# START POSITION CONFIGURATION
# ============================================================================
# Option 1: Use a glider deployment ID to fetch the last surfacing position
#           Set GLIDER_DEPLOYMENT_ID to the deployment ID string
# Option 2: Manually specify a starting position
#           Set GLIDER_DEPLOYMENT_ID to None and fill in MANUAL_START_LAT/LON
# ============================================================================

# Glider deployment ID (set to None to use manual start position)
GLIDER_DEPLOYMENT_ID = "redwing-20251011T1511"

# Manual start position (only used if GLIDER_DEPLOYMENT_ID is None)
MANUAL_START_LAT = 0                     # Decimal degrees
MANUAL_START_LON = 0                   # Decimal degrees

# ============================================================================
# START TIME CONFIGURATION
# ============================================================================
# REAL_TIME = True:  Use current datetime as start time (for current forecasts)
# REAL_TIME = False: Use MANUAL_START_TIME as start time (for historical runs)
# ============================================================================
REAL_TIME = False
MANUAL_START_TIME = "2025-01-15 12:00:00"  # UTC time string (only used if REAL_TIME=False)

# Map extent [lon_min, lon_max, lat_min, lat_max]
MAP_EXTENT = [-40.25, -9.75, 19.75, 40.25]

# =============================================================================
# LUSITANIA CONFIGURATION
# =============================================================================
# File naming: YYYYMMDDHH.nc (e.g., 2026010300.nc)
# Each file contains one complete day: from 00:00 GMT of the date to 00:00 GMT
# of the next day. For example, 2026010300.nc covers 2026-01-03 00:00 GMT to
# 2026-01-04 00:00 GMT.
#
# The model runs daily with a hindcast period (day -2), current day, and
# 6 days ahead forecast, providing approximately 3-day forecast capability.
# =============================================================================

LUSITANIA_CONFIG = {
    'base_url': 'https://thredds.atlanticsense.com/thredds',
    'catalog_path': '/catalog/atlDatasets/Lusitania/catalog.html',
    'opendap_base': '/dodsC/atlDatasets/Lusitania/',
    'file_pattern': r'(\d{10})\.nc',  # YYYYMMDDHH.nc
    'timeout': 30,
}


# =============================================================================
# LUSITANIA MODEL FUNCTIONS
# =============================================================================

def _parse_lusitania_catalog(timeout: int = 30):
    """
    Parse the Lusitania THREDDS catalog to get available files.

    Returns
    -------
    list of tuple
        List of (datetime, filename) tuples sorted by date ascending
    """
    catalog_url = (
        f"{LUSITANIA_CONFIG['base_url']}{LUSITANIA_CONFIG['catalog_path']}"
    )

    try:
        response = requests.get(catalog_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to fetch Lusitania catalog: {exc}")
        return []

    # Parse HTML to find netCDF file links
    pattern = re.compile(LUSITANIA_CONFIG['file_pattern'])
    available_files = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            text = link.get_text(strip=True)
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                try:
                    file_date = datetime.strptime(date_str, '%Y%m%d%H')
                    available_files.append((file_date, text))
                except ValueError as exc:
                    logger.warning(f"Could not parse date from {text}: {exc}")
                    continue
    else:
        # Fallback: use regex on raw HTML
        for match in pattern.finditer(response.text):
            date_str = match.group(1)
            filename = f"{date_str}.nc"
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d%H')
                if (file_date, filename) not in available_files:
                    available_files.append((file_date, filename))
            except ValueError as exc:
                logger.warning(f"Could not parse date from {filename}: {exc}")
                continue

    available_files.sort(key=lambda x: x[0])

    logger.info(f"Found {len(available_files)} Lusitania files in catalog")
    if available_files:
        logger.info(f"Date range: {available_files[0][0]} to {available_files[-1][0]}")

    return available_files


@lru_cache(maxsize=1)
def _get_lusitania_catalog():
    """Get cached catalog listing (refreshes once per session)."""
    return _parse_lusitania_catalog(timeout=LUSITANIA_CONFIG['timeout'])


def _find_lusitania_file(target_time, available_files=None, method='nearest'):
    """
    Find the Lusitania file nearest to the target time.

    Parameters
    ----------
    target_time : datetime
        Target time to match
    available_files : list, optional
        List of (datetime, filename) tuples. If None, fetches from catalog.
    method : str, optional
        Selection method: 'nearest', 'before', or 'after'

    Returns
    -------
    tuple or None
        (datetime, filename) of the best matching file, or None if no match
    """
    if available_files is None:
        available_files = _get_lusitania_catalog()

    if not available_files:
        logger.warning("No Lusitania files available in catalog")
        return None

    # Make target_time naive if it has timezone info
    if target_time.tzinfo is not None:
        target_time = target_time.replace(tzinfo=None)

    if method == 'before':
        candidates = [f for f in available_files if f[0] <= target_time]
        if not candidates:
            logger.warning(f"No Lusitania files available before {target_time}")
            return None
        return candidates[-1]

    elif method == 'after':
        candidates = [f for f in available_files if f[0] >= target_time]
        if not candidates:
            logger.warning(f"No Lusitania files available after {target_time}")
            return None
        return candidates[0]

    else:  # 'nearest'
        best_match = None
        min_diff = None

        for file_date, filename in available_files:
            diff = abs((file_date - target_time).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best_match = (file_date, filename)

        if best_match:
            logger.info(
                f"Nearest Lusitania file to {target_time}: "
                f"{best_match[1]} (offset: {min_diff/3600:.1f} hours)"
            )

        return best_match


def lusitania_uv(target_time=None, rename=True, method='nearest'):
    """
    Load Lusitania model data from the AtlanticSense THREDDS server.

    Parameters
    ----------
    target_time : datetime, optional
        Target time to load. If None, loads the most recent available file.
    rename : bool, optional
        If True, rename velocity variables to 'water_u' and 'water_v'.
    method : str, optional
        Time matching method: 'nearest', 'before', or 'after'.

    Returns
    -------
    xr.Dataset or None
        Dataset containing velocity variables, or None if loading fails.
    """
    # If no target time, use most recent file
    if target_time is None:
        available = _get_lusitania_catalog()
        if not available:
            logger.error("No Lusitania files available")
            return None
        file_date, filename = available[-1]
        logger.info(f"Loading most recent Lusitania file: {filename}")
    else:
        result = _find_lusitania_file(target_time, method=method)
        if result is None:
            return None
        file_date, filename = result

    # Construct OPeNDAP URL
    opendap_url = (
        f"{LUSITANIA_CONFIG['base_url']}"
        f"{LUSITANIA_CONFIG['opendap_base']}{filename}"
    )

    logger.info(f"Loading Lusitania data from: {opendap_url}")

    try:
        ds = xr.open_dataset(opendap_url)

        logger.info(f"Lusitania variables: {list(ds.data_vars)}")
        logger.info(f"Lusitania coordinates: {list(ds.coords)}")
        logger.info(f"Lusitania dimensions: {dict(ds.dims)}")

        ds.attrs['model'] = 'Lusitania'
        ds.attrs['source_file'] = filename
        ds.attrs['source_url'] = opendap_url

        if rename:
            rename_map = {}
            # Map to OpenDrift-compatible names (water_u, water_v)
            u_candidates = ['uo', 'u', 'water_u', 'u_velocity', 'U', 'ucur']
            v_candidates = ['vo', 'v', 'water_v', 'v_velocity', 'V', 'vcur']

            for u_name in u_candidates:
                if u_name in ds.data_vars and u_name != 'water_u':
                    rename_map[u_name] = 'water_u'
                    break

            for v_name in v_candidates:
                if v_name in ds.data_vars and v_name != 'water_v':
                    rename_map[v_name] = 'water_v'
                    break

            if rename_map:
                logger.info(f"Renaming variables: {rename_map}")
                ds = ds.rename(rename_map)

        return ds

    except Exception as exc:
        logger.error(f"Failed to load Lusitania data: {exc}")
        return None


def lusitania_uv_multi(start_time, rename=True):
    """
    Load all available Lusitania files from start_time onwards and concatenate them.

    Each Lusitania file contains ~1 day of data. This function loads all files
    from the start_time onwards to maximize forecast coverage.

    Parameters
    ----------
    start_time : datetime
        Start time - will load all files from this time onwards.
    rename : bool, optional
        If True, rename velocity variables to 'water_u' and 'water_v'.

    Returns
    -------
    xr.Dataset or None
        Concatenated dataset containing all available time steps, or None if loading fails.
    """
    available = _get_lusitania_catalog()
    if not available:
        logger.error("No Lusitania files available")
        return None

    # Make start_time naive if it has timezone info
    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    # Find all files from start_time onwards (or the closest one before if none start exactly at start_time)
    # First, find the file that contains start_time (method='before')
    start_result = _find_lusitania_file(start_time, available_files=available, method='before')
    if start_result is None:
        # If no file before, try to get the first available
        start_result = available[0] if available else None
    if start_result is None:
        logger.error("Could not find starting Lusitania file")
        return None

    start_file_date = start_result[0]

    # Get all files from start_file_date onwards
    files_to_load = [(fd, fn) for fd, fn in available if fd >= start_file_date]

    if not files_to_load:
        logger.warning(f"No Lusitania files available from {start_time}")
        return None

    logger.info(f"Loading {len(files_to_load)} Lusitania files from {files_to_load[0][0]} to {files_to_load[-1][0]}")

    datasets = []
    for file_date, filename in files_to_load:
        opendap_url = (
            f"{LUSITANIA_CONFIG['base_url']}"
            f"{LUSITANIA_CONFIG['opendap_base']}{filename}"
        )

        try:
            logger.info(f"  Loading: {filename}")
            ds = xr.open_dataset(opendap_url)

            if rename:
                rename_map = {}
                u_candidates = ['uo', 'u', 'water_u', 'u_velocity', 'U', 'ucur']
                v_candidates = ['vo', 'v', 'water_v', 'v_velocity', 'V', 'vcur']

                for u_name in u_candidates:
                    if u_name in ds.data_vars and u_name != 'water_u':
                        rename_map[u_name] = 'water_u'
                        break

                for v_name in v_candidates:
                    if v_name in ds.data_vars and v_name != 'water_v':
                        rename_map[v_name] = 'water_v'
                        break

                if rename_map:
                    ds = ds.rename(rename_map)

            datasets.append(ds)

        except Exception as exc:
            logger.warning(f"  Failed to load {filename}: {exc}")
            continue

    if not datasets:
        logger.error("Failed to load any Lusitania files")
        return None

    # Concatenate along time dimension
    logger.info(f"Concatenating {len(datasets)} Lusitania datasets...")
    try:
        combined = xr.concat(datasets, dim='time')
        combined = combined.sortby('time')

        # Remove duplicate time steps if any
        _, unique_indices = np.unique(combined['time'].values, return_index=True)
        combined = combined.isel(time=sorted(unique_indices))

        combined.attrs['model'] = 'Lusitania'
        combined.attrs['source_files'] = [fn for _, fn in files_to_load]

        time_range = pd.to_datetime(combined['time'].values)
        logger.info(f"Lusitania combined dataset: {len(time_range)} time steps from {time_range[0]} to {time_range[-1]}")

        return combined

    except Exception as exc:
        logger.error(f"Failed to concatenate Lusitania datasets: {exc}")
        return None


def dm_to_dd(dm_coord):
    """Convert degrees-minutes to decimal degrees."""
    sign = 1 if dm_coord > 0 else -1
    dm_coord = abs(dm_coord)
    degrees = int(dm_coord // 100)
    minutes = dm_coord % 100
    dd_coord = degrees + (minutes / 60)
    return sign * dd_coord

def get_glider_position(deployment_id):
    """
    Query the RUCOOL glider surfacings API for ALL positions 
    to create a historical track.
    """
    surfacings_url = f"https://marine.rutgers.edu/cool/data/gliders/api/surfacings/?deployment={deployment_id}"
    response = requests.get(surfacings_url)
    data = response.json()

    if not data.get("data"):
        raise ValueError(f"No surfacing data found for deployment '{deployment_id}'")

    # The API returns most recent first; reverse it for a chronological track
    all_surfacings = data["data"][::-1]

    # Extract all latitudes and longitudes for the track
    track_lats = []
    track_lons = []
    for s in all_surfacings:
        if s.get("gps_lat_degrees") is not None:
            track_lats.append(s["gps_lat_degrees"])
            track_lons.append(s["gps_lon_degrees"])
        else:
            track_lats.append(dm_to_dd(s["gps_lat"]))
            track_lons.append(dm_to_dd(s["gps_lon"]))

    # Most recent point (start of the forecast)
    last_s = data["data"][0]
    dt_utc = datetime.fromtimestamp(last_s.get("connect_time_epoch"), timezone.utc)

    return {
        "name": last_s.get("glider_name", deployment_id),
        "lat": track_lats[-1],
        "lon": track_lons[-1],
        "track_lats": track_lats,
        "track_lons": track_lons,
        "last_surfacing_time": dt_utc.strftime('%Y-%m-%d %H:%M:%S')
    }

#def get_glider_position(deployment_id):
#    """
#    Query the RUCOOL glider surfacings API for the last available position
#    from a specific deployment.
#    """
#    surfacings_url = f"https://marine.rutgers.edu/cool/data/gliders/api/surfacings/?deployment={deployment_id}"
#    response = requests.get(surfacings_url)
#    data = response.json()
#
#    if not data.get("data"):
#        raise ValueError(f"No surfacing data found for deployment '{deployment_id}'")
#
#    # Get the most recent surfacing event (first item in the array)
#    last_surfacing = data["data"][0]
#
#    # Use decimal degrees directly if available, otherwise convert
#    if "gps_lat_degrees" in last_surfacing and last_surfacing["gps_lat_degrees"] is not None:
#        lat_dd = last_surfacing["gps_lat_degrees"]
#        lon_dd = last_surfacing["gps_lon_degrees"]
#    else:
#        lat_dd = dm_to_dd(last_surfacing["gps_lat"])
#        lon_dd = dm_to_dd(last_surfacing["gps_lon"])
#
#    epoch_time = last_surfacing.get("connect_time_epoch")
#    dt_utc = datetime.fromtimestamp(epoch_time, timezone.utc)
#    # dt_utc.astimezone(None) # Convert to local timezone for display
#    formatted_time = dt_utc.strftime('%Y-%m-%d %H:%M:%S')
#
#    # print("UTC Date:", dt_utc)
#
#    return {
#        "name": last_surfacing.get("glider_name", deployment_id),
#        "lat": lat_dd,
#        "lon": lon_dd,
#        "last_surfacing_time": formatted_time
#    }


def get_manual_start_position():
    """Return a start position dict from manual configuration values."""
    return {
        "name": "Manual",
        "lat": MANUAL_START_LAT,
        "lon": MANUAL_START_LON,
        "last_surfacing_time": MANUAL_START_TIME
    }


def get_start_time():
    """
    Get the start time based on configuration.
    If REAL_TIME=True, returns current UTC time rounded to nearest hour.
    If REAL_TIME=False, returns MANUAL_START_TIME.
    """
    if REAL_TIME:
        now = datetime.now(tz=None)  # Local time, will be treated as UTC
        # Round to nearest hour
        if now.minute >= 30:
            now += timedelta(hours=1)
        return now.replace(minute=0, second=0, microsecond=0)
    else:
        return datetime.strptime(MANUAL_START_TIME, "%Y-%m-%d %H:%M:%S")


def extend_forecast_to_persistence(ds, model_name, output_file, target_days=120, time_step_hours=6):
    """
    Extend a forecast dataset to a target duration by persisting the last time step.

    Parameters:
    -----------
    ds : xarray.Dataset
        The original forecast dataset with time dimension
    model_name : str
        Name of the model (for logging)
    output_file : str
        Path to save the extended NetCDF file
    target_days : int
        Total duration in days for the extended forecast
    time_step_hours : int
        Time step interval in hours for the extended portion

    Returns:
    --------
    str : Path to the extended NetCDF file
    """
    print(f"  {model_name}: Extending forecast to {target_days} days with {time_step_hours}h time steps...")

    # Check if dataset has time dimension with data
    if 'time' not in ds.dims or ds.sizes['time'] == 0:
        raise ValueError(f"{model_name} dataset has no time dimension or is empty")

    # Sort by time to ensure correct ordering
    ds = ds.sortby('time')

    # Get the time range from the original dataset
    original_times = pd.to_datetime(ds['time'].values)
    start_time = original_times[0]
    last_forecast_time = original_times[-1]
    target_end_time = start_time + pd.Timedelta(days=target_days)

    print(f"    Original forecast: {start_time} to {last_forecast_time}")
    print(f"    Target end time: {target_end_time}")

    # If already long enough, just return
    if last_forecast_time >= target_end_time:
        print(f"    Forecast already covers target duration, saving as-is...")
        ds.to_netcdf(output_file)
        return output_file

    # Generate new time steps from end of forecast to target end
    persistence_times = pd.date_range(
        start=last_forecast_time + pd.Timedelta(hours=time_step_hours),
        end=target_end_time,
        freq=f'{time_step_hours}h'
    )

    print(f"    Adding {len(persistence_times)} persistence time steps...")

    # Get the last time step data
    last_step = ds.isel(time=-1)

    # Create persistence dataset efficiently in one operation
    if len(persistence_times) > 0:
        persistence_ds = last_step.expand_dims(time=persistence_times).assign_coords(time=persistence_times)
        extended_ds = xr.concat([ds, persistence_ds], dim='time')
    else:
        extended_ds = ds

    # Preserve/restore CF-compliant coordinate attributes for OpenDrift compatibility
    # OpenDrift needs these attributes to recognize geospatial coordinates
    for coord_name in extended_ds.coords:
        coord_lower = coord_name.lower()
        if coord_lower in ['lon', 'longitude']:
            extended_ds[coord_name].attrs.update({
                'standard_name': 'longitude',
                'units': 'degrees_east',
                'axis': 'X'
            })
        elif coord_lower in ['lat', 'latitude']:
            extended_ds[coord_name].attrs.update({
                'standard_name': 'latitude',
                'units': 'degrees_north',
                'axis': 'Y'
            })
        elif coord_lower == 'time':
            extended_ds[coord_name].attrs.update({
                'standard_name': 'time',
                'axis': 'T'
            })

    # Save to file
    if os.path.exists(output_file):
        os.remove(output_file)
    extended_ds.to_netcdf(output_file)

    final_times = pd.to_datetime(extended_ds['time'].values)
    print(f"    Extended forecast: {final_times[0]} to {final_times[-1]} ({len(final_times)} time steps)")
    print(f"    Saved to: {output_file}")

    return output_file


def get_run_identifier(start_time, glider_name=None):
    """
    Generate a unique identifier for this run based on start time and optional glider name.

    Returns a string like: "redwing_20250115T120000" or "manual_20250115T120000"
    """
    time_str = start_time.strftime("%Y%m%dT%H%M%S")
    if glider_name and glider_name != "Manual":
        # Extract just the glider name (before the deployment date)
        name = glider_name.split('-')[0].lower()
        return f"{name}_{time_str}"
    return f"manual_{time_str}"


def get_forecast_filename(model_name, ds, suffix="raw"):
    """
    Generate a filename for downloaded model data based on initialization and end times.

    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "espc", "cmems")
    ds : xarray.Dataset
        The dataset containing time dimension
    suffix : str
        File suffix (e.g., "raw", "extended")

    Returns:
    --------
    str : Filename like "espc_20250122T12_to_20250129T12_raw.nc"
    """
    times = pd.to_datetime(ds['time'].values)
    init_time = times[0].strftime("%Y%m%dT%H")
    end_time = times[-1].strftime("%Y%m%dT%H")
    return f"{model_name.lower()}_{init_time}_to_{end_time}_{suffix}.nc"


def prepare_output_directories(run_id, start_time):
    """
    Build and create the output directory tree for a run, grouped by YYYY_MM.
    Returns a dict with paths for the run root, data, projections, and figures.
    """
    month_folder = start_time.strftime("%Y_%m")
    run_root = os.path.join(OUTPUT_ROOT, month_folder, run_id)
    data_dir = os.path.join(run_root, "data")
    projections_dir = os.path.join(run_root, "projections")
    figures_dir = os.path.join(run_root, "figures")

    for path in (data_dir, projections_dir, figures_dir):
        os.makedirs(path, exist_ok=True)

    return {
        "run_root": run_root,
        "data": data_dir,
        "projections": projections_dir,
        "figures": figures_dir
    }


def create_static_persistence_file(ds, model_name, output_dir):
    """
    Creates a static NetCDF file from the last time step of the input dataset.
    This static file will have NO time dimension, acting as a permanent fallback.
    """
    print(f"  {model_name}: Creating static persistence file from last time step...")
    
    # 1. Select the last time step
    # .isel(time=-1) grabs the last index. 
    ds_last = ds.isel(time=-1)
    
    # 2. Drop the time coordinate entirely
    # This forces OpenDrift to treat it as a static map (valid for all time)
    ds_last = ds_last.drop_vars(['time'], errors='ignore')
    
    # Quick debug to ensure we aren't persisting zero velocity
    # Attempts to find standard velocity names
    for v in ['water_u', 'uo', 'u']:
        if v in ds_last:
            u_mean = float(ds_last[v].mean())
            print(f"  DEBUG {model_name}: Persistence field Mean {v} = {u_mean:.4f} m/s")
            break

    # 3. Save to disk
    filename = os.path.join(output_dir, f"{model_name}_static_persistence.nc")
    if os.path.exists(filename):
        os.remove(filename)
        
    ds_last.to_netcdf(filename)
    print(f"  {model_name}: Saved static persistence file to {filename}")
    
    return filename


def run_extended_forecast(glider_info, num_particles, projection_days, output_dir,
                          model_name, readers, horizontal_diffusivity=0):
    """
    Run OpenDrift forecast from glider position.
    accepts 'readers' which can be a single reader or a list of readers.
    """
    start_time = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    if start_time.minute > 0 or start_time.second > 0:
        start_time += timedelta(hours=1)
        start_time = start_time.replace(minute=0, second=0, microsecond=0)

    end_time = start_time + timedelta(days=projection_days)

    o = OceanDrift(loglevel=50)
    if horizontal_diffusivity > 0:
        o.set_config('drift:horizontal_diffusivity', horizontal_diffusivity)
    
    # Add the list of readers (Order matters: [Dynamic, Static])
    o.add_reader(readers)

    o.seed_elements(
        lon=glider_info['lon'],
        lat=glider_info['lat'],
        time=start_time,
        number=num_particles,
        radius=500 if num_particles > 1 else 0
    )

    output_file = os.path.join(
        output_dir,
        f"redwing_{model_name}_{projection_days}days_diff{horizontal_diffusivity}_forecast.nc"
    )

    # Use 1-hour time steps for the simulation
    o.run(end_time=end_time, time_step=3600, outfile=output_file)

    return output_file, start_time, end_time


# =============================================================================
# BATHYMETRY
# =============================================================================

def load_bathymetry(extent):
    """Load SRTM15 bathymetry data subsetted to the given map extent."""
    bathy = xr.open_dataset(BATHY_FILE)
    bathy = bathy.sel(
        lon=slice(extent[0], extent[1]),
        lat=slice(extent[2], extent[3]),
    )
    bathy = bathy.rename({'lon': 'longitude', 'lat': 'latitude'})
    return bathy


def add_tricolor_bathymetry(ax, extent):
    """Add tricolor bathymetry contourf to a cartopy axes."""
    bathy = load_bathymetry(extent)
    ax.contourf(
        bathy['longitude'], bathy['latitude'], bathy['z'],
        BATHY_LEVELS, colors=BATHY_COLORS,
        transform=ccrs.PlateCarree(), zorder=0.05,
    )
    bathy.close()


def _nice_interval(span, target_ticks=5):
    """
    Return a 'nice' tick interval that yields roughly *target_ticks* ticks.

    Only returns intervals that are multiples of 0.25, 0.5, 1, 2, 5, 10, etc.
    This ensures clean degree intervals for lat/lon coordinates.
    """
    if span <= 0:
        return 1

    raw = span / target_ticks

    # Allowed intervals in ascending order
    allowed = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]

    # Find the best interval
    for interval in allowed:
        if interval >= raw:
            return interval

    # If span is huge, use the largest interval
    return allowed[-1]


def style_ticks(ax, extent):
    """
    Apply tick and tick-label styling matching redwing_hindcast_verification.py.

    Major and minor intervals are chosen adaptively so that each axis
    gets roughly 4-8 labelled ticks regardless of the extent size.
    Labels use cartopy LongitudeFormatter / LatitudeFormatter with bold font.

    Returns (major_x, major_y, minor_x, minor_y) arrays for use with gridline locators.
    """
    import math

    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    # Choose adaptive intervals
    major_dx = _nice_interval(lon_span)
    major_dy = _nice_interval(lat_span)
    minor_dx = major_dx / 5
    minor_dy = major_dy / 5

    # Snap first tick outward to a nice boundary just inside the extent
    major_x = np.arange(math.ceil(lon_min / major_dx) * major_dx,
                        lon_max + major_dx * 0.01, major_dx)
    major_y = np.arange(math.ceil(lat_min / major_dy) * major_dy,
                        lat_max + major_dy * 0.01, major_dy)
    minor_x = np.arange(math.ceil(lon_min / minor_dx) * minor_dx,
                        lon_max + minor_dx * 0.01, minor_dx)
    minor_y = np.arange(math.ceil(lat_min / minor_dy) * minor_dy,
                        lat_max + minor_dy * 0.01, minor_dy)

    ax.set_xticks(major_x, crs=ccrs.PlateCarree())
    ax.set_yticks(major_y, crs=ccrs.PlateCarree())
    ax.set_xticks(minor_x, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(minor_y, minor=True, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.tick_params(
        axis='both', which='major',
        labelsize=12, direction='out',
        length=6, width=1,
        top=True, right=True,
    )
    ax.tick_params(
        axis='both', which='minor',
        direction='out', length=3, width=1,
        top=True, right=True,
    )

    return major_x, major_y, minor_x, minor_y


# =============================================================================
# PLOTTING
# =============================================================================

def create_comparison_plot(all_outputs, glider_info, figures_dir, projection_days, extent):
    """Create a comparison plot showing different diffusivity values."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_facecolor('white')
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Tricolor bathymetry
    add_tricolor_bathymetry(ax, extent)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='#a0a0a0', linewidth=0.25), zorder=0.1)

    tickx, ticky, minor_tickx, minor_ticky = style_ticks(ax, extent)

    # Major gridlines
    gl_major = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_major.xlocator = mticker.FixedLocator(tickx)
    gl_major.ylocator = mticker.FixedLocator(ticky)
    gl_major.top_labels = False
    gl_major.right_labels = False

    # Minor gridlines
    gl_minor = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_minor.xlocator = mticker.FixedLocator(minor_tickx)
    gl_minor.ylocator = mticker.FixedLocator(minor_ticky)

    colors = plt.cm.viridis(np.linspace(0, 1, len(DIFFUSIVITY_VALUES)))

    for idx, (diff_val, outputs) in enumerate(all_outputs.items()):
        color = colors[idx]
        for model_name, ds in outputs.items():
            lons = ds['lon'].values
            lats = ds['lat'].values

            mean_lon = np.nanmean(lons, axis=0)
            mean_lat = np.nanmean(lats, axis=0)

            linestyle = '-' if model_name == 'ESPC' else '--'
            ax.plot(mean_lon, mean_lat, color=color, linestyle=linestyle, linewidth=2, label=f'{model_name} Diff={diff_val}')

            for i in range(min(5, ds.dims['trajectory'])):
                ax.plot(lons[i, :], lats[i, :], color=color, alpha=0.2, linewidth=0.5, linestyle=linestyle)

            ax.scatter(mean_lon[-1], mean_lat[-1], color=color, s=50, marker='o' if model_name == 'ESPC' else 's', zorder=5)

    ax.scatter(glider_info['lon'], glider_info['lat'], color='red', s=150, marker='*', zorder=10, label=f"Start: {glider_info['last_surfacing_time']}")
    main_legend = ax.legend(loc='upper left', fontsize=8)
    ax.add_artist(main_legend)

    # Bathymetry legend
    bathy_handles = [
        Patch(facecolor='lightsteelblue', edgecolor='none', label='0\u2013100 m'),
        Patch(facecolor=tuple(cfeature.COLORS['water']), edgecolor='none', label='100\u20131000 m'),
        Patch(facecolor='cornflowerblue', edgecolor='none', label='1000+ m'),
    ]
    ax.legend(handles=bathy_handles, loc='lower left', fontsize=9,
              title='Bathymetry', title_fontsize=10)

    ax.set_title(f"Redwing Glider {projection_days}-Day Projection\nComparing Diffusivity Values")

    os.makedirs(figures_dir, exist_ok=True)
    # Format start time for filename
    start_dt = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    time_str = start_dt.strftime("%Y%m%dT%H")
    output_path = os.path.join(figures_dir, f"redwing_{time_str}_{projection_days}day_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved: {output_path}")
    return output_path


def create_single_run_map(model_outputs, glider_info, output_dir, projection_days, diffusivity, extent):
    """Create a static map for a single projection run."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_facecolor('white')
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Tricolor bathymetry
    add_tricolor_bathymetry(ax, extent)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='#a0a0a0', linewidth=0.25), zorder=0.1)

    tickx, ticky, minor_tickx, minor_ticky = style_ticks(ax, extent)

    # Major gridlines
    gl_major = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_major.xlocator = mticker.FixedLocator(tickx)
    gl_major.ylocator = mticker.FixedLocator(ticky)
    gl_major.top_labels = False
    gl_major.right_labels = False

    # Minor gridlines
    gl_minor = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_minor.xlocator = mticker.FixedLocator(minor_tickx)
    gl_minor.ylocator = mticker.FixedLocator(minor_ticky)

    # Plot the historical glider track
    ax.plot(glider_info['track_lons'], glider_info['track_lats'], color='black', linewidth=1.5, linestyle='-',
            transform=ccrs.PlateCarree(), label="Glider Trail", zorder=8)

    colors = {'ESPC': 'orange', 'CMEMS': 'magenta', 'Lusitania': 'cyan'}

    for model_name, ds in model_outputs.items():
        lons = ds['lon'].values
        lats = ds['lat'].values

        for i in range(ds.dims['trajectory']):
            ax.plot(lons[i, :], lats[i, :], color=colors[model_name], alpha=0.3, linewidth=1, label=model_name if i == 0 else None)
            ax.scatter(lons[i, -1], lats[i, -1], color=colors[model_name], s=20, zorder=5, alpha=0.5)

    ax.scatter(glider_info['lon'], glider_info['lat'], color='red', s=100, marker='*', zorder=10)

    # Build legend handles in prescribed order: Glider Track, Start, then models
    main_handles = [
        Line2D([], [], color='black', linewidth=3, label='Glider Trail'),
        Line2D([], [], marker='*', color='red', linestyle='None', markersize=14, label='Current Position'),
    ]
    for mname in ['ESPC', 'CMEMS', 'Lusitania']:
        if mname in model_outputs:
            main_handles.append(
                Line2D([], [], color=colors[mname], linewidth=3, label=mname)
            )
    main_legend = ax.legend(handles=main_handles, loc='upper left', fontsize=10)
    ax.add_artist(main_legend)

    # Bathymetry legend
    bathy_handles = [
        Patch(facecolor='lightsteelblue', edgecolor='none', label='0\u2013100 m'),
        Patch(facecolor=tuple(cfeature.COLORS['water']), edgecolor='none', label='100\u20131000 m'),
        Patch(facecolor='cornflowerblue', edgecolor='none', label='1000+ m'),
    ]
    ax.legend(handles=bathy_handles, loc='lower left', fontsize=9,
              title='Bathymetry', title_fontsize=10)

    ax.set_title(f"Redwing {projection_days}-Day Projection (Diffusivity={diffusivity} m\u00b2/s)\nStart: {glider_info['last_surfacing_time']} UTC")

    # Format start time for filename
    start_dt = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    time_str = start_dt.strftime("%Y%m%dT%H")
    output_path = os.path.join(output_dir, f"redwing_{time_str}_{projection_days}days_diff{diffusivity}_map.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Map saved: {output_path}")
    return output_path


def create_animation(model_outputs, glider_info, output_dir, projection_days, diffusivity, extent,
                     quiver_model=None, quiver_data=None, persistence_times=None):
    """
    Create an animation showing drifter trajectories over time.

    Parameters:
    -----------
    model_outputs : dict
        Dictionary of model name -> xarray Dataset with trajectory data
    glider_info : dict
        Glider position and metadata
    output_dir : str
        Directory to save output files
    projection_days : int
        Number of days in projection
    diffusivity : float
        Horizontal diffusivity value
    extent : list
        Map extent [lon_min, lon_max, lat_min, lat_max]
    quiver_model : str or None
        Model name for quiver overlay ('ESPC', 'CMEMS', or None for no quivers)
    quiver_data : dict or None
        Dictionary with 'ESPC' and/or 'CMEMS' xarray Datasets containing u/v fields
    persistence_times : dict or None
        Dictionary with model names -> datetime when persistence (forecast freeze) begins
    """
    first_model = list(model_outputs.keys())[0]
    times = model_outputs[first_model]['time'].values

    total_frames = len(times)
    frame_step = max(1, total_frames // 200)
    frame_indices = list(range(0, total_frames, frame_step))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    quiver_label = f"_{quiver_model.lower()}_quivers" if quiver_model else ""
    print(f"    Creating animation{quiver_label} with {len(frame_indices)} frames (step={frame_step})...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_facecolor('white')
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Tricolor bathymetry
    add_tricolor_bathymetry(ax, extent)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='#a0a0a0', linewidth=0.25), zorder=0.1)

    tickx, ticky, minor_tickx, minor_ticky = style_ticks(ax, extent)

    # Major gridlines
    gl_major = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_major.xlocator = mticker.FixedLocator(tickx)
    gl_major.ylocator = mticker.FixedLocator(ticky)
    gl_major.top_labels = False
    gl_major.right_labels = False

    # Minor gridlines
    gl_minor = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_minor.xlocator = mticker.FixedLocator(minor_tickx)
    gl_minor.ylocator = mticker.FixedLocator(minor_ticky)

    # Plot the historical glider track
    ax.plot(glider_info['track_lons'], glider_info['track_lats'],
            color='black', linewidth=1.5, linestyle='-',
            transform=ccrs.PlateCarree(), label="Glider Track", zorder=8)

    colors = {'ESPC': 'orange', 'CMEMS': 'magenta', 'Lusitania': 'cyan'}

    ax.scatter(glider_info['lon'], glider_info['lat'], color='red', s=150, marker='*', zorder=10, label="Start")

    model_scatters = {}
    model_trails_forecast = {}  # Thicker lines for forecast period
    model_trails_persistence = {}  # Thinner lines for persistence/extended period

    # Line widths: forecast = 4, persistence = 2
    FORECAST_LINEWIDTH = 4
    PERSISTENCE_LINEWIDTH = 2

    for model_name, ds in model_outputs.items():
        n_traj = ds.dims['trajectory']
        model_scatters[model_name] = ax.scatter([], [], color=colors[model_name], s=40, label=model_name, zorder=5)
        # Create two sets of trails - one for forecast (thick) and one for persistence (thin)
        model_trails_forecast[model_name] = [ax.plot([], [], color=colors[model_name], alpha=0.3, linewidth=FORECAST_LINEWIDTH)[0] for _ in range(n_traj)]
        model_trails_persistence[model_name] = [ax.plot([], [], color=colors[model_name], alpha=0.3, linewidth=PERSISTENCE_LINEWIDTH)[0] for _ in range(n_traj)]

    # Pre-compute persistence frame indices for each model
    persistence_frame_indices = {}
    if persistence_times:
        for model_name in model_outputs.keys():
            if model_name in persistence_times:
                persist_time = persistence_times[model_name]
                # Find the frame index where persistence starts
                for idx, t in enumerate(times):
                    if pd.to_datetime(t) >= persist_time:
                        persistence_frame_indices[model_name] = idx
                        break
                else:
                    persistence_frame_indices[model_name] = len(times)  # Never reaches persistence

    # Current visualization setup (quiver or streamplot)
    quiver_plot = None
    streamplot_container = [None]  # Use list to allow modification in nested function
    contourf_container = [None]  # Container for speed contour fill
    use_streamplot = CURRENT_STYLE == 'streamplot'
    Q_lon, Q_lat = None, None
    stream_lons, stream_lats = None, None
    contour_lons, contour_lats = None, None  # For contourf

    # Speed contour settings
    SPEED_LEVELS = np.linspace(0, 1.5, 16)  # 0 to 1.5 m/s in 16 levels
    SPEED_CMAP = cmo.speed

    # Determine subsample factor based on model
    if quiver_model == 'ESPC':
        quiver_subsample = QUIVER_SUBSAMPLE_ESPC
    elif quiver_model == 'Lusitania':
        quiver_subsample = QUIVER_SUBSAMPLE_LUSITANIA
    else:
        quiver_subsample = QUIVER_SUBSAMPLE_CMEMS

    if quiver_model and quiver_data and quiver_model in quiver_data:
        qds = quiver_data[quiver_model]
        # Get coordinate names
        lon_name = 'lon' if 'lon' in qds.coords else 'longitude'
        lat_name = 'lat' if 'lat' in qds.coords else 'latitude'

        # Store coordinates for contourf (always use full resolution)
        contour_lons = qds[lon_name].values
        contour_lats = qds[lat_name].values

        if use_streamplot:
            # For streamplot, use full resolution data
            stream_lons = qds[lon_name].values
            stream_lats = qds[lat_name].values
        else:
            # Subsample for quiver plotting
            q_lons = qds[lon_name].values[::quiver_subsample]
            q_lats = qds[lat_name].values[::quiver_subsample]

            # Create meshgrid for quiver
            Q_lon, Q_lat = np.meshgrid(q_lons, q_lats)

            # Initialize empty quiver
            quiver_plot = ax.quiver(
                Q_lon, Q_lat, np.zeros_like(Q_lon), np.zeros_like(Q_lat),
                transform=ccrs.PlateCarree(), color='black', alpha=0.6,
                scale=60, width=0.002, zorder=2
            )

    # Add colorbar for speed (only if we have quiver data)
    cbar = None
    if quiver_model and quiver_data and quiver_model in quiver_data:
        # Create a dummy mappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=SPEED_CMAP, norm=Normalize(vmin=0, vmax=1.5))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
        cbar.set_label('Current Speed (m/s)', fontsize=10)

    # Build legend handles in prescribed order: Glider Track, Start, then models
    main_handles = [
        Line2D([], [], color='black', linewidth=3, label='Glider Track'),
        Line2D([], [], marker='*', color='red', linestyle='None', markersize=14, label='Start'),
    ]
    for mname in ['ESPC', 'CMEMS', 'Lusitania']:
        if mname in model_outputs:
            main_handles.append(
                Line2D([], [], color=colors[mname], linewidth=3, label=mname)
            )
    legend_title = f"{quiver_model} Currents" if quiver_model else None
    main_legend = ax.legend(handles=main_handles, loc='upper left', fontsize=10,
                            title=legend_title)
    ax.add_artist(main_legend)

    # Bathymetry legend (persists across all frames)
    bathy_handles = [
        Patch(facecolor='lightsteelblue', edgecolor='none', label='0\u2013100 m'),
        Patch(facecolor=tuple(cfeature.COLORS['water']), edgecolor='none', label='100\u20131000 m'),
        Patch(facecolor='cornflowerblue', edgecolor='none', label='1000+ m'),
    ]
    ax.legend(handles=bathy_handles, loc='lower left', fontsize=9,
              title='Bathymetry', title_fontsize=10)

    title = ax.set_title("")

    def init():
        for model_name in model_outputs:
            model_scatters[model_name].set_offsets(np.empty((0, 2)))
            for trail in model_trails_forecast[model_name]:
                trail.set_data([], [])
            for trail in model_trails_persistence[model_name]:
                trail.set_data([], [])
        if quiver_plot is not None:
            quiver_plot.set_UVC(np.zeros_like(Q_lon), np.zeros_like(Q_lat))
        return []

    def update(frame_num):
        frame = frame_indices[frame_num]
        current_time = times[frame]
        current_time_dt = pd.to_datetime(current_time)
        start_time_anim = times[0]
        days_since_init = (current_time - start_time_anim) / np.timedelta64(1, 'D')
        start_str = np.datetime_as_string(start_time_anim, unit='h')
        time_str = np.datetime_as_string(current_time, unit='h')

        # Check if we're in persistence mode for the quiver model
        persistence_label = ""
        if quiver_model and persistence_times and quiver_model in persistence_times:
            if current_time_dt >= persistence_times[quiver_model]:
                persistence_label = " [PERSISTENCE]"

        title.set_text(
            f"Redwing {projection_days}-Day Projection (Diff={diffusivity}){quiver_label.replace('_', ' ').title()}\n"
            f"Start: {start_str} UTC | Current: {time_str} UTC{persistence_label} | Days since start: {days_since_init:.2f}"
        )

        artists = []

        for model_name, ds in model_outputs.items():
            lons_full = ds['lon'].values[:, :frame+1]
            lats_full = ds['lat'].values[:, :frame+1]
            current_lons = lons_full[:, -1]
            current_lats = lats_full[:, -1]
            model_scatters[model_name].set_offsets(np.c_[current_lons, current_lats])
            artists.append(model_scatters[model_name])

            # Get the persistence frame index for this model
            persist_idx = persistence_frame_indices.get(model_name, frame + 1)

            for i, (trail_fc, trail_ps) in enumerate(zip(model_trails_forecast[model_name], model_trails_persistence[model_name])):
                # Split the trajectory at the persistence point
                if frame < persist_idx:
                    # Still in forecast period - only draw forecast line
                    trail_fc.set_data(lons_full[i, :], lats_full[i, :])
                    trail_ps.set_data([], [])
                else:
                    # In persistence period - draw both lines
                    # Forecast line: from start to persistence point (include persist_idx for continuity)
                    trail_fc.set_data(lons_full[i, :persist_idx+1], lats_full[i, :persist_idx+1])
                    # Persistence line: from persistence point to current (overlap at persist_idx for continuity)
                    trail_ps.set_data(lons_full[i, persist_idx:], lats_full[i, persist_idx:])

                artists.append(trail_fc)
                artists.append(trail_ps)

        # Update current visualization (contourf, quiver or streamplot)
        if quiver_model and quiver_data and quiver_model in quiver_data:
            qds = quiver_data[quiver_model]

            # Get velocity variable names
            if 'water_u' in qds:
                u_name, v_name = 'water_u', 'water_v'
            elif 'uo' in qds:
                u_name, v_name = 'uo', 'vo'
            else:
                u_name, v_name = 'u', 'v'

            # Find nearest time in quiver data - get FULL resolution data first
            # Detect the actual time dimension on the velocity variable (may be 'time' or 'time1', etc.)
            u_da = qds[u_name]
            time_dim = next((d for d in u_da.dims if 'time' in d.lower()), None)
            if time_dim is not None:
                # Get times from the best available time coordinate
                time_coord_name = 'time' if 'time' in qds.coords else time_dim
                q_times = pd.to_datetime(qds[time_coord_name].values)
                # Find closest time (will use last available if past forecast end)
                time_idx = np.abs(q_times - current_time_dt).argmin()
                u_full = qds[u_name].isel(**{time_dim: time_idx}).values
                v_full = qds[v_name].isel(**{time_dim: time_idx}).values
            else:
                # Static field (no time dimension)
                u_full = qds[u_name].values
                v_full = qds[v_name].values

            # Calculate speed from full resolution data for contourf
            speed_full = np.sqrt(u_full**2 + v_full**2)

            # Remove previous contourf collections if they exist
            if contourf_container[0] is not None:
                contourf_container[0].remove()
                contourf_container[0] = None

            # Draw speed contourf (background layer, zorder=1)
            if contour_lons is not None and contour_lats is not None:
                contourf_container[0] = ax.contourf(
                    contour_lons, contour_lats, speed_full,
                    levels=SPEED_LEVELS, cmap=SPEED_CMAP,
                    transform=ccrs.PlateCarree(), zorder=1, extend='max'
                )

            # Now prepare data for quiver/streamplot
            if use_streamplot:
                u_data = np.nan_to_num(u_full, nan=0)
                v_data = np.nan_to_num(v_full, nan=0)
            else:
                u_data = np.nan_to_num(u_full[::quiver_subsample, ::quiver_subsample], nan=0)
                v_data = np.nan_to_num(v_full[::quiver_subsample, ::quiver_subsample], nan=0)

            if use_streamplot:
                # Remove previous streamplot if it exists
                if streamplot_container[0] is not None:
                    streamplot_container[0].lines.remove()
                    for art in ax.get_children():
                        if isinstance(art, matplotlib.patches.FancyArrowPatch):
                            art.remove()

                # Create new streamplot
                streamplot_container[0] = ax.streamplot(
                    stream_lons, stream_lats, u_data, v_data,
                    transform=ccrs.PlateCarree(), color='black',
                    density=STREAMPLOT_DENSITY, linewidth=0.5, arrowsize=0.8, zorder=2
                )
            else:
                # Normalize vectors to unit length (show direction only)
                magnitude = np.sqrt(u_data**2 + v_data**2)
                magnitude = np.where(magnitude == 0, 1, magnitude)  # Avoid division by zero
                u_data = u_data / magnitude
                v_data = v_data / magnitude

                quiver_plot.set_UVC(u_data, v_data)
                artists.append(quiver_plot)

        return artists

    # Use blit=False when showing currents (contourf/streamplot can't be updated in place)
    use_blit = not quiver_model
    ani = FuncAnimation(fig, update, init_func=init, frames=len(frame_indices), interval=100, blit=use_blit)
    # Format start time for filename
    start_dt = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    time_str = start_dt.strftime("%Y%m%dT%H")

    # Build filename with quiver suffix
    quiver_suffix = f"_{quiver_model.lower()}_quivers" if quiver_model else ""

    # Try ffmpeg for MP4, fall back to GIF if unavailable
    from matplotlib.animation import writers
    if writers.is_available('ffmpeg'):
        output_path = os.path.join(output_dir, f"redwing_{time_str}_{projection_days}days_diff{diffusivity}{quiver_suffix}_animation.mp4")
        ani.save(output_path, writer='ffmpeg', dpi=100)
    else:
        print("    WARNING: ffmpeg not available, saving as GIF instead")
        output_path = os.path.join(output_dir, f"redwing_{time_str}_{projection_days}days_diff{diffusivity}{quiver_suffix}_animation.gif")
        ani.save(output_path, writer='pillow', dpi=100)

    plt.close()
    print(f"    Animation saved: {output_path}")
    return output_path

# Archive configuration for flat structure
ARCHIVE_BASE = "/www/web/rucool/media/sentinel/drift/archive"
ARCHIVE_MOVIES = os.path.join(ARCHIVE_BASE, "90day_movies")
ARCHIVE_MAPS = os.path.join(ARCHIVE_BASE, "90day_maps")

def archive_90day_outputs(map_path, animation_path):
    """Copies the 90-day map and animation to a flat archive structure."""
    try:
        # Create archive directories if they don't exist
        os.makedirs(ARCHIVE_MOVIES, exist_ok=True)
        os.makedirs(ARCHIVE_MAPS, exist_ok=True)

        # Copy Map
        map_filename = os.path.basename(map_path)
        shutil.copy2(map_path, os.path.join(ARCHIVE_MAPS, map_filename))

        # Copy Animation
        anim_filename = os.path.basename(animation_path)
        shutil.copy2(animation_path, os.path.join(ARCHIVE_MOVIES, anim_filename))

        print(f"    Archived to: {ARCHIVE_BASE}")
    except Exception as e:
        print(f"    Error archiving 90-day files: {e}")


class _Tee:
    """Duplicate writes to both the original stream and a log file."""
    def __init__(self, stream, log_path):
        self._stream = stream
        self._file = open(log_path, 'w', buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def close(self):
        self._file.close()


def main():
    print("=" * 70)
    print("REDWING GLIDER EXTENDED PROJECTION SCRIPT (PERSISTENCE FIX)")
    print("=" * 70)
    print(f"Projection lengths: {PROJECTION_DAYS} days")
    print(f"Diffusivity values: {DIFFUSIVITY_VALUES} m²/s")
    print(f"Particles: {NUM_PARTICLES}")

    # [1/5] Get start position and time
    print("\n[1/5] Fetching start position and time...")
    try:
        if GLIDER_DEPLOYMENT_ID:
            print(f"  Using glider deployment: {GLIDER_DEPLOYMENT_ID}")
            glider_info = get_glider_position(GLIDER_DEPLOYMENT_ID)
        # else:
        #     print("  Using manual start position...")
        #     glider_info = get_manual_start_position()

        # Use the glider's actual surfacing time
        start_time = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
        # start_time = glider_info['last_surfacing_time']

        # Round up to nearest hour if needed
        if start_time.minute > 0 or start_time.second > 0:
            start_time += timedelta(hours=1)
            start_time = start_time.replace(minute=0, second=0, microsecond=0)

        print(f"  Name: {glider_info['name']}")
        print(f"  Last surfacing: {glider_info['last_surfacing_time']}")
        lat_dir = 'N' if glider_info['lat'] >= 0 else 'S'
        lon_dir = 'E' if glider_info['lon'] >= 0 else 'W'
        print(f"  Position: {abs(glider_info['lat']):.4f}{lat_dir}, {abs(glider_info['lon']):.4f}{lon_dir}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    max_projection_days = max(PROJECTION_DAYS)
    target_end_time = start_time + timedelta(days=max_projection_days)

    # Map extent
    extent = MAP_EXTENT

    # Generate run identifier for file naming
    run_id = get_run_identifier(start_time, glider_info['name'])
    print(f"\n  Run identifier: {run_id}")
    print(f"  Max projection: {max_projection_days} days")
    print(f"  Target end time: {target_end_time}")

    # Prepare output directories grouped by YYYY_MM
    output_dirs = prepare_output_directories(run_id, start_time)
    output_dir = output_dirs["run_root"]
    data_dir = output_dirs["data"]
    projections_dir = output_dirs["projections"]
    figures_dir = output_dirs["figures"]
    print(f"  Output base: {OUTPUT_ROOT}")
    print(f"  Run directory: {output_dir}")
    print(f"  Output grouping: {start_time.strftime('%Y_%m')}/")

    # Start logging to file — all subsequent print() calls go to both
    # the console and the log file.
    log_path = os.path.join(output_dir, f"run_{run_id}.log")
    _tee = _Tee(sys.stdout, log_path)
    sys.stdout = _tee
    # Also attach a FileHandler so logger.* calls are captured too
    _file_handler = logging.FileHandler(log_path)
    _file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(_file_handler)
    print(f"  Logging to: {log_path}")

    # [2/5] Load ESPC (HYCOM) data
    print("\n[2/5] Loading ESPC (HYCOM) data...")
    readers_espc = []
    espc_ds = None
    espc_raw_file = None
    espc_extended_file = None

    if not ENABLE_ESPC:
        print("  ESPC model disabled (ENABLE_ESPC=False)")
    else:
        espc_forecast_end = start_time + timedelta(days=7)
        try:
            # Check for existing raw files matching this model/time pattern
            existing_espc_raw = [f for f in os.listdir(data_dir) if f.startswith("espc_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

            if existing_espc_raw:
                # Use the first matching raw file found
                espc_raw_file = os.path.join(data_dir, existing_espc_raw[0])
                print(f"  ESPC raw data file exists: {espc_raw_file}")
                print("  Loading from disk...")
                with xr.open_dataset(espc_raw_file) as ds:
                    espc_ds = ds.load()
                # HYCOM FMRC may serve water_v on 'time1' instead of 'time' (server-side quirk).
                # Prefer 'time' if all variables already use it; only normalize if 'time1' is
                # actually present on a variable's dimensions.
                if any('time1' in espc_ds[v].dims for v in espc_ds.data_vars):
                    u_times = espc_ds['time'].values
                    for var in list(espc_ds.data_vars):
                        if 'time1' in espc_ds[var].dims:
                            aligned = espc_ds[var].sel(time1=u_times, method='nearest')
                            aligned = aligned.assign_coords(time1=u_times).rename({'time1': 'time'})
                            espc_ds[var] = aligned
                    if 'time1' in espc_ds.dims:
                        espc_ds = espc_ds.drop_dims('time1')
            else:
                hycom_url = 'https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd'
                print("  Downloading ESPC data...")

                # ESPC uses 0-360 longitude.
                # We must query using 0-360, but then CONVERT to -180/180 so OpenDrift understands it.
                espc_lon_min = MAP_EXTENT[0] + 360 if MAP_EXTENT[0] < 0 else MAP_EXTENT[0]
                espc_lon_max = MAP_EXTENT[1] + 360 if MAP_EXTENT[1] < 0 else MAP_EXTENT[1]

                # Handle crossing 0/360 boundary if necessary (simple slice usually works for Atlantic if contiguous)
                # Also select time1 to prevent water_v from coming in unsubsetted
                _raw = xr.open_dataset(hycom_url, drop_variables=['tau'])
                _sel_kwargs = dict(
                    time=slice(start_time, espc_forecast_end),
                    lat=slice(MAP_EXTENT[2] - BUFFER_DEG, MAP_EXTENT[3] + BUFFER_DEG),
                    lon=slice(espc_lon_min - BUFFER_DEG, espc_lon_max + BUFFER_DEG),
                    depth=0,
                )
                if 'time1' in _raw.dims:
                    _sel_kwargs['time1'] = slice(start_time, espc_forecast_end)
                espc_ds = _raw.sel(**_sel_kwargs)
                del _raw

                # Check if we got any time steps
                if espc_ds.sizes.get('time', 0) == 0:
                    raise ValueError(
                        f"No ESPC data available for time range {start_time} to {espc_forecast_end}. "
                        "The forecast server may not have historical data for this period."
                    )

                print(f"  Downloaded {espc_ds.sizes['time']} time steps")
                print("  Loading ESPC into memory...")
                espc_ds = espc_ds.load()

                # HYCOM FMRC may serve water_v on 'time1' instead of 'time' (server-side quirk).
                # Prefer 'time' if all variables already use it; only normalize if 'time1' is
                # actually present on a variable's dimensions.
                if any('time1' in espc_ds[v].dims for v in espc_ds.data_vars):
                    u_times = espc_ds['time'].values
                    for var in list(espc_ds.data_vars):
                        if 'time1' in espc_ds[var].dims:
                            aligned = espc_ds[var].sel(time1=u_times, method='nearest')
                            aligned = aligned.assign_coords(time1=u_times).rename({'time1': 'time'})
                            espc_ds[var] = aligned
                    if 'time1' in espc_ds.dims:
                        espc_ds = espc_ds.drop_dims('time1')

                # --- CRITICAL FIX: Convert Longitude from 0-360 to -180-180 ---
                print("  Converting ESPC longitude from 0-360 to -180/180...")
                espc_ds.coords['lon'] = (espc_ds.coords['lon'] + 180) % 360 - 180
                espc_ds = espc_ds.sortby('lon')
                # --------------------------------------------------------------

                # Generate filename based on actual init and end times
                espc_raw_filename = get_forecast_filename("espc", espc_ds, "raw")
                espc_raw_file = os.path.join(data_dir, espc_raw_filename)
                espc_ds.to_netcdf(espc_raw_file)
                print(f"  Saved raw ESPC data to: {espc_raw_file}")

            # Check for existing extended file
            existing_espc_ext = [f for f in os.listdir(data_dir) if f.startswith("espc_") and f.endswith("_extended.nc")] if os.path.exists(data_dir) else []
            if existing_espc_ext:
                espc_extended_file = os.path.join(data_dir, existing_espc_ext[0])
                print(f"  ESPC extended file already exists: {espc_extended_file}")
            else:
                # Generate extended filename based on raw data times
                espc_extended_filename = get_forecast_filename("espc", espc_ds, "extended")
                espc_extended_file = os.path.join(data_dir, espc_extended_filename)

                # Extend forecast to 120 days with 6-hour time steps
                extend_forecast_to_persistence(
                    espc_ds, "ESPC", espc_extended_file,
                    target_days=max_projection_days, time_step_hours=3
                )

            # Create reader from extended file
            r_extended = reader_netCDF_CF_generic.Reader(espc_extended_file)
            readers_espc.append(r_extended)

            print("  ESPC Reader ready (extended forecast)")

        except Exception as e:
            print(f"  ERROR loading ESPC: {e}")
            import traceback
            traceback.print_exc()

    # [3/5] Load CMEMS data
    print("\n[3/5] Loading CMEMS data...")
    readers_cmems = []
    cmems_ds = None
    cmems_raw_file = None
    cmems_extended_file = None

    if not ENABLE_CMEMS:
        print("  CMEMS model disabled (ENABLE_CMEMS=False)")
    else:
        cmems_forecast_end = start_time + timedelta(days=10)
        try:
            # Check for existing raw files matching this model/time pattern
            existing_cmems_raw = [f for f in os.listdir(data_dir) if f.startswith("cmems_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

            if existing_cmems_raw:
                # Use the first matching raw file found
                cmems_raw_file = os.path.join(data_dir, existing_cmems_raw[0])
                print(f"  CMEMS raw data file exists: {cmems_raw_file}")
                print("  Loading from disk...")
                with xr.open_dataset(cmems_raw_file) as ds:
                    cmems_ds = ds.load()
            else:
                # Download to a temporary filename first
                temp_cmems_file = os.path.join(data_dir, "cmems_temp_download.nc")
                print("  Downloading CMEMS subset...")

                # --- FIX: USE DYNAMIC EXTENT (Like Tim's script) ---
                # Center the download on the glider, not the fixed map
                # This guarantees the glider is inside the data domain
                dl_lon_min = glider_info['lon'] - BUFFER_DEG
                dl_lon_max = glider_info['lon'] + BUFFER_DEG
                dl_lat_min = glider_info['lat'] - BUFFER_DEG
                dl_lat_max = glider_info['lat'] + BUFFER_DEG
                # ---------------------------------------------------

                cm.subset(
                    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
                    variables=["uo", "vo"],
                    minimum_longitude=dl_lon_min,
                    maximum_longitude=dl_lon_max,
                    minimum_latitude=dl_lat_min,
                    maximum_latitude=dl_lat_max,
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=cmems_forecast_end.strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_depth=0,
                    maximum_depth=1,
                    output_filename="cmems_temp_download.nc",
                    output_directory=data_dir,
                    force_download=True,  # Added to ensure fresh data
                    username="maristizabalvar",
                    password="MariaCMEMS2018"
                )

                # Process CMEMS data
                print("  Processing CMEMS data...")
                with xr.open_dataset(temp_cmems_file) as cmems_raw:
                    cmems_ds = cmems_raw
                    if 'depth' in cmems_ds.coords:
                        cmems_ds = cmems_ds.sel(depth=0, method='nearest')
                    # Load into memory
                    cmems_ds = cmems_ds.load()

                # Generate filename based on actual init and end times
                cmems_raw_filename = get_forecast_filename("cmems", cmems_ds, "raw")
                cmems_raw_file = os.path.join(data_dir, cmems_raw_filename)
                cmems_ds.to_netcdf(cmems_raw_file)
                print(f"  Saved raw CMEMS data to: {cmems_raw_file}")

                # Remove temporary file
                if os.path.exists(temp_cmems_file):
                    os.remove(temp_cmems_file)

            # Check for existing extended file
            existing_cmems_ext = [f for f in os.listdir(data_dir) if f.startswith("cmems_") and f.endswith("_extended.nc")] if os.path.exists(data_dir) else []

            if existing_cmems_ext:
                cmems_extended_file = os.path.join(data_dir, existing_cmems_ext[0])
                print(f"  CMEMS extended file already exists: {cmems_extended_file}")
            else:
                # Generate extended filename based on raw data times
                cmems_extended_filename = get_forecast_filename("cmems", cmems_ds, "extended")
                cmems_extended_file = os.path.join(data_dir, cmems_extended_filename)

                # Extend forecast to 120 days with 6-hour time steps
                extend_forecast_to_persistence(
                    cmems_ds, "CMEMS", cmems_extended_file,
                    target_days=max_projection_days, time_step_hours=6
                )

            # Create reader from extended file
            r_extended = reader_netCDF_CF_generic.Reader(cmems_extended_file)
            readers_cmems.append(r_extended)

            print("  CMEMS Reader ready (extended forecast)")

        except Exception as e:
            print(f"  ERROR loading CMEMS: {e}")
            import traceback
            traceback.print_exc()

    # [3.5/5] Load Lusitania data
    print("\n[3.5/5] Loading Lusitania data...")
    readers_lusitania = []
    lusitania_ds = None
    lusitania_raw_file = None
    lusitania_extended_file = None

    if not ENABLE_LUSITANIA:
        print("  Lusitania model disabled (ENABLE_LUSITANIA=False)")
    else:
        try:
            # Check for existing raw files matching this model/time pattern
            existing_lusitania_raw = [f for f in os.listdir(data_dir) if f.startswith("lusitania_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

            if existing_lusitania_raw:
                # Use the first matching raw file found
                lusitania_raw_file = os.path.join(data_dir, existing_lusitania_raw[0])
                print(f"  Lusitania raw data file exists: {lusitania_raw_file}")
                print("  Loading from disk...")
                with xr.open_dataset(lusitania_raw_file) as ds:
                    lusitania_ds = ds.load()

                # Show time range of cached data
                if 'time' in lusitania_ds.dims:
                    cached_times = pd.to_datetime(lusitania_ds['time'].values)
                    print(f"  Cached data time range: {cached_times[0]} to {cached_times[-1]} ({len(cached_times)} time steps)")
                    days_coverage = (cached_times[-1] - cached_times[0]).total_seconds() / 86400
                    print(f"  Coverage: {days_coverage:.1f} days")
                    if days_coverage < 5:
                        print(f"  WARNING: Cached file has limited coverage. Consider deleting {lusitania_raw_file}")
                        print(f"           and the corresponding *_extended.nc file to re-download with multi-file loading.")
            else:
                print("  Downloading Lusitania data from THREDDS...")
                print("  Loading all available files from start time onwards...")

                # Load all Lusitania files from start_time onwards
                lusitania_full = lusitania_uv_multi(start_time=start_time, rename=True)

                if lusitania_full is None:
                    raise ValueError("Could not load Lusitania data from THREDDS server")

                # Subset to the region of interest centered on glider
                dl_lon_min = glider_info['lon'] - BUFFER_DEG
                dl_lon_max = glider_info['lon'] + BUFFER_DEG
                dl_lat_min = glider_info['lat'] - BUFFER_DEG
                dl_lat_max = glider_info['lat'] + BUFFER_DEG

                # Get coordinate names
                lon_name = 'lon' if 'lon' in lusitania_full.coords else 'longitude'
                lat_name = 'lat' if 'lat' in lusitania_full.coords else 'latitude'

                # Check coordinate ordering for proper slicing
                lons = lusitania_full[lon_name].values
                lats = lusitania_full[lat_name].values

                # Handle longitude slicing (check if ascending or descending)
                if lons[0] > lons[-1]:  # Descending
                    lon_slice = slice(dl_lon_max, dl_lon_min)
                else:  # Ascending
                    lon_slice = slice(dl_lon_min, dl_lon_max)

                # Handle latitude slicing (check if ascending or descending)
                if lats[0] > lats[-1]:  # Descending (common in ocean models: 90 to -90)
                    lat_slice = slice(dl_lat_max, dl_lat_min)
                else:  # Ascending
                    lat_slice = slice(dl_lat_min, dl_lat_max)

                print(f"  Lusitania spatial extent: lon=[{lons.min():.2f}, {lons.max():.2f}], lat=[{lats.min():.2f}, {lats.max():.2f}]")
                print(f"  Subsetting to: lon=[{dl_lon_min:.2f}, {dl_lon_max:.2f}], lat=[{dl_lat_min:.2f}, {dl_lat_max:.2f}]")

                # Subset spatially
                lusitania_ds = lusitania_full.sel(
                    {lon_name: lon_slice,
                     lat_name: lat_slice}
                )

                # Select surface depth if depth dimension exists
                if 'depth' in lusitania_ds.coords:
                    lusitania_ds = lusitania_ds.sel(depth=0, method='nearest')

                # Load into memory
                print("  Loading Lusitania into memory...")
                lusitania_ds = lusitania_ds.load()

                # Show comprehensive time coverage info
                if 'time' in lusitania_ds.dims:
                    lus_times = pd.to_datetime(lusitania_ds['time'].values)
                    days_coverage = (lus_times[-1] - lus_times[0]).total_seconds() / 86400
                    print(f"  Lusitania loaded: {len(lus_times)} time steps")
                    print(f"  Time range: {lus_times[0]} to {lus_times[-1]}")
                    print(f"  Coverage: {days_coverage:.1f} days")
                else:
                    print(f"  Downloaded {lusitania_ds.sizes.get('time', 1)} time steps (no time dimension found)")

                # Generate filename based on actual init and end times
                lusitania_raw_filename = get_forecast_filename("lusitania", lusitania_ds, "raw")
                lusitania_raw_file = os.path.join(data_dir, lusitania_raw_filename)
                lusitania_ds.to_netcdf(lusitania_raw_file)
                print(f"  Saved raw Lusitania data to: {lusitania_raw_file}")

            # Check for existing extended file
            existing_lusitania_ext = [f for f in os.listdir(data_dir) if f.startswith("lusitania_") and f.endswith("_extended.nc")] if os.path.exists(data_dir) else []

            if existing_lusitania_ext:
                lusitania_extended_file = os.path.join(data_dir, existing_lusitania_ext[0])
                print(f"  Lusitania extended file already exists: {lusitania_extended_file}")
            else:
                # Generate extended filename based on raw data times
                lusitania_extended_filename = get_forecast_filename("lusitania", lusitania_ds, "extended")
                lusitania_extended_file = os.path.join(data_dir, lusitania_extended_filename)

                # Extend forecast to max days with 6-hour time steps
                extend_forecast_to_persistence(
                    lusitania_ds, "Lusitania", lusitania_extended_file,
                    target_days=max_projection_days, time_step_hours=6
                )

            # Create reader from extended file
            r_extended = reader_netCDF_CF_generic.Reader(lusitania_extended_file)
            readers_lusitania.append(r_extended)

            print("  Lusitania Reader ready (extended forecast)")

        except Exception as e:
            print(f"  ERROR loading Lusitania: {e}")
            import traceback
            traceback.print_exc()

    # --- Model Readiness Summary ---
    # Printed once after all data loading so it's easy to spot which model
    # failed before any projections run.
    print("\n" + "-" * 60)
    print("MODEL READINESS SUMMARY")
    print("-" * 60)
    _readiness_rows = [
        ("ESPC",      ENABLE_ESPC,      espc_ds,      readers_espc),
        ("CMEMS",     ENABLE_CMEMS,     cmems_ds,     readers_cmems),
        ("Lusitania", ENABLE_LUSITANIA, lusitania_ds, readers_lusitania),
    ]
    _any_not_ready = False
    for _name, _enabled, _ds, _readers in _readiness_rows:
        if not _enabled:
            print(f"  {_name:<12} DISABLED")
            continue
        if _readers:
            if _ds is not None and 'time' in _ds.dims:
                _times = pd.to_datetime(_ds['time'].values)
                _days = (_times[-1] - _times[0]).total_seconds() / 86400
                print(f"  {_name:<12} READY   | {len(_times)} steps, {_days:.1f} days "
                      f"({_times[0].date()} to {_times[-1].date()})")
            else:
                print(f"  {_name:<12} READY   | (pre-existing extended file)")
        else:
            _any_not_ready = True
            print(f"  {_name:<12} NOT READY --> data load failed; will be SKIPPED in all plots")
    if _any_not_ready:
        print("\n  WARNING: One or more enabled models failed to load.")
        print("  Check the error messages above for details.")
    print("-" * 60)

    # Prepare quiver data and persistence times
    quiver_data = {}
    persistence_times = {}

    if PLOT_CURRENTS:
        print("\n  Preparing quiver data and persistence times...")
        if espc_ds is not None:
            quiver_data['ESPC'] = espc_ds
            # Persistence starts after the last original forecast time
            espc_times = pd.to_datetime(espc_ds['time'].values)
            persistence_times['ESPC'] = espc_times[-1].to_pydatetime()
            print(f"    ESPC persistence starts: {persistence_times['ESPC']}")

        if cmems_ds is not None:
            quiver_data['CMEMS'] = cmems_ds
            # Persistence starts after the last original forecast time
            cmems_times = pd.to_datetime(cmems_ds['time'].values)
            persistence_times['CMEMS'] = cmems_times[-1].to_pydatetime()
            print(f"    CMEMS persistence starts: {persistence_times['CMEMS']}")

        if lusitania_ds is not None:
            quiver_data['Lusitania'] = lusitania_ds
            # Persistence starts after the last original forecast time
            lusitania_times = pd.to_datetime(lusitania_ds['time'].values)
            persistence_times['Lusitania'] = lusitania_times[-1].to_pydatetime()
            print(f"    Lusitania persistence starts: {persistence_times['Lusitania']}")

    # [4/5] Run projections
    print("\n[4/5] Running projections...")
    all_results = {}

    for proj_days in PROJECTION_DAYS:
        print(f"\n{'='*60}")
        print(f"PROJECTION: {proj_days} DAYS")
        print('='*60)
        all_results[proj_days] = {}

        for diff_val in DIFFUSIVITY_VALUES:
            print(f"\n  Diffusivity: {diff_val} m²/s")
            run_folder = os.path.join(projections_dir, f"{proj_days}day_diff{diff_val}")
            os.makedirs(run_folder, exist_ok=True)

            model_outputs = {}

            # Run ESPC (Pass list of readers)
            if readers_espc:
                print(f"    Running ESPC...")
                try:
                    output_file, _, _ = run_extended_forecast(
                        glider_info, NUM_PARTICLES, proj_days,
                        run_folder, "ESPC", readers_espc,
                        horizontal_diffusivity=diff_val
                    )
                    model_outputs['ESPC'] = xr.open_dataset(output_file)
                    print(f"    ESPC complete: {output_file}")
                except Exception as e:
                    print(f"    ESPC ERROR: {e}")

            # Run CMEMS (Pass list of readers)
            if readers_cmems:
                print(f"    Running CMEMS...")
                try:
                    output_file, _, _ = run_extended_forecast(
                        glider_info, NUM_PARTICLES, proj_days,
                        run_folder, "CMEMS", readers_cmems,
                        horizontal_diffusivity=diff_val
                    )
                    model_outputs['CMEMS'] = xr.open_dataset(output_file)
                    print(f"    CMEMS complete: {output_file}")
                except Exception as e:
                    print(f"    CMEMS ERROR: {e}")

            # Run Lusitania (Pass list of readers)
            if readers_lusitania:
                print(f"    Running Lusitania...")
                try:
                    output_file, _, _ = run_extended_forecast(
                        glider_info, NUM_PARTICLES, proj_days,
                        run_folder, "Lusitania", readers_lusitania,
                        horizontal_diffusivity=diff_val
                    )
                    model_outputs['Lusitania'] = xr.open_dataset(output_file)
                    print(f"    Lusitania complete: {output_file}")
                except Exception as e:
                    print(f"    Lusitania ERROR: {e}")

            # --- Per-run model output diagnostic ---
            # Compare what was enabled/readable against what actually landed in model_outputs.
            _expected = []
            if ENABLE_ESPC:       _expected.append(("ESPC",      readers_espc))
            if ENABLE_CMEMS:      _expected.append(("CMEMS",     readers_cmems))
            if ENABLE_LUSITANIA:  _expected.append(("Lusitania", readers_lusitania))

            _missing = [
                (n, "data load failed (no reader created)" if not r
                    else "OpenDrift run failed (see error above)")
                for n, r in _expected
                if n not in model_outputs
            ]
            if _missing:
                print(f"\n  *** MISSING MODELS for {proj_days}-day / diff={diff_val} ***")
                for _n, _reason in _missing:
                    print(f"    {_n}: {_reason}")
                print("  These models will NOT appear in the map or animation.")
            else:
                _present = list(model_outputs.keys())
                print(f"\n  All expected models present: {_present}")

            all_results[proj_days][diff_val] = model_outputs

            # Generate individual map/animation
            if model_outputs:
                map_path = create_single_run_map(model_outputs, glider_info, run_folder, proj_days, diff_val, extent)

                # Create animations - either 1 (no quivers) or 3 (no quivers, CMEMS quivers, ESPC quivers)
                animation_paths = []

                # Always create the base animation without quivers
                animation_path = create_animation(
                    model_outputs, glider_info, run_folder, proj_days, diff_val, extent,
                    quiver_model=None, quiver_data=quiver_data, persistence_times=persistence_times
                )
                animation_paths.append(animation_path)

                # If PLOT_CURRENTS is enabled, create additional animations with quivers
                if PLOT_CURRENTS:
                    if 'CMEMS' in quiver_data:
                        cmems_anim_path = create_animation(
                            model_outputs, glider_info, run_folder, proj_days, diff_val, extent,
                            quiver_model='CMEMS', quiver_data=quiver_data, persistence_times=persistence_times
                        )
                        animation_paths.append(cmems_anim_path)

                    if 'ESPC' in quiver_data:
                        espc_anim_path = create_animation(
                            model_outputs, glider_info, run_folder, proj_days, diff_val, extent,
                            quiver_model='ESPC', quiver_data=quiver_data, persistence_times=persistence_times
                        )
                        animation_paths.append(espc_anim_path)

                    if 'Lusitania' in quiver_data:
                        lusitania_anim_path = create_animation(
                            model_outputs, glider_info, run_folder, proj_days, diff_val, extent,
                            quiver_model='Lusitania', quiver_data=quiver_data, persistence_times=persistence_times
                        )
                        animation_paths.append(lusitania_anim_path)

                # Copy 90-day outputs to latest files
                if proj_days == 90:
                    print(f"    Copying 90-day outputs to latest files...")
                    # Use same extension as the generated animation (.mp4 or .gif)
                    anim_ext = os.path.splitext(animation_path)[1]
                    latest_animation = os.path.join(LATEST_DIR, f"90day_drift_latest{anim_ext}")
                    latest_map = os.path.join(LATEST_DIR, "90day_drift_latest_map.png")
                    shutil.copy2(animation_path, latest_animation)
                    shutil.copy2(map_path, latest_map)
                    archive_90day_outputs(map_path, animation_path)
                    print(f"    Copied to: {latest_animation}")
                    print(f"    Copied to: {latest_map}")

    # [5/5] Create comparison plots
    print("\n[5/5] Creating comparison plots...")
    for proj_days in PROJECTION_DAYS:
        if all_results[proj_days]:
            create_comparison_plot(all_results[proj_days], glider_info, figures_dir, proj_days, extent)

    # Cleanup
    for proj_days in PROJECTION_DAYS:
        for diff_val in DIFFUSIVITY_VALUES:
            for ds in all_results[proj_days].get(diff_val, {}).values():
                ds.close()

    print("\n" + "=" * 70)
    print("EXTENDED PROJECTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Log saved to:          {log_path}")

    # Restore stdout and close the log file
    sys.stdout = _tee._stream
    logging.getLogger().removeHandler(_file_handler)
    _file_handler.close()
    _tee.close()


if __name__ == "__main__":
    main()
