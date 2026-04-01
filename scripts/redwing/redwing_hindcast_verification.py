"""
Hindcast Verification Script: Redwing Glider
---------------------------------------------
Runs a drifter simulation starting from a past date and overlays the
actual observed glider track for forecast verification.

Method:
1. Fetch all glider surfacings and find the position at HINDCAST_START_TIME.
2. Download model forecast data (ESPC ~7 days, CMEMS ~10 days, Lusitania ~3 days)
   valid for that historical period.
3. Run OpenDrift particle simulation forward from that position/time.
4. Fetch the actual glider track from HINDCAST_START_TIME to present.
5. Plot simulated drifter tracks and observed glider track together.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re
import logging
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import shutil
import requests
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import copernicusmarine as cm
import cartopy.mpl.ticker as cticker
import cmocean.cm as cmo
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import platform

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

# =============================================================================
# CONFIGURATION
# =============================================================================

BUFFER_DEG = 30.0  # Spatial buffer in degrees

# Horizontal diffusivity values to test (m^2/s)
DIFFUSIVITY_VALUES = [100]

# Number of particles
NUM_PARTICLES = 20

# Model enable/disable flags
ENABLE_ESPC = True
ENABLE_CMEMS = True
ENABLE_LUSITANIA = True

# Cache control - set to False to force fresh downloads instead of using cached data
USE_CACHED_DATA = True

# Current visualization options
PLOT_CURRENTS = True
CURRENT_STYLE = 'quiver'  # 'quiver' or 'streamplot'
QUIVER_SUBSAMPLE_CMEMS = 6
QUIVER_SUBSAMPLE_ESPC = 8
QUIVER_SUBSAMPLE_LUSITANIA = 6
STREAMPLOT_DENSITY = 3

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
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "redwing_hindcast_verification")
    FORECAST_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "redwing")
    LATEST_DIR = OUTPUT_ROOT
else:  # Linux server
    OUTPUT_ROOT = "/www/web/rucool/media/sentinel/drift/redwing_hindcast_verification"
    FORECAST_OUTPUT_ROOT = "/www/web/rucool/media/sentinel/drift/redwing"
    LATEST_DIR = "/www/web/rucool/media/sentinel"

# =============================================================================
# GLIDER DEPLOYMENT
# =============================================================================
GLIDER_DEPLOYMENT_ID = "redwing-20251011T1511"

# =============================================================================
# HINDCAST START TIME
# =============================================================================
# Automatically set to 8 days before current UTC time, rounded to the nearest
# hour. This allows the script to run on a daily cron job without manual edits.
HINDCAST_LOOKBACK_DAYS = 8
_now_utc = datetime.utcnow()
_start_raw = _now_utc - timedelta(days=HINDCAST_LOOKBACK_DAYS)
# Round to nearest hour
if _start_raw.minute >= 30:
    _start_rounded = _start_raw.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
else:
    _start_rounded = _start_raw.replace(minute=0, second=0, microsecond=0)
HINDCAST_START_TIME = _start_rounded

# Map extent [lon_min, lon_max, lat_min, lat_max]
MAP_EXTENT = [-40.25, -9.75, 19.75, 40.25]

# =============================================================================
# LUSITANIA CONFIGURATION
# =============================================================================
LUSITANIA_CONFIG = {
    'base_url': 'https://thredds.atlanticsense.com/thredds',
    'catalog_path': '/catalog/atlDatasets/Lusitania/catalog.html',
    'opendap_base': '/dodsC/atlDatasets/Lusitania/',
    'file_pattern': r'(\d{10})\.nc',  # YYYYMMDDHH.nc
    'timeout': 30,
}


# =============================================================================
# GLIDER TRACK FUNCTIONS
# =============================================================================

def dm_to_dd(dm_coord):
    """Convert degrees-minutes to decimal degrees."""
    sign = 1 if dm_coord > 0 else -1
    dm_coord = abs(dm_coord)
    degrees = int(dm_coord // 100)
    minutes = dm_coord % 100
    dd_coord = degrees + (minutes / 60)
    return sign * dd_coord


def find_cached_forecast_data(model_prefix, sim_start, suffix="raw"):
    """
    Search existing forecast run directories for cached model data files
    whose time range covers the simulation start time.

    Looks in FORECAST_OUTPUT_ROOT/YYYY_MM/*/data/ for files like
    espc_20260125T15_to_20260201T12_raw.nc

    Parameters
    ----------
    model_prefix : str
        Model name prefix (e.g., "espc", "cmems", "lusitania")
    sim_start : datetime
        The simulation start time - the cached file must contain this time.
    suffix : str
        File suffix to look for ("raw" or "extended")

    Returns
    -------
    str or None
        Path to a matching cached file, or None if not found.
    """
    if not os.path.isdir(FORECAST_OUTPUT_ROOT):
        return None

    prefix = model_prefix.lower()
    best_match = None
    best_distance = None

    # Scan all month folders and run directories
    for month_folder in sorted(os.listdir(FORECAST_OUTPUT_ROOT), reverse=True):
        month_path = os.path.join(FORECAST_OUTPUT_ROOT, month_folder)
        if not os.path.isdir(month_path):
            continue

        for run_folder in sorted(os.listdir(month_path), reverse=True):
            data_dir = os.path.join(month_path, run_folder, "data")
            if not os.path.isdir(data_dir):
                continue

            for fname in os.listdir(data_dir):
                if not fname.startswith(f"{prefix}_") or not fname.endswith(f"_{suffix}.nc"):
                    continue

                # Parse time range from filename: model_YYYYMMDDTHH_to_YYYYMMDDTHH_suffix.nc
                parts = fname.replace(f"_{suffix}.nc", "").split("_to_")
                if len(parts) != 2:
                    continue

                try:
                    file_start_str = parts[0].replace(f"{prefix}_", "")
                    file_start = datetime.strptime(file_start_str, "%Y%m%dT%H")
                    file_end = datetime.strptime(parts[1], "%Y%m%dT%H")
                except ValueError:
                    continue

                # Check if this file's time range covers our simulation start
                if file_start <= sim_start <= file_end:
                    distance = abs((file_start - sim_start).total_seconds())
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_match = os.path.join(data_dir, fname)

    if best_match:
        logger.info(f"Found cached {prefix} data: {best_match}")
    return best_match


def get_all_surfacings(deployment_id):
    """
    Fetch ALL surfacing records for a deployment from the COOL API.

    Returns a DataFrame with time index and latitude/longitude columns,
    sorted by time ascending.
    """
    surfacings_url = f"https://marine.rutgers.edu/cool/data/gliders/api/surfacings/?deployment={deployment_id}"
    response = requests.get(surfacings_url, timeout=60)
    response.raise_for_status()
    data = response.json()

    records = data.get("data", [])
    if not records:
        raise ValueError(f"No surfacing data found for deployment '{deployment_id}'")

    rows = []
    for rec in records:
        epoch = rec.get("gps_timestamp_epoch")
        if epoch is None:
            continue

        dt_utc = datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None)

        if "gps_lat_degrees" in rec and rec["gps_lat_degrees"] is not None:
            lat = float(rec["gps_lat_degrees"])
            lon = float(rec["gps_lon_degrees"])
        elif "gps_lat" in rec and rec["gps_lat"] is not None:
            lat = dm_to_dd(rec["gps_lat"])
            lon = dm_to_dd(rec["gps_lon"])
        else:
            continue

        rows.append({"time": dt_utc, "latitude": lat, "longitude": lon})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df.sort_values("time").set_index("time")
    logger.info(f"Fetched {len(df)} surfacings for {deployment_id}")
    logger.info(f"  Time range: {df.index[0]} to {df.index[-1]}")
    return df


def get_glider_position_at_time(surfacings_df, target_time):
    """
    Find the surfacing closest to target_time and return a glider_info dict
    compatible with the simulation functions.
    """
    if isinstance(target_time, str):
        target_time = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S")

    time_diffs = abs(surfacings_df.index - target_time)
    closest_idx = time_diffs.argmin()
    closest = surfacings_df.iloc[closest_idx]

    return {
        "lat": closest["latitude"],
        "lon": closest["longitude"],
        "last_surfacing_time": surfacings_df.index[closest_idx].strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_glider_track(surfacings_df, start_time, end_time=None):
    """
    Extract the glider track between start_time and end_time.

    Returns a DataFrame with time index and latitude/longitude columns.
    """
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    if end_time is not None and isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    mask = surfacings_df.index >= start_time
    if end_time is not None:
        mask = mask & (surfacings_df.index <= end_time)

    track = surfacings_df[mask]
    logger.info(f"Glider track: {len(track)} points from {track.index[0]} to {track.index[-1]}")
    return track


# =============================================================================
# LUSITANIA MODEL FUNCTIONS (reused from original)
# =============================================================================

def _parse_lusitania_catalog(timeout=30):
    """Parse the Lusitania THREDDS catalog to get available files."""
    catalog_url = f"{LUSITANIA_CONFIG['base_url']}{LUSITANIA_CONFIG['catalog_path']}"
    try:
        response = requests.get(catalog_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to fetch Lusitania catalog: {exc}")
        return []

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
                except ValueError:
                    continue
    else:
        for match in pattern.finditer(response.text):
            date_str = match.group(1)
            filename = f"{date_str}.nc"
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d%H')
                if (file_date, filename) not in available_files:
                    available_files.append((file_date, filename))
            except ValueError:
                continue

    available_files.sort(key=lambda x: x[0])
    logger.info(f"Found {len(available_files)} Lusitania files in catalog")
    return available_files


@lru_cache(maxsize=1)
def _get_lusitania_catalog():
    return _parse_lusitania_catalog(timeout=LUSITANIA_CONFIG['timeout'])


def _find_lusitania_file(target_time, available_files=None, method='nearest'):
    """Find the Lusitania file nearest to the target time."""
    if available_files is None:
        available_files = _get_lusitania_catalog()
    if not available_files:
        return None

    if target_time.tzinfo is not None:
        target_time = target_time.replace(tzinfo=None)

    if method == 'before':
        candidates = [f for f in available_files if f[0] <= target_time]
        return candidates[-1] if candidates else None
    elif method == 'after':
        candidates = [f for f in available_files if f[0] >= target_time]
        return candidates[0] if candidates else None
    else:
        best_match = None
        min_diff = None
        for file_date, filename in available_files:
            diff = abs((file_date - target_time).total_seconds())
            if min_diff is None or diff < min_diff:
                min_diff = diff
                best_match = (file_date, filename)
        return best_match


def lusitania_uv_multi(start_time, rename=True):
    """Load all available Lusitania files from start_time onwards and concatenate."""
    available = _get_lusitania_catalog()
    if not available:
        logger.error("No Lusitania files available")
        return None

    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    start_result = _find_lusitania_file(start_time, available_files=available, method='before')
    if start_result is None:
        start_result = available[0] if available else None
    if start_result is None:
        return None

    start_file_date = start_result[0]
    files_to_load = [(fd, fn) for fd, fn in available if fd >= start_file_date]
    if not files_to_load:
        return None

    logger.info(f"Loading {len(files_to_load)} Lusitania files")

    datasets = []
    for file_date, filename in files_to_load:
        opendap_url = f"{LUSITANIA_CONFIG['base_url']}{LUSITANIA_CONFIG['opendap_base']}{filename}"
        try:
            ds = xr.open_dataset(opendap_url)
            if rename:
                rename_map = {}
                for u_name in ['uo', 'u', 'water_u']:
                    if u_name in ds.data_vars and u_name != 'water_u':
                        rename_map[u_name] = 'water_u'
                        break
                for v_name in ['vo', 'v', 'water_v']:
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
        return None

    combined = xr.concat(datasets, dim='time').sortby('time')
    _, unique_indices = np.unique(combined['time'].values, return_index=True)
    combined = combined.isel(time=sorted(unique_indices))
    combined.attrs['model'] = 'Lusitania'
    return combined


# =============================================================================
# FORECAST EXTENSION & SIMULATION
# =============================================================================

def extend_forecast_to_persistence(ds, model_name, output_file, target_days=10, time_step_hours=6):
    """Extend a forecast dataset to target_days by persisting the last time step."""
    print(f"  {model_name}: Extending forecast to {target_days} days...")

    if 'time' not in ds.dims or ds.sizes['time'] == 0:
        raise ValueError(f"{model_name} dataset has no time dimension or is empty")

    ds = ds.sortby('time')
    original_times = pd.to_datetime(ds['time'].values)
    start_time = original_times[0]
    last_forecast_time = original_times[-1]
    target_end_time = start_time + pd.Timedelta(days=target_days)

    print(f"    Original forecast: {start_time} to {last_forecast_time}")
    print(f"    Target end time: {target_end_time}")

    if last_forecast_time >= target_end_time:
        ds.to_netcdf(output_file)
        return output_file

    persistence_times = pd.date_range(
        start=last_forecast_time + pd.Timedelta(hours=time_step_hours),
        end=target_end_time,
        freq=f'{time_step_hours}h'
    )

    print(f"    Adding {len(persistence_times)} persistence time steps...")

    last_step = ds.isel(time=-1)
    if len(persistence_times) > 0:
        persistence_ds = last_step.expand_dims(time=persistence_times).assign_coords(time=persistence_times)
        extended_ds = xr.concat([ds, persistence_ds], dim='time')
    else:
        extended_ds = ds

    # Restore CF-compliant coordinate attributes
    for coord_name in extended_ds.coords:
        coord_lower = coord_name.lower()
        if coord_lower in ['lon', 'longitude']:
            extended_ds[coord_name].attrs.update({
                'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'
            })
        elif coord_lower in ['lat', 'latitude']:
            extended_ds[coord_name].attrs.update({
                'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'
            })
        elif coord_lower == 'time':
            extended_ds[coord_name].attrs.update({
                'standard_name': 'time', 'axis': 'T'
            })

    if os.path.exists(output_file):
        os.remove(output_file)
    extended_ds.to_netcdf(output_file)
    print(f"    Saved to: {output_file}")
    return output_file


def get_forecast_filename(model_name, ds, suffix="raw"):
    """Generate a filename based on model init and end times."""
    times = pd.to_datetime(ds['time'].values)
    init_time = times[0].strftime("%Y%m%dT%H")
    end_time = times[-1].strftime("%Y%m%dT%H")
    return f"{model_name.lower()}_{init_time}_to_{end_time}_{suffix}.nc"


def run_simulation(glider_info, num_particles, projection_days, output_dir,
                   model_name, readers, horizontal_diffusivity=0):
    """Run OpenDrift forecast from glider position."""
    start_time = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    if start_time.minute > 0 or start_time.second > 0:
        start_time += timedelta(hours=1)
        start_time = start_time.replace(minute=0, second=0, microsecond=0)

    end_time = start_time + timedelta(days=projection_days)

    o = OceanDrift(loglevel=50)
    if horizontal_diffusivity > 0:
        o.set_config('drift:horizontal_diffusivity', horizontal_diffusivity)

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
        f"hindcast_{model_name}_{projection_days}days_diff{horizontal_diffusivity}.nc"
    )

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
    Apply tick and tick-label styling matching redwing_map.py.

    Major and minor intervals are chosen adaptively so that each axis
    gets roughly 4-8 labelled ticks regardless of the extent size.
    Labels use cartopy LongitudeFormatter / LatitudeFormatter with
    bold font.

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

def compute_zoomed_extent(model_outputs, actual_track, glider_info, buffer_deg=1.0):
    """
    Compute a tight bounding box around all trajectory data and the observed track.

    Returns [lon_min, lon_max, lat_min, lat_max] with buffer_deg padding.
    """
    all_lons = [glider_info['lon']]
    all_lats = [glider_info['lat']]

    for ds in model_outputs.values():
        lons = ds['lon'].values
        lats = ds['lat'].values
        all_lons.extend([np.nanmin(lons), np.nanmax(lons)])
        all_lats.extend([np.nanmin(lats), np.nanmax(lats)])

    all_lons.extend(actual_track['longitude'].values.tolist())
    all_lats.extend(actual_track['latitude'].values.tolist())

    return [
        min(all_lons) - buffer_deg,
        max(all_lons) + buffer_deg,
        min(all_lats) - buffer_deg,
        max(all_lats) + buffer_deg,
    ]


def compute_moderate_extent(full_extent, tight_extent, zoom_factor=0.5):
    """
    Compute a moderate extent between full and tight extents.

    Parameters
    ----------
    full_extent : list
        Full map extent [lon_min, lon_max, lat_min, lat_max]
    tight_extent : list
        Tight zoomed extent
    zoom_factor : float
        Interpolation factor between full (0.0) and tight (1.0).
        Default 0.5 gives halfway between.

    Returns
    -------
    list
        Moderate extent [lon_min, lon_max, lat_min, lat_max]
    """
    return [
        full_extent[0] + zoom_factor * (tight_extent[0] - full_extent[0]),
        full_extent[1] + zoom_factor * (tight_extent[1] - full_extent[1]),
        full_extent[2] + zoom_factor * (tight_extent[2] - full_extent[2]),
        full_extent[3] + zoom_factor * (tight_extent[3] - full_extent[3]),
    ]


def create_verification_map(model_outputs, glider_info, actual_track, output_dir,
                            projection_days, diffusivity, extent, persistence_times=None,
                            suffix=""):
    """Create a static map showing simulated drifter tracks and the actual glider track."""
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

    # Minor gridlines (same appearance as major)
    gl_minor = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_minor.xlocator = mticker.FixedLocator(minor_tickx)
    gl_minor.ylocator = mticker.FixedLocator(minor_ticky)

    model_colors = {'ESPC': 'orange', 'CMEMS': 'magenta', 'Lusitania': 'cyan'}

    # Track which models have persistence for legend building
    models_with_persistence = set()

    # Plot simulated tracks — solid line for forecast, dotted line with filled
    # circle markers for persistence
    for model_name, ds in model_outputs.items():
        lons = ds['lon'].values
        lats = ds['lat'].values
        color = model_colors.get(model_name, 'blue')
        sim_times = pd.to_datetime(ds['time'].values)

        # Find persistence split index for this model
        persist_idx = None
        if persistence_times and model_name in persistence_times:
            persist_dt = persistence_times[model_name]
            persist_idx = int(np.abs(sim_times - persist_dt).argmin())

        for i in range(ds.dims['trajectory']):
            if persist_idx is not None and persist_idx < lons.shape[1] - 1:
                models_with_persistence.add(model_name)
                # Forecast portion: solid colored line
                ax.plot(lons[i, :persist_idx+1], lats[i, :persist_idx+1],
                        color=color, alpha=0.5, linewidth=2)
                # Persistence portion: filled circle markers only (no line), every 2nd point
                ax.scatter(lons[i, persist_idx::1], lats[i, persist_idx::1],
                           color=color, s=12, alpha=0.5, zorder=5,
                           edgecolors='black', linewidths=0.3)
            else:
                # No persistence split — all forecast
                ax.plot(lons[i, :], lats[i, :], color=color, alpha=0.5, linewidth=2)
            ax.scatter(lons[i, -1], lats[i, -1], color=color, s=20, zorder=5, alpha=0.5)

    # Plot actual glider track
    ax.plot(actual_track['longitude'].values, actual_track['latitude'].values,
            color='limegreen', linewidth=2.5, zorder=8)
    ax.scatter(actual_track['longitude'].values, actual_track['latitude'].values,
               color='limegreen', s=15, zorder=9, edgecolors='darkgreen', linewidths=0.5)
    # Mark final observed position
    ax.scatter(actual_track['longitude'].iloc[-1], actual_track['latitude'].iloc[-1],
               color='limegreen', s=80, zorder=10, marker='D', edgecolors='black', linewidths=1)

    # Start position
    ax.scatter(glider_info['lon'], glider_info['lat'], color='red', s=150, marker='*', zorder=10)

    # --- Main legend (upper left) with prescribed order ---
    legend_order = [
        ('Observed Track', Line2D([], [], color='limegreen', linewidth=3,
                                  label='Observed Track')),
        ('Start', Line2D([], [], marker='*', color='red', linestyle='None',
                         markersize=14, label='Start')),
    ]
    # Append model entries in CMEMS-first, then ESPC, then Lusitania order
    for mname in ['CMEMS', 'ESPC', 'Lusitania']:
        if mname not in model_outputs:
            continue
        color = model_colors.get(mname, 'blue')
        legend_order.append(
            (mname, Line2D([], [], color=color, linewidth=3,
                           label=mname))
        )
        if mname in models_with_persistence:
            legend_order.append(
                (f'{mname} (persistence)',
                 Line2D([], [], marker='o', color=color, linestyle='None',
                        markersize=4, alpha=0.5, markeredgecolor='black',
                        markeredgewidth=0.3, label=f'{mname} (persistence)'))
            )

    main_handles = [h for _, h in legend_order]
    main_labels = [lbl for lbl, _ in legend_order]
    main_legend = ax.legend(main_handles, main_labels, loc='upper left',
                            fontsize=10)
    ax.add_artist(main_legend)

    # --- Bathymetry legend (lower left) ---
    bathy_handles = [
        Patch(facecolor='lightsteelblue', edgecolor='none',
              label='0\u2013100 m'),
        Patch(facecolor=cfeature.COLORS['water'], edgecolor='none',
              label='100\u20131000 m'),
        Patch(facecolor='cornflowerblue', edgecolor='none',
              label='1000+ m'),
    ]
    ax.legend(handles=bathy_handles, loc='lower left', fontsize=9,
              title='Bathymetry', title_fontsize=10)
    zoom_label = " (Zoomed)" if suffix else ""
    ax.set_title(
        f"Hindcast Verification: {projection_days}-Day Simulation vs Observed{zoom_label}\n"
        f"Start: {glider_info['last_surfacing_time']} UTC | Diffusivity={diffusivity} m\u00b2/s"
    )

    start_dt = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    time_str = start_dt.strftime("%Y%m%dT%H")
    suffix_str = f"_{suffix}" if suffix else ""
    output_path = os.path.join(output_dir, f"hindcast_{time_str}_{projection_days}days_diff{diffusivity}_map{suffix_str}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Map saved: {output_path}")
    return output_path


def create_verification_animation(model_outputs, glider_info, actual_track, output_dir,
                                  projection_days, diffusivity, extent,
                                  quiver_model=None, quiver_data=None, persistence_times=None):
    """
    Create an animation showing simulated drifter trajectories and the actual
    glider track progressively revealed over time.
    """
    first_model = list(model_outputs.keys())[0]
    times = model_outputs[first_model]['time'].values

    total_frames = len(times)
    frame_step = max(1, total_frames // 200)
    frame_indices = list(range(0, total_frames, frame_step))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)

    quiver_label = f"_{quiver_model.lower()}_quivers" if quiver_model else ""
    print(f"    Creating animation{quiver_label} with {len(frame_indices)} frames...")

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

    # Minor gridlines (same appearance as major)
    gl_minor = ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.5)
    gl_minor.xlocator = mticker.FixedLocator(minor_tickx)
    gl_minor.ylocator = mticker.FixedLocator(minor_ticky)

    model_colors = {'ESPC': 'orange', 'CMEMS': 'magenta', 'Lusitania': 'cyan'}

    # Track which models have persistence for legend building
    models_with_persistence = set()
    if persistence_times:
        for mname in model_outputs.keys():
            if mname in persistence_times:
                models_with_persistence.add(mname)

    # Start marker
    ax.scatter(glider_info['lon'], glider_info['lat'], color='red', s=150, marker='*', zorder=10)

    # Simulated track artists
    model_scatters = {}
    model_trails_forecast = {}
    model_trails_persist = {}

    for model_name, ds in model_outputs.items():
        n_traj = ds.dims['trajectory']
        color = model_colors.get(model_name, 'blue')
        model_scatters[model_name] = ax.scatter([], [], color=color, s=40, zorder=5)
        model_trails_forecast[model_name] = [ax.plot([], [], color=color, alpha=0.3, linewidth=3)[0] for _ in range(n_traj)]
        # Persistence: filled circle markers only (no line)
        model_trails_persist[model_name] = [ax.scatter([], [], color=color, s=12, alpha=0.3, zorder=5, edgecolors='black', linewidths=0.3) for _ in range(n_traj)]

    # Actual track artists
    observed_trail, = ax.plot([], [], color='limegreen', linewidth=2.5, zorder=8)
    observed_scatter = ax.scatter([], [], color='limegreen', s=30, zorder=9, edgecolors='darkgreen', linewidths=0.5)

    # Pre-convert actual track times for lookup
    track_times = actual_track.index.values
    track_lons = actual_track['longitude'].values
    track_lats = actual_track['latitude'].values

    # Pre-compute persistence frame indices
    persistence_frame_indices = {}
    if persistence_times:
        for model_name in model_outputs.keys():
            if model_name in persistence_times:
                persist_time = persistence_times[model_name]
                for idx, t in enumerate(times):
                    if pd.to_datetime(t) >= persist_time:
                        persistence_frame_indices[model_name] = idx
                        break
                else:
                    persistence_frame_indices[model_name] = len(times)

    # Current visualization setup
    quiver_plot = None
    streamplot_container = [None]
    contourf_container = [None]
    use_streamplot = CURRENT_STYLE == 'streamplot'
    Q_lon, Q_lat = None, None
    stream_lons, stream_lats = None, None
    contour_lons, contour_lats = None, None

    SPEED_LEVELS = np.linspace(0, 1.5, 16)
    SPEED_CMAP = cmo.speed

    if quiver_model == 'ESPC':
        quiver_subsample = QUIVER_SUBSAMPLE_ESPC
    elif quiver_model == 'Lusitania':
        quiver_subsample = QUIVER_SUBSAMPLE_LUSITANIA
    else:
        quiver_subsample = QUIVER_SUBSAMPLE_CMEMS

    if quiver_model and quiver_data and quiver_model in quiver_data:
        qds = quiver_data[quiver_model]
        lon_name = 'lon' if 'lon' in qds.coords else 'longitude'
        lat_name = 'lat' if 'lat' in qds.coords else 'latitude'

        contour_lons = qds[lon_name].values
        contour_lats = qds[lat_name].values

        if use_streamplot:
            stream_lons = qds[lon_name].values
            stream_lats = qds[lat_name].values
        else:
            q_lons = qds[lon_name].values[::quiver_subsample]
            q_lats = qds[lat_name].values[::quiver_subsample]
            Q_lon, Q_lat = np.meshgrid(q_lons, q_lats)
            quiver_plot = ax.quiver(
                Q_lon, Q_lat, np.zeros_like(Q_lon), np.zeros_like(Q_lat),
                transform=ccrs.PlateCarree(), color='black', alpha=0.6,
                scale=60, width=0.002, zorder=2
            )

    cbar = None
    if quiver_model and quiver_data and quiver_model in quiver_data:
        sm = plt.cm.ScalarMappable(cmap=SPEED_CMAP, norm=Normalize(vmin=0, vmax=1.5))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
        cbar.set_label('Current Speed (m/s)', fontsize=10)

    # --- Main legend (upper left) with prescribed order ---
    legend_order = [
        ('Observed Track', Line2D([], [], color='limegreen', linewidth=3,
                                  label='Observed Track')),
        ('Start', Line2D([], [], marker='*', color='red', linestyle='None',
                         markersize=14, label='Start')),
    ]
    # Append model entries in CMEMS-first, then ESPC, then Lusitania order
    for mname in ['CMEMS', 'ESPC', 'Lusitania']:
        if mname not in model_outputs:
            continue
        color = model_colors.get(mname, 'blue')
        legend_order.append(
            (mname, Line2D([], [], color=color, linewidth=3,
                           label=mname))
        )
        if mname in models_with_persistence:
            legend_order.append(
                (f'{mname} (persistence)',
                 Line2D([], [], marker='o', color=color, linestyle='None',
                        markersize=4, alpha=0.5, markeredgecolor='black',
                        markeredgewidth=0.3, label=f'{mname} (persistence)'))
            )

    main_handles = [h for _, h in legend_order]
    main_labels = [lbl for lbl, _ in legend_order]

    legend_title = f"{quiver_model} Currents" if quiver_model else None
    main_legend = ax.legend(main_handles, main_labels, loc='upper left',
                            fontsize=10, title=legend_title)
    ax.add_artist(main_legend)

    # --- Bathymetry legend (lower left) ---
    bathy_handles = [
        Patch(facecolor='lightsteelblue', edgecolor='none',
              label='0\u2013100 m'),
        Patch(facecolor=cfeature.COLORS['water'], edgecolor='none',
              label='100\u20131000 m'),
        Patch(facecolor='cornflowerblue', edgecolor='none',
              label='1000+ m'),
    ]
    ax.legend(handles=bathy_handles, loc='lower left', fontsize=9,
              title='Bathymetry', title_fontsize=10)

    title = ax.set_title("")

    def init():
        for model_name in model_outputs:
            model_scatters[model_name].set_offsets(np.empty((0, 2)))
            for trail in model_trails_forecast[model_name]:
                trail.set_data([], [])
            for scat in model_trails_persist[model_name]:
                scat.set_offsets(np.empty((0, 2)))
        observed_trail.set_data([], [])
        observed_scatter.set_offsets(np.empty((0, 2)))
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

        persistence_label = ""
        if quiver_model and persistence_times and quiver_model in persistence_times:
            if current_time_dt >= persistence_times[quiver_model]:
                persistence_label = " [PERSISTENCE]"

        title.set_text(
            f"Hindcast Verification (Diff={diffusivity}){quiver_label.replace('_', ' ').title()}\n"
            f"Start: {start_str} UTC | Current: {time_str} UTC{persistence_label} | Days since start: {days_since_init:.1f}"
        )

        artists = []

        # Update simulated tracks
        for model_name, ds in model_outputs.items():
            lons_full = ds['lon'].values[:, :frame+1]
            lats_full = ds['lat'].values[:, :frame+1]
            current_lons = lons_full[:, -1]
            current_lats = lats_full[:, -1]
            model_scatters[model_name].set_offsets(np.c_[current_lons, current_lats])
            artists.append(model_scatters[model_name])

            persist_idx = persistence_frame_indices.get(model_name, frame + 1)
            for i in range(len(model_trails_forecast[model_name])):
                if frame < persist_idx:
                    # Still in forecast — draw everything as solid forecast line
                    model_trails_forecast[model_name][i].set_data(lons_full[i, :], lats_full[i, :])
                    model_trails_persist[model_name][i].set_offsets(np.empty((0, 2)))
                else:
                    # Split into forecast (solid) + persistence (dots only)
                    model_trails_forecast[model_name][i].set_data(lons_full[i, :persist_idx+1], lats_full[i, :persist_idx+1])
                    model_trails_persist[model_name][i].set_offsets(np.c_[lons_full[i, persist_idx::1], lats_full[i, persist_idx::1]])
                artists.append(model_trails_forecast[model_name][i])
                artists.append(model_trails_persist[model_name][i])

        # Update actual glider track (progressive reveal)
        track_mask = track_times <= np.datetime64(current_time_dt)
        visible_lons = track_lons[track_mask]
        visible_lats = track_lats[track_mask]
        if len(visible_lons) > 0:
            observed_trail.set_data(visible_lons, visible_lats)
            observed_scatter.set_offsets(np.c_[visible_lons, visible_lats])
        artists.append(observed_trail)
        artists.append(observed_scatter)

        # Update current visualization
        if quiver_model and quiver_data and quiver_model in quiver_data:
            qds = quiver_data[quiver_model]

            if 'water_u' in qds:
                u_name, v_name = 'water_u', 'water_v'
            elif 'uo' in qds:
                u_name, v_name = 'uo', 'vo'
            else:
                u_name, v_name = 'u', 'v'

            def _get_time_dim(da):
                return 'time' if 'time' in da.dims else 'time1' if 'time1' in da.dims else None

            u_time_dim = _get_time_dim(qds[u_name])
            v_time_dim = _get_time_dim(qds[v_name])
            ref_time_dim = u_time_dim or v_time_dim
            if ref_time_dim is not None:
                q_times = pd.to_datetime(qds[ref_time_dim].values)
                time_idx = np.abs(q_times - current_time_dt).argmin()
                u_full = qds[u_name].isel({u_time_dim: time_idx}).values if u_time_dim else qds[u_name].values
                v_full = qds[v_name].isel({v_time_dim: time_idx}).values if v_time_dim else qds[v_name].values
            else:
                u_full = qds[u_name].values
                v_full = qds[v_name].values

            speed_full = np.sqrt(u_full**2 + v_full**2)

            if contourf_container[0] is not None:
                contourf_container[0].remove()
                contourf_container[0] = None

            if contour_lons is not None and contour_lats is not None:
                contourf_container[0] = ax.contourf(
                    contour_lons, contour_lats, speed_full,
                    levels=SPEED_LEVELS, cmap=SPEED_CMAP,
                    transform=ccrs.PlateCarree(), zorder=1, extend='max'
                )

            if use_streamplot:
                u_data = np.nan_to_num(u_full, nan=0)
                v_data = np.nan_to_num(v_full, nan=0)
            else:
                u_data = np.nan_to_num(u_full[::quiver_subsample, ::quiver_subsample], nan=0)
                v_data = np.nan_to_num(v_full[::quiver_subsample, ::quiver_subsample], nan=0)

            if use_streamplot:
                if streamplot_container[0] is not None:
                    streamplot_container[0].lines.remove()
                    for art in ax.get_children():
                        if isinstance(art, matplotlib.patches.FancyArrowPatch):
                            art.remove()
                streamplot_container[0] = ax.streamplot(
                    stream_lons, stream_lats, u_data, v_data,
                    transform=ccrs.PlateCarree(), color='black',
                    density=STREAMPLOT_DENSITY, linewidth=0.5, arrowsize=0.8, zorder=2
                )
            else:
                magnitude = np.sqrt(u_data**2 + v_data**2)
                magnitude = np.where(magnitude == 0, 1, magnitude)
                u_data = u_data / magnitude
                v_data = v_data / magnitude
                quiver_plot.set_UVC(u_data, v_data)
                artists.append(quiver_plot)

        return artists

    use_blit = not quiver_model
    ani = FuncAnimation(fig, update, init_func=init, frames=len(frame_indices), interval=100, blit=use_blit)

    start_dt = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    time_str = start_dt.strftime("%Y%m%dT%H")
    quiver_suffix = f"_{quiver_model.lower()}_quivers" if quiver_model else ""

    from matplotlib.animation import writers
    if writers.is_available('ffmpeg'):
        output_path = os.path.join(output_dir, f"hindcast_{time_str}_{projection_days}days_diff{diffusivity}{quiver_suffix}_animation.mp4")
        ani.save(output_path, writer='ffmpeg', dpi=100)
    else:
        print("    WARNING: ffmpeg not available, saving as GIF instead")
        output_path = os.path.join(output_dir, f"hindcast_{time_str}_{projection_days}days_diff{diffusivity}{quiver_suffix}_animation.gif")
        ani.save(output_path, writer='pillow', dpi=100)

    plt.close()
    print(f"    Animation saved: {output_path}")
    return output_path


# =============================================================================
# ARCHIVE
# =============================================================================

ARCHIVE_BASE = "/www/web/rucool/media/sentinel/drift/archive"
ARCHIVE_MOVIES = os.path.join(ARCHIVE_BASE, "hindcast_movies")
ARCHIVE_MAPS = os.path.join(ARCHIVE_BASE, "hindcast_maps")


def archive_hindcast_outputs(map_path, animation_path, zoomed_map_path=None):
    """Copies hindcast map(s) and animation to a flat archive structure."""
    try:
        os.makedirs(ARCHIVE_MOVIES, exist_ok=True)
        os.makedirs(ARCHIVE_MAPS, exist_ok=True)

        # Copy map
        shutil.copy2(map_path, os.path.join(ARCHIVE_MAPS, os.path.basename(map_path)))

        # Copy zoomed map (if produced)
        if zoomed_map_path and os.path.exists(zoomed_map_path):
            shutil.copy2(zoomed_map_path, os.path.join(ARCHIVE_MAPS, os.path.basename(zoomed_map_path)))

        # Copy animation
        shutil.copy2(animation_path, os.path.join(ARCHIVE_MOVIES, os.path.basename(animation_path)))

        print(f"    Archived to: {ARCHIVE_BASE}")
    except Exception as e:
        print(f"    Error archiving hindcast files: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REDWING GLIDER HINDCAST VERIFICATION")
    print("=" * 70)

    start_time = HINDCAST_START_TIME
    print(f"Hindcast start time: {start_time}")
    print(f"Diffusivity values: {DIFFUSIVITY_VALUES} m\u00b2/s")
    print(f"Particles: {NUM_PARTICLES}")

    # [1/6] Fetch all glider surfacings
    print("\n[1/6] Fetching glider surfacings...")
    all_surfacings = get_all_surfacings(GLIDER_DEPLOYMENT_ID)

    # Find position at hindcast start
    glider_info = get_glider_position_at_time(all_surfacings, start_time)
    glider_info['name'] = GLIDER_DEPLOYMENT_ID

    print(f"  Start position: {glider_info['lat']:.4f}N, {abs(glider_info['lon']):.4f}W")
    print(f"  Closest surfacing: {glider_info['last_surfacing_time']}")

    # Round start time for simulation
    sim_start = datetime.strptime(glider_info['last_surfacing_time'], "%Y-%m-%d %H:%M:%S")
    if sim_start.minute > 0 or sim_start.second > 0:
        sim_start += timedelta(hours=1)
        sim_start = sim_start.replace(minute=0, second=0, microsecond=0)

    # Determine projection length: ESPC ~7 days, CMEMS ~10 days
    # Use the longest available forecast as projection length
    projection_days = 10 if ENABLE_CMEMS else 7

    sim_end = sim_start + timedelta(days=projection_days)
    print(f"  Simulation period: {sim_start} to {sim_end} ({projection_days} days)")

    # [2/6] Get actual glider track for comparison
    print("\n[2/6] Extracting actual glider track...")
    actual_track = get_glider_track(all_surfacings, start_time, sim_end)
    print(f"  Observed track: {len(actual_track)} surfacing positions")

    extent = MAP_EXTENT

    # Prepare output directory
    time_str = sim_start.strftime("%Y%m%dT%H%M%S")
    run_id = f"hindcast_{time_str}"
    month_folder = sim_start.strftime("%Y_%m")
    output_dir = os.path.join(OUTPUT_ROOT, month_folder, run_id)
    data_dir = os.path.join(output_dir, "data")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # [3/6] Load ESPC (HYCOM) data
    print("\n[3/6] Loading ESPC (HYCOM) data...")
    readers_espc = []
    espc_ds = None

    if not ENABLE_ESPC:
        print("  ESPC disabled")
    else:
        espc_forecast_end = sim_start + timedelta(days=7)
        try:
            # Check cache only if USE_CACHED_DATA is True
            if USE_CACHED_DATA:
                # 1) Check this run's local data dir
                existing_espc = [f for f in os.listdir(data_dir) if f.startswith("espc_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

                if existing_espc:
                    espc_raw_file = os.path.join(data_dir, existing_espc[0])
                    print(f"  Loading cached (local): {espc_raw_file}")
                    with xr.open_dataset(espc_raw_file) as ds:
                        espc_ds = ds.load()

                else:
                    # 2) Check existing forecast run directories
                    cached = find_cached_forecast_data("espc", sim_start, suffix="raw")
                    if cached:
                        print(f"  Found in forecast cache: {cached}")
                        with xr.open_dataset(cached) as ds:
                            espc_ds = ds.load()
                    else:
                        espc_ds = None
            else:
                print("  USE_CACHED_DATA=False, forcing fresh download")
                espc_ds = None

            # Download from server if no cached data found or caching disabled
            if espc_ds is None:
                hycom_url = 'https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd'
                print("  Downloading ESPC data...")
                print(f"    Request: {sim_start} to {espc_forecast_end} (depth=0)")

                espc_lon_min = MAP_EXTENT[0] + 360 if MAP_EXTENT[0] < 0 else MAP_EXTENT[0]
                espc_lon_max = MAP_EXTENT[1] + 360 if MAP_EXTENT[1] < 0 else MAP_EXTENT[1]

                _espc_raw = xr.open_dataset(hycom_url, drop_variables=['tau'])
                _time_dim = 'time1' if 'time1' in _espc_raw.dims else 'time'
                espc_ds = _espc_raw.sel(
                    {_time_dim: slice(sim_start, espc_forecast_end)},
                    lat=slice(MAP_EXTENT[2] - BUFFER_DEG, MAP_EXTENT[3] + BUFFER_DEG),
                    lon=slice(espc_lon_min - BUFFER_DEG, espc_lon_max + BUFFER_DEG),
                    depth=0,
                )

                _time_dim = 'time1' if 'time1' in espc_ds.dims else 'time'
                if espc_ds.sizes.get(_time_dim, 0) == 0:
                    raise ValueError(f"No ESPC data for {sim_start} to {espc_forecast_end}")

                print(f"  Downloaded {espc_ds.sizes[_time_dim]} time steps")
                espc_ds = espc_ds.load()

                # Print actual date range and depth info
                espc_times = pd.to_datetime(espc_ds[_time_dim].values)
                espc_depth = espc_ds['depth'].values if 'depth' in espc_ds.coords else [0]
                print(f"    Actual data: {espc_times[0]} to {espc_times[-1]}")
                print(f"    Depth: {espc_depth}")

                # Convert longitude from 0-360 to -180/180
                espc_ds.coords['lon'] = (espc_ds.coords['lon'] + 180) % 360 - 180
                espc_ds = espc_ds.sortby('lon')

                espc_raw_file = os.path.join(data_dir, get_forecast_filename("espc", espc_ds, "raw"))
                if not os.path.exists(espc_raw_file):
                    espc_ds.to_netcdf(espc_raw_file)
                    print(f"  Saved: {espc_raw_file}")

            # Extend if needed (in case forecast is shorter than projection_days)
            existing_ext = [f for f in os.listdir(data_dir) if f.startswith("espc_") and f.endswith("_extended.nc")]
            if existing_ext:
                espc_extended_file = os.path.join(data_dir, existing_ext[0])
            else:
                espc_extended_file = os.path.join(data_dir, get_forecast_filename("espc", espc_ds, "extended"))
                extend_forecast_to_persistence(espc_ds, "ESPC", espc_extended_file,
                                               target_days=projection_days, time_step_hours=3)

            readers_espc.append(reader_netCDF_CF_generic.Reader(espc_extended_file))
            print("  ESPC reader ready")

        except Exception as e:
            print(f"  ERROR loading ESPC: {e}")
            import traceback
            traceback.print_exc()

    # [4/6] Load CMEMS data
    print("\n[4/6] Loading CMEMS data...")
    readers_cmems = []
    cmems_ds = None

    if not ENABLE_CMEMS:
        print("  CMEMS disabled")
    else:
        cmems_forecast_end = sim_start + timedelta(days=10)
        try:
            # Check cache only if USE_CACHED_DATA is True
            if USE_CACHED_DATA:
                # 1) Check this run's local data dir
                existing_cmems = [f for f in os.listdir(data_dir) if f.startswith("cmems_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

                if existing_cmems:
                    cmems_raw_file = os.path.join(data_dir, existing_cmems[0])
                    print(f"  Loading cached (local): {cmems_raw_file}")
                    with xr.open_dataset(cmems_raw_file) as ds:
                        cmems_ds = ds.load()

                else:
                    # 2) Check existing forecast run directories
                    cached = find_cached_forecast_data("cmems", sim_start, suffix="raw")
                    if cached:
                        print(f"  Found in forecast cache: {cached}")
                        with xr.open_dataset(cached) as ds:
                            cmems_ds = ds
                            if 'depth' in cmems_ds.dims:
                                cmems_ds = cmems_ds.sel(depth=0, method='nearest')
                            cmems_ds = cmems_ds.load()
                    else:
                        cmems_ds = None
            else:
                print("  USE_CACHED_DATA=False, forcing fresh download")
                cmems_ds = None

            # Download from server if no cached data found or caching disabled
            if cmems_ds is None:
                temp_cmems_file = os.path.join(data_dir, "cmems_temp_download.nc")
                print("  Downloading CMEMS subset...")

                dl_lon_min = glider_info['lon'] - BUFFER_DEG
                dl_lon_max = glider_info['lon'] + BUFFER_DEG
                dl_lat_min = glider_info['lat'] - BUFFER_DEG
                dl_lat_max = glider_info['lat'] + BUFFER_DEG

                print(f"    Request: {sim_start.strftime('%Y-%m-%d %H:%M:%S')} to {cmems_forecast_end.strftime('%Y-%m-%d %H:%M:%S')} (depth: 0-1 m)")

                cm.subset(
                    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
                    variables=["uo", "vo"],
                    minimum_longitude=dl_lon_min,
                    maximum_longitude=dl_lon_max,
                    minimum_latitude=dl_lat_min,
                    maximum_latitude=dl_lat_max,
                    start_datetime=sim_start.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=cmems_forecast_end.strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_depth=0,
                    maximum_depth=1,
                    output_filename="cmems_temp_download.nc",
                    output_directory=data_dir,
                    username="maristizabalvar",
                    password="MariaCMEMS2018"
                )

                with xr.open_dataset(temp_cmems_file) as cmems_raw:
                    cmems_ds = cmems_raw
                    if 'depth' in cmems_ds.dims:
                        cmems_ds = cmems_ds.sel(depth=0, method='nearest')
                    cmems_ds = cmems_ds.load()

                # Print actual date range and depth info
                cmems_times = pd.to_datetime(cmems_ds['time'].values)
                cmems_depth = cmems_ds['depth'].values if 'depth' in cmems_ds.coords else [0]
                print(f"    Actual data: {cmems_times[0]} to {cmems_times[-1]}")
                print(f"    Depth: {cmems_depth}")

                if os.path.exists(temp_cmems_file):
                    os.remove(temp_cmems_file)

                cmems_raw_file = os.path.join(data_dir, get_forecast_filename("cmems", cmems_ds, "raw"))
                if not os.path.exists(cmems_raw_file):
                    cmems_ds.to_netcdf(cmems_raw_file)
                    print(f"  Saved: {cmems_raw_file}")

            existing_ext = [f for f in os.listdir(data_dir) if f.startswith("cmems_") and f.endswith("_extended.nc")]
            if existing_ext:
                cmems_extended_file = os.path.join(data_dir, existing_ext[0])
            else:
                cmems_extended_file = os.path.join(data_dir, get_forecast_filename("cmems", cmems_ds, "extended"))
                extend_forecast_to_persistence(cmems_ds, "CMEMS", cmems_extended_file,
                                               target_days=projection_days, time_step_hours=6)

            readers_cmems.append(reader_netCDF_CF_generic.Reader(cmems_extended_file))
            print("  CMEMS reader ready")

        except Exception as e:
            print(f"  ERROR loading CMEMS: {e}")
            import traceback
            traceback.print_exc()

    # [4.5/6] Load Lusitania data
    print("\n[4.5/6] Loading Lusitania data...")
    readers_lusitania = []
    lusitania_ds = None

    if not ENABLE_LUSITANIA:
        print("  Lusitania disabled")
    else:
        try:
            # 1) Check this run's local data dir
            existing_lusitania = [f for f in os.listdir(data_dir) if f.startswith("lusitania_") and f.endswith("_raw.nc")] if os.path.exists(data_dir) else []

            if existing_lusitania:
                lusitania_raw_file = os.path.join(data_dir, existing_lusitania[0])
                print(f"  Loading cached (local): {lusitania_raw_file}")
                with xr.open_dataset(lusitania_raw_file) as ds:
                    lusitania_ds = ds.load()

            else:
                # 2) Check existing forecast run directories
                cached = find_cached_forecast_data("lusitania", sim_start, suffix="raw")
                if cached:
                    print(f"  Found in forecast cache: {cached}")
                    with xr.open_dataset(cached) as ds:
                        lusitania_ds = ds
                        if 'depth' in lusitania_ds.dims:
                            lusitania_ds = lusitania_ds.sel(depth=0, method='nearest')
                        lusitania_ds = lusitania_ds.load()
                else:
                    # 3) Download from server
                    print("  Downloading Lusitania data...")
                    lusitania_full = lusitania_uv_multi(start_time=sim_start, rename=True)
                    if lusitania_full is None:
                        raise ValueError("Could not load Lusitania data")

                    dl_lon_min = glider_info['lon'] - BUFFER_DEG
                    dl_lon_max = glider_info['lon'] + BUFFER_DEG
                    dl_lat_min = glider_info['lat'] - BUFFER_DEG
                    dl_lat_max = glider_info['lat'] + BUFFER_DEG

                    lon_name = 'lon' if 'lon' in lusitania_full.coords else 'longitude'
                    lat_name = 'lat' if 'lat' in lusitania_full.coords else 'latitude'

                    lons = lusitania_full[lon_name].values
                    lats = lusitania_full[lat_name].values

                    lon_slice = slice(dl_lon_max, dl_lon_min) if lons[0] > lons[-1] else slice(dl_lon_min, dl_lon_max)
                    lat_slice = slice(dl_lat_max, dl_lat_min) if lats[0] > lats[-1] else slice(dl_lat_min, dl_lat_max)

                    lusitania_ds = lusitania_full.sel({lon_name: lon_slice, lat_name: lat_slice})
                    if 'depth' in lusitania_ds.dims:
                        lusitania_ds = lusitania_ds.sel(depth=0, method='nearest')
                    lusitania_ds = lusitania_ds.load()

                lusitania_raw_file = os.path.join(data_dir, get_forecast_filename("lusitania", lusitania_ds, "raw"))
                if not os.path.exists(lusitania_raw_file):
                    lusitania_ds.to_netcdf(lusitania_raw_file)
                    print(f"  Saved: {lusitania_raw_file}")

            existing_ext = [f for f in os.listdir(data_dir) if f.startswith("lusitania_") and f.endswith("_extended.nc")]
            if existing_ext:
                lusitania_extended_file = os.path.join(data_dir, existing_ext[0])
            else:
                lusitania_extended_file = os.path.join(data_dir, get_forecast_filename("lusitania", lusitania_ds, "extended"))
                extend_forecast_to_persistence(lusitania_ds, "Lusitania", lusitania_extended_file,
                                               target_days=projection_days, time_step_hours=6)

            readers_lusitania.append(reader_netCDF_CF_generic.Reader(lusitania_extended_file))
            print("  Lusitania reader ready")

        except Exception as e:
            print(f"  ERROR loading Lusitania: {e}")
            import traceback
            traceback.print_exc()

    # Prepare quiver data and persistence times
    quiver_data = {}
    persistence_times = {}

    if PLOT_CURRENTS:
        if espc_ds is not None:
            _espc_time_dim = 'time1' if 'time1' in espc_ds.dims else 'time'
            quiver_data['ESPC'] = espc_ds
            espc_times = pd.to_datetime(espc_ds[_espc_time_dim].values)
            persistence_times['ESPC'] = espc_times[-1].to_pydatetime()
        if cmems_ds is not None:
            quiver_data['CMEMS'] = cmems_ds
            cmems_times = pd.to_datetime(cmems_ds['time'].values)
            persistence_times['CMEMS'] = cmems_times[-1].to_pydatetime()
        if lusitania_ds is not None:
            quiver_data['Lusitania'] = lusitania_ds
            lusitania_times = pd.to_datetime(lusitania_ds['time'].values)
            persistence_times['Lusitania'] = lusitania_times[-1].to_pydatetime()

    # [5/6] Run simulations
    print("\n[5/6] Running simulations...")

    for diff_val in DIFFUSIVITY_VALUES:
        print(f"\n  Diffusivity: {diff_val} m\u00b2/s")
        run_folder = os.path.join(figures_dir, f"diff{diff_val}")
        os.makedirs(run_folder, exist_ok=True)

        model_outputs = {}

        if readers_espc:
            print("    Running ESPC...")
            try:
                output_file, _, _ = run_simulation(
                    glider_info, NUM_PARTICLES, projection_days,
                    run_folder, "ESPC", readers_espc,
                    horizontal_diffusivity=diff_val
                )
                model_outputs['ESPC'] = xr.open_dataset(output_file)
                print(f"    ESPC complete")
            except Exception as e:
                print(f"    ESPC ERROR: {e}")

        if readers_cmems:
            print("    Running CMEMS...")
            try:
                output_file, _, _ = run_simulation(
                    glider_info, NUM_PARTICLES, projection_days,
                    run_folder, "CMEMS", readers_cmems,
                    horizontal_diffusivity=diff_val
                )
                model_outputs['CMEMS'] = xr.open_dataset(output_file)
                print(f"    CMEMS complete")
            except Exception as e:
                print(f"    CMEMS ERROR: {e}")

        if readers_lusitania:
            print("    Running Lusitania...")
            try:
                output_file, _, _ = run_simulation(
                    glider_info, NUM_PARTICLES, projection_days,
                    run_folder, "Lusitania", readers_lusitania,
                    horizontal_diffusivity=diff_val
                )
                model_outputs['Lusitania'] = xr.open_dataset(output_file)
                print(f"    Lusitania complete")
            except Exception as e:
                print(f"    Lusitania ERROR: {e}")

        # [6/6] Create verification plots
        if model_outputs:
            print("\n[6/6] Creating verification plots...")

            # Full extent map
            map_path = create_verification_map(
                model_outputs, glider_info, actual_track,
                run_folder, projection_days, diff_val, extent,
                persistence_times=persistence_times
            )

            # Zoomed map (tight bounding box around all data)
            zoomed_extent = compute_zoomed_extent(model_outputs, actual_track, glider_info)
            zoomed_map_path = create_verification_map(
                model_outputs, glider_info, actual_track,
                run_folder, projection_days, diff_val, zoomed_extent,
                persistence_times=persistence_times, suffix="zoomed"
            )

            # Moderate extent for animations (between full and tight zoom)
            moderate_extent = compute_moderate_extent(extent, zoomed_extent, zoom_factor=0.6)

            # Base animation (no currents) with moderate zoom
            animation_path = create_verification_animation(
                model_outputs, glider_info, actual_track,
                run_folder, projection_days, diff_val, moderate_extent,
                quiver_model=None, quiver_data=quiver_data,
                persistence_times=persistence_times
            )

            # Current overlay animations with moderate zoom
            if PLOT_CURRENTS:
                for qmodel in quiver_data:
                    create_verification_animation(
                        model_outputs, glider_info, actual_track,
                        run_folder, projection_days, diff_val, moderate_extent,
                        quiver_model=qmodel, quiver_data=quiver_data,
                        persistence_times=persistence_times
                    )

            # Copy diff=100 outputs to latest files
            if diff_val == 100:
                print(f"    Copying diff=100 outputs to latest files...")
                anim_ext = os.path.splitext(animation_path)[1]
                latest_animation = os.path.join(LATEST_DIR, f"hindcast_verification_latest{anim_ext}")
                latest_map = os.path.join(LATEST_DIR, "hindcast_verification_latest_map.png")
                latest_zoomed_map = os.path.join(LATEST_DIR, "hindcast_verification_latest_map_zoomed.png")
                shutil.copy2(animation_path, latest_animation)
                shutil.copy2(map_path, latest_map)
                shutil.copy2(zoomed_map_path, latest_zoomed_map)
                print(f"    Copied to: {latest_animation}")
                print(f"    Copied to: {latest_map}")
                print(f"    Copied to: {latest_zoomed_map}")
                archive_hindcast_outputs(map_path, animation_path, zoomed_map_path)

        # Cleanup
        for ds in model_outputs.values():
            ds.close()

    print("\n" + "=" * 70)
    print("HINDCAST VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
