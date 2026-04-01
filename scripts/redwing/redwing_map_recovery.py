import datetime as dt
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from functools import lru_cache
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import shutil
import cmocean
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from cool_maps.plot import create, add_bathymetry
import cartopy.mpl.ticker as cticker
from oceans.ocfis import uv2spdir
import cartopy.feature as cfeature
import requests

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is optional at runtime
    gpd = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configuration dictionary for all parameters
CONFIG = {
    'paths': {
        'save_path': '/Users/mikesmith/Documents/',
        'bathy_file': '/Users/mikesmith/Documents/github/ioos_model_comparisons/sentinel_bathy.nc',  # Use relative path by default
        'eez_path': '/Users/mikesmith/Downloads/World_Exclusive_Economic_Zones_Boundaries-shp/World_Exclusive_Economic_Zones_Boundaries.shp'
    },
    # 'paths': {
    #       'save_path': '/www/web/rucool/media/sentinel/canaries',
    #       'bathy_file': '/home/michaesm/sentinel/sentinel_bathy.nc',  # Use relative path by default
    #       'eez_path': '/home/hurricaneadm/data/World_Exclusive_Economic_Zones_Boundaries-shp/World_Exclusive_Economic_Zones_Boundaries.shp',
    #       },
    'models': {
        'plot_model_data': True,
        'plot_rtofs': False,
        'plot_espc': True,
        'plot_cmems': True,
        'plot_lusitania': True,
    },
    'glider': {
        'id': "redwing-20251011T1511",
        'api_url': "https://marine.rutgers.edu/cool/data/gliders/api/surfacings/",
        'api_timeout': 30,
    },
    'region': {
        'name': "Sentinel Mission",
        'folder': "sentinel",
        # 'extent': [-72.25+15, -44.75+15, 34.75-3, 45.25-3],  # [lon_min, lon_max, lat_min, lat_max]
        'extent': [-40.25, -9.75, 24.75, 40.25]
    },
    'currents': {
        'enabled': True,
        'depths': [0, 150, 200],
        'limits': [0, 90, 10],
        'coarsen': {'rtofs': 7, 'espc': 8},
        'streamplot': {
            'density': 3,
            'linewidth': 0.75,
            'color': 'black',
        }
    },
    'depth_average': {
        'min_depth': 0,
        'max_depth': 1000,
        'depth_step': 1,
    },
    'plotting': {
        'figsize': (12, 7),
        'dpi': 300,
        'legend_columns': 7,
        'glider_track_linewidth': 4,
        'glider_marker_size': 10,
        'waypoint_marker_size': 6,  # Added for waypoint
    },
    'bathymetry': {
        'enabled': False,  # Set to True to enable bathymetry plotting
        'contour_levels': (-1000, -100),
        'filled_levels': [-8000, -1000, -100, 0],
        'filled_colors': ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue'],
    }
}

PLOCAN_LOCATION = {
    "name": "PLOCAN",
    "latitude": 27.9921099522062,
    "longitude": -15.368678085796802,
}

# Gulf Stream boundary service constants
GULF_STREAM_SERVICE_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "Gulf_Stream_Boundary/FeatureServer/0/query"
)
GULF_STREAM_QUERY_PARAMS = {
    "where": "1=1",
    "outFields": "*",
    "outSR": "4326",
    "f": "geojson",
}


# Projections
MAP_PROJECTION = ccrs.Mercator()
DATA_PROJECTION = ccrs.PlateCarree()


# ============================================================================
# CUSTOM LEGEND HANDLER FOR TARGET MARKER
# ============================================================================

class TargetHandler(HandlerBase):
    """Custom legend handler to overlay multiple markers as a target/bullseye."""
    
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """Create overlaid circle artists for the legend entry."""
        # Center position in legend box
        x_center = width / 2.0
        y_center = height / 2.0
        
        artists = []
        for handle in orig_handle:
            # Create a copy of the marker at the center position
            marker = Line2D([x_center], [y_center],
                          marker=handle.get_marker(),
                          markersize=handle.get_markersize(),
                          markerfacecolor=handle.get_markerfacecolor(),
                          markeredgecolor=handle.get_markeredgecolor(),
                          markeredgewidth=handle.get_markeredgewidth(),
                          transform=trans)
            artists.append(marker)
        
        return artists


# ============================================================================
# GULF STREAM BOUNDARIES
# ============================================================================

@lru_cache(maxsize=1)
def load_gulf_stream_boundaries():
    """Fetch Gulf Stream boundary features from the ArcGIS service."""
    if gpd is None:
        logger.warning(
            "Geopandas is required to plot the Gulf Stream north wall; skipping."
        )
        return None

    params = urlencode(GULF_STREAM_QUERY_PARAMS)
    url = f"{GULF_STREAM_SERVICE_URL}?{params}"

    try:
        gdf = gpd.read_file(url)
    except Exception as exc:
        logger.warning("Failed to load Gulf Stream boundaries: %s", exc)
        return None

    if "name" not in gdf.columns:
        logger.warning("Gulf Stream boundary payload missing 'name' column.")
        return None

    return gdf


def get_gulf_stream_wall(branch: str = "north"):
    """
    Return the requested Gulf Stream boundary branch (e.g., north wall).

    Parameters
    ----------
    branch : str, optional
        Named branch within the dataset ('north' or 'south').

    Returns
    -------
    geopandas.GeoDataFrame or None
        GeoDataFrame containing the requested boundary or None if unavailable.
    """
    gdf = load_gulf_stream_boundaries()
    if gdf is None:
        return None

    column = gdf["name"].fillna("").str.lower()
    subset = gdf[column == branch.lower()]
    if subset.empty:
        logger.warning("No Gulf Stream boundary entries found for '%s'.", branch)
        return None

    return subset


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lon180to360(array: np.ndarray) -> np.ndarray:
    """Convert longitude from -180:180 to 0:360 range."""
    array = np.array(array)
    return np.mod(array, 360)


def lon360to180(array: np.ndarray) -> np.ndarray:
    """Convert longitude from 0:360 to -180:180 range."""
    array = np.array(array)
    return np.mod(array + 180, 360) - 180


def ddmm_to_degrees(val) -> float:
    """Convert DDMM.MMMM strings to signed decimal degrees."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    val_float = float(val)
    sign = -1 if val_float < 0 else 1
    v = abs(val_float)
    degrees = int(v // 100)
    minutes = v - 100 * degrees
    return sign * (degrees + minutes / 60.0)


def expand_extent(extent: list, buffer: float = 1.0) -> list:
    """Expand geographic extent by a buffer amount."""
    return np.add(extent, [-buffer, buffer, -buffer, buffer]).tolist()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def fetch_glider_surfacings(
    deployment_id: str,
    base_url: str,
    timeout: int = 30
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Fetch glider surfacing data from the COOL API.
    
    Parameters
    ----------
    deployment_id : str
        Glider deployment identifier
    base_url : str
        API base URL
    timeout : int, optional
        Request timeout in seconds
        
    Returns
    -------
    tuple
        (DataFrame with time index and latitude/longitude columns, 
         waypoint dict or None)
    """
    params = {"deployment": deployment_id}
    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.error(f"Unable to retrieve glider surfacings: {exc}")
        return pd.DataFrame(columns=["latitude", "longitude"]), None

    records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not records:
        logger.warning("No glider surfacings returned from API.")
        return pd.DataFrame(columns=["latitude", "longitude"]), None

    df = pd.json_normalize(records)
    df["gps_time"] = pd.to_datetime(df.get("gps_timestamp_epoch"), unit="s", utc=True, errors="coerce")

    # Parse latitude
    if "gps_lat_degrees" in df.columns:
        lat = pd.to_numeric(df["gps_lat_degrees"], errors="coerce")
    else:
        lat_raw = df.get("gps_lat")
        lat = lat_raw.map(ddmm_to_degrees) if lat_raw is not None else np.nan
    df["latitude"] = lat

    # Parse longitude
    if "gps_lon_degrees" in df.columns:
        lon = pd.to_numeric(df["gps_lon_degrees"], errors="coerce")
    else:
        lon_raw = df.get("gps_lon")
        lon = lon_raw.map(ddmm_to_degrees) if lon_raw is not None else np.nan
    df["longitude"] = lon

    df = df.dropna(subset=["gps_time", "latitude", "longitude"])
    if df.empty:
        logger.warning("Glider API response contained no usable position records.")
        return pd.DataFrame(columns=["latitude", "longitude"]), None

    df["gps_time"] = df["gps_time"].dt.tz_convert(None)
    df = df.sort_values("gps_time").set_index("gps_time")
    df.index.name = "time"

    # Extract waypoint information from most recent surfacing with complete waypoint data
    waypoint_info = None
    waypoint_cols = ["waypoint_lat", "waypoint_lon", "waypoint_bearing_degrees", "waypoint_range_meters"]
    
    # Check if waypoint columns exist in the dataframe
    if all(col in df.columns for col in waypoint_cols):
        # Sort by time descending and find first record with all waypoint data
        for idx in df.index[::-1]:  # iterate from most recent
            row = df.loc[idx]
            
            # Check if all waypoint fields are valid (not NaN and not empty string)
            all_valid = True
            for col in waypoint_cols:
                val = row[col]
                # Handle both scalar and potential Series
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    # If it's a Series or array-like, check if empty or all NaN
                    all_valid = False
                    break
                if pd.isna(val) or val == '' or val is None:
                    all_valid = False
                    break
            
            if all_valid:
                # Parse waypoint lat/lon
                wp_lat_raw = row["waypoint_lat"]
                wp_lon_raw = row["waypoint_lon"]
                
                # Convert to degrees if needed
                try:
                    if isinstance(wp_lat_raw, str) or (isinstance(wp_lat_raw, float) and abs(wp_lat_raw) > 90):
                        wp_lat = ddmm_to_degrees(wp_lat_raw)
                    else:
                        wp_lat = float(wp_lat_raw)
                    
                    if isinstance(wp_lon_raw, str) or (isinstance(wp_lon_raw, float) and abs(wp_lon_raw) > 180):
                        wp_lon = ddmm_to_degrees(wp_lon_raw)
                    else:
                        wp_lon = float(wp_lon_raw)
                    
                    waypoint_info = {
                        'latitude': wp_lat,
                        'longitude': wp_lon,
                        'bearing': float(row["waypoint_bearing_degrees"]),
                        'range': float(row["waypoint_range_meters"]),
                        'time': idx
                    }
                    logger.info(f"Found waypoint: lat={wp_lat:.4f}, lon={wp_lon:.4f}, "
                               f"bearing={waypoint_info['bearing']:.1f}°, "
                               f"range={waypoint_info['range']:.0f}m")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse waypoint at {idx}: {e}")
                    continue
    
    if waypoint_info is None:
        logger.info("No waypoint data found in surfacings")

    return df[["latitude", "longitude"]], waypoint_info


def load_bathymetry(bathy_path: str) -> Optional[xr.Dataset]:
    """
    Load bathymetry dataset.
    
    Parameters
    ----------
    bathy_path : str
        Path to bathymetry netCDF file
        
    Returns
    -------
    xr.Dataset or None
        Bathymetry dataset or None if loading fails
    """
    try:
        return xr.open_dataset(bathy_path)
    except FileNotFoundError:
        logger.warning(f"Bathymetry file not found: {bathy_path}")
        return None
    except Exception as exc:
        logger.warning(f"Error loading bathymetry: {exc}")
        return None


def load_rtofs(extent: list) -> Tuple[Optional[xr.Dataset], Optional[Dict]]:
    """
    Load RTOFS model data.
    
    Parameters
    ----------
    extent : list
        Geographic extent [lon_min, lon_max, lat_min, lat_max]
        
    Returns
    -------
    tuple
        (dataset, grid_info_dict) or (None, None) if loading fails
    """
    try:
        from ioos_model_comparisons.models import rtofs as r
        rds = r()
        
        grid_info = {
            'lons': rds.lon.values[0, :],
            'lats': rds.lat.values[:, 0],
            'x': rds.x.values,
            'y': rds.y.values
        }
        
        logger.info("RTOFS data loaded successfully")
        return rds, grid_info
    except Exception as exc:
        logger.error(f"Failed to load RTOFS data: {exc}")
        return None, None


def load_espc(extent: list, reference_date: dt.datetime) -> Tuple[Optional[xr.Dataset], bool]:
    """
    Load ESPC model data.
    
    Parameters
    ----------
    extent : list
        Geographic extent [lon_min, lon_max, lat_min, lat_max]
    reference_date : datetime
        Reference date to determine archive vs operational
        
    Returns
    -------
    tuple
        (dataset, is_archive) or (None, False) if loading fails
    """
    now_naive = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None)
    today_naive = pd.to_datetime(reference_date).to_pydatetime().replace(tzinfo=None)
    archive_espc = (now_naive - today_naive) > dt.timedelta(days=8)
    
    try:
        if archive_espc:
            from ioos_model_comparisons.models import ESPC as g
            gobj = g(year=reference_date.year)
            espc_ds = gobj.get_combined_subset(
                [extent[0], extent[1]],
                [extent[2], extent[3]]
            )
            logger.info("ESPC archive data loaded successfully")
            return espc_ds, True
        else:
            from ioos_model_comparisons.models import espc_uv
            espc_u = espc_uv(rename=True)
            logger.info("ESPC operational data loaded successfully")
            return espc_u, False
    except Exception as exc:
        logger.error(f"ESPC data load failed: {exc}")
        return None, False


def load_cmems(extent: list) -> Optional[xr.Dataset]:
    """
    Load CMEMS model data.
    
    Parameters
    ----------
    extent : list
        Geographic extent [lon_min, lon_max, lat_min, lat_max]
        
    Returns
    -------
    xr.Dataset or None
        CMEMS dataset or None if loading fails
    """
    try:
        from ioos_model_comparisons.models import CMEMS as c
        cobj = c()
        cds = cobj.get_combined_subset(
            [extent[0], extent[1]],
            [extent[2], extent[3]]
        )
        logger.info("CMEMS data loaded successfully")
        return cds
    except Exception as exc:
        logger.error(f"Failed to load CMEMS data: {exc}")
        return None
    

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
import datetime as dt
import re
from typing import Optional, List, Tuple
from functools import lru_cache
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
def _parse_lusitania_catalog(timeout: int = 30) -> List[Tuple[dt.datetime, str]]:
    """
    Parse the Lusitania_Vertical THREDDS catalog to get available files.
    
    Parameters
    ----------
    timeout : int, optional
        Request timeout in seconds
        
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
                    file_date = dt.datetime.strptime(date_str, '%Y%m%d%H')
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
                file_date = dt.datetime.strptime(date_str, '%Y%m%d%H')
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
def _get_lusitania_catalog() -> List[Tuple[dt.datetime, str]]:
    """Get cached catalog listing (refreshes once per session)."""
    return _parse_lusitania_catalog(timeout=LUSITANIA_CONFIG['timeout'])


def _find_lusitania_file(
    target_time: dt.datetime,
    available_files: Optional[List[Tuple[dt.datetime, str]]] = None,
    method: str = 'nearest'
) -> Optional[Tuple[dt.datetime, str]]:
    """
    Find the Lusitania file nearest to the target time.
    
    Since each file contains one full day (00:00 GMT to 00:00 GMT next day),
    for a target time in the middle of a day, use method='before' to ensure
    you get the file that contains that timestamp.
    
    Parameters
    ----------
    target_time : datetime
        Target time to match
    available_files : list, optional
        List of (datetime, filename) tuples. If None, fetches from catalog.
    method : str, optional
        Selection method: 
        - 'nearest': Find file with minimum time difference (default)
        - 'before': Find latest file with start time <= target (recommended 
          for getting the file containing a specific timestamp)
        - 'after': Find earliest file with start time >= target
        
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


def lusitania_uv(target_time: Optional[dt.datetime] = None, rename: bool = True, method: str = 'nearest'):
    """
    Load Lusitania model data from the AtlanticSense THREDDS server.
    
    This function crawls the THREDDS catalog to find the file closest to the
    requested time, then loads the data via OPeNDAP.
    
    File Structure
    --------------
    Each file (e.g., 2026010300.nc) contains one complete day of data:
    from 00:00 GMT of the file date to 00:00 GMT of the next day.
    
    The model runs daily with hindcast (day -2), current day, and 6-day 
    forecast, providing approximately 3-day forecast capability.
    
    Parameters
    ----------
    target_time : datetime, optional
        Target time to load. If None, loads the most recent available file.
        The function will find the file whose date contains this time.
    rename : bool, optional
        If True, rename velocity variables to 'u' and 'v' for consistency.
        Default True.
    method : str, optional
        Time matching method: 'nearest', 'before', or 'after'. Default 'nearest'.
        
    Returns
    -------
    xr.Dataset or None
        Dataset containing velocity variables, or None if loading fails.
        
    Examples
    --------
    >>> # Load most recent data
    >>> ds = lusitania_uv()
    
    >>> # Load data for a specific date (will get file containing that time)
    >>> ds = lusitania_uv(target_time=dt.datetime(2026, 1, 7, 12, 0))
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
            u_candidates = ['uo', 'water_u', 'u_velocity', 'U', 'ucur']
            v_candidates = ['vo', 'water_v', 'v_velocity', 'V', 'vcur']
            
            for u_name in u_candidates:
                if u_name in ds.data_vars:
                    rename_map[u_name] = 'u'
                    break
            
            for v_name in v_candidates:
                if v_name in ds.data_vars:
                    rename_map[v_name] = 'v'
                    break
            
            if rename_map:
                logger.info(f"Renaming variables: {rename_map}")
                ds = ds.rename(rename_map)
        
        return ds
        
    except Exception as exc:
        logger.error(f"Failed to load Lusitania data: {exc}")
        return None

def load_lusitania(reference_date: dt.datetime) -> Tuple[Optional[xr.Dataset], bool]:
    """
    Load Lusitania model data.
    
    Parameters
    ----------
    extent : list
        Geographic extent [lon_min, lon_max, lat_min, lat_max]
    reference_date : datetime
        Reference date to determine archive vs operational
        
    Returns
    -------
    tuple
        (dataset, is_archive) or (None, False) if loading fails
    """
    # now_naive = pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None)
    # today_naive = pd.to_datetime(reference_date).to_pydatetime().replace(tzinfo=None)
    # archive_espc = (now_naive - today_naive) > dt.timedelta(days=8)
    
    try:
        # if archive_espc:
        #     from ioos_model_comparisons.models import lusitania_uv as g
        #     gobj = g(year=reference_date.year)
        #     espc_ds = gobj.get_combined_subset(
        #         [extent[0], extent[1]],
        #         [extent[2], extent[3]]
        #     )
        #     logger.info("ESPC archive data loaded successfully")
        #     return espc_ds, True
        # else:
        # from ioos_model_comparisons.models import lusitania_uv
        lusitania_u = lusitania_uv(target_time=reference_date, rename=True,  method='before')
        logger.info("Lusitania operational data loaded successfully")
        return lusitania_u, False
    except Exception as exc:
        logger.error(f"Lusitania data load failed: {exc}")
        return None, False
    
# ============================================================================
# CURRENT PROCESSING FUNCTIONS
# ============================================================================

def compute_depth_avg_currents(
    ds: xr.Dataset,
    min_depth: float = 0,
    max_depth: float = 1000,
    depth_step: float = 1,
    depth_dim_hint: str = "depth",
) -> Optional[xr.Dataset]:
    """
    Interpolate currents to 1 m vertical spacing between min_depth and
    max_depth and return the depth-averaged u/v fields.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing u and v variables with a vertical dimension
    min_depth : float, optional
        Shallow bound (meters, positive down) to include in the average
    max_depth : float, optional
        Deep bound (meters, positive down) to include in the average
    depth_step : float, optional
        Vertical resolution (meters) used for interpolation prior to averaging
    depth_dim_hint : str, optional
        Hint for the name of the depth dimension if it is not 'depth'

    Returns
    -------
    xr.Dataset or None
        Dataset containing depth-averaged u and v on the original
        horizontal grid. Returns None when the required inputs are missing.
    """
    if ds is None:
        return None
    if not isinstance(ds, xr.Dataset):
        logger.warning("Depth-average skipped: dataset expected.")
        return None
    missing = [var for var in ("u", "v") if var not in ds]
    if missing:
        logger.warning(f"Depth-average skipped: missing variables {missing}.")
        return None

    # Attempt to locate the vertical dimension used by the model
    candidate_dims = [
        depth_dim_hint,
        "depth",
        "Depth",
        "depthu",
        "depthv",
        "z",
        "lev",
        "level",
    ]
    depth_dim = None
    for cand in candidate_dims:
        if cand in ds["u"].dims:
            depth_dim = cand
            break
    if depth_dim is None:
        for cand in candidate_dims:
            if cand in ds["v"].dims:
                depth_dim = cand
                break
    if depth_dim is None:
        logger.warning("Depth-average skipped: unable to determine depth dimension.")
        return None
    if depth_dim not in ds.coords:
        logger.warning(f"Depth-average skipped: coordinate '{depth_dim}' missing from dataset.")
        return None

    ds_uv = ds[["u", "v"]]
    depth_coord = ds_uv.coords.get(depth_dim)
    if depth_coord is None:
        logger.warning(f"Depth-average skipped: coordinate '{depth_dim}' missing from dataset.")
        return None

    depth_values = depth_coord.astype(float)
    finite_depths = depth_values.where(np.isfinite(depth_values), drop=True)
    if finite_depths.size == 0:
        logger.warning("Depth-average skipped: no finite depth values found.")
        return None

    raw_depths = np.asarray(finite_depths.values)
    if raw_depths.ndim == 0:
        raw_depths = np.array([float(raw_depths)])

    # Handle negative depths (oceanographic convention)
    if np.all(raw_depths <= 0):
        positive_depths = np.abs(raw_depths)
        ds_uv = ds_uv.assign_coords({depth_dim: positive_depths})
        finite_depths = xr.DataArray(
            positive_depths,
            coords={depth_dim: positive_depths},
            dims=depth_dim,
        )
    else:
        positive_depths = raw_depths

    start_depth = max(min_depth, float(np.nanmin(positive_depths)))
    end_depth = min(max_depth, float(np.nanmax(positive_depths)))
    if end_depth <= start_depth:
        logger.warning("Depth-average skipped: depth range outside available data.")
        return None

    target_depths = np.arange(start_depth, end_depth + depth_step, depth_step)
    if target_depths.size == 0:
        logger.warning("Depth-average skipped: no target depths generated.")
        return None

    # Select and interpolate the subset before averaging
    ds_uv = ds_uv.sortby(depth_dim)
    ds_trimmed = ds_uv.sel({depth_dim: slice(start_depth, end_depth)})
    ds_interp = ds_trimmed.interp({depth_dim: target_depths})

    depth_avg = ds_interp.mean(dim=depth_dim, skipna=True)
    depth_avg.attrs = dict(ds.attrs)
    depth_avg.attrs["depth_average"] = {
        "min": float(start_depth),
        "max": float(end_depth),
        "step": float(depth_step),
    }
    depth_avg.attrs["product"] = "Depth-averaged currents"
    for var in ("u", "v"):
        original_attrs = getattr(ds[var], "attrs", None) or {}
        depth_avg[var].attrs = dict(original_attrs)
        depth_avg[var].attrs["vertical_aggregation"] = (
            f"mean {start_depth}-{end_depth} m"
        )

    return depth_avg


def currents_to_cm_per_s(ds: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
    """Convert velocity components from m/s to cm/s for plotting."""
    if ds is None or not isinstance(ds, xr.Dataset):
        return ds

    scaled = ds.copy()
    for comp in ("u", "v"):
        if comp in scaled:
            scaled[comp] = scaled[comp] * 100
            attrs = dict(getattr(ds[comp], "attrs", {}))
            attrs["units"] = "cm/s"
            scaled[comp].attrs = attrs

    scaled.attrs = dict(getattr(ds, "attrs", {}))
    scaled.attrs["velocity_units"] = "cm/s"
    return scaled

def map_add_currents(
    ax,
    ds: xr.Dataset,
    density: int = 2,
    linewidth: float = 0.75,
    color: str = 'black',
    transform=DATA_PROJECTION
):
    """
    Add current streamlines to map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes
    ds : xr.Dataset
        Dataset containing u and v current components
    density : int, optional
        Density of streamlines
    linewidth : float, optional
        Line width for streamplot
    color : str, optional
        Line color for streamplot
    transform : cartopy.crs, optional
        Coordinate reference system

    Returns
    -------
    matplotlib streamplot object
    """
    angle, speed = uv2spdir(ds['u'], ds['v'])

    lons = ds.lon.squeeze().data
    lats = ds.lat.squeeze().data
    u = ds.u.squeeze().data
    v = ds.v.squeeze().data
    
    sargs = {
        "transform": transform,
        "density": density,
        "linewidth": linewidth
    }
    
    if color:
        sargs["color"] = color
    else:
        sargs["color"] = speed
        sargs["cmap"] = cmocean.cm.speed
        
    q = ax.streamplot(lons, lats, u, v, **sargs)
    return q


def map_add_eez(ax, zorder=1, color='white', linewidth=0.75, linestyle='-'):
    # reader = Reader('').geometries()

    # filtered = []
    # for record in reader.records():
    #     if record.attributes['LINE_TYPE'] == 'Straight Baseline':
    #         continue
    #     else:
    #         filtered.append(record.geometry)
    from cartopy.io.shapereader import Reader        
    shape_feature = cfeature.ShapelyFeature(
        Reader(CONFIG['paths']['eez_path']).geometries(),
        ccrs.PlateCarree(),
        linestyle=linestyle,
        linewidth=linewidth,
        edgecolor=color, 
        facecolor='none'
        )
    h = ax.add_feature(shape_feature, zorder=zorder)
    return h

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_redwing_map(
    model_ds: Optional[xr.Dataset] = None,
    bathy: Optional[xr.Dataset] = None,
    gliders: Optional[pd.DataFrame] = None,
    waypoint: Optional[Dict] = None,
    config: Dict = None,
    path_save: str = None,
    model_name: Optional[str] = None
):
    """
    Generate map visualization of glider track with optional model currents.
    
    Parameters
    ----------
    model_ds : xr.Dataset, optional
        Model dataset containing u, v currents
    bathy : xr.Dataset, optional
        Bathymetry dataset
    gliders : pd.DataFrame, optional
        Glider track data with latitude/longitude
    waypoint : dict, optional
        Waypoint information with lat, lon, bearing, range
    config : dict, optional
        Configuration dictionary
    path_save : str, optional
        Directory to save output files
    model_name : str, optional
        Name of the model for labeling
    """
    if config is None:
        config = CONFIG
    
    if path_save is None:
        path_save = config['paths']['save_path']
    
    Path(path_save).mkdir(parents=True, exist_ok=True)

    # Extract configuration
    figsize = config['plotting']['figsize']
    dpi = config['plotting']['dpi']
    extent = config['region']['extent']
    bathy_config = config['bathymetry']
    bathy_enabled = bathy_config.get('enabled', True) and bathy is not None
    stream_config = config['currents']['streamplot']
    # north_wall_gdf = get_gulf_stream_wall("north")
    gulf_stream_color = 'lime'
    gulf_stream_label = 'Gulf Stream North Wall'

    # Prepare metadata
    ds_time = None
    surface_ds = None
    depth_avg_ds = None
    latest_glider_text = None
    model_label = model_name

    if gliders is not None and hasattr(gliders, "empty") and not gliders.empty:
        latest_glider_ts = pd.to_datetime(gliders.index.max())
        latest_glider_text = latest_glider_ts.strftime('%Y-%m-%d %H:%MZ')

    # Process model data if available
    if model_ds is not None:
        ds_time = pd.to_datetime(model_ds.time.data)
        logger.info("Plotting currents @ 0m")
        surface_ds = model_ds.sel(depth=0, method='nearest')
        
        # === DIAGNOSTIC ===
        logger.info(f"model_ds dims: {model_ds.dims}")
        logger.info(f"model_ds u shape: {model_ds['u'].shape}")
        logger.info(f"surface_ds dims: {surface_ds.dims}")
        logger.info(f"surface_ds u shape: {surface_ds['u'].shape}")
        logger.info(f"surface_ds lon shape: {surface_ds['lon'].shape}")
        logger.info(f"surface_ds lat shape: {surface_ds['lat'].shape}")
        # === END DIAGNOSTIC ===
        
        depth_avg_ds = compute_depth_avg_currents(
            model_ds,
            min_depth=config['depth_average']['min_depth'],
            max_depth=config['depth_average']['max_depth'],
            depth_step=config['depth_average']['depth_step']
        )
        surface_ds = currents_to_cm_per_s(surface_ds)
        depth_avg_ds = currents_to_cm_per_s(depth_avg_ds)
        if model_label is None and hasattr(model_ds, "attrs"):
            model_label = model_ds.attrs.get('model')
    else:
        logger.info("Generating map without model currents")

    if model_label is None:
        model_label = "Track"
    else:
        model_label = str(model_label)

    # Create filename-safe slug
    slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in model_label)
    slug = '-'.join(filter(None, slug.split('-')))
    if not slug:
        slug = 'track'

    # Contour arguments for speed magnitude
    qargs = {
        'transform': DATA_PROJECTION,
        'cmap': cmocean.cm.speed,
        'extend': "max",
        'levels': np.arange(
            config['currents']['limits'][0],
            config['currents']['limits'][1],
            config['currents']['limits'][2]
        )
    }

    def init_axis(ax, add_legend: bool = False, add_text: bool = False):
        """Initialize map axis with bathymetry and glider track."""
        create(extent, ax=ax, ticks=False, landcolor='sandybrown')
        
        if bathy_enabled and bathy is not None:
            add_bathymetry(
                ax,
                bathy.longitude.values,
                bathy.latitude.values,
                bathy.z.values,
                levels=bathy_config['contour_levels'],
                zorder=1.5
            )
            ax.contourf(
                bathy['longitude'],
                bathy['latitude'],
                bathy['z'],
                bathy_config['filled_levels'],
                colors=bathy_config['filled_colors'],
                transform=DATA_PROJECTION,
                ticks=False
            )

        # if north_wall_gdf is not None and not north_wall_gdf.empty:
        #     try:
        #         north_wall_gdf.plot(
        #             ax=ax,
        #             transform=DATA_PROJECTION,
        #             linewidth=2,
        #             color=gulf_stream_color,
        #             zorder=4000
        #         )
        #         if add_legend:
        #             ax.plot(
        #                 [],
        #                 [],
        #                 color=gulf_stream_color,
        #                 linewidth=2,
        #                 label=gulf_stream_label
        #             )
        #     except Exception as exc:
        #         logger.warning("Failed to plot Gulf Stream north wall: %s", exc)
        
        if gliders is not None and not gliders.empty:
            ax.plot(
                gliders['longitude'].iloc[-1],
                gliders['latitude'].iloc[-1],
                marker='^',
                color='red',
                markersize=config['plotting']['glider_marker_size'],
                markeredgecolor='black',
                transform=DATA_PROJECTION,
                zorder=10001,
                label='Latest Position',
                linestyle="None"
            )
            ax.plot(
                gliders['longitude'],
                gliders['latitude'],
                color='red',
                linewidth=config['plotting']['glider_track_linewidth'],
                transform=DATA_PROJECTION,
                zorder=10000,
                label='Redwing Track'
            )

            # # Create custom legend
            # custom_handles = [
            #     h[0],  # Your PLOCAN point
            #     plt.Line2D([], [], color='red', linewidth=2, label='EEZ'), # Your EEZ line
            # ]

            if add_legend:
                ax.legend(loc='lower left', fontsize=10).set_zorder(100000)
        
        # Plot waypoint if available (as a target/bullseye)
        if waypoint is not None:
            wp_lon = waypoint['longitude']
            wp_lat = waypoint['latitude']
            base_size = config['plotting']['waypoint_marker_size']
            
            # Outer ring (black)
            ax.plot(
                wp_lon, wp_lat,
                marker='o',
                color='black',
                markersize=base_size * 1.8,
                transform=DATA_PROJECTION,
                zorder=10002,
                linestyle="None"
            )
            # Middle ring (white)
            ax.plot(
                wp_lon, wp_lat,
                marker='o',
                color='white',
                markersize=base_size * 1.3,
                transform=DATA_PROJECTION,
                zorder=10003,
                linestyle="None"
            )
            # Inner ring (red)
            ax.plot(
                wp_lon, wp_lat,
                marker='o',
                color='red',
                markersize=base_size * 0.8,
                transform=DATA_PROJECTION,
                zorder=10004,
                linestyle="None"
            )
            # Center bullseye (white)
            ax.plot(
                wp_lon, wp_lat,
                marker='o',
                color='white',
                markersize=base_size * 0.3,
                transform=DATA_PROJECTION,
                zorder=10005,
                linestyle="None"
            )

            h = ax.plot(
                PLOCAN_LOCATION["longitude"],
                PLOCAN_LOCATION["latitude"],
                marker='o',
                color='lime',
                markersize=6,
                markeredgecolor='black',
                transform=DATA_PROJECTION,
                zorder=10002,
                # label=PLOCAN_LOCATION["name"],
                linestyle="None"
            )

            he = map_add_eez(ax, color='white', linewidth=2)
            he.set_zorder(10000)
            
            if add_legend:
                # Create custom legend entry for the target
                # Plot invisible proxy artists for the legend
                
                # Get existing legend handles and labels
                handles, labels = ax.get_legend_handles_labels()
                
                # Create composite target for legend using multiple overlaid markers
                target_handles = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                           markersize=base_size * 1.2, linestyle='None', markeredgecolor='none'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                           markersize=base_size * 0.9, linestyle='None', markeredgecolor='none'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                           markersize=base_size * 0.6, linestyle='None', markeredgecolor='none'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                           markersize=base_size * 0.2, linestyle='None', markeredgecolor='none'),
                ]
                
                # Add the target as a tuple (matplotlib will overlay them with custom handler)
                handles.append(tuple(target_handles))
                labels.append('Waypoint')

                handles.append(h[0])
                labels.append(PLOCAN_LOCATION["name"])

                handles.append(plt.Line2D([], [], color='white', linewidth=2))
                labels.append('EEZ')
                
                ax.legend(handles, labels, loc='upper right', fontsize=10, 
                         handler_map={tuple: TargetHandler()}).set_zorder(100000)
        
        if add_text and latest_glider_text:
            ax.text(
                # 0.98, 0.02, #to the left of the plot
                0.275, 0.02,  # to the right of the plot
                f'Latest glider profile: {latest_glider_text}',
                transform=ax.transAxes,
                fontsize=6,
                fontweight='bold',
                ha='right',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        
        # Configure axis ticks
        # [-72.25, -44.75, 34.75, 45.25]
        # ax.set_xticks(np.arange(-70+15, -44+15, 5), crs=ccrs.PlateCarree())
        # ax.set_yticks(np.arange(35, 45, 5), crs=ccrs.PlateCarree())
        # ax.set_xticks(np.arange(-72+15, -44+15, 1), minor=True, crs=ccrs.PlateCarree())
        # ax.set_yticks(np.arange(35-3, 45-2, 1), minor=True, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-40, -5, 5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(25, 41, 5), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-40, -10, 1), minor=True, crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(25, 40, 1), minor=True, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        ax.tick_params(
            axis='both', which='major',
            labelsize=12, direction='out',
            length=6, width=1,
            top=True, right=True
        )
        ax.tick_params(
            axis='both', which='minor',
            direction='out', length=3, width=1,
            top=True, right=True
        )

    def render_and_save_map(
        ds_plot: Optional[xr.Dataset],
        title_text: str,
        slug_suffix: str,
        add_legend: bool,
        add_text: bool,
        show_colorbar: bool = True
    ):
        """Render a single map and save to file."""
        # Get actual generation time for this specific image
        actual_generated_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=MAP_PROJECTION)
        init_axis(ax, add_legend=add_legend, add_text=add_text)
        
        m = None
        if ds_plot is not None:
            try:
                _, mag = uv2spdir(ds_plot['u'], ds_plot['v'])
                m = ax.contourf(ds_plot["lon"], ds_plot["lat"], mag, **qargs)
            except Exception as error:
                logger.error(f"Contour failed for {slug_suffix}: {error}")

            map_add_currents(
                ax, ds_plot,
                density=stream_config['density'],
                linewidth=stream_config['linewidth'],
                color=stream_config['color']
            )
            
            if show_colorbar and m is not None:
                cb = fig.colorbar(m, ax=ax, orientation="vertical", shrink=0.6, aspect=20)
                cb.ax.tick_params(labelsize=12)
                cb.set_label('Current Speed (cm/s)', fontsize=12, fontweight='bold')
                cb.formatter = FormatStrFormatter('%.0f')
                cb.update_ticks()

        ax.set_title(title_text, fontsize=18, fontweight='bold')

        # Generate filename
        safe_time_val = (title_time or actual_generated_time).replace(':', '').replace(' ', '_')
        filename = f"redwing_{safe_time_val}_{slug_suffix}.png"

        # Determine save directory for timestamped file
        model_dir = Path(path_save)
        if model_ds is not None and hasattr(model_ds, 'attrs') and model_ds.attrs.get('model'):
            model_dir = model_dir / str(model_ds.attrs['model'])
        model_dir.mkdir(parents=True, exist_ok=True)

        save_file = model_dir / filename
        
        # Alias file is always one directory up from timestamped file
        alias_file = model_dir.parent / f"redwing_latest_{slug_suffix}.png"

        fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        shutil.copyfile(save_file, alias_file)

        plt.close(fig)
        logger.info(f"Saved map: {save_file}")
        logger.info(f"Saved alias: {alias_file}")

    # Determine title time
    if ds_time is not None:
        title_time = ds_time.strftime("%Y-%m-%dT%HZ")
    elif latest_glider_text:
        title_time = latest_glider_text
    else:
        title_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')

    # Generate surface current map
    if surface_ds is not None:
        model_title = model_label
        if 'surface' not in model_title.lower():
            model_title = f'{model_title} Surface Currents'
        title_str = f"Sentinel Mission - Redwing Landing Area\n {model_title} - {title_time}"
    else:
        title_str = f"Sentinel Mission - Redwing Landing Area\n Latest Update - {title_time}"

    render_and_save_map(
        surface_ds,
        title_str,
        slug,
        add_legend=True,
        add_text=True,
        show_colorbar=(surface_ds is not None)
    )

    # Generate depth-averaged map if available
    if depth_avg_ds is not None:
        depth_slug = f"{slug}-depthavg"
        min_d = config['depth_average']['min_depth']
        max_d = config['depth_average']['max_depth']
        depth_title = (
            f"Sentinel Mission - Redwing Landing Area\n"
            f"{model_label} Depth-Averaged ({min_d}-{max_d} m) - {title_time}"
        )
        render_and_save_map(
            depth_avg_ds,
            depth_title,
            depth_slug,
            add_legend=True,
            add_text=True,
            show_colorbar=True
        )


def subset_model_data(
    model_ds: xr.Dataset,
    extent: list,
    grid_info: Optional[Dict] = None,
    model_name: str = "Model"
) -> xr.Dataset:
    """
    Subset model dataset to specified extent.
    
    Parameters
    ----------
    model_ds : xr.Dataset
        Full model dataset
    extent : list
        Geographic extent [lon_min, lon_max, lat_min, lat_max]
    grid_info : dict, optional
        Grid information for RTOFS-style indexing
    model_name : str, optional
        Model name to add as attribute
        
    Returns
    -------
    xr.Dataset
        Subsetted dataset
    """
    extent_data = expand_extent(extent, buffer=1.0)
    
    # Handle RTOFS-style indexing with grid info
    if grid_info is not None:
        lons_ind = np.interp(extent_data[:2], grid_info['lons'], grid_info['x'])
        lats_ind = np.interp(extent_data[2:], grid_info['lats'], grid_info['y'])
        
        extent_ind = [
            int(np.floor(lons_ind[0])),
            int(np.ceil(lons_ind[1])),
            int(np.floor(lats_ind[0])),
            int(np.ceil(lats_ind[1]))
        ]
        
        subset = model_ds.sel(
            x=slice(extent_ind[0], extent_ind[1]),
            y=slice(extent_ind[2], extent_ind[3])
        )
    else:
        # Handle standard lat/lon indexing
        # Try multiple coordinate name conventions
        lon_coords = ['lon', 'longitude']
        lat_coords = ['lat', 'latitude']
        
        lon_key = next((c for c in lon_coords if c in model_ds.coords), None)
        lat_key = next((c for c in lat_coords if c in model_ds.coords), None)
        
        if lon_key and lat_key:
            # Check if longitude is in 0-360 range
            lon_max = float(model_ds[lon_key].max())
            is_360 = lon_max > 180
            
            if is_360:
                # Convert extent to 0-360
                lon_min_360 = extent_data[0] % 360
                lon_max_360 = extent_data[1] % 360
                
                logger.info(f"{model_name} subsetting: lon=[{lon_min_360:.2f}, {lon_max_360:.2f}], lat=[{extent_data[2]:.2f}, {extent_data[3]:.2f}]")
                
                # Use .where().dropna() for more robust subsetting
                subset = model_ds.where(
                    (model_ds[lon_key] >= lon_min_360) & 
                    (model_ds[lon_key] <= lon_max_360) &
                    (model_ds[lat_key] >= extent_data[2]) & 
                    (model_ds[lat_key] <= extent_data[3]),
                    drop=True
                )
                
                logger.info(f"{model_name} after subset: lon size={subset[lon_key].size}, lat size={subset[lat_key].size}")
                
                # Convert longitude back to -180:180
                new_lon = lon360to180(subset[lon_key].values)
                subset = subset.assign_coords({lon_key: new_lon})
                
            else:
                logger.info(f"{model_name} subsetting: lon=[{extent_data[0]:.2f}, {extent_data[1]:.2f}], lat=[{extent_data[2]:.2f}, {extent_data[3]:.2f}]")
                
                subset = model_ds.where(
                    (model_ds[lon_key] >= extent_data[0]) & 
                    (model_ds[lon_key] <= extent_data[1]) &
                    (model_ds[lat_key] >= extent_data[2]) & 
                    (model_ds[lat_key] <= extent_data[3]),
                    drop=True
                )
                
                logger.info(f"{model_name} after subset: lon size={subset[lon_key].size}, lat size={subset[lat_key].size}")
            
            # Standardize coordinate names
            if lon_key != 'lon':
                subset = subset.rename({lon_key: 'lon'})
            if lat_key != 'lat':
                subset = subset.rename({lat_key: 'lat'})
        else:
            logger.warning(f"Could not find standard lat/lon coordinates in {model_name}")
            subset = model_ds
    
    subset.attrs['model'] = model_name
    return subset


def process_and_plot_time(
    reference_time: dt.datetime,
    config: Dict,
    bathy_data: Optional[xr.Dataset],
    glider_data: pd.DataFrame,
    waypoint_data: Optional[Dict] = None,
    rtofs_data: Optional[Tuple[xr.Dataset, Dict]] = None,
    espc_data: Optional[Tuple[xr.Dataset, bool]] = None,
    cmems_data: Optional[xr.Dataset] = None,
    lusitania_data: Optional[Tuple[xr.Dataset, bool]] = None
):
    """
    Process all model data for a given time and generate plots.
    
    Parameters
    ----------
    reference_time : datetime
        Time to process
    config : dict
        Configuration dictionary
    bathy_data : xr.Dataset or None
        Bathymetry dataset (None if disabled)
    glider_data : pd.DataFrame
        Glider track data
    waypoint_data : dict, optional
        Waypoint information
    rtofs_data : tuple, optional
        (RTOFS dataset, grid info)
    espc_data : tuple, optional
        (ESPC dataset, is_archive flag)
    cmems_data : xr.Dataset, optional
        CMEMS dataset
    """
    logger.info(f"Processing time: {reference_time}")
    
    extent = config['region']['extent']
    extent_data = expand_extent(extent, buffer=1.0)
    
    # Subset bathymetry if available and enabled
    bathy = None
    if bathy_data is not None and config['bathymetry'].get('enabled', True):
        # Check bathymetry longitude convention
        bathy_lon_max = float(bathy_data.longitude.max())
        if bathy_lon_max > 180:
            # Bathymetry uses 0-360, convert extent
            bathy_lon_slice = slice(
                (extent_data[0] - 1) % 360,
                (extent_data[1] + 1) % 360
            )
        else:
            bathy_lon_slice = slice(extent_data[0] - 1, extent_data[1] + 1)
        
        bathy = bathy_data.sel(
            longitude=bathy_lon_slice,
            latitude=slice(extent_data[2] - 1, extent_data[3] + 1)
        )
        
        # Convert bathy longitude to -180:180 if needed
        if bathy_lon_max > 180:
            bathy = bathy.assign_coords(longitude=lon360to180(bathy.longitude.values))
        
        logger.info(f"Bathymetry subset: lon={bathy.longitude.size}, lat={bathy.latitude.size}")
    else:
        logger.info("Bathymetry plotting disabled or data unavailable")
    
    # Process RTOFS
    if rtofs_data is not None:
        rds, grid_info = rtofs_data
        try:
            rds_time = rds.sel(time=reference_time, method="nearest")
            rds_slice = subset_model_data(
                rds_time, extent, grid_info=grid_info, model_name='RTOFS'
            )
            logger.info("RTOFS: Processing")
            plot_redwing_map(
                rds_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='RTOFS'
            )
        except Exception as error:
            logger.error(f"RTOFS processing failed: {error}")
    
    # Process ESPC
    if espc_data is not None:
        espc_ds, is_archive = espc_data
        try:
            espc_time = espc_ds.sel(time=reference_time, method="nearest")
            if 'time1' in espc_time.dims:
                espc_time = espc_time.sel(time1=reference_time, method="nearest")
            
            # === DIAGNOSTIC START ===
            logger.info(f"ESPC dims: {espc_time.dims}")
            logger.info(f"ESPC coords: {list(espc_time.coords)}")
            for coord in espc_time.coords:
                c = espc_time[coord]
                if c.ndim == 1 and c.size > 1:
                    logger.info(f"  {coord}: min={float(c.min()):.2f}, max={float(c.max()):.2f}, size={c.size}")
            logger.info(f"Target extent: {extent}")
            logger.info(f"Target lon360: {lon180to360(extent[:2])}")
            # === DIAGNOSTIC END ===
            
            espc_slice = subset_model_data(
                espc_time, extent, model_name='ESPC'
            )
            logger.info("ESPC: Processing")
            plot_redwing_map(
                espc_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='ESPC'
            )
        except Exception as error:
            logger.error(f"ESPC processing failed: {error}")
    
    # Process CMEMS
    if cmems_data is not None:
        try:
            u = cmems_data['u'].sel(time=reference_time, method="nearest")
            v = cmems_data['v'].sel(time=reference_time, method="nearest")
            
            cds_time = xr.Dataset({'u': u, 'v': v})
            cds_slice = subset_model_data(
                cds_time, extent, model_name='Copernicus'
            )
            logger.info("CMEMS: Processing")
            plot_redwing_map(
                cds_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='Copernicus'
            )
        except Exception as error:
            logger.error(f"CMEMS processing failed: {error}")

    # Process Lusitania
    if lusitania_data is not None:
        lusitania_ds, is_archive = lusitania_data
        try:
            lusitania_time = lusitania_ds.sel(time=reference_time, method="nearest")
            
            # === DIAGNOSTIC START ===
            logger.info(f"Lusitania dims: {lusitania_time.dims}")
            logger.info(f"Lusitania coords: {list(lusitania_time.coords)}")
            for coord in lusitania_time.coords:
                c = lusitania_time[coord]
                if c.ndim == 1 and c.size > 1:
                    logger.info(f"  {coord}: min={float(c.min()):.2f}, max={float(c.max()):.2f}, size={c.size}")
            logger.info(f"Target extent: {extent}")
            # logger.info(f"Target lon360: {lon180to360(extent[:2])}")
            # === DIAGNOSTIC END ===
            
            lusitania_slice = subset_model_data(
                lusitania_time, extent, model_name='Lusitania'
            )
            logger.info("Lusitania: Processing")
            plot_redwing_map(
                lusitania_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='Lusitania'
            )
        except Exception as error:
            logger.error(f"Lusitania processing failed: {error}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    start_time = time.time()
    
    logger.info("Starting Redwing plotting script")
    logger.info(f"Configuration: {CONFIG['models']}")
    logger.info(f"Bathymetry enabled: {CONFIG['bathymetry'].get('enabled', True)}")
    
    # Validate model flags
    models_enabled = CONFIG['models']
    if not models_enabled['plot_model_data']:
        models_enabled['plot_rtofs'] = False
        models_enabled['plot_espc'] = False
        models_enabled['plot_cmems'] = False
        models_enabled['plot_lusitania'] = False
    
    # Load glider data
    logger.info(f"Fetching glider data for: {CONFIG['glider']['id']}")
    glider_data, waypoint_data = fetch_glider_surfacings(
        CONFIG['glider']['id'],
        CONFIG['glider']['api_url'],
        CONFIG['glider']['api_timeout']
    )
    
    # Determine reference date
    if not glider_data.empty:
        latest_glider_date = glider_data.index.max()
        logger.info(f"Latest glider date: {latest_glider_date}")
        reference_date = latest_glider_date
    else:
        reference_date = dt.datetime.now(dt.timezone.utc)
        logger.warning("No glider data available, using current time")
    
    # Load bathymetry (if enabled)
    bathy_data = None
    if CONFIG['bathymetry'].get('enabled', True):
        logger.info("Loading bathymetry data")
        bathy_data = load_bathymetry(CONFIG['paths']['bathy_file'])
        if bathy_data is None:
            logger.warning("Bathymetry loading failed, continuing without bathymetry")
    else:
        logger.info("Bathymetry plotting disabled in config")
    
    # Load model data if requested
    rtofs_data = None
    espc_data = None
    cmems_data = None
    lusitania_data = None  # <-- FIX 1: Initialize to None
    
    if models_enabled['plot_rtofs']:
        logger.info("Loading RTOFS data")
        rtofs_data = load_rtofs(CONFIG['region']['extent'])
    
    if models_enabled['plot_espc']:
        logger.info("Loading ESPC data")
        espc_data = load_espc(CONFIG['region']['extent'], reference_date)
    
    if models_enabled['plot_cmems']:
        logger.info("Loading CMEMS data")
        cmems_data = load_cmems(CONFIG['region']['extent'])

    if models_enabled['plot_lusitania']:
        logger.info("Loading Lusitania data")
        lusitania_data = load_lusitania(reference_date)
    
    # Check if any models were successfully loaded
    models_available = any([
        rtofs_data is not None and rtofs_data[0] is not None,
        espc_data is not None and espc_data[0] is not None,
        cmems_data is not None,
        lusitania_data is not None and lusitania_data[0] is not None  # <-- FIX 2: Add tuple check
    ])
    
    if models_enabled['plot_model_data'] and models_available:
        # Process and plot with model data
        process_and_plot_time(
            reference_date,
            CONFIG,
            bathy_data,
            glider_data,
            waypoint_data=waypoint_data,
            rtofs_data=rtofs_data,
            espc_data=espc_data,
            cmems_data=cmems_data,
            lusitania_data=lusitania_data
        )
    else:
        # Plot track only without model data
        logger.info("Generating track-only map")
        extent_data = expand_extent(CONFIG['region']['extent'], buffer=1.0)
        
        bathy = None
        if bathy_data is not None and CONFIG['bathymetry'].get('enabled', True):
            bathy_lon_max = float(bathy_data.longitude.max())
            if bathy_lon_max > 180:
                bathy_lon_slice = slice(
                    (extent_data[0] - 1) % 360,
                    (extent_data[1] + 1) % 360
                )
            else:
                bathy_lon_slice = slice(extent_data[0] - 1, extent_data[1] + 1)
            
            bathy = bathy_data.sel(
                longitude=bathy_lon_slice,
                latitude=slice(extent_data[2] - 1, extent_data[3] + 1)
            )
            
            if bathy_lon_max > 180:
                bathy = bathy.assign_coords(longitude=lon360to180(bathy.longitude.values))
        
        plot_redwing_map(
            model_ds=None,
            bathy=bathy,
            gliders=glider_data,
            waypoint=waypoint_data,
            config=CONFIG,
            model_name=None
        )
    
    elapsed = time.time() - start_time
    logger.info(f'Execution completed in {elapsed:.2f} seconds')


if __name__ == "__main__":
    main()