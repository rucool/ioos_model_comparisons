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
except ImportError:
    gpd = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'paths': {
        'save_path': '/Users/mikesmith/Documents/',
        # 'save_path': '/www/web/rucool/media/',
        'eez_path': '/Users/mikesmith/Downloads/World_Exclusive_Economic_Zones_Boundaries-shp/World_Exclusive_Economic_Zones_Boundaries.shp'
        # 'eez_path': '/home/hurricaneadm/data/World_Exclusive_Economic_Zones_Boundaries-shp/World_Exclusive_Economic_Zones_Boundaries.shp',
    },
    'models': {
        'plot_model_data': True,
        'plot_rtofs': True,
        'plot_espc': True,
        'plot_cmems': True,
    },
    'glider': {
        'name': "ru29",  # Glider platform name; most recent deployment is resolved automatically
        'deployments_url': "https://marine.rutgers.edu/cool/data/gliders/api/deployments/",
        'api_url': "https://marine.rutgers.edu/cool/data/gliders/api/surfacings/",
        'api_timeout': 30,
    },
    'region': {
        'name': "Tropical Western Atlantic",
        'folder': "tropical_western_atlantic",
        'extent': [-70.25, -40.75, 0, 25],
    },
    'currents': {
        'enabled': True,
        'depths': [0, 100, 150, 200],
        'limits': [0, 120, 10],
        'limits_depth_avg': [0, 40, 5],
        'auto_colorbar': False,  # If True, derive colorbar limits from data; if False, use 'limits'
        'streamplot': {
            'density': 3,
            'linewidth': 0.5,
            'color': 'black',
        }
    },
    'depth_average': {
        'min_depth': 0,
        'max_depth': 1000,
        'depth_step': 1,
    },
    'plotting': {
        'figsize': (12, 8.5),
        'dpi': 300,
        'legend_columns': 9,
        'glider_track_linewidth': 4,
        'glider_marker_size': 10,
        'waypoint_marker_size': 6,
        'show_waypoint': False,
    },
    'zoom': {
        'enabled': True,
        'lon_min': -62,
        'lon_max': -55,
        'lat_buffer': 3.0,  # degrees above/below latest glider latitude
        'lat_max': 16,      # northern bound is at least this value
        'figsize': (10, 10),
    },
    'bathymetry': {
        'enabled': False,
        'contour_levels': (-1000, -100),
        'filled_levels': [-8000, -1000, -100, 0],
        'filled_colors': ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue'],
    }
}

# Projections
MAP_PROJECTION = ccrs.Mercator()
DATA_PROJECTION = ccrs.PlateCarree()


# ============================================================================
# CUSTOM LEGEND HANDLER FOR TARGET MARKER
# ============================================================================

class TargetHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x_center = width / 2.0
        y_center = height / 2.0
        artists = []
        for handle in orig_handle:
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
# UTILITY FUNCTIONS
# ============================================================================

def lon180to360(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return np.mod(array, 360)


def lon360to180(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return np.mod(array + 180, 360) - 180


def ddmm_to_degrees(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return np.nan
    val_float = float(val)
    sign = -1 if val_float < 0 else 1
    v = abs(val_float)
    degrees = int(v // 100)
    minutes = v - 100 * degrees
    return sign * (degrees + minutes / 60.0)


def expand_extent(extent: list, buffer: float = 1.0) -> list:
    return np.add(extent, [-buffer, buffer, -buffer, buffer]).tolist()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def resolve_latest_deployment(
    glider_name: str,
    deployments_url: str,
    timeout: int = 30
) -> str:
    """
    Query the COOL glider deployments API and return the deployment_name of the
    most recent deployment for the given glider platform.

    Parameters
    ----------
    glider_name : str
        Glider platform name (e.g. 'ru29')
    deployments_url : str
        URL for the deployments API endpoint
    timeout : int, optional
        Request timeout in seconds

    Returns
    -------
    str
        Most recent deployment name (e.g. 'ru29-20260623T2102')
    """
    try:
        response = requests.get(deployments_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch deployments from {deployments_url}: {exc}")

    records = payload.get("data", payload) if isinstance(payload, dict) else payload
    matches = [
        r for r in records
        if r.get("glider_name", "").lower() == glider_name.lower()
    ]

    if not matches:
        raise ValueError(f"No deployments found for glider '{glider_name}'")

    latest = max(matches, key=lambda r: r.get("start_date_epoch") or 0)
    deployment_name = latest["deployment_name"]
    logger.info(f"Resolved latest deployment for '{glider_name}': {deployment_name}")
    return deployment_name

def fetch_glider_surfacings(
    deployment_id: str,
    base_url: str,
    timeout: int = 30
) -> Tuple[pd.DataFrame, Optional[Dict]]:
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

    if "gps_lat_degrees" in df.columns:
        lat = pd.to_numeric(df["gps_lat_degrees"], errors="coerce")
    else:
        lat_raw = df.get("gps_lat")
        lat = lat_raw.map(ddmm_to_degrees) if lat_raw is not None else np.nan
    df["latitude"] = lat

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

    waypoint_info = None
    waypoint_cols = ["waypoint_lat", "waypoint_lon", "waypoint_bearing_degrees", "waypoint_range_meters"]

    if all(col in df.columns for col in waypoint_cols):
        for idx in df.index[::-1]:
            row = df.loc[idx]
            all_valid = True
            for col in waypoint_cols:
                val = row[col]
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    all_valid = False
                    break
                if pd.isna(val) or val == '' or val is None:
                    all_valid = False
                    break

            if all_valid:
                wp_lat_raw = row["waypoint_lat"]
                wp_lon_raw = row["waypoint_lon"]
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
                               f"bearing={waypoint_info['bearing']:.1f}, "
                               f"range={waypoint_info['range']:.0f}m")
                    break
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse waypoint at {idx}: {e}")
                    continue

    if waypoint_info is None:
        logger.info("No waypoint data found in surfacings")

    return df[["latitude", "longitude"]], waypoint_info


def load_bathymetry(extent: list) -> Optional[xr.Dataset]:
    try:
        from ioos_model_comparisons.platforms import get_bathymetry
        ds = get_bathymetry(bbox=extent)
        logger.info("GEBCO bathymetry loaded successfully")
        return ds
    except Exception as exc:
        logger.warning(f"Failed to load GEBCO bathymetry: {exc}")
        return None


def load_rtofs(extent: list) -> Tuple[Optional[xr.Dataset], Optional[Dict]]:
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
    if ds is None:
        return None
    if not isinstance(ds, xr.Dataset):
        logger.warning("Depth-average skipped: dataset expected.")
        return None
    missing = [var for var in ("u", "v") if var not in ds]
    if missing:
        logger.warning(f"Depth-average skipped: missing variables {missing}.")
        return None

    candidate_dims = [depth_dim_hint, "depth", "Depth", "depthu", "depthv", "z", "lev", "level"]
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


def regrid_curvilinear(ds: Optional[xr.Dataset], resolution: float = 0.25) -> Optional[xr.Dataset]:
    if ds is None:
        return ds
    lon = ds['lon'].values
    lat = ds['lat'].values
    if lon.ndim == 1:
        return ds

    from scipy.interpolate import griddata

    lon_reg = np.arange(float(np.nanmin(lon)), float(np.nanmax(lon)) + resolution, resolution)
    lat_reg = np.arange(float(np.nanmin(lat)), float(np.nanmax(lat)) + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_reg, lat_reg)
    src_pts = np.column_stack([lon.ravel(), lat.ravel()])

    new_vars = {}
    for var in ds.data_vars:
        vals = ds[var].values
        if vals.ndim != 2:
            continue
        flat = vals.ravel()
        mask = np.isfinite(flat) & np.isfinite(src_pts[:, 0]) & np.isfinite(src_pts[:, 1])
        if mask.sum() < 4:
            continue
        new_vars[var] = xr.DataArray(
            griddata(src_pts[mask], flat[mask], (lon_grid, lat_grid), method='linear'),
            dims=['lat', 'lon']
        )

    new_ds = xr.Dataset(new_vars, coords={'lon': lon_reg, 'lat': lat_reg})
    new_ds.attrs = ds.attrs
    return new_ds


def map_add_currents(
    ax,
    ds: xr.Dataset,
    density: int = 2,
    linewidth: float = 0.75,
    color: str = 'black',
    transform=DATA_PROJECTION
):
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

def plot_twa_map(
    model_ds: Optional[xr.Dataset] = None,
    bathy: Optional[xr.Dataset] = None,
    gliders: Optional[pd.DataFrame] = None,
    waypoint: Optional[Dict] = None,
    config: Dict = None,
    path_save: str = None,
    model_name: Optional[str] = None
):
    if config is None:
        config = CONFIG

    if path_save is None:
        path_save = config['paths']['save_path']

    Path(path_save).mkdir(parents=True, exist_ok=True)

    figsize = config['plotting']['figsize']
    dpi = config['plotting']['dpi']
    extent = config['region']['extent']
    bathy_config = config['bathymetry']
    bathy_enabled = bathy_config.get('enabled', True) and bathy is not None
    stream_config = config['currents']['streamplot']

    ds_time = None
    surface_ds = None
    depth_avg_ds = None
    latest_glider_text = None
    model_label = model_name

    if gliders is not None and hasattr(gliders, "empty") and not gliders.empty:
        latest_glider_ts = pd.to_datetime(gliders.index.max())
        latest_glider_text = latest_glider_ts.strftime('%Y-%m-%d %H:%MZ')

    if model_ds is not None:
        ds_time = pd.to_datetime(model_ds.time.data)
        logger.info("Plotting currents @ 0m")
        surface_ds = model_ds.sel(depth=0, method='nearest')

        logger.info(f"model_ds dims: {model_ds.dims}")
        logger.info(f"model_ds u shape: {model_ds['u'].shape}")
        logger.info(f"surface_ds dims: {surface_ds.dims}")
        logger.info(f"surface_ds u shape: {surface_ds['u'].shape}")
        logger.info(f"surface_ds lon shape: {surface_ds['lon'].shape}")
        logger.info(f"surface_ds lat shape: {surface_ds['lat'].shape}")

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

    slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in model_label)
    slug = '-'.join(filter(None, slug.split('-')))
    if not slug:
        slug = 'track'

    auto_colorbar = config['currents'].get('auto_colorbar', False)
    qargs_surface = {
        'transform': DATA_PROJECTION,
        'cmap': cmocean.cm.speed,
        'extend': "max",
    }
    qargs_depth_avg = {
        'transform': DATA_PROJECTION,
        'cmap': cmocean.cm.speed,
        'extend': "max",
    }
    if not auto_colorbar:
        lim = config['currents']['limits']
        qargs_surface['levels'] = np.arange(lim[0], lim[1], lim[2])
        lim_da = config['currents'].get('limits_depth_avg', lim)
        qargs_depth_avg['levels'] = np.arange(lim_da[0], lim_da[1], lim_da[2])

    def init_axis(ax, plot_extent, add_legend: bool = False):
        create(plot_extent, ax=ax, ticks=False)

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
                label='RU29 Track'
            )

        he = map_add_eez(ax, color='white', linewidth=2)
        he.set_zorder(10000)

        if config['plotting']['show_waypoint'] and waypoint is not None:
            wp_lon = waypoint['longitude']
            wp_lat = waypoint['latitude']
            base_size = config['plotting']['waypoint_marker_size']

            ax.plot(wp_lon, wp_lat, marker='o', color='black',
                    markersize=base_size * 1.8, transform=DATA_PROJECTION,
                    zorder=10002, linestyle="None")
            ax.plot(wp_lon, wp_lat, marker='o', color='white',
                    markersize=base_size * 1.3, transform=DATA_PROJECTION,
                    zorder=10003, linestyle="None")
            ax.plot(wp_lon, wp_lat, marker='o', color='red',
                    markersize=base_size * 0.8, transform=DATA_PROJECTION,
                    zorder=10004, linestyle="None")
            ax.plot(wp_lon, wp_lat, marker='o', color='white',
                    markersize=base_size * 0.3, transform=DATA_PROJECTION,
                    zorder=10005, linestyle="None")

            if add_legend:
                handles, labels = ax.get_legend_handles_labels()
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
                handles.append(tuple(target_handles))
                labels.append('Waypoint')
                handles.append(plt.Line2D([], [], color='white', linewidth=2))
                labels.append('EEZ')
                ax.legend(handles, labels, loc='upper left', fontsize=10,
                         handler_map={tuple: TargetHandler()}).set_zorder(100000)
        else:
            if add_legend:
                handles, labels = ax.get_legend_handles_labels()
                handles.append(plt.Line2D([], [], color='white', linewidth=2))
                labels.append('EEZ')
                ax.legend(handles, labels, loc='upper left', fontsize=10).set_zorder(100000)

        # Compute tick intervals based on the extent size
        lon_span = plot_extent[1] - plot_extent[0]
        lat_span = plot_extent[3] - plot_extent[2]
        span = max(lon_span, lat_span)
        if span <= 8:
            major_step, minor_step = 1, 0.25
        elif span <= 20:
            major_step, minor_step = 2, 0.5
        else:
            major_step, minor_step = 5, 1

        ax.set_xticks(np.arange(np.floor(plot_extent[0]), plot_extent[1] + 1e-6, major_step), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(np.floor(plot_extent[2]), plot_extent[3] + 1e-6, major_step), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(np.floor(plot_extent[0]), plot_extent[1] + 1e-6, minor_step), minor=True, crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(np.floor(plot_extent[2]), plot_extent[3] + 1e-6, minor_step), minor=True, crs=ccrs.PlateCarree())
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
        show_colorbar: bool = True,
        plot_extent: Optional[list] = None,
        plot_figsize: Optional[tuple] = None,
        contour_args: Optional[dict] = None,
    ):
        actual_generated_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')
        active_extent = list(plot_extent if plot_extent is not None else extent)
        active_figsize = plot_figsize if plot_figsize is not None else figsize
        active_qargs = contour_args if contour_args is not None else qargs_surface

        # Clip plot extent to actual data coverage to avoid blank ocean background
        if ds_plot is not None:
            valid_lon = ds_plot['lon'].values[np.isfinite(ds_plot['lon'].values)]
            valid_lat = ds_plot['lat'].values[np.isfinite(ds_plot['lat'].values)]
            if valid_lon.size and valid_lat.size:
                active_extent[0] = max(active_extent[0], float(valid_lon.min()))
                active_extent[1] = min(active_extent[1], float(valid_lon.max()))
                active_extent[2] = max(active_extent[2], float(valid_lat.min()))
                active_extent[3] = min(active_extent[3], float(valid_lat.max()))

        fig = plt.figure(figsize=active_figsize)
        ax = fig.add_subplot(1, 1, 1, projection=MAP_PROJECTION)
        init_axis(ax, active_extent, add_legend=add_legend)

        m = None
        if ds_plot is not None:
            ds_plot = regrid_curvilinear(ds_plot)
            try:
                _, mag = uv2spdir(ds_plot['u'], ds_plot['v'])
                m = ax.contourf(ds_plot["lon"], ds_plot["lat"], mag, **active_qargs)
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
                if auto_colorbar:
                    cb.formatter = FormatStrFormatter('%.2f')
                else:
                    cb.formatter = FormatStrFormatter('%.0f')
                cb.update_ticks()

        ax.set_title(title_text, fontsize=18, fontweight='bold')
        ax.text(
            1.0, -0.07,
            f'Image generated: {actual_generated_time}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=7, fontweight='bold'
        )

        safe_time_val = (title_time or actual_generated_time).replace(':', '').replace(' ', '_')
        glider_name = config['glider']['name']
        filename = f"{glider_name}_twa_{safe_time_val}_{slug_suffix}.png"

        glider_dir = Path(path_save) / glider_name
        model_dir = glider_dir
        if model_ds is not None and hasattr(model_ds, 'attrs') and model_ds.attrs.get('model'):
            model_dir = glider_dir / str(model_ds.attrs['model'])
        model_dir.mkdir(parents=True, exist_ok=True)

        save_file = model_dir / filename
        alias_file = glider_dir / f"{glider_name}_twa_latest_{slug_suffix}.png"

        fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.03)
        shutil.copyfile(save_file, alias_file)

        plt.close(fig)
        logger.info(f"Saved map: {save_file}")
        logger.info(f"Saved alias: {alias_file}")

    if ds_time is not None:
        title_time = ds_time.strftime("%Y-%m-%dT%HZ")
    elif latest_glider_text:
        title_time = latest_glider_text
    else:
        title_time = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')

    region_name = config['region']['name']
    model_title = model_label
    if surface_ds is not None:
        if 'surface' not in model_title.lower():
            model_title = f'{model_title} Surface Currents'
        title_str = f"{region_name} - RU29 Track\n {model_title} - {title_time}"
    else:
        title_str = f"{region_name} - RU29 Track\n Latest Update - {title_time}"

    render_and_save_map(
        surface_ds,
        title_str,
        slug,
        add_legend=True,
        show_colorbar=(surface_ds is not None)
    )

    if depth_avg_ds is not None:
        depth_slug = f"{slug}-depthavg"
        min_d = config['depth_average']['min_depth']
        max_d = config['depth_average']['max_depth']
        depth_title = (
            f"{region_name} - RU29 Track\n"
            f"{model_label} Depth-Averaged ({min_d}-{max_d} m) - {title_time}"
        )
        render_and_save_map(
            depth_avg_ds,
            depth_title,
            depth_slug,
            add_legend=True,
            show_colorbar=True,
            contour_args=qargs_depth_avg,
        )

    # Zoom maps centered on latest glider position
    zoom_config = config.get('zoom', {})
    if zoom_config.get('enabled', False) and gliders is not None and not gliders.empty:
        lat0 = gliders['latitude'].iloc[-1]
        lat_buf = zoom_config.get('lat_buffer', 3.0)
        lat_north = max(lat0 + lat_buf, zoom_config.get('lat_max', lat0 + lat_buf))
        zoom_extent = [
            zoom_config.get('lon_min', -62),
            zoom_config.get('lon_max', -55),
            lat0 - lat_buf,
            lat_north,
        ]
        zoom_figsize = zoom_config.get('figsize', (10, 10))
        logger.info(f"Generating zoom maps: extent={zoom_extent}")

        if surface_ds is not None:
            zoom_surface_title = (
                f"RU29 - {model_title}\n{title_time}"
            )
            render_and_save_map(
                surface_ds,
                zoom_surface_title,
                f"{slug}-zoom",
                add_legend=True,
                show_colorbar=True,
                plot_extent=zoom_extent,
                plot_figsize=zoom_figsize,
                contour_args=qargs_surface,
            )

        if depth_avg_ds is not None:
            zoom_depth_title = (
                f"RU29 - {model_label} Depth-Averaged ({min_d}-{max_d} m)\n{title_time}"
            )
            render_and_save_map(
                depth_avg_ds,
                zoom_depth_title,
                f"{slug}-depthavg-zoom",
                add_legend=True,
                show_colorbar=True,
                plot_extent=zoom_extent,
                plot_figsize=zoom_figsize,
                contour_args=qargs_depth_avg,
            )


def subset_model_data(
    model_ds: xr.Dataset,
    extent: list,
    grid_info: Optional[Dict] = None,
    model_name: str = "Model"
) -> xr.Dataset:
    extent_data = expand_extent(extent, buffer=1.0)

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
        lon_coords = ['lon', 'longitude']
        lat_coords = ['lat', 'latitude']
        lon_key = next((c for c in lon_coords if c in model_ds.coords), None)
        lat_key = next((c for c in lat_coords if c in model_ds.coords), None)

        if lon_key and lat_key:
            lon_max = float(model_ds[lon_key].max())
            is_360 = lon_max > 180

            if is_360:
                lon_min_360 = extent_data[0] % 360
                lon_max_360 = extent_data[1] % 360

                logger.info(f"{model_name} subsetting: lon=[{lon_min_360:.2f}, {lon_max_360:.2f}], lat=[{extent_data[2]:.2f}, {extent_data[3]:.2f}]")

                lat_mask = (
                    (model_ds[lat_key] >= extent_data[2]) &
                    (model_ds[lat_key] <= extent_data[3])
                )
                if lon_min_360 > lon_max_360:
                    lon_mask = (
                        (model_ds[lon_key] >= lon_min_360) |
                        (model_ds[lon_key] <= lon_max_360)
                    )
                else:
                    lon_mask = (
                        (model_ds[lon_key] >= lon_min_360) &
                        (model_ds[lon_key] <= lon_max_360)
                    )
                subset = model_ds.where(lon_mask & lat_mask, drop=True)

                logger.info(f"{model_name} after subset: lon size={subset[lon_key].size}, lat size={subset[lat_key].size}")

                new_lon = lon360to180(subset[lon_key].values)
                subset = subset.assign_coords({lon_key: new_lon})
                subset = subset.sortby(lon_key)
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

            if lon_key != 'lon':
                subset = subset.rename({lon_key: 'lon'})
            if lat_key != 'lat':
                subset = subset.rename({lat_key: 'lat'})

            in_bounds = (
                (subset['lon'] >= extent_data[0]) & (subset['lon'] <= extent_data[1]) &
                (subset['lat'] >= extent_data[2]) & (subset['lat'] <= extent_data[3])
            )
            subset = subset.where(in_bounds)
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
):
    logger.info(f"Processing time: {reference_time}")

    extent = config['region']['extent']
    extent_data = expand_extent(extent, buffer=1.0)

    bathy = None
    if bathy_data is not None and config['bathymetry'].get('enabled', True):
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

        logger.info(f"Bathymetry subset: lon={bathy.longitude.size}, lat={bathy.latitude.size}")
    else:
        logger.info("Bathymetry plotting disabled or data unavailable")

    if rtofs_data is not None:
        rds, grid_info = rtofs_data
        try:
            rds_time = rds.sel(time=reference_time, method="nearest")
            rds_slice = subset_model_data(
                rds_time, extent, grid_info=grid_info, model_name='RTOFS'
            )
            logger.info("RTOFS: Processing")
            plot_twa_map(
                rds_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='RTOFS'
            )
        except Exception as error:
            logger.error(f"RTOFS processing failed: {error}")

    if espc_data is not None:
        espc_ds, is_archive = espc_data
        try:
            espc_time = espc_ds.sel(time=reference_time, method="nearest")
            if 'time1' in espc_time.dims:
                espc_time = espc_time.sel(time1=reference_time, method="nearest")

            logger.info(f"ESPC dims: {espc_time.dims}")
            logger.info(f"ESPC coords: {list(espc_time.coords)}")
            for coord in espc_time.coords:
                c = espc_time[coord]
                if c.ndim == 1 and c.size > 1:
                    logger.info(f"  {coord}: min={float(c.min()):.2f}, max={float(c.max()):.2f}, size={c.size}")
            logger.info(f"Target extent: {extent}")
            logger.info(f"Target lon360: {lon180to360(extent[:2])}")

            espc_slice = subset_model_data(
                espc_time, extent, model_name='ESPC'
            )
            logger.info("ESPC: Processing")
            plot_twa_map(
                espc_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='ESPC'
            )
        except Exception as error:
            logger.error(f"ESPC processing failed: {error}")

    if cmems_data is not None:
        try:
            u = cmems_data['u'].sel(time=reference_time, method="nearest")
            v = cmems_data['v'].sel(time=reference_time, method="nearest")

            cds_time = xr.Dataset({'u': u, 'v': v})
            cds_slice = subset_model_data(
                cds_time, extent, model_name='Copernicus'
            )
            logger.info("CMEMS: Processing")
            plot_twa_map(
                cds_slice,
                bathy=bathy,
                gliders=glider_data,
                waypoint=waypoint_data,
                config=config,
                model_name='Copernicus'
            )
        except Exception as error:
            logger.error(f"CMEMS processing failed: {error}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    start_time = time.time()

    logger.info("Starting Redwing TWA plotting script")
    logger.info(f"Configuration: {CONFIG['models']}")
    logger.info(f"Bathymetry enabled: {CONFIG['bathymetry'].get('enabled', True)}")

    models_enabled = CONFIG['models']
    if not models_enabled['plot_model_data']:
        models_enabled['plot_rtofs'] = False
        models_enabled['plot_espc'] = False
        models_enabled['plot_cmems'] = False

    glider_name = CONFIG['glider']['name']
    logger.info(f"Resolving latest deployment for glider: {glider_name}")
    deployment_id = resolve_latest_deployment(
        glider_name,
        CONFIG['glider']['deployments_url'],
        CONFIG['glider']['api_timeout']
    )

    logger.info(f"Fetching glider data for: {deployment_id}")
    glider_data, waypoint_data = fetch_glider_surfacings(
        deployment_id,
        CONFIG['glider']['api_url'],
        CONFIG['glider']['api_timeout']
    )

    if not glider_data.empty:
        latest_glider_date = glider_data.index.max()
        logger.info(f"Latest glider date: {latest_glider_date}")
        reference_date = latest_glider_date
    else:
        reference_date = dt.datetime.now(dt.timezone.utc)
        logger.warning("No glider data available, using current time")

    bathy_data = None
    if CONFIG['bathymetry'].get('enabled', True):
        logger.info("Loading bathymetry data")
        bathy_data = load_bathymetry(CONFIG['region']['extent'])
        if bathy_data is None:
            logger.warning("Bathymetry loading failed, continuing without bathymetry")
    else:
        logger.info("Bathymetry plotting disabled in config")

    rtofs_data = None
    espc_data = None
    cmems_data = None

    if models_enabled['plot_rtofs']:
        logger.info("Loading RTOFS data")
        rtofs_data = load_rtofs(CONFIG['region']['extent'])

    if models_enabled['plot_espc']:
        logger.info("Loading ESPC data")
        espc_data = load_espc(CONFIG['region']['extent'], reference_date)

    if models_enabled['plot_cmems']:
        logger.info("Loading CMEMS data")
        cmems_data = load_cmems(CONFIG['region']['extent'])

    models_available = any([
        rtofs_data is not None and rtofs_data[0] is not None,
        espc_data is not None and espc_data[0] is not None,
        cmems_data is not None,
    ])

    if models_enabled['plot_model_data'] and models_available:
        process_and_plot_time(
            reference_date,
            CONFIG,
            bathy_data,
            glider_data,
            waypoint_data=waypoint_data,
            rtofs_data=rtofs_data,
            espc_data=espc_data,
            cmems_data=cmems_data,
        )
    else:
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

        plot_twa_map(
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
