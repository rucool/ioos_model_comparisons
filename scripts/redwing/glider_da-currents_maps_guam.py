"""
Depth-averaged currents comparison maps for Guam: RTOFS vs ESPC and RTOFS vs
CMEMS, side by side.

RTOFS OPeNDAP does not cover the Guam domain, so RTOFS is read from the same
pre-processed binary NetCDFs that
scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py produces
(rtofs_archv/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_guam.nc), instead of
ioos_model_comparisons.models.rtofs().

Output is saved into the same maps/<region> tree used by
rtofs-gofs-cmems-amseas.py and rtofs_binary_model_comparisons.py:
    <path_plots>/maps/guam/currents_depthavg/YYYY/MM/
        guam_<timestamp>_currents-depthavg_<model1>-vs-<model2>.png
"""
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cool_maps.plot import create, add_bathymetry
from oceans.ocfis import uv2spdir
import cartopy.feature as cfeature

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.regions import region_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

REGION_NAME = "Guam"
GUAM_EXTENT = [129.75, 160.25, 4.75, 25.25]  # matches region_config('guam')
RTOFS_BINARY_REGION = "guam"
REGION_FOLDER = region_config(regions=[RTOFS_BINARY_REGION])["folder"]

# Same top-level maps directory that rtofs-gofs-cmems-amseas.py and
# rtofs_binary_model_comparisons.py write Guam maps into.
PATH_SAVE = conf.path_plots / "maps" / REGION_FOLDER

CONFIG = {
    'paths': {
        'eez_path': '/home/hurricaneadm/data/World_Exclusive_Economic_Zones_Boundaries-shp/World_Exclusive_Economic_Zones_Boundaries.shp',
        # RTOFS OPeNDAP does not cover Guam; use the pre-processed binary NetCDFs
        # produced by scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py instead.
        'rtofs_binary_dir': '/home/hurricaneadm/data/rtofs_archv',
        # 'rtofs_binary_dir': str(Path.home() / 'Downloads' / 'rtofs_global'),  # local dev
    },
    'models': {
        'plot_espc': True,
        'plot_cmems': True,
    },
    'depth_average': {
        'min_depth': 0,
        'max_depth': 1000,
        'depth_step': 1,
    },
    'currents': {
        'limits_depth_avg': [0, 0.5, 0.05],  # m/s
        'auto_colorbar': False,
        'streamplot': {
            'density': 3,
            'linewidth': 0.5,
            'color': 'black',
        }
    },
    'plotting': {
        'figsize': (16, 8),
        'dpi': 300,
    },
    'bathymetry': {
        'enabled': False,
        'contour_levels': (-1000, -100),
        'filled_levels': [-8000, -1000, -100, 0],
        'filled_colors': ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue'],
    }
}

MAP_PROJECTION = ccrs.Mercator()
DATA_PROJECTION = ccrs.PlateCarree()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lon180to360(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return np.mod(array, 360)


def lon360to180(array: np.ndarray) -> np.ndarray:
    array = np.array(array)
    return np.mod(array + 180, 360) - 180


def expand_extent(extent: list, buffer: float = 1.0) -> list:
    return np.add(extent, [-buffer, buffer, -buffer, buffer]).tolist()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_bathymetry(extent: list) -> Optional[xr.Dataset]:
    try:
        from ioos_model_comparisons.platforms import get_bathymetry
        ds = get_bathymetry(bbox=extent)
        logger.info("GEBCO bathymetry loaded successfully")
        return ds
    except Exception as exc:
        logger.warning(f"Failed to load GEBCO bathymetry: {exc}")
        return None


def find_rtofs_binary_files(data_dir: Path) -> list:
    """Return sorted list of pre-processed RTOFS Guam NetCDFs under data_dir.

    Expects the same directory layout produced by
    scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py:
        <data_dir>/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_guam.nc
    """
    return sorted(Path(data_dir).glob(f"*/*/*/rtofs_glo_*_{RTOFS_BINARY_REGION}.nc"))


def parse_rtofs_binary_valid_time(nc_path: Path) -> pd.Timestamp:
    """Extract valid time from filename like rtofs_glo_20260629T06_guam.nc"""
    time_part = nc_path.stem.split("_")[2]
    return pd.Timestamp(dt.datetime.strptime(time_part, "%Y%m%dT%H"))


def load_rtofs_binary_file(nc_path: Path) -> xr.Dataset:
    """Load one pre-processed RTOFS NetCDF and rename variables to match the
    schema (u/v/depth) used by the rest of this script."""
    ds = xr.open_dataset(nc_path)
    valid_time = parse_rtofs_binary_valid_time(nc_path)
    ds = ds.rename({
        "temp": "temperature",
        "salin": "salinity",
        "u-vel.": "u",
        "v-vel.": "v",
        "z": "depth",
    })
    ds = ds.assign_coords(time=valid_time)
    # NOTE: keep u/v as data variables (not coords via set_coords) — compute_depth_avg_currents()
    # reduces over the depth dimension with .mean(), which silently drops any coordinate
    # variable that depends on the reduced dim.
    ds.attrs["model"] = "RTOFS"
    ds["u"].attrs["units"] = "m/s"
    ds["v"].attrs["units"] = "m/s"
    return ds


def load_latest_rtofs_binary() -> Optional[xr.Dataset]:
    """Load the most recent pre-processed RTOFS binary NetCDF for Guam."""
    data_dir = Path(CONFIG['paths']['rtofs_binary_dir'])
    try:
        files = find_rtofs_binary_files(data_dir)
        if not files:
            logger.error("No RTOFS binary files found for '%s' under %s", RTOFS_BINARY_REGION, data_dir)
            return None
        latest_file = files[-1]
        rds = load_rtofs_binary_file(latest_file)
        logger.info("RTOFS binary data loaded from %s (valid %s)", latest_file, pd.Timestamp(rds.time.values))
        return rds
    except Exception as exc:
        logger.error(f"Failed to load RTOFS binary data: {exc}")
        return None


def load_espc(extent: list, reference_date: dt.datetime):
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
            return espc_ds
        else:
            from ioos_model_comparisons.models import espc_uv
            espc_u = espc_uv(rename=True)
            logger.info("ESPC operational data loaded successfully")
            return espc_u
    except Exception as exc:
        logger.error(f"ESPC data load failed: {exc}")
        return None


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
    if "u" not in depth_avg.data_vars or "v" not in depth_avg.data_vars:
        # .mean() silently drops non-index coordinate variables that depend on the
        # reduced dim, so if u/v were ever marked as coords upstream they vanish here.
        logger.warning("Depth-average skipped: 'u'/'v' lost during depth reduction.")
        return None
    depth_avg.attrs = dict(ds.attrs)
    depth_avg.attrs["depth_average"] = {
        "min": float(start_depth),
        "max": float(end_depth),
        "step": float(depth_step),
    }
    depth_avg.attrs["product"] = "Depth-averaged currents"

    return depth_avg


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
    lons = ds.lon.squeeze().data
    lats = ds.lat.squeeze().data
    u = ds.u.squeeze().data
    v = ds.v.squeeze().data

    sargs = {
        "transform": transform,
        "density": density,
        "linewidth": linewidth,
        "color": color,
    }

    return ax.streamplot(lons, lats, u, v, **sargs)


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
    return ax.add_feature(shape_feature, zorder=zorder)


def subset_model_data(model_ds: xr.Dataset, extent: list, model_name: str = "Model") -> xr.Dataset:
    extent_data = expand_extent(extent, buffer=1.0)

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


# ============================================================================
# PLOTTING
# ============================================================================

def plot_depth_avg_comparison(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    model1_name: str,
    model2_name: str,
    extent: list,
    bathy: Optional[xr.Dataset],
    reference_time: pd.Timestamp,
    min_depth: float,
    max_depth: float,
    config: Dict,
):
    figsize = config['plotting']['figsize']
    dpi = config['plotting']['dpi']
    bathy_config = config['bathymetry']
    bathy_enabled = bathy_config.get('enabled', True) and bathy is not None
    stream_config = config['currents']['streamplot']

    auto_colorbar = config['currents'].get('auto_colorbar', False)
    qargs = {'transform': DATA_PROJECTION, 'cmap': cmocean.cm.speed, 'extend': 'max'}
    if not auto_colorbar:
        lim = config['currents']['limits_depth_avg']
        qargs['levels'] = np.arange(lim[0], lim[1], lim[2])

    fig, axs = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw=dict(projection=MAP_PROJECTION),
    )

    mappable = None
    for ax, ds, label in ((axs[0], ds1, model1_name), (axs[1], ds2, model2_name)):
        create(extent, ax=ax, ticks=True)

        if bathy_enabled:
            add_bathymetry(
                ax, bathy.longitude.values, bathy.latitude.values, bathy.z.values,
                levels=bathy_config['contour_levels'], zorder=1.5,
            )
            ax.contourf(
                bathy['longitude'], bathy['latitude'], bathy['z'],
                bathy_config['filled_levels'], colors=bathy_config['filled_colors'],
                transform=DATA_PROJECTION,
            )

        ds_plot = regrid_curvilinear(ds)
        _, mag = uv2spdir(ds_plot['u'], ds_plot['v'])
        mappable = ax.contourf(ds_plot['lon'], ds_plot['lat'], mag, **qargs)
        map_add_currents(
            ax, ds_plot,
            density=stream_config['density'],
            linewidth=stream_config['linewidth'],
            color=stream_config['color'],
        )
        map_add_eez(ax, color='red', linewidth=1.5)
        ax.set_title(label.upper(), fontsize=16, fontweight='bold')

    cb = fig.colorbar(mappable, ax=axs, orientation='horizontal', shrink=0.8, aspect=40, pad=0.08)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('Depth-Averaged Current Speed (m/s)', fontsize=12, fontweight='bold')
    cb.formatter = FormatStrFormatter('%.2f')
    cb.update_ticks()

    time_str = reference_time.strftime('%Y-%m-%dT%H:%MZ')
    fig.suptitle(
        f"{REGION_NAME} Depth-Averaged Currents ({min_depth:.0f}-{max_depth:.0f} m)\n{time_str}",
        fontsize=20, fontweight='bold',
    )

    save_dir = PATH_SAVE / "currents_depthavg" / reference_time.strftime('%Y/%m')
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp_tag = reference_time.strftime('%Y-%m-%dT%H%M%SZ')
    sname = f"{REGION_FOLDER}_{timestamp_tag}_currents-depthavg_{model1_name.lower()}-vs-{model2_name.lower()}.png"
    save_file = save_dir / sname

    fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    logger.info("Saved: %s", save_file)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    start_time = time.time()
    logger.info("Starting Guam depth-averaged currents comparison script")

    load_extent = expand_extent(GUAM_EXTENT, buffer=1.0)

    bathy_data = None
    if CONFIG['bathymetry'].get('enabled', True):
        bathy_data = load_bathymetry(load_extent)
        if bathy_data is None:
            logger.warning("Bathymetry load failed; continuing without it.")

    bathy = None
    if bathy_data is not None:
        bathy = bathy_data.sel(
            longitude=slice(load_extent[0], load_extent[1]),
            latitude=slice(load_extent[2], load_extent[3]),
        )

    rds_raw = load_latest_rtofs_binary()
    if rds_raw is None:
        logger.error("No RTOFS binary data available; exiting.")
        return

    reference_time = pd.Timestamp(rds_raw.time.values)
    logger.info("RTOFS reference time: %s", reference_time)

    rtofs_subset = subset_model_data(rds_raw, GUAM_EXTENT, model_name='RTOFS')
    min_d = CONFIG['depth_average']['min_depth']
    max_d = CONFIG['depth_average']['max_depth']
    depth_step = CONFIG['depth_average']['depth_step']

    rtofs_depth_avg = compute_depth_avg_currents(
        rtofs_subset, min_depth=min_d, max_depth=max_d, depth_step=depth_step,
    )
    if rtofs_depth_avg is None:
        logger.error("Failed to compute RTOFS depth-averaged currents; exiting.")
        return

    if CONFIG['models'].get('plot_espc', True):
        try:
            espc_ds = load_espc(load_extent, reference_time)
            if espc_ds is not None:
                espc_time = espc_ds.sel(time=reference_time, method='nearest')
                if 'time1' in espc_time.dims:
                    espc_time = espc_time.sel(time1=reference_time, method='nearest')
                espc_subset = subset_model_data(espc_time, GUAM_EXTENT, model_name='ESPC')
                espc_depth_avg = compute_depth_avg_currents(
                    espc_subset, min_depth=min_d, max_depth=max_d, depth_step=depth_step,
                )
                if espc_depth_avg is not None:
                    plot_depth_avg_comparison(
                        rtofs_depth_avg, espc_depth_avg, 'RTOFS', 'ESPC',
                        GUAM_EXTENT, bathy, reference_time, min_d, max_d, CONFIG,
                    )
                else:
                    logger.warning("ESPC depth-average failed; skipping RTOFS-vs-ESPC map.")
        except Exception as exc:
            logger.error("RTOFS-vs-ESPC comparison failed: %s", exc)

    if CONFIG['models'].get('plot_cmems', True):
        try:
            cmems_ds = load_cmems(load_extent)
            if cmems_ds is not None:
                u = cmems_ds['u'].sel(time=reference_time, method='nearest')
                v = cmems_ds['v'].sel(time=reference_time, method='nearest')
                cmems_time = xr.Dataset({'u': u, 'v': v})
                cmems_subset = subset_model_data(cmems_time, GUAM_EXTENT, model_name='Copernicus')
                cmems_depth_avg = compute_depth_avg_currents(
                    cmems_subset, min_depth=min_d, max_depth=max_d, depth_step=depth_step,
                )
                if cmems_depth_avg is not None:
                    plot_depth_avg_comparison(
                        rtofs_depth_avg, cmems_depth_avg, 'RTOFS', 'Copernicus',
                        GUAM_EXTENT, bathy, reference_time, min_d, max_d, CONFIG,
                    )
                else:
                    logger.warning("CMEMS depth-average failed; skipping RTOFS-vs-CMEMS map.")
        except Exception as exc:
            logger.error("RTOFS-vs-CMEMS comparison failed: %s", exc)

    elapsed = time.time() - start_time
    logger.info("Execution completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
