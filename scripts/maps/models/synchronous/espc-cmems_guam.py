"""
ESPC vs CMEMS side-by-side comparison maps for Guam.

Produces temperature, salinity, and current-speed panels for every 6-hourly
timestep in the most recent `conf.days` window. ESPC timestamps are used to
anchor the window so `.sel(time=...)` lookups always align.
"""
import matplotlib
matplotlib.use('Agg')  # must be before any pyplot import

import copy
import datetime as dt
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from oceans.ocfis import uv2spdir

import ioos_model_comparisons.configs as conf
from cool_maps.plot import create, get_bathymetry
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import CMEMS, espc_ts, espc_uv
from ioos_model_comparisons.platforms import (
    ARGO_GOOD_QC_FLAGS,
    ARGO_LOCATION_QC_VARIABLES,
    filter_argo_by_qc,
    get_active_gliders,
    get_argo_floats_by_time,
)
from ioos_model_comparisons.plotting import (
    cmaps,
    map_add_currents,
    map_add_eez,
    plot_regional_assets_single_color,
)
from ioos_model_comparisons.regions import region_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tstr = '%Y-%m-%d %H:%M:%S'

MAP_PROJECTION = ccrs.Mercator()
DATA_PROJECTION = ccrs.PlateCarree()

# ── Worker-level globals (populated by worker_initializer) ─────────────────
_worker_gds_ts = None
_worker_gds_uv = None
_worker_cds = None
_worker_argo_data = None
_worker_glider_data = None
_worker_bathy_data = None
_worker_region_configs = None
_worker_config = None
_worker_path_save = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_recent_date_list(days, available_times=None):
    """
    Build a 6-hourly time window covering the last `days` days.

    When model timestamps are available (ESPC), prefer those exact times so
    `.sel(time=...)` lookups stay aligned with the dataset, but restrict the
    run to the synoptic cycle: 00Z, 06Z, 12Z, 18Z, including the next day's
    00Z when available as the current day's 24Z.
    """
    current_day_end = (
        pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)
        + pd.Timedelta(days=1)
    )
    window_end = current_day_end
    window_start = window_end - pd.Timedelta(days=days)

    if available_times is not None:
        times = pd.DatetimeIndex(pd.to_datetime(available_times, utc=True)).tz_convert(None)
        times = times.sort_values().unique()
        times = times[times <= current_day_end]
        times = times[times.hour.isin([0, 6, 12, 18])]
        if len(times):
            window_end = times[-1]
            window_start = window_end - pd.Timedelta(days=days)
            window = times[times >= window_start]
            if len(window):
                return pd.DatetimeIndex(window)

    return pd.date_range(window_start, window_end, freq='6h')


# ── Map / plot helpers ─────────────────────────────────────────────────────

def _init_guam_axis(ax, extent, bathy=None, label_left=True, label_right=False):
    create(extent, ax=ax, ticks=False)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    if bathy is not None:
        try:
            lons_b, lats_b = np.meshgrid(bathy.longitude.values, bathy.latitude.values)
            ax.contour(lons_b, lats_b, bathy.z.values, (-1000, -100),
                       linewidths=0.75, alpha=0.5, colors='k',
                       transform=DATA_PROJECTION, zorder=1.5)
            ax.contourf(
                bathy['longitude'],
                bathy['latitude'],
                bathy['z'],
                [-8000, -1000, -100, 0],
                colors=['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue'],
                transform=DATA_PROJECTION,
                transform_first=True,
            )
        except Exception as e:
            logger.warning(f"Bathymetry render failed: {e}")

    h_marker, = ax.plot(
        144.73, 13.41,
        marker='D', markersize=9,
        markeredgecolor='red', markeredgewidth=2,
        color='None',
        transform=DATA_PROJECTION,
        zorder=10000,
        label='Guam',
    )
    leg = ax.legend(handles=[h_marker], loc='lower right', fontsize=10,
                    framealpha=0.8, edgecolor='black')
    leg.set_zorder(10000)

    ax.set_xticks(np.arange(130, 161, 10), crs=DATA_PROJECTION)
    ax.set_xticks(np.arange(130, 161, 5), minor=True, crs=DATA_PROJECTION)
    ax.set_yticks(np.arange(5, 26, 5), crs=DATA_PROJECTION)
    ax.set_yticks(np.arange(5, 26, 5), minor=True, crs=DATA_PROJECTION)

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    gl = ax.gridlines(crs=DATA_PROJECTION, draw_labels=False,
                      linestyle='--', color='black', alpha=0.5)
    gl.xlocator = plt.MultipleLocator(10)
    gl.ylocator = plt.MultipleLocator(5)
    gl.set_zorder(9000)

    ax.tick_params(
        axis='both', which='major',
        labelsize=12, direction='out',
        length=6, width=1,
        top=True, right=True,
        labelleft=label_left, labelright=label_right,
        labelbottom=True, labeltop=False,
    )
    ax.tick_params(
        axis='both', which='minor',
        direction='out', length=6, width=1,
        top=True, right=True,
    )

    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks():
            tick.label1.set_fontweight('bold')
            tick.label2.set_fontweight('bold')
        for tick in axis.get_minor_ticks():
            tick.label1.set_fontweight('bold')
            tick.label2.set_fontweight('bold')


def _build_figure():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 7),
        subplot_kw={'projection': MAP_PROJECTION},
    )
    return fig, axes[0], axes[1]


def plot_guam_comparison(ds_espc, ds_cmems, region,
                         bathy=None, argo=None, gliders=None,
                         eez=False, path_save=os.getcwd(),
                         dpi=300, overwrite=False):
    time_pd = pd.to_datetime(ds_espc.time.data)
    time_str = time_pd.strftime('%Y-%m-%dT%H:%MZ')
    year = time_pd.strftime('%Y')
    month = time_pd.strftime('%m')
    extent = region['extent']

    for var_key, var_list in region['variables'].items():
        var_label = ' '.join(var_key.split('_')).title()

        for item in var_list:
            depth = item['depth']
            logger.info(f"Plotting {var_key} @ {depth}m")

            save_dir = path_save / f"{var_key}_{depth}m" / f"{year}/{month}"
            os.makedirs(save_dir, exist_ok=True)
            sname = (
                f'{"-".join(region["folder"].split("_"))}_'
                f'{time_pd.strftime("%Y-%m-%dT%H%M%SZ")}_'
                f'{var_key}-{depth}m_espc-vs-cmems'
            )
            save_file = save_dir / f"{sname}.png"

            if save_file.is_file() and not overwrite:
                logger.info(f"{save_file} exists. Skipping.")
                continue

            try:
                espc_slice = ds_espc[var_key].sel(depth=depth, method='nearest')
                for _tdim in [d for d in espc_slice.dims if 'time' in d.lower()]:
                    espc_slice = espc_slice.isel({_tdim: 0})

                cmems_slice = ds_cmems[var_key].sel(depth=depth, method='nearest')
                for _tdim in [d for d in cmems_slice.dims if 'time' in d.lower()]:
                    cmems_slice = cmems_slice.isel({_tdim: 0})
            except Exception as e:
                logger.error(f"Could not select {var_key} @ {depth}m: {e}")
                continue

            fig, ax1, ax2 = _build_figure()

            _init_guam_axis(ax1, extent, bathy=bathy, label_left=True, label_right=False)
            _init_guam_axis(ax2, extent, bathy=bathy, label_left=False, label_right=True)

            rargs = {'argo': argo, 'gliders': gliders, 'transform': DATA_PROJECTION, 'time': time_pd}
            plot_regional_assets_single_color(ax1, **rargs)
            plot_regional_assets_single_color(ax2, **rargs)

            if eez:
                map_add_eez(ax1, zorder=20, color='white', linewidth=1)
                map_add_eez(ax2, zorder=20, color='white', linewidth=1)

            vargs = {
                'transform': DATA_PROJECTION,
                'transform_first': True,
                'cmap': cmaps(var_key),
                'extend': 'both',
            }
            if 'limits' in item:
                vargs['vmin'] = item['limits'][0]
                vargs['vmax'] = item['limits'][1]
                vargs['levels'] = np.arange(item['limits'][0], item['limits'][1], item['limits'][2])

            if espc_slice['lon'].ndim == 1 and espc_slice['lat'].ndim == 1:
                e_lons, e_lats = np.meshgrid(espc_slice['lon'], espc_slice['lat'])
            else:
                e_lons, e_lats = espc_slice['lon'], espc_slice['lat']
            h1 = ax1.contourf(e_lons, e_lats, espc_slice.squeeze(), **vargs)

            if cmems_slice['lon'].ndim == 1 and cmems_slice['lat'].ndim == 1:
                c_lons, c_lats = np.meshgrid(cmems_slice['lon'], cmems_slice['lat'])
            else:
                c_lons, c_lats = cmems_slice['lon'], cmems_slice['lat']
            ax2.contourf(c_lons, c_lats, cmems_slice.squeeze(), **vargs)

            ax1.set_title("ESPC", fontsize=16, fontweight='bold', pad=6)
            ax2.set_title("CMEMS", fontsize=16, fontweight='bold', pad=6)

            fig.tight_layout(rect=[0, 0.10, 1, 0.95], w_pad=1.5)

            units = getattr(espc_slice, 'units', '')
            cb_label = f'{var_label} ({units})' if units else var_label
            cb = fig.colorbar(h1, ax=[ax1, ax2], orientation='horizontal',
                              fraction=0.03, pad=0.1, aspect=60)
            cb.ax.tick_params(labelsize=11)
            cb.set_label(cb_label, fontsize=12, fontweight='bold')

            axes_top = max(ax.get_position().y1 for ax in [ax1, ax2])
            fig.suptitle(f"{var_label} ({depth} m) — {time_str}", fontsize=18, fontweight='bold',
                         y=axes_top + 0.08)

            fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            logger.info(f"Saved: {save_file}")


def plot_guam_streamplot(ds_espc_uv, ds_cmems, region,
                         bathy=None, argo=None, gliders=None,
                         eez=True, path_save=os.getcwd(),
                         dpi=300, overwrite=False):
    time_pd = pd.to_datetime(ds_espc_uv.time.data)
    time_str = time_pd.strftime('%Y-%m-%dT%H:%MZ')
    year = time_pd.strftime('%Y')
    month = time_pd.strftime('%m')
    extent = region['extent']

    cdict = region['currents']
    coarsen_cfg = cdict.get('coarsen', {})
    stream_kwargs = cdict.get('kwargs', {})

    for depth in cdict['depths']:
        logger.info(f"Plotting currents @ {depth}m")

        save_dir = path_save / f"currents_{depth}m" / f"{year}/{month}"
        os.makedirs(save_dir, exist_ok=True)
        sname = (
            f'{"-".join(region["folder"].split("_"))}_'
            f'{time_pd.strftime("%Y-%m-%dT%H%M%SZ")}_'
            f'currents-{depth}m_espc-vs-cmems'
        )
        save_file = save_dir / f"{sname}.png"

        if save_file.is_file() and not overwrite:
            logger.info(f"{save_file} exists. Skipping.")
            continue

        try:
            espc_depth = ds_espc_uv.sel(depth=depth, method='nearest')
        except Exception as e:
            logger.error(f"Could not select ESPC currents @ {depth}m: {e}")
            continue

        try:
            cmems_depth = ds_cmems.sel(depth=depth, method='nearest')
        except Exception as e:
            logger.warning(f"Could not select CMEMS currents @ {depth}m (trying without method): {e}")
            try:
                cmems_depth = ds_cmems
            except Exception:
                continue

        _, mag_espc = uv2spdir(espc_depth['u'], espc_depth['v'])
        _, mag_cmems = uv2spdir(cmems_depth['u'], cmems_depth['v'])

        if depth == 1500:
            levels = np.arange(0, 0.4, 0.05)
        else:
            lims = cdict['limits']
            levels = np.arange(lims[0], lims[1] + lims[2], lims[2])

        qargs = {
            'transform': DATA_PROJECTION,
            'transform_first': True,
            'cmap': cmocean.cm.speed,
            'extend': 'max',
            'levels': levels,
        }

        fig, ax1, ax2 = _build_figure()

        _init_guam_axis(ax1, extent, bathy=bathy, label_left=True, label_right=False)
        _init_guam_axis(ax2, extent, bathy=bathy, label_left=False, label_right=True)

        rargs = {'argo': argo, 'gliders': gliders, 'transform': DATA_PROJECTION, 'time': time_pd}
        plot_regional_assets_single_color(ax1, **rargs)
        plot_regional_assets_single_color(ax2, **rargs)

        if espc_depth['lon'].ndim == 1 and espc_depth['lat'].ndim == 1:
            e_lons, e_lats = np.meshgrid(espc_depth['lon'], espc_depth['lat'])
        else:
            e_lons, e_lats = espc_depth['lon'], espc_depth['lat']
        m1 = ax1.contourf(e_lons, e_lats, mag_espc, **qargs)

        if cmems_depth['lon'].ndim == 1 and cmems_depth['lat'].ndim == 1:
            c_lons, c_lats = np.meshgrid(cmems_depth['lon'], cmems_depth['lat'])
        else:
            c_lons, c_lats = cmems_depth['lon'], cmems_depth['lat']
        ax2.contourf(c_lons, c_lats, mag_cmems, **qargs)

        espc_coarsen = coarsen_cfg.get('espc', 8)
        cmems_coarsen = coarsen_cfg.get('cmems', 8)
        map_add_currents(ax1, espc_depth, coarsen=espc_coarsen, **stream_kwargs)
        map_add_currents(ax2, cmems_depth, coarsen=cmems_coarsen, **stream_kwargs)

        if eez:
            map_add_eez(ax1, zorder=200, color='white', linewidth=1)
            map_add_eez(ax2, zorder=200, color='white', linewidth=1)

        ax1.set_title("ESPC", fontsize=16, fontweight='bold', pad=6)
        ax2.set_title("CMEMS", fontsize=16, fontweight='bold', pad=6)

        fig.tight_layout(rect=[0, 0.10, 1, 0.95], w_pad=1.5)

        cb = fig.colorbar(m1, ax=[ax1, ax2], orientation='horizontal',
                          fraction=0.03, pad=0.1, aspect=60)
        cb.ax.tick_params(labelsize=11)
        cb.set_label('Current Speed (m/s)', fontsize=12, fontweight='bold')

        axes_top = max(ax.get_position().y1 for ax in [ax1, ax2])
        fig.suptitle(f"Currents ({depth} m) — {time_str}", fontsize=18, fontweight='bold',
                     y=axes_top + 0.08)

        fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        logger.info(f"Saved: {save_file}")


# ── Data loading helpers ───────────────────────────────────────────────────

def subset_data_lonlat(data, lon_extent, lat_extent):
    try:
        return data.sel(
            lon=slice(lon_extent[0], lon_extent[1]),
            lat=slice(lat_extent[2], lat_extent[3]),
        )
    except Exception as e:
        logger.error(f"Error during lon/lat data subsetting: {e}")
        return None


def load_or_fetch_bathymetry(extent, cache_dir):
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"bathy_{ext_str}.pkl"
    if cache_file.exists():
        logger.info(f"Loading bathymetry from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Fetching bathymetry for extent {extent}...")
    bathy_data = get_bathymetry(extent)
    bathy_data = xr.Dataset({
        'z': xr.DataArray(
            bathy_data.z.values,
            dims=['latitude', 'longitude'],
            coords={
                'latitude': bathy_data.latitude.values,
                'longitude': bathy_data.longitude.values,
            },
        )
    })

    with open(cache_file, 'wb') as f:
        pickle.dump(bathy_data, f)
    return bathy_data


# ── Worker initializer ─────────────────────────────────────────────────────

def worker_initializer(
    global_extent_buffered,
    lon_transform,
    argo_data,
    glider_data,
    bathy_data,
    region_configs,
    config_dict,
    path_save,
):
    global _worker_gds_ts, _worker_gds_uv, _worker_cds
    global _worker_argo_data, _worker_glider_data, _worker_bathy_data
    global _worker_region_configs, _worker_config, _worker_path_save

    _worker_argo_data = argo_data
    _worker_glider_data = glider_data
    _worker_bathy_data = bathy_data
    _worker_region_configs = region_configs
    _worker_config = config_dict
    _worker_path_save = path_save

    if config_dict['plot_espc']:
        logger.info("Loading ESPC-TS (FMRC best-forecast)...")
        gds_ts_local = espc_ts(rename=True)
        _worker_gds_ts = gds_ts_local.sel(
            lon=slice(lon_transform[0], lon_transform[1]),
            lat=slice(global_extent_buffered[2], global_extent_buffered[3]),
        )

        logger.info("Loading ESPC-UV (FMRC best-forecast)...")
        gds_uv_local = espc_uv(rename=True)
        _worker_gds_uv = gds_uv_local.sel(
            lon=slice(lon_transform[0], lon_transform[1]),
            lat=slice(global_extent_buffered[2], global_extent_buffered[3]),
        )

    if config_dict['plot_cmems']:
        logger.info("Authenticating with CMEMS...")
        cobj = CMEMS()
        lon_slice = slice(global_extent_buffered[0], global_extent_buffered[1])
        lat_slice = slice(global_extent_buffered[2], global_extent_buffered[3])
        _worker_cds = {
            'temperature': cobj.get_variable('temperature').sel(longitude=lon_slice, latitude=lat_slice),
            'salinity': cobj.get_variable('salinity').sel(longitude=lon_slice, latitude=lat_slice),
            'u': cobj.get_variable('uo').sel(longitude=lon_slice, latitude=lat_slice),
            'v': cobj.get_variable('vo').sel(longitude=lon_slice, latitude=lat_slice),
        }

    logger.info("Worker %s initialized", mp.current_process().name)


# ── Per-timestamp processing ───────────────────────────────────────────────

def process_time(ctime):
    """Fetch model data and produce all plots for a single timestep."""
    gds_ts = _worker_gds_ts
    gds_uv = _worker_gds_uv
    cds = _worker_cds
    region_configs = _worker_region_configs
    argo_data = _worker_argo_data
    glider_data = _worker_glider_data
    bathy_data = _worker_bathy_data
    path_save = _worker_path_save
    config = _worker_config

    logger.info(f"Processing {ctime}")
    plots_generated = []

    for region_key in config['regions']:
        region = region_configs[region_key]
        extent = region['extent']
        extended = np.add(extent, [-1, 1, -1, 1]).tolist()
        lon360 = lon180to360(extended[:2])

        search_window_t0 = (ctime - dt.timedelta(hours=config['search_hours'])).strftime(tstr)
        search_window_t1 = ctime.strftime(tstr)

        gdt_flag = False
        gdt_ts_data = None
        if config['plot_espc'] and gds_ts is not None:
            try:
                data = gds_ts.sel(time=ctime)
                gdt_ts_data = subset_data_lonlat(data, lon360, extended)
                logger.info(f"  [{ctime}] ESPC-TS selected for {region_key}")
                gdt_flag = gdt_ts_data is not None
            except (KeyError, ValueError) as e:
                logger.warning(f"  [{ctime}] ESPC-TS not available for {region_key}: {e}")
            except Exception as e:
                logger.error(f"  [{ctime}] ESPC-TS error for {region_key}: {e}\n{traceback.format_exc()}")

        gdv_flag = False
        gdt_uv_data = None
        if config['plot_espc'] and gds_uv is not None:
            try:
                data = gds_uv.sel(time=ctime)
                gdt_uv_data = subset_data_lonlat(data, lon360, extended)
                logger.info(f"  [{ctime}] ESPC-UV selected for {region_key}")
                gdv_flag = gdt_uv_data is not None
            except (KeyError, ValueError) as e:
                logger.warning(f"  [{ctime}] ESPC-UV not available for {region_key}: {e}")
            except Exception as e:
                logger.error(f"  [{ctime}] ESPC-UV error for {region_key}: {e}\n{traceback.format_exc()}")

        cdt_flag = False
        cdt_data = None
        if config['plot_cmems'] and cds is not None:
            try:
                temperature = cds['temperature'].sel(time=ctime, method='nearest').sel(
                    longitude=slice(extent[0], extent[1]),
                    latitude=slice(extent[2], extent[3]),
                )
                salinity = cds['salinity'].sel(time=ctime, method='nearest').sel(
                    longitude=slice(extent[0], extent[1]),
                    latitude=slice(extent[2], extent[3]),
                )
                u = cds['u'].sel(time=ctime, method='nearest').sel(
                    longitude=slice(extent[0], extent[1]),
                    latitude=slice(extent[2], extent[3]),
                )
                v = cds['v'].sel(time=ctime, method='nearest').sel(
                    longitude=slice(extent[0], extent[1]),
                    latitude=slice(extent[2], extent[3]),
                )
                data = xr.Dataset({'temperature': temperature, 'salinity': salinity, 'u': u, 'v': v})
                data = data.rename({k: v for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items() if k in data.coords})
                logger.info(f"  [{ctime}] CMEMS fetched for {region_key}")
                cdt_flag = True
                cdt_data = data
            except Exception as e:
                logger.warning(f"  [{ctime}] CMEMS not available for {region_key}: {e}")

        # ── Platform subsets ───────────────────────────────────────────
        argo_region = pd.DataFrame()
        if not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']
            mask = (
                (extended[0] <= lon) & (lon <= extended[1]) &
                (extended[2] <= lat) & (lat <= extended[3])
            )
            argo_region = argo_data[mask].sort_index()
            idx = pd.IndexSlice
            try:
                argo_region = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]
            except KeyError:
                argo_region = pd.DataFrame()

        glider_region = pd.DataFrame()
        if not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']
            mask = (
                (extended[0] <= lon) & (lon <= extended[1]) &
                (extended[2] <= lat) & (lat <= extended[3])
            )
            glider_region = glider_data[mask]
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time')) &
                (glider_region.index.get_level_values('time') < search_window_t1)
            ]

        kwargs = {
            'bathy':     bathy_data,
            'eez':       region.get('eez', True),
            'path_save': path_save / region['folder'],
            'dpi':       config['dpi'],
            'overwrite': config['overwrite'],
            'argo':      argo_region if not argo_region.empty else None,
            'gliders':   glider_region if not glider_region.empty else None,
        }

        # ── Plots ──────────────────────────────────────────────────────
        if gdt_flag and cdt_flag:
            try:
                plot_guam_comparison(gdt_ts_data, cdt_data, region, **kwargs)
                plots_generated.append(f"{region_key} | T/S | ESPC vs CMEMS")
            except Exception as e:
                logger.error(f"  [{ctime}] T/S plot failed for {region_key}: {e}\n{traceback.format_exc()}")

        if gdv_flag and cdt_flag:
            try:
                plot_guam_streamplot(gdt_uv_data, cdt_data, region, **kwargs)
                plots_generated.append(f"{region_key} | Currents | ESPC vs CMEMS")
            except Exception as e:
                logger.error(f"  [{ctime}] Currents plot failed for {region_key}: {e}\n{traceback.format_exc()}")

    return {'ctime': ctime, 'plots': plots_generated}


# ── Main ───────────────────────────────────────────────────────────────────

def main(parallel=True, max_workers=None):
    start_time_exec = time.time()

    plot_espc = True
    plot_cmems = True
    overwrite = False

    conf.days = 1
    conf.regions = ['guam']

    path_save = conf.path_plots / "maps"
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)

    region_configs = {r: region_config(r) for r in conf.regions}

    extent_list = [cfg["extent"] for cfg in region_configs.values()]
    extent_df = pd.DataFrame(extent_list, columns=['lonmin', 'lonmax', 'latmin', 'latmax'])
    global_extent = [
        extent_df.lonmin.min(), extent_df.lonmax.max(),
        extent_df.latmin.min(), extent_df.latmax.max(),
    ]
    global_extent_buffered = [
        global_extent[0] - 3, global_extent[1] + 3,
        global_extent[2] - 3, global_extent[3] + 3,
    ]
    lon_transform = lon180to360(global_extent_buffered[:2])

    # ── ESPC: open once to anchor the date window ──────────────────────────
    gds_ts_probe = None
    if plot_espc:
        try:
            logger.info("Loading ESPC-TS (FMRC best-forecast)...")
            gds_ts_probe = espc_ts(rename=True)
        except Exception as e:
            logger.error(f"ESPC-TS load failed: {e}\n{traceback.format_exc()}")

    # Build date list from ESPC timestamps so .sel() always aligns
    date_list = _build_recent_date_list(
        conf.days,
        gds_ts_probe.time.values if gds_ts_probe is not None else None,
    )
    date_start = date_list[0]
    date_end   = date_list[-1]
    search_start = date_start - dt.timedelta(hours=conf.search_hours)
    del gds_ts_probe

    logger.info(
        "Using %d timestamps from %s through %s",
        len(date_list), date_start, date_end,
    )
    for t in date_list:
        logger.info(f"  {t}")

    # ── Platform and bathymetry data ───────────────────────────────────────
    argo_data = pd.DataFrame()
    if conf.argo:
        try:
            argo_data = get_argo_floats_by_time(
                global_extent,
                search_start,
                date_end,
                include_qc=ARGO_LOCATION_QC_VARIABLES,
            )
            raw_argo_count = len(argo_data)
            argo_data = filter_argo_by_qc(
                argo_data,
                qc_columns=ARGO_LOCATION_QC_VARIABLES,
                allowed_flags=ARGO_GOOD_QC_FLAGS,
            )
            logger.info("Argo QC retained %s/%s samples.", len(argo_data), raw_argo_count)
            logger.info(f"Argo data {'loaded' if not argo_data.empty else 'not available'}.")
        except Exception as e:
            logger.error(f"Failed to load Argo data: {e}")

    glider_data = pd.DataFrame()
    if conf.gliders:
        try:
            glider_data = get_active_gliders(global_extent, search_start, date_end, parallel=False, timeout=60)
            logger.info(f"Glider data {'loaded' if not glider_data.empty else 'not available'}.")
            glider_data.index = glider_data.index.set_levels(
                glider_data.index.levels[0].str.rsplit("-", n=1).str[0],
                level="glider"
            )
        except Exception as e:
            logger.error(f"Failed to load Glider data: {e}")

    bathy_data = None
    if conf.bathy:
        try:
            bathy_data = load_or_fetch_bathymetry(global_extent, cache_dir)
            logger.info("Bathymetry data loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Bathymetry data: {e}")

    config_dict = {
        'regions': conf.regions,
        'dpi': 300,
        'search_hours': conf.search_hours,
        'plot_espc': plot_espc,
        'plot_cmems': plot_cmems,
        'overwrite': overwrite,
    }

    init_args = (
        global_extent_buffered,
        lon_transform,
        argo_data,
        glider_data,
        bathy_data,
        region_configs,
        config_dict,
        path_save,
    )

    all_results = []
    if not parallel:
        worker_initializer(*init_args)
        for ctime in date_list:
            result = process_time(ctime)
            all_results.append(result)
    else:
        max_workers = max_workers or min(4, mp.cpu_count())
        logger.info("Using %s worker processes", max_workers)

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_initializer,
            initargs=init_args,
        ) as executor:
            futures = {executor.submit(process_time, ctime): ctime for ctime in date_list}
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                ctime = futures[future]
                completed += 1
                try:
                    result = future.result()
                    all_results.append(result)
                    logger.info("[%s/%s] %s done (%s plots)", completed, total, ctime, len(result['plots']))
                except Exception as e:
                    logger.error("[%s/%s] %s ERROR: %s\n%s", completed, total, ctime, e, traceback.format_exc())
                    all_results.append({'ctime': ctime, 'plots': []})

    # ── Summary ────────────────────────────────────────────────────────────
    total_plots = sum(len(r['plots']) for r in all_results)
    print(f"\n{'='*60}")
    print(f"SUMMARY — {total_plots} plot(s) generated")
    print(f"{'='*60}")
    for result in sorted(all_results, key=lambda r: r['ctime'] if isinstance(r, dict) else pd.Timestamp.min):
        if not isinstance(result, dict):
            continue
        ctime_str = result['ctime'].strftime('%Y-%m-%d %H:%MZ')
        print(f"\n  {ctime_str} ({len(result['plots'])} plots)")
        for p in result['plots']:
            print(f"    • {p}")
    elapsed = time.time() - start_time_exec
    print(f"\nTotal execution time: {elapsed / 60:.2f} min")


if __name__ == "__main__":
    main(parallel=True, max_workers=4)
