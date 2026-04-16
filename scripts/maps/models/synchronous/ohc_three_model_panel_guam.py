"""
OHC two-model side-by-side plots for Guam: ESPC | CMEMS.

Produces one figure per 6-hourly timestep over the most recent `conf.days`
window ending on the latest current-day model time available.
Cached OHC NetCDF files are used when available to skip recomputation.
"""
import matplotlib
matplotlib.use('Agg')  # must be set before any pyplot import

import datetime as dt
import logging
import multiprocessing as mp
import pickle
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

try:
    import httpx
    _HTTPX_ERRORS = (httpx.HTTPError,)
except ImportError:
    _HTTPX_ERRORS = ()

import cartopy.crs as ccrs
import ioos_model_comparisons.configs as conf
import numpy as np
import pandas as pd
import xarray as xr
from cool_maps.plot import get_bathymetry
from ioos_model_comparisons.calc import lon180to360, compute_ohc_vectorized
from ioos_model_comparisons.models import CMEMS, espc_ts
from ioos_model_comparisons.platforms import (
    ARGO_GOOD_QC_FLAGS,
    ARGO_LOCATION_QC_VARIABLES,
    filter_argo_by_qc,
    get_active_gliders,
    get_argo_floats_by_time,
)
from ioos_model_comparisons.plotting import plot_ohc_all_models
from ioos_model_comparisons.regions import region_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GUAM_TRANSFORM = {
    'map': ccrs.Mercator(),
    'data': ccrs.PlateCarree(),
}

# ── Worker-level globals (populated by worker_initializer) ─────────────────
_worker_eds = None
_worker_cds = None
_worker_bathy_data = {}
_worker_argo_data = None
_worker_glider_data = None
_worker_region_configs = None
_worker_config = None
_worker_path_save = None
_worker_cache_dir = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_with_timeout(ds, timeout=120):
    """Load an xarray Dataset/DataArray, raising TimeoutError if it hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ds.load)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Data load timed out after {timeout}s — server may be unavailable")
        except _HTTPX_ERRORS as e:
            raise RuntimeError(f"HTTP error during data load (server may be unavailable): {e}") from e


def _ohc_cache_path(cache_dir, region, ctime, model_key):
    tstr = ctime.strftime('%Y%m%dT%H%M%SZ')
    return Path(cache_dir) / "ohc" / region / f"{tstr}_{model_key}.nc"


def _save_ohc(da, cache_dir, region, ctime, model_key):
    path = _ohc_cache_path(cache_dir, region, ctime, model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset(name='ohc').to_netcdf(path)


def _load_ohc(cache_dir, region, ctime, model_key):
    path = _ohc_cache_path(cache_dir, region, ctime, model_key)
    if path.exists():
        return xr.open_dataset(path)['ohc']
    return None


def _rename_lon_lat(ds):
    rename = {k: v for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items() if k in ds.coords}
    return ds.rename(rename) if rename else ds


def _build_recent_date_list(days, available_times=None):
    """
    Build a 6-hourly time window covering the last `days` days.

    When model timestamps are available, prefer those exact times so later
    `.sel(time=...)` lookups stay aligned with the dataset. The window ends at
    the current day's 24Z (next day's 00Z) when available, with a generic
    6-hour fallback if the probe fails.
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

    return pd.date_range(window_start, window_end, freq="6h")


def load_or_fetch_argo(extent, start_time, end_time, cache_dir):
    import pickle
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"argo_qcv1_{ext_str}_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.pkl"
    if cache_file.exists():
        logger.info(f"Loading Argo from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    logger.info("Fetching Argo data...")
    argo_data = get_argo_floats_by_time(
        extent,
        start_time,
        end_time,
        include_qc=ARGO_LOCATION_QC_VARIABLES,
    )
    raw_argo_count = len(argo_data)
    argo_data = filter_argo_by_qc(
        argo_data,
        qc_columns=ARGO_LOCATION_QC_VARIABLES,
        allowed_flags=ARGO_GOOD_QC_FLAGS,
    )
    logger.info("Argo QC retained %s/%s samples.", len(argo_data), raw_argo_count)
    with open(cache_file, 'wb') as f:
        pickle.dump(argo_data, f)
    return argo_data


def load_or_fetch_gliders(extent, start_time, end_time, cache_dir, parallel=False):
    import pickle
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"gliders_{ext_str}_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.pkl"
    if cache_file.exists():
        logger.info(f"Loading gliders from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    logger.info("Fetching glider data...")
    glider_data = get_active_gliders(extent, start_time, end_time, parallel=parallel)
    with open(cache_file, 'wb') as f:
        pickle.dump(glider_data, f)
    return glider_data


def load_or_fetch_bathymetry(extent, cache_dir):
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"bathy_{ext_str}.pkl"
    if cache_file.exists():
        logger.info(f"Loading bathymetry from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    logger.info(f"Fetching bathymetry for extent {extent}...")
    bathy_data = get_bathymetry(extent)
    # Convert to xarray so the result is picklable
    bathy_data = xr.Dataset({
        'z': xr.DataArray(bathy_data.z.values, dims=['latitude', 'longitude'],
                          coords={'latitude': bathy_data.latitude.values,
                                  'longitude': bathy_data.longitude.values})
    })
    with open(cache_file, 'wb') as f:
        pickle.dump(bathy_data, f)
    return bathy_data


# ── Worker initializer ─────────────────────────────────────────────────────

def worker_initializer(
    global_extent_buffered,
    lon_transform,
    bathy_data,
    argo_data,
    glider_data,
    region_configs,
    config_dict,
    path_save,
    cache_dir,
):
    global _worker_eds, _worker_cds
    global _worker_bathy_data, _worker_argo_data, _worker_glider_data
    global _worker_region_configs, _worker_config, _worker_path_save
    global _worker_cache_dir

    _worker_bathy_data = bathy_data
    _worker_argo_data = argo_data
    _worker_glider_data = glider_data
    _worker_region_configs = region_configs
    _worker_config = config_dict
    _worker_path_save = path_save
    _worker_cache_dir = cache_dir

    if config_dict['plot_espc']:
        logger.info("ESPC: using FMRC best-forecast")
        eds_local = espc_ts(rename=True, chunks={'time': 1, 'depth': 5})
        _worker_eds = eds_local.sel(
            lon=slice(lon_transform[0], lon_transform[1]),
            lat=slice(global_extent_buffered[2], global_extent_buffered[3]),
            depth=slice(0, 400),
        )

    if config_dict['plot_cmems']:
        cobj = CMEMS()
        lon_slice = slice(global_extent_buffered[0], global_extent_buffered[1])
        lat_slice = slice(global_extent_buffered[2], global_extent_buffered[3])
        depth_slice = slice(0, 400)
        _worker_cds = {
            'temperature': cobj.get_variable('temperature').sel(longitude=lon_slice, latitude=lat_slice, depth=depth_slice),
            'salinity': cobj.get_variable('salinity').sel(longitude=lon_slice, latitude=lat_slice, depth=depth_slice),
        }

    logger.info("Worker %s initialized", mp.current_process().name)


# ── Per-timestamp processing ───────────────────────────────────────────────

def process_time(ctime):
    """Fetch OHC for all models at a single timestep and generate plots."""
    eds = _worker_eds
    cds = _worker_cds
    bathy_by_region = _worker_bathy_data
    argo_data = _worker_argo_data
    glider_data = _worker_glider_data
    region_configs = _worker_region_configs
    config = _worker_config
    path_save = _worker_path_save
    cache_dir = _worker_cache_dir

    logger.info(f"Processing {ctime}")
    plots_generated = []

    eds_time = None
    cds_time = None
    edt_flag = cdt_flag = False

    if config['plot_espc'] and eds is not None:
        try:
            eds_time = eds.sel(time=ctime)
            edt_flag = True
        except KeyError:
            logger.warning(f"ESPC: no data for {ctime}")

    if config['plot_cmems'] and cds is not None:
        try:
            temperature = cds['temperature'].sel(time=ctime, method='nearest')
            salinity = cds['salinity'].sel(time=ctime, method='nearest')
            cds_time = xr.Dataset({'temperature': temperature, 'salinity': salinity})
            cds_time = _rename_lon_lat(cds_time)
            cds_time.attrs['model'] = 'CMEMS'
            cdt_flag = True
        except KeyError:
            logger.warning(f"CMEMS: no data for {ctime}")

    for region_key in config['regions']:
        configs = region_configs[region_key]
        extent = configs["extent"]
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()
        bathy_data = bathy_by_region.get(region_key)

        search_window_t0 = ctime - pd.Timedelta(hours=config.get('search_hours', conf.search_hours))
        search_window_t1 = ctime

        ohc_espc = None
        if edt_flag:
            try:
                ohc_espc = _load_ohc(cache_dir, region_key, ctime, 'espc')
                if ohc_espc is None:
                    eds_slice = eds_time.sel(
                        lon=slice(extent_data[0] - 2, extent_data[1] + 2),
                        lat=slice(extent_data[2] - 2, extent_data[3] + 2),
                    )
                    logger.info(f"  [{ctime}] Loading ESPC data ({region_key})...")
                    _load_with_timeout(eds_slice)
                    ohc_espc = compute_ohc_vectorized(eds_slice)
                    _save_ohc(ohc_espc, cache_dir, region_key, ctime, 'espc')
                    logger.info(f"  [{ctime}] Cached ESPC OHC → {_ohc_cache_path(cache_dir, region_key, ctime, 'espc')}")
                else:
                    logger.info(f"  [{ctime}] Loaded ESPC OHC from cache")
            except Exception as e:
                logger.error(f"  [{ctime}] ESPC OHC failed for {region_key}: {e}\n{traceback.format_exc()}")

        ohc_cmems = None
        if cdt_flag:
            try:
                ohc_cmems = _load_ohc(cache_dir, region_key, ctime, 'cmems')
                if ohc_cmems is None:
                    cds_slice = cds_time.sel(
                        lon=slice(extent_data[0], extent_data[1]),
                        lat=slice(extent_data[2], extent_data[3]),
                    )
                    _load_with_timeout(cds_slice)
                    ohc_cmems = compute_ohc_vectorized(cds_slice)
                    _save_ohc(ohc_cmems, cache_dir, region_key, ctime, 'cmems')
                    logger.info(f"  [{ctime}] Cached CMEMS OHC → {_ohc_cache_path(cache_dir, region_key, ctime, 'cmems')}")
                else:
                    logger.info(f"  [{ctime}] Loaded CMEMS OHC from cache")
            except Exception as e:
                logger.error(f"  [{ctime}] CMEMS OHC failed for {region_key}: {e}\n{traceback.format_exc()}")

        if ohc_espc is None and ohc_cmems is None:
            logger.warning(f"  [{ctime}] No OHC data for {region_key}, skipping plot")
            continue

        # ── Platforms ─────────────────────────────────────────────────
        argo = gliders = pd.DataFrame()
        if not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            argo_region = argo_data[mask].copy().sort_index()
            try:
                argo = argo_region.loc[pd.IndexSlice[:, search_window_t0:search_window_t1], :]
            except KeyError:
                pass

        if not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            glider_region = glider_data[mask]
            gliders = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                & (glider_region.index.get_level_values('time') < search_window_t1)
            ]

        # ── Plot ──────────────────────────────────────────────────────
        try:
            plot_ohc_all_models(
                ohc_rtofs=None,
                ohc_espc=ohc_espc,
                ohc_cmems=ohc_cmems,
                extent=extent,
                region_name=configs["name"],
                ctime=ctime,
                bathy=bathy_data,
                eez=configs.get('eez', True),
                path_save=path_save / configs["folder"],
                transform=GUAM_TRANSFORM,
                dpi=config['dpi'],
                overwrite=config.get('overwrite', False),
                argo=argo,
                gliders=gliders,
                ohc_min=0,
                ohc_max=150,
                ohc_stride=10,
                location_marker=dict(lon=144.73, lat=13.41, label='Guam'),
            )
            plots_generated.append(f"{region_key} | OHC | ESPC vs CMEMS")
        except Exception as e:
            logger.error(f"  [{ctime}] Plot failed for {region_key}: {e}\n{traceback.format_exc()}")

    return {'ctime': ctime, 'plots': plots_generated}


# ── Main ───────────────────────────────────────────────────────────────────

def main(parallel=True, max_workers=None):
    start_time_exec = time.time()

    plot_espc = True
    plot_cmems = True
    replot = False

    conf.days = 10
    path_save = conf.path_plots / "maps"

    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)

    conf.regions = ['guam']
    region_configs = {r: region_config(r) for r in conf.regions}

    extent_list = [cfg["extent"] for cfg in region_configs.values()]
    global_extent = [
        min(e[0] for e in extent_list), max(e[1] for e in extent_list),
        min(e[2] for e in extent_list), max(e[3] for e in extent_list),
    ]
    global_extent_buffered = [
        global_extent[0] - 3, global_extent[1] + 3,
        global_extent[2] - 3, global_extent[3] + 3,
    ]
    # Guam lons are already in 0-360 space
    lon_transform = lon180to360(global_extent_buffered[:2])

    logger.info(f"Global extent: {global_extent}")

    # ── ESPC: probe timestamps once to define the recent window ────────────
    eds_probe = None
    if plot_espc:
        try:
            logger.info("ESPC: using FMRC best-forecast")
            eds_probe = espc_ts(rename=True, chunks={'time': 1, 'depth': 5})
        except Exception as e:
            logger.error(f"ESPC load failed: {e}\n{traceback.format_exc()}")

    date_list = _build_recent_date_list(conf.days, eds_probe.time.values if eds_probe is not None else None)
    date_start = date_list[0]
    date_end = date_list[-1]
    search_start = date_start - dt.timedelta(hours=conf.search_hours)
    del eds_probe

    logger.info(
        "Using %s timestamps from %s through %s",
        len(date_list),
        date_start,
        date_end,
    )
    for t in date_list:
        logger.info(f"  {t}")

    # ── Platform and bathymetry data ───────────────────────────────────────
    argo_data = pd.DataFrame()
    if conf.argo:
        try:
            argo_data = load_or_fetch_argo(global_extent, search_start, date_end, cache_dir)
        except Exception as e:
            logger.error(f"Argo fetch failed, continuing without Argo data: {e}\n{traceback.format_exc()}")

    glider_data = pd.DataFrame()
    if conf.gliders:
        try:
            glider_data = load_or_fetch_gliders(global_extent, search_start, date_end, cache_dir)
        except Exception as e:
            logger.error(f"Glider fetch failed, continuing without glider data: {e}\n{traceback.format_exc()}")

    bathy_data = {}
    if conf.bathy:
        try:
            bathy_data = {
                region: load_or_fetch_bathymetry(region_configs[region]["extent"], cache_dir)
                for region in conf.regions
            }
        except Exception as e:
            logger.error(f"Bathymetry fetch failed, continuing without bathymetry: {e}\n{traceback.format_exc()}")

    config_dict = {
        'regions': conf.regions,
        'dpi': conf.dpi,
        'search_hours': conf.search_hours,
        'plot_espc': plot_espc,
        'plot_cmems': plot_cmems,
        'overwrite': replot,
    }

    init_args = (
        global_extent_buffered,
        lon_transform,
        bathy_data,
        argo_data,
        glider_data,
        region_configs,
        config_dict,
        path_save,
        cache_dir,
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
