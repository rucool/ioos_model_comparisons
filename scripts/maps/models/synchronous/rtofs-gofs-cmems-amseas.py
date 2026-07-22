import datetime as dt
import time
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import copy
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import rtofs, amseas, CMEMS, espc_ts, espc_uv, cnaps
from ioos_model_comparisons.platforms import (
    get_active_gliders,
    get_argo_floats_by_time, get_goes
    )
from ioos_model_comparisons.plotting import (
    plot_model_region_comparison,
    plot_model_region_comparison_streamplot,
    plot_sst
)
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.db import (
    apply_colorbar_overrides,
    ensure_plot_index,
    fetch_completed_plot_keys,
    log_plots,
    needs_replot,
)
from cool_maps.plot import get_bathymetry

matplotlib.use('agg')  # Set matplotlib to use non-interactive backend

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Save path ─────────────────────────────────────────────────────────────────
path_save = conf.path_plots / "maps"

SCRIPT_ID = "wfs"

# ── Model selection flags ─────────────────────────────────────────────────────
plot_rtofs  = True
plot_para   = False
plot_espc   = True
plot_cmems  = True
plot_amseas = False
plot_cnaps  = False
plot_doppio = False

# ── Parallelization ───────────────────────────────────────────────────────────
parallel    = True
max_workers = 4   # one worker per concurrent timestamp

# ── Keyword arguments for map plots ──────────────────────────────────────────
kwargs = {
    'transform': conf.projection,
    'dpi':       conf.dpi,
    'overwrite': False,
    'colorbar':  True,
    'legend':    True,
}

# ── Date / region configuration ───────────────────────────────────────────────
conf.days    = 1
conf.regions = [
    'caribbean',
    'gom', 
    'tropical_western_atlantic', 
    'mab', 
    'sab', 
    'west_florida_shelf',
    'windward',
    ]
# conf.regions = ['mab']
today       = dt.date.today()
date_start  = today - dt.timedelta(days=conf.days)
date_end    = today + dt.timedelta(days=1)
freq        = '6H'
date_list   = pd.date_range(date_start, date_end, freq=freq)
date_list_2 = pd.date_range(date_start - dt.timedelta(days=1), date_end, freq=freq)

search_start  = date_list[0] - dt.timedelta(hours=conf.search_hours)
extent_list   = [region_config(region)["extent"] for region in conf.regions]
extent_df     = pd.DataFrame(extent_list, columns=['lonmin', 'lonmax', 'latmin', 'latmax'])
global_extent = [extent_df.lonmin.min(), extent_df.lonmax.max(),
                 extent_df.latmin.min(), extent_df.latmax.max()]

# ── Worker-level globals ──────────────────────────────────────────────────────
# Populated by worker_initializer() in each spawned process.
# Also used directly in serial mode after worker_initializer() is called in main().
_worker_rds    = None   # RTOFS dataset
_worker_rdsp   = None   # RTOFS Parallel dataset
_worker_gds_ts = None   # ESPC temperature/salinity dataset
_worker_gds_uv = None   # ESPC UV dataset
_worker_cmems  = None   # CMEMS instance (one per worker)
_worker_am     = None   # AMSEAS dataset
_worker_cn     = None   # CNAPS dataset
_doppio_cache  = {}     # {ctime: xr.Dataset} pre-fetched in main, read-only in workers

# Platform data defaults (overwritten by worker_initializer)
argo_data     = pd.DataFrame()
glider_data   = pd.DataFrame()
sst_sorted_16 = None
sst_sorted_19 = None
bathy_data    = None
grid_lons   = None
grid_lats   = None
grid_x      = None
grid_y      = None


# ── Model loader helper ───────────────────────────────────────────────────────
def load_model(model_func, model_name, source=None, rename=True):
    try:
        logger.info(f'Loading {model_name} model data.')
        model_data = model_func(rename=rename, source=source) if source else model_func(rename=rename)
        logger.info(f'{model_name} model data loaded successfully.')
        return model_data
    except Exception as e:
        logger.error(f"Failed to load {model_name} model data: {e}")
        return None


# ── Worker initializer ────────────────────────────────────────────────────────
def worker_initializer(argo_df, glider_df, sst_data_16, sst_data_19, bathy, region_cfgs, kw, path_sv, doppio_cache):
    """Called once per worker process by ProcessPoolExecutor.

    Re-opens OPeNDAP model connections (not picklable) and assigns pre-loaded
    platform data (pandas/numpy, picklable) from the main process.
    """
    global _worker_rds, _worker_rdsp, _worker_gds_ts, _worker_gds_uv
    global _worker_cmems, _worker_am, _worker_cn, _doppio_cache
    global argo_data, glider_data, sst_sorted_16, sst_sorted_19, bathy_data
    global grid_lons, grid_lats, grid_x, grid_y

    _worker_rds    = load_model(rtofs,   'RTOFS')                             if plot_rtofs  else None
    _worker_rdsp   = load_model(rtofs,   'RTOFS Parallel', source='parallel') if plot_para   else None
    _worker_gds_ts = load_model(espc_ts, 'ESPC TS')                           if plot_espc   else None
    _worker_gds_uv = load_model(espc_uv, 'ESPC UV')                           if plot_espc   else None
    _worker_am     = load_model(amseas,  'AMSEAS')                             if plot_amseas else None
    _worker_cn     = load_model(cnaps,   'CNAPS')                              if plot_cnaps  else None
    _doppio_cache  = doppio_cache  # pre-fetched in main; no OPeNDAP in workers
    _worker_cmems  = CMEMS()                                                   if plot_cmems  else None

    argo_data     = argo_df
    glider_data   = glider_df
    sst_sorted_16 = sst_data_16
    sst_sorted_19 = sst_data_19
    bathy_data    = bathy

    if _worker_rds is not None:
        grid_lons = _worker_rds.lon.values[0, :]
        grid_lats = _worker_rds.lat.values[:, 0]
        grid_x    = _worker_rds.x.values
        grid_y    = _worker_rds.y.values


# ── Per-timestamp processing ──────────────────────────────────────────────────
def process_time(ctime):
    """Process all regions for a single timestamp. Runs inside a worker process."""
    rdt_flag,  rdt    = attempt_data_load(_worker_rds,    ctime, "RTOFS")
    rdtp_flag, rdtp   = attempt_data_load(_worker_rdsp,   ctime, "RTOFS Parallel") if plot_para   else (False, None)
    amt_flag,  amt    = attempt_data_load(_worker_am,     ctime, "AMSEAS")          if plot_amseas else (False, None)
    cnt_flag,  cnt    = attempt_data_load(_worker_cn,     ctime, "CNAPS")           if plot_cnaps  else (False, None)
    if plot_doppio:
        dpt = _doppio_cache.get(ctime)
        dpt_flag = dpt is not None
    else:
        dpt_flag, dpt = False, None
    gdt_flag,  gdt_ts = attempt_data_load(_worker_gds_ts, ctime, "ESPC TS")         if plot_espc   else (False, None)
    _,         gdt_uv = attempt_data_load(_worker_gds_uv, ctime, "ESPC UV")         if plot_espc   else (False, None)

    plots_generated = []
    all_pending_logs = []
    for item in conf.regions:
        region    = region_config(item)
        region    = apply_colorbar_overrides(item, region)
        cmems_extent = np.add(region['extent'], [-1, 1, -1, 1]).tolist()
        cdt_flag, cdt = attempt_cmems_data_load(_worker_cmems, ctime, cmems_extent) if plot_cmems else (False, None)

        plots, pending_logs = process_region(
            ctime, rdt_flag, rdt, rdtp_flag, rdtp,
            gdt_flag, gdt_ts, gdt_uv, cdt_flag, cdt,
            amt_flag, amt, cnt_flag, cnt, dpt_flag, dpt, region
        )
        plots_generated.extend(plots)
        all_pending_logs.extend(pending_logs)

    return {'ctime': ctime, 'plots': plots_generated, 'pending_logs': all_pending_logs}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global argo_data, glider_data, sst_sorted_16, sst_sorted_19, bathy_data

    start_time_exec = time.time()

    # Load platform data once in the main process (passed to workers via initargs)
    if conf.argo:
        try:
            argo_data = get_argo_floats_by_time(global_extent, search_start, date_end)
            logger.info(f"Argo data loaded ({len(argo_data)} records).")
        except Exception as e:
            logger.error(f"Failed to load Argo data: {e}")
            argo_data = pd.DataFrame()

    if conf.gliders:
        try:
            glider_data = get_active_gliders(global_extent, search_start, date_end, parallel=False, timeout=60)
            logger.info(f"Glider data loaded ({len(glider_data)} records).")
            # When a glider has multiple active deployments, keep only the most
            # recent one — the deployment suffix is a sortable timestamp string.
            dep_ids = glider_data.index.get_level_values('glider').unique()
            short_to_latest = {}
            for dep_id in dep_ids:
                short = dep_id.rsplit("-", 1)[0]
                if short not in short_to_latest or dep_id > short_to_latest[short]:
                    short_to_latest[short] = dep_id
            keep = set(short_to_latest.values())
            glider_data = glider_data[glider_data.index.get_level_values('glider').isin(keep)]
            glider_data.index = glider_data.index.remove_unused_levels()
            glider_data.index = glider_data.index.set_levels(
                glider_data.index.levels[0].str.rsplit("-", n=1).str[0], level="glider"
            )
        except Exception as e:
            logger.error(f"Failed to load Glider data: {e}")
            glider_data = pd.DataFrame()

    try:
        bathy_data = get_bathymetry(global_extent) if conf.bathy else None
        logger.info("Bathymetry data loaded.")
    except Exception as e:
        logger.error(f"Failed to load Bathymetry data: {e}")
        bathy_data = None

    try:
        sst_sorted_16 = get_goes('goes16')
        logger.info("GOES-16 SST data loaded.")
    except Exception as e:
        logger.error(f"Failed to load GOES-16 SST data: {e}")
        sst_sorted_16 = None

    try:
        sst_sorted_19 = get_goes('goes19')
        logger.info("GOES-19 SST data loaded.")
    except Exception as e:
        logger.error(f"Failed to load GOES-19 SST data: {e}")
        sst_sorted_19 = None

    has_argo    = isinstance(argo_data,   pd.DataFrame) and not argo_data.empty
    has_gliders = isinstance(glider_data, pd.DataFrame) and not glider_data.empty

    ensure_plot_index()

    # ── Pre-check: skip timestamps whose outputs all already exist ───────────
    date_list_pending = pre_check_date_list(date_list, current_has_argo=has_argo, current_has_gliders=has_gliders,
                                             overwrite=kwargs.get('overwrite', False))
    if len(date_list_pending) == 0:
        logger.info("All outputs already exist. Nothing to do.")
        return

    # ── Pre-fetch Doppio sequentially (avoids concurrent OPeNDAP connections) ──
    doppio_cache = {}
    if plot_doppio:
        logger.info("Pre-fetching Doppio data for all pending timestamps...")
        from ioos_model_comparisons.models import Doppio
        _dop = Doppio()
        for ctime in date_list_pending:
            try:
                doppio_cache[ctime] = _dop.sel(time=ctime)
                logger.info(f"Doppio pre-fetched: {ctime}")
            except Exception as e:
                logger.warning(f"Doppio pre-fetch failed for {ctime}: {e}")
                doppio_cache[ctime] = None

    # ── Dispatch timestamps ───────────────────────────────────────────────────
    all_results = []
    init_args   = (argo_data, glider_data, sst_sorted_16, sst_sorted_19, bathy_data, conf.regions, kwargs, path_save, doppio_cache)

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers,
                                 initializer=worker_initializer,
                                 initargs=init_args) as executor:
            futures   = {executor.submit(process_time, ct): ct for ct in date_list_pending}
            completed = 0
            total     = len(futures)
            for future in as_completed(futures):
                ct = futures[future]
                completed += 1
                try:
                    result = future.result()
                    all_results.append(result)
                    log_plots(SCRIPT_ID, result['pending_logs'], has_argo=has_argo, has_gliders=has_gliders)
                    logger.info(f"[{completed}/{total}] {ct} done ({len(result['plots'])} plots)")
                except Exception as exc:
                    logger.error(f"[{completed}/{total}] {ct} ERROR: {exc}\n{traceback.format_exc()}")
    else:
        # Serial fallback: populate worker globals in-process, then loop
        worker_initializer(*init_args)
        for ct in date_list_pending:
            result = process_time(ct)
            all_results.append(result)
            log_plots(SCRIPT_ID, result['pending_logs'])

    # ── Summary ───────────────────────────────────────────────────────────────
    total_plots = sum(len(r['plots']) for r in all_results if isinstance(r, dict))
    elapsed     = time.time() - start_time_exec
    print(f"\n{'='*60}")
    print(f"SUMMARY — {total_plots} plot(s) generated")
    print(f"{'='*60}")
    for result in sorted(all_results,
                         key=lambda r: r['ctime'] if isinstance(r, dict) else pd.Timestamp.min):
        if not isinstance(result, dict):
            continue
        print(f"\n  {result['ctime'].strftime('%Y-%m-%d %H:%MZ')} ({len(result['plots'])} plots)")
        for p in result['plots']:
            print(f"    • {p}")
    print(f"\nTotal execution time: {elapsed / 60:.2f} min")
    logger.info(f"All processing complete. Total execution time: {elapsed / 60:.2f} min.")


def attempt_data_load(model, ctime, model_name):
    """Attempt to load data for a given model and time."""
    try:
        if model is None:
            raise ValueError(f"{model_name} model data is not available.")
        data = model.sel(time=ctime)
        # Some FMRC datasets (e.g. ESPC) expose a secondary time dimension (time1,
        # time2, …) alongside the primary "time".  sel(time=ctime) collapses the
        # primary dimension but leaves the others intact, producing an unexpected
        # extra axis that breaks downstream squeeze()/contourf calls.  Select the
        # nearest value on any remaining time-like dimension to flatten them.
        for dim in list(data.dims):
            if 'time' in dim.lower():
                try:
                    data = data.sel({dim: ctime}, method='nearest')
                except Exception:
                    data = data.isel({dim: 0})
        logger.info(f"{model_name}: Data successfully loaded for time {ctime}.")
        return True, data
    except (KeyError, ValueError, RuntimeError) as e:
        logger.warning(f"{model_name}: Data not available for time {ctime} - {e}")
        return False, None


def attempt_cmems_data_load(cmems_instance, ctime, extent):
    """Attempt to load CMEMS data for a given time and region extent."""
    try:
        if cmems_instance is None:
            raise ValueError("CMEMS instance is not initialized.")

        lon_extent = extent[:2]
        lat_extent = extent[2:]

        data = cmems_instance.get_combined_subset(lon_extent, lat_extent, time=ctime)

        if data is None:
            raise ValueError(f"No valid CMEMS data found for time {ctime}.")

        logger.info(f"CMEMS: Data successfully loaded for time {ctime}.")
        return True, data

    except (KeyError, ValueError, Exception) as e:
        logger.warning(f"CMEMS: Data not available for time {ctime} - {e}")
        return False, None


def subset_data_curvilinear(data, extent):
    """Subset a curvilinear-grid dataset to a bounding-box extent.

    Expects 2-D ``lon``/``lat`` coordinates and ``y``/``x`` dimensions.
    Returns the smallest isel rectangle that contains all points within extent.
    Falls back to the full grid if no points are found.
    """
    try:
        lon = data['lon'].values
        lat = data['lat'].values
        lonmin, lonmax, latmin, latmax = extent
        mask = (lon >= lonmin) & (lon <= lonmax) & (lat >= latmin) & (lat <= latmax)
        eta_i, xi_i = np.where(mask)
        if len(eta_i) == 0:
            return data
        return data.isel(
            y=slice(int(eta_i.min()), int(eta_i.max()) + 1),
            x=slice(int(xi_i.min()),  int(xi_i.max())  + 1),
        )
    except Exception as e:
        logger.error(f"Error during curvilinear data subsetting: {e}")
        return data


def process_region(ctime, rdt_flag, rdt, rdtp_flag, rdtp, gdt_flag, gdt_ts, gdt_uv, cdt_flag, cdt,
                   amt_flag, amt, cnt_flag, cnt, dpt_flag, dpt, region):
    """Process a specific region for the given time."""
    extent = region['extent']
    logger.info(f"Subsetting data for region: {region['name']} with extent {extent} at time {ctime}")
    kwargs['path_save'] = path_save / region['folder']

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)

    if 'eez' in region:
        kwargs["eez"] = region["eez"]

    if region['currents']['bool']:
        kwargs['currents'] = region['currents']

    if 'figure' in region:
        if 'legend' in region['figure']:
            kwargs['cols'] = region['figure']['legend']['columns']
        if 'figsize' in region['figure']:
            kwargs['figsize'] = region['figure']['figsize']

    extended = np.add(extent, [-1, 1, -1, 1]).tolist()
    lon360   = lon180to360(extended[:2])

    try:
        # Subset data based on the region extent
        rds_sub  = subset_data(rdt,    extended, grid_lons, grid_lats, grid_x, grid_y) if rdt_flag  else None
        rdtp_sub = subset_data(rdtp,   extended, grid_lons, grid_lats, grid_x, grid_y) if rdtp_flag else None
        gds_ts_s = subset_data_lonlat(gdt_ts, lon360, extended) if gdt_flag else None
        if gds_ts_s is not None:
            gds_ts_s['lon'] = lon360to180(gds_ts_s['lon'])
        gds_uv_s = subset_data_lonlat(gdt_uv, lon360, extended) if gdt_flag else None
        if gds_uv_s is not None:
            gds_uv_s['lon'] = lon360to180(gds_uv_s['lon'])
        cds_sub  = cdt
        am_sub   = subset_data_lonlat(amt, lon360, extended) if amt_flag else None
        dop_sub  = subset_data_curvilinear(dpt, extended)    if dpt_flag else None

        # Subset platform data to this region and time window
        if not argo_data.empty:
            lon  = argo_data['lon']
            lat  = argo_data['lat']
            mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
            argo_region = argo_data[mask]
            argo_region.sort_index(inplace=True)
            idx = pd.IndexSlice
            kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

        if not glider_data.empty:
            lon  = glider_data['lon']
            lat  = glider_data['lat']
            mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
            glider_region = glider_data[mask]
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                &
                (glider_region.index.get_level_values('time') < search_window_t1)
            ]
            kwargs['gliders'] = glider_region

        # Process SST data for each satellite
        sst16 = process_sst_data(sst_sorted_16, extent, ctime) if sst_sorted_16 is not None else None
        sst19 = process_sst_data(sst_sorted_19, extent, ctime) if sst_sorted_19 is not None else None
        if sst16 is None and sst19 is None:
            logger.warning(f"SST data unavailable for region {region['name']} at time {ctime}")

        # Plot data
        plots        = []
        pending_logs = []
        ts_dt        = pd.to_datetime(ctime).to_pydatetime()

        if rdt_flag and gdt_flag:
            try:
                plot_model_region_comparison(rds_sub, gds_ts_s, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs ESPC (TS)")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'espc'))
                logger.info(f"Successfully plotted RTOFS vs ESPC TS for region {region['name']} at time {ctime}")
            except Exception as e:
                logger.error(f"Failed to process RTOFS vs ESPC TS at {ctime} for region {region['name']}: {e}", exc_info=True)

            try:
                plot_model_region_comparison_streamplot(rds_sub, gds_uv_s, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs ESPC (currents)")
                pending_logs.extend(_sp_records(region, ts_dt, 'rtofs', 'espc'))
                logger.info(f"Successfully plotted RTOFS vs ESPC currents for region {region['name']} at time {ctime}")
            except Exception as e:
                logger.error(f"Failed to process RTOFS vs ESPC currents at {ctime} for region {region['name']}: {e}", exc_info=True)

        if rdt_flag and rdtp_flag:
            try:
                plot_model_region_comparison(rds_sub, rdtp_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs Parallel (TS)")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'rtofs'))
                logger.info(f"Successfully plotted RTOFS vs Parallel TS for region {region['name']} at time {ctime}")
            except Exception as e:
                logger.error(f"Failed to process RTOFS vs Parallel TS at {ctime} for region {region['name']}: {e}", exc_info=True)

            try:
                plot_model_region_comparison_streamplot(rds_sub, rdtp_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs Parallel (currents)")
                pending_logs.extend(_sp_records(region, ts_dt, 'rtofs', 'rtofs'))
                logger.info(f"Successfully plotted RTOFS vs Parallel currents for region {region['name']} at time {ctime}")
            except Exception as e:
                logger.error(f"Failed to process RTOFS vs Parallel currents at {ctime} for region {region['name']}: {e}", exc_info=True)

        try:
            if rdt_flag and cdt_flag:
                plot_model_region_comparison(rds_sub, cds_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs CMEMS")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'cmems'))
                logger.info(f"Successfully plotted RTOFS vs CMEMS for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs CMEMS at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and cdt_flag:
                plot_model_region_comparison_streamplot(rds_sub, cds_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs CMEMS (currents)")
                pending_logs.extend(_sp_records(region, ts_dt, 'rtofs', 'cmems'))
                logger.info(f"Successfully plotted RTOFS vs CMEMS currents for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs CMEMS currents at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and amt_flag:
                plot_model_region_comparison(rds_sub, am_sub, region, **kwargs)
                plot_model_region_comparison_streamplot(rds_sub, am_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs AMSEAS")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'amseas'))
                pending_logs.extend(_sp_records(region, ts_dt, 'rtofs', 'amseas'))
                logger.info(f"Successfully plotted RTOFS vs AMSEAS for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs AMSEAS at {ctime} for region {region['name']}: {e}")

        try:
            if rdt_flag and dpt_flag:
                plot_model_region_comparison(rds_sub, dop_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs Doppio")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'doppio'))
                logger.info(f"Successfully plotted RTOFS vs Doppio for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs Doppio at {ctime} for region {region['name']}: {e}", exc_info=True)

        try:
            if rdt_flag and dpt_flag:
                plot_model_region_comparison_streamplot(rds_sub, dop_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs Doppio (currents)")
                pending_logs.extend(_sp_records(region, ts_dt, 'rtofs', 'doppio'))
                logger.info(f"Successfully plotted RTOFS vs Doppio currents for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs Doppio currents at {ctime} for region {region['name']}: {e}", exc_info=True)

        if sst16 is not None:
            plot_sst(rds_sub, sst16, region, satellite='GOES-16', **remove_kwargs(['eez', 'currents', 'legend']))
            plots.append(f"{region['name']} | SST GOES-16")
            pending_logs.append(_sst_record(region, ts_dt, 'GOES16'))
            logger.info(f"Successfully plotted GOES-16 SST for region {region['name']} at time {ctime}")

        if sst19 is not None:
            plot_sst(rds_sub, sst19, region, satellite='GOES-19', **remove_kwargs(['eez', 'currents', 'legend']))
            plots.append(f"{region['name']} | SST GOES-19")
            pending_logs.append(_sst_record(region, ts_dt, 'GOES19'))
            logger.info(f"Successfully plotted GOES-19 SST for region {region['name']} at time {ctime}")

        return plots, pending_logs

    except Exception as e:
        logger.error(f"Failed to process region {region['name']} at time {ctime}: {e}")
        return [], []


def subset_data(data, extent, grid_lons, grid_lats, grid_x, grid_y):
    """Subset data based on the region extent."""
    try:
        lons_ind = np.interp(extent[:2], grid_lons, grid_x)
        lats_ind = np.interp(extent[2:], grid_lats, grid_y)
        extent_ind = [int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1])),
                      int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1]))]
        logger.debug(f"Subsetting data for extent indices: {extent_ind}")
        return data.isel(x=slice(extent_ind[0], extent_ind[1]),
                         y=slice(extent_ind[2], extent_ind[3])).set_coords(['u', 'v'])
    except Exception as e:
        logger.error(f"Error during data subsetting: {e}")
        return None


def subset_data_lonlat(data, lon_extent, lat_extent):
    """Subset data using longitude and latitude extents."""
    try:
        logger.debug(f"Subsetting {data.attrs['model']} data for lon extent: {lon_extent} and lat extent: {lat_extent}")
        return data.sel(lon=slice(lon_extent[0], lon_extent[1]),
                        lat=slice(lat_extent[2], lat_extent[3]))
    except Exception as e:
        logger.error(f"Error during lon/lat data subsetting: {e}")
        return None


def process_sst_data(sst_data: xr.DataArray, extent: list, ctime: dt.datetime) -> xr.DataArray:
    """Subset SST data to extent/time and normalize to SST_C (°C).

    GOES-16 (hourly): selects the nearest timestamp to ctime.
    GOES-19 (daily avg): selects the daily average for the date of ctime.
    """
    try:
        logger.debug(f"Processing SST data for extent: {extent} at time {ctime}")
        spatial = sst_data.sel(lon=slice(extent[0], extent[1]),
                               lat=slice(extent[2], extent[3]))
        if 'cleaned_sst' in sst_data.data_vars:
            # GOES-19: daily average in Celsius — select by date only
            sst = spatial.sel(time=str(ctime.date()), method='nearest')
            sst['SST_C'] = sst['cleaned_sst']
        else:
            # GOES-16: hourly in Kelvin — select nearest to requested hour
            sst = spatial.sel(time=str(ctime), method='nearest')
            sst['SST_C'] = (('lat', 'lon'), kelvin_to_celsius(sst['SST'].values))
        return sst
    except Exception as e:
        logger.error(f"Error during SST data processing: {e}")
        return None


def kelvin_to_celsius(kelvin_temps: np.ndarray) -> np.ndarray:
    """Convert temperature from Kelvin to Celsius."""
    try:
        return kelvin_temps - 273.15
    except Exception as e:
        logger.error(f"Error converting temperature from Kelvin to Celsius: {e}")
        return kelvin_temps


def remove_kwargs(keys: list) -> dict:
    """Return a copy of kwargs with specified keys removed."""
    try:
        new_kwargs = copy.deepcopy(kwargs)
        for key in keys:
            new_kwargs.pop(key, None)
        return new_kwargs
    except Exception as e:
        logger.error(f"Error removing kwargs: {e}")
        return kwargs


def _expected_outputs(ctime, region) -> set:
    """Return the set of output paths still needed for this (ctime, region).

    Mirrors the filename logic used by each plot_* function so that the
    main process can decide whether a timestamp is worth dispatching before
    any workers are spun up or model data is loaded.
    """
    t      = pd.to_datetime(ctime)
    tstr   = t.strftime("%Y-%m-%dT%H%M%SZ")
    folder = region['folder']
    dashed = "-".join(folder.split("_"))
    base   = path_save / folder
    overwrite_flag = kwargs.get('overwrite', False)
    needed = set()

    def _need(path):
        if overwrite_flag or not path.is_file():
            needed.add(path)

    # ── plot_model_region_comparison (one file per variable × depth) ──────────
    model_pairs = []
    if plot_rtofs and plot_espc:
        model_pairs.append(('rtofs', 'espc'))
    if plot_rtofs and plot_para:
        model_pairs.append(('rtofs', 'rtofs'))
    if plot_rtofs and plot_cmems:
        model_pairs.append(('rtofs', 'cmems'))
    if plot_rtofs and plot_amseas:
        model_pairs.append(('rtofs', 'amseas'))
    if plot_rtofs and plot_cnaps:
        model_pairs.append(('rtofs', 'cnaps'))
    if plot_rtofs and plot_doppio:
        model_pairs.append(('rtofs', 'doppio'))

    for m1, m2 in model_pairs:
        for k, depth_list in region['variables'].items():
            for depth_entry in depth_list:
                depth = depth_entry['depth']
                d = base / f"{k}_{depth}m" / t.strftime('%Y/%m')
                _need(d / f'{dashed}_{tstr}_{k}-{depth}m_{m1}-vs-{m2}.png')

    # ── plot_model_region_comparison_streamplot (one file per depth) ──────────
    if region['currents']['bool']:
        stream_pairs = []
        if plot_rtofs and plot_espc:
            stream_pairs.append(('rtofs', 'espc'))
        if plot_rtofs and plot_para:
            stream_pairs.append(('rtofs', 'rtofs'))
        if plot_rtofs and plot_cmems:
            stream_pairs.append(('rtofs', 'cmems'))
        if plot_rtofs and plot_amseas:
            stream_pairs.append(('rtofs', 'amseas'))
        if plot_rtofs and plot_doppio:
            stream_pairs.append(('rtofs', 'doppio'))

        for m1, m2 in stream_pairs:
            for depth in region['currents']['depths']:
                d = base / f"currents_{depth}m" / t.strftime('%Y/%m')
                _need(d / f'{folder}_{tstr}_currents-{depth}m_{m1}-vs-{m2}.png')

    # ── plot_sst (no .png extension — matches how plot_sst saves) ────────────
    if plot_rtofs:
        sst_tags = ((['GOES16'] if sst_sorted_16 is not None else []) +
                    (['GOES19'] if sst_sorted_19 is not None else []))
        for satellite_tag in sst_tags:
            d = base / "temperature_0m" / t.strftime('%Y/%m')
            _need(d / f'{dashed}_{tstr}_temperature-0m_rtofs-vs-{satellite_tag}')

    return needed


def _expected_plot_keys(ctime, region) -> set:
    """Return the set of (region, iso_ts, plot_type, variable, depth, m1, m2) tuples
    expected for this (ctime, region) — no file I/O.

    Mirrors _expected_outputs() but yields structured keys for MongoDB comparison.
    """
    t      = pd.to_datetime(ctime)
    tstr_k = t.strftime("%Y-%m-%dT%H%M%SZ")
    rname  = region['name']
    keys   = set()

    model_pairs = []
    if plot_rtofs and plot_espc:
        model_pairs.append(('rtofs', 'espc'))
    if plot_rtofs and plot_para:
        model_pairs.append(('rtofs', 'rtofs'))
    if plot_rtofs and plot_cmems:
        model_pairs.append(('rtofs', 'cmems'))
    if plot_rtofs and plot_amseas:
        model_pairs.append(('rtofs', 'amseas'))
    if plot_rtofs and plot_cnaps:
        model_pairs.append(('rtofs', 'cnaps'))
    if plot_rtofs and plot_doppio:
        model_pairs.append(('rtofs', 'doppio'))

    for m1, m2 in model_pairs:
        for k, depth_list in region['variables'].items():
            for depth_entry in depth_list:
                keys.add((rname, tstr_k, "model_comparison", k, depth_entry['depth'], m1, m2))

    if region['currents']['bool']:
        stream_pairs = []
        if plot_rtofs and plot_espc:
            stream_pairs.append(('rtofs', 'espc'))
        if plot_rtofs and plot_para:
            stream_pairs.append(('rtofs', 'rtofs'))
        if plot_rtofs and plot_cmems:
            stream_pairs.append(('rtofs', 'cmems'))
        if plot_rtofs and plot_amseas:
            stream_pairs.append(('rtofs', 'amseas'))
        if plot_rtofs and plot_doppio:
            stream_pairs.append(('rtofs', 'doppio'))
        for m1, m2 in stream_pairs:
            for depth in region['currents']['depths']:
                keys.add((rname, tstr_k, "streamplot", "currents", depth, m1, m2))

    if plot_rtofs:
        for sat in (['GOES16'] if sst_sorted_16 is not None else []) + \
                   (['GOES19'] if sst_sorted_19 is not None else []):
            keys.add((rname, tstr_k, "sst", "temperature", 0, "rtofs", sat))

    return keys


def _mc_records(region, ts_dt, m1, m2):
    return [
        {"region": region['name'], "timestamp": ts_dt, "plot_type": "model_comparison",
         "variable": k, "depth": de['depth'], "model1": m1, "model2": m2}
        for k, dl in region['variables'].items() for de in dl
    ]


def _sp_records(region, ts_dt, m1, m2):
    return [
        {"region": region['name'], "timestamp": ts_dt, "plot_type": "streamplot",
         "variable": "currents", "depth": depth, "model1": m1, "model2": m2}
        for depth in region['currents']['depths']
    ]


def _sst_record(region, ts_dt, satellite_tag):
    return {"region": region['name'], "timestamp": ts_dt, "plot_type": "sst",
            "variable": "temperature", "depth": 0, "model1": "rtofs", "model2": satellite_tag}


def pre_check_date_list(date_list, current_has_argo=False, current_has_gliders=False, overwrite=False) -> pd.DatetimeIndex:
    """Return a filtered DatetimeIndex with only timestamps that need processing.

    Tries MongoDB first (one round-trip for all timestamps); falls back to
    per-file filesystem checks if the database is unavailable.

    A timestamp is requeued when:
      - any expected plot record is missing from MongoDB, OR
      - a record exists but was plotted without Argo/glider data that is now available.

    overwrite=True bypasses this check entirely and returns date_list unchanged —
    otherwise a timestamp already logged as done in MongoDB is skipped even with
    overwrite set, since the per-file overwrite check inside plot_* never runs
    for timestamps dropped at this stage.
    """
    if overwrite:
        logger.info(f"Pre-check: overwrite=True — reprocessing all {len(date_list)} timestamp(s).")
        return pd.DatetimeIndex(date_list)

    needed_dates = []
    skipped = 0

    # Build expected plot keys per timestamp — no I/O, just logic
    expected_by_ts = {}
    for ctime in date_list:
        keys = set()
        for item in conf.regions:
            region = region_config(item)
            region = apply_colorbar_overrides(item, region)
            keys |= _expected_plot_keys(ctime, region)
        expected_by_ts[ctime] = keys

    done_keys = fetch_completed_plot_keys(SCRIPT_ID, list(date_list))

    if done_keys is not None:
        for ctime in date_list:
            if any(needs_replot(k, done_keys, current_has_argo, current_has_gliders)
                   for k in expected_by_ts[ctime]):
                needed_dates.append(ctime)
            else:
                skipped += 1
        logger.info(f"Pre-check (MongoDB): {skipped}/{len(date_list)} timestamp(s) fully done — skipping.")
    else:
        logger.warning("MongoDB unavailable — falling back to file-existence pre-check.")
        for ctime in date_list:
            pending = set()
            for item in conf.regions:
                region = region_config(item)
                region = apply_colorbar_overrides(item, region)
                pending |= _expected_outputs(ctime, region)
            if pending:
                needed_dates.append(ctime)
            else:
                skipped += 1
        logger.info(f"Pre-check (filesystem): {skipped}/{len(date_list)} timestamp(s) fully done — skipping.")

    logger.info(f"Pre-check: {len(needed_dates)}/{len(date_list)} timestamp(s) queued.")
    return pd.DatetimeIndex(needed_dates)


if __name__ == "__main__":
    main()
