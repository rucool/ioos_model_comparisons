"""One-off: salinity-only comparison plots (RTOFS vs ESPC, RTOFS vs CMEMS)
with the salinity colorbar minimum forced to 31 at every depth, overriding
whatever regions.py / MongoDB colorbar config would otherwise set.

Trimmed from rtofs-gofs-cmems-amseas.py — no temperature, currents, or SST
plots are generated.
"""
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
from ioos_model_comparisons.models import rtofs, CMEMS, espc_ts
from ioos_model_comparisons.platforms import (
    get_active_gliders,
    get_argo_floats_by_time,
    )
from ioos_model_comparisons.plotting import plot_model_region_comparison
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.db import (
    apply_colorbar_overrides,
    ensure_plot_index,
    log_plots,
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

# Distinct SCRIPT_ID so this one-off's plot log doesn't interact with the
# production "wfs" pre-check/log.
SCRIPT_ID = "wfs_salinity_min31"

# ── Model selection flags ─────────────────────────────────────────────────────
plot_rtofs = True
plot_espc  = True
plot_cmems = True

# ── Salinity colorbar override for this one-off run ──────────────────────────
SALINITY_MIN = 30

# ── Parallelization ───────────────────────────────────────────────────────────
parallel    = True
max_workers = 4

# ── Keyword arguments for map plots ──────────────────────────────────────────
kwargs = {
    'transform': conf.projection,
    'dpi':       conf.dpi,
    'overwrite': True,
    'colorbar':  True,
    'legend':    True,
}

# ── Date / region configuration ───────────────────────────────────────────────
conf.days    = 1
conf.regions = [
    'gom',
    ]
date_start = pd.Timestamp(2026, 7, 20, 0, 0, 0)
date_end   = pd.Timestamp(2026, 7, 20, 0, 0, 0)
freq        = '6H'
date_list   = pd.date_range(date_start, date_end, freq=freq)

search_start  = date_list[0] - dt.timedelta(hours=conf.search_hours)
extent_list   = [region_config(region)["extent"] for region in conf.regions]
extent_df     = pd.DataFrame(extent_list, columns=['lonmin', 'lonmax', 'latmin', 'latmax'])
global_extent = [extent_df.lonmin.min(), extent_df.lonmax.max(),
                 extent_df.latmin.min(), extent_df.latmax.max()]

# ── Worker-level globals ──────────────────────────────────────────────────────
_worker_rds    = None   # RTOFS dataset
_worker_gds_ts = None   # ESPC temperature/salinity dataset
_worker_cmems  = None   # CMEMS instance (one per worker)

# Platform data defaults (overwritten by worker_initializer)
argo_data   = pd.DataFrame()
glider_data = pd.DataFrame()
bathy_data  = None
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


def region_salinity_only(item):
    """region_config + MongoDB colorbar overrides, then restricted to
    salinity-only with the minimum limit forced to SALINITY_MIN at every
    depth. Applied after apply_colorbar_overrides so it always wins."""
    region = region_config(item)
    region = apply_colorbar_overrides(item, region)
    region = copy.deepcopy(region)
    region['variables'] = {'salinity': region['variables']['salinity']}
    for depth_entry in region['variables']['salinity']:
        limits = list(depth_entry['limits'])
        limits[0] = SALINITY_MIN
        depth_entry['limits'] = limits
    region['currents'] = dict(region.get('currents', {}), bool=False)
    return region


# ── Worker initializer ────────────────────────────────────────────────────────
def worker_initializer(argo_df, glider_df, bathy):
    """Called once per worker process by ProcessPoolExecutor.

    Re-opens OPeNDAP model connections (not picklable) and assigns pre-loaded
    platform data (pandas/numpy, picklable) from the main process.
    """
    global _worker_rds, _worker_gds_ts, _worker_cmems
    global argo_data, glider_data, bathy_data
    global grid_lons, grid_lats, grid_x, grid_y

    _worker_rds    = load_model(rtofs,   'RTOFS')   if plot_rtofs else None
    _worker_gds_ts = load_model(espc_ts, 'ESPC TS') if plot_espc  else None
    _worker_cmems  = CMEMS()                        if plot_cmems else None

    argo_data   = argo_df
    glider_data = glider_df
    bathy_data  = bathy

    if _worker_rds is not None:
        grid_lons = _worker_rds.lon.values[0, :]
        grid_lats = _worker_rds.lat.values[:, 0]
        grid_x    = _worker_rds.x.values
        grid_y    = _worker_rds.y.values


# ── Per-timestamp processing ──────────────────────────────────────────────────
def process_time(ctime):
    """Process all regions for a single timestamp. Runs inside a worker process."""
    rdt_flag, rdt    = attempt_data_load(_worker_rds,    ctime, "RTOFS")
    gdt_flag, gdt_ts = attempt_data_load(_worker_gds_ts, ctime, "ESPC TS") if plot_espc else (False, None)

    plots_generated  = []
    all_pending_logs = []
    for item in conf.regions:
        region       = region_salinity_only(item)
        cmems_extent = np.add(region['extent'], [-1, 1, -1, 1]).tolist()
        cdt_flag, cdt = attempt_cmems_data_load(_worker_cmems, ctime, cmems_extent) if plot_cmems else (False, None)

        plots, pending_logs = process_region(
            ctime, rdt_flag, rdt, gdt_flag, gdt_ts, cdt_flag, cdt, region
        )
        plots_generated.extend(plots)
        all_pending_logs.extend(pending_logs)

    return {'ctime': ctime, 'plots': plots_generated, 'pending_logs': all_pending_logs}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global argo_data, glider_data, bathy_data

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

    ensure_plot_index()

    # One-off: always replot every configured timestamp — no MongoDB/filesystem
    # pre-check. Relies on kwargs['overwrite'] = True to replace existing files.
    date_list_pending = date_list

    # ── Dispatch timestamps ───────────────────────────────────────────────────
    all_results = []
    init_args   = (argo_data, glider_data, bathy_data)

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
                    log_plots(SCRIPT_ID, result['pending_logs'])
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


def process_region(ctime, rdt_flag, rdt, gdt_flag, gdt_ts, cdt_flag, cdt, region):
    """Process a specific region for the given time — salinity comparisons only."""
    extent = region['extent']
    logger.info(f"Subsetting data for region: {region['name']} with extent {extent} at time {ctime}")
    kwargs['path_save'] = path_save / region['folder']

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)

    if 'eez' in region:
        kwargs["eez"] = region["eez"]

    if 'figure' in region:
        if 'legend' in region['figure']:
            kwargs['cols'] = region['figure']['legend']['columns']
        if 'figsize' in region['figure']:
            kwargs['figsize'] = region['figure']['figsize']

    extended = np.add(extent, [-1, 1, -1, 1]).tolist()
    lon360   = lon180to360(extended[:2])

    try:
        # Subset data based on the region extent
        rds_sub  = subset_data(rdt, extended, grid_lons, grid_lats, grid_x, grid_y) if rdt_flag else None
        gds_ts_s = subset_data_lonlat(gdt_ts, lon360, extended) if gdt_flag else None
        if gds_ts_s is not None:
            gds_ts_s['lon'] = lon360to180(gds_ts_s['lon'])
        cds_sub = cdt

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

        plots        = []
        pending_logs = []
        ts_dt        = pd.to_datetime(ctime).to_pydatetime()

        if rdt_flag and gdt_flag:
            try:
                plot_model_region_comparison(rds_sub, gds_ts_s, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs ESPC (salinity)")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'espc'))
                logger.info(f"Successfully plotted RTOFS vs ESPC salinity for region {region['name']} at time {ctime}")
            except Exception as e:
                logger.error(f"Failed to process RTOFS vs ESPC salinity at {ctime} for region {region['name']}: {e}", exc_info=True)

        try:
            if rdt_flag and cdt_flag:
                plot_model_region_comparison(rds_sub, cds_sub, region, **kwargs)
                plots.append(f"{region['name']} | RTOFS vs CMEMS (salinity)")
                pending_logs.extend(_mc_records(region, ts_dt, 'rtofs', 'cmems'))
                logger.info(f"Successfully plotted RTOFS vs CMEMS salinity for region {region['name']} at time {ctime}")
        except Exception as e:
            logger.error(f"Failed to process RTOFS vs CMEMS salinity at {ctime} for region {region['name']}: {e}")

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


def _mc_records(region, ts_dt, m1, m2):
    return [
        {"region": region['name'], "timestamp": ts_dt, "plot_type": "model_comparison",
         "variable": k, "depth": de['depth'], "model1": m1, "model2": m2}
        for k, dl in region['variables'].items() for de in dl
    ]


if __name__ == "__main__":
    main()
