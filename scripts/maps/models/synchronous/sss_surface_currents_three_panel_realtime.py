"""
Sea surface salinity and surface currents three-panel plots — real-time.

Produces separate salinity and current-speed products for the most recent
RTOFS timestamps. ESPC and CMEMS are regridded to the RTOFS curvilinear grid
using xesmf so all model panels share the same map grid.
"""
import datetime as dt
import time
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import multiprocessing as mp

import ioos_model_comparisons.configs as conf
import numpy as np
import pandas as pd
import xarray as xr
from ioos_model_comparisons.calc import lon180to360, lon360to180, regrid_to_rtofs
from ioos_model_comparisons.models import CMEMS as c
from ioos_model_comparisons.models import espc_ts
from ioos_model_comparisons.models import rtofs as r
from ioos_model_comparisons.plotting import plot_sss_three_panel, plot_speed_diff_three_panel
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
from cool_maps.plot import get_bathymetry

# ── Worker-level globals (populated by worker_initializer) ─────────────────
_worker_rds = None
_worker_eds = None
_worker_cds = None
_worker_grid_lons = None
_worker_grid_lats = None
_worker_grid_x = None
_worker_grid_y = None
_worker_bathy_data = {}  # dict keyed by region name
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


def _build_surface_dataset(ctime, model_name, salinity=None, u=None, v=None, lon=None, lat=None):
    data_vars = {}
    if salinity is not None:
        data_vars['salinity'] = salinity
    if u is not None:
        data_vars['u'] = u
    if v is not None:
        data_vars['v'] = v
    if not data_vars:
        return None

    ds = xr.Dataset(data_vars)

    coord_source = salinity if salinity is not None else u if u is not None else v
    if lon is None and coord_source is not None and 'lon' in coord_source.coords:
        lon = coord_source['lon']
    if lat is None and coord_source is not None and 'lat' in coord_source.coords:
        lat = coord_source['lat']

    assign_coords = {'time': pd.Timestamp(ctime).to_datetime64(), 'depth': 0.0}
    if lon is not None:
        assign_coords['lon'] = lon
    if lat is not None:
        assign_coords['lat'] = lat

    ds = ds.assign_coords(**assign_coords)
    ds.attrs['model'] = model_name
    return ds


def load_or_fetch_argo(extent, start_time, end_time, cache_dir):
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"argo_{ext_str}_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.pkl"
    if cache_file.exists():
        print(f"Loading Argo from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    print("Fetching Argo data...")
    argo_data = get_argo_floats_by_time(extent, start_time, end_time)
    with open(cache_file, 'wb') as f:
        pickle.dump(argo_data, f)
    return argo_data


def load_or_fetch_gliders(extent, start_time, end_time, cache_dir, parallel=False):
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"gliders_{ext_str}_{start_time.strftime('%Y%m%d_%H%M')}_{end_time.strftime('%Y%m%d_%H%M')}.pkl"
    if cache_file.exists():
        print(f"Loading gliders from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    print("Fetching glider data...")
    glider_data = get_active_gliders(extent, start_time, end_time, parallel=parallel)
    with open(cache_file, 'wb') as f:
        pickle.dump(glider_data, f)
    return glider_data


def load_or_fetch_bathymetry(extent, cache_dir):
    ext_str = f"{extent[0]}_{extent[1]}_{extent[2]}_{extent[3]}"
    cache_file = cache_dir / f"bathy_{ext_str}.pkl"
    if cache_file.exists():
        print(f"Loading bathymetry from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    print(f"Fetching bathymetry for extent {extent}...")
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
    global_extent,
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
    global _worker_rds, _worker_eds, _worker_cds
    global _worker_grid_lons, _worker_grid_lats, _worker_grid_x, _worker_grid_y
    global _worker_bathy_data, _worker_argo_data, _worker_glider_data
    global _worker_region_configs, _worker_config, _worker_path_save
    global _worker_cache_dir

    _worker_cache_dir = cache_dir
    _worker_bathy_data = bathy_data
    _worker_argo_data = argo_data
    _worker_glider_data = glider_data
    _worker_region_configs = region_configs
    _worker_config = config_dict
    _worker_path_save = path_save

    # ── RTOFS ──────────────────────────────────────────────────────────
    if config_dict['plot_rtofs']:
        rds_local = r()
        rds_local = rds_local[['salinity', 'u', 'v']]
        lons_ind = np.interp(global_extent_buffered[:2], rds_local.lon.values[0, :], rds_local.x.values)
        lats_ind = np.interp(global_extent_buffered[2:], rds_local.lat.values[:, 0], rds_local.y.values)
        rds_local = rds_local.isel(
            x=slice(np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int)),
            y=slice(np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int)),
        )
        _worker_rds = rds_local
        _worker_grid_lons = rds_local.lon.values[0, :]
        _worker_grid_lats = rds_local.lat.values[:, 0]
        _worker_grid_x = rds_local.x.values
        _worker_grid_y = rds_local.y.values

    # ── ESPC (FMRC best-forecast for real-time) ────────────────────────
    if config_dict['plot_espc']:
        print("ESPC: using FMRC best-forecast (real-time)")
        eds_ts = espc_ts(rename=True, chunks={'time': 1, 'depth': 5})
        eds_ts_sub = eds_ts.sel(
            lon=slice(lon_transform[0], lon_transform[1]),
            lat=slice(global_extent_buffered[2], global_extent_buffered[3]),
            depth=slice(0, 10),
        )

        url_uv = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
        eds_uv = xr.open_dataset(url_uv, drop_variables='tau', chunks={'time': 1, 'depth': 5})
        eds_uv_sub = eds_uv.sel(
            lon=slice(lon_transform[0], lon_transform[1]),
            lat=slice(global_extent_buffered[2], global_extent_buffered[3]),
            depth=slice(0, 10),
        )

        _worker_eds = {
            'salinity': eds_ts_sub['salinity'],
            'u': eds_uv_sub['water_u'],
            'v': eds_uv_sub['water_v'],
        }

    # ── CMEMS ──────────────────────────────────────────────────────────
    if config_dict['plot_cmems']:
        cobj = c()
        lon_slice = slice(global_extent_buffered[0], global_extent_buffered[1])
        lat_slice = slice(global_extent_buffered[2], global_extent_buffered[3])
        depth_slice = slice(0, 10)  # near-surface only
        _worker_cds = {
            'salinity': cobj.get_variable('salinity').sel(longitude=lon_slice, latitude=lat_slice, depth=depth_slice),
            'u':        cobj.get_variable('uo').sel(longitude=lon_slice, latitude=lat_slice, depth=depth_slice),
            'v':        cobj.get_variable('vo').sel(longitude=lon_slice, latitude=lat_slice, depth=depth_slice),
        }

    print(f"Worker {mp.current_process().name} initialized")


# ── Per-timestamp processing ───────────────────────────────────────────────

def process_time(ctime):
    rds = _worker_rds
    eds = _worker_eds
    cds = _worker_cds
    grid_lons = _worker_grid_lons
    grid_lats = _worker_grid_lats
    grid_x = _worker_grid_x
    grid_y = _worker_grid_y
    bathy_by_region = _worker_bathy_data
    argo_data = _worker_argo_data
    glider_data = _worker_glider_data
    region_configs = _worker_region_configs
    config = _worker_config
    path_save = _worker_path_save
    cache_dir = _worker_cache_dir

    rdt_flag = edt_flag = cdt_flag = False
    argo = gliders = pd.DataFrame()
    plots_generated = []

    search_window_t0 = ctime - pd.Timedelta(hours=config.get('search_hours', 24 * 5))
    search_window_t1 = ctime

    # ── Time selection ─────────────────────────────────────────────────
    if config['plot_rtofs'] and rds is not None:
        try:
            rds_time = rds.sel(time=ctime)
            rds_time.load()
            rdt_flag = True
        except KeyError:
            pass

    if config['plot_espc'] and eds is not None:
        try:
            sal = eds['salinity'].sel(time=ctime)
            uo = eds['u'].sel(time=ctime)
            vo = eds['v'].sel(time=ctime)
            sal['lon'] = lon360to180(sal['lon'])
            uo['lon'] = lon360to180(uo['lon'])
            vo['lon'] = lon360to180(vo['lon'])
            eds_time = xr.Dataset({'salinity': sal, 'u': uo, 'v': vo})
            edt_flag = True
        except KeyError:
            pass

    if config['plot_cmems'] and cds is not None:
        try:
            sal = cds['salinity'].sel(time=ctime, method='nearest')
            uo = cds['u'].sel(time=ctime, method='nearest')
            vo = cds['v'].sel(time=ctime, method='nearest')
            cds_time = xr.Dataset({'salinity': sal, 'u': uo, 'v': vo})
            rename_map = {k: v for k, v in {'longitude': 'lon', 'latitude': 'lat'}.items()
                          if k in cds_time.coords}
            if rename_map:
                cds_time = cds_time.rename(rename_map)
            cds_time.attrs['model'] = 'CMEMS'
            cdt_flag = True
        except KeyError:
            pass

    if not rdt_flag:
        print(f"RTOFS data missing for {ctime} — skipping (need RTOFS as base grid).")
        return f"Skipped {ctime}: no RTOFS"

    # ── Per-region loop ────────────────────────────────────────────────
    for region in config['regions']:
        configs = region_configs[region]
        extent = configs["extent"]
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()
        bathy_data = bathy_by_region.get(region)

        lons_ind = np.interp(extent_data[:2], grid_lons, grid_x)
        lats_ind = np.interp(extent_data[2:], grid_lats, grid_y)
        extent_ind = [
            np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int),
            np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int),
        ]
        rds_slice = rds_time.sel(
            x=slice(extent_ind[0], extent_ind[1]),
            y=slice(extent_ind[2], extent_ind[3]),
        )

        # Extract RTOFS surface (depth index 0)
        try:
            rds_surface = rds_slice.isel(depth=0)
            sss_rtofs = rds_surface['salinity']
            sss_rtofs = sss_rtofs.assign_coords(lon=rds_surface['lon'], lat=rds_surface['lat'])
            u_rtofs = rds_surface['u'].assign_coords(lon=rds_surface['lon'], lat=rds_surface['lat'])
            v_rtofs = rds_surface['v'].assign_coords(lon=rds_surface['lon'], lat=rds_surface['lat'])
            rtofs_lon2d = rds_surface['lon'].values
            rtofs_lat2d = rds_surface['lat'].values
        except Exception as err:
            print(f"  RTOFS surface extraction failed for {region} at {ctime}: {err}")
            continue

        sss_espc = None
        u_espc = None
        v_espc = None

        if edt_flag:
            try:
                eds_slice = eds_time.sel(
                    lon=slice(extent_data[0] - 2, extent_data[1] + 2),
                    lat=slice(extent_data[2] - 2, extent_data[3] + 2),
                )
                eds_surface = eds_slice.isel(depth=0)
                print(f"  Loading ESPC surface data ({region})...")
                _load_with_timeout(eds_surface)

                sss_espc = regrid_to_rtofs(
                    eds_surface['salinity'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'espc'),
                    weights_dir=cache_dir,
                )
                sss_espc = sss_espc.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )

                u_espc = regrid_to_rtofs(
                    eds_surface['u'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'espc'),
                    weights_dir=cache_dir,
                )
                u_espc = u_espc.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )

                v_espc = regrid_to_rtofs(
                    eds_surface['v'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'espc'),
                    weights_dir=cache_dir,
                )
                v_espc = v_espc.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )
            except Exception as err:
                print(f"  ESPC SSS/currents/regrid failed for {region} at {ctime}: {err}")

        sss_cmems = None
        u_cmems = None
        v_cmems = None

        if cdt_flag:
            try:
                cds_slice = cds_time.sel(
                    lon=slice(extent_data[0], extent_data[1]),
                    lat=slice(extent_data[2], extent_data[3]),
                )
                # Select surface depth (index 0)
                cds_surface = cds_slice.isel(depth=0)
                print(f"  Loading CMEMS surface data ({region})...")
                _load_with_timeout(cds_surface)

                # Regrid SSS to RTOFS grid (reuses the regridder for u/v via same cache key)
                sss_cmems = regrid_to_rtofs(
                    cds_surface['salinity'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'cmems'),
                    weights_dir=cache_dir,
                )
                sss_cmems = sss_cmems.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )

                u_cmems = regrid_to_rtofs(
                    cds_surface['u'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'cmems'),
                    weights_dir=cache_dir,
                )
                u_cmems = u_cmems.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )

                v_cmems = regrid_to_rtofs(
                    cds_surface['v'], rtofs_lon2d, rtofs_lat2d,
                    cache_key=(region, 'cmems'),
                    weights_dir=cache_dir,
                )
                v_cmems = v_cmems.assign_coords(
                    lon=xr.DataArray(rtofs_lon2d, dims=['y', 'x']),
                    lat=xr.DataArray(rtofs_lat2d, dims=['y', 'x']),
                )
            except Exception as err:
                print(f"  CMEMS SSS/currents/regrid failed for {region} at {ctime}: {err}")

        if argo_data is not None and not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            argo_region = argo_data[mask].copy()
            argo_region.sort_index(inplace=True)
            try:
                argo = argo_region.loc[pd.IndexSlice[:, search_window_t0:search_window_t1], :]
            except KeyError:
                argo = pd.DataFrame()

        if glider_data is not None and not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            glider_region = glider_data[mask]
            gliders = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                & (glider_region.index.get_level_values('time') < search_window_t1)
            ]

        salinity_cfg = next((item for item in configs.get('salinity', []) if item.get('depth') == 0), None)
        current_limits = configs.get('currents', {}).get('limits', [0, 1.5, 0.1])
        salinity_limits = salinity_cfg.get('limits', [30, 38, 0.5]) if salinity_cfg else [30, 38, 0.5]

        speed_rtofs = _build_surface_dataset(ctime, 'RTOFS', u=u_rtofs, v=v_rtofs) if u_rtofs is not None and v_rtofs is not None else None
        speed_espc = _build_surface_dataset(ctime, 'ESPC', u=u_espc, v=v_espc) if u_espc is not None and v_espc is not None else None
        speed_cmems = _build_surface_dataset(ctime, 'COPERNICUS', u=u_cmems, v=v_cmems) if u_cmems is not None and v_cmems is not None else None

        common_kwargs = dict(
            extent=extent,
            region_name=configs["name"],
            path_save=path_save / configs["folder"],
            bathy=bathy_data,
            argo=argo,
            gliders=gliders,
            eez=configs.get('eez', True),
            transform=config['projection'],
            dpi=config['dpi'],
            overwrite=config.get('overwrite', False),
        )

        if sss_espc is not None:
            plot_sss_three_panel(
                sss_rtofs,
                sss_espc,
                comp_model_name='ESPC',
                ctime=ctime,
                sss_min=salinity_limits[0],
                sss_max=salinity_limits[1],
                sss_stride=salinity_limits[2],
                diff_lim=max(1.0, salinity_limits[2] * 6),
                **common_kwargs,
            )
            plots_generated.append(f"{region} | Sea Surface Salinity | RTOFS vs ESPC")

        if sss_cmems is not None:
            plot_sss_three_panel(
                sss_rtofs,
                sss_cmems,
                comp_model_name='COPERNICUS',
                ctime=ctime,
                sss_min=salinity_limits[0],
                sss_max=salinity_limits[1],
                sss_stride=salinity_limits[2],
                diff_lim=max(1.0, salinity_limits[2] * 6),
                **common_kwargs,
            )
            plots_generated.append(f"{region} | Sea Surface Salinity | RTOFS vs COPERNICUS")

        if speed_rtofs is not None and speed_espc is not None:
            plot_speed_diff_three_panel(
                speed_rtofs,
                speed_espc,
                comp_model_name='ESPC',
                ctime=ctime,
                speed_min=current_limits[0],
                speed_max=current_limits[1],
                speed_stride=current_limits[2],
                diff_lim=current_limits[1],
                **common_kwargs,
            )
            plots_generated.append(f"{region} | Surface Currents | RTOFS vs ESPC")

        if speed_rtofs is not None and speed_cmems is not None:
            plot_speed_diff_three_panel(
                speed_rtofs,
                speed_cmems,
                comp_model_name='COPERNICUS',
                ctime=ctime,
                speed_min=current_limits[0],
                speed_max=current_limits[1],
                speed_stride=current_limits[2],
                diff_lim=current_limits[1],
                **common_kwargs,
            )
            plots_generated.append(f"{region} | Surface Currents | RTOFS vs COPERNICUS")

    return {'ctime': ctime, 'plots': plots_generated}


# ── Main ───────────────────────────────────────────────────────────────────

def main(parallel=True, max_workers=None):
    start_time_exec = time.time()

    plot_rtofs = True
    plot_espc = True
    plot_cmems = True
    replot = False

    conf.days = 2
    path_save = conf.path_plots / "adaptive_sampling_guidance" / "maps"

    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)

    # ── Real-time: use whatever RTOFS actually has most recently ──────
    print("Probing RTOFS for available timestamps...")
    _rtofs_probe = r()
    rtofs_times = pd.DatetimeIndex(_rtofs_probe.time.values)
    del _rtofs_probe

    # Take the last N timestamps based on conf.days (RTOFS has 4 timestamps per day)
    date_list = pd.DatetimeIndex(rtofs_times[-(conf.days * 4):])

    date_start = date_list[0]
    date_end = date_list[-1]
    search_start = date_start - dt.timedelta(hours=conf.search_hours)

    print(f"Real-time mode: using {len(date_list)} RTOFS timestamps on {date_list[0].date()}")
    for t in date_list:
        print(f"  {t}")

    conf.regions = ['gom_west', 'gom_east', 'caribbean', 'east_coast', 'tropical_western_atlantic']
    region_configs = {region: region_config([region]) for region in conf.regions}
    extent_list = [cfg["extent"] for cfg in region_configs.values()]

    extent_df = pd.DataFrame(np.array(extent_list), columns=["lonmin", "lonmax", "latmin", "latmax"])
    global_extent = [
        extent_df.lonmin.min(), extent_df.lonmax.max(),
        extent_df.latmin.min(), extent_df.latmax.max(),
    ]
    global_extent_buffered = [
        global_extent[0] - 3, global_extent[1] + 3,
        global_extent[2] - 3, global_extent[3] + 3,
    ]
    lon_transform = lon180to360(global_extent_buffered[:2])

    print(f"Global extent: {global_extent}")

    if conf.bathy:
        bathy_data = {
            region: load_or_fetch_bathymetry(region_configs[region]["extent"], cache_dir)
            for region in conf.regions
        }
    else:
        bathy_data = {}
    argo_data = load_or_fetch_argo(global_extent, search_start, date_end, cache_dir) if conf.argo else pd.DataFrame()
    glider_data = load_or_fetch_gliders(global_extent, search_start, date_end, cache_dir) if conf.gliders else pd.DataFrame()

    config_dict = {
        'regions': conf.regions,
        'projection': conf.projection,
        'dpi': conf.dpi,
        'search_hours': conf.search_hours,
        'plot_rtofs': plot_rtofs,
        'plot_espc': plot_espc,
        'plot_cmems': plot_cmems,
        'overwrite': replot,
    }

    init_args = (
        global_extent,
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
        print(f"Using {max_workers} worker processes")

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
                    print(f"[{completed}/{total}] {ctime} done ({len(result['plots'])} plots)")
                except Exception as exc:
                    print(f"[{completed}/{total}] {ctime} ERROR: {exc}")

    # ── Summary ────────────────────────────────────────────────────────
    total_plots = sum(len(r['plots']) for r in all_results if isinstance(r, dict))
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
