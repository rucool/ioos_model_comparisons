import datetime as dt
import time

import ioos_model_comparisons.configs as conf
import matplotlib
import numpy as np
import pandas as pd
from ioos_model_comparisons.calc import (lon180to360,
                             lon360to180,
                             density,
                             ocean_heat_content,
                             )
import requests
from ioos_model_comparisons.platforms import (get_active_gliders,
                                  get_argo_floats_by_time,
                                  get_ohc,
                                  )
from ioos_model_comparisons.plotting import plot_ohc
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.db import (
    apply_colorbar_overrides,
    ensure_plot_index,
    fetch_completed_plot_keys,
    log_plots,
)
from shapely.errors import TopologicalError
from cool_maps.plot import get_bathymetry
import concurrent.futures

import xarray as xr

startTime = time.time()
matplotlib.use('agg')

SCRIPT_ID = "ohc"

parallel = True # utilize parallel processing?
max_workers = 8

# Which models should we plot?
plot_rtofs = True
plot_espc = True
plot_cmems = True
plot_nesdis = True
plot_para = False
# Set path to save plots
path_save = (conf.path_plots / "maps")

# Regions are split into two groups because RTOFS is tiled — Atlantic/Gulf/
# Caribbean regions use the default (east) tile, Pacific regions need the
# 'west' tile. Everything else (ESPC, CMEMS, Argo, gliders, bathy) is global
# and doesn't care about the split.
ATLANTIC_REGIONS = ['mab', 'sab', 'gom', 'caribbean', 'tropical_western_atlantic', 'windward']
PACIFIC_REGIONS  = ['hawaii', 'mexico_pacific']

conf.days = .5
conf.regions = ATLANTIC_REGIONS + PACIFIC_REGIONS
conf.regions =  ['caribbean']

conf.argo = True
conf.gliders = True
conf.bathy = True

# initialize keyword arguments. Grab anything from configs.py
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = False

# Get today and yesterday dates
today = dt.date.today()
date_end = today + dt.timedelta(days=2)
date_start = today - dt.timedelta(days=conf.days)

freq = '6H'
# Create dates that we want to plot
date_list = pd.date_range(date_start, date_end, freq=freq)

search_start = date_list[0] - dt.timedelta(hours=conf.search_hours)


def _group_extent(regions):
    """Combined [lonmin, lonmax, latmin, latmax] extent across a list of regions."""
    extent_list = [region_config(region)["extent"] for region in regions]
    extent_df = pd.DataFrame(
        np.array(extent_list),
        columns=['lonmin', 'lonmax', 'latmin', 'latmax']
        )
    return [
        extent_df.lonmin.min(),
        extent_df.lonmax.max(),
        extent_df.latmin.min(),
        extent_df.latmax.max()
        ]

# Global extent across every region — used for Argo/glider/CMEMS/bathy, none
# of which care about the RTOFS east/west tile split.
global_extent = _group_extent(conf.regions)
atlantic_extent = _group_extent(ATLANTIC_REGIONS)
pacific_extent = _group_extent(PACIFIC_REGIONS)

lon_transform = lon180to360(global_extent[:2])

if conf.argo:
    argo_data = get_argo_floats_by_time(global_extent, search_start, date_end)
else:
    argo_data = pd.DataFrame()

if conf.gliders:
    glider_data = get_active_gliders(global_extent, search_start, date_end,
                                     parallel=False)
else:
    glider_data = pd.DataFrame()


def drop_first_deployment_per_base(glider_data: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the full (glider-with-deployment) index, but for bases that have multiple
    deployments (as identified by a trailing -YYYYMMDDTHHMM), drop the earliest one.

    Works even if base names contain hyphens (e.g., 'usf-jaialai').
    """
    # Expect MultiIndex with [glider, time]
    lvl0, lvl1 = glider_data.index.names[:2]

    df = glider_data.reset_index()

    # Extract optional trailing deployment stamp. Example match:
    #   SG678-20250624T1621 -> base='SG678', stamp='20250624T1621'
    #   usf-jaialai         -> base=NaN,   stamp=NaN  (no trailing stamp)
    ext = df[lvl0].str.extract(r'^(?P<base>.+)-(?P<stamp>\d{8}T\d{4})$')
    # base_glider = extracted base when a stamp exists; otherwise the original name
    df["glider_base"] = ext["base"].where(ext["base"].notna(), df[lvl0])

    # Parse deployment datetime if present; otherwise NaT
    df["deploy_dt"] = pd.to_datetime(ext["stamp"], format="%Y%m%dT%H%M", errors="coerce")

    # Build per-(base, full_deployment_label) keys
    dep_keys = (
        df.groupby(["glider_base", lvl0])
          .agg(
              deploy_dt=("deploy_dt", "min"),
              first_obs_time=(lvl1, "min"),
          )
          .reset_index()
    )
    # Use parsed stamp if available; fallback to earliest observation time
    dep_keys["order_key"] = dep_keys["deploy_dt"].fillna(dep_keys["first_obs_time"])

    # Sort to get stable ranking (tie-breaker on full deployment label)
    dep_keys = dep_keys.sort_values(["glider_base", "order_key", lvl0])
    dep_keys["rank_in_base"] = dep_keys.groupby("glider_base").cumcount()

    # Identify bases with >1 deployments (only those with stamps actually create >1 rows)
    dep_counts = dep_keys.groupby("glider_base")[lvl0].transform("nunique")

    # Earliest deployment per base to drop (only when count > 1)
    to_drop_deployments = set(
        dep_keys.loc[(dep_counts > 1) & (dep_keys["rank_in_base"] == 0), lvl0]
    )

    # Drop all rows belonging to those earliest deployments
    out = df[~df[lvl0].isin(to_drop_deployments)].set_index([lvl0, lvl1]).sort_index()

    return out

glider_data = drop_first_deployment_per_base(glider_data)

if not glider_data.empty:
    # Split the 'glider' index by the last '-'
    glider_data.index = glider_data.index.set_levels([
        glider_data.index.levels[0].str.rsplit('-', n=1).str[0],  # Corrected by specifying `n=1` as a keyword
        glider_data.index.levels[1]
    ])

if conf.bathy:
    bathy_dict = {}
    for region in conf.regions:
        bathy_dict[region] = get_bathymetry(region_config(region)["extent"])


def _load_rtofs_grid(source, extent):
    """Load RTOFS, subset to *extent*, and return (dataset, grid_lons, grid_lats, grid_x, grid_y)."""
    from ioos_model_comparisons.models import rtofs as r

    rds_ = r(source=source) if source else r()
    rds_ = rds_[['temperature', 'salinity']]
    lons_ind = np.interp(extent[:2], rds_.lon.values[0, :], rds_.x.values)
    lats_ind = np.interp(extent[2:], rds_.lat.values[:, 0], rds_.y.values)

    rds_ = rds_.isel(
        x=slice(np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int)),
        y=slice(np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int))
        )

    grid_lons_ = rds_.lon.values[0, :]
    grid_lats_ = rds_.lat.values[:, 0]
    grid_x_ = rds_.x.values
    grid_y_ = rds_.y.values
    return rds_, grid_lons_, grid_lats_, grid_x_, grid_y_


if plot_rtofs:
    rds_atl, grid_lons_atl, grid_lats_atl, grid_x_atl, grid_y_atl = _load_rtofs_grid(None, atlantic_extent)
    rds_pac, grid_lons_pac, grid_lats_pac, grid_x_pac, grid_y_pac = _load_rtofs_grid('west', pacific_extent)

if plot_para:
    from ioos_model_comparisons.models import rtofs as r

    # Load RTOFS and subset to global_extent of regions we are looking at.
    rdsp = r(source='parallel')
    rdsp = rdsp[['temperature', 'salinity']]
    lons_ind = np.interp(global_extent[:2], rdsp.lon.values[0,:], rdsp.x.values)
    lats_ind = np.interp(global_extent[2:], rdsp.lat.values[:,0], rdsp.y.values)

    rdsp = rdsp.isel(
        x=slice(np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int)),
        y=slice(np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int))
        )

if plot_espc:
    from ioos_model_comparisons.models import espc_ts

    # Load GOFS
    gds = espc_ts(rename=True)

if plot_cmems:
    from ioos_model_comparisons.models import CMEMS as c

    # Load Copernicus
    cds = c()
    cds = cds.get_combined_subset(global_extent[:2], global_extent[2:])

plot_hurricanes = True
# ctimes within this many hours of now use the real-time NHC feed;
# older ctimes fall back to the IBTrACS archive automatically.
_REALTIME_WINDOW_HOURS = 48

_rt_obj = None      # cached realtime.Realtime()
_archive_obj = None  # cached tracks.TrackDataset


def _load_realtime():
    global _rt_obj
    if _rt_obj is None:
        from tropycal import realtime as trealtime
        _rt_obj = trealtime.Realtime()
    return _rt_obj


def _load_archive():
    global _archive_obj
    if _archive_obj is None:
        from tropycal import tracks as trtracks
        _archive_obj = trtracks.TrackDataset(
            basin='north_atlantic', source='ibtracs', include_btk=True
        )
    return _archive_obj


def _archive_storms_at(ctime):
    import types
    basin = _load_archive()
    year = ctime.year
    active_stms, active_fcts = [], []

    for sid in list(basin.data.keys()):
        # Quick year check from storm ID (e.g. 'AL092024' → 2024)
        try:
            if int(sid[-4:]) != year:
                continue
        except (ValueError, IndexError):
            continue
        try:
            storm = basin.get_storm(sid)
            # Archive Storm objects use .time; realtime Storm objects use .date
            times = getattr(storm, 'time', None) or getattr(storm, 'date', None)
            if not times:
                continue
            dates = [pd.Timestamp(d) for d in times]
            if not (min(dates) <= ctime <= max(dates)):
                continue
            idx = [i for i, d in enumerate(dates) if d <= ctime]
            if not idx:
                continue
            s = types.SimpleNamespace(
                invest=getattr(storm, 'invest', False),
                lon=[storm.lon[i] for i in idx],
                lat=[storm.lat[i] for i in idx],
                vmax=[storm.vmax[i] for i in idx],
                name=storm.name,
                basin=storm.basin,
            )
            active_stms.append(s)
            active_fcts.append(storm.get_nhc_forecast_dict(ctime))
        except Exception:
            continue

    return active_stms, active_fcts


def _get_storms_for_time(ctime):
    """Return (storms, forecasts) for ctime using realtime or archive as appropriate."""
    if not plot_hurricanes:
        return [], []
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_hours = (now - ctime).total_seconds() / 3600
    try:
        if age_hours <= _REALTIME_WINDOW_HOURS:
            rt = _load_realtime()
            keys = rt.list_active_storms(basin='north_atlantic')
            stms = [rt.get_storm(k) for k in keys]
            fcts = [s.get_forecast_realtime(True) for s in stms]
            if stms:
                return stms, fcts
        # Archive fallback: ctime is old, or realtime returned no storms
        return _archive_storms_at(ctime)
    except Exception as e:
        print(f"Warning: hurricane fetch failed for {ctime}: {e}")
        return [], []

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'


def _ohc_record(region, ts_dt, m1, m2):
    return {"region": region['name'], "timestamp": ts_dt, "plot_type": "ocean_heat_content",
            "variable": "ohc", "depth": 0, "model1": m1, "model2": m2}


def _expected_plot_keys(ctime, region) -> set:
    """Return the set of (region, iso_ts, plot_type, variable, depth, m1, m2) tuples
    expected for this (ctime, region) — no file I/O.

    OHC produces one plot per model pair per region (no variable/depth breakdown).
    """
    if 'ocean_heat_content' not in region:
        return set()

    t      = pd.to_datetime(ctime)
    tstr_k = t.strftime("%Y-%m-%dT%H%M%SZ")
    rname  = region['name']
    keys   = set()

    model_pairs = []
    if plot_rtofs and plot_espc:
        model_pairs.append(('rtofs', 'espc'))
    if plot_rtofs and plot_cmems:
        model_pairs.append(('rtofs', 'cmems'))
    if plot_rtofs and plot_nesdis:
        model_pairs.append(('rtofs', 'nesdis'))
    if plot_rtofs and plot_para:
        model_pairs.append(('rtofs', 'rtofs'))

    for m1, m2 in model_pairs:
        keys.add((rname, tstr_k, "ocean_heat_content", "ohc", 0, m1, m2))

    return keys


def pre_check_date_list(date_list, overwrite=False) -> pd.DatetimeIndex:
    """Return a filtered DatetimeIndex with only timestamps that still need processing.

    Tries MongoDB first (one round-trip); falls back to processing all
    timestamps if the database is unavailable.

    overwrite=True bypasses this check entirely and returns date_list unchanged —
    otherwise a timestamp already logged as done in MongoDB is skipped even with
    overwrite set, since the per-file overwrite check inside plot_* never runs
    for timestamps dropped at this stage.
    """
    if overwrite:
        print(f"Pre-check: overwrite=True — reprocessing all {len(date_list)} timestamp(s).")
        return pd.DatetimeIndex(date_list)

    expected_by_ts = {}
    for ctime in date_list:
        keys = set()
        for item in conf.regions:
            region = apply_colorbar_overrides(item, region_config(item))
            keys |= _expected_plot_keys(ctime, region)
        expected_by_ts[ctime] = keys

    done_keys = fetch_completed_plot_keys(SCRIPT_ID, list(date_list))

    needed_dates = []
    skipped = 0
    if done_keys is not None:
        for ctime in date_list:
            if expected_by_ts[ctime].issubset(done_keys):
                skipped += 1
            else:
                needed_dates.append(ctime)
        print(f"Pre-check (MongoDB): {skipped}/{len(date_list)} timestamp(s) fully done — skipping.")
    else:
        print("MongoDB unavailable — processing all timestamps.")
        needed_dates = list(date_list)

    print(f"Pre-check: {len(needed_dates)}/{len(date_list)} timestamp(s) queued.")
    return pd.DatetimeIndex(needed_dates)


# for ctime in date_list:
def plot_ctime(ctime):
    print(f"Checking if {ctime} exists for each model.")
    pending_logs = []
    ts_dt = pd.to_datetime(ctime).to_pydatetime()

    if plot_para:
        try:
            rdsp_time = rdsp.sel(time=ctime)
            print(f"RTOFS Parallel: True")
            rdtp_flag = True
        except KeyError as error:
            print(f"RTOFS Parallel: False")
            rdtp_flag = False
    else:
        rdtp_flag = False

    if plot_espc:
        try:
            gds_time = gds.sel(time=ctime)
            print(f"GOFS: True")
            gdt_flag = True
        except KeyError as error:
            print(f"GOFS: False")
            gdt_flag = False
    else:
        gdt_flag = False

    if plot_cmems:
        try:
            cds_time = cds.sel(time=ctime) #CMEMS
            print(f"CMEMS: True")
            cdt_flag = True
        except KeyError:
            print(f"CMEMS: False")
            cdt_flag = False
        print("\n")
    else:
        cdt_flag = False

    if plot_nesdis:
        try:
            nds = get_ohc(time=ctime)
            nds.attrs['model'] = 'NESDIS'
            nds = nds.rename({'longitude': 'lon', 'latitude': 'lat'})
            print(f"NESDIS: True")
            ndt_flag = True
        except (requests.exceptions.HTTPError, Exception) as e:
            print(f"NESDIS: False - {e}")
            ndt_flag = False
    else:
        ndt_flag = False

    _storms, _forecasts = _get_storms_for_time(ctime)

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)

    for r in conf.regions:
        is_pacific = r in PACIFIC_REGIONS
        rds_grp        = rds_pac if is_pacific else rds_atl
        grid_lons_grp  = grid_lons_pac if is_pacific else grid_lons_atl
        grid_lats_grp  = grid_lats_pac if is_pacific else grid_lats_atl
        grid_x_grp     = grid_x_pac if is_pacific else grid_x_atl
        grid_y_grp     = grid_y_pac if is_pacific else grid_y_atl

        if plot_rtofs:
            try:
                rds_time = rds_grp.sel(time=ctime)
                print(f"RTOFS ({'pacific' if is_pacific else 'atlantic'}): True")
                rdt_flag = True
            except KeyError:
                print(f"RTOFS ({'pacific' if is_pacific else 'atlantic'}): False")
                rdt_flag = False
        else:
            rdt_flag = False

        configs = apply_colorbar_overrides(r, region_config(r))

        # Save the extent of the region being plotted to a variable.
        extent = configs['extent']

        # Increase the extent a little bit to grab slightly more data than the region
        # we want to plot. Otherwise, we will have areas of the plot with no data.
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()

        # convert from 360 to 180 lon
        lon360 = lon180to360(extent_data[:2])

        # Add the following to keyword arguments to ocean_heat_content function
        kwargs['path_save'] = path_save / configs['folder']
        kwargs['eez'] = configs['eez']

        key = 'ocean_heat_content'
        if key in configs:
            if 'limits' in configs[key]:
                kwargs['limits'] = configs[key]['limits']
        else:
            continue

        if 'legend' in configs['figure']:
            kwargs['cols'] = configs['figure']['legend']['columns']

        if 'figsize' in configs['figure']:
            kwargs['figsize'] = configs['figure']['figsize']

        if conf.bathy:
            kwargs['bathy'] = bathy_dict[r]

        if rdt_flag:
            # Find x, y indexes of the area we want to subset
            lons_ind = np.interp(extent_data[:2], grid_lons_grp, grid_x_grp)
            lats_ind = np.interp(extent_data[2:], grid_lats_grp, grid_y_grp)

            # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice
            # in order to widen the area of the extent slightly.
            extent_ind = [
                np.floor(lons_ind[0]).astype(int),
                np.ceil(lons_ind[1]).astype(int),
                np.floor(lats_ind[0]).astype(int),
                np.ceil(lats_ind[1]).astype(int)
                ]

            # Use .isel selector on x/y since we know indexes that we want to slice
            rds_slice = rds_time.sel(
                x=slice(extent_ind[0], extent_ind[1]),
                y=slice(extent_ind[2], extent_ind[3]),
                )
            rds_slice.load()

            rds_slice['density'] = xr.apply_ufunc(density,
                        rds_slice['temperature'],
                        -rds_slice['depth'],
                        rds_slice['salinity'],
                        rds_slice['lat'],
                        rds_slice['lon']
                        )
            rds_slice['ohc'] = xr.apply_ufunc(ocean_heat_content,
                                rds_slice.depth,
                                rds_slice.temperature,
                                rds_slice.density,
                                input_core_dims=[['depth'], ['depth'], ['depth']],
                                vectorize=True,
                                )

        if rdtp_flag:
            # Find x, y indexes of the area we want to subset
            lons_ind = np.interp(extent_data[:2], grid_lons_grp, grid_x_grp)
            lats_ind = np.interp(extent_data[2:], grid_lats_grp, grid_y_grp)

            # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice
            # in order to widen the area of the extent slightly.
            extent_ind = [
                np.floor(lons_ind[0]).astype(int),
                np.ceil(lons_ind[1]).astype(int),
                np.floor(lats_ind[0]).astype(int),
                np.ceil(lats_ind[1]).astype(int)
                ]

            # Use .isel selector on x/y since we know indexes that we want to slice
            rdsp_slice = rdsp_time.sel(
                x=slice(extent_ind[0], extent_ind[1]),
                y=slice(extent_ind[2], extent_ind[3])
                )
            rdsp_slice.load()

            rdsp_slice['density'] = xr.apply_ufunc(density,
                                    rdsp_slice['temperature'],
                                    -rdsp_slice['depth'],
                                    rdsp_slice['salinity'],
                                    rdsp_slice['lat'],
                                    rdsp_slice['lon']
                                    )
            rdsp_slice['ohc'] = xr.apply_ufunc(ocean_heat_content,
                                rdsp_slice.depth,
                                rdsp_slice.temperature,
                                rdsp_slice.density,
                                input_core_dims=[['depth'], ['depth'], ['depth']],
                                vectorize=True)

        if gdt_flag:
            # subset dataset to the proper extents for each region
            gds_slice = gds_time.sel(
                lon=slice(lon180to360(extent_data[0]), lon180to360(extent_data[1])),
                lat=slice(extent_data[2], extent_data[3])
            )

            # Convert from 0,360 lon to -180,180
            gds_slice['lon'] = lon360to180(gds_slice['lon'])

            gds_slice['density'] = xr.apply_ufunc(density,
                        gds_slice['temperature'],
                        -gds_slice['depth'],
                        gds_slice['salinity'],
                        gds_slice['lat'],
                        gds_slice['lon']
                        )
            gds_slice['ohc'] = xr.apply_ufunc(ocean_heat_content,
                                gds_slice.depth,
                                gds_slice.temperature,
                                gds_slice.density,
                                input_core_dims=[['depth'], ['depth'], ['depth']],
                                vectorize=True)

        if cdt_flag:
            cds_slice= cds_time.sel(
                lon=slice(extent_data[0], extent_data[1]),
                lat=slice(extent_data[2], extent_data[3])
                )
            cds_slice.load()

            cds_slice['density'] = xr.apply_ufunc(density,
                                    cds_slice['temperature'],
                                    -cds_slice['depth'],
                                    cds_slice['salinity'],
                                    cds_slice['lat'],
                                    cds_slice['lon']
                                    )
            cds_slice['ohc'] = xr.apply_ufunc(ocean_heat_content,
                                cds_slice.depth,
                                cds_slice.temperature,
                                cds_slice.density,
                                input_core_dims=[['depth'], ['depth'], ['depth']],
                                vectorize=True)

        if ndt_flag:
            nds_slice = nds.sel(
                lon=slice(extent_data[0], extent_data[1]),
                lat=slice(extent_data[2], extent_data[3])
            ).squeeze()

        # Subset downloaded Argo data to this region and time
        if not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']

            # Mask out anything beyond the extent
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            argo_region = argo_data[mask]
            argo_region.sort_index(inplace=True)

            # Mask out any argo floats beyond the time window
            idx = pd.IndexSlice
            kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

        # Subset downloaded glider data to this region and time
        if not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']

            # Mask out anything beyond the extent
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            glider_region = glider_data[mask]

            # Mask out any gliders beyond the time window
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                &
                (glider_region.index.get_level_values('time') < search_window_t1)
                ]
            kwargs['gliders'] = glider_region

        try:
            if rdt_flag and gdt_flag:
                plot_ohc(rds_slice, gds_slice, extent, configs['name'],
                         storms=_storms, forecasts=_forecasts, **kwargs)
                pending_logs.append(_ohc_record(configs, ts_dt, 'rtofs', 'espc'))

            if rdt_flag and cdt_flag:
                plot_ohc(rds_slice, cds_slice, extent, configs['name'],
                         storms=_storms, forecasts=_forecasts, **kwargs)
                pending_logs.append(_ohc_record(configs, ts_dt, 'rtofs', 'cmems'))

            if rdt_flag and ndt_flag:
                plot_ohc(rds_slice, nds_slice, extent, configs['name'],
                         storms=_storms, forecasts=_forecasts, **kwargs)
                pending_logs.append(_ohc_record(configs, ts_dt, 'rtofs', 'nesdis'))

            if rdt_flag and rdtp_flag:
                plot_ohc(rds_slice, rdsp_slice, extent, configs['name'],
                         storms=_storms, forecasts=_forecasts, **kwargs)
                pending_logs.append(_ohc_record(configs, ts_dt, 'rtofs', 'rtofs'))

            # Delete some keyword arguments that may not be defined in all
            # regions. We don't want to plot the regions with wrong inputs
            if 'figsize' in kwargs:
                del kwargs['figsize']

            if 'limits' in kwargs:
                del kwargs['limits']

            if 'eez' in kwargs:
                del kwargs['eez']

            if 'gliders' in kwargs:
                del kwargs['gliders']

            if 'argo' in kwargs:
                del kwargs['argo']

        except TopologicalError as error:
            print("Error: {error}")
            continue

    return pending_logs

def main():
    ensure_plot_index()
    date_list_pending = pre_check_date_list(date_list, overwrite=kwargs.get('overwrite', False))

    if len(date_list_pending) == 0:
        print("All outputs already exist. Nothing to do.")
        return

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(plot_ctime, ct): ct for ct in date_list_pending}
            for future in concurrent.futures.as_completed(futures):
                ct = futures[future]
                try:
                    pending_logs = future.result()
                    log_plots(SCRIPT_ID, pending_logs)
                except Exception as exc:
                    print(f"{ct} ERROR: {exc}")
    else:
        for ctime in date_list_pending:
            pending_logs = plot_ctime(ctime)
            log_plots(SCRIPT_ID, pending_logs)

    print('Execution time in seconds: ' + str(time.time() - startTime))


if __name__ == "__main__":
    main()
