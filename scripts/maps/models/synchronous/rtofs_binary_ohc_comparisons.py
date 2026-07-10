"""
Compare ocean heat content (OHC) from pre-processed RTOFS binary NetCDFs
against ESPC and CMEMS, using the same plot_ohc function as
ocean_heat_content.py but reading RTOFS from the rtofs_global/ directory
structure instead of OPeNDAP.

Reads the most recent day's files from:
    rtofs_global/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_{region}.nc

Usage:
    python3 scripts/maps/models/synchronous/rtofs_binary_ohc_comparisons.py
    python3 scripts/maps/models/synchronous/rtofs_binary_ohc_comparisons.py --regions guam hawaii
    python3 scripts/maps/models/synchronous/rtofs_binary_ohc_comparisons.py --source global --regions guam
"""
import matplotlib
matplotlib.use("agg")

import argparse
import copy
import datetime as dt
import logging
import logging.handlers
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import (
    density,
    lon180to360,
    lon360to180,
    ocean_heat_content,
)
from ioos_model_comparisons.models import CMEMS, espc_ts
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
from ioos_model_comparisons.plotting import plot_ohc
from ioos_model_comparisons.regions import region_config
from cool_maps.plot import get_bathymetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

path_save = conf.path_plots / "maps"

plot_espc = True
plot_cmems = True
plot_hurricanes = True

_REALTIME_WINDOW_HOURS = 48
_rt_obj = None
_archive_obj = {}  # keyed by basin string


def _basin_for_extent(extent):
    """Pick the tropycal basin string from [lonmin, lonmax, latmin, latmax]."""
    center_lon = (extent[0] + extent[1]) / 2
    center_lat = (extent[2] + extent[3]) / 2
    if center_lat < 0:
        return 'south_indian' if center_lon > 40 else 'south_pacific'
    if -100 <= center_lon <= 30:
        return 'north_atlantic'
    if 30 < center_lon <= 100:
        return 'north_indian'
    # Pacific: center_lon > 100 (western Pacific) or < -100 (eastern Pacific)
    return 'west_pacific' if center_lon > 0 else 'east_pacific'


def _load_realtime():
    global _rt_obj
    if _rt_obj is None:
        from tropycal import realtime as trealtime
        _rt_obj = trealtime.Realtime()
    return _rt_obj


def _load_archive(basin):
    global _archive_obj
    if basin not in _archive_obj:
        from tropycal import tracks as trtracks
        _archive_obj[basin] = trtracks.TrackDataset(
            basin=basin, source='ibtracs', include_btk=True
        )
    return _archive_obj[basin]


def _archive_storms_at(ctime, basin):
    basin_ds = _load_archive(basin)
    year = ctime.year
    active_stms, active_fcts = [], []
    for sid in list(basin_ds.data.keys()):
        try:
            if int(sid[-4:]) != year:
                continue
        except (ValueError, IndexError):
            continue
        try:
            storm = basin_ds.get_storm(sid)
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


def _get_storms_for_time(ctime, extent):
    if not plot_hurricanes:
        return [], []
    basin = _basin_for_extent(extent)
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_hours = (now - ctime).total_seconds() / 3600
    try:
        if age_hours <= _REALTIME_WINDOW_HOURS:
            logger.info("Tropycal: realtime feed, basin=%s (%.1fh old)", basin, age_hours)
            rt = _load_realtime()
            keys = rt.list_active_storms()
            logger.info("Tropycal realtime: %d active storm(s) globally — %s", len(keys), keys)
            stms = [rt.get_storm(k) for k in keys]
            fcts = [s.get_forecast_realtime(True) for s in stms]
            if stms:
                return stms, fcts
        logger.info("Tropycal: IBTrACS archive, basin=%s, time=%s", basin, ctime)
        stms, fcts = _archive_storms_at(ctime, basin)
        logger.info("Tropycal archive: %d storm(s) found", len(stms))
        return stms, fcts
    except Exception as e:
        logger.warning("Hurricane fetch failed for %s: %s", ctime, e, exc_info=True)
        return [], []

kwargs = {
    "transform": conf.projection,
    "dpi": conf.dpi,
    "overwrite": False,
}


def find_files_in_range(data_dir, source, region_name, t_start, t_end, hours=None):
    if source == "global":
        pattern = "*_global.nc"
    else:
        pattern = f"*_{region_name}.nc"
    candidates = sorted(data_dir.glob(f"*/*/*/{pattern}"))
    if not candidates:
        return []
    if t_start is None and t_end is None:
        latest_dir = candidates[-1].parent
        candidates = [f for f in candidates if f.parent == latest_dir]
    else:
        candidates = [
            f for f in candidates
            if (t_start is None or parse_valid_time(f) >= t_start)
            and (t_end is None or parse_valid_time(f) <= t_end)
        ]
    if hours is not None:
        candidates = [f for f in candidates if parse_valid_time(f).hour in hours]
    return candidates


def parse_valid_time(nc_path):
    stem = nc_path.stem
    time_part = stem.split("_")[2]
    return pd.Timestamp(dt.datetime.strptime(time_part, "%Y%m%dT%H"))


def compute_ohc(ds):
    """Compute density and ocean heat content on a dataset with
    temperature, salinity, depth, lat, lon."""
    ds["density"] = xr.apply_ufunc(
        density,
        ds["temperature"],
        -ds["depth"],
        ds["salinity"],
        ds["lat"],
        ds["lon"],
    )
    ds["ohc"] = xr.apply_ufunc(
        ocean_heat_content,
        ds.depth,
        ds.temperature,
        ds.density,
        input_core_dims=[["depth"], ["depth"], ["depth"]],
        vectorize=True,
    )
    return ds


def load_rtofs_binary(nc_path, extent=None):
    ds = xr.open_dataset(nc_path)

    if extent is not None:
        ds = ds.sel(
            lon=slice(extent[0], extent[1]),
            lat=slice(extent[2], extent[3]),
        )

    valid_time = parse_valid_time(nc_path)

    ds = ds.rename({
        "temp": "temperature",
        "salin": "salinity",
        "u-vel.": "u",
        "v-vel.": "v",
        "z": "depth",
    })
    ds = ds.assign_coords(time=valid_time)
    ds.attrs["model"] = "RTOFS"
    ds["temperature"].attrs["units"] = "degC"
    ds["salinity"].attrs["units"] = "PSU"

    ds.load()
    ds = compute_ohc(ds)
    return ds


def load_model(model_func, model_name, **kw):
    try:
        logger.info("Loading %s model data.", model_name)
        data = model_func(**kw)
        logger.info("%s loaded successfully.", model_name)
        return data
    except Exception as e:
        logger.error("Failed to load %s: %s", model_name, e)
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare OHC from RTOFS binary NetCDFs against other models",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path.home() / "Downloads" / "rtofs_global",
    )
    parser.add_argument(
        "--source", default="region",
        choices=["region", "global"],
    )
    parser.add_argument(
        "--regions", nargs="*",
        default=["guam"],
    )
    parser.add_argument(
        "--start", type=pd.Timestamp, default=None,
        metavar="DATETIME",
        help="Start of time range, e.g. 2026-07-01 or 2026-07-01T06 (default: latest day only)",
    )
    parser.add_argument(
        "--end", type=pd.Timestamp, default=None,
        metavar="DATETIME",
        help="End of time range, e.g. 2026-07-07 or 2026-07-07T18 (default: latest day only)",
    )
    parser.add_argument(
        "--hours", nargs="+", type=int, default=None, metavar="H",
        help="Only plot files whose valid time is one of these UTC hours, e.g. --hours 0 12",
    )
    parser.add_argument("--no-espc", action="store_true")
    parser.add_argument("--no-cmems", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
        help="Regenerate plots even if they already exist")
    parser.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="Write log output to this file in addition to stdout",
    )
    return parser.parse_args()


def main():
    global plot_espc, plot_cmems

    args = parse_args()

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            args.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    if args.no_espc:
        plot_espc = False
    if args.no_cmems:
        plot_cmems = False
    if args.overwrite:
        kwargs["overwrite"] = True

    start_time = time.time()

    # Load comparison models
    gds = load_model(espc_ts, "ESPC", rename=True) if plot_espc else None

    # CMEMS: load full global, subset per-region later
    cmems_instance = None
    if plot_cmems:
        try:
            cmems_instance = CMEMS()
            logger.info("CMEMS initialized.")
        except Exception as e:
            logger.error("Failed to initialize CMEMS: %s", e)

    # Platform data
    region_configs = {r: region_config(r) for r in args.regions}
    extents = [rc["extent"] for rc in region_configs.values()]
    extent_df = pd.DataFrame(extents, columns=["lonmin", "lonmax", "latmin", "latmax"])
    global_extent = [
        extent_df.lonmin.min(), extent_df.lonmax.max(),
        extent_df.latmin.min(), extent_df.latmax.max(),
    ]

    today = dt.date.today()
    fetch_start = (args.start or pd.Timestamp(today - dt.timedelta(days=1))) - dt.timedelta(hours=conf.search_hours)
    fetch_end   = args.end   or pd.Timestamp(today + dt.timedelta(days=1))
    tstr = "%Y-%m-%d %H:%M:%S"

    argo_data = pd.DataFrame()
    if conf.argo:
        try:
            argo_data = get_argo_floats_by_time(global_extent, fetch_start, fetch_end)
            logger.info("Argo data loaded (%d records).", len(argo_data))
        except Exception as e:
            logger.error("Failed to load Argo data: %s", e)

    glider_data = pd.DataFrame()
    if conf.gliders:
        try:
            glider_data = get_active_gliders(
                global_extent, fetch_start, fetch_end,
                parallel=False, timeout=60,
            )
            logger.info("Glider data loaded (%d records).", len(glider_data))
            if isinstance(glider_data.index, pd.MultiIndex):
                glider_data.index = glider_data.index.set_levels(
                    glider_data.index.levels[0].str.rsplit("-", n=1).str[0], level="glider",
                )
        except Exception as e:
            logger.error("Failed to load Glider data: %s", e)

    bathy_dict = {}
    if conf.bathy:
        for r in args.regions:
            try:
                bathy_dict[r] = get_bathymetry(region_configs[r]["extent"])
            except Exception as e:
                logger.error("Failed to load bathymetry for %s: %s", r, e)

    if args.start or args.end:
        logger.info("Time range: %s → %s", args.start or "earliest", args.end or "latest")
    else:
        logger.info("No time range specified — using latest available day.")

    total_plots = 0
    errors = 0

    for region_name in args.regions:
        rc = region_configs[region_name]
        extent = rc["extent"]
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()
        lon360 = lon180to360(extent_data[:2])

        if "ocean_heat_content" not in rc or rc["ocean_heat_content"] is None:
            logger.info("No OHC config for %s, skipping.", region_name)
            continue

        nc_files = find_files_in_range(args.data_dir, args.source, region_name, args.start, args.end, hours=args.hours)
        if not nc_files:
            logger.warning("No files found for %s in the specified time range.", region_name)
            continue

        logger.info("Found %d files for %s", len(nc_files), region_name)

        for nc_file in nc_files:
            ctime = parse_valid_time(nc_file)
            logger.info("Processing OHC for %s @ %s", region_name, ctime)
            _storms, _forecasts = _get_storms_for_time(ctime, extent)

            kw = copy.deepcopy(kwargs)
            kw["path_save"] = path_save / rc["folder"]
            kw["eez"] = rc.get("eez", False)

            if "limits" in rc["ocean_heat_content"]:
                kw["limits"] = rc["ocean_heat_content"]["limits"]

            if "figure" in rc:
                if "legend" in rc["figure"]:
                    kw["cols"] = rc["figure"]["legend"]["columns"]
                if "figsize" in rc["figure"]:
                    kw["figsize"] = rc["figure"]["figsize"]

            if region_name in bathy_dict:
                kw["bathy"] = bathy_dict[region_name]

            # Load and compute RTOFS OHC
            rds = load_rtofs_binary(
                nc_file,
                extent=extent_data if args.source == "global" else None,
            )

            # Platform data
            search_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
            search_t1 = ctime.strftime(tstr)

            if not argo_data.empty:
                lon, lat = argo_data["lon"], argo_data["lat"]
                mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
                argo_region = argo_data[mask].sort_index()
                kw["argo"] = argo_region.loc[pd.IndexSlice[:, search_t0:search_t1], :]

            if not glider_data.empty:
                lon, lat = glider_data["lon"], glider_data["lat"]
                mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
                glider_region = glider_data[mask]
                glider_region = glider_region[
                    (search_t0 < glider_region.index.get_level_values("time"))
                    & (glider_region.index.get_level_values("time") < search_t1)
                ]
                kw["gliders"] = glider_region

            # RTOFS vs ESPC
            if gds is not None:
                try:
                    gds_time = gds.sel(time=ctime)
                    gds_slice = gds_time.sel(
                        lon=slice(lon180to360(extent_data[0]), lon180to360(extent_data[1])),
                        lat=slice(extent_data[2], extent_data[3]),
                    )
                    gds_slice["lon"] = lon360to180(gds_slice["lon"])
                    gds_slice = compute_ohc(gds_slice)
                    plot_ohc(rds, gds_slice, extent, rc["name"],
                             storms=_storms, forecasts=_forecasts, **kw)
                    total_plots += 1
                    logger.info("Plotted RTOFS vs ESPC OHC for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs ESPC OHC: %s", e, exc_info=True)
                    errors += 1

            # RTOFS vs CMEMS
            if cmems_instance is not None:
                try:
                    cds_sub = cmems_instance.get_combined_subset(
                        extent_data[:2], extent_data[2:], time=ctime,
                    )
                    if cds_sub is not None:
                        cds_sub = compute_ohc(cds_sub)
                        plot_ohc(rds, cds_sub, extent, rc["name"],
                                 storms=_storms, forecasts=_forecasts, **kw)
                        total_plots += 1
                        logger.info("Plotted RTOFS vs CMEMS OHC for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs CMEMS OHC: %s", e, exc_info=True)
                    errors += 1

            rds.close()

    elapsed = time.time() - start_time
    logger.info("Done. %d plot(s) generated, %d error(s) in %.1f min.", total_plots, errors, elapsed / 60)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
