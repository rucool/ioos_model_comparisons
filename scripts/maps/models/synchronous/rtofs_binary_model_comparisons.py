"""
Compare pre-processed RTOFS binary NetCDFs against other operational models
(ESPC, CMEMS, AMSEAS) using the same plotting functions as
rtofs-gofs-cmems-amseas.py, but reading RTOFS from the rtofs_global/
directory structure instead of OPeNDAP.

Reads the most recent day's files from:
    rtofs_global/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_{region}.nc

Usage:
    python3 scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py

    # Use global files instead of region-specific ones
    python3 scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py --source global

    # Custom data directory
    python3 scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py \
        --data-dir /data/rtofs_global
"""
import matplotlib
matplotlib.use("agg")

import argparse
import copy
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import (
    CMEMS,
    amseas,
    espc_ts,
    espc_ts_archive,
    espc_uv,
    espc_uv_archive,
)
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time, get_goes
from ioos_model_comparisons.plotting import (
    plot_model_region_comparison,
    plot_model_region_comparison_streamplot,
    plot_sst,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

path_save = conf.path_plots / "maps"

SCRIPT_ID = "rtofs_binary"

# Model selection
plot_espc = True
plot_cmems = True
plot_amseas = False

# ESPC only keeps ~1 week of history in the FMRC "best" aggregation; older
# requests must go through the year-based archive endpoints instead.
ESPC_ARCHIVE_CUTOFF_DAYS = 7
_espc_ts_cache = {}
_espc_uv_cache = {}

kwargs = {
    "transform": conf.projection,
    "dpi": conf.dpi,
    "overwrite": True,
    "colorbar": True,
    "legend": True,
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
    """Extract valid time from filename like rtofs_glo_20260629T06_guam.nc"""
    stem = nc_path.stem
    time_part = stem.split("_")[2]
    return pd.Timestamp(dt.datetime.strptime(time_part, "%Y%m%dT%H"))


def load_rtofs_binary(nc_path, extent=None):
    """Load a pre-processed RTOFS NetCDF and adapt it to the format
    expected by plot_model_region_comparison."""
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
    ds = ds.set_coords(["u", "v"])
    ds.attrs["model"] = "RTOFS"
    ds["temperature"].attrs["units"] = "degC"
    ds["salinity"].attrs["units"] = "PSU"
    ds["u"].attrs["units"] = "m/s"
    ds["v"].attrs["units"] = "m/s"

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


def get_espc_ts(ctime):
    """Return the ESPC TS dataset covering ctime, using the FMRC best-forecast
    for recent times and falling back to the year-based archive otherwise."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=ESPC_ARCHIVE_CUTOFF_DAYS)
    key = "best" if ctime >= cutoff else ctime.year
    if key not in _espc_ts_cache:
        if key == "best":
            _espc_ts_cache[key] = load_model(espc_ts, "ESPC TS", rename=True)
        else:
            _espc_ts_cache[key] = load_model(
                espc_ts_archive, f"ESPC TS Archive {key}", rename=True, year=key,
            )
    return _espc_ts_cache[key]


def get_espc_uv(ctime):
    """Return the ESPC UV dataset covering ctime, using the FMRC best-forecast
    for recent times and falling back to the year-based archive otherwise."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=ESPC_ARCHIVE_CUTOFF_DAYS)
    key = "best" if ctime >= cutoff else ctime.year
    if key not in _espc_uv_cache:
        if key == "best":
            _espc_uv_cache[key] = load_model(espc_uv, "ESPC UV", rename=True)
        else:
            _espc_uv_cache[key] = load_model(
                espc_uv_archive, f"ESPC UV Archive {key}", rename=True, year=key,
            )
    return _espc_uv_cache[key]


def attempt_data_load(model, ctime, model_name):
    try:
        if model is None:
            return False, None
        data = model.sel(time=ctime)
        for dim in list(data.dims):
            if "time" in dim.lower():
                try:
                    data = data.sel({dim: ctime}, method="nearest")
                except Exception:
                    data = data.isel({dim: 0})
        return True, data
    except (KeyError, ValueError) as e:
        logger.warning("%s: Data not available for %s - %s", model_name, ctime, e)
        return False, None


def attempt_cmems_load(cmems_instance, ctime, extent):
    try:
        if cmems_instance is None:
            return False, None
        data = cmems_instance.get_combined_subset(extent[:2], extent[2:], time=ctime)
        if data is None:
            return False, None
        return True, data
    except Exception as e:
        logger.warning("CMEMS: Data not available for %s - %s", ctime, e)
        return False, None


def subset_data_lonlat(data, lon_extent, lat_extent):
    try:
        return data.sel(
            lon=slice(lon_extent[0], lon_extent[1]),
            lat=slice(lat_extent[2], lat_extent[3]),
        )
    except Exception as e:
        logger.error("Error during lon/lat subsetting: %s", e)
        return None


def process_sst_data(sst_data, extent, ctime):
    try:
        spatial = sst_data.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
        if "cleaned_sst" in sst_data.data_vars:
            sst = spatial.sel(time=str(ctime.date()), method="nearest")
            sst["SST_C"] = sst["cleaned_sst"]
        else:
            sst = spatial.sel(time=str(ctime), method="nearest")
            sst["SST_C"] = (("lat", "lon"), sst["SST"].values - 273.15)
        return sst
    except Exception as e:
        logger.error("Error processing SST: %s", e)
        return None


def _mc_records(region, ts_dt, m2):
    return [
        {"region": region["name"], "timestamp": ts_dt, "plot_type": "model_comparison",
         "variable": k, "depth": de["depth"], "model1": "rtofs", "model2": m2}
        for k, dl in region["variables"].items() for de in dl
    ]


def _sp_records(region, ts_dt, m2):
    return [
        {"region": region["name"], "timestamp": ts_dt, "plot_type": "streamplot",
         "variable": "currents", "depth": depth, "model1": "rtofs", "model2": m2}
        for depth in region["currents"]["depths"]
    ]


def _sst_record(region, ts_dt, satellite_tag):
    return {"region": region["name"], "timestamp": ts_dt, "plot_type": "sst",
            "variable": "temperature", "depth": 0, "model1": "rtofs", "model2": satellite_tag}


def _expected_plot_keys(ctime, region, has_sst16, has_sst19) -> set:
    """Return the set of (region, iso_ts, plot_type, variable, depth, m1, m2) tuples
    expected for this (ctime, region) — no I/O. Mirrors the plot calls in main()
    so the pre-check can decide whether a file is worth (re)processing."""
    t      = pd.to_datetime(ctime)
    tstr_k = t.strftime("%Y-%m-%dT%H%M%SZ")
    rname  = region["name"]
    keys   = set()

    model_pairs = []
    if plot_espc:
        model_pairs.append("espc")
    if plot_cmems:
        model_pairs.append("cmems")
    if plot_amseas:
        model_pairs.append("amseas")

    for m2 in model_pairs:
        for k, depth_list in region["variables"].items():
            for depth_entry in depth_list:
                keys.add((rname, tstr_k, "model_comparison", k, depth_entry["depth"], "rtofs", m2))

    if region.get("currents", {}).get("bool"):
        for m2 in model_pairs:
            for depth in region["currents"]["depths"]:
                keys.add((rname, tstr_k, "streamplot", "currents", depth, "rtofs", m2))

    for sat in (["GOES16"] if has_sst16 else []) + (["GOES19"] if has_sst19 else []):
        keys.add((rname, tstr_k, "sst", "temperature", 0, "rtofs", sat))

    return keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare RTOFS binary NetCDFs against other models",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path.home() / "Downloads" / "rtofs_global",
    )
    parser.add_argument(
        "--source", default="region",
        choices=["region", "global"],
        help="Use region-specific files or subset from the global file (default: region)",
    )
    parser.add_argument(
        "--regions", nargs="*",
        default=["guam"],
        help="Regions to plot (default: guam)",
    )
    parser.add_argument(
        "--start", type=pd.Timestamp, default=None, metavar="DATETIME",
        help="Start of time range, e.g. 2026-07-01 or 2026-07-01T06 (default: latest day only)",
    )
    parser.add_argument(
        "--end", type=pd.Timestamp, default=None, metavar="DATETIME",
        help="End of time range, e.g. 2026-07-07 or 2026-07-07T18 (default: latest day only)",
    )
    parser.add_argument(
        "--hours", nargs="+", type=int, default=None, metavar="H",
        help="Only plot files whose valid time is one of these UTC hours, e.g. --hours 0 12",
    )
    parser.add_argument(
        "--variables", nargs="+", default=None, metavar="VAR",
        help="Only plot these variables, e.g. --variables temperature salinity",
    )
    parser.add_argument(
        "--depths", nargs="+", type=int, default=None, metavar="D",
        help="Only plot these depths (m), e.g. --depths 0 200",
    )
    parser.add_argument("--no-espc", action="store_true")
    parser.add_argument("--no-cmems", action="store_true")
    parser.add_argument("--amseas", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    global plot_espc, plot_cmems, plot_amseas

    args = parse_args()
    if args.no_espc:
        plot_espc = False
    if args.no_cmems:
        plot_cmems = False
    if args.amseas:
        plot_amseas = True
    if args.overwrite:
        kwargs["overwrite"] = True

    start_time = time.time()

    # Load comparison models (ESPC is loaded lazily, per-time, in the loop
    # below since old requests need the archive rather than the FMRC best).
    cmems_instance = CMEMS() if plot_cmems else None
    am = load_model(amseas, "AMSEAS") if plot_amseas else None

    # Load platform data
    region_configs = [region_config(r) for r in args.regions]
    extents = [rc["extent"] for rc in region_configs]
    extent_df = pd.DataFrame(extents, columns=["lonmin", "lonmax", "latmin", "latmax"])
    global_extent = [
        extent_df.lonmin.min(), extent_df.lonmax.max(),
        extent_df.latmin.min(), extent_df.latmax.max(),
    ]

    today = dt.date.today()
    fetch_start = (args.start or pd.Timestamp(today - dt.timedelta(days=1))) - dt.timedelta(hours=conf.search_hours)
    fetch_end   = args.end   or pd.Timestamp(today + dt.timedelta(days=1))

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
            glider_data = get_active_gliders(global_extent, fetch_start, fetch_end, parallel=False, timeout=60)
            logger.info("Glider data loaded (%d records).", len(glider_data))
            if isinstance(glider_data.index, pd.MultiIndex):
                glider_data.index = glider_data.index.set_levels(
                    glider_data.index.levels[0].str.rsplit("-", n=1).str[0], level="glider",
                )
        except Exception as e:
            logger.error("Failed to load Glider data: %s", e)

    bathy_data = None
    if conf.bathy:
        try:
            bathy_data = get_bathymetry(global_extent)
        except Exception as e:
            logger.error("Failed to load bathymetry: %s", e)

    sst_16 = None
    sst_19 = None
    try:
        sst_16 = get_goes("goes16")
    except Exception as e:
        logger.error("Failed to load GOES-16 SST: %s", e)
    try:
        sst_19 = get_goes("goes19")
    except Exception as e:
        logger.error("Failed to load GOES-19 SST: %s", e)

    if args.start or args.end:
        logger.info("Time range: %s → %s", args.start or "earliest", args.end or "latest")
    else:
        logger.info("No time range specified — using latest available day.")

    has_argo    = isinstance(argo_data,   pd.DataFrame) and not argo_data.empty
    has_gliders = isinstance(glider_data, pd.DataFrame) and not glider_data.empty

    ensure_plot_index()

    total_plots = 0

    for region_name in args.regions:
        region = region_config(region_name)
        region = apply_colorbar_overrides(region_name, region)
        extent = region["extent"]
        extended = np.add(extent, [-1, 1, -1, 1]).tolist()
        lon360 = lon180to360(extended[:2])

        nc_files = find_files_in_range(args.data_dir, args.source, region_name, args.start, args.end, hours=args.hours)
        if not nc_files:
            logger.warning("No files found for %s in the specified time range.", region_name)
            continue

        logger.info("Found %d files for %s", len(nc_files), region_name)
        kwargs["path_save"] = path_save / region["folder"]

        if "eez" in region:
            kwargs["eez"] = region["eez"]
        if region.get("currents", {}).get("bool"):
            kwargs["currents"] = region["currents"]
        if "figure" in region:
            if "legend" in region["figure"]:
                kwargs["cols"] = region["figure"]["legend"]["columns"]
            if "figsize" in region["figure"]:
                kwargs["figsize"] = region["figure"]["figsize"]

        if args.variables or args.depths:
            filtered = {}
            for varname, depth_list in region.get("variables", {}).items():
                if args.variables and varname not in args.variables:
                    continue
                if args.depths:
                    depth_list = [d for d in depth_list if d["depth"] in args.depths]
                if depth_list:
                    filtered[varname] = depth_list
            region = {**region, "variables": filtered}
            logger.info("Variable filter applied: %s", filtered)

        # ── Pre-check: skip files whose expected plots are already logged ────
        # (bypassed entirely by --overwrite, since MongoDB tracking doesn't
        # otherwise know a re-run should force reprocessing)
        file_ctimes = [(f, parse_valid_time(f)) for f in nc_files]
        done_keys = None if kwargs.get("overwrite") else fetch_completed_plot_keys(
            SCRIPT_ID, [ct for _, ct in file_ctimes]
        )
        if kwargs.get("overwrite"):
            logger.info("Pre-check: overwrite=True — reprocessing all %d file(s) for %s.",
                        len(nc_files), region_name)
        elif done_keys is not None:
            pending_files = [
                f for f, ctime in file_ctimes
                if any(
                    needs_replot(k, done_keys, has_argo, has_gliders)
                    for k in _expected_plot_keys(ctime, region, sst_16 is not None, sst_19 is not None)
                )
            ]
            skipped = len(nc_files) - len(pending_files)
            if skipped:
                logger.info("Pre-check (MongoDB): %d/%d file(s) for %s already done — skipping.",
                            skipped, len(nc_files), region_name)
            nc_files = pending_files
            if not nc_files:
                continue
        else:
            logger.warning("MongoDB unavailable — processing all found files for %s.", region_name)

        for nc_file in nc_files:
            ctime = parse_valid_time(nc_file)
            ts_dt = pd.to_datetime(ctime).to_pydatetime()
            pending_logs = []
            logger.info("Processing %s @ %s", region_name, ctime)

            rds = load_rtofs_binary(
                nc_file,
                extent=extended if args.source == "global" else None,
            )

            # Load comparison model data for this time
            gds_ts = get_espc_ts(ctime) if plot_espc else None
            gds_uv = get_espc_uv(ctime) if plot_espc else None
            gdt_flag, gdt_ts_sub = attempt_data_load(gds_ts, ctime, "ESPC TS")
            _, gdt_uv_sub = attempt_data_load(gds_uv, ctime, "ESPC UV")
            cdt_flag, cdt = attempt_cmems_load(cmems_instance, ctime, extended)
            amt_flag, amt_sub = attempt_data_load(am, ctime, "AMSEAS")

            if gdt_flag and gdt_ts_sub is not None:
                gdt_ts_sub = subset_data_lonlat(gdt_ts_sub, lon360, extended)
                if gdt_ts_sub is not None:
                    gdt_ts_sub["lon"] = lon360to180(gdt_ts_sub["lon"])
            if gdt_flag and gdt_uv_sub is not None:
                gdt_uv_sub = subset_data_lonlat(gdt_uv_sub, lon360, extended)
                if gdt_uv_sub is not None:
                    gdt_uv_sub["lon"] = lon360to180(gdt_uv_sub["lon"])
            if amt_flag and amt_sub is not None:
                amt_sub = subset_data_lonlat(amt_sub, lon360, extended)

            # Platform data for this region/time
            tstr = "%Y-%m-%d %H:%M:%S"
            search_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
            search_t1 = ctime.strftime(tstr)

            kw = copy.deepcopy(kwargs)
            if not argo_data.empty:
                lon, lat = argo_data["lon"], argo_data["lat"]
                mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
                argo_region = argo_data[mask].sort_index()
                kw["argo"] = argo_region.loc[pd.IndexSlice[:, search_t0:search_t1], :]

            if not glider_data.empty:
                lon, lat = glider_data["lon"], glider_data["lat"]
                mask = (extended[0] <= lon) & (lon <= extended[1]) & (extended[2] <= lat) & (lat <= extended[3])
                glider_region = glider_data[mask]
                glider_region = glider_region[
                    (search_t0 < glider_region.index.get_level_values("time"))
                    & (glider_region.index.get_level_values("time") < search_t1)
                ]
                kw["gliders"] = glider_region

            if bathy_data is not None:
                kw["bathy"] = bathy_data

            # RTOFS vs ESPC
            if gdt_flag and gdt_ts_sub is not None:
                try:
                    plot_model_region_comparison(rds, gdt_ts_sub, region, **kw)
                    total_plots += 1
                    pending_logs.extend(_mc_records(region, ts_dt, "espc"))
                    logger.info("Plotted RTOFS vs ESPC TS for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs ESPC TS: %s", e, exc_info=True)

                try:
                    if gdt_uv_sub is not None:
                        plot_model_region_comparison_streamplot(rds, gdt_uv_sub, region, **kw)
                        total_plots += 1
                        pending_logs.extend(_sp_records(region, ts_dt, "espc"))
                        logger.info("Plotted RTOFS vs ESPC currents for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs ESPC currents: %s", e, exc_info=True)

            # RTOFS vs CMEMS
            if cdt_flag and cdt is not None:
                try:
                    plot_model_region_comparison(rds, cdt, region, **kw)
                    total_plots += 1
                    pending_logs.extend(_mc_records(region, ts_dt, "cmems"))
                    logger.info("Plotted RTOFS vs CMEMS TS for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs CMEMS TS: %s", e, exc_info=True)

                try:
                    plot_model_region_comparison_streamplot(rds, cdt, region, **kw)
                    total_plots += 1
                    pending_logs.extend(_sp_records(region, ts_dt, "cmems"))
                    logger.info("Plotted RTOFS vs CMEMS currents for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs CMEMS currents: %s", e, exc_info=True)

            # RTOFS vs AMSEAS
            if amt_flag and amt_sub is not None:
                try:
                    plot_model_region_comparison(rds, amt_sub, region, **kw)
                    plot_model_region_comparison_streamplot(rds, amt_sub, region, **kw)
                    total_plots += 2
                    pending_logs.extend(_mc_records(region, ts_dt, "amseas"))
                    pending_logs.extend(_sp_records(region, ts_dt, "amseas"))
                    logger.info("Plotted RTOFS vs AMSEAS for %s @ %s", region_name, ctime)
                except Exception as e:
                    logger.error("Failed RTOFS vs AMSEAS: %s", e, exc_info=True)

            # SST
            kw_sst = {k: v for k, v in kw.items() if k not in ("eez", "currents", "legend")}
            if sst_16 is not None:
                sst = process_sst_data(sst_16, extent, ctime)
                if sst is not None:
                    try:
                        plot_sst(rds, sst, region, satellite="GOES-16", **kw_sst)
                        total_plots += 1
                        pending_logs.append(_sst_record(region, ts_dt, "GOES16"))
                    except Exception as e:
                        logger.error("Failed GOES-16 SST: %s", e)

            if sst_19 is not None:
                sst = process_sst_data(sst_19, extent, ctime)
                if sst is not None:
                    try:
                        plot_sst(rds, sst, region, satellite="GOES-19", **kw_sst)
                        total_plots += 1
                        pending_logs.append(_sst_record(region, ts_dt, "GOES19"))
                    except Exception as e:
                        logger.error("Failed GOES-19 SST: %s", e)

            log_plots(SCRIPT_ID, pending_logs, has_argo=has_argo, has_gliders=has_gliders)
            rds.close()

    elapsed = time.time() - start_time
    logger.info("Done. %d plots generated in %.1f min.", total_plots, elapsed / 60)


if __name__ == "__main__":
    main()
