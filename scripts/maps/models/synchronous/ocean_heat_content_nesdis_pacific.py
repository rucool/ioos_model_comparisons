"""
Compare NESDIS satellite-derived ocean heat content (OHC) against
pre-processed RTOFS binary NetCDFs, for Pacific regions (Hawaii, Guam, ...).

Mirrors rtofs_binary_ohc_comparisons.py's binary-file handling (same
find_files_in_range/load_rtofs_binary pattern, same MongoDB dedup via db.py)
but compares against NESDIS (via ioos_model_comparisons.platforms.get_ohc)
instead of ESPC/CMEMS. get_ohc() auto-picks the na/np CoastWatch basin and
handles the np basin's 0-360 longitude convention, so no special-casing is
needed here beyond passing each region's own extent.

Reads the most recent day's files from:
    rtofs_global/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_{region}.nc

Usage:
    python3 scripts/maps/models/synchronous/ocean_heat_content_nesdis_pacific.py
    python3 scripts/maps/models/synchronous/ocean_heat_content_nesdis_pacific.py --regions guam hawaii
    python3 scripts/maps/models/synchronous/ocean_heat_content_nesdis_pacific.py --source global --regions hawaii
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
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import density, ocean_heat_content
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time, get_ohc
from ioos_model_comparisons.plotting import plot_ohc, region_transform
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

SCRIPT_ID = "ohc_nesdis_pacific"

kwargs = {
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
    """Extract valid time from filename like rtofs_glo_20260629T06_hawaii.nc"""
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


def get_nesdis_ohc(extent, ctime):
    """Fetch NESDIS OHC for a single region's own extent (not a combined
    multi-region bbox) - keeps this out of get_ohc()'s basin math entirely,
    since a bbox spanning e.g. both Guam and Hawaii would straddle the
    antimeridian and produce a reversed (min > max) longitude query."""
    try:
        amt = get_ohc(bbox=extent, time=ctime)
    except requests.exceptions.HTTPError:
        amt = None
    if amt is None:
        return None
    amt.attrs["model"] = "NESDIS"
    return amt.rename({"longitude": "lon", "latitude": "lat"})


def _ohc_record(region, ts_dt, m1, m2):
    return {"region": region["name"], "timestamp": ts_dt, "plot_type": "ocean_heat_content",
            "variable": "ohc", "depth": 0, "model1": m1, "model2": m2}


def _expected_plot_keys(ctime, region) -> set:
    """Return the set of (region, iso_ts, plot_type, variable, depth, m1, m2) tuples
    expected for this (ctime, region) — no I/O.

    OHC produces one plot per model pair per region (no variable/depth breakdown).
    """
    if "ocean_heat_content" not in region or region["ocean_heat_content"] is None:
        return set()

    t      = pd.to_datetime(ctime)
    tstr_k = t.strftime("%Y-%m-%dT%H%M%SZ")
    rname  = region["name"]

    return {(rname, tstr_k, "ocean_heat_content", "ohc", 0, "rtofs", "nesdis")}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare OHC from RTOFS binary NetCDFs against NESDIS satellite OHC",
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
        default=["hawaii"],
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
    parser.add_argument("--overwrite", action="store_true",
        help="Regenerate plots even if they already exist")
    parser.add_argument(
        "--log-file", type=Path, default=None, metavar="PATH",
        help="Write log output to this file in addition to stdout",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            args.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    if args.overwrite:
        kwargs["overwrite"] = True

    start_time = time.time()

    # Platform data
    region_configs = {r: apply_colorbar_overrides(r, region_config(r)) for r in args.regions}
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

    has_argo    = isinstance(argo_data,   pd.DataFrame) and not argo_data.empty
    has_gliders = isinstance(glider_data, pd.DataFrame) and not glider_data.empty

    ensure_plot_index()

    total_plots = 0
    errors = 0

    for region_name in args.regions:
        rc = region_configs[region_name]
        extent = rc["extent"]
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()

        if "ocean_heat_content" not in rc or rc["ocean_heat_content"] is None:
            logger.info("No OHC config for %s, skipping.", region_name)
            continue

        nc_files = find_files_in_range(args.data_dir, args.source, region_name, args.start, args.end, hours=args.hours)
        if not nc_files:
            logger.warning("No files found for %s in the specified time range.", region_name)
            continue

        logger.info("Found %d files for %s", len(nc_files), region_name)

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
                    for k in _expected_plot_keys(ctime, rc)
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
            logger.info("Processing NESDIS OHC for %s @ %s", region_name, ctime)

            kw = copy.deepcopy(kwargs)
            kw["transform"] = region_transform(extent)
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

            # RTOFS vs NESDIS
            amt = get_nesdis_ohc(extent_data, ctime)
            if amt is not None:
                try:
                    amt_slice = amt.sel(
                        lon=slice(extent_data[0], extent_data[1]),
                        lat=slice(extent_data[2], extent_data[3]),
                    )
                    plot_ohc(rds, amt_slice.squeeze(), extent, rc["name"], **kw)
                    total_plots += 1
                    pending_logs.append(_ohc_record(rc, ts_dt, "rtofs", "nesdis"))
                    logger.info("Plotted RTOFS vs NESDIS OHC for %s @ %s", region_name, ctime)
                except TypeError as e:
                    logger.warning("Skipping RTOFS vs NESDIS OHC (bad data shape): %s", e)
                except Exception as e:
                    logger.error("Failed RTOFS vs NESDIS OHC: %s", e, exc_info=True)
                    errors += 1
            else:
                logger.info("NESDIS: no data for %s @ %s", region_name, ctime)

            log_plots(SCRIPT_ID, pending_logs, has_argo=has_argo, has_gliders=has_gliders)
            rds.close()

    elapsed = time.time() - start_time
    logger.info("Done. %d plot(s) generated, %d error(s) in %.1f min.", total_plots, errors, elapsed / 60)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
