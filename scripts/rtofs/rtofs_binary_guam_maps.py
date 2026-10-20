"""
Plot Guam depth maps from pre-processed RTOFS z-level NetCDFs.

Reads the most recent day's files from the rtofs_global directory structure:
    rtofs_global/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_guam.nc

Produces temperature, salinity, and current-speed maps at every depth
defined in region_config('guam').

Usage:
    python3 scripts/rtofs/rtofs_binary_guam_maps.py
    python3 scripts/rtofs/rtofs_binary_guam_maps.py --data-dir /data/rtofs_global
    python3 scripts/rtofs/rtofs_binary_guam_maps.py --overwrite
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cool_maps.plot import create

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.regions import region_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REGION = "guam"
DATA_PROJECTION = ccrs.PlateCarree()
MAP_PROJECTION = ccrs.Mercator()

CMAPS = {
    "temperature": cmocean.cm.thermal,
    "salinity": cmocean.cm.haline,
    "currents": cmocean.cm.speed,
}


def find_latest_day(data_dir, region):
    """Walk the YYYY/MM/YYYYMMDD tree and return the most recent day
    directory that contains files for the given region."""
    candidates = sorted(data_dir.glob(f"*/*/*/rtofs_glo_*_{region}.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"No {region} NetCDF files found under {data_dir}"
        )
    latest_day_dir = candidates[-1].parent
    return latest_day_dir


def parse_valid_time(nc_path):
    """Extract valid time from filename like rtofs_glo_20260629T06_guam.nc"""
    stem = nc_path.stem
    time_part = stem.split("_")[2]
    return datetime.strptime(time_part, "%Y%m%dT%H")


def save_figure(fig, save_file, dpi=300):
    os.makedirs(save_file.parent, exist_ok=True)
    fig.savefig(save_file, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info("Saved: %s", save_file)


def plot_scalar(ds, var_name, depth, limits, extent, time_str, save_file,
                cmap, label, units, dpi=300):
    vmin, vmax, stride = limits
    levels = np.arange(vmin, vmax + stride, stride)

    LONS, LATS = np.meshgrid(ds.lon.values, ds.lat.values)
    da = ds[var_name].sel(z=depth)

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw=dict(projection=MAP_PROJECTION),
    )
    create(extent, ax=ax, ticks=True)

    cs = ax.contourf(
        LONS, LATS, da.values,
        levels=levels, cmap=cmap,
        transform=DATA_PROJECTION, transform_first=True, extend="both",
    )
    cb = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.06, aspect=40)
    cb.set_label(units, fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    ax.set_title(
        f"{label} ({depth} m) — {time_str}",
        fontsize=14, fontweight="bold", pad=10,
    )

    save_figure(fig, save_file, dpi=dpi)


def plot_currents(ds, depth, limits, extent, time_str, save_file, dpi=300):
    vmin, vmax, stride = limits
    levels = np.arange(vmin, vmax + stride, stride)

    LONS, LATS = np.meshgrid(ds.lon.values, ds.lat.values)
    u = ds["u-vel."].sel(z=depth)
    v = ds["v-vel."].sel(z=depth)
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw=dict(projection=MAP_PROJECTION),
    )
    create(extent, ax=ax, ticks=True)

    cs = ax.contourf(
        LONS, LATS, speed.values,
        levels=levels, cmap=CMAPS["currents"],
        transform=DATA_PROJECTION, transform_first=True, extend="max",
    )
    ax.streamplot(
        ds.lon.values, ds.lat.values,
        u.values, v.values,
        transform=DATA_PROJECTION, color="black", density=3, linewidth=0.5,
    )

    cb = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.06, aspect=40)
    cb.set_label("m/s", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    ax.set_title(
        f"Current Speed ({depth} m) — {time_str}",
        fontsize=14, fontweight="bold", pad=10,
    )

    save_figure(fig, save_file, dpi=dpi)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Guam depth maps from pre-processed RTOFS NetCDFs",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path.home() / "Downloads" / "rtofs_global",
        help="Root directory of processed NetCDFs (default: ~/Downloads/rtofs_global)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    region = region_config(regions=[REGION])
    extent = region["extent"]

    latest_dir = find_latest_day(args.data_dir, REGION)
    nc_files = sorted(latest_dir.glob(f"rtofs_glo_*_{REGION}.nc"))
    logger.info("Found %d files in %s", len(nc_files), latest_dir)

    path_save = conf.path_plots / "maps" / region["folder"]

    for nc_file in nc_files:
        valid_time = parse_valid_time(nc_file)
        time_str = valid_time.strftime("%Y-%m-%dT%H:%MZ")
        year = valid_time.strftime("%Y")
        month = valid_time.strftime("%m")
        timestamp_tag = valid_time.strftime("%Y-%m-%dT%H%M%SZ")

        logger.info("Processing %s (valid: %s)", nc_file.name, time_str)
        ds = xr.open_dataset(nc_file)

        for var_key, var_list in region["variables"].items():
            for item in var_list:
                depth = item["depth"]
                if "limits" not in item:
                    logger.warning("No limits for %s @ %dm, skipping.", var_key, depth)
                    continue

                var_name = "temp" if var_key == "temperature" else "salin"
                cmap = CMAPS.get(var_key, cmocean.cm.thermal)
                units = "°C" if var_key == "temperature" else "PSU"
                label = var_key.replace("_", " ").title()

                save_dir = path_save / f"{var_key}_{depth}m" / year / month
                sname = f"guam_{timestamp_tag}_{var_key}-{depth}m_rtofs-binary.png"
                save_file = save_dir / sname

                if save_file.is_file() and not args.overwrite:
                    logger.info("%s exists, skipping.", save_file)
                    continue

                logger.info("Plotting %s @ %dm", var_key, depth)
                plot_scalar(
                    ds, var_name, depth, item["limits"], extent, time_str,
                    save_file, cmap, label, units, dpi=args.dpi,
                )

        if region.get("currents"):
            cdict = region["currents"]
            for depth in cdict["depths"]:
                save_dir = path_save / f"currents_{depth}m" / year / month
                sname = f"guam_{timestamp_tag}_currents-{depth}m_rtofs-binary.png"
                save_file = save_dir / sname

                if save_file.is_file() and not args.overwrite:
                    logger.info("%s exists, skipping.", save_file)
                    continue

                logger.info("Plotting currents @ %dm", depth)
                plot_currents(
                    ds, depth, cdict["limits"], extent, time_str,
                    save_file, dpi=args.dpi,
                )

        ds.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
