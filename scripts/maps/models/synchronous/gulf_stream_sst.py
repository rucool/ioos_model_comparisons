"""
Compare RTOFS sea surface temperature against ESPC, CMEMS, ECCOFS, and GOES-16/
GOES-19 satellite SST across the Gulf Stream — from the Mid Atlantic Bight
shelf break out into the open mid-Atlantic — using both the "turbo" and
cmocean "thermal" colormaps, one PNG per colormap per comparison per
timestamp.

Extent, depth (surface only), and the temperature colorbar range come from
the "gulf_stream" region in ioos_model_comparisons/regions.py.

ECCOFS only publishes one nowcast per calendar day (00Z), so RTOFS-vs-ECCOFS
is only attempted at 00Z timestamps — see ECCOFS_HOUR.

Usage:
    python3 scripts/maps/models/synchronous/gulf_stream_sst.py

    # Explicit time range
    python3 scripts/maps/models/synchronous/gulf_stream_sst.py \
        --start 2026-08-01T00 --end 2026-08-03T00 --freq 6H

    # Only some comparisons, skip already-plotted outputs
    python3 scripts/maps/models/synchronous/gulf_stream_sst.py --no-cmems --no-overwrite

    # Only the turbo colormap
    python3 scripts/maps/models/synchronous/gulf_stream_sst.py --no-thermal
"""
import matplotlib
matplotlib.use("agg")

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cmocean
from cool_maps.plot import add_bathymetry, add_features, add_ticks, get_bathymetry

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import rtofs, espc_ts, CMEMS, ECCOFSFullDepth
from ioos_model_comparisons.platforms import get_goes
from ioos_model_comparisons.plotting import add_colorbar
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.db import apply_colorbar_overrides

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

path_save = conf.path_plots / "maps"

REGION = "gulf_stream"

# ECCOFS only has one nowcast/day (00Z) — see ECCOFSFullDepth in models.py —
# so RTOFS-vs-ECCOFS is only attempted at this hour.
ECCOFS_HOUR = 0

# name -> (matplotlib/cmocean colormap object, short tag used in filenames)
COLORMAPS = {
    "turbo":   (plt.get_cmap("turbo"), "turbo"),
    "thermal": (cmocean.cm.thermal,    "thermal"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare RTOFS surface temperature against ESPC/CMEMS/ECCOFS across the Gulf Stream, in turbo and cmocean thermal.",
    )
    parser.add_argument(
        "--start", type=pd.Timestamp, default=None, metavar="DATETIME",
        help="Start of time range, e.g. 2026-08-01 or 2026-08-01T18 (default: RTOFS's latest available time)",
    )
    parser.add_argument(
        "--end", type=pd.Timestamp, default=None, metavar="DATETIME",
        help="End of time range (default: same as --start / latest available time)",
    )
    parser.add_argument(
        "--freq", default="6H",
        help="Frequency for the generated timestamp list when --start/--end are given (default: 6H)",
    )
    parser.add_argument("--no-espc", action="store_true", help="Disable RTOFS vs ESPC")
    parser.add_argument("--no-cmems", action="store_true", help="Disable RTOFS vs CMEMS")
    parser.add_argument("--no-eccofs", action="store_true", help="Disable RTOFS vs ECCOFS")
    parser.add_argument("--no-goes16", action="store_true", help="Disable RTOFS vs GOES-16 SST")
    parser.add_argument("--no-goes19", action="store_true", help="Disable RTOFS vs GOES-19 SST")
    parser.add_argument("--no-thermal", action="store_true", help="Skip the cmocean 'thermal' colormap, only plot 'turbo'")
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Skip regenerating plots that already exist (default: overwrite existing plots)",
    )
    return parser.parse_args()


def load_model(model_func, model_name, **kw):
    try:
        logger.info(f"Loading {model_name} model data.")
        data = model_func(**kw)
        logger.info(f"{model_name} model data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None


def attempt_data_load(model, ctime, model_name):
    """Select the nearest available timestep, collapsing any secondary
    time-like dimensions (e.g. ESPC's FMRC time1/time2) down to 2-D."""
    try:
        if model is None:
            raise ValueError(f"{model_name} model data is not available.")
        data = model.sel(time=ctime, method="nearest")
        for dim in list(data.dims):
            if "time" in dim.lower():
                data = data.isel({dim: 0})
        actual = pd.Timestamp(data.time.values) if "time" in data.coords else None
        if actual is not None and abs((actual - pd.Timestamp(ctime)).total_seconds()) > 3 * 3600:
            logger.warning(f"{model_name}: nearest available time to {ctime} is {actual} (>3h away)")
        return True, data
    except Exception as e:
        logger.warning(f"{model_name}: data not available near {ctime} - {e}")
        return False, None


def subset_data(data, extent, grid_lons, grid_lats, grid_x, grid_y):
    """Subset a curvilinear RTOFS-style (x/y + 2-D lon/lat) dataset to a bounding box."""
    lons_ind = np.interp(extent[:2], grid_lons, grid_x)
    lats_ind = np.interp(extent[2:], grid_lats, grid_y)
    extent_ind = [int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1])),
                  int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1]))]
    return data.isel(x=slice(extent_ind[0], extent_ind[1]),
                      y=slice(extent_ind[2], extent_ind[3]))


def subset_data_lonlat(data, lon_extent, lat_extent):
    """Subset a 1-D lon/lat dataset (e.g. ESPC) using lon/lat slices."""
    return data.sel(lon=slice(lon_extent[0], lon_extent[1]),
                     lat=slice(lat_extent[2], lat_extent[3]))


def subset_data_curvilinear(data, extent):
    """Subset a curvilinear-grid dataset (e.g. ECCOFS) with 2-D lon/lat coords."""
    lon = data["lon"].values
    lat = data["lat"].values
    lonmin, lonmax, latmin, latmax = extent
    mask = (lon >= lonmin) & (lon <= lonmax) & (lat >= latmin) & (lat <= latmax)
    eta_i, xi_i = np.where(mask)
    if len(eta_i) == 0:
        return data
    return data.isel(
        y=slice(int(eta_i.min()), int(eta_i.max()) + 1),
        x=slice(int(xi_i.min()), int(xi_i.max()) + 1),
    )


def surface_temperature(ds):
    try:
        return ds["temperature"].sel(depth=0, method="nearest")
    except KeyError:
        return ds["temperature"].isel(depth=0)


def process_goes_sst(sst_data, extent, ctime):
    """Subset a GOES SST dataset to `extent`/`ctime`, returning a Celsius DataArray.
    GOES-16 (hourly, Kelvin, 'SST') selects the nearest timestamp; GOES-19
    (daily average, already Celsius, 'cleaned_sst') selects the nearest day."""
    spatial = sst_data.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
    if "cleaned_sst" in sst_data.data_vars:
        sst = spatial.sel(time=str(pd.Timestamp(ctime).date()), method="nearest")
        return sst["cleaned_sst"]
    sst = spatial.sel(time=str(ctime), method="nearest")
    return sst["SST"] - 273.15


def plot_temperature_comparison(rds, ods, region, cmap, cmap_tag, bathy=None, overwrite=False):
    """Two-panel RTOFS-vs-other surface temperature comparison, saved as one PNG.
    Returns the saved path, or None if it already existed and overwrite=False."""
    extent = region["extent"]
    folder = region["folder"]
    dashed = "-".join(folder.split("_"))
    depth  = 0
    time1  = pd.to_datetime(rds.time.values)
    time2  = pd.to_datetime(ods.time.values)
    other  = ods.attrs["model"]

    save_dir  = path_save / folder / f"temperature_{depth}m_{cmap_tag}" / time1.strftime("%Y/%m")
    save_dir.mkdir(parents=True, exist_ok=True)
    sname = (f"{dashed}_{time1.strftime('%Y-%m-%dT%H%M%SZ')}_temperature-{depth}m_"
             f"rtofs-vs-{other.lower()}-{cmap_tag}.png")
    save_file = save_dir / sname

    if save_file.is_file() and not overwrite:
        logger.info(f"{save_file} exists. Overwrite=False. Skipping.")
        return None

    vmin, vmax, stride = region["variables"]["temperature"][0]["limits"]
    levels = np.arange(vmin, vmax + stride, stride)

    # Matches plot_model_region_comparison()'s grid exactly (map row + a
    # short row below it) — dropping the second row entirely, rather than
    # just leaving it empty, throws off constrained-layout's sizing and
    # leaves a large blank gap above the maps.
    fig, _ = plt.subplot_mosaic(
        "RG\nLL",
        figsize=region["figure"]["figsize"],
        layout="constrained",
        subplot_kw={"projection": conf.projection["map"]},
        gridspec_kw={"height_ratios": [4, 1]},
    )
    ax1, ax2, ax3 = fig.axes
    ax3.set_axis_off()

    for ax in (ax1, ax2):
        # crs must be given explicitly: set_extent(extent) with no crs re-
        # projects the extent rectangle using a coarser line-subdivision
        # threshold for non-PlateCarree map projections (Mercator here),
        # which overshoots the requested bounds — up to ~0.75° north for
        # this region. Passing crs=data avoids that and honors `extent`
        # exactly.
        ax.set_extent(extent, crs=conf.projection["data"])
        add_features(ax)
        if bathy is not None:
            try:
                add_bathymetry(ax, bathy.longitude.values, bathy.latitude.values,
                                bathy.z.values, levels=(-1000, -100), zorder=1.5)
            except ValueError:
                logger.warning("Bathymetry deeper than specified levels.")
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    cargs = dict(levels=levels, cmap=cmap, extend="both", transform=conf.projection["data"])
    h = ax1.contourf(rds["lon"], rds["lat"], rds.squeeze(), **cargs)
    ax2.contourf(ods["lon"], ods["lat"], ods.squeeze(), **cargs)

    cb = add_colorbar([ax1, ax2], h, location="bottom")
    cb.set_label("Sea Water Temperature (°C)", fontsize=12)

    ax1.set_title(f"RTOFS\n{time1.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=15, fontweight="bold")
    ax2.set_title(f"{other}\n{time2.strftime('%Y-%m-%d %H:%M UTC')}", fontsize=15, fontweight="bold")
    fig.suptitle(f"Sea Water Temperature ({depth} m)", fontsize=17, fontweight="bold")

    fig.savefig(save_file, dpi=conf.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"Saved {save_file}")
    return save_file


def main():
    args = parse_args()
    region = region_config(REGION)
    region = apply_colorbar_overrides(REGION, region)
    extent = region["extent"]
    colormaps = {k: v for k, v in COLORMAPS.items() if not (k == "thermal" and args.no_thermal)}

    # Data is fetched a bit wider than the plotted extent: cartopy's
    # set_extent() expands the axes view to match the panel's box aspect
    # ratio (needed here since the Gulf Stream domain is unusually wide/
    # short), which can push the visible view beyond `extent` — padding the
    # underlying data fetch means that expansion still finds real data
    # instead of blank background. Same convention as the padding used
    # throughout rtofs-gofs-cmems-amseas.py.
    extended     = np.add(extent, [-1, 1, -1, 1]).tolist()
    extended_360 = lon180to360(extended[:2])

    rds_full = load_model(rtofs, "RTOFS")
    if rds_full is None:
        logger.error("RTOFS failed to load — nothing to compare against. Exiting.")
        return
    grid_lons = rds_full.lon.values[0, :]
    grid_lats = rds_full.lat.values[:, 0]
    grid_x    = rds_full.x.values
    grid_y    = rds_full.y.values

    plot_espc   = not args.no_espc
    plot_cmems  = not args.no_cmems
    plot_eccofs = not args.no_eccofs
    plot_goes16 = not args.no_goes16
    plot_goes19 = not args.no_goes19

    espc_full = load_model(espc_ts, "ESPC", rename=True) if plot_espc else None
    cmems     = CMEMS() if plot_cmems else None
    eccofs    = ECCOFSFullDepth() if plot_eccofs else None
    sst16     = load_model(get_goes, "GOES-16", satellite="goes16") if plot_goes16 else None
    sst19     = load_model(get_goes, "GOES-19", satellite="goes19") if plot_goes19 else None

    try:
        bathy = get_bathymetry(extended)
        logger.info("Bathymetry data loaded.")
    except Exception as e:
        logger.error(f"Failed to load bathymetry: {e}")
        bathy = None

    if args.start is None and args.end is None:
        date_list = [pd.Timestamp(rds_full.time.values[-1])]
        logger.info(f"No time range specified — using RTOFS's latest available time: {date_list[0]}")
    else:
        start = args.start or args.end
        end   = args.end or args.start
        date_list = pd.date_range(start, end, freq=args.freq)
        logger.info(f"Time range: {start} → {end} (freq={args.freq})")

    overwrite   = not args.no_overwrite
    total_plots = 0

    for ctime in date_list:
        rdt_flag, rdt = attempt_data_load(rds_full, ctime, "RTOFS")
        if not rdt_flag:
            continue
        rds_sub = subset_data(rdt, extended, grid_lons, grid_lats, grid_x, grid_y)
        rtofs_temp = surface_temperature(rds_sub)
        rtofs_temp.attrs.setdefault("model", "RTOFS")

        comparisons = []

        if plot_espc:
            gdt_flag, gdt = attempt_data_load(espc_full, ctime, "ESPC")
            if gdt_flag:
                gds_sub = subset_data_lonlat(gdt, extended_360, extended)
                gds_sub = gds_sub.assign_coords(lon=lon360to180(gds_sub["lon"]))
                espc_temp = surface_temperature(gds_sub)
                espc_temp.attrs["model"] = "ESPC"
                comparisons.append(espc_temp)

        if plot_cmems:
            try:
                cds = cmems.get_combined_subset(extended[:2], extended[2:], time=ctime)
                cmems_temp = surface_temperature(cds)
                cmems_temp.attrs["model"] = "CMEMS"
                comparisons.append(cmems_temp)
            except Exception as e:
                logger.warning(f"CMEMS: data not available near {ctime} - {e}")

        if plot_eccofs and pd.Timestamp(ctime).hour == ECCOFS_HOUR:
            try:
                ect = eccofs.sel(time=ctime)
                ect_sub = subset_data_curvilinear(ect, extended)
                eccofs_temp = surface_temperature(ect_sub)
                eccofs_temp.attrs["model"] = "ECCOFS"
                comparisons.append(eccofs_temp)
            except Exception as e:
                logger.warning(f"ECCOFS: data not available near {ctime} - {e}")

        if plot_goes16 and sst16 is not None:
            try:
                goes16_temp = process_goes_sst(sst16, extended, ctime)
                goes16_temp.attrs["model"] = "GOES-16"
                comparisons.append(goes16_temp)
            except Exception as e:
                logger.warning(f"GOES-16: data not available near {ctime} - {e}")

        if plot_goes19 and sst19 is not None:
            try:
                goes19_temp = process_goes_sst(sst19, extended, ctime)
                goes19_temp.attrs["model"] = "GOES-19"
                comparisons.append(goes19_temp)
            except Exception as e:
                logger.warning(f"GOES-19: data not available near {ctime} - {e}")

        for comp_temp in comparisons:
            for cmap, cmap_tag in colormaps.values():
                saved = plot_temperature_comparison(rtofs_temp, comp_temp, region, cmap, cmap_tag, bathy=bathy, overwrite=overwrite)
                if saved is not None:
                    total_plots += 1

    logger.info(f"Done. {total_plots} plot(s) generated.")


if __name__ == "__main__":
    main()
