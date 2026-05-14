#!/usr/bin/env python
import io
import os
import re
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

import cartopy.crs as ccrs
import requests
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ioos_model_comparisons.calc import lon180to360, lon360to180, depth_interpolate
import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt

save_dir = conf.path_plots / 'profiles' / 'fvon'
os.makedirs(save_dir, exist_ok=True)

# Configs
parallel = True
replot = False   # set True to overwrite existing plots
depth = 400

# Which models should we plot?
plot_rtofs = False
plot_espc = True
plot_cmems = True

days = 6

# Allowlist of WIGOS IDs to plot — set to None to process all instruments.
# wigos_ids_filter = [
#     "0-22000-0-QNE94JY",
#     "0-22000-0-DZK2PNH",
#     "0-22000-0-PECEQRL",
#     "0-22000-0-FYNTR3N",
# ]
wigos_ids_filter = None

# Debug filter — set to limit processing to a single instrument/date.
# Set to None to process everything normally.
debug_instrument = None   # e.g. "1159"
debug_date = None         # e.g. "2026-04-20"
dpi = conf.dpi

depths = slice(0, depth)

# THREDDS access — catalog XML to enumerate files, HTTPServer to fetch them.
# This avoids S3 Parquet entirely; each .nc is one tow profile (35–268 KB).
_THREDDS = "https://thredds.aodn.org.au/thredds"
_CATALOG_URL = _THREDDS + "/catalog/IMOS/SOOP/SOOP-FishSOOP/REALTIME/{year}/{month:02d}/catalog.xml"
_FILE_URL    = _THREDDS + "/fileServer/IMOS/SOOP/SOOP-FishSOOP/REALTIME/{year}/{month:02d}/{fname}"

# Broad ocean basin regions — same convention as fvon_profile_model_comparisons.py.
# Extent: [lonmin, lonmax, latmin, latmax].  lonmin > lonmax means antimeridian crossing.
# FishSOOP is Australian, so only indian_ocean / south_pacific / southern_ocean will have data.
FVON_REGIONS = [
    # {"name": "North Atlantic",  "folder": "north_atlantic",  "extent": [-100,  30,   0,  70]},
    # {"name": "South Atlantic",  "folder": "south_atlantic",  "extent": [ -65,  25, -60,   0]},
    # {"name": "Mediterranean",   "folder": "mediterranean",   "extent": [  -8,  42,  30,  48]},
    # {"name": "Indian Ocean",    "folder": "indian_ocean",    "extent": [  25, 120, -60,  30]},
    # {"name": "North Pacific",   "folder": "north_pacific",   "extent": [ 100, -110,  0,  70]},
    {"name": "South Pacific",   "folder": "south_pacific",   "extent": [ 140,  -65, -60,   0]},
    # {"name": "Southern Ocean",  "folder": "southern_ocean",  "extent": [-180,  180, -90, -60]},
    # {"name": "Arctic",          "folder": "arctic",          "extent": [-180,  180,  65,  90]},
]


def filter_region(df, extent):
    """Filter a DataFrame to rows within extent, handling antimeridian-crossing regions."""
    lonmin, lonmax, latmin, latmax = extent
    lat_mask = (df['lat'] >= latmin) & (df['lat'] <= latmax)
    if lonmin <= lonmax:
        lon_mask = (df['lon'] >= lonmin) & (df['lon'] <= lonmax)
    else:
        lon_mask = (df['lon'] >= lonmin) | (df['lon'] <= lonmax)
    return df[lat_mask & lon_mask]


# Paired (depth_limit, tick_stride) — 5–7 ticks per axis.
_DEPTH_CONFIGS = [
    (10,    5), (15,    5), (20,    5), (25,    5), (30,    5),
    (40,   10), (50,   10), (75,   15), (100,  20), (150,  25),
    (200,  50), (250,  50), (300,  50), (400, 100), (500, 100),
    (750, 150), (1000, 200),
]


def smart_depth_axis(ax, obs_max_depth):
    """Set y-axis limit and ticks based on the observation's actual max depth."""
    lim, stride = next(
        ((lim, stride) for lim, stride in _DEPTH_CONFIGS if lim >= obs_max_depth),
        _DEPTH_CONFIGS[-1],
    )
    ticks = np.arange(0, lim + stride, stride)
    ax.set_ylim(lim, 0)
    ax.set_yticks(ticks)
    return lim


def extract_downcast(df):
    """Return only the downcast portion of a tow profile.

    Sorts by measurement time, finds the index of maximum depth, then keeps
    only rows up to and including that point. This removes the upcast and
    eliminates the zigzag caused by interleaved up/downcast data at the same depths.
    """
    df = df.sort_values('time').reset_index(drop=True)
    max_depth_idx = df['depth'].idxmax()
    return df.iloc[: max_depth_idx + 1]


def local_map_extent(lon, lat, km=500):
    """Return a ±km extent as [lonmin, lonmax, latmin, latmax].

    Converts km to degrees separately for lat and lon so the box is
    approximately square in distance space. Longitude is NOT clamped to ±180;
    the caller must use a projection centered at lon so cartopy renders it
    correctly near the antimeridian.
    """
    lat_delta = km / 111.0
    lon_delta = km / (111.0 * np.cos(np.radians(lat)))
    return [
        lon - lon_delta,
        lon + lon_delta,
        max(lat - lat_delta, -90),
        min(lat + lat_delta,  90),
    ]


def line_limits(fax, delta=1.0):
    """Get the min/max x-data across all lines on a Matplotlib axis."""
    mins = [np.nanmin(line.get_xdata()) for line in fax.lines]
    maxs = [np.nanmax(line.get_xdata()) for line in fax.lines]
    return min(mins) - delta, max(maxs) + delta


# ---------------------------------------------------------------------------
# THREDDS catalog helpers
# ---------------------------------------------------------------------------

def _catalog_filenames(year, month):
    """Return all .nc filenames listed in the THREDDS catalog for a given month."""
    url = _CATALOG_URL.format(year=year, month=month)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Catalog fetch failed {year}/{month:02d}: {e}")
        return []

    ns = "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"
    root = ET.fromstring(resp.content)
    return [
        el.attrib["name"]
        for el in root.iter(f"{{{ns}}}dataset")
        if el.attrib.get("name", "").endswith(".nc")
    ]


def _fname_to_dt(fname):
    """Parse the tow-start timestamp embedded in a FishSOOP filename."""
    # e.g. IMOS_SOOP-FishSOOP_TP_20260401T013753Z_FV01_84.nc
    m = re.search(r"_(\d{8}T\d{6})Z_", fname)
    if m:
        return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
    return None


def _instrument_id(fname):
    """Extract the instrument serial number from the filename (e.g. '84')."""
    m = re.search(r"FV01_(\w+)\.nc$", fname)
    return m.group(1) if m else fname


def _wigos_id(ds):
    """Extract the WIGOS ID from the global 'abstract' attribute."""
    abstract = ds.attrs.get("abstract", "")
    m = re.search(r"WIGOS ID ([\w-]+)", abstract)
    return m.group(1) if m else "unknown"


def _download_profile(year, month, fname):
    """Download one FishSOOP NetCDF file and return a cleaned DataFrame."""
    url = _FILE_URL.format(year=year, month=month, fname=fname)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Download failed {fname}: {e}")
        return None

    try:
        ds = xr.open_dataset(io.BytesIO(resp.content))
    except Exception as e:
        print(f"Cannot parse {fname}: {e}")
        return None

    df = ds[["TIME", "LATITUDE", "LONGITUDE", "DEPTH",
             "TEMPERATURE", "TEMPERATURE_quality_control"]].to_dataframe().reset_index()

    # Good-data QC flag only
    df = df[df["TEMPERATURE_quality_control"] == 1]
    df = df.dropna(subset=["TEMPERATURE", "DEPTH"])

    if df.empty:
        return None

    df = df.rename(columns={
        "LATITUDE":    "lat",
        "LONGITUDE":   "lon",
        "TEMPERATURE": "temperature",
        "DEPTH":       "depth",
    })

    # TIME is decoded by xarray; ensure tz-naive
    df["time"] = pd.to_datetime(df["TIME"]).dt.tz_localize(None)
    df["wmo_id"]   = _instrument_id(fname)
    df["wigos_id"] = _wigos_id(ds)
    df["tow_id"]   = fname          # unique per tow — used as profile key

    # Keep only columns needed downstream
    return df[["wmo_id", "wigos_id", "tow_id", "time", "lat", "lon", "depth", "temperature"]]


def fetch_fishsoop(date_start, date_end, max_workers=12):
    """
    Query THREDDS catalog XMLs for the date range, download matching files in
    parallel with ThreadPoolExecutor, and return a single concatenated DataFrame.
    Much faster than reading S3 Parquet — each file is 35–268 KB.
    """
    months = pd.period_range(
        pd.Period(date_start, "M"), pd.Period(date_end, "M"), freq="M"
    )

    files_in_range = []
    for period in months:
        y, mo = period.year, period.month
        for fname in _catalog_filenames(y, mo):
            dt = _fname_to_dt(fname)
            if dt is not None and date_start <= dt <= date_end:
                files_in_range.append((y, mo, fname))

    if not files_in_range:
        return pd.DataFrame()

    print(f"Downloading {len(files_in_range)} FishSOOP files…")
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_download_profile, y, mo, fname): fname
            for y, mo, fname in files_in_range
        }
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined["depth"] <= depth]
    print(
        f"Loaded {len(combined)} records across "
        f"{combined['wmo_id'].nunique()} instruments"
    )
    return combined


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

if plot_rtofs:
    from ioos_model_comparisons.models import rtofs
    rds = rtofs().sel(depth=depths)
    rlons = rds.lon.data[0, :]
    rlats = rds.lat.data[:, 0]
    rx = rds.x.data
    ry = rds.y.data

if plot_espc:
    from ioos_model_comparisons.models import espc_ts
    espc_loaded = espc_ts(rename=True).sel(depth=depths)

if plot_cmems:
    from ioos_model_comparisons.models import CMEMS
    cobj = CMEMS()

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

date_end   = pd.Timestamp.utcnow().tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor("1d")

then = pd.Timestamp((pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y-%m-%d"))

date_fmt = "%Y-%m-%dT%H:%MZ"

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

print(f"Fetching FishSOOP data: {date_start} to {date_end}")
fishsoop = fetch_fishsoop(date_start, date_end)


# ---------------------------------------------------------------------------
# Per-region processing
# ---------------------------------------------------------------------------

def process_fishsoop_region(region):
    region_name   = region["name"]
    region_folder = region["folder"]
    extent        = region["extent"]

    region_df = filter_region(fishsoop, extent)

    if region_df.empty:
        print(f"No FishSOOP data in {region_name}")
        return

    print(f"Region: {region_name} — {region_df['wmo_id'].nunique()} instruments")

    region_save_dir = save_dir / region_folder
    os.makedirs(region_save_dir, exist_ok=True)

    symlink_dir = region_save_dir / "last_14_days"
    os.makedirs(symlink_dir, exist_ok=True)

    # Prune expired symlinks
    for f in sorted(glob.glob(os.path.join(symlink_dir, "*.png"))):
        m = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}", f)
        if m:
            dt_obj = datetime.strptime(m.group(), "%Y-%m-%dT%H%M%S")
            if (datetime.now() - dt_obj).days > 14:
                print(f"Removing expired symlink: {f}")
                os.remove(f)

    # Each tow_id is one profile file
    for tow_id, df in region_df.groupby("tow_id"):
        if len(df) < 3:
            continue

        # WIGOS ID allowlist — skip instruments not in the filter list
        if wigos_ids_filter and df["wigos_id"].iloc[0] not in wigos_ids_filter:
            continue

        # Debug filters — skip if set and doesn't match
        if debug_instrument and str(df["wmo_id"].iloc[0]) != str(debug_instrument):
            continue
        if debug_date and not df["time"].dt.date.eq(pd.Timestamp(debug_date).date()).any():
            continue

        wmo_id   = df["wmo_id"].iloc[0]
        wigos_id = df["wigos_id"].iloc[0]
        ctime    = df["time"].median()
        tstart   = df["time"].min()
        tend     = df["time"].max()

        tstr     = ctime.strftime(date_fmt)
        save_str = f"{wigos_id}-profile-{ctime.strftime('%Y-%m-%dT%H%M%SZ')}.png"
        tdir     = region_save_dir / ctime.strftime("%Y") / ctime.strftime("%m") / ctime.strftime("%d")
        os.makedirs(tdir, exist_ok=True)
        full_file = tdir / save_str

        if full_file.is_file() and not replot:
            print(f"{full_file} already exists. Skipping.")
            continue

        print(f"Processing FishSOOP {wigos_id} (#{wmo_id}) tow at {ctime}")

        lon = df["lon"].mean()
        lat = df["lat"].mean()
        mlon360 = lon180to360(lon)

        alon   = round(float(lon), 2)
        alat   = round(float(lat), 2)
        alabel = f"WIGOS {wigos_id} [#{wmo_id}]\n[{alon:.2f}, {alat:.2f}]"
        leg_str = f"FishSOOP WIGOS {wigos_id} [#{wmo_id}]\n"
        elapsed  = tend - tstart
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes  = remainder // 60
        leg_str += f"Start:   {tstart.strftime(date_fmt)}\n"
        leg_str += f"End:     {tend.strftime(date_fmt)}\n"
        leg_str += f"Elapsed: {hours}h {minutes:02d}m\n"

        # Cast average — split into downcast/upcast by time, bin each onto the
        # same grid (same depth_min, depth_max, stride), then average.
        daily_avg = None
        daily_avg_flag = False
        try:
            tow_time = df[["time", "depth", "temperature"]].sort_values("time").reset_index(drop=True)
            max_depth_idx = tow_time["depth"].idxmax()
            downcast = tow_time.iloc[:max_depth_idx + 1][["depth", "temperature"]]
            upcast   = tow_time.iloc[max_depth_idx:][["depth", "temperature"]]

            # Shared 1m grid — must be passed explicitly so both casts land on
            # identical depth values and groupby can average them correctly.
            shared_bins = np.arange(0, int(tow_time["depth"].max()) + 1, 1, dtype=float)

            casts = []
            for cast in [downcast, upcast]:
                if len(cast) >= 3:
                    binned = depth_interpolate(
                        cast,
                        depth_var="depth",
                        bins=shared_bins,
                    ).set_index("depth")[["temperature"]]
                    casts.append(binned)

            if len(casts) == 2:
                daily_avg = pd.concat(casts).groupby(level=0).mean().reset_index()
                daily_avg_flag = True
            elif len(casts) == 1:
                daily_avg = casts[0].reset_index()
                daily_avg_flag = True
        except Exception as e:
            print(f"Cast average failed for {wmo_id}: {e}")

        if plot_espc:
            try:
                gdsi = espc_loaded.sel(lon=mlon360, lat=lat, method="nearest")
                gdsi = gdsi.sel(time=ctime, method="nearest")
                gdsi["lon"] = lon360to180(gdsi["lon"])
                glon = gdsi.lon.data.round(2)
                glat = gdsi.lat.data.round(2)
                glabel = f"ESPC [{glon}, {glat}]"
                leg_str += f"ESPC : {pd.to_datetime(gdsi.time.data).strftime(date_fmt)}\n"
                espc_flag = True
            except KeyError as error:
                print(f"ESPC: False - {error}")
                espc_flag = False
        else:
            espc_flag = False

        if plot_rtofs:
            try:
                rlonI = np.interp(lon, rlons, rx)
                rlatI = np.interp(lat, rlats, ry)
                rdsp  = rds.sel(time=ctime, method="nearest")
                rdsi  = rdsp.sel(x=rlonI, y=rlatI, method="nearest")
                rdsi.load()
                rlon     = rdsi.lon.data.round(2)
                rlat_val = rdsi.lat.data.round(2)
                rlabel   = f"RTOFS [{rlon:.2f}, {rlat_val:.2f}]"
                leg_str += f"RTOFS: {pd.to_datetime(rdsi.time.data).strftime(date_fmt)}\n"
                rtofs_flag = True
            except KeyError as error:
                print(f"RTOFS: False - {error}")
                rtofs_flag = False
        else:
            rtofs_flag = False

        if plot_cmems:
            try:
                cdsi     = cobj.get_point(lon, lat, ctime)
                cdsi     = cdsi.sel(depth=depths)
                clon     = cdsi.lon.data.round(2)
                clat_val = cdsi.lat.data.round(2)
                clabel   = f"CMEMS [{clon:.2f}, {clat_val:.2f}]"
                leg_str += f"CMEMS: {pd.to_datetime(cdsi.time.data).strftime(date_fmt)}\n"
                cmems_flag = True
            except KeyError as error:
                print(f"CMEMS: False - {error}")
                cmems_flag = False
        else:
            cmems_flag = False

        # Plot
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        widths  = [1, 1.5]
        heights = [1, 2, 1]
        gs = fig.add_gridspec(3, 2, width_ratios=widths, height_ratios=heights)

        ax1 = fig.add_subplot(gs[:, 0])                                                           # Temperature profile
        ax2 = fig.add_subplot(gs[0, 1])                                                           # Info text
        ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.Mercator(central_longitude=float(lon)))   # Map centered at profile
        ax4 = fig.add_subplot(gs[2, 1])                                                           # Legend

        ax1.plot(df["temperature"], df["depth"], "b-o", label=alabel)

        if daily_avg_flag:
            ax1.plot(daily_avg["temperature"], daily_avg["depth"], "-o", color="cyan", linewidth=2, label=f"{wmo_id} avg")

        if espc_flag:
            ax1.plot(gdsi["temperature"], gdsi["depth"], "-o", color="green",   label=glabel)
        if rtofs_flag:
            ax1.plot(rdsi["temperature"], rdsi["depth"], "-o", color="red",     label=rlabel)
        if cmems_flag:
            ax1.plot(cdsi["temperature"], cdsi["depth"], "-o", color="magenta", label=clabel)

        obs_max_depth = df["depth"].max()
        depth_limit = smart_depth_axis(ax1, obs_max_depth)

        # X-limits: span all lines (obs + models) within the displayed depth range
        # (the smart y-axis limit, not raw obs max) so clipped model values are included.
        # NaN values (sparse model grids) are dropped.
        try:
            all_t = []
            for line in ax1.lines:
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                mask = (y >= 0) & (y <= depth_limit) & np.isfinite(x)
                if mask.any():
                    all_t.extend(x[mask].tolist())
            tmin = np.nanmin(all_t) - 0.5
            tmax = np.nanmax(all_t) + 0.5
        except ValueError:
            tmin, tmax = -2, 35

        ax1.set_xlim(tmin, tmax)
        ax1.grid(True, linestyle="--", linewidth=0.5)
        ax1.tick_params(axis="both", labelsize=13)
        ax1.set_xlabel("Temperature (˚C)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Depth (m)",        fontsize=14, fontweight="bold")

        text = ax2.text(0.125, 1.0, leg_str, ha="left", va="top", size=15, fontweight="bold")
        text.set_path_effects([path_effects.Normal()])
        ax2.set_axis_off()

        map_extent = local_map_extent(lon, lat)
        cplt.create(map_extent, ax=ax3, bathymetry=False)
        cplt.add_ticks(ax3, map_extent, fontsize=8)
        ax3.plot(lon, lat, "bo", transform=conf.projection["data"], zorder=101)

        h, l = ax1.get_legend_handles_labels()
        ax4.legend(h, l, ncol=1, loc="center", fontsize=12)
        ax4.set_axis_off()

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        plt.savefig(full_file, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        if ctime > then:
            symlink_path = symlink_dir / save_str
            if not symlink_path.exists():
                os.symlink(full_file, symlink_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if fishsoop.empty:
        print("No FishSOOP data found. Exiting.")
        return

    if parallel:
        import concurrent.futures
        workers = 6 if isinstance(parallel, bool) else parallel
        print(f"Running in parallel mode with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(process_fishsoop_region, FVON_REGIONS)
    else:
        for region in FVON_REGIONS:
            process_fishsoop_region(region)


if __name__ == "__main__":
    main()
