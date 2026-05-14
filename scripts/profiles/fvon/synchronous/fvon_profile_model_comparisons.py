#!/usr/bin/env python
import os
import re
import glob
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ioos_model_comparisons.calc import lon180to360, lon360to180
import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt

save_dir = conf.path_plots / 'profiles' / 'fvon'


# Configs
parallel = False
replot = False   # set True to overwrite existing plots
depth = 400

# Which models should we plot?
plot_rtofs = True
plot_espc = True
plot_cmems = True

days = 7
dpi = conf.dpi

depths = slice(0, depth)

# Broad ocean basin regions used to organize output folders.
# Extent format: [lonmin, lonmax, latmin, latmax] in -180/180 space.
# For regions that cross the antimeridian (Pacific), lonmin > lonmax.
FVON_REGIONS = [
    {"name": "Bahamas", "folder": "bahamas", "extent": [-82.5, -63,    16,    31]},
    # {"name": "Fiji",    "folder": "fiji",    "extent": [139.75, -169.75, -30.25, -4.75]},  # crosses antimeridian
]


def filter_region(df, extent):
    """Filter a DataFrame to rows within the given extent, handling antimeridian-crossing regions."""
    lonmin, lonmax, latmin, latmax = extent
    lat_mask = (df['lat'] >= latmin) & (df['lat'] <= latmax)
    if lonmin <= lonmax:
        lon_mask = (df['lon'] >= lonmin) & (df['lon'] <= lonmax)
    else:  # region crosses the antimeridian
        lon_mask = (df['lon'] >= lonmin) | (df['lon'] <= lonmax)
    return df[lat_mask & lon_mask]


# Load models
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

# Date range: last 14 days ending now
date_end = pd.Timestamp.utcnow().tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor('1d')

# For symlink cleanup
then = pd.Timestamp.today() - pd.Timedelta(days=14)
then = pd.Timestamp(then.strftime('%Y-%m-%d'))

# Formatter for date
date_fmt = "%Y-%m-%dT%H:%MZ"

# Fetch FVON data from PMEL ERDDAP
erddap_url = (
    "https://data.pmel.noaa.gov/generic/erddap/tabledap/fvon_nrt_data.csv"
    "?wmo_platform_code,time,latitude,longitude,depth,temperature,segment_type"
    f"&time>={date_start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    f"&time<={date_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    '&orderBy("wmo_platform_code,time")'
)

print(f"Fetching FVON data: {date_start} to {date_end}")
try:
    fvon = pd.read_csv(erddap_url, skiprows=[1], parse_dates=['time'])
    fvon = fvon.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    fvon['time'] = pd.to_datetime(fvon['time']).dt.tz_convert(None)
    fvon = fvon[fvon['depth'] <= depth]
    fvon = fvon.dropna(subset=['temperature'])
    # Keep downcast only — avoids zigzag from interleaved up/downcast data
    fvon = fvon[fvon['segment_type'] == 'Profiling Down']
    print(f"Loaded {len(fvon)} FVON records across {fvon['wmo_platform_code'].nunique()} platforms")
except Exception as e:
    print(f"Error fetching FVON data: {e}")
    fvon = pd.DataFrame()


# Paired (depth_limit, tick_stride) configs — chosen so each limit gets 5-7 ticks.
_DEPTH_CONFIGS = [
    (10,    5),
    (15,    5),
    (20,    5),
    (25,    5),
    (30,    5),
    (40,   10),
    (50,   10),
    (75,   15),
    (100,  20),
    (150,  25),
    (200,  50),
    (250,  50),
    (300,  50),
    (400, 100),
    (500, 100),
    (750, 150),
    (1000, 200),
]


def smart_depth_axis(ax, obs_max_depth):
    """Set y-axis depth limit and ticks based on the observation's actual max depth.

    Finds the smallest 'nice' depth ceiling >= obs_max_depth, then applies a
    matching tick stride so the axis always has 5-7 evenly-spaced labels.
    Model lines that extend deeper are clipped by the ylim automatically.
    """
    lim, stride = next(
        ((lim, stride) for lim, stride in _DEPTH_CONFIGS if lim >= obs_max_depth),
        _DEPTH_CONFIGS[-1],
    )
    ticks = np.arange(0, lim + stride, stride)
    ax.set_ylim(lim, 0)
    ax.set_yticks(ticks)
    return lim


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
    """Get the min and max x-data across all lines on a Matplotlib axis."""
    mins = [np.nanmin(line.get_xdata()) for line in fax.lines]
    maxs = [np.nanmax(line.get_xdata()) for line in fax.lines]
    return min(mins) - delta, max(maxs) + delta


def _ensure_symlink(full_file, symlink_dir, save_str, ctime, then):
    """Create a symlink in last_14_days/ pointing to the dated subfolder file."""
    if ctime <= then:
        return
    symlink_path = symlink_dir / save_str
    if not symlink_path.is_symlink():
        rel_target = os.path.relpath(full_file.resolve(), symlink_dir.resolve())
        os.symlink(rel_target, symlink_path)


def process_fvon_region(region):
    region_name = region['name']
    region_folder = region['folder']
    extent = region['extent']

    region_df = filter_region(fvon, extent)

    if region_df.empty:
        print(f"No FVON data found in {region_name}")
        return

    print(f"Region: {region_name} — {len(region_df['wmo_platform_code'].unique())} platforms")

    region_save_dir = save_dir / region_folder
    os.makedirs(region_save_dir, exist_ok=True)

    # One symlink directory per region — all platforms visible in one place
    symlink_dir = region_save_dir / 'last_14_days'
    os.makedirs(symlink_dir, exist_ok=True)

    # Remove symlinks older than 14 days
    for f in sorted(glob.glob(os.path.join(symlink_dir, '*.png'))):
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}', f)
        if match:
            date_time_obj = datetime.strptime(match.group(), '%Y-%m-%dT%H%M%S')
            if (datetime.now() - date_time_obj).days > 14:
                print(f"Removing expired symlink: {f}")
                os.remove(f)

    # Each unique (platform, time) represents one profile cast
    for (wmo_id, ctime), df in region_df.groupby(['wmo_platform_code', 'time']):
        if df.empty or len(df) < 3:
            continue

        tstr = ctime.strftime(date_fmt)
        save_str = f'{wmo_id}-profile-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'
        tdir = region_save_dir / ctime.strftime("%Y") / ctime.strftime("%m") / ctime.strftime("%d")
        os.makedirs(tdir, exist_ok=True)
        full_file = tdir / save_str

        if full_file.is_file() and not replot:
            print(f"{full_file} already exists. Skipping.")
            _ensure_symlink(full_file, symlink_dir, save_str, ctime, then)
            continue

        print(f"Processing FVON {wmo_id} profile at {ctime}")

        lon = df['lon'].mean()
        lat = df['lat'].mean()
        mlon360 = lon180to360(lon)

        alon = round(float(lon), 2)
        alat = round(float(lat), 2)
        alabel = f'{wmo_id} [{alon:.2f}, {alat:.2f}]'
        leg_str = f'FVON #{wmo_id}\n'
        leg_str += f'FVON: {tstr}\n'

        if plot_espc:
            try:
                gdsi = espc_loaded.sel(lon=mlon360, lat=lat, method='nearest')
                gdsi = gdsi.sel(time=ctime, method='nearest')
                gdsi['lon'] = lon360to180(gdsi['lon'])
                glon = gdsi.lon.data.round(2)
                glat = gdsi.lat.data.round(2)
                glabel = f'ESPC [{glon}, {glat}]'
                leg_str += f'ESPC : {pd.to_datetime(gdsi.time.data).strftime(date_fmt)}\n'
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
                rdsp = rds.sel(time=ctime, method='nearest')
                rdsi = rdsp.sel(x=rlonI, y=rlatI, method='nearest')
                rdsi.load()
                rlon = rdsi.lon.data.round(2)
                rlat_val = rdsi.lat.data.round(2)
                rlabel = f'RTOFS [{rlon:.2f}, {rlat_val:.2f}]'
                leg_str += f'RTOFS: {pd.to_datetime(rdsi.time.data).strftime(date_fmt)}\n'
                rtofs_flag = True
            except KeyError as error:
                print(f"RTOFS: False - {error}")
                rtofs_flag = False
        else:
            rtofs_flag = False

        if plot_cmems:
            try:
                cdsi = cobj.get_point(lon, lat, ctime)
                cdsi = cdsi.sel(depth=depths)
                clon = cdsi.lon.data.round(2)
                clat_val = cdsi.lat.data.round(2)
                clabel = f"CMEMS [{clon:.2f}, {clat_val:.2f}]"
                leg_str += f'CMEMS: {pd.to_datetime(cdsi.time.data).strftime(date_fmt)}\n'
                cmems_flag = True
            except KeyError as error:
                print(f"CMEMS: False - {error}")
                cmems_flag = False
        else:
            cmems_flag = False

        # Plot: temperature profile + info + map + legend
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        widths = [1, 1.5]
        heights = [1, 2, 1]

        gs = fig.add_gridspec(3, 2, width_ratios=widths, height_ratios=heights)

        ax1 = fig.add_subplot(gs[:, 0])                                                           # Temperature profile
        ax2 = fig.add_subplot(gs[0, 1])                                                           # Info text
        ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.Mercator(central_longitude=float(lon)))   # Map centered at profile
        ax4 = fig.add_subplot(gs[2, 1])                                                           # Legend

        # FVON observation
        ax1.plot(df['temperature'], df['depth'], 'b-o', label=alabel)

        # ESPC
        if espc_flag:
            ax1.plot(gdsi['temperature'], gdsi['depth'], '-o', color='green', label=glabel)

        # RTOFS
        if rtofs_flag:
            ax1.plot(rdsi['temperature'], rdsi['depth'], '-o', color='red', label=rlabel)

        # CMEMS
        if cmems_flag:
            ax1.plot(cdsi['temperature'], cdsi['depth'], '-o', color='magenta', label=clabel)

        obs_max_depth = df['depth'].max()
        smart_depth_axis(ax1, obs_max_depth)

        # X-limits: only consider data within the observation's depth range so
        # model profiles that extend deeper don't widen the temperature axis.
        try:
            all_t = []
            for line in ax1.lines:
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                mask = (y >= 0) & (y <= obs_max_depth)
                if mask.any():
                    all_t.extend(x[mask].tolist())
            tmin = min(all_t) - 0.5
            tmax = max(all_t) + 0.5
        except ValueError:
            tmin, tmax = -2, 35

        ax1.set_xlim(tmin, tmax)
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

        text = ax2.text(0.125, 1.0, leg_str, ha='left', va='top', size=15, fontweight='bold')
        text.set_path_effects([path_effects.Normal()])
        ax2.set_axis_off()

        # Small local map centered on profile location
        map_extent = local_map_extent(lon, lat)
        cplt.create(map_extent, ax=ax3, bathymetry=False)
        cplt.add_ticks(ax3, map_extent, fontsize=8)
        ax3.plot(lon, lat, 'bo', transform=conf.projection['data'], zorder=101)

        h, l = ax1.get_legend_handles_labels()
        ax4.legend(h, l, ncol=1, loc='center', fontsize=12)
        ax4.set_axis_off()

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        _ensure_symlink(full_file, symlink_dir, save_str, ctime, then)


def main():
    if fvon.empty:
        print("No FVON data found. Exiting.")
        return

    if parallel:
        import concurrent.futures
        workers = 6 if isinstance(parallel, bool) else parallel
        print(f"Running in parallel mode with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(process_fvon_region, FVON_REGIONS)
    else:
        for region in FVON_REGIONS:
            process_fvon_region(region)


if __name__ == "__main__":
    main()
