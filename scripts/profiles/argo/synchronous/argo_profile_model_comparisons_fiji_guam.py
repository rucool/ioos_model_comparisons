#!/usr/bin/env python
"""
Argo profile vs model comparisons for Pacific regions (Fiji, Guam).

ESPC + CMEMS only — no RTOFS (these regions are beyond RTOFS coverage).

Fiji notes
----------
- Region extent stored in 0-360 space: [139.75, 190.25, -30.25, -4.75]
- Argo API limited to [139.75, 180, -30.25, -4.75] (APIs use -180/180 lon)
- Map inset uses Mercator(central_longitude=180) to keep the domain contiguous
- CMEMS get_point() uses standard -180/180 lon directly — no antimeridian handling needed

Guam notes
----------
- Region extent: [129.75, 160.25, 4.75, 25.25] — all positive lons, no special handling
"""
import os
import glob
import re
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gsw import z_from_p

import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt
from cool_maps.plot import get_bathymetry
from ioos_model_comparisons.calc import lon180to360, lon360to180, density, ocean_heat_content
from ioos_model_comparisons.models import CMEMS, espc_ts
from ioos_model_comparisons.platforms import get_argo_floats_by_time
from ioos_model_comparisons.regions import region_config

# ── Config ──────────────────────────────────────────────────────────────────

save_dir = conf.path_plots / 'profiles' / 'argo'

parallel = True
depth = 400
days = 9
dpi = conf.dpi

plot_espc = True
plot_cmems = True

float_id = [2903887, 5906192]  # Set to None to plot all floats
sal_xlim =None   # Set to None to auto-scale salinity axis
temp_xlim = None      # Set to None to auto-scale temperature axis
density_xlim = None  # Set to None to auto-scale density axis

# Fiji platform extent — limited to 139.75–180°E since Argo ERDDAP uses -180/180
# and floats east of 180° are very sparse.
FIJI_PLATFORM_EXTENT = [139.75, 180, -30.25, -4.75]

# Per-region map projection for the location inset panel
REGION_MAP_PROJECTION = {
    'fiji': ccrs.Mercator(central_longitude=180),
    'guam': ccrs.Mercator(),
}

DATA_PROJECTION = ccrs.PlateCarree()

conf.regions = ['guam' ]

# ── Date range ──────────────────────────────────────────────────────────────

date_end = pd.Timestamp.now(tz='UTC').tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor('1d')

then = pd.Timestamp.today() - pd.Timedelta(days=14)
then = pd.Timestamp(then.strftime('%Y-%m-%d'))

date_fmt = "%Y-%m-%dT%H:%MZ"
vars = ['platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal']
depths = slice(0, depth)

# ── Global argo fetch extent (covers Fiji platform zone + Guam) ─────────────

guam_extent = region_config('guam')['extent']
global_extent = [
    min(FIJI_PLATFORM_EXTENT[0], guam_extent[0]),
    max(FIJI_PLATFORM_EXTENT[1], guam_extent[1]),
    min(FIJI_PLATFORM_EXTENT[2], guam_extent[2]),
    max(FIJI_PLATFORM_EXTENT[3], guam_extent[3]),
]

# ── Load models once ────────────────────────────────────────────────────────

espc_loaded = espc_ts(rename=True).sel(depth=depths) if plot_espc else None
cobj = CMEMS() if plot_cmems else None

# ── Argo floats ─────────────────────────────────────────────────────────────

floats = get_argo_floats_by_time(
    global_extent,
    date_start,
    date_end,
    variables=vars,
)
floats['depth'] = -z_from_p(floats['pres (decibar)'], floats['lat'])
floats = floats[floats['depth'] <= depth]

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']


# ── Helpers ─────────────────────────────────────────────────────────────────

def line_limits(fax, delta=1):
    mins = [np.nanmin(line.get_xdata()) for line in fax.lines]
    maxs = [np.nanmax(line.get_xdata()) for line in fax.lines]
    return min(mins) - delta, max(maxs) + delta


# ── Per-region processing ────────────────────────────────────────────────────

def process_argo(region_key):
    region = region_config(region_key)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')

    temp_save_dir = save_dir / region['folder']
    os.makedirs(temp_save_dir, exist_ok=True)

    symlink_dir = temp_save_dir / 'last_14_days'
    os.makedirs(symlink_dir, exist_ok=True)

    # Prune stale symlinks
    for f in sorted(glob.glob(os.path.join(symlink_dir, '*.png'))):
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}', f)
        if match:
            date_time_obj = datetime.strptime(match.group(), '%Y-%m-%dT%H%M%S')
            if (datetime.now() - date_time_obj).days > 14:
                print(f"The file {f} is older than 14 days.")
                os.remove(f)

    try:
        bathy = get_bathymetry(extent)
        bathy_flag = True
    except Exception:
        bathy_flag = False

    # Fiji: API only covers up to 180°E, so filter on the platform extent
    platform_extent = FIJI_PLATFORM_EXTENT if region_key == 'fiji' else extent
    map_proj = REGION_MAP_PROJECTION.get(region_key, ccrs.Mercator())

    if floats.empty:
        print(f"No Argo floats found in {region['name']}")
        return

    argo_region = floats[
        (platform_extent[0] <= floats['lon']) & (floats['lon'] <= platform_extent[1]) &
        (platform_extent[2] <= floats['lat']) & (floats['lat'] <= platform_extent[3])
    ]

    if argo_region.empty:
        print(f"No Argo floats in {region['name']} after region filter")
        return

    if float_id is not None:
        ids = [float_id] if not isinstance(float_id, list) else float_id
        argo_region = argo_region[argo_region.index.get_level_values('argo').isin(ids)]
        if argo_region.empty:
            print(f"Float(s) {ids} not found in {region['name']}")
            return

    for gname, df in argo_region.reset_index().groupby(['argo', 'time']):
        wmo = gname[0]
        ctime = gname[1]
        print(f"Checking ARGO {wmo} for new profiles")

        tstr = ctime.strftime(date_fmt)
        save_str = f'{wmo}-profile-{ctime.strftime("%Y-%m-%dT%H%MZ")}.png'
        tdir = temp_save_dir / ctime.strftime("%Y") / ctime.strftime("%m") / ctime.strftime("%d")
        os.makedirs(tdir, exist_ok=True)
        full_file = tdir / save_str
        if full_file.is_file():
            print(f"Skipping ARGO {wmo} profile at {ctime}: {full_file} already exists.")
            if ctime > then:
                try:
                    os.symlink(full_file, symlink_dir / save_str)
                except FileExistsError:
                    pass
            continue

        print(f"Processing ARGO {wmo} profile that occurred at {ctime}")

        df = df.assign(depth=-z_from_p(df['pres (decibar)'].values, df['lat'].values))
        depth_mask = df['depth'] <= depth
        df = df[depth_mask].copy()

        if df.empty:
            continue

        df = df.assign(
            density=density(
                df['temp (degree_Celsius)'].values,
                -df['depth'].values,
                df['psal (PSU)'].values,
                df['lat'].values,
                df['lon'].values,
            )
        )

        ohc_float = ocean_heat_content(df['depth'], df['temp (degree_Celsius)'], df['density'])

        lon, lat = df['lon'].unique()[-1], df['lat'].unique()[-1]
        mlon360 = lon180to360(lon)

        alon = round(lon, 2)
        alat = round(lat, 2)
        alabel = f'{wmo} [{alon}, {alat}]'
        leg_str = f'Argo #{wmo}\n'
        leg_str += f'ARGO: {tstr}\n'

        # ── ESPC ──────────────────────────────────────────────────────
        espc_flag = False
        gdsi = None
        if plot_espc and espc_loaded is not None:
            try:
                gdsi = espc_loaded.sel(lon=mlon360, lat=lat, method='nearest')
                gdsi = gdsi.sel(time=ctime, method='nearest')
                gdsi['lon'] = lon360to180(gdsi['lon'])
                gdsi['density'] = density(gdsi.temperature, -gdsi.depth, gdsi.salinity, gdsi.lat, gdsi.lon)
                ohc_espc = ocean_heat_content(gdsi['depth'].values, gdsi['temperature'].values, gdsi['density'].values)
                glon = gdsi.lon.data.round(2)
                glat = gdsi.lat.data.round(2)
                glabel = f'ESPC [{glon}, {glat}]'
                leg_str += f'ESPC : {pd.to_datetime(gdsi.time.data).strftime(date_fmt)}\n'
                espc_flag = True
            except KeyError as error:
                print(f"ESPC: False - {error}")

        # ── CMEMS ─────────────────────────────────────────────────────
        cmems_flag = False
        cdsi = None
        if plot_cmems and cobj is not None:
            try:
                cdsi = cobj.get_point(lon, lat, ctime)
                cdsi = cdsi.sel(depth=depths)
                cdsi['density'] = density(cdsi.temperature, -cdsi.depth, cdsi.salinity, cdsi.lat, cdsi.lon)
                ohc_cmems = ocean_heat_content(cdsi['depth'].values, cdsi['temperature'].values, cdsi['density'].values)
                clon = cdsi.lon.data.round(2)
                clat = cdsi.lat.data.round(2)
                clabel = f'CMEMS [{clon:.2f}, {clat:.2f}]'
                leg_str += f'CMEMS: {pd.to_datetime(cdsi.time.data).strftime(date_fmt)}\n'
                cmems_flag = True
            except KeyError as error:
                print(f"CMEMS: False - {error}")

        # ── Figure ────────────────────────────────────────────────────────
        fig = plt.figure(constrained_layout=True, figsize=(16, 6))
        widths = [1, 1, 1, 1.5]
        heights = [1, 2, 1]
        gs = fig.add_gridspec(3, 4, width_ratios=widths, height_ratios=heights)

        ax1 = fig.add_subplot(gs[:, 0])                        # Temperature
        ax2 = fig.add_subplot(gs[:, 1], sharey=ax1)            # Salinity
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3 = fig.add_subplot(gs[:, 2], sharey=ax1)            # Density
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax4 = fig.add_subplot(gs[0, -1])                       # Info text
        ax5 = fig.add_subplot(gs[1, -1], projection=map_proj)  # Map
        ax6 = fig.add_subplot(gs[2, -1])                       # Legend

        # Argo
        ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
        ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
        ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)

        # ESPC
        if espc_flag:
            ax1.plot(gdsi['temperature'], gdsi['depth'], linestyle='-', marker='o', color='green', label=glabel)
            ax2.plot(gdsi['salinity'], gdsi['depth'], linestyle='-', marker='o', color='green', label=glabel)
            ax3.plot(gdsi['density'], gdsi['depth'], linestyle='-', marker='o', color='green', label=glabel)

        # CMEMS
        if cmems_flag:
            ax1.plot(cdsi['temperature'], cdsi['depth'], linestyle='-', marker='o', color='magenta', label=clabel)
            ax2.plot(cdsi['salinity'], cdsi['depth'], linestyle='-', marker='o', color='magenta', label=clabel)
            ax3.plot(cdsi['density'], cdsi['depth'], linestyle='-', marker='o', color='magenta', label=clabel)

        try:
            tmin, tmax = line_limits(ax1, delta=.5)
            smin, smax = line_limits(ax2, delta=.25)
            dmin, dmax = line_limits(ax3, delta=.5)
        except ValueError:
            print('Some kind of error')
            pass

        ax1.set_ylim([depth, 0])
        ax1.set_xlim(temp_xlim if temp_xlim is not None else (tmin, tmax))
        ax1.grid(True, linestyle='--', linewidth=.5)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

        ax2.set_ylim([depth, 0])
        ax2.set_xlim(sal_xlim if sal_xlim is not None else (smin, smax))
        ax2.grid(True, linestyle='--', linewidth=.5)
        ax2.tick_params(axis='both', labelsize=13)
        ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')

        ax3.set_ylim([depth, 0])
        ax3.set_xlim(density_xlim if density_xlim is not None else (dmin, dmax))
        ax3.grid(True, linestyle='--', linewidth=.5)
        ax3.tick_params(axis='both', labelsize=13)
        ax3.set_xlabel('Density', fontsize=14, fontweight='bold')

        text = ax4.text(0.125, 1.0, leg_str, ha='left', va='top', size=13, fontweight='bold')
        text.set_path_effects([path_effects.Normal()])
        ax4.set_axis_off()

        cplt.create(extent, ax=ax5, bathymetry=False)
        cplt.add_ticks(ax5, extent, fontsize=8)
        ax5.plot(lon, lat, 'bo', transform=DATA_PROJECTION, zorder=101)

        h, l = ax2.get_legend_handles_labels()
        ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
        ax6.set_axis_off()

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        ohc_string = 'Ocean Heat Content (kJ/cm^2) - '
        try:
            v = np.nanmean(ohc_float)
            ohc_string += f"Argo: {v:.4f},  " if not np.isnan(v) else "Argo: N/A,  "
        except Exception:
            pass
        try:
            if np.isnan(ohc_espc):
                ohc_string += 'ESPC: N/A,  '
            else:
                ohc_string += f"ESPC: {ohc_espc:.4f},  "
        except Exception:
            pass
        try:
            if np.isnan(ohc_cmems):
                ohc_string += 'CMEMS: N/A,  '
            else:
                ohc_string += f"CMEMS: {ohc_cmems:.4f},  "
        except Exception:
            pass

        plt.figtext(0.4, 0.001, ohc_string, ha="center", fontsize=10, fontstyle='italic')

        plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        if ctime > then:
            try:
                os.symlink(full_file, symlink_dir / save_str)
            except FileExistsError:
                pass


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if parallel:
        import concurrent.futures
        workers = parallel if isinstance(parallel, int) else len(conf.regions)
        print(f"Running in parallel mode with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(process_argo, conf.regions)
    else:
        for region in conf.regions:
            process_argo(region)


if __name__ == "__main__":
    main()
