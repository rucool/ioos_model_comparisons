#!/usr/bin/env python
"""
Argo profile vs model comparisons for Pacific regions (Fiji, Guam).

ESPC + CMEMS + RTOFS binary — where pre-processed RTOFS regional NetCDFs exist.

RTOFS files are read from a directory tree created by grab_rtofs_archv_aws.py:
    rtofs_global/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_{region}.nc

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
import datetime as dt
import json
import os
import glob
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from gsw import z_from_p

import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt
from cool_maps.plot import get_bathymetry
from ioos_model_comparisons.calc import lon180to360, lon360to180, density, ocean_heat_content
from ioos_model_comparisons.models import CMEMS, espc_ts, espc_ts_archive
from ioos_model_comparisons.platforms import (
    ARGO_GOOD_QC_FLAGS,
    get_argo_floats_by_time,
)
from ioos_model_comparisons.regions import region_config

# ── Config ──────────────────────────────────────────────────────────────────

save_dir = conf.path_plots / 'profiles' / 'argo'

# RTOFS binary pre-processed NetCDF directory (created by grab_rtofs_archv_aws.py)
RTOFS_DATA_DIR = Path("/home/hurricaneadm/data/rtofs_archv")

# Max age difference between an Argo profile and the nearest RTOFS file (hours)
RTOFS_MAX_HOURS = 12

parallel = True
depth = 400
days = 30
dpi = conf.dpi

plot_espc = True
plot_cmems = True
plot_rtofs = True

float_id = None   # Set to a WMO number to plot only that float
sal_xlim = None   # Set to None to auto-scale salinity axis
temp_xlim = None  # Set to None to auto-scale temperature axis
density_xlim = None  # Set to None to auto-scale density axis

# Fiji platform extent — limited to 139.75–180°E since Argo ERDDAP uses -180/180
FIJI_PLATFORM_EXTENT = [139.75, 180, -30.25, -4.75]

# Per-region map projection for the location inset panel
REGION_MAP_PROJECTION = {
    'fiji': ccrs.Mercator(central_longitude=180),
}

DATA_PROJECTION = ccrs.PlateCarree()

conf.regions = ['guam', 'fiji']

# ── Date range ──────────────────────────────────────────────────────────────

date_end = pd.Timestamp.now(tz='UTC').tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor('1d')

then = pd.Timestamp.today() - pd.Timedelta(days=14)
then = pd.Timestamp(then.strftime('%Y-%m-%d'))

date_fmt = "%Y-%m-%dT%H:%MZ"
vars = ['platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal']
depths = slice(0, depth)

ARGO_TEMP_POINT_QC_COLUMNS = ('temp_qc',)
ARGO_SALINITY_POINT_QC_COLUMNS = ('psal_qc',)
ARGO_DENSITY_POINT_QC_COLUMNS = ('pres_qc', 'temp_qc', 'psal_qc')
QC_FLAGGED_HANDLE = Line2D(
    [0], [0],
    linestyle='None',
    marker='o',
    markerfacecolor='none',
    markeredgecolor='red',
    markeredgewidth=1.5,
    markersize=7,
    label='Argo QC flagged',
)

# ── Global argo fetch extent (covers Fiji platform zone + Guam) ─────────────

guam_extent = region_config('guam')['extent']
global_extent = [
    min(FIJI_PLATFORM_EXTENT[0], guam_extent[0]),
    max(FIJI_PLATFORM_EXTENT[1], guam_extent[1]),
    min(FIJI_PLATFORM_EXTENT[2], guam_extent[2]),
    max(FIJI_PLATFORM_EXTENT[3], guam_extent[3]),
]

# ── Load models once ────────────────────────────────────────────────────────

espc_loaded = None
if plot_espc:
    # Use FMRC best-forecast if date_start is within the last 10 days,
    # otherwise fall back to the year-based archive.
    espc_cutoff = pd.Timestamp.now() - pd.Timedelta(days=10)
    if date_start >= espc_cutoff:
        print("ESPC: using FMRC best-forecast (recent data)")
        espc_loaded = espc_ts(rename=True).sel(depth=depths)
    else:
        print(f"ESPC: using archive for year {date_start.year}")
        espc_loaded = espc_ts_archive(rename=True, year=date_start.year).sel(depth=depths)
cobj = CMEMS() if plot_cmems else None

# ── Argo floats ─────────────────────────────────────────────────────────────

floats = get_argo_floats_by_time(
    global_extent,
    date_start,
    date_end,
    variables=vars,
    include_qc=True,
)
floats['depth'] = -z_from_p(floats['pres (decibar)'], floats['lat'])
floats = floats[floats['depth'] <= depth]

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']


# ── RTOFS binary helpers ─────────────────────────────────────────────────────

def _parse_rtofs_time(nc_path):
    """Parse valid time from rtofs_glo_YYYYMMDDTHH_{region}.nc filename."""
    stem = Path(nc_path).stem
    time_part = stem.split("_")[2]
    return pd.Timestamp(dt.datetime.strptime(time_part, "%Y%m%dT%H"))


def find_rtofs_file(region_name, target_time, data_dir=RTOFS_DATA_DIR, max_hours=RTOFS_MAX_HOURS):
    """Return the RTOFS NetCDF path whose valid time is closest to target_time.

    Returns None if no file is found or if the closest file is further than
    max_hours from target_time.
    """
    candidates = sorted(data_dir.glob(f"*/*/*/rtofs_glo_*_{region_name}.nc"))
    if not candidates:
        return None

    target = pd.Timestamp(target_time)
    best = min(candidates, key=lambda p: abs((_parse_rtofs_time(p) - target).total_seconds()))
    diff_h = abs((_parse_rtofs_time(best) - target).total_seconds()) / 3600
    if diff_h > max_hours:
        return None
    return best


def load_rtofs_point(nc_path, lon, lat, max_depth=400):
    """Extract nearest-point profile from a pre-processed RTOFS binary NetCDF.

    Returns an xarray.Dataset with dims (depth,) and variables
    temperature, salinity, renamed from temp/salin/z.
    Returns None on failure.
    """
    try:
        ds = xr.open_dataset(nc_path)

        # The file may use 0–360 lons (especially for Fiji / Pacific regions).
        # Convert the lookup lon to match.
        file_lon_min = float(ds.lon.min())
        lookup_lon = lon180to360(lon) if file_lon_min > 90 and lon < 0 else lon

        point = ds.sel(lat=lat, lon=lookup_lon, method='nearest')
        point = point.rename({"temp": "temperature", "salin": "salinity", "z": "depth"})
        point = point.sel(depth=slice(0, max_depth))

        # Drop u/v — not needed for profiles
        drop_vars = [v for v in ("u-vel.", "v-vel.") if v in point]
        if drop_vars:
            point = point.drop_vars(drop_vars)

        point["temperature"].attrs["units"] = "degC"
        point["salinity"].attrs["units"] = "PSU"
        point.load()
        return point
    except Exception as e:
        print(f"RTOFS binary: load failed ({e})")
        return None


# ── Common helpers ───────────────────────────────────────────────────────────

def normalize_qc_flag(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def qc_flag_mask(df, columns):
    columns = [col for col in columns if col in df.columns]
    if not columns:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in columns:
        flags = df[col].map(normalize_qc_flag)
        mask |= flags.notna() & ~flags.isin(ARGO_GOOD_QC_FLAGS)
    return mask.fillna(False)


def profile_qc_value(df, column):
    if column not in df.columns:
        return 'NA'
    values = [v for v in df[column].map(normalize_qc_flag).dropna().unique() if v]
    if not values:
        return 'NA'
    if len(values) == 1:
        return values[0]
    return ','.join(values[:3]) + ('+' if len(values) > 3 else '')


def add_flagged_points(ax, x, y, mask):
    if bool(mask.any()):
        ax.scatter(
            x[mask], y[mask],
            s=42,
            facecolors='none',
            edgecolors='red',
            linewidths=1.5,
            zorder=20,
        )


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
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{4}Z', f)
        if match:
            date_time_obj = datetime.strptime(match.group(), '%Y-%m-%dT%H%MZ')
            if (datetime.now() - date_time_obj).days > 14:
                print(f"The file {f} is older than 14 days.")
                os.remove(f)

    locations_file = symlink_dir / 'locations.json'
    if locations_file.exists():
        try:
            with open(locations_file, 'r') as f:
                locations = json.load(f)
            locations = {k: v for k, v in locations.items() if (symlink_dir / k).exists()}
            with open(locations_file, 'w') as f:
                json.dump(locations, f)
        except Exception as e:
            print(f"Error cleaning up locations.json: {e}")

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

    # Pass 1: gather profile + model data for every new surfacing, grouped by
    # float. Axis limits are then computed per-float so that multiple
    # surfacings of the same buoy share the same temperature/salinity/density
    # axes and can be visually compared.
    records_by_wmo = defaultdict(list)

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

        temp_flagged = qc_flag_mask(df, ARGO_TEMP_POINT_QC_COLUMNS)
        salinity_flagged = qc_flag_mask(df, ARGO_SALINITY_POINT_QC_COLUMNS)
        density_flagged = qc_flag_mask(df, ARGO_DENSITY_POINT_QC_COLUMNS)
        any_flagged = bool(temp_flagged.any() or salinity_flagged.any() or density_flagged.any())

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
        glabel = None
        ohc_espc = np.nan
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
        clabel = None
        ohc_cmems = np.nan
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

        # ── RTOFS binary ───────────────────────────────────────────────
        rtofs_flag = False
        rdsi = None
        rlabel = None
        ohc_rtofs = np.nan
        if plot_rtofs:
            rtofs_nc = find_rtofs_file(region_key, ctime)
            if rtofs_nc is not None:
                rdsi = load_rtofs_point(rtofs_nc, lon, lat, max_depth=depth)
                if rdsi is not None:
                    try:
                        rdsi['density'] = density(
                            rdsi['temperature'],
                            -rdsi['depth'],
                            rdsi['salinity'],
                            float(rdsi.lat),
                            float(rdsi.lon),
                        )
                        ohc_rtofs = ocean_heat_content(
                            rdsi['depth'].values,
                            rdsi['temperature'].values,
                            rdsi['density'].values,
                        )
                        rlon = float(rdsi.lon.values)
                        rlat = float(rdsi.lat.values)
                        # Convert back to -180/180 for display
                        if rlon > 180:
                            rlon -= 360
                        rlabel = f'RTOFS [{rlon:.2f}, {rlat:.2f}]'
                        rtime = _parse_rtofs_time(rtofs_nc)
                        leg_str += f'RTOFS: {rtime.strftime(date_fmt)}\n'
                        rtofs_flag = True
                    except Exception as e:
                        print(f"RTOFS binary: compute failed ({e})")
            else:
                print(f"RTOFS binary: no file found within {RTOFS_MAX_HOURS}h of {ctime} for region '{region_key}'")

        records_by_wmo[wmo].append(dict(
            wmo=wmo, ctime=ctime, tstr=tstr, save_str=save_str, full_file=full_file,
            df=df, temp_flagged=temp_flagged, salinity_flagged=salinity_flagged,
            density_flagged=density_flagged, any_flagged=any_flagged, ohc_float=ohc_float,
            lon=lon, lat=lat, alabel=alabel, leg_str=leg_str,
            espc_flag=espc_flag, gdsi=gdsi, glabel=glabel, ohc_espc=ohc_espc,
            cmems_flag=cmems_flag, cdsi=cdsi, clabel=clabel, ohc_cmems=ohc_cmems,
            rtofs_flag=rtofs_flag, rdsi=rdsi, rlabel=rlabel, ohc_rtofs=ohc_rtofs,
        ))

    # ── Shared axis limits per float ─────────────────────────────────────
    def _bounds(arrays, delta):
        values = np.concatenate([np.asarray(a).ravel() for a in arrays])
        values = values[~np.isnan(values)]
        if values.size == 0:
            return None
        return float(values.min()) - delta, float(values.max()) + delta

    def _wmo_limits(records):
        temp_arrays, sal_arrays, dens_arrays = [], [], []
        for rec in records:
            temp_arrays.append(rec['df']['temp (degree_Celsius)'].values)
            sal_arrays.append(rec['df']['psal (PSU)'].values)
            dens_arrays.append(rec['df']['density'].values)
            for key, flag in (('gdsi', 'espc_flag'), ('cdsi', 'cmems_flag'), ('rdsi', 'rtofs_flag')):
                if rec[flag]:
                    temp_arrays.append(rec[key]['temperature'].values)
                    sal_arrays.append(rec[key]['salinity'].values)
                    dens_arrays.append(rec[key]['density'].values)
        return _bounds(temp_arrays, .5), _bounds(sal_arrays, .25), _bounds(dens_arrays, .5)

    # Pass 2: render one figure per surfacing, using per-float shared limits
    # (unless the module-level *_xlim overrides are set).
    for wmo, records in records_by_wmo.items():
        wmo_tlim, wmo_slim, wmo_dlim = _wmo_limits(records)

        for rec in records:
            ctime = rec['ctime']
            tstr = rec['tstr']
            save_str = rec['save_str']
            full_file = rec['full_file']
            df = rec['df']
            temp_flagged = rec['temp_flagged']
            salinity_flagged = rec['salinity_flagged']
            density_flagged = rec['density_flagged']
            any_flagged = rec['any_flagged']
            ohc_float = rec['ohc_float']
            lon, lat = rec['lon'], rec['lat']
            alabel = rec['alabel']
            leg_str = rec['leg_str']
            espc_flag, gdsi, glabel, ohc_espc = rec['espc_flag'], rec['gdsi'], rec['glabel'], rec['ohc_espc']
            cmems_flag, cdsi, clabel, ohc_cmems = rec['cmems_flag'], rec['cdsi'], rec['clabel'], rec['ohc_cmems']
            rtofs_flag, rdsi, rlabel, ohc_rtofs = rec['rtofs_flag'], rec['rdsi'], rec['rlabel'], rec['ohc_rtofs']

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
            add_flagged_points(ax1, df['temp (degree_Celsius)'], df['depth'], temp_flagged)
            add_flagged_points(ax2, df['psal (PSU)'], df['depth'], salinity_flagged)
            add_flagged_points(ax3, df['density'], df['depth'], density_flagged)

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

            # RTOFS binary
            if rtofs_flag:
                ax1.plot(rdsi['temperature'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)
                ax2.plot(rdsi['salinity'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)
                ax3.plot(rdsi['density'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)

            ax1.set_ylim([depth, 0])
            ax1.set_xlim(temp_xlim if temp_xlim is not None else wmo_tlim)
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

            ax2.set_ylim([depth, 0])
            ax2.set_xlim(sal_xlim if sal_xlim is not None else wmo_slim)
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')

            ax3.set_ylim([depth, 0])
            ax3.set_xlim(density_xlim if density_xlim is not None else wmo_dlim)
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
            if any_flagged:
                h.append(QC_FLAGGED_HANDLE)
                l.append(QC_FLAGGED_HANDLE.get_label())
            # Sort alphabetically by label, keeping QC flagged handle at the end
            paired = sorted(zip(l, h), key=lambda x: (x[0] != 'Argo QC flagged', x[0]))
            l, h = zip(*paired) if paired else ([], [])
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
            if np.isnan(ohc_espc):
                ohc_string += 'ESPC: N/A,  '
            else:
                ohc_string += f"ESPC: {ohc_espc:.4f},  "
            if np.isnan(ohc_cmems):
                ohc_string += 'CMEMS: N/A,  '
            else:
                ohc_string += f"CMEMS: {ohc_cmems:.4f},  "
            if np.isnan(ohc_rtofs):
                ohc_string += 'RTOFS: N/A'
            else:
                ohc_string += f"RTOFS: {ohc_rtofs:.4f}"

            plt.figtext(0.4, 0.001, ohc_string, ha="center", fontsize=10, fontstyle='italic')

            plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            if ctime > then:
                try:
                    os.symlink(full_file, symlink_dir / save_str)
                except FileExistsError:
                    pass

                locations_file = symlink_dir / 'locations.json'
                locations = {}
                if locations_file.exists():
                    try:
                        with open(locations_file, 'r') as f:
                            locations = json.load(f)
                    except Exception:
                        pass
                locations[save_str] = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'wmo': str(wmo),
                    'time': tstr
                }
                try:
                    with open(locations_file, 'w') as f:
                        json.dump(locations, f)
                except Exception as e:
                    print(f"Error saving locations.json: {e}")


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
