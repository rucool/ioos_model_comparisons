#!/usr/bin/env python
"""
Argo profile vs model comparisons for floats 4903137 and 4903138 (Puerto Rico
area, "Caribbean" region) that are not yet mirrored to the
IFREMER Argo GDAC and are therefore fetched from the OSMC ERDDAP instead:

    https://osmc.noaa.gov/erddap/tabledap/OSMC_flattened.html

ESPC + CMEMS + RTOFS (thredds) — same model-comparison machinery as
argo_profile_model_comparisons.py, but sourcing Argo profiles from
OSMC_flattened (platform_code, time, latitude, longitude, observation_depth,
ztmp, zsal) instead of the IFREMER ArgoFloats dataset.

OSMC_flattened has no per-point QC flags, so there is no QC-flag overlay here.

RTOFS is read live from the Rutgers thredds "us_east" scraped dataset, which
covers the Caribbean region directly — no pre-processed regional binary
NetCDFs needed here (unlike the Pacific regions in
argo_profile_model_comparisons_rtofs_binary.py, which fall outside that
thredds domain).
"""
import json
import os
import glob
import re
from collections import defaultdict
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from erddapy import ERDDAP

import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt
from cool_maps.plot import get_bathymetry
from ioos_model_comparisons.calc import lon180to360, lon360to180, density, ocean_heat_content
from ioos_model_comparisons.models import CMEMS, espc_ts, espc_ts_archive, rtofs
from ioos_model_comparisons.regions import region_config

# ── Config ──────────────────────────────────────────────────────────────────

save_dir = conf.path_plots / 'profiles' / 'argo'

depth = 400
dpi = conf.dpi

plot_espc = True
plot_cmems = True
plot_rtofs = True

# These two floats only appear on OSMC, not IFREMER, so fetch by platform code.
OSMC_PLATFORM_IDS = ['4903137', '4903138']
region_key = 'caribbean'

DATA_PROJECTION = ccrs.PlateCarree()
MAP_PROJECTION = ccrs.Mercator()

# ── Date range ──────────────────────────────────────────────────────────────

# Deployment/start date for these floats on OSMC.
date_start = pd.Timestamp('2026-06-30')
date_end = pd.Timestamp.now(tz='UTC').tz_localize(None)

then = pd.Timestamp.today() - pd.Timedelta(days=14)
then = pd.Timestamp(then.strftime('%Y-%m-%d'))

date_fmt = "%Y-%m-%dT%H:%MZ"
depths = slice(0, depth)


# ── OSMC Argo fetch ──────────────────────────────────────────────────────────

def get_osmc_floats(platform_ids, time_start, time_end):
    """Fetch Argo profiles from the OSMC ERDDAP (OSMC_flattened) for a list
    of platform codes not (yet) available via the IFREMER Argo GDAC.

    Returns a dataframe indexed by (argo, time) with columns lon, lat,
    depth, temp, psal — matching the shape consumed by process_argo() below.
    """
    e = ERDDAP(server='OSMC', protocol='tabledap', response='csv')
    e.dataset_id = 'OSMC_flattened'
    e.variables = [
        'platform_code', 'time', 'latitude', 'longitude',
        'observation_depth', 'ztmp', 'zsal',
    ]

    frames = []
    for platform_id in platform_ids:
        e.constraints = {
            'time>=': time_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'time<=': time_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'platform_code=': str(platform_id),
        }
        try:
            df = e.to_pandas(
                index_col='time (UTC)',
                parse_dates=True,
                skiprows=(1,),
            ).dropna().tz_localize(None)
        except Exception as err:
            print(f"OSMC: failed to fetch platform {platform_id}: {err}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).reset_index().rename(columns={
        'time (UTC)': 'time',
        'platform_code': 'argo',
        'latitude (degrees_north)': 'lat',
        'longitude (degrees_east)': 'lon',
        'observation_depth': 'depth',
        'ztmp (Deg C)': 'temp',
        'zsal': 'psal',
    })
    df['argo'] = df['argo'].astype(str)
    df = df.set_index(['argo', 'time']).sort_index()
    return df


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
rds = rtofs().sel(depth=depths) if plot_rtofs else None

# ── Argo floats ─────────────────────────────────────────────────────────────

floats = get_osmc_floats(OSMC_PLATFORM_IDS, date_start, date_end)
floats = floats[floats['depth'] <= depth]

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']


# ── Per-region processing ────────────────────────────────────────────────────

def process_argo():
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

    if plot_rtofs:
        # Setting RTOFS lon/lat/x/y to their own variables speeds up lookups
        rlons = rds.lon.data[0, :]
        rlats = rds.lat.data[:, 0]
        rx = rds.x.data
        ry = rds.y.data

    if floats.empty:
        print("No OSMC Argo floats found")
        return

    argo_region = floats

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

        df = df.sort_values('depth').copy()

        if df.empty:
            continue

        df = df.assign(
            density=density(
                df['temp'].values,
                -df['depth'].values,
                df['psal'].values,
                df['lat'].values,
                df['lon'].values,
            )
        )

        ohc_float = ocean_heat_content(df['depth'], df['temp'], df['density'])

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

        # ── RTOFS (thredds) ──────────────────────────────────────────────
        rtofs_flag = False
        rdsi = None
        rlabel = None
        ohc_rtofs = np.nan
        if plot_rtofs:
            try:
                # Interpolate lon/lat to the RTOFS curvilinear grid's x/y index
                rlonI = np.interp(lon, rlons, rx)
                rlatI = np.interp(lat, rlats, ry)

                rdsp = rds.sel(time=ctime, method='nearest')
                rdsi = rdsp.sel(x=rlonI, y=rlatI, method='nearest')
                rdsi.load()

                rdsi['density'] = density(rdsi.temperature, -rdsi.depth, rdsi.salinity, rdsi.lat, rdsi.lon)
                ohc_rtofs = ocean_heat_content(
                    rdsi['depth'].values,
                    rdsi['temperature'].values,
                    rdsi['density'].values,
                )
                rlon = rdsi.lon.data.round(2)
                rlat = rdsi.lat.data.round(2)
                rlabel = f'RTOFS [{rlon:.2f}, {rlat:.2f}]'
                leg_str += f'RTOFS: {pd.to_datetime(rdsi.time.data).strftime(date_fmt)}\n'
                rtofs_flag = True
            except KeyError as error:
                print(f"RTOFS: False - {error}")

        records_by_wmo[wmo].append(dict(
            wmo=wmo, ctime=ctime, tstr=tstr, save_str=save_str, full_file=full_file,
            df=df, ohc_float=ohc_float,
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
            temp_arrays.append(rec['df']['temp'].values)
            sal_arrays.append(rec['df']['psal'].values)
            dens_arrays.append(rec['df']['density'].values)
            for key, flag in (('gdsi', 'espc_flag'), ('cdsi', 'cmems_flag'), ('rdsi', 'rtofs_flag')):
                if rec[flag]:
                    temp_arrays.append(rec[key]['temperature'].values)
                    sal_arrays.append(rec[key]['salinity'].values)
                    dens_arrays.append(rec[key]['density'].values)
        return _bounds(temp_arrays, .5), _bounds(sal_arrays, .25), _bounds(dens_arrays, .5)

    # Pass 2: render one figure per surfacing, using per-float shared limits.
    for wmo, records in records_by_wmo.items():
        wmo_tlim, wmo_slim, wmo_dlim = _wmo_limits(records)

        for rec in records:
            ctime = rec['ctime']
            tstr = rec['tstr']
            save_str = rec['save_str']
            full_file = rec['full_file']
            df = rec['df']
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
            ax5 = fig.add_subplot(gs[1, -1], projection=MAP_PROJECTION)  # Map
            ax6 = fig.add_subplot(gs[2, -1])                       # Legend

            # Argo
            ax1.plot(df['temp'], df['depth'], 'b-o', label=alabel)
            ax2.plot(df['psal'], df['depth'], 'b-o', label=alabel)
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

            # RTOFS
            if rtofs_flag:
                ax1.plot(rdsi['temperature'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)
                ax2.plot(rdsi['salinity'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)
                ax3.plot(rdsi['density'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)

            ax1.set_ylim([depth, 0])
            ax1.set_xlim(wmo_tlim)
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

            ax2.set_ylim([depth, 0])
            ax2.set_xlim(wmo_slim)
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')

            ax3.set_ylim([depth, 0])
            ax3.set_xlim(wmo_dlim)
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
            paired = sorted(zip(l, h), key=lambda x: x[0])
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
    process_argo()


if __name__ == "__main__":
    main()
