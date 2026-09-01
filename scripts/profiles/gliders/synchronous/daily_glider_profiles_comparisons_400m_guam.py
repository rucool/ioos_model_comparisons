#!/usr/bin/env python
"""
Daily glider profile vs model comparisons for Guam.

RTOFS OPeNDAP does not cover the Guam domain, so RTOFS is read from the same
pre-processed binary NetCDFs that
scripts/maps/models/synchronous/rtofs_binary_model_comparisons.py produces
(rtofs_archv/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_guam.nc), instead of
ioos_model_comparisons.models.rtofs(). See argo_profile_model_comparisons_rtofs_binary.py
for the Argo equivalent of this pattern.
"""
# %%
import datetime as dt
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from pathlib import Path
from cool_maps.plot import (
    create
    )

import ioos_model_comparisons.configs as configs
import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import (density,
                                         ocean_heat_content,
                                         depth_bin,
                                         depth_interpolate,
                                         lon180to360
                                         )
from ioos_model_comparisons.platforms import get_active_gliders, get_ohc
from ioos_model_comparisons.regions import region_config
import pandas
import re
from datetime import datetime
import cartopy.feature as cfeature
import math
from cool_maps.plot import get_bathymetry
import logging
logging.basicConfig(level=logging.INFO)  # or adjust logging level as needed

# %%
# set path to save plots
path_save = (configs.path_plots / "profiles" / "gliders")

# Create and maintain last_14_days directory
import json as _json
import glob as _glob
import fcntl as _fcntl
symlink_dir = path_save / 'last_14_days'
os.makedirs(symlink_dir, exist_ok=True)

for f in sorted(_glob.glob(os.path.join(symlink_dir, '*.png'))):
    match = re.search(r'_(\d{4})(\d{2})(\d{2})_to_', f)
    if match:
        file_date = dt.datetime.strptime(f"{match.group(1)}{match.group(2)}{match.group(3)}", '%Y%m%d')
        if (dt.datetime.now() - file_date).days > 14:
            os.remove(f)

_loc14_file = symlink_dir / 'locations.json'
_loc14_lock = symlink_dir / 'locations.lock'
if _loc14_file.exists():
    try:
        with open(_loc14_lock, 'w') as _lf:
            _fcntl.flock(_lf, _fcntl.LOCK_EX)
            with open(_loc14_file, 'r') as _f:
                _loc14 = _json.load(_f)
            _loc14 = {k: v for k, v in _loc14.items() if (symlink_dir / k).exists()}
            with open(_loc14_file, 'w') as _f:
                _json.dump(_loc14, _f)
    except Exception as _e:
        print(f"Error cleaning up last_14_days/locations.json: {_e}")

# dac access
parallel = False
timeout = 60
days = 7
today = dt.date.today()
spatial_interp = False
workers = 4

# Region selection
conf.regions = ['guam']
RTOFS_BINARY_REGION = 'guam'

# RTOFS binary pre-processed NetCDF directory (created by grab_rtofs_archv_aws.py)
RTOFS_DATA_DIR = Path("/home/hurricaneadm/data/rtofs_archv")

# Max age difference between a glider profile and the nearest RTOFS file (hours)
RTOFS_MAX_HOURS = 12

# Model selection
plot_rtofs = True
plot_espc = True
plot_cmems = True
plot_para = False

# Subplot selection
plot_temperature = True
plot_salinity = True
plot_density = True

# Depth
depth = 400

# Glider profile averaging method: 'bin' or 'interpolate'
glider_depth_method = 'interpolate'

# Time threshold (in hours). If a profile time is greater than this, we won't
# grab the corresponding profile from the model
time_threshold = 6 # hours

# Create a date list.
date_list = [today - dt.timedelta(days=x+1) for x in range(days)]
date_list.insert(0, today + dt.timedelta(days=1))
date_list.reverse()

# Get time bounds for the current day
t0 = date_list[0]
t1 = date_list[-1]
# %% Look for datasets in IOOS glider dac
vars = ['time', 'latitude', 'longitude', 'depth', 'temperature', 'salinity',
        'density', 'profile_id']

region_gliders = []
for region in conf.regions:
    print('Region:', region)
    extent = region_config(region)["extent"]
    gliders = get_active_gliders(extent, t0, t1,
                            variables=vars,
                            timeout=timeout,
                            parallel=False).reset_index()
    gliders['region'] = region
    region_gliders.append(gliders)

gliders = pd.concat(region_gliders)

def pick_region_map(regions, point):
    distances = []
    for region in regions:
        extent = region_config(region)["extent"]
        x0, x1, y0, y1 = extent
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        distance = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        distances.append((distance, region, (center_x, center_y)))
    return min(distances, key=lambda x: x[0])

# %% Load models
if plot_espc:
    from ioos_model_comparisons.models import espc_ts #ESPC
    print('Loading ESPC')
    espc_loaded = espc_ts(rename=True)
    print('ESPC loaded')
    glabel = f'ESPC' # Legend labels

if plot_cmems:
    from ioos_model_comparisons.models import CMEMS
    print('Loading CMEMS')

    # Read Copernicus
    cobj = CMEMS()
    print('CMEMS loaded')
    clabel = f"CMEMS" # Legend labels

rlabel = 'RTOFS' # Legend label

# Convert time threshold to a Timedelta so that we can compare timedeltas.
time_threshold= pd.Timedelta(hours=time_threshold)

# %% Define functions
def line_limits(fax, delta=1):
    """Function to get the minimum and maximum of a series of lines from a
    Matplotlib axis.

    Args:
        fax (_type_): Matplotlib Axes
        delta (float, optional): Delta for . Defaults to 1.

    Returns:
        _type_: _description_
    """
    mins = [np.nanmin(line.get_xdata()) for line in fax.lines]
    maxs = [np.nanmax(line.get_xdata()) for line in fax.lines]
    return min(mins)-delta, max(maxs)+delta

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']

def round_to_nearest_ten(n):
    if n % 10 >= 5:
        return ((n // 10) + 1) * 10
    else:
        return (n // 10) * 10


# ── RTOFS binary helpers ─────────────────────────────────────────────────────
# Guam is not covered by the RTOFS OPeNDAP grid, so profiles are pulled from
# the pre-processed regional NetCDFs instead (see module docstring).

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

        # The file may use 0–360 lons. Convert the lookup lon to match.
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


def plot_glider_profiles(id, gliders):
    print('Plotting ' + id)

    # Subset the glider dataframe by a given id
    df = gliders[gliders['glider'] == id]

    # Remove any duplicate glider entries
    df = list(df.groupby('region'))[0][1]

    # Get extent for inset map
    # Find which region it's in most recently
    found = pick_region_map(conf.regions, (df.lon.iloc[-1], df.lat.iloc[-1]))
    extent = region_config(found[1])["extent"]

    # Extract glider id and deployment timestamp from dac id
    match = re.search(r'(.*)-(\d{8}T\d{4})', id)
    glid = match.group(1)
    datetime_str = match.group(2)
    deployed = datetime.strptime(datetime_str, '%Y%m%dT%H%M')
    print('Glider ID:', glid)
    print('Deployed:', deployed)

    alabel = f'{glid}'

    for t in list(df.groupby(df['time'].dt.date)):
        tdf = t[1]
        t0 = t[0]
        t1 = t[0] + dt.timedelta(days=1)

        spath = path_save / str(today.year) / t0.strftime('%m-%d')
        os.makedirs(spath, exist_ok=True)

        fullfile = spath / f"{id}_{t0.strftime('%Y%m%d')}_to_{t1.strftime('%Y%m%d')}_400m.png"

        # Initialize plot
        fig = plt.figure(constrained_layout=True, figsize=(16, 8))
        widths = [1, 1, 1, 1]
        heights = [1, 2, 1]

        gs = fig.add_gridspec(3, 4, width_ratios=widths,
                                height_ratios=heights)

        tax = fig.add_subplot(gs[:, 0]) # Temperature
        sax = fig.add_subplot(gs[:, 1], sharey=tax)  # Salinity
        plt.setp(sax.get_yticklabels(), visible=False)
        dax = fig.add_subplot(gs[:, 2], sharey=tax) # Density
        plt.setp(dax.get_yticklabels(), visible=False)
        ax4 = fig.add_subplot(gs[0, -1]) # Title
        mpax = fig.add_subplot(gs[1, -1], projection=configs.projection['map']) # Map
        lax = fig.add_subplot(gs[2, -1]) # Legend

        lon_track = []
        lat_track = []

        # Filter glider depth
        tdf = tdf[(tdf["depth"] > 0.5) & (tdf["depth"] <= 400)]

        # Groupby glider profiles
        maxd = []
        ohc_glider = []

        # Creating individual arrays
        array1 = np.arange(0, 10, 2) # From 0 to 10 with step size 2
        array2 = np.arange(10, 101, 5) # From 10 to 100 with step size 5 (101 is the stop point to include 100)
        array3 = np.arange(110, 401, 10) # From 110 to 1000 with step size 10 (1001 is the stop point to include 1000)

        # Concatenating the arrays for bins to interpolate to
        bins = np.concatenate((array1, array2, array3))

        binned = []
        if not tdf.empty:
            for name, pdf in tdf.groupby(['profile_id', 'time', 'lon', 'lat']):
                if not pdf.empty:
                    print(f'plotting profile {name}')
                    pdf['density'] = density(pdf['temperature'].values, -pdf['depth'].values, pdf['salinity'].values, pdf['lat'].values, pdf['lon'].values)
                    if glider_depth_method == 'interpolate':
                        tmp_depth = depth_interpolate(pdf.select_dtypes(exclude=['object']), depth_var='depth', bins=bins)
                    else:
                        tmp_depth = depth_bin(pdf.select_dtypes(exclude=['object']), depth_var='depth', aggregation='mean', bins=bins)
                    binned.append(tmp_depth)
                    pid = name[0]
                    time_glider = name[1]
                    lon_glider = name[2].round(2)
                    lat_glider = name[3].round(2)
                    lon_track.append(lon_glider)
                    lat_track.append(lat_glider)

                    print(f"Glider: {id}, Profile ID: {pid}, Time: {time_glider}")

                    # Filter salinity and temperature that are more than 4 standard deviations
                    # from the mean
                    try:
                        pdf = pdf[np.abs(stats.zscore(pdf['salinity'])) < 4]  #  salinity
                        pdf = pdf[np.abs(stats.zscore(pdf['temperature'])) < 4]  #  temperature
                    except pandas.errors.IndexingError:
                        pass

                    # Save as Pd.Series for easier recalling of columns
                    depth_glider = pdf['depth']
                    temp_glider = pdf['temperature']
                    salinity_glider = pdf['salinity']
                    density_glider = pdf['density']

                    # Plot glider profiles
                    tax.plot(temp_glider, depth_glider, '.', color='cyan', linestyle='None', label='_nolegend_')
                    sax.plot(salinity_glider, depth_glider, '.', color='cyan', linestyle='None', label='_nolegend_')
                    dax.plot(density_glider, depth_glider, '.', color='cyan', linestyle='None', label='_nolegend_')

                    try:
                        maxd.append(np.nanmax(depth_glider))
                    except:
                        continue
                    ohc = ocean_heat_content(depth_glider, temp_glider, density_glider)
                    ohc_glider.append(ohc)
                else:
                    print('Test')
                    continue
        else:
            continue

        bin_avg = pd.concat(binned).groupby('depth').mean().reset_index()

        mlon = tdf['lon'].mean()
        mlat = tdf['lat'].mean()

        mlon360 = lon180to360(mlon)
        try:
            nesdis = get_ohc(extent, time_glider.date())
        except:
            nesdis = None

        if nesdis:
            nesdis = nesdis.squeeze()
            ohc_nesdis = nesdis.sel(longitude=mlon, latitude=mlat, method='nearest')
            ohc_nesdis = ohc_nesdis.ohc.values

        if plot_espc:
            # Select the nearest model time to the glider time for this profile
            gds = espc_loaded.sel(lon=mlon360, lat=mlat, method='nearest')
            gds = gds.sel(time=time_glider, method="nearest")

            gds = gds.sel(depth=slice(0, depth)).squeeze()
            # FMRC datasets retain a reftime dimension after time selection; drop it
            extra_dims = [d for d in gds.dims if d != 'depth']
            if extra_dims:
                gds = gds.isel({d: 0 for d in extra_dims})
            gds['salinity'].load()
            gds['temperature'].load()

            # Calculate density
            d_g = density(gds['temperature'].values, -gds['depth'].values, gds['salinity'].values, float(gds['lat']), float(gds['lon']))
            gds['density'] = (('depth'), d_g)

            print(f"ESPC - Time: {pd.to_datetime(gds.time.values)}")

            ohc_espc = ocean_heat_content(gds['depth'].values, gds['temperature'].values, gds['density'].values)

        rtofs_flag = False
        if plot_rtofs:
            # RTOFS binary (Guam is not covered by RTOFS OPeNDAP)
            rtofs_nc = find_rtofs_file(RTOFS_BINARY_REGION, time_glider)
            if rtofs_nc is not None:
                rds = load_rtofs_point(rtofs_nc, mlon, mlat, max_depth=depth)
                if rds is not None:
                    try:
                        rds['density'] = (('depth'), density(
                            rds['temperature'].values, -rds['depth'].values,
                            rds['salinity'].values, float(rds.lat), float(rds.lon),
                        ))
                        ohc_rtofs = ocean_heat_content(rds['depth'].values, rds['temperature'].values, rds['density'].values)
                        print(f"RTOFS binary - Time: {_parse_rtofs_time(rtofs_nc)}")
                        rtofs_flag = True
                    except Exception as e:
                        print(f"RTOFS binary: compute failed ({e})")
            else:
                print(f"RTOFS binary: no file found within {RTOFS_MAX_HOURS}h of {time_glider} for region '{RTOFS_BINARY_REGION}'")
            if not rtofs_flag:
                ohc_rtofs = np.nan

        if plot_cmems:
            # CMEMS
            cds = cobj.get_point(mlon, mlat, time_glider, interp=spatial_interp)
            cds = cds.sel(depth=slice(0, depth)).squeeze()

            cds['salinity'].load()
            cds['temperature'].load()

            print(f"CMEMS - Time: {pd.to_datetime(cds.time.values)}")

            # Calculate density
            d_c = density(cds['temperature'].values, -cds['depth'].values, cds['salinity'].values, float(cds['lat']), float(cds['lon']))
            cds['density'] = (('depth'), d_c)
            ohc_cmems = ocean_heat_content(cds['depth'].values, cds['temperature'].values, cds['density'].values)

        # Plot model profiles
        if rtofs_flag:
            tax.plot(rds['temperature'], rds['depth'], '.-', linewidth=5, color='red',  label='_nolegend_')
            sax.plot(rds['salinity'], rds['depth'], '.-', linewidth=5, color='red',  label='_nolegend_')
            dax.plot(rds['density'], rds['depth'], '.-', linewidth=5, color='red', label='_nolegend_')

        if plot_espc:
            tax.plot(gds['temperature'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
            sax.plot(gds['salinity'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
            dax.plot(gds['density'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')

        if plot_cmems:
            tax.plot(cds['temperature'], cds["depth"], '.-', color="magenta", label='_nolegend_')
            sax.plot(cds['salinity'], cds["depth"], '.-', color="magenta", label='_nolegend_')
            dax.plot(cds['density'], cds["depth"], '.-', color="magenta", label='_nolegend_')

        # Plot glider profile
        tax.plot(bin_avg['temperature'], bin_avg['depth'], '-o', color='blue', label=f"{alabel} (Average Profile)")
        sax.plot(bin_avg['salinity'], bin_avg['depth'], '-o', color='blue', label=f"{alabel} (Average Profile)")
        dax.plot(bin_avg['density'], bin_avg['depth'], '-o', color='blue', label=f"{alabel} (Average Profile)")

        # Plot model profiles
        if rtofs_flag:
            tax.plot(rds['temperature'], rds['depth'], '-o', color='red', label=rlabel)
            sax.plot(rds['salinity'], rds['depth'], '-o', color='red', label=rlabel)
            dax.plot(rds['density'], rds['depth'], '-o', color='red', label=rlabel)

        if plot_espc:
            tax.plot(gds['temperature'], gds["depth"], '-o', color="green", label=glabel)
            sax.plot(gds['salinity'], gds["depth"], '-o', color="green", label=glabel)
            dax.plot(gds['density'], gds["depth"], '-o', color="green", label=glabel)

        if plot_cmems:
            tax.plot(cds['temperature'], cds["depth"], '-o', color="magenta", label=clabel)
            sax.plot(cds['salinity'], cds["depth"], '-o', color="magenta", label=clabel)
            dax.plot(cds['density'], cds["depth"], '-o', color="magenta", label=clabel)
        try:
            # Get min and max of each plot. Add a delta to each for x limits
            tmin, tmax = line_limits(tax, delta=.5)
            smin, smax = line_limits(sax, delta=.25)
            dmin, dmax = line_limits(dax, delta=.5)
        except ValueError:
            print('Some kind of error')
            pass

        md = np.nanmax(maxd)

        if md < 400:
            ylim = [md, 0]
            if md < 50:
                yticks = np.arange(0, md+5, 5)
            elif md <= 100:
                yticks = np.arange(0, md+10, 10)
            elif md < 200:
                yticks = np.arange(0, md+10, 20)
            elif md < 300:
                yticks = np.arange(0, md+25, 25)
            else:
                yticks = np.arange(0, 425, 25)
        else:
            ylim = [401, 0]
            yticks = np.arange(0, 425, 25)

        # Adjust plots
        tax.set_xlim([tmin, tmax])
        tax.set_ylim(ylim)
        tax.set_yticks(yticks)
        tax.set_ylabel('Depth (m)', fontsize=13, fontweight="bold")
        tax.set_xlabel('Temperature ($^oC$)', fontsize=13, fontweight="bold")
        tax.grid(True, linestyle='--', linewidth=0.5)

        sax.set_xlim([smin, smax])
        sax.set_ylim(ylim)
        sax.set_xlabel('Salinity', fontsize=13, fontweight="bold")
        sax.grid(True, linestyle='--', linewidth=0.5)

        dax.set_xlim([dmin, dmax])
        dax.set_ylim(ylim)
        dax.set_xlabel('Density (kg m-3)', fontsize=13, fontweight="bold")
        dax.grid(True, linestyle='--', linewidth=0.5)
        # Rotate the x-axis labels by 45 degrees (you can adjust this angle)
        dax.tick_params(axis='x', labelrotation=45)

        if spatial_interp:
            method = "Interpolation"
        else:
            method = "Nearest-Neighbor"

        title_str = (f'Comparison Date: { tdf["time"].min().strftime("%Y-%m-%d") }\n\n'
                    f'Glider: {glid}\n'
                    f'Profiles: { tdf["profile_id"].nunique() }\n'
                    f'First: { str(tdf["time"].min()) }\n'
                    f'Last: { str(tdf["time"].max()) }\n'
                    f'Method: {method}\n'
                    )

        # Add text to title axis
        text = ax4.text(-0.1, 1.0,
                        title_str,
                        ha='left', va='top', size=13, fontweight='bold')

        text.set_path_effects([path_effects.Normal()])
        ax4.set_axis_off()

        lon_track = np.array(lon_track)
        lat_track = np.array(lat_track)
        dx = 2/2
        dy = 1.25/2
        extent_main = [lon_track.min() - .2, lon_track.max() + .2, lat_track.min() - .2, lat_track.max() + .2]
        extent_inset = [lon_track.min() - dx, lon_track.max() + dx, lat_track.min() - dy, lat_track.max() + dy]

        # Create a map in the map axis
        create(extent, ax=mpax, bathymetry=False)
        mpax.plot(lon_track, lat_track, '.-w',
                markeredgecolor='black',
                markersize=8,
                linewidth=4,
                transform=configs.projection['data'],
                zorder=999,
                )

        mpax.plot(lon_track[-1], lat_track[-1],
                marker='.',
                markeredgecolor='black',
                markerfacecolor='red',
                markersize=10,
                transform=configs.projection['data'],
                zorder=1000
                )
        mpax.tick_params(axis='x', labelrotation=45)

        h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

        # Create custom legend item of cyan dot for glider profiles
        from matplotlib.lines import Line2D
        glider_profile_legend = Line2D([0], [0], marker='.', color='w', label='Glider Profiles',
                              markerfacecolor='cyan', markersize=10)
        h = [glider_profile_legend] + h
        l = [f'{glid} (Raw Data Points)'] + l

        lax.legend(h, l, ncol=1, loc='center', fontsize=13)
        lax.set_axis_off()

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        ohc_string = 'Ocean Heat Content (kJ/cm^2) - '
        try:
            if np.isnan(np.nanmean(ohc_glider)):
                ohc_string += 'Glider: N/A,  '
            else:
                ohc_string += f"Glider: {np.nanmean(ohc_glider):.4f},  "
        except:
            pass

        if np.isnan(ohc_rtofs):
            ohc_string += 'RTOFS: N/A,  '
        else:
            ohc_string += f"RTOFS: {ohc_rtofs:.4f},  "

        try:
            if np.isnan(ohc_espc):
                ohc_string += 'ESPC: N/A,  '
            else:
                ohc_string += f"ESPC: {ohc_espc:.4f},  "
        except:
            pass

        try:
            if np.isnan(ohc_cmems):
                ohc_string += 'CMEMS: N/A,  '
            else:
                ohc_string += f"CMEMS: {ohc_cmems:.4f},  "
        except:
            pass

        if nesdis:
            try:
                ohc_string += f"NESDIS: {ohc_nesdis:.4f},  "
            except:
                pass

        plt.figtext(0.4, 0.001, ohc_string, ha="center", fontsize=10, fontstyle='italic')

        plt.savefig(fullfile, dpi=configs.dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Save locations.json
        locations_file = spath / 'locations.json'
        import json
        locations = {}
        if locations_file.exists():
            try:
                with open(locations_file, 'r') as f:
                    locations = json.load(f)
            except:
                pass

        loc_entry = {
            'lat': float(mlat),
            'lon': float(mlon),
            'glider_id': str(id),
            'time': str(t0.strftime('%Y-%m-%d'))
        }
        locations[fullfile.name] = loc_entry
        try:
            with open(locations_file, 'w') as f:
                json.dump(locations, f)
        except Exception as e:
            print(f"Error saving locations.json: {e}")

        # Update last_14_days symlink and locations.json
        symlink_target = symlink_dir / fullfile.name
        if not symlink_target.exists():
            try:
                os.symlink(fullfile, symlink_target)
            except Exception as e:
                print(f"Error creating symlink: {e}")
        try:
            with open(_loc14_lock, 'w') as lf:
                _fcntl.flock(lf, _fcntl.LOCK_EX)
                loc14 = {}
                if _loc14_file.exists():
                    with open(_loc14_file, 'r') as f:
                        loc14 = json.load(f)
                loc14[fullfile.name] = loc_entry
                with open(_loc14_file, 'w') as f:
                    json.dump(loc14, f)
        except Exception as e:
            print(f"Error updating last_14_days/locations.json: {e}")


from functools import partial

def driver(gliders, id):
    plot_glider_profiles(id, gliders)

def main():
    active_gliders = gliders.glider.unique().tolist()
    if parallel:
        import concurrent.futures

        # Use partial to input half of the function inputs.
        f = partial(driver, gliders)

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(f, active_gliders)
    else:
        for id in active_gliders:
            plot_glider_profiles(id, gliders)


if __name__ == "__main__":
    main()
