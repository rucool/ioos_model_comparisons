#!/usr/bin/env python
# %%
import datetime as dt
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from cool_maps.plot import (
    create
    )

import ioos_model_comparisons.configs as configs
import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import (density,
                                         ocean_heat_content,
                                         depth_bin,
                                         depth_interpolate,
                                         difference,
                                         lon180to360
                                         )
from ioos_model_comparisons.platforms import get_active_gliders, get_ohc, get_glider_by_id
from ioos_model_comparisons.regions import region_config
import pandas
import matplotlib.pyplot as plt
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

# dac access
parallel = True
timeout = 400
days = 2
today = dt.date.today()
interp = False
workers = 4

# Region selection
conf.regions = ['caribbean', 'gom', 'mab', 'sab']
# conf.regions = ['mab']
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
depth = 1000

# Time threshold (in hours). If a profile time is greater than this, we won't 
# grab the corresponding profile from the model
time_threshold = 6 # hours

# Get extent for all configured regions to download argo/glider data one time
extent_list = []

# Create a date list.
date_list = [today - dt.timedelta(days=x+1) for x in range(days)]
# date_list.insert(0, today)
date_list.insert(0, today + dt.timedelta(days=1))
date_list.reverse()

# Get time bounds for the current day
t0 = date_list[0]
t1 = date_list[-1]

# t0 = dt.datetime(2024, 6, 13)
# t1 = dt.datetime(2024, 7, 10)
# %% Look for datasets in IOOS glider dac
vars = ['time', 'latitude', 'longitude', 'depth', 'temperature', 'salinity',
        'density', 'profile_id']

region_gliders = []
# for region in conf.regions:
#     print('Region:', region)
#     extent = region_config(region)["extent"]
#     gliders = get_active_gliders(extent, t0, t1, 
#                             variables=vars,
#                             timeout=timeout, 
#                             parallel=False).reset_index()
#     gliders['region'] = region
#     region_gliders.append(gliders)

try:
    redwing = get_glider_by_id('redwing-20251011T1511', start=t0, end=t1)
    redwing.reset_index(inplace=True)
    redwing['glider'] = 'redwing-20251011T1511'
    redwing['region'] = 'sentinel'
    redwing.rename(columns={'longitude': 'lon', 'latitude': 'lat'}, inplace=True)
    region_gliders.append(redwing)
except Exception as e:
    logging.error(f"Failed to load redwing glider data: {e}")

gliders = pd.concat(region_gliders)

# Remove redwing-20251011T1511 and region ='mab' if exists
gliders = gliders[~((gliders['glider'] == 'redwing-20251011T1511') & (gliders['region'] == 'mab'))]

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
    # Read ESPC output
    # espc_loaded = ESPC(uv=False)
    espc_loaded = espc_ts(rename=True)

    print('ESPC loaded')
    glabel = f'ESPC' # Legend labels

if plot_para:
    from ioos_model_comparisons.models import rtofs
    print('Loading RTOFS Parallel')
    # RTOFS Parallel
    rtofs_para = rtofs(source='parallel').sel(depth=slice(0,depth))
    print('RTOFS Parallel loaded')
    # from glob import glob
    # import xarray as xr
    # import os
    rplabel = "RTOFS-P"
    # url = '/Users/mikesmith/Downloads/rtofs.parallel.v2.3/'
    # rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
    # rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

    # rtofs_para = xr.open_mfdataset(rtofs_files)
    # rtofs_para = rtofs_para.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth', 'X': 'x', 'Y': 'y'})
    rtofs_para.attrs['model'] = "RTOFS-P"

if plot_rtofs:
    from ioos_model_comparisons.models import rtofs
    print('Loading RTOFS')
    # Read RTOFS grid and time
    rtofs = rtofs().sel(depth=slice(0,depth))
    print('RTOFS loaded')
    rlabel = f'RTOFS' # Legend labels

    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlon = rtofs.lon.data[0, :]
    rlat = rtofs.lat.data[:, 0]
    rx = rtofs.x.data
    ry = rtofs.y.data

if plot_cmems:
    from ioos_model_comparisons.models import CMEMS
    print('Loading CMEMS')
    
    # Read Copernicus
    cobj = CMEMS()
    print('CMEMS loaded')
    clabel = f"CMEMS" # Legend labels

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

def plot_glider_profiles(id, gliders):
    print('Plotting ' + id)

    # Subset the glider dataframe by a given id
    df = gliders[gliders['glider'] == id]

    # Remove any duplicate glider entries
    df = list(df.groupby('region'))[0][1]

    if 'redwing' in id:
        extent =  [-77.25, -49.75, 34.75, 45.25]
    else:
        # Get extent for inset map
        # Find which region it's in most recently
        found = pick_region_map(conf.regions, (df.lon.iloc[-1], df.lat.iloc[-1]))
        extent = region_config(found[1])["extent"]

    # # Get extent for inset map
    # # Find which region it's in most recently
    # found = pick_region_map(conf.regions, (df.lon.iloc[-1], df.lat.iloc[-1]))
    # extent = region_config(found[1])["extent"]

    try:
        bathy = get_bathymetry(extent)
        bathy_flag = True
    except:
        bathy_flag = False
        pass

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
        
        fullfile = spath / f"{id}_{t0.strftime('%Y%m%d')}_to_{t1.strftime('%Y%m%d')}_{depth}m.png"
        diff_file = spath / f"{id}_{t0.strftime('%Y%m%d')}_to_{t1.strftime('%Y%m%d')}_{depth}m_difference.png"

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
        tdf = tdf[tdf["depth"] <= depth]

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
                    tmp_depth = depth_bin(pdf.select_dtypes(exclude=['object']), depth_var='depth', depth_min=-1, depth_max=depth+10, stride=5, aggregation='mean')
                    # tmp_depth = depth_interpolate(pdf, depth_min=0, depth_max=400, bins=bins)
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

        # time_glider_str = time_glider.strftime("%Y-%m-%d")
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
            # gds = espc_loaded.get_point(mlon, mlat, time_glider, interp=False) #Archive data
            gds = espc_loaded.sel(lon=mlon360, lat=mlat, method='nearest')
            gds = gds.sel(time=time_glider, method="nearest")

            # espc_loaded
            gds = gds.sel(depth=slice(0,depth)).squeeze()
            gds['salinity'].load()
            gds['temperature'].load()

            # Calculate density
            gds['density'] = density(gds['temperature'].values, -gds['depth'].values, gds['salinity'].values, gds['lat'].values, gds['lon'].values)

            print(f"ESPC - Time: {pd.to_datetime(gds.time.values)}")

            ohc_espc = ocean_heat_content(gds['depth'].values, gds['temperature'].values, gds['density'].values)

        if plot_rtofs:
            # RTOFS
            rds = rtofs.sel(time=time_glider, method="nearest")
            print(f"RTOFS - Time: {pd.to_datetime(rds.time.values)}")

            # interpolating lon and lat to x and y index of the rtofs grid
            rlonI = np.interp(mlon, rlon, rx) # lon -> x
            rlatI = np.interp(mlat, rlat, ry) # lat -> y

            if interp:
                rds = rds.interp(
                    x=rlonI,
                    y=rlatI,
                )
            else:
                rds = rds.sel(
                    x=np.round(rlonI),
                    y=np.round(rlatI),
                    method='nearest'
                    )
            
            # Calculate density 
            d_g = density(rds['temperature'].values, -rds['depth'].values, rds['salinity'].values, rds['lat'].values, rds['lon'].values)
            rds['density'] = (('depth'), d_g)
            ohc_rtofs = ocean_heat_content(rds['depth'].values, rds['temperature'].values, rds['density'].values)

        if plot_para:
            # RTOFS
            rdsp = rtofs_para.sel(time=time_glider, method="nearest")
            print(f"RTOFS (parallel) - Time: {pd.to_datetime(rdsp.time.values)}")

            # interpolating lon and lat to x and y index of the rtofs grid
            rlonI = np.interp(mlon, rlon, rx) # lon -> x
            rlatI = np.interp(mlat, rlat, ry) # lat -> y

            if interp:
                rdsp = rdsp.interp(
                    x=rlonI,
                    y=rlatI,
                )
            else:
                rdsp = rdsp.sel(
                    x=np.round(rlonI),
                    y=np.round(rlatI),
                    method='nearest'
                    )
            
            # Calculate density 
            d_g = density(rdsp['temperature'].values, -rdsp['depth'].values, rdsp['salinity'].values, rdsp['lat'].values, rdsp['lon'].values)
            rdsp['density'] = (('depth'), d_g)
            ohc_rtofsp = ocean_heat_content(rdsp['depth'].values, rdsp['temperature'].values, rdsp['density'].values)
            
        if plot_cmems:
            # CMEMS
            if interp:
                cds = cobj.get_point(mlon, mlat, time_glider, interp=True)
            else:
                cds = cobj.get_point(mlon, mlat, time_glider, interp=False)
            cds = cds.sel(depth=slice(0,depth)).squeeze()

            cds['salinity'].load()
            cds['temperature'].load()

            print(f"CMEMS - Time: {pd.to_datetime(cds.time.values)}")
            # delta_time = np.abs(time_glider - pd.to_datetime(cds.time.values))
            # print(f"Threshold time: {delta_time}")
            # if delta_time < time_threshold:
            #     print(f"Difference between profile and nearest CMEMS time is {delta_time}. Interpolating to profile")

            # Calculate density
            cds['density'] = density(cds['temperature'].values, -cds['depth'].values, cds['salinity'].values, cds['lat'].values, cds['lon'].values)
            ohc_cmems = ocean_heat_content(cds['depth'].values, cds['temperature'].values, cds['density'].values)
            
        # Plot model profiles
        if plot_rtofs:
            tax.plot(rds['temperature'], rds['depth'], '.-', linewidth=5, color='red',  label='_nolegend_')
            sax.plot(rds['salinity'], rds['depth'], '.-', linewidth=5, color='red',  label='_nolegend_')
            dax.plot(rds['density'], rds['depth'], '.-', linewidth=5, color='red', label='_nolegend_')

        if plot_para:
            tax.plot(rdsp['temperature'], rdsp['depth'], '.-', color='black', markeredgecolor='black',  markersize=12, label='_nolegend_')
            sax.plot(rdsp['salinity'], rdsp['depth'], '.-', color='black', markeredgecolor='black', markersize=12, label='_nolegend_')
            dax.plot(rdsp['density'], rdsp['depth'], '.-', color='black', markeredgecolor='black', markersize=12, label='_nolegend_')

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
        rlabel = 'RTOFS'
        rplabel = 'RTOFS (Parallel)'
        if plot_rtofs:
            tax.plot(rds['temperature'], rds['depth'], '-o', color='red', label=rlabel)
            sax.plot(rds['salinity'], rds['depth'], '-o', color='red', label=rlabel)
            dax.plot(rds['density'], rds['depth'], '-o', color='red', label=rlabel)

        if plot_para:
            tax.plot(rdsp['temperature'], rdsp['depth'], '-o', color='orange', label=rplabel)
            sax.plot(rdsp['salinity'], rdsp['depth'], '-o', color='orange', label=rplabel)
            dax.plot(rdsp['density'], rdsp['depth'], '-o', color='orange', label=rplabel)

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
            ylim = [1010, 0] 
            yticks = np.arange(0, 1100, 100)
            
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

        if interp:
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
        # lonmin, lonmax, latmin, latmax = extent_inset
    
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

        # if bathy_flag:
        #     mpax.contourf(bathy['longitude'],
        #                 bathy['latitude'],
        #                 bathy['z'],
        #                 levels, colors=colors, transform=configs.projection['data'], ticks=False)
        
        # Create inset axis for glider track
        # axin = map_add_inset(mpax, extent=extent, zoom_extent=extent_inset)
        # add_features(axin)
        # if bathy_flag:
        #     axin.contourf(bathy['longitude'],
        #         bathy['latitude'],
        #         bathy['elevation'],
        #         levels, colors=colors, transform=configs.projection['data'], ticks=False)

        # axin.plot(lon_track, lat_track, '.-w', 
        #           markeredgecolor='black',
        #           markersize=8,
        #           linewidth=4,
        #           transform=configs.projection['data'],
        #           zorder=999
        #           )

        # axin.plot(lon_track[-1], lat_track[-1],  
        #         marker='.',
        #         markeredgecolor='black',
        #         markerfacecolor='red',
        #         markersize=8,
        #         transform=configs.projection['data'],
        #         zorder=1000
        #         )

        h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

        # Create custom legend item of cyan dot for glider profiles
        from matplotlib.lines import Line2D
        glider_profile_legend = Line2D([0], [0], marker='.', color='w', label='Glider Profiles',
                              markerfacecolor='cyan', markersize=10)
        h = [glider_profile_legend] + h
        l = [f'{glid} (Raw Data Points)'] + l
        # h.append(glider_profile_legend)
        # l.append(f'{id} (Daily Profiles)')

        lax.legend(h, l, ncol=1, loc='center', fontsize=13)
        lax.set_axis_off()

        # plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')

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
        
        try:
            if np.isnan(ohc_rtofs):
                ohc_string += 'RTOFS: N/A,  '
            else:
                ohc_string += f"RTOFS: {ohc_rtofs:.4f},  "
        except:
            pass

        try:
            if np.isnan(ohc_rtofsp):
                ohc_string += 'RTOFS (Parallel): N/A,  '
            else:
                ohc_string += f"RTOFS (Parallel): {ohc_rtofsp:.4f},  "
        except:
            pass
        
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

        # # Interpolate model to glider depth
        # if plot_rtofs:
        #     rdsi = rds.interp(depth=bin_avg['depth'])

        # if plot_para:
        #     rdsip = rdsp.interp(depth=bin_avg['depth'])

        # if plot_espc:
        #     gdsi = gds.interp(depth=bin_avg['depth'])

        # if plot_cmems:
        #     cdsi = cds.interp(depth=bin_avg['depth'])
            
        # # plot difference
        # fig = plt.figure(constrained_layout=True, figsize=(16, 6))
        # widths = [1, 1, 1, 1]
        # heights = [1, 2, 1]

        # gs = fig.add_gridspec(3, 4, width_ratios=widths,
        #                         height_ratios=heights)

        # tax = fig.add_subplot(gs[:, 0]) # Temperature
        # sax = fig.add_subplot(gs[:, 1], sharey=tax)  # Salinity
        # plt.setp(sax.get_yticklabels(), visible=False)
        # dax = fig.add_subplot(gs[:, 2], sharey=tax) # Density
        # plt.setp(dax.get_yticklabels(), visible=False)
        # ax4 = fig.add_subplot(gs[0, -1]) # Title
        # mpax = fig.add_subplot(gs[1, -1], projection=configs.projection['map']) # Map
        # lax = fig.add_subplot(gs[2, -1]) # Legend

        # # Temperature 
        # diff_g = difference(gdsi['temperature'], df['temp (degree_Celsius)'])
        # diff_r = difference(rdsi['temperature'], bin_avg['temperature'])
        # diff_rd = difference(rdsip['temperature'], bin_avg['temperature'])
        # diff_c = difference(cds['temperature'], df['temp (degree_Celsius)'])
            
        # # glabel = f'{diff_g[1]}, {diff_g[2]}'
        # rlabel = f'{diff_r[1]}, {diff_r[2]}'
        # rlabel1 = f'{diff_rd[1]}, {diff_rd[2]}'
        # # clabel = f"{diff_c[1]}, {diff_c[2]}"
            
        # # ax1.plot(diff_g[0], gdsi['depth'], 'g-o', label=glabel)
        # tax.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
        # tax.plot(diff_rd[0], bin_avg['depth'], '-o', color='orange', label=rlabel1)
        # # ax1.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)
        # tax.axvline(0)
        # tax.set_ylim([400, 0])
        # tax.set_xlim([-3, 3])
        # tax.grid(True, linestyle='--', linewidth=.5)
        # tax.tick_params(axis='both', labelsize=13)
        # tax.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
        # tax.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
        # tax.legend(title="bias, rms", loc=3, fontsize='small',)

        # # Salinity
        # # diff_g = difference(gdsi['salinity'], df['psal (PSU)'])
        # diff_r = difference(rdsi['salinity'], bin_avg['salinity'])
        # diff_rd = difference(rdsip['salinity'], bin_avg['salinity'])
        # # diff_c = difference(cdsi['salinity'], df['psal (PSU)'])
        
        # # glabel = f'{diff_g[1]}, {diff_g[2]}'
        # rlabel = f'{diff_r[1]}, {diff_r[2]}'
        # rlabel1 = f'{diff_rd[1]}, {diff_rd[2]}'
        # # clabel = f"{diff_c[1]}, {diff_c[2]}"
            
        # # ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
        # # ax2.plot(diff_g[0], gdsi['depth'], 'g-o', label=glabel)
        # sax.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
        # sax.plot(diff_rd[0], bin_avg['depth'], '-o', color='orange', label=rlabel1)
        # # ax2.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)
        # sax.axvline(0)
        # sax.set_ylim([400, 0])
        # sax.set_xlim([-1, 1])
        # sax.grid(True, linestyle='--', linewidth=.5)
        # sax.tick_params(axis='both', labelsize=13)
        # sax.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
        # sax.legend(title="bias, rms", loc=3, fontsize='small',)
        # # ax2.set_ylabel('Depth (m)', fontsize=14)

        # # Density
        # # diff_g = difference(gdsi['density'], df['density'])
        # diff_r = difference(rdsi['density'], bin_avg['density'])
        # diff_rd = difference(rdsip['density'], bin_avg['density'])
        # # diff_c = difference(cdsi['density'], df['density'])
        
        # # glabel = f'{diff_g[1]}, {diff_g[2]}'
        # rlabel = f'{diff_r[1]}, {diff_r[2]}'
        # rlabel1 = f'{diff_rd[1]}, {diff_rd[2]}'
        # # clabel = f"{diff_c[1]}, {diff_c[2]}"
        
        # # ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
        # # ax3.plot(diff_g[0], gdsi['depth'],'g-o', label=glabel)
        # dax.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
        # dax.plot(diff_rd[0], bin_avg['depth'], '-o', color='orange', label=rlabel1)
        # # ax3.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)
        # dax.set_ylim([400, 0])
        # dax.set_xlim([-.3, .3])
        # dax.grid(True, linestyle='--', linewidth=.5)
        # dax.tick_params(axis='both', labelsize=13)
        # dax.set_xlabel('Density', fontsize=14, fontweight='bold')
        # dax.legend(title="bias, rms", loc=3, fontsize='small',)
        # dax.axvline(0)
        # ax3.set_ylabel('Depth (m)', fontsize=14)

        # text = ax4.text(0.125, 1.0, 
        #                 f'Glider: {glid}\n'
        #                 f'ARGO:  { tstr }\n'
        #                 f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
        #                 f'GOFS : {pd.to_datetime(gdsi.time.data)}\n'
        #                 f'CMEMS: {pd.to_datetime(cdsi.time.data)}',
        #                 ha='left', va='top', size=15, fontweight='bold')
        
        # text = ax4.text(0.125, 1.0, 
        #         f'Glider: {glid}\n'
        #         f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
        #         f'RTOFS (Parallel): {pd.to_datetime(rdsp.time.data)}\n',
        #         ha='left', va='top', size=15, fontweight='bold')


        # text.set_path_effects([path_effects.Normal()])
        # ax4.set_axis_off()

        # # map_create(extent, ax=ax5, ticks=False)
        # # map_add_ticks(ax5, extent, fontsize=10)
        # create(extent, ax=mpax)
        
        # mpax.plot(bin_avg['lon'][0], bin_avg['lat'][0], 'ro', transform=conf.projection['data'], zorder=10000)

        # h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

        # # lax.legend(h, [f'GOFS [{ glon }, { glat }]', f'RTOFS [{ rlon }, { rlat }]', f"Copernicus [{ clon }, { clat }]"], ncol=1, loc='center', fontsize=12)
        # lax.legend(h, [ f'RTOFS', 'RTOFS (Parallel)'], ncol=1, loc='center', fontsize=12)
        # lax.set_axis_off()
        
        # # plt.figtext(0.15, 0.001, f'Depths interpolated to every {conf.stride}m', ha="center", fontsize=10, fontstyle='italic')

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.9)
        
        # # diff_file = temp_save_dir / f'{wmo}-profile-difference-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png' 

        # plt.savefig(diff_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.close()
        
from functools import partial
from joblib import Parallel, delayed

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
        
        # Use joblib to enable. The last argument in the function is what you input.
        # results = Parallel(n_jobs=workers)(delayed(f)(x) for x in active_gliders)
    else:
        for id in active_gliders:
            plot_glider_profiles(id, gliders)


if __name__ == "__main__":
    main()

# %%
