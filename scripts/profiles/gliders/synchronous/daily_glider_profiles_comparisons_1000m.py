#!/usr/bin/env python
# %%
import datetime as dt
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from cool_maps.plot import add_features, create

import ioos_model_comparisons.configs as configs
import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import density, lon180to360, lon360to180, ocean_heat_content
from ioos_model_comparisons.models import cmems, gofs, rtofs
from ioos_model_comparisons.platforms import get_active_gliders, get_bathymetry
from ioos_model_comparisons.plotting import map_add_inset
from ioos_model_comparisons.regions import region_config

import matplotlib.pyplot as plt
import cmocean 
import re
from datetime import datetime
import cartopy.feature as cfeature

# %%
# set path to save plots
path_save = (configs.path_plots / "profiles" / "gliders")

# dac access
parallel = True
timeout = 60
days = 1
today = dt.date.today()
interp = False

# Model selection
plot_rtofs = True
plot_gofs = True
plot_cmems = True
plot_amseas = True

# Subplot selection
plot_temperature = True
plot_salinity = True
plot_density = True

# Time threshold (in hours). If a profile time is greater than this, we won't 
# grab the corresponding profile from the model
time_threshold = 6 # hours

# Get extent for all configured regions to download argo/glider data one time
extent_list = []
# conf.regions= ['mab']

# extent_df = pd.DataFrame(
#     np.array(extent_list),
#     columns=['lonmin', 'lonmax', 'latmin', 'latmax']
#     )

# global_extent = [
#     extent_df.lonmin.min(),
#     extent_df.lonmax.max(),
#     extent_df.latmin.min(),
#     extent_df.latmax.max()
#     ]

# Create a date list.
date_list = [today + dt.timedelta(days=x+1) for x in range(days)]
date_list.insert(0, today)
# date_list.reverse()

# Get time bounds for the current day
t0 = date_list[0]
t1 = date_list[1]

# %% Look for datasets in IOOS glider dac
vars = ['time', 'latitude', 'longitude', 'depth', 'temperature', 'salinity',
        'density', 'profile_id']

region_gliders = []

conf.regions = ['mab', 'sab', 'caribbean', 'gom', 'passengers']
# conf.regions = ['mab']
for region in conf.regions:
    # extent_list.append(region_config(region)["extent"])
    extent = region_config(region)["extent"]
    gliders = get_active_gliders(extent, t0, t1, 
                            variables=vars,
                            timeout=timeout, 
                            parallel=False).reset_index()
    gliders['region'] = region
    region_gliders.append(gliders)

gliders = pd.concat(region_gliders)
# # Get all active gliders in regions between times
# gliders = get_active_gliders(global_extent, t0, t1, 
#                              variables=vars,
#                              timeout=timeout, 
#                              parallel=False).reset_index()

# for region in conf.regions:
#     extent = region_config(region)["extent"]
#     gliders = get_active_gliders(extent, t0, t1, 
#                             variables=vars,
#                             timeout=timeout, 
#                             parallel=False).reset_index()


# %% Load models
if plot_gofs:
    # Read GOFS 3.1 output
    gofs = gofs(rename=True).sel(depth=slice(0,1000))
    glabel = f'GOFS' # Legend labels

if plot_rtofs:
    # Read RTOFS grid and time
    rtofs = rtofs().sel(depth=slice(0,1000))
    rlabel = f'RTOFS' # Legend labels

    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlon = rtofs.lon.data[0, :]
    rlat = rtofs.lat.data[:, 0]
    rx = rtofs.x.data
    ry = rtofs.y.data

if plot_cmems:
    # Read Copernicus
    cmems = cmems(rename=True).sel(depth=slice(0,1000))
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

spath = path_save / str(today.year) / today.strftime('%m-%d')
os.makedirs(spath, exist_ok=True)

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']

def plot_glider_profiles(id, gliders):
    print('Plotting ' + id)

    # Subset the glider dataframe by a given id
    df = gliders[gliders['glider'] == id]

    # Remove any duplicate glider entries
    df = list(df.groupby('region'))[0][1]

    # Get extent for inset map
    # try:
    region_name = df['region'].iloc[0]
    extent = region_config(region_name)["extent"]

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
    fullfile = spath / f"{id}_{t0.strftime('%Y%m%d')}_to_{t1.strftime('%Y%m%d')}_1000m.png"

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
    df = df[df["depth"] <= 1000]

    # Groupby glider profiles
    maxd = []
    ohc_glider = []
    for name, pdf in df.groupby(['profile_id', 'time', 'lon', 'lat']):
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
            pdf = pdf[np.abs(stats.zscore(df['salinity'])) < 4]  #  salinity
            pdf = pdf[np.abs(stats.zscore(df['temperature'])) < 4]  #  temperature
        except KeyError:
            pass

        # Save as Pd.Series for easier recalling of columns 
        depth_glider = pdf['depth']
        temp_glider = pdf['temperature']
        salinity_glider = pdf['salinity']
        density_glider = pdf['density']

        # Plot glider profiles
        tax.plot(temp_glider, depth_glider, '.', color='cyan', label='_nolegend_')
        sax.plot(salinity_glider, depth_glider, '.', color='cyan', label='_nolegend_')
        dax.plot(density_glider, depth_glider, '.', color='cyan', label='_nolegend_')
        maxd.append(np.nanmax(depth_glider))
        ohc = ocean_heat_content(depth_glider, temp_glider, density_glider)
        ohc_glider.append(ohc)

    if np.nanmax(maxd) < 400:
        plt.close()
        return
    
    mlon = df['lon'].mean()
    mlat = df['lat'].mean()
    
    if plot_gofs:
        # Convert glider lon from -180,180 to 0,359
        lon_glider_gofs = lon180to360(mlon)
    
        # Select the nearest model time to the glider time for this profile
        gds = gofs.sel(time=time_glider, method="nearest")

        # Interpolate the model to the nearest point
        if interp:
            gds = gds.interp(
                lon=lon_glider_gofs,
                lat=mlat,
                )
        else:
            # select nearest neighbor grid point
            gds = gds.sel(
                lon=lon_glider_gofs,
                lat=mlat,
                method="nearest"
            )
        
        # Convert lon from 0,259 to -180,180
        gds['lon'] = lon360to180(gds['lon'])

        # Calculate density
        gds['density'] = density(gds['temperature'].values, -gds['depth'].values, gds['salinity'].values, gds['lat'].values, gds['lon'].values)

        print(f"GOFS - Time: {pd.to_datetime(gds.time.values)}")

        ohc_gofs = ocean_heat_content(gds['depth'].values, gds['temperature'].values, gds['density'].values)

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
        rds['density'] = density(rds['temperature'].values, -rds['depth'].values, rds['salinity'].values, rds['lat'].values, rds['lon'].values)
        ohc_rtofs = ocean_heat_content(rds['depth'].values, rds['temperature'].values, rds['density'].values)

    if plot_cmems:
        # CMEMS
        cds = cmems.sel(time=time_glider, method="nearest")
        print(f"CMEMS - Time: {pd.to_datetime(cds.time.values)}")
        # delta_time = np.abs(time_glider - pd.to_datetime(cds.time.values))
        # print(f"Threshold time: {delta_time}")
        # if delta_time < time_threshold:
        #     print(f"Difference between profile and nearest CMEMS time is {delta_time}. Interpolating to profile")

        if interp:
            cds = cds.interp(
                lon=mlon,
                lat=mlat
            )
        else:
            cds = cds.sel(
                lon=mlon,
                lat=mlat,
                method='nearest'
            )
        # Calculate density
        cds['density'] = density(cds['temperature'].values, -cds['depth'].values, cds['salinity'].values, cds['lat'].values, cds['lon'].values)
        ohc_cmems = ocean_heat_content(cds['depth'].values, cds['temperature'].values, cds['density'].values)
        
    # Plot model profiles
    if plot_rtofs:
        tax.plot(rds['temperature'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')
        sax.plot(rds['salinity'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')
        dax.plot(rds['density'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')

    if plot_gofs:
        tax.plot(gds['temperature'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
        sax.plot(gds['salinity'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
        dax.plot(gds['density'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')

    if plot_cmems:
        tax.plot(cds['temperature'], cds["depth"], '.-', color="magenta", label='_nolegend_')
        sax.plot(cds['salinity'], cds["depth"], '.-', color="magenta", label='_nolegend_')
        dax.plot(cds['density'], cds["depth"], '.-', color="magenta", label='_nolegend_')

    # Plot glider profile
    tax.plot(temp_glider, depth_glider, '-o', color='blue', label=alabel)
    sax.plot(salinity_glider, depth_glider, '-o', color='blue', label=alabel)
    dax.plot(density_glider, depth_glider, '-o', color='blue', label=alabel)

    # Plot model profiles
    if plot_rtofs:
        tax.plot(rds['temperature'], rds['depth'], '-o', color='red', label=rlabel)
        sax.plot(rds['salinity'], rds['depth'], '-o', color='red', label=rlabel)
        dax.plot(rds['density'], rds['depth'], '-o', color='red', label=rlabel)

    if plot_gofs:
        tax.plot(gds['temperature'], gds["depth"], '-o', color="green", label=glabel)
        sax.plot(gds['salinity'], gds["depth"], '-o', color="green", label=glabel)
        dax.plot(gds['density'], gds["depth"], '-o', color="green", label=glabel)

    if plot_cmems:        
        tax.plot(cds['temperature'], cds["depth"], '-o', color="magenta", label=clabel)
        sax.plot(cds['salinity'], cds["depth"], '-o', color="magenta", label=clabel)    
        dax.plot(cds['density'], cds["depth"], '-o', color="magenta", label=clabel)

    # Get min and max of each plot. Add a delta to each for x limits
    tmin, tmax = line_limits(tax, delta=.5)
    smin, smax = line_limits(sax, delta=.25)
    dmin, dmax = line_limits(dax, delta=.5)

    md = np.nanmax(maxd)

    if md < 1000:
        ylim = [md, 0]
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

    if interp:
        method = "Interpolation"
    else:
        method = "Nearest-Neighbor"

    title_str = (f'Glider: {glid}\n'
                 f'Deployed: { deployed.strftime("%Y-%m-%d") }\n'
                 f'Profiles: { df["profile_id"].nunique() }\n'
                 f'First: { str(df["time"].min()) }\n'
                 f'Last: { str(df["time"].max()) }\n'
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
    create(extent_main, ax=mpax, bathymetry=False)
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
            markersize=8,
            transform=configs.projection['data'],
            zorder=1000
            )

    if bathy_flag:        
        mpax.contourf(bathy['longitude'],
                    bathy['latitude'],
                    bathy['elevation'],
                    levels, colors=colors, transform=configs.projection['data'], ticks=False)
        
    # Create inset axis for glider track
    axin = map_add_inset(mpax, extent=extent, zoom_extent=extent_inset)
    add_features(axin)
    if bathy_flag:
         axin.contourf(bathy['longitude'],
                  bathy['latitude'],
                  bathy['elevation'],
                  levels, colors=colors, transform=configs.projection['data'], ticks=False)
    
    axin.plot(lon_track, lat_track, '.-w', 
              markeredgecolor='black',
              markersize=8,
              linewidth=4,
              transform=configs.projection['data'],
              zorder=999
              )
   
    axin.plot(lon_track[-1], lat_track[-1],  
            marker='.',
            markeredgecolor='black',
            markerfacecolor='red',
            markersize=8,
            transform=configs.projection['data'],
            zorder=1000
            )

    h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

    lax.legend(h, l, ncol=1, loc='center', fontsize=13)
    lax.set_axis_off()

    # plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    ohc_string = 'Ocean Heat Content (Profiles) - '
    try:
        if np.isnan(np.nanmean(ohc_glider)):
            ohc_string += 'Glider: N/A,  '
        else:
            ohc_string += f"Glider: {np.nanmean(ohc_glider):.4f} kJ/cm^2,  "
    except:
        pass
    
    try:
        if np.isnan(ohc_rtofs):
            ohc_string += 'RTOFS: N/A,  '
        else:
            ohc_string += f"RTOFS: {ohc_rtofs:.4f} kJ/cm^2,  "
    except:
        pass
    
    try:           
        if np.isnan(ohc_gofs):
            ohc_string += 'GOFS: N/A,  '
        else:
            ohc_string += f"GOFS: {ohc_gofs:.4f} kJ/cm^2,  "
    except:
        pass
           
    try:
        if np.isnan(ohc_cmems):
            ohc_string += 'CMEMS: N/A,  '
        else:
            ohc_string += f"CMEMS: {ohc_cmems:.4f} kJ/cm^2,  "
    except:
        pass
    
    plt.figtext(0.4, 0.001, ohc_string, ha="center", fontsize=10, fontstyle='italic')

    plt.savefig(fullfile, dpi=configs.dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
    
from functools import partial
from joblib import Parallel, delayed

def driver(gliders, id):
    plot_glider_profiles(id, gliders)
    
def main():    
    # import concurrent.futures
    active_gliders = gliders.glider.unique().tolist()
        
    if parallel:
        import concurrent.futures
        workers = 6

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
