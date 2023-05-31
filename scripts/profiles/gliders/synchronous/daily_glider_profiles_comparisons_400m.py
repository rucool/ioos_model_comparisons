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
from ioos_model_comparisons.calc import density, lon180to360, lon360to180
from ioos_model_comparisons.models import cmems, gofs, rtofs
from ioos_model_comparisons.platforms import get_active_gliders
from ioos_model_comparisons.plotting import map_add_inset
from ioos_model_comparisons.regions import region_config

# %%
# set path to save plots
path_save = (configs.path_plots / "profiles" / "gliders")

# dac access
parallel = True
timeout = 60

# Get extent for all configured regions to download argo/glider data one time
extent_list = []
# conf.regions= ['mab']
for region in conf.regions:
    extent_list.append(region_config(region)["extent"])

extent_df = pd.DataFrame(
    np.array(extent_list),
    columns=['lonmin', 'lonmax', 'latmin', 'latmax']
    )

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

global_extent = [
    extent_df.lonmin.min(),
    extent_df.lonmax.max(),
    extent_df.latmin.min(),
    extent_df.latmax.max()
    ]

gliders = get_active_gliders(global_extent, t0, t1, 
                             variables=vars,
                             timeout=timeout, 
                             parallel=False).reset_index()

# %% Load models
if plot_gofs:
    # Read GOFS 3.1 output
    gofs = gofs(rename=True).sel(depth=slice(0,400))
    glabel = f'GOFS' # Legend labels

if plot_rtofs:
    # Read RTOFS grid and time
    rtofs = rtofs().sel(depth=slice(0,400))
    rlabel = f'RTOFS' # Legend labels

    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlon = rtofs.lon.data[0, :]
    rlat = rtofs.lat.data[:, 0]
    rx = rtofs.x.data
    ry = rtofs.y.data

if plot_cmems:
    # Read Copernicus
    cmems = cmems(rename=True).sel(depth=slice(0,400))
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

def plot_glider_profiles(id):
    print('Plotting ' + id)
    
    split = id.split('-')
    glid = split[0]
    deployed = pd.to_datetime(split[1]).strftime('%Y-%m-%dT%H:%MZ')
    df = gliders[gliders['glider'] == id]
    
    alabel = f'{glid}'
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
    df = df[df["depth"] <= 400]

    # Groupby glider profiles
    maxd = []
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

    if np.nanmax(maxd) < 400:
        xlim = [np.nanmax(maxd) * 1.1, 0]
    else:
        xlim = [400, 0]
        
    # Adjust plots
    tax.set_xlim([tmax, tmin])
    tax.set_ylim(xlim)
    tax.set_ylabel('Depth (m)', fontsize=13, fontweight="bold")
    tax.set_xlabel('Temperature ($^oC$)', fontsize=13, fontweight="bold")
    tax.grid(True, linestyle='--', linewidth=0.5)

    sax.set_xlim([smax, smin])
    sax.set_ylim(xlim)  
    sax.set_xlabel('Salinity', fontsize=13, fontweight="bold")
    sax.grid(True, linestyle='--', linewidth=0.5)

    dax.set_xlim([dmax, dmin])
    dax.set_ylim(xlim)
    dax.set_xlabel('Density (kg m-3)', fontsize=13, fontweight="bold")
    dax.grid(True, linestyle='--', linewidth=0.5)

    if interp:
        method = "Interpolation"
    else:
        method = "Nearest-Neighbor"

    title_str = (f'Glider: {glid}\n'
                 f'Deployed: {deployed}\n'
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
    dx = 2
    dy = 1.25
    # extent_inset = [lon_track.min(), lon_track.max(), lat_track.min(), lat_track.max()]
    extent_main = [lon_track.min() - dx, lon_track.max() + dx, lat_track.min() - dy, lat_track.max() + dy]
    # lonmin, lonmax, latmin, latmax = extent_inset

    # Create a map in the map axis     
    create(extent_main, ax=mpax)
    mpax.plot(lon_track, lat_track, '.-w', 
              markeredgecolor='black',
              markersize=3, 
              transform=configs.projection['data']
              )
    
    # Create inset axis for glider track
    # axin = map_add_inset(mpax, extent=extent_inset)
    # add_features(axin, extent_inset)
    # axin.plot(lon_track, lat_track, '.-w',
    #           markeredgecolor='black',
    #           markersize=5,
    #           transform=configs.projection['data']
    #           )

    h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

    lax.legend(h, l, ncol=1, loc='center', fontsize=13)
    lax.set_axis_off()

    # plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 

    plt.savefig(fullfile, dpi=configs.dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
       
active_gliders = gliders.glider.unique().tolist()

def main():    
    if parallel:
        import concurrent.futures
        workers = 6
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(plot_glider_profiles, active_gliders)
    else:
        for glider in active_gliders:
            plot_glider_profiles(glider)

if __name__ == "__main__":

    main()
