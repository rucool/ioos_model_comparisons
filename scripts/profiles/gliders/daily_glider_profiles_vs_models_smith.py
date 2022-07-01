#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import os
import matplotlib.patheffects as path_effects
from hurricanes.plotting import map_add_features, map_create, map_add_inset
import hurricanes.configs as configs
from hurricanes.models import gofs, rtofs, cmems
from hurricanes.platforms import get_active_gliders
from hurricanes.calc import lon180to360, lon360to180, density

# %%
# set path to save plots
path_save = (configs.path_plots / "profiles" / "gliders")

# lat and lon bounds
extent = [-110.0, -10.0, 5.0, 45.0]
days = 5
today = dt.date.today()
interp = False

# Map Configs
dx = 1.75
dy = 1.25

# Time threshold (in hours). If a profile time is greater than this, we won't 
# grab the corresponding profile from the model
time_threshold = 6 # hours

# Create a date list.
date_list = [today + dt.timedelta(days=x+1) for x in range(days)]
date_list.insert(0, today)

# Get time bounds for the current day
t0 = date_list[0]
t1 = date_list[1]

spath = path_save / str(today.year) / today.strftime('%m-%d')
os.makedirs(spath, exist_ok=True)

# %% Look for datasets in IOOS glider dac
vars = ['time', 'latitude', 'longitude', 'depth', 'temperature', 'salinity',
        'density', 'profile_id']
gliders = get_active_gliders(extent, t0, t1, variables=vars, parallel=True).reset_index()

# Read GOFS 3.1 output
gofs = gofs(rename=True).sel(depth=slice(0,400))

# Read RTOFS grid and time
rtofs = rtofs().sel(depth=slice(0,400))

# Setting RTOFS lon and lat to their own variables speeds up the script
rlon = rtofs.lon.data[0, :]
rlat = rtofs.lat.data[:, 0]
rx = rtofs.x.data
ry = rtofs.y.data

# Read Copernicus
cmems = cmems(rename=True).sel(depth=slice(0,400))

# Convert time threshold to a Timedelta so that we can compare timedeltas.
time_threshold= pd.Timedelta(hours=time_threshold) 

# Legend labels
glabel = f'GOFS'
rlabel = f'RTOFS'
clabel = f"CMEMS"

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

# Looping through all gliders found
for id, df in gliders.groupby('glider'):
    print('Plotting ' + id)
    alabel = f'{id}'
    fullfile = spath / f"{id}_{t0.strftime('%Y%m%d')}_to_{t1.strftime('%Y%m%d')}.png"

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
    for name, pdf in df.groupby(['profile_id', 'time', 'lon', 'lat']):
        pid = name[0]
        time_glider = name[1] 
        lon_glider = name[2].round(2)
        lat_glider = name[3].round(2)
        lon_track.append(lon_glider)
        lat_track.append(lat_glider)
        
        print(f"Glider: {id}, Profile ID: {pid}, Time: {time_glider}")
        
        # Filter salinity and temperature that are more than 3 standard deviations
        # from the mean
        try:
            pdf = pdf[np.abs(stats.zscore(df['salinity'])) < 3]  #  salinity
            pdf = pdf[np.abs(stats.zscore(df['temperature'])) < 3]  #  temperature
        except KeyError:
            pass

        # Save as Pd.Series for easier recalling of columns 
        depth_glider = pdf['depth']
        temp_glider = pdf['temperature']
        salinity_glider = pdf['salinity']
        density_glider = pdf['density']

        # delta_z = 0.3 # fraction
        # zn = int(np.round(np.nanmax(dg) / delta_z)) # expand the depth by dz
        # from gsw import p_from_z, z_from_p
        # import seawater
        
        # Calculate density for gofs profile
        # gdsi["pressure"] = xr.apply_ufunc(seawater.eos80.pres, gdsi.depth, gdsi.lat)
        # gdsi["density"] = xr.apply_ufunc(seawater.eos80.dens, gdsi.salinity, gdsi.temperature, gdsi.pressure)
        
        # Extract profiles from operational models
        # pdf['pressure_sea'] = seawater.eos80.pres(pdf['depth'].values, pdf['lat'])
        # pdf['density_sea'] = seawater.eos80.dens(pdf['salinity'].values, pdf['temperature'].values, pdf['pressure_sea'].values)

        
        # pdf['pressure_gsw'] = p_from_z(-pdf['depth'].values,pdf['lat'].values)
        # pdf['depth_gsw'] = z_from_p(pdf['pressure_gsw'].values, pdf['lat'].values)
        # pdf['density_gsw'] = density(pdf['temperature'].values, -pdf['depth'].values, pdf['salinity'].values, pdf['lat'].values, pdf['lon'].values)
        
        # GOFS3.1
        # Convert glider lon from -180,180 to 0,359
        lon_glider_gofs = lon180to360(lon_glider)

        # Select the nearest model time to the glider time for this profile
        gds = gofs.sel(time=time_glider, method="nearest")

        # Interpolate the model to the nearest point
        if interp:
            gds = gds.interp(
                lon=lon_glider_gofs,
                lat=lat_glider,
                )
        else:
            # select nearest neighbor grid point
            gds = gds.sel(
                lon=lon_glider_gofs,
                lat=lat_glider,
                method="nearest"
            )
        
        # Convert lon from 0,259 to -180,180
        gds['lon'] = lon360to180(gds['lon'])

        # Calculate density
        gds['density'] = density(gds['temperature'].values, -gds['depth'].values, gds['salinity'].values, gds['lat'].values, gds['lon'].values)

        print(f"GOFS - Time: {pd.to_datetime(gds.time.values)}")

        # RTOFS
        rds = rtofs.sel(time=time_glider, method="nearest")
        print(f"RTOFS - Time: {pd.to_datetime(rds.time.values)}")

        # interpolating lon and lat to x and y index of the rtofs grid
        rlonI = np.interp(lon_glider, rlon, rx) # lon -> x
        rlatI = np.interp(lat_glider, rlat, ry) # lat -> y

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

        # CMEMS
        cds = cmems.sel(time=time_glider, method="nearest")
        print(f"CMEMS - Time: {pd.to_datetime(cds.time.values)}")
        delta_time = np.abs(time_glider - pd.to_datetime(cds.time.values))
        # print(f"Threshold time: {delta_time}")
        # if delta_time < time_threshold:
        #     print(f"Difference between profile and nearest CMEMS time is {delta_time}. Interpolating to profile")

        if interp:
            cds = cds.interp(
                lon=lon_glider,
                lat=lat_glider
            )
        else:
            cds = cds.sel(
                lon=lon_glider,
                lat=lat_glider,
                method='nearest'
            )
        # Calculate density
        cds['density'] = density(cds['temperature'].values, -cds['depth'].values, cds['salinity'].values, cds['lat'].values, cds['lon'].values)
        
        # Temperature profile
        tax.plot(temp_glider, depth_glider, '.-', color='cyan', label='_nolegend_')
        tax.plot(rds['temperature'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')
        tax.plot(gds['temperature'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
        tax.plot(cds['temperature'], cds["depth"], '.-', color="magenta", label='_nolegend_')
        
        # Salinity profile
        sax.plot(salinity_glider, depth_glider, '.-', color='cyan', label='_nolegend_')
        sax.plot(rds['salinity'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')
        sax.plot(gds['salinity'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
        sax.plot(cds['salinity'], cds["depth"], '.-', color="magenta", label='_nolegend_')

        # Density profile
        dax.plot(density_glider, depth_glider, '.-', color='cyan', label='_nolegend_')
        dax.plot(rds['density'], rds['depth'], '.-', color='lightcoral', label='_nolegend_')
        dax.plot(gds['density'], gds["depth"], '.-', color="mediumseagreen", label='_nolegend_')
        dax.plot(cds['density'], cds["depth"], '.-', color="magenta", label='_nolegend_')

    # Temperature profile
    tax.plot(temp_glider, depth_glider, '-o', color='blue', label=alabel)
    tax.plot(rds['temperature'], rds['depth'], '-o', color='red', label=rlabel)
    tax.plot(gds['temperature'], gds["depth"], '-o', color="green", label=glabel)
    tax.plot(cds['temperature'], cds["depth"], '-o', color="purple", label=clabel)
    
    # Salinity profile
    sax.plot(salinity_glider, depth_glider, '-o', color='blue', label=alabel)
    sax.plot(rds['salinity'], rds['depth'], '-o', color='red', label=rlabel)
    sax.plot(gds['salinity'], gds["depth"], '-o', color="green", label=glabel)
    sax.plot(cds['salinity'], cds["depth"], '-o', color="purple", label=clabel)

    # Density profile
    dax.plot(density_glider, depth_glider, '-o', color='blue', label=alabel)
    dax.plot(rds['density'], rds['depth'], '-o', color='red', label=rlabel)
    dax.plot(gds['density'], gds["depth"], '-o', color="green", label=glabel)
    dax.plot(cds['density'], cds["depth"], '-o', color="purple", label=clabel)

    # Get min and max of each plot. Add a delta to each for x limits
    tmin, tmax = line_limits(tax, delta=.5)
    smin, smax = line_limits(sax, delta=.25)
    dmin, dmax = line_limits(dax, delta=.5)
    
    # Adjust plots
    tax.set_xlim([tmin, tmax])
    tax.set_ylim([400, 1])
    tax.set_ylabel('Depth (m)', fontsize=13, fontweight="bold")
    tax.set_xlabel('Temperature ($^oC$)', fontsize=13, fontweight="bold")
    tax.grid(True, linestyle='--', linewidth=0.5)

    sax.set_xlim([smin, smax])
    sax.set_ylim([400, 1])  
    sax.set_xlabel('Salinity', fontsize=13, fontweight="bold")
    sax.grid(True, linestyle='--', linewidth=0.5)

    dax.set_xlim([dmin, dmax])
    dax.set_ylim([400, 1])
    dax.set_xlabel('Density (kg m-3)', fontsize=13, fontweight="bold")
    dax.grid(True, linestyle='--', linewidth=0.5)

    if interp:
        method = "Interpolation"
    else:
        method = "Nearest-Neighbor"
        
    title_str = (f'Glider {id}\n'
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
    extent_inset = [lon_track.min(), lon_track.max(), lat_track.min(), lat_track.max()]
    extent_main = [lon_track.min() - dx, lon_track.max() + dx, lat_track.min() - dy, lat_track.max() + dy]
    lonmin, lonmax, latmin, latmax = extent_inset

    # Create a map in the map axis  
    map_create(extent_main, ax=mpax)
    mpax.plot(lon_track, lat_track, '.-w', 
              markeredgecolor='black',
              markersize=3, 
              transform=configs.projection['data']
              )

    # Create inset axis for glider track
    axin = map_add_inset(mpax, extent=extent_inset)
    map_add_features(axin, extent_inset)
    axin.plot(lon_track, lat_track, '.-w',
              markeredgecolor='black',
              markersize=5,
              transform=configs.projection['data']
              )

    h, l = sax.get_legend_handles_labels()  # get labels and handles from ax1

    lax.legend(h, l, ncol=1, loc='center', fontsize=13)
    lax.set_axis_off()

    # plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 

    plt.savefig(fullfile, dpi=configs.dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()    