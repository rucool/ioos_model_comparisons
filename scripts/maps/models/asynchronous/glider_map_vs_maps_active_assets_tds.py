#!/usr/bin/env python

import datetime as dt
import os
import pickle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Reader
from ioos_model_comparisons.calc import find_nearest, lon180to360, lon360to180
from ioos_model_comparisons.models import gofs, rtofs
from ioos_model_comparisons.platforms import (get_active_gliders, get_argo_floats_by_time,
                                  get_bathymetry)
from ioos_model_comparisons.plotting import (export_fig, map_add_bathymetry,
                                 map_add_features, map_add_ticks)
from ioos_model_comparisons.regions import region_config
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_gliders = (path_data / "gliders")
path_argo = (path_data / "argo")

# Paths to save plots and figures
path_plot = (root_dir / "plots")
path_plot_script = (path_plot / script_name.replace(".py", "")) # create plot path
path_plot_maps = (path_plot_script / "mapfigs")


# Make sure above paths exist
os.makedirs(path_plot_maps, exist_ok=True)
os.makedirs(path_plot_script, exist_ok=True)

# User defined variables
# models = ["rtofs", "gofs"] # Which model: rtofs or gofs.. or both?
models = ["gofs"] # Which model: rtofs or gofs.. or both?
t0 = dt.datetime(2021, 5, 1)
t1 = dt.datetime(2021, 12, 1)
# t0 = dt.datetime(2021, 8, 1)
# t1 = dt.datetime(2021, 8, 7)
freq = '1D' # time interval for each plot
dpi = 150 # dots per inch for savefig resolution

# Cartopy mapping arguments
projection_data = ccrs.PlateCarree() # Projection the data is in
projection_map = ccrs.Mercator() # Projection to plot the map in

# Plot the following: True or False
temp = True 
salinity = False 
bathy = True
argo = True
eez = False

# Rewrite images? If the script failed for some reason and you want to restart
# from where you left off, set this to False. Otherwise, set it to True to
# rewrite any plots
overwrite = True

# GOM
# depth = 0
# temp_range = [22, 31, .25] # [min, max, step]
# haline_range = [34, 36.5, .1]
# depth = 100
# temp_range = [17, 28, .5] # [min, max, step]
# haline_range = [36, 36.9, .1]
# depth = 200
# temp_range = [12, 24, .5] # [min, max, step]
# haline_range = [35.5, 37, .1] # [min, max, step]

# Caribbean
# region_key = "caribbean" # specify the region in order to grab the extents of that region
# subdir = "2021_hurricane_season_caribbean"
# depth = 0
# temp_range = [24, 30, .5] # [min, max, step]
# haline_range = [34.6 , 37, .1]
# depth = 100
# temp_range = [17, 28, 1] # [min, max, step]
# haline_range = [36, 37.3, .1]
# depth = 150
# temp_range = [17, 26, .5] # [min, max, step]
# haline_range = [36, 37.6, .1]
# depth = 200
# temp_range = [14, 24, 1] # [min, max, step]
# haline_range = [35.5, 37, .1] # [min, max, step]

# NG645 GoM/SAB
# depth = 0
# temp_range = [24, 31, .5] # [min, max, step]
# haline_range = [35.8, 36.8, .05]
# depth = 100
# temp_range = [17, 28, .5] # [min, max, step]
# haline_range = [36, 36.9, .1]
# depth = 200
# temp_range = [12, 24, .5] # [min, max, step]
# haline_range = [35.5, 37, .1] # [min, max, step]

# Caribbean PR/VI
region_key = "prvi" # specify the region in order to grab the extents of that region
subdir = "puerto_rico-us_virgin_islands"
# depth = 0
# temp_range = [24, 30, .5] # [min, max, step]
# haline_range = [36 , 37, .05]
depth = 100
temp_range = [20, 28, .5] # [min, max, step]
haline_range = [36, 37.2, .1]
# depth = 150
# temp_range = [18.5, 26, .5] # [min, max, step]
# haline_range = [36.1, 37.2, .1]
# depth = 200
# temp_range = [14, 24, 1] # [min, max, step]
# haline_range = [35.5, 37, .1] # [min, max, step]

# South Atlantic Bight
# depth = 0
# temp_range = [21, 30.5, .5] # [min, max, step]
# haline_range = [35.3 , 36.9, .1]
# depth = 150
# temp_range = [16.5, 25.5, .5] # [min, max, step]
# haline_range = [36, 37, .1]
# depth = 200
# temp_range = [13, 23, 1] # [min, max, step]
# haline_range = [35.7, 36.9, .1] # [min, max, step]

# Mid Atlantic Bight
# depth = 0
# temp_range = [11, 30, 1] # [min, max, step]
# haline_range = [31.25, 36.5, .25]
# depth = 30
# temp_range = [9, 29, 1] # [min, max, step]
# haline_range = [32, 36.75, .25]
# depth = 200
# temp_range = [10, 23, 1] # [min, max, step]
# haline_range = [35.5, 36.8, .1] # [min, max, step]

 
###############################################################################
# Don't really need to change anything from below this block until the end
###############################################################################
def plot_map(sfig, ds, glider_trails=None, gliders_day=None, argo_locs=None):
    tds = ds
    var = tds.name

    # Create save file name
    save_dir = path_plot_script / model / str(depth) / var
    save_file = f"{model}_{t0.strftime(dfmt)}.png"

    if Path(save_dir / save_file).is_file():
        if not overwrite:
            return
            
    with open(sfig, "rb") as file:
        fig = pickle.load(file)
        ax = fig.axes[0]

        # Create filled contour of dataarray
        if var == 'salinity':
            kwargs['cmap'] = cmocean.cm.haline
            kwargs['vmin'] = haline_range[0]
            kwargs['vmax'] = haline_range[1]
            kwargs['levels'] = np.arange(kwargs['vmin'], kwargs['vmax'], haline_range[2])
        elif var == 'temperature':
            kwargs['cmap'] = cmocean.cm.thermal
            kwargs['vmin'] = temp_range[0]
            kwargs['vmax'] = temp_range[1]
            kwargs['levels'] = np.arange(kwargs['vmin'], kwargs['vmax'], temp_range[2])
            
        h2 = ax.contourf(tds['lon'], tds['lat'], tds, **kwargs)

        # Make some space for a colorbar axes
        axins = inset_axes(ax,  # here using axis of the lowest plot
            width="2.5%",  # width = 5% of parent_bbox width
            height="100%",  # height : 340% good for a (4x4) Grid
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
            )

        # Create a colorbar of the contourf
        cb = fig.colorbar(h2, cax=axins)

        # Adjust colorbar tick parameters and create title
        cb.ax.tick_params(labelsize=12)
        cb.set_label(var.title(), fontsize=13)

        # Plot last 2 weeks of each gliders track
        if not glider_trails.empty:
            for g, new_df in glider_trails.groupby(level=0):
                h = ax.plot(
                    new_df['lon'],
                    new_df['lat'],
                    "-",
                    color="gray",
                    linewidth=4, 
                    transform=projection_data
                    )

        # Plot past day of each gliders track
        if not gliders_day.empty:
            for g, new_df in gliders_day.groupby(level=0):
                h = ax.plot(
                    new_df['lon'],
                    new_df['lat'],
                    "-",
                    color="white",
                    linewidth=5, 
                    transform=projection_data
                    )
                
        # Plot argo floats
        if not argo_locs.empty:
            h = ax.plot(argo_locs["lon"], argo_locs["lat"],
                        "o",
                        color="limegreen",
                        markeredgecolor="black",
                        markersize=8,
                        transform=projection_data)
        ax.grid(True, color='k', linestyle='--', alpha=.5, linewidth=.5)

        # Add title
        ax.set_title(
            f"{model.upper()} - {t0.strftime(dfmt)} - {var.title()} ({depth} m)",
            fontsize=18,
            fontweight="bold",
            )

        # Save figure
        export_fig(save_dir, save_file, dpi=dpi)
        plt.close()

path_plot_script = (path_plot_script / subdir)
region = region_config(region_key)
extent = region["extent"]
region_name = region["name"]

# Convert date_s and date_e to strings
dstr = "%Y%m%d"
sstr = t0.strftime(dstr)
estr = t1.strftime(dstr)

# # Read glider data directly from the erddap server
pkl_g = f"gliders_{subdir}_{sstr}_{estr}_locations.pkl"

try:
    gliders = pd.read_pickle(path_gliders / pkl_g)
except FileNotFoundError:
    print(f"Downloading {sstr} to {estr} glider data from NGDAC")
    glider_vars = ['depth', 'latitude', 'longitude']
    gliders = get_active_gliders(extent, t0=t0, t1=t1, variables=glider_vars, parallel=False)
    # print(f"Downloading {glider} data from DAC")
    # glider_df = get_glider_by_id(dataset_id=glider)
    gliders.to_pickle(path_gliders / pkl_g)

ranges = pd.date_range(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"), freq=freq)

# Read argo data from the erddap server
pkl_a = f"argo_{subdir}_{sstr}_{estr}.pkl"

try:
    argo_df = pd.read_pickle(path_argo / pkl_a)
except:
    print(f"Downloading {sstr} to {estr} ARGO data from ifremer.fr")
    argo_df = get_argo_floats_by_time(extent, t0, t1)
    argo_df.to_pickle(path_argo / pkl_a)

# Keyword arguments for filled contour plot
kwargs = {}
kwargs['transform'] = projection_data
kwargs['extend'] = 'both'

# Create a map figure and serialize it if one doesn't already exist for this glider
region_name = "_".join(region["name"].split(' ')).lower()

if eez:
    eez_file = path_data / "eez/World_EEZ_v11_20191118/eez_boundaries_v11.shp"
    shape_feature = cfeature.ShapelyFeature(
        Reader(eez_file).geometries(),
        ccrs.PlateCarree(), edgecolor='grey', facecolor='none'
        )
    sfig = (path_plot_maps / f"{region_key}_eez_fig.pkl")
else:
    sfig = (path_plot_maps / f"{region_key}_fig.pkl")

if not sfig.exists():    
    # Create figure 
    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw=dict(projection=projection_map)
    )

    # Make the map pretty
    map_add_features(ax, extent)# zorder=0)
    if bathy:
        # Load bathymetry
        bathy = get_bathymetry(extent)
        map_add_bathymetry(ax,
                        bathy['longitude'],
                        bathy['latitude'],
                        bathy['elevation'], 
                        [-1000, -100], 
                        zorder=1.5)
    map_add_ticks(ax, extent)
    
    if eez:
        ax.add_feature(shape_feature, zorder=10)

    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    with open(sfig, 'wb') as file:
        pickle.dump(fig, file)
        
dfmt = "%Y-%m-%d" # date format
tfmt = "%Y-%m-%d 00:00:00" # date format with time

for model in models:
    if model == "gofs":
        # Load and subset GOFS data to the proper extents for each region
        ds = gofs(rename=True)
        glons = lon180to360(extent[:2])
        ds = ds.sel(
            lon=slice(glons[0]-1, glons[1]+1),
            lat=slice(extent[2]-1, extent[3]+1)
        )
        ds["lon"] = lon360to180(ds["lon"])
    elif model == "rtofs":
        ds = rtofs().set_coords(['lon', 'lat'])
        xds = ds.isel(time=0)
        
        # Save rtofs lon and lat as variables to speed up indexing calculation
        rtofs_lon = xds.lon.values
        rtofs_lat = xds.lat.values
        xds.close()

        # Find index of nearest lon and lat points
        _, lon1_ind = find_nearest(rtofs_lon[0, :], extent[0]-1)
        _, lon2_ind = find_nearest(rtofs_lon[0, :], extent[1]+1)

        _, lat1_ind = find_nearest(rtofs_lat[:, 0], extent[2]-1)
        _, lat2_ind = find_nearest(rtofs_lat[:, 0], extent[3]+1)

        rtofs_extent = [lon1_ind, lon2_ind, lat1_ind, lat2_ind]

        # subset the dataset to the extent of the region
        ds = ds.isel(
            x=slice(rtofs_extent[0], rtofs_extent[1]),
            y=slice(rtofs_extent[2], rtofs_extent[3])
            ).squeeze()

    # Enumerate through date ranges starting with the second value in ranges
    # Start at the second value so we don't error out on the very last item in the ranges
    for index, value in enumerate(ranges[1:], start=1):
        # Save time range as variables
        t1 = ranges[index]
        t0 = ranges[index-1]
        print(f"{index-1} to {index}, {t0} to {t1}")

        # Calculate t0week and t1week for glider and argo trails to persist
        t0week = (t0 - dt.timedelta(days=14))
        tN = (t0 - dt.timedelta(days=1))

        # Get center time range to plot the days model run at 12Z
        ctime = (pd.to_datetime(t0) + dt.timedelta(hours=12))
            
        # Subset glider erddap dataframe
        if not gliders.empty:
            glider_trails = gliders.loc[pd.IndexSlice[:, t0week: t1.strftime(tfmt)], :]  # 2 weeks
            gliders_day = gliders.loc[pd.IndexSlice[:, t0:t1.strftime(tfmt)], :]  # 1 day
        else:
            glider_trails = pd.DataFrame()
            gliders_day = pd.DataFrame()

        # Subset argo erddap dataframe to current day and previous day 
        # (make each argo persist for 2 days)
        if not argo_df.empty:
            argo_day = argo_df.loc[pd.IndexSlice[:, tN.strftime(dfmt):t0.strftime(dfmt)], :]
        else:
            argo_day = pd.DataFrame()

        # Select 12Z of this day to plot the surface map
        try:
            tds = ds.sel(time=ctime, depth=depth).squeeze()
        except KeyError:
            continue
            
        if temp:
            xds = tds["temperature"].squeeze() 
            plot_map(sfig, xds, glider_trails, gliders_day, argo_day)

        if salinity:
            xds = tds["salinity"].squeeze()
            plot_map(sfig, xds, glider_trails, gliders_day, argo_day)
