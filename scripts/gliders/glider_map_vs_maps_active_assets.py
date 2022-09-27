#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from ioos_model_comparisons.calc import find_nearest, convert_lon_360_to_180
from ioos_model_comparisons.plotting import map_add_ticks, map_add_features, map_add_bathymetry, export_fig
from ioos_model_comparisons.platforms import get_bathymetry, Argo, get_active_gliders
from ioos_model_comparisons.regions import region_config
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cmocean
from pathlib import Path
import os
from glob import glob
import pickle
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
path_ibtracs = (path_data / "ibtracs/IBTrACS.NA.v04r00.nc")
path_rtofs = (path_data / "rtofs")
path_gofs = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"

# Paths to save plots and figures
path_plot = (root_dir / "plots")
path_plot_maps = (path_plot / "mapfigs")
path_plot_script = (path_plot / script_name.replace(".py", "")) # create plot path

# Make sure above paths exist
os.makedirs(path_plot_maps, exist_ok=True)
os.makedirs(path_plot_script, exist_ok=True)

# User defined variables
storm = None # ibtracs_id for hurricane
freq = '1D' # interval of time for each plot
model = "rtofs" # Which model: rtofs or gofs
region = "gom"

# # 0m
# depth = 0
# temp_range = [22, 31, .25] # [min, max, step]
# haline_range = [34, 36.5, .1]

# 100m
depth = 100
temp_range = [12, 24, .5] # [min, max, step]
haline_range = [36, 36.9, .1]

# 200m
# depth = 200
# temp_range = [12, 24, .5] # [min, max, step]
# haline_range = [35.5, 37, .1] # [min, max, step]

t0 = dt.datetime(2021, 5, 1)
t1 = dt.datetime(2021, 12, 1)
subdir = "2021_hurricane_season"
dpi = 150

# Plot the following
bathy = True
argo = True
temp = False
salinity = True

path_plot_script = (path_plot_script / subdir)
region = region_config(region, model)

# Set cartopy information
extent = region["extent"]
region_name = region["name"]
projection_map = ccrs.Mercator()
projection_data = ccrs.PlateCarree()

# Convert date_s and date_e to strings
dstr = "%Y%m%d"
sstr = t0.strftime(dstr)
estr = t1.strftime(dstr)

# # Read glider data directly from the erddap server
pkl_g = f"gliders_{sstr}_{estr}_locations.pkl"

try:
    gliders = pd.read_pickle(path_gliders / pkl_g)
except FileNotFoundError:
    print(f"Downloading {sstr} to {estr} glider data from NGDAC")
    glider_vars = ['depth', 'latitude', 'longitude']
    gliders = get_active_gliders(extent, t0=t0, t1=t1, variables=glider_vars, parallel=True)
    # print(f"Downloading {glider} data from DAC")
    # glider_df = get_glider_by_id(dataset_id=glider)
    gliders.to_pickle(path_gliders / pkl_g)

ranges = pd.date_range(t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"), freq=freq)

# Read argo data from the erddap server
pkl_a = f"argo_{sstr}_{estr}.pkl"

try:
    argo_df = pd.read_pickle(path_argo / pkl_a)
except:
    print(f"Downloading {sstr} to {estr} ARGO data from ifremer.fr")

    #initialize connection to Argo server
    client = Argo()
    
    # Subset by lons -> lats -> times
    client.lons(extent[0], extent[1]).lats(extent[2], extent[3]).times(t0, t1)
    argo_df = client.get()
    argo_df = argo_df.to_pickle(path_argo / pkl_a)

# Keyword arguments for filled contour plot
kwargs = {}
kwargs['transform'] = projection_data
kwargs['extend'] = 'both'

def plot_map(sfig, ds, glider_trails, gliders_day, argo_locs):
    tds = ds
    var = tds.name
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
                    new_df['longitude'],
                    new_df['latitude'],
                    "-",
                    color="gray",
                    linewidth=4, 
                    transform=projection_data
                    )

        # Plot past day of each gliders track
        if not gliders_day.empty:
            for g, new_df in gliders_day.groupby(level=0):
                h = ax.plot(
                    new_df['longitude'],
                    new_df['latitude'],
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
        save_dir = path_plot_script / model / str(depth) / var 
        export_fig(save_dir, f"{model}_{t0.strftime(dfmt)}.png", dpi=dpi)
        plt.close()


if model == "gofs":
    # Load and subset GOFS data to the proper extents for each region
    ds = xr.open_dataset(path_gofs, drop_variables="tau")
    ds["lon"] = convert_lon_360_to_180(ds["lon"])
    ds = ds.sel(
        lon=slice(extent[0], extent[1]),
        lat=slice(extent[2], extent[3])
    )
    # ds['lon'] = ds['lon'] - 360  # Convert model lon to glider lon
elif model == "rtofs":
    # Load in RTOFS files locally
    rtofs_file_dates = []
    rtofs_file_paths = []
    for date in pd.date_range(t0, t1).to_list():
        tstr = date.strftime('rtofs.%Y%m%d')
        files = sorted(glob(os.path.join(path_rtofs, tstr, '*.nc')))
        for f in files:
            if f == '':
                continue
            else:
                date_list = f.split('rtofs/rtofs.')[1].split('/')
                rtofs_file_dates.append(pd.to_datetime(date_list[0]) + dt.timedelta(hours=int(date_list[1].split('_')[3].strip('f'))))
                rtofs_file_paths.append(f)

    ds = xr.open_mfdataset(
        rtofs_file_paths,
        concat_dim="MT",
        combine="nested",
        data_vars='minimal',
        coords='minimal',
        compat='override',
        parallel=True)
    ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat',
                    'MT': 'time', 'Depth': 'depth'})
    ds[["lon", "lat"]].load()

    # rtofs_df = pd.DataFrame(list(zip(rtofs_file_dates, rtofs_file_paths)), columns=['date', 'paths'])
    # rtofs_df.set_index('date', inplace=True)
    # with xr.open_dataset(rtofs_df['paths'][0]) as tds:
    xds = ds.isel(time=0)
    
    # Save rtofs lon and lat as variables to speed up indexing calculation
    rtofs_lon = xds.lon.values
    rtofs_lat = xds.lat.values

    # Find index of nearest lon and lat points
    _, lon1_ind = find_nearest(rtofs_lon[0, :], extent[0])
    _, lon2_ind = find_nearest(rtofs_lon[0, :], extent[1])

    _, lat1_ind = find_nearest(rtofs_lat[:, 0], extent[2])
    _, lat2_ind = find_nearest(rtofs_lat[:, 0], extent[3])

    rtofs_extent = [lon1_ind, lon2_ind, lat1_ind, lat2_ind]

    # subset the dataset to the extent of the region
    ds = ds.isel(
        X=slice(rtofs_extent[0], rtofs_extent[1]),
        Y=slice(rtofs_extent[2], rtofs_extent[3])
        ).squeeze()

# Create a map figure and serialize it if one doesn't already exist for this glider
region_name = "_".join(region["name"].split(' ')).lower()
sfig = (path_plot_maps / f"{region_name}_fig.pkl")

if not sfig.exists():
    if bathy:
        # Load bathymetry
        bathy = get_bathymetry(extent)
    
    # Create figure 
    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw=dict(projection=projection_map)
    )

    # Make the map pretty
    map_add_features(ax, extent)# zorder=0)
    map_add_bathymetry(ax, bathy, projection_data, [-1000, -100], zorder=1.5)
    map_add_ticks(ax, extent)

    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    with open(sfig, 'wb') as file:
        pickle.dump(fig, file)

dfmt = "%Y-%m-%d" # date format
tfmt = "%Y-%m-%d 00:00:00" # date format with time

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
    glider_trails = gliders.loc[pd.IndexSlice[:, t0week: t1.strftime(tfmt)], :]  # 2 weeks
    gliders_day = gliders.loc[pd.IndexSlice[:, t0:t1.strftime(tfmt)], :]  # 1 day

    # Subset argo erddap dataframe to current day and previous day 
    # (make each argo persist for 2 days)
    argo_day = argo_df.loc[pd.IndexSlice[:, tN.strftime(dfmt):t0.strftime(dfmt)], :]
    
    if model == "rtofs":
        # Load RTOFS file that corresponds to ctime
        tds = ds.sel(time=ctime, depth=depth).squeeze()
            # ds = xr.open_dataset(rtofs_df[rtofs_df.index==ctime]['paths'][0])
            # ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat',  
            #                 'MT': 'time', 'Depth': 'depth'})
            # # subset the dataset to the extent of the region
            # ds = ds.isel(
            #     X=slice(rtofs_extent[0], rtofs_extent[1]),
            #     Y=slice(rtofs_extent[2], rtofs_extent[3])
            #     ).squeeze()
    elif model == "gofs":
        tds = ds.sel(time=ctime, depth=depth).squeeze()
        
    if temp:
        xds = tds["temperature"].squeeze()
        plot_map(sfig, xds, glider_trails, gliders_day, argo_day)
   
    if salinity:
        xds = tds["salinity"].squeeze()
        plot_map(sfig, xds, glider_trails, gliders_day, argo_day)
