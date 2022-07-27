#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from hurricanes.calc import find_nearest, lon180to360, lon360to180
from hurricanes.plotting import map_add_ticks, map_add_features, map_add_bathymetry, export_fig
from hurricanes.platforms import get_glider_by_id, get_bathymetry, get_argo_floats_by_time
from hurricanes.models import gofs, rtofs
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cmocean
from pathlib import Path
import os
from glob import glob
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

matplotlib.use('agg')

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

# Paths to save plots and figures
path_plot = (root_dir / "plots" / script_name) # create plot path
path_plot_maps = (path_plot / "mapfigs")

os.makedirs(path_plot_maps, exist_ok=True)

# User defined variables
storm_id = 2281 # ibtracs_id for hurricane
freq = '1D' # interval of time for each plot
model = "rtofs" # Which model: rtofs or gofs

# Plot the following
bathy = True
argo = False
plot_temperature = False
plot_salinity = True
replot = True

# Plot information
extent = [-75.5, -65, 34, 42] # cartopy extent format
glider = "ng738-20211002T0000"

# temp_range = [12, 21.5, .5] # [min, max, step]
# haline_range = [35.3, 36.7, .1] # [min, max, step]
# depth = 200 

# temp_range = [13, 26, 1] # [min, max, step]
# haline_range = [35.3, 36.7, .1] # [min, max, step]
# depth = 100 

# temp_range = [13, 29, 1] # [min, max, step]
# haline_range = [35, 36.6, .1] # [min, max, step]
# depth = 50

temp_range = [12, 29, 1] # [min, max, step]
haline_range = [34.4, 36.6, .1] # [min, max, step]
depth = 0

# Set cartopy information
projection_map = ccrs.Mercator()
projection_data = ccrs.PlateCarree()

# Read glider data directly from the erddap server
pkl_g = f"{glider}_data.pkl"

try:
    glider_df = pd.read_pickle(path_gliders / pkl_g)
except FileNotFoundError:
    print(f"Downloading {glider} data from DAC")
    glider_df = get_glider_by_id(dataset_id=glider)
    glider_df = glider_df.to_pickle(path_gliders / pkl_g)

# Create lon and lat variables for the entire glider track
glider_lon = glider_df['longitude (degrees_east)']
glider_lat = glider_df['latitude (degrees_north)']

# Create date range from start date to end date of glider data
date_s = glider_df.index[0].floor(freq="1D")
date_e = glider_df.index[-1].ceil(freq='1D')

# center_time = dt.datetime(2021, 8, 29)
# start_date = center_time - dt.timedelta(days=7)
# end_date = center_time + dt.timedelta(days=7)

# Convert date_s and date_e to strings
dstr = "%Y%m%d"
sstr = date_s.strftime(dstr)
estr = date_e.strftime(dstr)

ranges = pd.date_range(date_s.strftime("%Y-%m-%d"), date_e.strftime("%Y-%m-%d"), freq=freq)

if argo:
    # Read argo data from the erddap server
    pkl_a = f"argo_floats_{date_s}_{date_e}.pkl"
    
    try:
        argo_df = pd.read_pickle(path_argo / pkl_a)
    except:
        print(f"Downloading {date_s} to {date_e} ARGO data from ifremer.fr")
        argo_df = get_argo_floats_by_time(extent, date_s, date_e)
        argo_df = argo_df.to_pickle(path_argo / pkl_a)

# read in ibtracs hurricane packs and convert to dataframe
if storm_id:
    hurricane = xr.open_dataset(path_ibtracs)
    time_h = pd.to_datetime(hurricane.time[storm_id,:])
    lon_h = hurricane.lon[storm_id,:]
    lat_h = hurricane.lat[storm_id,:]
    cat_h = hurricane.usa_sshs[storm_id,:]
    storm_df=pd.DataFrame(dict(time=time_h, lon=lon_h, lat=lat_h, cat=cat_h))
    storm_df.set_index("time", inplace=True)

    colors_map = {
        "nan": "white",
        "-5.0": "cyan",
        "-4.0":  "cyan",
        "-3.0":  "cyan",
        "-2.0":  "cyan",
        "-1.0":  "cyan",
        "0.0": "cyan",
        "1.0": "yellow",
        "2.0": "gold",
        "3.0": "navajowhite",
        "4.0": "darkorange",
        "5.0": "red",
    }

    size_map = {
        "nan": 1,
        "-5.0": 5,
        "-4.0": 5,
        "-3.0": 5,
        "-2.0": 5,
        "-1.0": 5,
        "0.0": 5,
        "1.0": 6,
        "2.0": 7,
        "3.0": 8,
        "4.0": 9,
        "5.0": 10,
    }


    # wrangle hurricane category data so that we can map it to color and size
    storm_df["cat"] = storm_df["cat"].astype(str)
    storm_df["colors"] = storm_df["cat"].map(colors_map)
    storm_df["size"] = storm_df["cat"].map(size_map)*15
else:
    storm_df = pd.DataFrame()

# dt_h = pd.to_datetime(time_h.data[0])
# name = ds.name[number]
# ida_wind_ib = ds.usa_wind[ida_storm_number,:]
# ida_press_ib = ds.usa_pres[ida_storm_number,:]
# ida_rmw_ib = ds.usa_rmw[ida_storm_number,:]

# Contour arguments
cargs = {}
cargs['transform'] = projection_data
cargs['extend'] = "both"

targs = {}
targs['cmap'] = cmocean.cm.thermal
targs['vmin'] = temp_range[0]
targs['vmax'] = temp_range[1]
targs['levels'] = np.arange(temp_range[0], temp_range[1], temp_range[2])

hargs = {}
hargs['cmap'] = cmocean.cm.haline
hargs['vmin'] = haline_range[0]
hargs['vmax'] = haline_range[1]
hargs['levels'] = np.arange(haline_range[0], haline_range[1], haline_range[2])

plot_vars = []
if plot_temperature:
    plot_vars.append('temperature')

if plot_salinity:
    plot_vars.append('salinity')
    
if bathy:
    # Load bathymetry
    bathy = get_bathymetry(extent)

if model == "gofs":
    # Load and subset GOFS data to the proper extents for each region
    ds = gofs(rename=True).sel(depth=depth)
    lon360 = lon180to360(extent[:2])
    ds = ds.sel(
        lon=slice(lon360[0], lon360[1]),
        lat=slice(extent[2], extent[3])
    )
    ds['lon'] = ds['lon'] - 360  # Convert model lon to glider lon
elif model == "rtofs":
    # Load RTOFS data
    ds = rtofs().sel(depth=depth)
    
    # Save rtofs lon and lat as variables to speed up indexing calculation
    rtofs_lon = ds.lon.values
    rtofs_lat = ds.lat.values

    # Find index of nearest lon and lat points
    _, lon1_ind = find_nearest(rtofs_lon[0, :], extent[0])
    _, lon2_ind = find_nearest(rtofs_lon[0, :], extent[1])

    _, lat1_ind = find_nearest(rtofs_lat[:, 0], extent[2])
    _, lat2_ind = find_nearest(rtofs_lat[:, 0], extent[3])

    rtofs_extent = [lon1_ind, lon2_ind, lat1_ind, lat2_ind]
    
    # subset the dataset to the extent of the region
    ds = ds.isel(
        x=slice(rtofs_extent[0], rtofs_extent[1]),
        y=slice(rtofs_extent[2], rtofs_extent[3])
        ).squeeze()
    
# # Select depth we are plotting
# ds = ds.sel(depth=depth)

# Create a map figure and serialize it if one doesn't already exist for this glider
sfig = (path_plot_maps / f"{glider}_fig.pkl")

if not sfig.exists():
    # Create figure 
    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw=dict(projection=projection_map)
    )

    # Make the map pretty
    map_add_features(ax, extent)# zorder=0)

    map_add_bathymetry(
        ax, bathy['longitude'], bathy['latitude'], bathy['elevation'], 
        levels=[-1000, -100], zorder=1.5)
    map_add_ticks(ax, extent)

    ax.plot(glider_lon, glider_lat,
            '-',
            linewidth=4,
            color='gray',
            transform=projection_data,
            # zorder=6
            )
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    with open(sfig, 'wb') as file:
        pickle.dump(fig, file)

for index, value in enumerate(ranges[1:], start=1):    
    # Save time range as variables
    t0 = ranges[index-1].strftime("%Y-%m-%d")
    t1 = ranges[index].strftime("%Y-%m-%d 00:00:00")
    t1s = ranges[index].strftime("%Y-%m-%d")
    print(f"{index-1} to {index}, {t0} to {t1}")
    ctime = pd.to_datetime(t0) + dt.timedelta(hours=12) # Center of time range
    datefmt = f"{t0}_to_{t1s}" # String for date range in filename

    if not storm_df.empty:
        tstorm = storm_df[:1]
        tstorm_day = storm_df[t0:t1]
        
    # Subset glider erddap dataframe
    glider_df_temp = glider_df[t0:t1]
    lon_gl = glider_df_temp['longitude (degrees_east)']
    lat_gl = glider_df_temp['latitude (degrees_north)']

    if argo:
        # Subset argo erddap dataframe
        argo_df_temp = argo_df.loc[pd.IndexSlice[:, t0:t1], :]

    with open(sfig, 'rb') as file:
        fig = pickle.load(file)
        ax = fig.axes[0]
    
    for v in plot_vars:
        # Create directory to save plots
        path_temp = path_plot / model / glider / f"{v}_{depth}m" 
        os.makedirs(path_temp, exist_ok=True)
        sname = f"{glider}_track_{v}_{depth}m_{datefmt}.png"

        if (path_temp / sname).is_file():
            if not replot:
                continue
    
        # with open(sfig, 'rb') as file:
        #     fig = pickle.load(file)
        #     ax = fig.axes[0]

        # Select variable in model
        try:
            tds = ds[v].sel(time=ctime).squeeze()
        except KeyError:
            continue

        # Create kwargs dict based off the variable being plotted.
        kwargs = {}
        kwargs.update(cargs)
        if v == 'temperature':
            kwargs.update(targs)
        elif v == 'salinity':
            kwargs.update(hargs)
            
        # Contour plot of the variable
        h1 = ax.contourf(tds['lon'], tds['lat'], tds.squeeze(), **kwargs)
        axins = inset_axes(ax,  # here using axis of the lowest plot
            width="2.5%",  # width = 5% of parent_bbox width
            height="100%",  # height : 340% good for a (4x4) Grid
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
            )
        cb = fig.colorbar(h1, cax=axins)
        cb.ax.tick_params(labelsize=12)
        # cb.set_label("Temperature (deg C)", fontsize=13)
        cb.set_label(f"{tds.name.title()} ({tds.units})", fontsize=13)

        # Plot subsetted glider track
        h3 = ax.plot(lon_gl, lat_gl,
            'w-',
            linewidth=5,
            transform=projection_data)

        if argo:
            h4 = ax.plot(argo_df_temp['lon'], argo_df_temp['lat'],
                        "o",
                        color="limegreen",
                        markeredgecolor="black",
                        markersize=6,
                        transform=projection_data)

        if not storm_df.empty:
            bt = storm_df.index.min() - dt.timedelta(days=5)
            et = storm_df.index.max() + dt.timedelta(days=5)

            if (pd.to_datetime(t0) > bt) & (pd.to_datetime(t1) < et):
                # Plot hurricanes (until that day)
                track_1 = ax.plot(storm_df["lon"], storm_df["lat"],
                                '-',
                                linewidth=2.5,
                                color="black",
                                # markersize=8, 
                                # # markerfillcolor="red",
                                # markeredgecolor="black", 
                                transform=projection_data, 
                                zorder=20)
                
                markers = ax.scatter(tstorm["lon"], tstorm["lat"],
                                    c="red", s=8, marker="o",
                                    transform=projection_data, zorder=20)

                # Plot hurricanes (track of that day)
                track_2 = ax.plot(tstorm_day["lon"], tstorm_day["lat"],
                                '-',
                                color="red",
                                linewidth=2.5,
                                transform=projection_data, 
                                zorder=20)
                
                markers = ax.scatter(tstorm_day["lon"], tstorm_day["lat"],
                                        c=tstorm_day["colors"], 
                                        s=tstorm_day["size"],
                                        marker='o',
                                        edgecolors="black",
                                        transform=projection_data,  zorder=20)
            
        ax.grid(True, color='k', linestyle='--', alpha=.5, linewidth=.5)

        # Add title
        ax.set_title(f"{model.upper()} - {v.title()} ({depth}m) @ {ctime}\n{glider} - {t0} to {t1}")

        # Save figure
        # export_fig(path_temp, sname, dpi=150)
        fig.savefig(path_temp / sname, dpi=150, bbox_inches='tight', pad_inches=0.1)

        # Remove handles so we can reuse figure
        # Delete contour handles 
        [x.remove() for x in h1.collections] # axes 1
        cb.remove()
        # plt.close()

    # if salinity:
    #     # Save salinity figure
    #     path_salt = path_plot / model / glider / f"salinity_{depth}m"
    #     os.makedirs(path_salt, exist_ok=True)
    #     sname = f"{glider}_track_salinity_{depth}m_{datefmt}.png"
    #     if (path_salt / sname).is_file():
    #         if not replot:
    #             continue
            
    #     with open(sfig, 'rb') as file:
    #         fig = pickle.load(file)
    #         ax = fig.axes[0]

    #         # Salinity
    #         if model == "gofs":
    #             tds = ds['salinity'].sel(time=ctime, depth=0).squeeze()
    #         elif model == "rtofs":
    #             tds = ds['salinity'].sel(time=ctime, depth=0).squeeze()

    #         # hargs["zorder"] = 3
    #         h2 = ax.contourf(tds['lon'], tds['lat'], tds.squeeze(), **hargs)
    #         axins = inset_axes(ax,  # here using axis of the lowest plot
    #             width="2.5%",  # width = 5% of parent_bbox width
    #             height="100%",  # height : 340% good for a (4x4) Grid
    #             loc='lower left',
    #             bbox_to_anchor=(1.05, 0., 1, 1),
    #             bbox_transform=ax.transAxes,
    #             borderpad=0
    #             )
    #         cb = fig.colorbar(h2, cax=axins)
    #         cb.ax.tick_params(labelsize=12)
    #         cb.set_label("Salinity", fontsize=13)

    #         # Plot subsetted glider track
    #         h3 = ax.plot(lon_gl, lat_gl,
    #             'w-',
    #             linewidth=5,
    #             transform=projection_data,
    #             )
    #         if argo:
    #             h4 = ax.plot(lon_argo, lat_argo,
    #                         "o",
    #                         color="limegreen",
    #                         markeredgecolor="black",
    #                         markersize=6,
    #                         transform=projection_data)
            
    #         if not storm_df.empty:
    #             bt = storm_df.index.min() - dt.timedelta(days=5)
    #             et = storm_df.index.max() + dt.timedelta(days=5)

    #             if (pd.to_datetime(t0) > bt) & (pd.to_datetime(t1) < et):
    #                 # Plot hurricanes (until that day)
    #                 track_1 = ax.plot(storm_df["lon"], storm_df["lat"],
    #                                 '-',
    #                                 linewidth=2.5,
    #                                 color="black",
    #                                 # markersize=8, 
    #                                 # # markerfillcolor="red",
    #                                 # markeredgecolor="black", 
    #                                 transform=projection_data, 
    #                                 zorder=20)
                    
    #                 markers = ax.scatter(tstorm["lon"], tstorm["lat"],
    #                                      c="red", s=8, marker="o",
    #                                      transform=projection_data, zorder=20)

    #                 # Plot hurricanes (track of that day)
    #                 track_2 = ax.plot(tstorm_day["lon"], tstorm_day["lat"],
    #                                 '-',
    #                                 color="red",
    #                                 linewidth=2.5,
    #                                 transform=projection_data, 
    #                                 zorder=20)
                    
    #                 markers = ax.scatter(tstorm_day["lon"], tstorm_day["lat"],
    #                                         c=tstorm_day["colors"], 
    #                                         s=tstorm_day["size"],
    #                                         marker='o',
    #                                         edgecolors="black",
    #                                         transform=projection_data,  zorder=20)
    #         ax.grid(True, color='k', linestyle='--', alpha=.25, linewidth=.75)

    #         # Adjust labels and title
    #         ax.set_title(f"RTOFS - Salinity ({depth}m) @ {ctime}\n{glider} - {t0} to {t1}")       
    #         export_fig(path_salt, sname, dpi=150)
