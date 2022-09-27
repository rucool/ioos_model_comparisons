#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from ioos_model_comparisons.calc import find_nearest
from ioos_model_comparisons.plotting import map_add_ticks, map_add_features, map_add_bathymetry, export_fig
from ioos_model_comparisons.platforms import strip_timezone, get_glider_by_id, get_bathymetry, get_argo_floats_by_time
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cmocean
from pathlib import Path
import os
from glob import glob
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Set paths
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name
data_path = (current_dir / "data").resolve() # create data path
plot_path = (current_dir / "plots").resolve() # create plot path
ibtracs_path = "~/Documents/data/ibtracs/IBTrACS.NA.v04r00.nc"

# User defined variables
glider_id = "ng645-20210613T0000"
ibtracs_id = 2281
url_gofs = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
freq = '1D' # interval of time for each plot
extent = [-98, -80, 18, 31] # cartopy extent format
# temp_range = [12, 24, .5] # [min, max, step]
temp_range = [5.5, 24, .5] # [min, max, step]
haline_range = [34, 37, .25] # [min, max, step]
projection_map = ccrs.Mercator()
projection_data = ccrs.PlateCarree()
bathy = True
argo = True
model = "rtofs"
# model = "gofs"

# Read glider data directly from the erddap server
# glider_erddap = get_glider_by_id(dataset_id=glider_id)
# glider_erddap = glider_erddap.to_pickle(data_path / "ng645-20210613T0000_data.pkl")
glider_erddap = pd.read_pickle(data_path / "ng645-20210613T0000_data.pkl")
# glider_erddap = strip_timezone(glider_erddap)

# Create lon and lat variables for the entire glider track
glider_lon = glider_erddap['longitude (degrees_east)']
glider_lat = glider_erddap['latitude (degrees_north)']

# Create date range from start date to end date of glider data
# start_date = glider_erddap.index[0].floor(freq="1D")
# end_date = glider_erddap.index[-1].ceil(freq='1D')

center_time = dt.datetime(2021, 8, 29)
start_date = center_time - dt.timedelta(days=7)
end_date = center_time + dt.timedelta(days=7)

ranges = pd.date_range(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), freq=freq)

# Read argo data from the erddap server
# argo_erddap = get_argo_floats_by_time(extent, start_date, end_date)
# argo_erddap = argo_erddap.to_pickle(data_path / "argo_floats_20210624_20210924.pkl")
argo_erddap = pd.read_pickle(data_path / "argo_floats_20210624_20210924.pkl")

# Read a precreated pickle file of gofs, rtofs, copernicus, and the glider data
df = pd.read_pickle(data_path / "combined_data.pkl")

# read in ibtracs hurricane packs and convert to dataframe
hurricane = xr.open_dataset(ibtracs_path)
time_h = pd.to_datetime(hurricane.time[ibtracs_id,:])
lon_h = hurricane.lon[ibtracs_id,:]
lat_h = hurricane.lat[ibtracs_id,:]
cat_h = hurricane.usa_sshs[ibtracs_id,:]
storm=pd.DataFrame(dict(time=time_h, lon=lon_h, lat=lat_h, cat=cat_h))
storm.set_index("time", inplace=True)

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
storm["cat"] = storm["cat"].astype(str)
storm["colors"] = storm["cat"].map(colors_map)
storm["size"] = storm["cat"].map(size_map)*15

# dt_h = pd.to_datetime(time_h.data[0])
# name = ds.name[number]
# ida_wind_ib = ds.usa_wind[ida_storm_number,:]
# ida_press_ib = ds.usa_pres[ida_storm_number,:]
# ida_rmw_ib = ds.usa_rmw[ida_storm_number,:]


# Create dictionary of plotting inputs for each variable
vars = {}

# Heat content
vars['ocean_heat_content'] = dict(name='Ocean Heat Content', units = 'kJ cm-2', flip_y=False, ylim=[0, 130])
vars['potential_energy_anomaly_100m'] = dict(name='Potential Energy Anomaly (100m)', units='J m-3', flip_y=False, ylim=[100, 650])

# Mixed Layer Depth
vars['mixed_layer_depth_from_temp'] = dict(name='MLD_temp', units='m', flip_y=True, ylim=[0, 55])
vars['mixed_layer_temp_from_temp'] = dict(name='MLT_temp', units='dec C', flip_y=False, ylim=[20, 32])

# Salinity
vars['salinity_surface'] = dict(name='Surface Salinity', units=' ', flip_y=False, ylim=[28, 38])
vars['salinity_max'] = dict(name='Max Salinity ', units=' ', flip_y=False, ylim=[36, 37.2])
vars['salinity_max_depth'] = dict(name='Max Salinity - Depth', units='m', flip_y=True, ylim=[0, 205])

# Average temperatures
vars['average_temp_mldt_to_100m'] = dict(name='Mean Temp. (MLDt to 100m)', units='deg C', flip_y=False, ylim=[20, 32])
vars['average_temp_000m_to_100m'] = dict(name='Mean Temp. (0 to 100m)', units='deg C', flip_y=False, ylim=[20, 32])
vars['average_temp_100m_to_200m'] = dict(name='Mean Temp. (100 to 200m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_200m_to_300m'] = dict(name='Mean Temp. (200 to 300m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_300m_to_400m'] = dict(name='Mean Temp. (300 to 400m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_400m_to_500m'] = dict(name='Mean Temp. (400 to 500m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_500m_to_600m'] = dict(name='Mean Temp. (500 to 600m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_600m_to_700m'] = dict(name='Mean Temp. (600 to 700m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_700m_to_800m'] = dict(name='Mean Temp. (700 to 800m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_800m_to_900m'] = dict(name='Mean Temp. (800 to 900m)', units='deg C', flip_y=False, ylim=[4, 25])
vars['average_temp_900m_to_1000m'] = dict(name='Mean Temp. (900 to 1000m)', units='deg C', flip_y=False, ylim=[4, 25])

# Temperature arguments
targs = {}
targs['vmin'] = temp_range[0]
targs['vmax'] = temp_range[1]
targs['transform'] = projection_data
targs['cmap'] = cmocean.cm.thermal
targs['levels'] = np.arange(temp_range[0], temp_range[1], temp_range[2])
targs['extend'] = 'both'

# Salinity arguments
hargs = {}
hargs['vmin'] = haline_range[0]
hargs['vmax'] = haline_range[1]
hargs['transform'] = projection_data
hargs['cmap'] = cmocean.cm.haline
hargs['levels'] = np.arange(haline_range[0], haline_range[1], haline_range[2])
hargs['extend'] = 'both'

if bathy:
    # Load bathymetry
    bathy = get_bathymetry(extent)

if model == "gofs":
    # Load and subset GOFS data to the proper extents for each region
    ds = xr.open_dataset(url_gofs, drop_variables="tau")
    ds = ds.sel(
        lon=slice(extent[0] + 359, extent[1] + 361),
        lat=slice(extent[2] - 1, extent[3] + 1)
    )
    ds['lon'] = ds['lon'] - 360  # Convert model lon to glider lon
elif model == "rtofs":
    # RTOFS
    # Load in RTOFS files locally
    rtofs_file_dates = []
    rtofs_file_paths = []
    for date in pd.date_range(start_date, end_date).to_list():
        tstr = date.strftime('rtofs.%Y%m%d')
        files = sorted(glob(os.path.join("/Users/mikesmith/Documents/data/rtofs/", tstr, '*.nc')))
        for f in files:
            if f == '':
                continue
            else:
                date_list = f.split('rtofs/rtofs.')[1].split('/')
                rtofs_file_dates.append(pd.to_datetime(date_list[0]) + dt.timedelta(hours=int(date_list[1].split('_')[3].strip('f'))))
                rtofs_file_paths.append(f)

    rtofs_df = pd.DataFrame(list(zip(rtofs_file_dates, rtofs_file_paths)), columns=['date', 'paths'])
    rtofs_df.set_index('date', inplace=True)

    with xr.open_dataset(rtofs_df['paths'][0]) as tds:
        # Save rtofs lon and lat as variables to speed up indexing calculation
        rtofs_lon = tds.Longitude.values
        rtofs_lat = tds.Latitude.values

    # Find index of nearest lon and lat points
    _, lon1_ind = find_nearest(rtofs_lon[0, :], extent[0])
    _, lon2_ind = find_nearest(rtofs_lon[0, :], extent[1])

    _, lat1_ind = find_nearest(rtofs_lat[:, 0], extent[2])
    _, lat2_ind = find_nearest(rtofs_lat[:, 0], extent[3])

    rtofs_extent = [lon1_ind, lon2_ind, lat1_ind, lat2_ind]

    # ds = xr.open_mfdataset(rtofs_file_paths, parallel=True)
    # ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat',
    #                 'MT': 'time', 'Depth': 'depth'})

    # # subset the dataset to the extent of the region
    # ds = ds.isel(
    #     X=slice(rtofs_extent[0], rtofs_extent[1]),
    #     Y=slice(rtofs_extent[2], rtofs_extent[3])
    #     ).squeeze()

# Create figure 
fig, ax = plt.subplots(
    figsize=(12, 9),
    subplot_kw=dict(projection=projection_map)
)

# Make the map pretty
map_add_features(ax, extent)# zorder=0)
map_add_bathymetry(ax, bathy, projection_data, [-1000, -100], zorder=1.5)
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

with open('plots.obj', 'wb') as file:
    pickle.dump(fig, file)

for index, value in enumerate(ranges[1:], start=1):    
    # Save time range as variables
    t0 = ranges[index-1].strftime("%Y-%m-%d")
    t1 = ranges[index].strftime("%Y-%m-%d 00:00:00")
    t1s = ranges[index].strftime("%Y-%m-%d")
    print(f"{index-1} to {index}, {t0} to {t1}")
    ctime = pd.to_datetime(t0) + dt.timedelta(hours=12) # Center of time range
    datefmt = f"{t0}_to_{t1s}" # String for date range in filename

    # Subset concatenated dataframe by t0 and t1
    tdf = df[t0:t1]

    if not storm.empty:
        tstorm = storm[:1]
        tstorm_day = storm[t0:t1]
        
    # Subset glider erddap dataframe
    glider_erddap_temp = glider_erddap[t0:t1]
    lon_gl = glider_erddap_temp['longitude (degrees_east)']
    lat_gl = glider_erddap_temp['latitude (degrees_north)']

    # Subset argo erddap dataframe
    argo_erddap_temp = argo_erddap[t0:t1]
    
    # Groupby platform and grab the first record in reach group
    # Argo reports each profile at the same longitude and latitude
    grouped = argo_erddap_temp.groupby(["platform_number"]).first()
    lon_argo = grouped['longitude (degrees_east)']
    lat_argo = grouped['latitude (degrees_north)']

    if model == "rtofs":
        # Load RTOFS file that corresponds to ctime
        ds = xr.open_dataset(rtofs_df[rtofs_df.index==ctime]['paths'][0])
        ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat',  
                        'MT': 'time', 'Depth': 'depth'})
        # subset the dataset to the extent of the region
        ds = ds.isel(
            X=slice(rtofs_extent[0], rtofs_extent[1]),
            Y=slice(rtofs_extent[2], rtofs_extent[3])
            ).squeeze()
    
    # with open('plots.obj', 'rb') as file:
    #     fig = pickle.load(file)
    #     ax = fig.axes[0]

    #     # Temperature
    #     if model == "gofs":
    #         temp = ds['water_temp'].sel(time=ctime, depth=200).squeeze()
    #     elif model == "rtofs":
    #         temp = ds['temperature'].sel(depth=200).squeeze()

    #     # Contour plot of the variable
    #     h1 = ax.contourf(temp['lon'], temp['lat'], temp.squeeze(), **targs)
    #     axins = inset_axes(ax,  # here using axis of the lowest plot
    #         width="2.5%",  # width = 5% of parent_bbox width
    #         height="100%",  # height : 340% good for a (4x4) Grid
    #         loc='lower left',
    #         bbox_to_anchor=(1.05, 0., 1, 1),
    #         bbox_transform=ax.transAxes,
    #         borderpad=0
    #         )
    #     cb = fig.colorbar(h1, cax=axins)
    #     cb.ax.tick_params(labelsize=12)
    #     cb.set_label("Temperature (deg C)", fontsize=13)

    #     # Plot subsetted glider track
    #     h3 = ax.plot(lon_gl, lat_gl,
    #         'w-',
    #         linewidth=5,
    #         transform=projection_data)

    #     h4 = ax.plot(lon_argo, lat_argo,
    #                  "o",
    #                  color="limegreen",
    #                  markeredgecolor="black",
    #                  markersize=6,
    #                  transform=projection_data)

    #     if not tstorm.empty:
    #         bt = storm.index.min() - dt.timedelta(days=5)
    #         et = storm.index.max() + dt.timedelta(days=5)

    #         if (pd.to_datetime(t0) > bt) & (pd.to_datetime(t1) < et):
    #             # Plot hurricanes (until that day)
    #             track_1 = ax.plot(storm["lon"], storm["lat"],
    #                             '-',
    #                             linewidth=2.5,
    #                             color="black",
    #                             # markersize=8, 
    #                             # # markerfillcolor="red",
    #                             # markeredgecolor="black", 
    #                             transform=projection_data, 
    #                             zorder=20)
                
    #             # markers = ax.scatter(tstorm["lon"], tstorm["lat"],
    #             #                      c="red", s=8, marker="o",
    #             #                      transform=projection_data, zorder=20)

    #             # Plot hurricanes (track of that day)
    #             track_2 = ax.plot(tstorm_day["lon"], tstorm_day["lat"],
    #                             '-',
    #                             color="red",
    #                             linewidth=2.5,
    #                             transform=projection_data, 
    #                             zorder=20)
                
    #             markers = ax.scatter(tstorm_day["lon"], tstorm_day["lat"],
    #                                     c=tstorm_day["colors"], 
    #                                     s=tstorm_day["size"],
    #                                     marker='o',
    #                                     edgecolors="black",
    #                                     transform=projection_data,  zorder=20)
            
    #     ax.grid(True, color='k', linestyle='--', alpha=.5, linewidth=.5)

    #     # Add title
    #     ax.set_title(f"Glider: ng645\n{t0} to {t1}\n{model} Temperature (200m) @ {ctime}")

    #     # Save salinity figure
    #     temp_path = plot_path / model / "temp" 
    #     os.makedirs(temp_path, exist_ok=True)
    #     export_fig(temp_path / f"glider_track_{datefmt}.png", dpi=300)
    #     plt.close()

    with open('plots.obj', 'rb') as file:
        fig = pickle.load(file)
        ax = fig.axes[0]

        # Salinity
        if model == "gofs":
            haline = ds['salinity'].sel(time=ctime, depth=0).squeeze()
        elif model == "rtofs":
            haline = ds['salinity'].sel(depth=0).squeeze()

        # hargs["zorder"] = 3
        h2 = ax.contourf(haline['lon'], haline['lat'], haline.squeeze(), **hargs)
        axins = inset_axes(ax,  # here using axis of the lowest plot
            width="2.5%",  # width = 5% of parent_bbox width
            height="100%",  # height : 340% good for a (4x4) Grid
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
            )
        cb = fig.colorbar(h2, cax=axins)
        cb.ax.tick_params(labelsize=12)
        cb.set_label("Salinity", fontsize=13)

        # Plot subsetted glider track
        h3 = ax.plot(lon_gl, lat_gl,
            'w-',
            linewidth=5,
            transform=projection_data,
            )
        h4 = ax.plot(lon_argo, lat_argo,
                     "o",
                     color="limegreen",
                     markeredgecolor="black",
                     markersize=6,
                     transform=projection_data)
        
        if not tstorm.empty:
            bt = storm.index.min() - dt.timedelta(days=5)
            et = storm.index.max() + dt.timedelta(days=5)

            if (pd.to_datetime(t0) > bt) & (pd.to_datetime(t1) < et):
                # Plot hurricanes (until that day)
                track_1 = ax.plot(storm["lon"], storm["lat"],
                                '-',
                                linewidth=2.5,
                                color="black",
                                # markersize=8, 
                                # # markerfillcolor="red",
                                # markeredgecolor="black", 
                                transform=projection_data, 
                                zorder=20)
                
                # markers = ax.scatter(tstorm["lon"], tstorm["lat"],
                #                      c="red", s=8, marker="o",
                #                      transform=projection_data, zorder=20)

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
        ax.grid(True, color='k', linestyle='--', alpha=.25, linewidth=.75)

        # Adjust labels and title
        ax.set_title(f"Glider: ng645\n {t0} to {t1}\n{model} Surface Salinity @ {ctime}")

        # Save salinity figure
        salt_path = plot_path / model / "haline"
        os.makedirs(salt_path, exist_ok=True)
        export_fig(salt_path / f"glider_track_{datefmt}.png", dpi=300)

    # # Split out data sources into separate dataframes
    # rtofs = tdf[tdf.source == 'rtofs']
    # gofs = tdf[tdf.source == 'gofs']
    # copernicus = tdf[tdf.source == 'copernicus']
    # glider = tdf[tdf.source == 'ng645']

    # for k, v in vars.items():
    #     var_title = v['name']
    #     if not v['units'] == ' ':
    #         var_title += f" ({v['units']})"

    #     # Create figure 
    #     fig, ax = plt.subplots(figsize=(13.333, 7.5))

    #     # Plot each model 
    #     h1 = ax.plot(glider.index, glider[k], 'b-o', markersize=2, label='ng645')
    #     h2 = ax.plot(gofs.index, gofs[k], 'g-o', markersize=2, label='gofs')
    #     h3 = ax.plot(rtofs.index, rtofs[k], 'r-o', markersize=2, label='rtofs')
    #     h4 = ax.plot(copernicus.index, copernicus[k], 'm-o', markersize=2, label='copernicus')

    #     # Set axis limits 
    #     ax.set_ylim(v['ylim'])

    #     # Invert axis if flip_y is True
    #     if v['flip_y']:
    #         ax.invert_yaxis()

    #     # Adjust axes labels
    #     ax.set_xlabel('Datetime (GMT)', fontsize=14, fontweight='bold')
    #     ax.set_ylabel(var_title, fontsize=14, fontweight='bold')

    #     # Create and set title
    #     title = f"ng645 Model Comparisons\n{v['name']} \n {t0} to {t1}"
    #     ax.set_title(title)

    #     # Add legend
    #     plt.legend()

    #     # Save figure 
    #     plt.savefig(f'/Users/mikesmith/Documents/temp/glider_{k}_{datefmt}.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
    #     plt.close()


