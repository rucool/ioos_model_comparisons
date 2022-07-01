import datetime as dt

import hurricanes.configs as configs
import numpy as np
import pandas as pd
from hurricanes.calc import lon180to360, lon360to180
from hurricanes.models import gofs, rtofs
from hurricanes.platforms import (get_active_gliders, get_argo_floats_by_time,
                                  get_bathymetry)
from hurricanes.plotting import (plot_model_region_comparison,)
                                #  plot_model_region_comparison_streamplot)
from hurricanes.regions import region_config
import matplotlib
import time

startTime = time.time()
matplotlib.use('agg')

# Set path to save plots
path_save = (configs.path_plots / "maps")

# initialize keyword arguments for map plotz
kwargs = dict()
kwargs['transform'] = configs.projection
kwargs['path_save'] = path_save
kwargs['dpi'] = configs.dpi
kwargs['overwrite'] = False

# configs.days = 2
# Get today and yesterday dates
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)
past = today - dt.timedelta(days=configs.days)

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# Create dates that we want to plot
date_list = pd.date_range(past, tomorrow, freq="6H", closed="right")
start = date_list[0] - dt.timedelta(hours=configs.search_hours)

# Get global extent for all regions
extent_list = []
for region in configs.regions:
    extent_list.append(region_config(region)["extent"])

extent_df = pd.DataFrame(
    np.array(extent_list),
    columns=['lonmin', 'lonmax', 'latmin', 'latmax']
    )
global_extent = [
    extent_df.lonmin.min(),
    extent_df.lonmax.max(),
    extent_df.latmin.min(),
    extent_df.latmax.max()
    ]

if configs.argo:
    argo_data = get_argo_floats_by_time(global_extent, start, tomorrow)
else:
    argo_data = pd.DataFrame()

if configs.gliders:
    glider_data = get_active_gliders(global_extent, start, tomorrow, parallel=False)
else:
    glider_data = pd.DataFrame()

if configs.bathy:
    bathy_data = get_bathymetry(global_extent)

# Load RTOFS DataSet
rds = rtofs() 

# Save rtofs lon and lat as variables to speed up indexing calculation
grid_lons = rds.lon.values[0,:]
grid_lats = rds.lat.values[:,0]
grid_x = rds.x.values
grid_y = rds.y.values

# Load GOFS DataSet
gds = gofs(rename=True)

# Loop through regions
# for item in configs.regions:
for item in ['west_florida_shelf']:
    region = region_config(item)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')

    # Create a map figure and serialize it if one doesn't already exist
    region_name = "_".join(region["name"].split(' ')).lower()
    kwargs['colorbar'] = True

    if 'eez' in region.keys():
        kwargs["eez"] = region["eez"]

    if region['currents']['bool']:
        kwargs['currents'] = region['currents']

    try:
        kwargs['bathy'] = bathy_data.sel(
            longitude=slice(extent[0] - 1, extent[1] + 1),
            latitude=slice(extent[2] - 1, extent[3] + 1)
        )
    except NameError:
        pass
            
    extent = np.add(extent, [-1, 1, -1, 1]).tolist()

    # Find x, y indexes of the area we want to subset
    lons_ind = np.interp(extent[:2], grid_lons, grid_x)
    lats_ind = np.interp(extent[2:], grid_lats, grid_y)

    # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice 
    # in order to widen the area of the extent slightly.
    extent_ind = [
        np.floor(lons_ind[0]).astype(int),
        np.ceil(lons_ind[1]).astype(int),
        np.floor(lats_ind[0]).astype(int),
        np.ceil(lats_ind[1]).astype(int)
        ]

    # Use .isel selector on x/y since we know indexes that we want to slice
    rds_sub = rds.isel(
        x=slice(extent_ind[0], extent_ind[1]), 
        y=slice(extent_ind[2], extent_ind[3])
        ).set_coords(['u', 'v'])
    
    # subset dataset to the proper extents for each region
    lon360 = lon180to360(extent[:2]) # convert from 360 to 180 lon
    gds_sub = gds.sel(
        lon=slice(lon360[0], lon360[1]),
        lat=slice(extent[2], extent[3])
    ).set_coords(['u', 'v'])
    
    # Convert from 0,360 lon to -180,180
    gds_sub['lon'] = lon360to180(gds_sub['lon'])

    # Iterate through dates 
    for t in date_list:
        search_window_t0 = (t - dt.timedelta(hours=configs.search_hours)).strftime(tstr)
        kwargs['t0'] = search_window_t0
        search_window_t1 = t.strftime(tstr)
        
        if not argo_data.empty:
            argo_lon = argo_data['lon']
            argo_lat = argo_data['lat']
            argo_region = argo_data[
                (extent[0] < argo_lon) & (argo_lon < extent[1]) & (extent[2] < argo_lat) & (argo_lat < extent[3])
            ]
            argo_region.sort_index(inplace=True)
            idx = pd.IndexSlice
            kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

        if not glider_data.empty:
            glider_lon = glider_data['lon']
            glider_lat = glider_data['lat']
            glider_region = glider_data[
                (extent[0] < glider_lon) & (glider_lon < extent[1]) & (extent[2] < glider_lat) & (glider_lat < extent[3])
                ]
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                &
                (glider_region.index.get_level_values('time') < search_window_t1)
                ]
            kwargs['gliders'] = glider_region
        try:
            plot_model_region_comparison(rds_sub.sel(time=t), gds_sub.sel(time=t), region, **kwargs)
        #     # plot_model_region_comparison_streamplot(rds_sub.sel(time=t), gds_sub.sel(time=t), region, **kwargs)
        except KeyError as e:
            print(e)
            continue

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))