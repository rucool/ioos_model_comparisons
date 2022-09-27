import datetime as dt

import ioos_model_comparisons.configs as configs
import numpy as np
import pandas as pd
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import gofs, rtofs
from ioos_model_comparisons.platforms import (get_active_gliders, get_argo_floats_by_time,
                                  get_bathymetry)
from ioos_model_comparisons.plotting import plot_model_region_comparison
from ioos_model_comparisons.regions import region_config
import matplotlib
import xarray as xr

matplotlib.use('agg')

# Set path to save plots
path_save = (configs.path_plots / "maps" / "rtofs_comparisons")

# initialize keyword arguments for map plotz
kwargs = dict()
kwargs['transform'] = configs.projection
kwargs['path_save'] = path_save
kwargs['dpi'] = configs.dpi

# Get today and yesterday dates
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)
past = today - dt.timedelta(days=3)
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
# ds1 = xr.open_mfdataset('/Users/mikesmith/Documents/rtofs/prod/*/*.nc')
# ds2 = xr.open_mfdataset('/Users/mikesmith/Documents/rtofs/v2.2/*/*.nc')
# ds1 = ds1.rename({"MT": 'time', "Depth": "depth", "Y": "y", "X": "x", "Latitude": "lat", "Longitude": "lon"})
# ds2 = ds2.rename({"MT": 'time', "Depth": "depth", "Y": "y", "X": "x", "Latitude": "lat", "Longitude": "lon"})
ds1 = rtofs()
ds2 = xr.open_mfdataset('/Users/mikesmith/Documents/data/rtofs/2.2/rtofs.20220621/*.nc')
ds2 = ds2.rename({"MT": 'time', "Depth": "depth", "Y": "y", "X": "x", "Latitude": "lat", "Longitude": "lon"})

# Loop through regions
for item in configs.regions:
    region = region_config(item)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')

    # Create a map figure and serialize it if one doesn't already exist
    region_name = "_".join(region["name"].split(' ')).lower()
    kwargs['colorbar'] = True

    if region['currents']['bool']:
        kwargs['currents'] = region['currents']

    if bathy_data:
        kwargs['bathy'] = bathy_data.sel(
            longitude=slice(extent[0] - 1, extent[1] + 1),
            latitude=slice(extent[2] - 1, extent[3] + 1)
        )
    extent = np.add(extent, [-1, 1, -1, 1]).tolist()
    
    # Save rtofs lon and lat as variables to speed up indexing calculation
    grid_lons = ds1.lon.values[0,:]
    grid_lats = ds1.lat.values[:,0]
    grid_x = ds1.x.values
    grid_y = ds1.y.values

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
    ds1_sub = ds1.isel(
        x=slice(extent_ind[0], extent_ind[1]), 
        y=slice(extent_ind[2], extent_ind[3])
        ).set_coords(['u', 'v'])
    ds1_sub.attrs['model'] = 'RTOFS 2.0'

    # Load RTOFS2.2 DataSet
    ds2_sub = ds2.isel(
        x=slice(extent_ind[0], extent_ind[1]), 
        y=slice(extent_ind[2], extent_ind[3])
        ).set_coords(['u', 'v'])
    ds2_sub.attrs['model'] = 'RTOFS 2.2'
   
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
            plot_model_region_comparison(ds1_sub.sel(time=t), ds2_sub.sel(time=t), region, **kwargs)
        except KeyError as e:
            print(e)
            continue
