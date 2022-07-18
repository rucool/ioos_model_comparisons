import datetime as dt
import time

import hurricanes.configs as conf
import matplotlib
import numpy as np
import pandas as pd
from hurricanes.calc import (lon180to360,
                             lon360to180, 
                             density, 
                             ocean_heat_content,
                             )
from hurricanes.platforms import (get_active_gliders, 
                                  get_argo_floats_by_time,
                                  get_bathymetry)
from hurricanes.plotting import plot_ohc
from hurricanes.regions import region_config
from shapely.errors import TopologicalError
from hurricanes.models import rtofs as r
from hurricanes.models import gofs as g
from hurricanes.models import cmems as c
import xarray as xr

startTime = time.time()
matplotlib.use('agg')

parallel = True # utilize parallel processing?

# Set path to save plots
path_save = (conf.path_plots / "maps")

# For debug
# conf.days = 1
# conf.regions = ['mab']
    
# Get today and yesterday dates
today = dt.date.today()
date_end = today + dt.timedelta(days=1)
date_start = today - dt.timedelta(days=conf.days)
freq = '6H'

# Create dates that we want to plot
date_list = pd.date_range(
    date_start, 
    date_end,
    freq=freq,
    closed="right")

search_start = date_list[0] - dt.timedelta(hours=conf.search_hours)

# Get global extent for all regions
extent_list = []
for region in conf.regions:
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

if conf.argo:
    argo_data = get_argo_floats_by_time(global_extent, search_start, date_end)
else:
    argo_data = pd.DataFrame()

if conf.gliders:
    glider_data = get_active_gliders(global_extent, search_start, date_end, 
                                     parallel=False)
else:
    glider_data = pd.DataFrame()

if conf.bathy:
    bathy_data = get_bathymetry(global_extent)

# Load RTOFS and subset to global_extent of regions we are looking at.
rds = r()
lons_ind = np.interp(global_extent[:2], rds.lon.values[0,:], rds.x.values)
lats_ind = np.interp(global_extent[2:], rds.lat.values[:,0], rds.y.values)

rds = rds.isel(
    x=slice(np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int)), 
    y=slice(np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int))
    )

# Save rtofs lon and lat as variables to speed up indexing calculation
grid_lons = rds.lon.values[0,:]
grid_lats = rds.lat.values[:,0]
grid_x = rds.x.values
grid_y = rds.y.values

# Load GOFS
g_extents = lon180to360(global_extent[:2])
gds = g(rename=True).sel(
    lon=slice(g_extents[0], g_extents[1]),
    lat=slice(global_extent[2], global_extent[3])
)

# Load Copernicus
cds = c(rename=True).sel(
    lon=slice(global_extent[0], global_extent[1]),
    lat=slice(global_extent[2], global_extent[3]) 
)

# initialize keyword arguments. Grab anything from configs.py
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = False

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# for ctime in date_list:
def plot_ctime(ctime):
    print(f"Checking if {ctime} exists for each model.")
    
    try:
        rds_time = rds.sel(time=ctime)
        # startTime = time.time()
        rds_time['density'] = xr.apply_ufunc(density, 
                                rds_time['temperature'], 
                                -rds_time['depth'],
                                rds_time['salinity'], 
                                rds_time['lat'], 
                                rds_time['lon']
                                )
        rds_time['ohc'] = xr.apply_ufunc(ocean_heat_content, 
                            rds_time.depth, 
                            rds_time.temperature, 
                            rds_time.density, 
                            input_core_dims=[['depth'], ['depth'], ['depth']], 
                            vectorize=True)
        # print('RTOFS - Execution time in seconds: ' + str(time.time() - startTime))
        print(f"RTOFS: True")
        rdt_flag = True
    except KeyError as error:
        print(f"RTOFS: False")
        rdt_flag = False

    try:
        gds_time = gds.sel(time=ctime)
        # startTime = time.time()
        gds_time['density'] = xr.apply_ufunc(density, 
                                gds_time['temperature'], 
                                -gds_time['depth'],
                                gds_time['salinity'], 
                                gds_time['lat'], 
                                gds_time['lon']
                                )
        gds_time['ohc'] = xr.apply_ufunc(ocean_heat_content, 
                            gds_time.depth, 
                            gds_time.temperature, 
                            gds_time.density, 
                            input_core_dims=[['depth'], ['depth'], ['depth']], 
                            vectorize=True)
        # print('GOFS - Execution time in seconds: ' + str(time.time() - startTime))
        print(f"GOFS: True")
        gdt_flag = True
    except KeyError as error:
        print(f"GOFS: False")
        gdt_flag = False

    try:
        cds_time = cds.sel(time=ctime) #CMEMS
        # startTime = time.time()
        cds_time['density'] = xr.apply_ufunc(density, 
                                cds_time['temperature'], 
                                -cds_time['depth'],
                                cds_time['salinity'], 
                                cds_time['lat'], 
                                cds_time['lon']
                                )
        cds_time['ohc'] = xr.apply_ufunc(ocean_heat_content, 
                            cds_time.depth, 
                            cds_time.temperature, 
                            cds_time.density, 
                            input_core_dims=[['depth'], ['depth'], ['depth']], 
                            vectorize=True)
        # print('CMEMS: Execution time in seconds: ' + str(time.time() - startTime))
        print(f"CMEMS: True")
        cdt_flag = True
    except KeyError:
        print(f"CMEMS: False")
        cdt_flag = False
    print("\n")

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)
    
    for r in conf.regions:
        configs = region_config(r)

        # Save the extent of the region being plotted to a variable.
        extent = configs['extent']

        # Increase the extent a little bit to grab slightly more data than the region 
        # we want to plot. Otherwise, we will have areas of the plot with no data.
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()

        # Add the following to keyword arguments to ocean_heat_content function
        kwargs['path_save'] = path_save / configs['folder']
        kwargs['eez'] = configs['eez']

        key = 'ocean_heat_content'
        if key in configs:
            # if 'figsize' in configs[key]:
                # kwargs['figsize'] = configs[key]['figsize']
            if 'limits' in configs[key]:
                kwargs['limits'] = configs[key]['limits']
        else:
            continue
                
        if 'legend' in configs['figure']:
            kwargs['cols'] = configs['figure']['legend']['columns']

        if 'figsize' in configs['figure']:
            kwargs['figsize'] = configs['figure']['figsize']

        try:
            kwargs['bathy'] = bathy_data.sel(
                longitude=slice(extent_data[0] - 1, extent_data[1] + 1),
                latitude=slice(extent_data[2] - 1, extent_data[3] + 1)
            )
        except NameError:
            pass

        if rdt_flag:
            # Find x, y indexes of the area we want to subset
            lons_ind = np.interp(extent_data[:2], grid_lons, grid_x)
            lats_ind = np.interp(extent_data[2:], grid_lats, grid_y)

            # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice 
            # in order to widen the area of the extent slightly.
            extent_ind = [
                np.floor(lons_ind[0]).astype(int),
                np.ceil(lons_ind[1]).astype(int),
                np.floor(lats_ind[0]).astype(int),
                np.ceil(lats_ind[1]).astype(int)
                ]

            # Use .isel selector on x/y since we know indexes that we want to slice
            rds_slice = rds_time.sel(
                x=slice(extent_ind[0], extent_ind[1]), 
                y=slice(extent_ind[2], extent_ind[3])
                )
            
        if gdt_flag:
            # subset dataset to the proper extents for each region
            lon360 = lon180to360(extent_data[:2]) # convert from 360 to 180 lon
            gds_slice = gds_time.sel(
                lon=slice(lon360[0], lon360[1]),
                lat=slice(extent_data[2], extent_data[3])
            )

            # Convert from 0,360 lon to -180,180
            gds_slice['lon'] = lon360to180(gds_slice['lon'])

        if cdt_flag:
            cds_slice= cds_time.sel(
                lon=slice(extent_data[0], extent_data[1]),
                lat=slice(extent_data[2], extent_data[3])
                )

        # Subset downloaded Argo data to this region and time
        if not argo_data.empty:
            lon = argo_data['lon']
            lat = argo_data['lat']

            # Mask out anything beyond the extent
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            argo_region = argo_data[mask]
            argo_region.sort_index(inplace=True)

            # Mask out any argo floats beyond the time window
            idx = pd.IndexSlice
            kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

        # Subset downloaded glider data to this region and time
        if not glider_data.empty:
            lon = glider_data['lon']
            lat = glider_data['lat']
            
            # Mask out anything beyond the extent
            mask = (extent[0] <= lon) & (lon <= extent[1]) & (extent[2] <= lat) & (lat <= extent[3])
            glider_region = glider_data[mask]

            # Mask out any gliders beyond the time window
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time'))
                &
                (glider_region.index.get_level_values('time') < search_window_t1)
                ]
            kwargs['gliders'] = glider_region
            
        try:
            if rdt_flag and gdt_flag:
                plot_ohc(rds_slice, gds_slice, extent, configs['name'], **kwargs)
                
            if rdt_flag and cdt_flag:
                plot_ohc(rds_slice, cds_slice, extent, configs['name'], **kwargs)

            # Delete some keyword arguments that may not be defined in all
            # regions. We don't want to plot the regions with wrong inputs 
            if 'figsize' in kwargs:
                del kwargs['figsize']

            if 'limits' in kwargs:
                del kwargs['limits']

            if 'eez' in kwargs:
                del kwargs['eez']

            if 'gliders' in kwargs:
                del kwargs['gliders']

            if 'argo' in kwargs:
                del kwargs['argo']
                           
        except TopologicalError as error:
            print("Error: {error}")
            continue

def main():
    if parallel:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            executor.map(plot_ctime, date_list)
    else:
        for ctime in date_list:
            plot_ctime(ctime)           

    print('Execution time in seconds: ' + str(time.time() - startTime))


if __name__ == "__main__":
    main()