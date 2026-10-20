import datetime as dt
import time

import ioos_model_comparisons.configs as conf
import matplotlib
import numpy as np
import pandas as pd
from ioos_model_comparisons.calc import (lon180to360,
                             lon360to180, 
                             density, 
                             ocean_heat_content,
                             )
from ioos_model_comparisons.platforms import (get_active_gliders, 
                                  get_argo_floats_by_time,
                                  get_glider_by_id
                                  )
from ioos_model_comparisons.plotting import plot_ohc_single, plot_salt_single, plot_temp_single
from ioos_model_comparisons.regions import region_config
from shapely.errors import TopologicalError
from ioos_model_comparisons.models import rtofs as r
from ioos_model_comparisons.models import gofs as g
from ioos_model_comparisons.models import CMEMS as c
from ioos_model_comparisons.models import amseas as a
from cool_maps.plot import get_bathymetry

import xarray as xr

startTime = time.time()
matplotlib.use('agg')

parallel = False # utilize parallel processing?

# Which models should we plot?
plot_rtofs = True
plot_gofs = False
plot_cmems = False
plot_amseas = False

# Set path to save plots
path_save = (conf.path_plots / "maps")

# For debug
# conf.days = 1
conf.regions = ['windward']
conf.gliders = True
conf.argo = True

target_extent = [-68.25, -56.25, 9.25, 19.75]

# initialize keyword arguments. Grab anything from configs.py
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = True
    
# Get today and yesterday dates
today = dt.date.today()
# today = dt.datetime(*dt.date.today().timetuple()[:3]) + dt.timedelta(hours=12)
date_end = today + dt.timedelta(days=1)
date_start = today - dt.timedelta(days=conf.days)

# date_start = pd.Timestamp(2024, 9, 26, 0, 0, 0)
# date_end = pd.Timestamp(2024, 9, 26, 0, 0, 0)

# date_start = pd.Timestamp(2024, 8, 12, 0, 0, 0)
# date_end = pd.Timestamp(2024, 9, 4, 12, 0, 0)

# date_start = pd.Timestamp(2024, 10, 7, 0, 0, 0)
# date_end = pd.Timestamp(2024, 10, 15, 0, 0, 0)
# date_start = pd.Timestamp(2024, 11, 3, 0, 0, 0)
# date_end = pd.Timestamp(2024, 11, 10, 0, 0 ,0)


date_start = pd.Timestamp(2025, 7, 15, 0, 0, 0)
date_end = pd.Timestamp(2025, 8, 2, 0, 0 ,0)

freq = '6H'

# Create dates that we want to plot
date_list = pd.date_range(date_start, date_end, freq=freq)

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
global_extent = target_extent
# global_extent = [-70, -66,  15, 17]
lon_transform = lon180to360(global_extent[:2])

# conf.argo = False
if conf.argo:
    argo_data = get_argo_floats_by_time(global_extent, search_start, date_end)
else:
    argo_data = pd.DataFrame()

# conf.gliders= False
if conf.gliders:
    # glider_data = get_glider_by_id("ru29-20240419T1430")
    glider_data = get_active_gliders(global_extent, search_start, date_end, 
                                     parallel=False)
                                    
else:
    glider_data = pd.DataFrame()

if conf.bathy:
    bathy_data = get_bathymetry(global_extent)

if plot_rtofs:
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

if plot_gofs:
    # Load GOFS
    gds = g(rename=True).sel(
        lon=slice(lon_transform[0], lon_transform[1]),
        lat=slice(global_extent[2], global_extent[3])
    )

if plot_cmems:
    # Load Copernicus
    cds = c(rename=True).sel(
        lon=slice(global_extent[0], global_extent[1]),
        lat=slice(global_extent[2], global_extent[3]) 
    )


if plot_amseas:
    # Load amseas
    am = a(rename=True)
    ads = am.sel(
        lon=slice(lon_transform[0], lon_transform[1]),
        lat=slice(global_extent[2], global_extent[3])
        )
    
# from tropycal import realtime
# # from tropycal.utils.generic_utils import wind_to_category, generate_nhc_cone

# realtime_obj = realtime.Realtime()

# # storms = [realtime_obj.get_storm(key) for key in realtime_obj.list_active_storms(basin='north_atlantic')]
# from tropycal import tracks
# # # import pandas as pd
# # from tropycal import realtime

# basin = tracks.TrackDataset(basin='north_atlantic', source='ibtracs', include_btk=True)
# storms = basin.get_storm(('rafael', 2024))

# # storm_dict = {}
# # for s in [storms[1]]:
# # # s.name == 'IDALIA':
# #     storm_dict[s.name] = {}
# #     storm_dict[s.name]['track'] = pd.DataFrame({"time": s.time, "lon": s.lon, "lat": s.lat}).set_index('time')
# #     storm_dict[s.name]['cone'] = {}
# #     for t in date_list:
# #         storm_dict[s.name]['cone'][t] = generate_nhc_cone(s.get_nhc_forecast_dict(t), s.basin, cone_days=5)

# kwargs['storms'] = storms


# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# for ctime in date_list:
def plot_ctime(ctime):
    print(f"Checking if {ctime} exists for each model.")
    
    if plot_rtofs:
        try:
            rds_time = rds.sel(time=ctime)
            startTime = time.time()
            # rds_time['density'] = xr.apply_ufunc(density, 
            #                         rds_time['temperature'], 
            #                         -rds_time['depth'],
            #                         rds_time['salinity'], 
            #                         rds_time['lat'], 
            #                         rds_time['lon']
            #                         )
            # rds_time['ohc'] = xr.apply_ufunc(ocean_heat_content, 
            #                     rds_time.depth, 
            #                     rds_time.temperature, 
            #                     rds_time.density, 
            #                     input_core_dims=[['depth'], ['depth'], ['depth']], 
            #                     vectorize=True)
            print('RTOFS - Execution time in seconds: ' + str(time.time() - startTime))
            print(f"RTOFS: True")
            rdt_flag = True
        except KeyError as error:
            print(f"RTOFS: False")
            rdt_flag = False
    else:
        rdt_flag = False

    if plot_gofs:
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
    else:
        gdt_flag = False

    if plot_cmems:
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
    else:
        cdt_flag = False

    if plot_amseas:
        try:
            amt = ads.sel(time=ctime)
            amt['density'] = xr.apply_ufunc(density, 
                            amt['temperature'], 
                            -amt['depth'],
                            amt['salinity'], 
                            amt['lat'], 
                            amt['lon']
                            )
            amt['ohc'] = xr.apply_ufunc(ocean_heat_content, 
                                amt.depth, 
                                amt.temperature, 
                                amt.density, 
                                input_core_dims=[['depth'], ['depth'], ['depth']], 
                                vectorize=True)
            print(f"AMSEAS: True")
            amt_flag = True
        except KeyError as error:
            print(f"AMSEAS: False - {error}")
            amt_flag = False
    else:
        amt_flag = False

    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr)
    
    for r in conf.regions:
        configs = region_config(r)

        # Save the extent of the region being plotted to a variable.
        extent = target_extent
        extent_data = extent

        # convert from 360 to 180 lon
        lon360 = lon180to360(extent_data[:2])

        # Add the following to keyword arguments to ocean_heat_content function
        kwargs['path_save'] = path_save / configs['folder']
        kwargs['eez'] = False

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
            kwargs['bathy'] = bathy_data#.sel(
            #     longitude=slice(extent_data[0] - 1, extent_data[1] + 1),
            #     latitude=slice(extent_data[2] - 1, extent_data[3] + 1)
            # )
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
            
        if amt_flag:
            # subset dataset to the proper extents for each region
            amt_slice = amt.sel(
                lon=slice(lon360[0], lon360[1]),
                lat=slice(extent_data[2], extent_data[3])
            )

            # Convert from 0,360 lon to -180,180
            amt_slice['lon'] = lon360to180(amt_slice['lon'])

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
            # kwargs['gliders'] = glider_data
            
        try:
            # kwargs['figsize'] = (14, 8)
            # dict(depth=0, limits=[27, 29.3, .1]),
            # dict(depth=150, limits=[16.5, 24.5, .5]),

            if rdt_flag:
                # plot_ohc_single(rds_slice.sel(depth=0), extent, configs['name'], **kwargs)
                # plot_salt_single(rds_slice.sel(depth=0), extent, configs['name'], min=33, max=36.1, stride=.1, **kwargs)
                # plot_salt_single(rds_slice.sel(depth=150), extent, configs['name'], min=34.75, max=37.75, stride=.25, map_pad=0, **kwargs)
                # plot_temp_single(rds_slice.sel(depth=0), extent, configs['name'], min=33, max=36, stride=.25, **kwargs)
                plot_temp_single(rds_slice.sel(depth=150), extent, configs['name'], min=15, max=27, stride=.5, map_pad=0, **kwargs)

            if gdt_flag:
                plot_ohc_single(gds_slice, extent, configs['name'], **kwargs)
         
            if cdt_flag:
                plot_ohc_single(cds_slice, extent, configs['name'], **kwargs)

            # if amt_flag:
                # plot_ohc(rds_slice, amt_slice, extent, configs['name'], **kwargs)

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
