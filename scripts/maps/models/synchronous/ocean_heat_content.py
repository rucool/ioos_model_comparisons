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
                                  )
from ioos_model_comparisons.plotting import plot_ohc
from ioos_model_comparisons.regions import region_config
from shapely.errors import TopologicalError
from cool_maps.plot import get_bathymetry

import xarray as xr

startTime = time.time()
matplotlib.use('agg')

parallel = False # utilize parallel processing?

# Which models should we plot?
plot_rtofs = True
plot_espc = True
plot_cmems = True
plot_amseas = False
plot_para = True

# Set path to save plots
path_save = (conf.path_plots / "maps")

# For debug
# conf.days = 1
conf.regions = ['mab', 'sab', 'gom', 'caribbean', 'tropical_western_atlantic']

# initialize keyword arguments. Grab anything from configs.py
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = False
    
# Get today and yesterday dates
today = dt.date.today()
# today = dt.datetime(*dt.date.today().timetuple()[:3]) + dt.timedelta(hours=12)
date_end = today + dt.timedelta(days=2)
date_start = today - dt.timedelta(days=conf.days)
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

lon_transform = lon180to360(global_extent[:2])

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

if plot_rtofs:
    from ioos_model_comparisons.models import rtofs as r

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

if plot_para:
    # Load RTOFS and subset to global_extent of regions we are looking at.
    rdsp = r(source='parallel')
    lons_ind = np.interp(global_extent[:2], rdsp.lon.values[0,:], rdsp.x.values)
    lats_ind = np.interp(global_extent[2:], rdsp.lat.values[:,0], rdsp.y.values)

    rdsp = rdsp.isel(
        x=slice(np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int)), 
        y=slice(np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int))
        )

    # # Save rtofs lon and lat as variables to speed up indexing calculation
    # grid_lons = rds.lon.values[0,:]
    # grid_lats = rds.lat.values[:,0]
    # grid_x = rds.x.values
    # grid_y = rds.y.values

if plot_espc:
    from ioos_model_comparisons.models import ESPC as g

    # Load GOFS
    gds = g()
    gds = gds.get_combined_subset(global_extent[:2], global_extent[2:])
    # .sel(
    #     lon=slice(lon_transform[0], lon_transform[1]),
    #     lat=slice(global_extent[2], global_extent[3])
    # )

if plot_cmems:
    from ioos_model_comparisons.models import CMEMS as c

    # Load Copernicus
    # cds = c(rename=True).sel(
    #     lon=slice(global_extent[0], global_extent[1]),
    #     lat=slice(global_extent[2], global_extent[3]) 
    # )
    cds = c()
    cds = cds.get_combined_subset(global_extent[:2], global_extent[2:])

if plot_amseas:
    from ioos_model_comparisons.models import amseas as a

    # Load amseas
    am = a(rename=True)
    ads = am.sel(
        lon=slice(lon_transform[0], lon_transform[1]),
        lat=slice(global_extent[2], global_extent[3])
        )
    
# from tropycal import realtime
# from tropycal.utils.generic_utils import wind_to_category, generate_nhc_cone

# realtime_obj = realtime.Realtime()

# storm_dict = {}
# storms = [realtime_obj.get_storm(key) for key in realtime_obj.list_active_storms(basin='north_atlantic')]
# for s in storms:
#     if s.name == 'IDALIA':
#         storm_dict[s.name] = {}
#         storm_dict[s.name]['track'] = pd.DataFrame({"date": s.date, "lon": s.lon, "lat": s.lat}).set_index('date')
#         storm_dict[s.name]['cone'] = {}

#         for t in date_list:
#             storm_dict[s.name]['cone'][t] = generate_nhc_cone(s.get_nhc_forecast_dict(t), s.basin, cone_days=5)

# kwargs['storms'] = storm_dict

# storm_dict = {}
# # storms = [realtime_obj.get_storm(key) for key in realtime_obj.list_active_storms(basin='north_atlantic')]
# from tropycal import tracks
# # import pandas as pd
# from tropycal import realtime

# basin = tracks.TrackDataset(basin='north_atlantic', source='ibtracs', include_btk=False)
# storms = basin.get_storm(('idalia', 2023))

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# for ctime in date_list:
def plot_ctime(ctime):
    print(f"Checking if {ctime} exists for each model.")
    
    if plot_rtofs:
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

    if plot_para:
        try:
            rdsp_time = rdsp.sel(time=ctime)
            # startTime = time.time()
            rdsp_time['density'] = xr.apply_ufunc(density, 
                                    rdsp_time['temperature'], 
                                    -rdsp_time['depth'],
                                    rdsp_time['salinity'], 
                                    rdsp_time['lat'], 
                                    rdsp_time['lon']
                                    )
            rdsp_time['ohc'] = xr.apply_ufunc(ocean_heat_content, 
                                rdsp_time.depth, 
                                rdsp_time.temperature, 
                                rdsp_time.density, 
                                input_core_dims=[['depth'], ['depth'], ['depth']], 
                                vectorize=True)
            # print('RTOFS - Execution time in seconds: ' + str(time.time() - startTime))
            print(f"RTOFS Parallel: True")
            rdtp_flag = True
        except KeyError as error:
            print(f"RTOFS Parallel: False")
            rdtp_flag = False  

    if plot_espc:
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
        extent = configs['extent']

        # Increase the extent a little bit to grab slightly more data than the region 
        # we want to plot. Otherwise, we will have areas of the plot with no data.
        extent_data = np.add(extent, [-1, 1, -1, 1]).tolist()

        # convert from 360 to 180 lon
        lon360 = lon180to360(extent_data[:2])

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

        if rdtp_flag:
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
            rdsp_slice = rdsp_time.sel(
                x=slice(extent_ind[0], extent_ind[1]), 
                y=slice(extent_ind[2], extent_ind[3])
                )
            
        if gdt_flag:
            # subset dataset to the proper extents for each region
            gds_slice = gds_time.sel(
                lon=slice(extent_data[0], extent_data[1]),
                lat=slice(extent_data[2], extent_data[3])
            )

            # Convert from 0,360 lon to -180,180
            # gds_slice['lon'] = lon360to180(gds_slice['lon'])

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
            
        try:
            if rdt_flag and gdt_flag:
                plot_ohc(rds_slice, gds_slice, extent, configs['name'], **kwargs)
                
            if rdt_flag and cdt_flag:
                plot_ohc(rds_slice, cds_slice, extent, configs['name'], **kwargs)

            if rdt_flag and amt_flag:
                plot_ohc(rds_slice, amt_slice, extent, configs['name'], **kwargs)

            if rdt_flag and rdtp_flag:
                plot_ohc(rds_slice, rdsp_slice, extent, configs['name'], **kwargs)

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