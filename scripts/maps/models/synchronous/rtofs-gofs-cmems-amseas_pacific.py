import datetime as dt

import ioos_model_comparisons.configs as conf
import numpy as np
import pandas as pd
from ioos_model_comparisons.calc import lon180to360, lon360to180
from ioos_model_comparisons.models import rtofs, amseas, CMEMS, espc
from ioos_model_comparisons.platforms import (get_active_gliders, 
                                  get_argo_floats_by_time)
from ioos_model_comparisons.plotting import (plot_model_region_comparison,
                                 plot_model_region_comparison_streamplot,
                                 plot_sst
                                 )
from ioos_model_comparisons.regions import region_config
import matplotlib
import time
import xarray as xr
import copy
from cool_maps.plot import get_bathymetry

startTime = time.time() # Start time to see how long the script took
matplotlib.use('agg')

# Set path to save plots
path_save = (conf.path_plots / "maps")

# Which models should we plot?
plot_rtofs = True
plot_para = False
plot_espc = True # ESPC
plot_cmems = True
plot_amseas = False
plot_cnaps = False

# initialize keyword arguments for map plots
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = False
kwargs['colorbar'] = True

# For debug purposes. Comment this out when commiting to repo.
conf.days = 2
conf.regions = ['hawaii', 'mexico_pacific']

# Get today and yesterday dates
today = dt.date.today()
date_end = today + dt.timedelta(days=1)
date_start = today - dt.timedelta(days=conf.days)

freq = '12H'

# Formatter for time
tstr = '%Y-%m-%d %H:%M:%S'

# Create dates that we want to plot
date_list = pd.date_range(date_start, date_end, freq=freq)
date_list_2 = pd.date_range(date_start - dt.timedelta(days=1), date_end, freq=freq)

# sstdata = xr.open_dataset('http://basin.ceoe.udel.edu/thredds/dodsC/GOESNOAASST.nc')
# sstdata = sstdata.rename({'latitude': 'lat', 'longitude': 'lon'})

# unique_times = ~sstdata['time'].to_series().duplicated()
# sst_unique = sstdata.isel(time=unique_times)
# sst_sorted = sst_unique.sortby('time')

# This is the initial time to start the search for argo/gliders
search_start = date_list[0] - dt.timedelta(hours=conf.search_hours)

# Get extent for all configured regions to download argo/glider data one time
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
    glider_data = get_active_gliders(global_extent, search_start, date_end, parallel=False, timeout=60)
else:
    glider_data = pd.DataFrame()

if conf.bathy:
    bathy_data = get_bathymetry(global_extent)

# Load RTOFS DataSet
print('Loading RTOFS')
rds = rtofs(source="west") 
print('RTOFS loaded')

# Save rtofs lon and lat as variables to speed up indexing calculation
grid_lons = rds.lon.values[0,:]
grid_lats = rds.lat.values[:,0]
grid_x = rds.x.values
grid_y = rds.y.values

if plot_para:
    # RTOFS Parallel
    # from glob import glob
    # import xarray as xr
    # import os
    # url = '/Users/mikesmith/Downloads/rtofs.parallel.v2.3/'
    # rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list_2]
    # rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])


    # rtofs_para = xr.open_mfdataset(rtofs_files)
    # rtofs_para = rtofs_para.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth', 'X': 'x', 'Y': 'y'})
    rtofs_para = rtofs(source='parallel') 
    print('RTOFS Parallel loaded')

    rtofs_para.attrs['model'] = 'RTOFS (Parallel)'


if plot_espc:
    print('Loading ESPC')
    # Load espc DataSet
    gds = espc(rename=True)
    print('ESPC loaded')

if plot_cmems:
    print('Loading CMEMS')
    # Load Copernicus
    cds = CMEMS() 
    print('CMEMS loaded')

if plot_amseas:
    print('Loading AMSEAS')
    # Load AMSEAS
    am = amseas(rename=True)
    print('AMSEAS loaded')

if plot_cnaps:
    print('Loading CNAPS')
    # Load CNAPS
    cn = cnaps(rename=True)
    print('CNAPS loaded')
    grid_lons_cn = cn.lon.values[0,:]
    grid_lats_cn = cn.lat.values[:,0]
    grid_x_cn = cn.xi_rho.values
    grid_y_cn = cn.eta_rho.values

def main():
    # Loop through times
    for ctime in date_list:
        print(f"Checking if {ctime} exists for each model.")
        try:
            rdt = rds.sel(time=ctime)
            print(f"RTOFS: True")
            rdt_flag = True
        except KeyError as error:
            print(f"RTOFS: False - {error}")
            rdt_flag = False
            
        if plot_para:
            try:
                rdtp = rtofs_para.sel(time=ctime)
                print(f"RTOFS Para: True")
                rdtp_flag = True
            except KeyError as error:
                print(f"RTOFS Para: False - {error}")
                rdtp_flag = False
        else:
            rdtp_flag = False

        if plot_espc:
            try:
                gdt = gds.sel(time=ctime)
                print(f"ESPC: True")
                gdt_flag = True
            except KeyError as error:
                print(f"ESPC: False - {error}")
                gdt_flag = False
        else:
            gdt_flag = False

        if plot_cmems:
            try:
                cdt = cds.data.sel(time=ctime)
                cdt.attrs['model'] = 'CMEMS'
                # cds.load(global_extent, ctime)
                # cdt = cds.data
                print(f"CMEMS: True")
                cdt_flag = True
            except KeyError as error:
                print(f"CMEMS: False - {error}")
                cdt_flag = False
        else:
            cdt_flag = False

        if plot_amseas:
            try:
                amt = am.sel(time=ctime)
                print(f"AMSEAS: True")
                amt_flag = True
            except KeyError as error:
                print(f"AMSEAS: False - {error}")
                amt_flag = False
        else:
            amt_flag = False

        if plot_cnaps:
            try:
                cnt = cn.sel(time=ctime)
                print(f"CNAPS: True")
                cnt_flag = True
            except KeyError as error:
                print(f"CNAPS: False - {error}")
                cnt_flag = False
        else:
            cnt_flag = False
            
        print("\n")
            
        search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
        search_window_t1 = ctime.strftime(tstr) 
        
        # Loop through regions
        for item in conf.regions:
            region = region_config(item)
            extent = region['extent']
            print(f'Region: {region["name"]}, Extent: {extent}')
            kwargs['path_save'] = path_save / region['folder']

            if 'eez' in region:
                kwargs["eez"] = region["eez"]

            if region['currents']['bool']:
                kwargs['currents'] = region['currents']

            if 'figure' in region:
                if 'legend' in region['figure']:
                    kwargs['cols'] = region['figure']['legend']['columns']

                if 'figsize' in region['figure']:
                    kwargs['figsize'] = region['figure']['figsize']

            try:
                kwargs['bathy'] = bathy_data.sel(
                    longitude=slice(extent[0] - 1, extent[1] + 1),
                    latitude=slice(extent[2] - 1, extent[3] + 1)
                )
            except NameError:
                pass
                    
            extended = np.add(extent, [-1, 1, -1, 1]).tolist()
            lon360 = lon180to360(extended[:2]) # convert from 360 to 180 lon

            # Find x, y indexes of the area we want to subset
            lons_ind = np.interp(extended[:2], grid_lons, grid_x)
            lats_ind = np.interp(extended[2:], grid_lats, grid_y)

            # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice 
            # in order to widen the area of the extent slightly.
            extent_ind = [
                np.floor(lons_ind[0]).astype(int),
                np.ceil(lons_ind[1]).astype(int),
                np.floor(lats_ind[0]).astype(int),
                np.ceil(lats_ind[1]).astype(int)
                ]
            
            # Use .isel selector on x/y since we know indexes that we want to slice
            rds_sub = rdt.isel(
                x=slice(extent_ind[0], extent_ind[1]), 
                y=slice(extent_ind[2], extent_ind[3])
                ).set_coords(['u', 'v'])

            if rdtp_flag:
                rdtp_sub = rdtp.isel(
                    x=slice(extent_ind[0], extent_ind[1]), 
                    y=slice(extent_ind[2], extent_ind[3])
                    ).set_coords(['u', 'v'])

            if cnt_flag:
                # CNAPS
                # Find x, y indexes of the area we want to subset
                lons_ind_cn = np.interp(extended[:2], grid_lons_cn, grid_x_cn)
                lats_ind_cn = np.interp(extended[2:], grid_lats_cn, grid_y_cn)

                # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice 
                # in order to widen the area of the extent slightly.
                extent_ind_cn = [
                    np.floor(lons_ind_cn[0]).astype(int),
                    np.ceil(lons_ind_cn[1]).astype(int),
                    np.floor(lats_ind_cn[0]).astype(int),
                    np.ceil(lats_ind_cn[1]).astype(int)
                    ]
                
                cnt_sub = cnt.isel(
                    xi_rho=slice(extent_ind[0], extent_ind[1]), 
                    eta_rho=slice(extent_ind[2], extent_ind[3])
                    ).set_coords(['u', 'v'])

            if gdt_flag:
                # subset dataset to the proper extents for each region
                gds_sub = gdt.sel(
                    lon=slice(lon360[0], lon360[1]),
                    lat=slice(extended[2], extended[3])
                ).set_coords(['u', 'v'])
                
                # Convert from 0,360 lon to -180,180
                gds_sub['lon'] = lon360to180(gds_sub['lon'])

            if cdt_flag:
                cds_sub = cdt.sel(
                    lon=slice(extended[0], extended[1]),
                    lat=slice(extended[2], extended[3])
                ).set_coords(['u', 'v'])

            if amt_flag:
                am_sub = amt.sel(
                    lon=slice(lon360[0], lon360[1]),
                    lat=slice(extended[2], extended[3])
                ).set_coords(['u', 'v'])

            # Check if any asset data was downloaded and subset it to the 
            # region and time being plotted
            # ARGO
            if not argo_data.empty:
                argo_lon = argo_data['lon']
                argo_lat = argo_data['lat']
                argo_region = argo_data[
                    (extended[0] <= argo_lon) & (argo_lon <= extended[1]) & (extended[2] <= argo_lat) & (argo_lat <= extended[3])
                ]
                argo_region.sort_index(inplace=True)
                idx = pd.IndexSlice
                kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

            # Gliders
            if not glider_data.empty:
                glider_lon = glider_data['lon']
                glider_lat = glider_data['lat']
                glider_region = glider_data[
                    (extended[0] <= glider_lon) & (glider_lon <= extended[1]) & (extended[2] <= glider_lat) & (glider_lat <= extended[3])
                    ]
                glider_region = glider_region[
                    (search_window_t0 <= glider_region.index.get_level_values('time'))
                    &
                    (glider_region.index.get_level_values('time') <= search_window_t1)
                    ]
                kwargs['gliders'] = glider_region
                
            def kelvin_to_celsius(kelvin_temps):
                # Convert each Kelvin temperature to Celsius
                celsius_temps = [k - 273.15 for k in kelvin_temps]
                return celsius_temps
            
            try:
                if rdt_flag and gdt_flag:
                    plot_model_region_comparison(rds_sub, gds_sub, region, **kwargs)
                    plot_model_region_comparison_streamplot(rds_sub, gds_sub, region, **kwargs)
            except Exception as e:
                print(f"Failed to process RTOFS vs ESPC at {ctime}")
                print(f"Error: {e}")

            try:
                if rdt_flag and cdt_flag:
                    plot_model_region_comparison(rds_sub, cds_sub.squeeze(), region, **kwargs)
                    plot_model_region_comparison_streamplot(rds_sub, cds_sub.squeeze(), region, **kwargs)
            except Exception as e:
                print(f"Failed to process RTOFS vs CMEMS at {ctime}")
                print(f"Error: {e}")

            try:
                if rdt_flag and rdtp_flag:
                    plot_model_region_comparison(rds_sub, rdtp_sub, region, **kwargs)
                    plot_model_region_comparison_streamplot(rds_sub, rdtp_sub, region, **kwargs)
            except Exception as e:
                print(f"Failed to process RTOFS vs RTOFS Para at {ctime}")
                print(f"Error: {e}")
                
            try:
                if rdt_flag and amt_flag:
                    plot_model_region_comparison(rds_sub, am_sub, region, **kwargs)
                    plot_model_region_comparison_streamplot(rds_sub, am_sub, region, **kwargs) 
            except Exception as e:
                print(f"Failed to process RTOFS vs AMSEAS at {ctime}")
                print(f"Error: {e}")
                
            # try:
            #     if rdt_flag and cnt_flag:
            #         plot_model_region_comparison(rds_sub, gds_sub, region, **kwargs)
            #         plot_model_region_comparison_streamplot(rds_sub, cnt_sub, region, **kwargs)
            # except Exception as e:
            #     print(f"Failed to process RTOFS vs GOFS at {ctime}")
            #     print(f"Error: {e}")
            # sst = sst_sorted.sel(
            #     lon=slice(extent[0],extent[1]),
            #     lat=slice(extent[2],extent[3])
            #     ).sel(time=str(ctime), method='nearest')

            # temp = kelvin_to_celsius(sst['SST'])
            # sst['SST_C'] = (('lat', 'lon'), temp)

            # swargs = copy.deepcopy(kwargs)
            # try:
            #     del swargs['eez']
            #     del swargs['currents']
            # except KeyError:
            #     pass
            # plot_sst(rds_sub, sst, region, **swargs)
 

if __name__ == "__main__":
    main()
    print('Execution time in seconds: ' + str(time.time() - startTime))