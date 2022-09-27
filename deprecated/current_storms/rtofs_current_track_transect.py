#! /usr/bin/env python3

"""
Author: Mike Smith
Last modified: Lori Garzio on 7/2/2021
Create a transect of RTOFS temperature and salinity by latitude and longitude for each model forecast for today under
current storm forecast tracks.
"""
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os
from glob import glob
from ioos_model_comparisons.storms_plt import plot_transect
from current_storms import current_forecast_track
import ioos_model_comparisons.storms as storms

# url = '/Users/garzio/Documents/rucool/hurricane_glider_project/RTOFS/RTOFS_6hourly_North_Atlantic/'
url = '/home/hurricaneadm/data/rtofs'  # on server
# save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/current_storm_tracks'
save_dir = '/www/home/lgarzio/public_html/hurricanes/current_storm_tracks'  # in server
days = 2  # number of days to search including today's date for RTOFS data (e.g. 2 = search for today and yesterday)
color_lims = dict(temp=dict(shallow=np.arange(6, 32)),
                  salt=dict(shallow=np.arange(34, 37.2, .2)))

today = dt.date.today()
now = dt.datetime.utcnow()
sdir = os.path.join(save_dir, now.strftime('%Y%m%d'), now.strftime('%Y%m%dT%H'))

# get forecast tracks for today
forecast_tracks = current_forecast_track.main(now, save_dir)

# define times to grab RTOFS data
date_list = [today - dt.timedelta(days=x) for x in range(days)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

# get RTOFS files for 0Z on the first day
md = (dt.datetime.strptime(rtofs_files[0].split('/')[-2].split('.')[-1], '%Y%m%d') - dt.timedelta(days=1)).strftime('rtofs.%Y%m%d')
rtofs_files.insert(0, os.path.join(url, md, 'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc'))

for f in rtofs_files:
    print(f)
    try:
        with xr.open_dataset(f) as ds:
            ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
            ds = ds.sel(depth=slice(0, 500))

            for tracks in forecast_tracks.items():
                ft = tracks[1]['forecast_track']
                targetlon, targetlat = storms.return_target_transect(ft['lon'], ft['lat'])

                date_str = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d %H:%M:%S')
                date_save = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%dT%H%M%SZ')
                save_dir_storm = os.path.join(save_dir, now.strftime('%Y%m%d'), now.strftime('%Y%m%dT%H'), tracks[0], 'transects', 'rtofs')
                os.makedirs(save_dir_storm, exist_ok=True)

                # get the custom temperature transect along the storm track
                print('Getting RTOFS custom temperature transect')
                temp, depth, lon_sub, lat_sub = storms.custom_transect(ds, 'temperature', targetlon, targetlat, 'rtofs')

                # plot temperature by longitude
                targs = {}
                targs['cmap'] = cmocean.cm.thermal
                targs['clab'] = 'Temperature ($^oC$)'
                targs['title'] = f'{tracks[0]}: Storm Track Forecast on {tracks[1]["forecast_time"].strftime("%Y-%m-%d %H:%M UTC")}\n RTOFS Temperature at {date_str} UTC'
                targs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_rtofs_transect_temp-lon-{date_save}.png')
                targs['levels'] = color_lims['temp']
                print('plotting temperature by longitude')
                plot_transect(lon_sub, -depth, temp, **targs)

                # plot temperature by latitude
                targs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_rtofs_transect_temp-lat-{date_save}.png')
                targs['xlab'] = 'Latitude'
                print('plotting temperature by latitude')
                plot_transect(lat_sub, -depth, temp, **targs)

                # get the custom salinity transect along the storm track
                print('Getting RTOFS custom salinity transect')
                salt, depth, lon_sub, lat_sub = storms.custom_transect(ds, 'salinity', targetlon, targetlat, 'rtofs')

                # plot salinity by longitude
                sargs = {}
                sargs['cmap'] = cmocean.cm.haline
                sargs['title'] = f'{tracks[0]}: Storm Track Forecast on {tracks[1]["forecast_time"].strftime("%Y-%m-%d %H:%M UTC")}\n RTOFS Salinity at {date_str} UTC'
                sargs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_rtofs_transect_salt-lon-{date_save}.png')
                sargs['levels'] = color_lims['salt']
                print('plotting salinity by longitude')
                plot_transect(lon_sub, -depth, salt, **sargs)

                # plot salinity by latitude
                sargs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_rtofs_transect_salt-lat-{date_save}.png')
                sargs['xlab'] = 'Latitude'
                print('plotting salinity by latitude')
                plot_transect(lat_sub, -depth, salt, **sargs)

    except OSError:
        continue
