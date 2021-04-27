#! /usr/bin/env python3

"""
Author: Mike Smith
Last modified: Lori Garzio on 4/22/2021
"""
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os
from src.storms_plt import plot_transect
from current_storms import current_forecast_track
import src.storms as storms

url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/current_storm_tracks'
color_lims = dict(temp=dict(shallow=np.arange(6, 32)),
                  salt=dict(shallow=np.arange(31.2, 37.2, .2)))

today = dt.date.today()

# get forecast tracks for today
forecast_tracks = current_forecast_track.main(today)

with xr.open_dataset(url, drop_variables='tau') as ds:
    ds = ds.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

    # Get all the datetimes for today at 0-500m
    ds = ds.sel(time=slice(pd.to_datetime(today), today + dt.timedelta(days=1)), depth=slice(0, 500))

    for tracks in forecast_tracks.items():
        ft = tracks[1]['forecast_track']
        targetlon, targetlat = storms.return_target_transect(ft['lon'], ft['lat'])
        targetlon_GOFS = storms.convert_target_gofs_lon(targetlon)

        for t in ds.time:
            gofs = ds.sel(time=t)
            date_str = pd.to_datetime(t.values).strftime('%Y-%m-%d %H:%M:%S')
            date_save = pd.to_datetime(t.values).strftime('%Y-%m-%dT%H%M%SZ')
            save_dir_storm = os.path.join(save_dir, today.strftime('%Y%m%d'), tracks[0], 'transects', 'gofs')
            os.makedirs(save_dir_storm, exist_ok=True)

            # get the custom temperature transect along the storm track
            print('Getting GOFS custom temperature transect')
            temp, depth, lon_sub, lat_sub = storms.custom_transect(gofs, 'temperature', targetlon_GOFS, targetlat, 'gofs')

            # plot temperature by longitude
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['clab'] = 'Temperature ($^oC$)'
            targs['title'] = f'{tracks[0]}: Storm Track Forecast on {today.strftime("%Y-%m-%d")}\n GOFS Temperature at {date_str} UTC'
            targs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_gofs_transect_temp-lon-{date_save}.png')
            #targs['levels'] = dict(shallow=np.arange(6, 32))
            targs['levels'] = color_lims['temp']
            print('plotting temperature by longitude')
            plot_transect(lon_sub, -depth, temp, **targs)

            # plot temperature by latitude
            targs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_gofs_transect_temp-lat-{date_save}.png')
            targs['xlab'] = 'Latitude'
            print('plotting temperature by latitude')
            plot_transect(lat_sub, -depth, temp, **targs)

            # get the custom salinity transect along the storm track
            print('Getting GOFS custom salinity transect')
            salt, depth, lon_sub, lat_sub = storms.custom_transect(gofs, 'salinity', targetlon_GOFS, targetlat, 'gofs')

            # plot salinity by longitude
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['title'] = f'{tracks[0]}: Storm Track Forecast on {today.strftime("%Y-%m-%d")}\n GOFS Salinity at {date_str} UTC'
            sargs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_gofs_transect_salt-lon-{date_save}.png')
            #sargs['levels'] = dict(shallow=np.arange(31.2, 37.2, .2))
            sargs['levels'] = color_lims['salt']
            print('plotting salinity by longitude')
            plot_transect(lon_sub, -depth, salt, **sargs)

            # plot salinity by latitude
            sargs['save_file'] = os.path.join(save_dir_storm, f'{tracks[0]}_gofs_transect_salt-lat-{date_save}.png')
            sargs['xlab'] = 'Latitude'
            print('plotting salinity by latitude')
            plot_transect(lat_sub, -depth, salt, **sargs)
