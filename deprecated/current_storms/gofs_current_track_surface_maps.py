#! /usr/bin/env python3

"""
Author: Mike Smith
Last modified: Lori Garzio on 8/17/2021
Create surface maps of GOFS temperature and salinity for each model forecast for today, overlaid with current storm
forecast tracks released today.
"""
import xarray as xr
import pandas as pd
import os
import cartopy.crs as ccrs
import datetime as dt
from hurricanes.platforms import get_argo_data, forecast_storm_region
from hurricanes.storms_plt import surface_map_storm_forecast
from current_storms import current_forecast_track
import hurricanes.gliders as gld
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
#save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/current_storm_tracks'
save_dir = '/www/home/lgarzio/public_html/hurricanes/current_storm_tracks'  # in server
#bathymetry = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
bathymetry = False
map_projection = ccrs.PlateCarree()
argo = True
gliders = True
dpi = 150
search_time = 48  # number of hours to search previous to today's date for gliders/argo
days = 1  # number of days to search previous to today's date for GOFS data

today = dt.date.today()
now = dt.datetime.utcnow()
sdir = os.path.join(save_dir, now.strftime('%Y%m'), now.strftime('%Y%m%d'), now.strftime('%Y%m%dT%H'))

# define times to grab GOFS data, matching the resolution of RTOFS data (every 6 hours)
time_start = today - dt.timedelta(days=days)
time_end = today + dt.timedelta(days=1)
ranges = pd.date_range(time_start, time_end, freq='6H')

# define times to search for gliders and argo floats
t0 = today - dt.timedelta(hours=search_time)
t1 = today + dt.timedelta(hours=24)

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'gofs'
kwargs['transform'] = map_projection
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)
else:
    bathy = False

# get forecast tracks for today
forecast_tracks = current_forecast_track.main(now, save_dir)

if forecast_tracks:
    for tracks in forecast_tracks.items():
        kwargs['forecast'] = tracks
        stm_region = forecast_storm_region(tracks[1]['forecast_track'])

        if len(stm_region) < 1:
            print('No region found for storm: {}'.format(tracks[0]))
        else:
            sdir_track = os.path.join(sdir, tracks[0])
            os.makedirs(sdir_track, exist_ok=True)
            kwargs['save_dir'] = sdir_track

            # Loop through regions - there should only be 1 for the current storm
            for region in stm_region.items():
                extent = region[1]['lonlat']
                if bathy:
                    kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - 1, extent[1] + 1),
                                                lat=slice(extent[2] - 1, extent[3] + 1))

                if argo:
                    argo_data = get_argo_data(extent, t0, t1)
                    if len(argo_data) > 0:
                        kwargs['argo'] = argo_data

                        argo_df = pd.DataFrame(argo_data)
                        argo_df.sort_values('time', inplace=True)
                        argo_savename = '{}_{}_argo_{}-{}.csv'.format(tracks[0], region[1]['code'],
                                                                      t0.strftime('%Y%m%d'), t1.strftime('%Y%m%d'))
                        argo_df.to_csv(os.path.join(sdir_track, argo_savename), index=False)
                    else:
                        kwargs['argo'] = False

                if gliders:
                    current_gliders = gld.glider_data(extent, t0, t1)
                    if len(current_gliders) > 0:
                        kwargs['gliders'] = current_gliders
                        gl_savename = '{}_{}_gliders_{}-{}.csv'.format(tracks[0], region[1]['code'],
                                                                       t0.strftime('%Y%m%d'), t1.strftime('%Y%m%d'))
                        gld.glider_summary(current_gliders, os.path.join(sdir_track, gl_savename))
                    else:
                        kwargs['gliders'] = False

                with xr.open_dataset(url, drop_variables='tau') as gofs:
                    gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

                    # Select date range
                    ds = gofs.sel(time=ranges[ranges <= pd.to_datetime(gofs.time.max().data)])
                    for t in ds.time:
                        print(f'Accessing GOFS: {str(t.dt.strftime("%Y-%m-%d %H:%M:%S").data)}')
                        tds = ds.sel(time=t)  # Select the latest time

                        # subset dataset to the proper extents for each region
                        sub = tds.sel(lon=slice(extent[0] + 359, extent[1] + 361), lat=slice(extent[2] - 2, extent[3] + 2))
                        sub['lon'] = sub['lon'] - 360  # Convert model lon to glider lon
                        surface_map_storm_forecast(sub, region, **kwargs)
