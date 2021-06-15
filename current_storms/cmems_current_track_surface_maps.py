#! /usr/bin/env python3

"""
Author: Mike Smith
Last modified: Lori Garzio on 4/27/2021
Create surface maps of CMEMS temperature and salinity for each model forecast for today, overlaid with current storm
forecast tracks released today.
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
import cartopy.crs as ccrs
import datetime as dt
from src.storms import get_argo_data, forecast_storm_region
from src.storms_plt import surface_map_storm_forecast
from current_storms import current_forecast_track
import src.gliders as gld
from scripts.harvest import grab_cmems

# for downloading the CMEMS dataset
cmems_nc_savedir = '/Users/garzio/Documents/rucool/hurricane_glider_project/CMEMS'
maxdepth = 10
username = 'user'
password = 'pwd'

# for plots
save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/current_storm_tracks'
#bathymetry = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
bathymetry = False
map_projection = ccrs.PlateCarree()
argo = True
gliders = True
dpi = 150
search_time = 48  # number of hours to search previous to today's date for gliders/argo

today = dt.date.today()
sdir = os.path.join(save_dir, today.strftime('%Y%m%d'))
tomorrow = today + dt.timedelta(days=1)

t0 = today - dt.timedelta(hours=search_time)
t1 = today + dt.timedelta(hours=24)

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'cmems'
kwargs['transform'] = map_projection
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)
else:
    bathy = False

# get forecast tracks for today
forecast_tracks = current_forecast_track.main(today)

for tracks in forecast_tracks.items():
    sdir_track = os.path.join(sdir, tracks[0])
    os.makedirs(sdir_track, exist_ok=True)
    kwargs['save_dir'] = sdir_track
    kwargs['forecast'] = tracks
    stm_region = forecast_storm_region(tracks[1]['forecast_track'])

    if len(stm_region) < 1:
        raise ValueError('No region found for storm: {}'.format(tracks[0]))

    # Loop through regions
    for region in stm_region.items():
        extent = region[1]['lonlat']
        dl_extent = np.add(extent, [0, 0, -.5, .5]).tolist()

        # download cmems dataset if it doesn't already exist
        fname = 'cmems_{}_{}_{}m.nc'.format(region[1]['code'], today.strftime('%Y%m%d'), maxdepth)
        ncsavedir = os.path.join(cmems_nc_savedir, today.strftime('%Y%m%d'))
        os.makedirs(ncsavedir, exist_ok=True)
        if not os.path.exists(os.path.join(ncsavedir, fname)):
            print('\nDownloading CMEMS file')
            grab_cmems.download_ds(ncsavedir, fname, today, tomorrow, dl_extent, maxdepth, username, password)

        url = os.path.join(ncsavedir, fname)
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

        with xr.open_dataset(url) as cmems:
            cmems = cmems.rename({'longitude': 'lon', 'latitude': 'lat', 'thetao': 'temperature', 'so': 'salinity'})

            # Get all the datetimes for today
            ds = cmems.sel(time=slice(pd.to_datetime(today), today + dt.timedelta(days=1)))
            for t in ds.time:
                tds = ds.sel(time=t)  # Select the latest time

                if bathy:
                    kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - 1, extent[1] + 1),
                                                lat=slice(extent[2] - 1, extent[3] + 1))

                # subset dataset to the proper extents for each region
                sub = tds.sel(lon=slice(extent[0] - 1, extent[1] + 1), lat=slice(extent[2] - 1, extent[3] + 1))
                surface_map_storm_forecast(sub, region, **kwargs)
