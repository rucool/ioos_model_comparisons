#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/7/2021
Create surface maps of RTOFS temperature and salinity overlaid with a user-specified glider track.
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import cartopy.crs as ccrs
import datetime as dt
import src.gliders as gld
from src.gliders_plt import surface_map_glider_track
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


gliders = ['maracoos_02-20210503T1937']
url = '/Users/garzio/Documents/rucool/hurricane_glider_project/RTOFS/RTOFS_6hourly_North_Atlantic/'
# url = '/home/hurricaneadm/data/rtofs'  # on server
save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
bathymetry = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
#bathymetry = False
map_projection = ccrs.PlateCarree()
dpi = 150
model_t0 = dt.datetime(2021, 5, 6, 0, 0)  # False
model_t1 = False
glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
glider_t1 = False  # dt.datetime(2021, 5, 5, 12, 0)
line_transect = True  # True or False  # get a straight line transect, rather than a transect along the glider track

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = glider_t0
gargs['time_end'] = glider_t1

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['transform'] = map_projection
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)
else:
    bathy = False

for glider in gliders:
    sdir_glider = os.path.join(save_dir, glider)
    os.makedirs(sdir_glider, exist_ok=True)
    kwargs['save_dir'] = sdir_glider

    glider_ds = gld.glider_dataset(glider, **gargs)
    kwargs['gliders'] = glider_ds
    glider_region = gld.glider_region(glider_ds)  # define the glider region

    gl_t0 = pd.to_datetime(np.nanmin(glider_ds.time.values))
    gl_t1 = pd.to_datetime(np.nanmax(glider_ds.time.values))

    if len(glider_region) < 1:
        raise ValueError('No region found for glider: {}'.format(glider))

    if line_transect:
        line_transect = gld.custom_gliderline_transects()
        x1 = line_transect[glider]['extent'][0]
        y1 = line_transect[glider]['extent'][1]
        x2 = line_transect[glider]['extent'][2]
        y2 = line_transect[glider]['extent'][3]
        kwargs['custom_transect'] = dict(lon=[x1, x2], lat=[y1, y2])

    # get RTOFS files
    if model_t0:
        mt0 = model_t0
    else:
        mt0 = gl_t0 - dt.timedelta(hours=6)
    if model_t1:
        mt1 = model_t1
    else:
        mt1 = gl_t1 + dt.timedelta(days=1)

    model_dates = [x.strftime('rtofs.%Y%m%d') for x in pd.date_range(mt0, mt1)]
    rtofs_files = [glob.glob(os.path.join(url, x, '*.nc')) for x in model_dates]
    rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

    # Loop through regions
    for region in glider_region.items():
        extent = region[1]['lonlat']
        if bathy:
            kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                        lat=slice(extent[2] - .1, extent[3] + .1))

        for f in rtofs_files:
            print(f)
            try:
                with xr.open_dataset(f) as ds:
                    ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
                    lat = ds.lat.data
                    lon = ds.lon.data

                    # subset the RTOFS grid
                    lonIndex = np.round(np.interp(extent[:2], lon[0, :], np.arange(0, len(lon[0, :])))).astype(int)
                    latIndex = np.round(np.interp(extent[2:], lat[:, 0], np.arange(0, len(lat[:, 0])))).astype(int)
                    sub = ds.sel(X=slice(lonIndex[0], lonIndex[1]), Y=slice(latIndex[0], latIndex[1]))
                    surface_map_glider_track(sub, region, **kwargs)
            except OSError:
                continue
