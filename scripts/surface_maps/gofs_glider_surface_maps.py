#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/7/2021
Create surface maps of GOFS temperature and salinity overlaid with a user-specified glider track.
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import datetime as dt
import src.gliders as gld
from src.gliders_plt import surface_map_glider_track
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


gliders = ['maracoos_02-20210503T1937']
url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
bathymetry = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
#bathymetry = False
map_projection = ccrs.PlateCarree()
dpi = 150
model_t0 = dt.datetime(2021, 5, 7, 0, 0)  # False
model_t1 = False
glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
glider_t1 = False  # dt.datetime(2021, 5, 5, 12, 0)


# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = glider_t0
gargs['time_end'] = glider_t1

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'gofs'
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

    # Loop through regions
    for region in glider_region.items():
        extent = region[1]['lonlat']
        if bathy:
            kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                        lat=slice(extent[2] - .1, extent[3] + .1))

        with xr.open_dataset(url, drop_variables='tau') as gofs:
            gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

            # Subset time range
            if model_t0:
                mt0 = model_t0
            else:
                mt0 = gl_t0 - dt.timedelta(hours=6)
            if model_t1:
                mt1 = model_t1
            else:
                mt1 = gl_t1 + dt.timedelta(days=1)
            ds = gofs.sel(time=slice(mt0, mt1))
            for t in ds.time:
                print(f'Accessing GOFS: {str(t.dt.strftime("%Y-%m-%d %H:%M:%S").data)}')
                tds = ds.sel(time=t)  # Select the latest time

                # subset dataset to the proper extents for each region
                sub = tds.sel(lon=slice(extent[0] + 359, extent[1] + 361), lat=slice(extent[2] - 1, extent[3] + 1))
                sub['lon'] = sub['lon'] - 360  # Convert model lon to glider lon
                surface_map_glider_track(sub, region, **kwargs)
