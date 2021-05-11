#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/7/2021
Last modified: Lori Garzio on 5/7/2021
Create a map of user-specified glider tracks
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import datetime as dt
import src.gliders as gld
from src.gliders_plt import glider_track
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


gliders = ['maracoos_02-20210503T1937', 'ru30-20210503T1929']
save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
bathymetry = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
#bathymetry = False
landcolor = 'none'
dpi = 150
glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
glider_t1 = False  # dt.datetime(2021, 5, 5, 12, 0)
line_transect = False  # True or False  # get a straight line transect, rather than a transect along the glider track
current_glider_location = False  # indicate the current glider location with a triangle marker

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = glider_t0
gargs['time_end'] = glider_t1

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['dpi'] = dpi
kwargs['landcolor'] = landcolor
kwargs['current_glider_loc'] = current_glider_location

if bathymetry:
    bathy = xr.open_dataset(bathymetry)
else:
    bathy = False

for glider in gliders:
    sdir_glider = os.path.join(save_dir, glider)
    os.makedirs(sdir_glider, exist_ok=True)
    kwargs['save_dir'] = sdir_glider

    glider_ds = gld.glider_dataset(glider, **gargs)
    glider_region = gld.glider_region(glider_ds)  # define the glider region

    gl_t0 = pd.to_datetime(np.nanmin(glider_ds.time.values))
    gl_t1 = pd.to_datetime(np.nanmax(glider_ds.time.values))

    if line_transect:
        line_transect = gld.custom_gliderline_transects()
        x1 = line_transect[glider]['extent'][0]
        y1 = line_transect[glider]['extent'][1]
        x2 = line_transect[glider]['extent'][2]
        y2 = line_transect[glider]['extent'][3]
        kwargs['custom_transect'] = dict(lon=[x1, x2], lat=[y1, y2])

    # Loop through regions
    for region in glider_region.items():
        if region[0] == 'zoomed':
            extent = region[1]['lonlat']
            if bathy:
                kwargs['bathy'] = bathy.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                            lat=slice(extent[2] - .1, extent[3] + .1))

            glider_track(glider_ds, region, **kwargs)
