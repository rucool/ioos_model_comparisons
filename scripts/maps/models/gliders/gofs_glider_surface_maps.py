#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/11/2021
Create surface maps of GOFS temperature and salinity overlaid with a user-specified glider track.
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import datetime as dt
import hurricanes.gliders as gld
from hurricanes.gliders_plt import surface_map_glider_track
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, bathymetry, m_t0, m_t1, g_t0, g_t1, lt, current_glider_location):
    url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = g_t0
    gargs['time_end'] = g_t1

    # initialize keyword arguments for map plot
    kwargs = dict()
    kwargs['model'] = 'gofs'
    kwargs['transform'] = ccrs.PlateCarree()
    kwargs['current_glider_loc'] = current_glider_location

    if bathymetry:
        bath = xr.open_dataset(bathymetry)
    else:
        bath = False

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

        if lt:
            lt = gld.custom_gliderline_transects()
            kwargs['custom_transect'] = dict(lon=[lt[glider]['extent'][0], lt[glider]['extent'][2]],
                                             lat=[lt[glider]['extent'][1], lt[glider]['extent'][3]])

        # Loop through regions
        for region in glider_region.items():
            extent = region[1]['lonlat']
            if bath:
                kwargs['bathy'] = bath.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                           lat=slice(extent[2] - .1, extent[3] + .1))

            with xr.open_dataset(url, drop_variables='tau') as gofs:
                gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

                # Subset time range
                if m_t0:
                    mt0 = m_t0
                else:
                    mt0 = gl_t0 - dt.timedelta(hours=6)
                if m_t1:
                    mt1 = m_t1
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


if __name__ == '__main__':
    glider_deployments = ['ru30-20210503T1929']
    sdir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
    bathy = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    # bathy = False
    model_t0 = dt.datetime(2021, 5, 10, 0, 0)  # False
    model_t1 = False
    glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
    glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
    line_transect = True  # True or False  # get a straight line transect, rather than a transect along the glider track
    curr_location = False  # indicate the current glider location with a triangle marker
    main(glider_deployments, sdir, bathy, model_t0, model_t1, glider_t0, glider_t1, line_transect, curr_location)
