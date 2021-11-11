#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/7/2021
Last modified: Lori Garzio on 5/11/2021
Create a map of user-specified glider tracks
"""
import xarray as xr
import pandas as pd
import os
import datetime as dt
import hurricanes.gliders as gld
from hurricanes.src import glider_track
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, bathymetry, landcolor, glider_t0, glider_t1, current_glider_location):
    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = glider_t0
    gargs['time_end'] = glider_t1

    # initialize keyword arguments for map plot
    kwargs = dict()
    kwargs['landcolor'] = landcolor
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
        glider_region = gld.glider_region(glider_ds)  # define the glider region

        # Loop through regions
        for region in glider_region.items():
            if region[0] == 'zoomed':
                extent = region[1]['lonlat']
                if bath:
                    kwargs['bathy'] = bath.sel(lon=slice(extent[0] - .1, extent[1] + .1),
                                               lat=slice(extent[2] - .1, extent[3] + .1))

                glider_track(glider_ds, region, **kwargs)


if __name__ == '__main__':
    glider_deployments = ['ru30-20210503T1929']
    sdir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
    bathy = '/Users/garzio/Documents/rucool/hurricane_glider_project/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
    # bathy = False
    land_color = 'none'
    glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
    glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
    curr_location = False  # indicate the current glider location with a triangle marker
    main(glider_deployments, sdir, bathy, land_color, glider_t0, glider_t1, curr_location)
