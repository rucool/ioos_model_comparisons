#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/11/2021
Create transects of RTOFS and user-specified glider(s) temperature and salinity. Creates one plot for each model time
and variable for a specified time range.
"""
import datetime as dt
import glob
import os

import cmocean
import numpy as np
import pandas as pd
import xarray as xr

import ioos_model_comparisons.gliders as gld
import ioos_model_comparisons.storms as storms
from ioos_model_comparisons.calc import calculate_transect
from ioos_model_comparisons.gliders_plt import plot_transect, plot_transects
from ioos_model_comparisons.models import rtofs

pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, m_t0, m_t1, g_t0, g_t1, lt, ylims, color_lims):
    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = g_t0
    gargs['time_end'] = g_t1
    gargs['filetype'] = 'dataframe'

    for glider in gliders:
        sdir_glider = os.path.join(save_dir, glider, 'transects', 'rtofs')
        os.makedirs(sdir_glider, exist_ok=True)
        glider_df = gld.glider_dataset(glider, **gargs)
        gl_t0 = pd.to_datetime(np.nanmin(glider_df['time']))
        gl_t1 = pd.to_datetime(np.nanmax(glider_df['time']))

        if lt:  # get the straight line transect
            lt = gld.custom_gliderline_transects()
            x1 = lt[glider]['extent'][0]
            y1 = lt[glider]['extent'][1]
            x2 = lt[glider]['extent'][2]
            y2 = lt[glider]['extent'][3]

            # calculate longitude and latitude of transect lines
            targetlon, targetlat, _ = calculate_transect(x1, y1, x2, y2)

        else:  # get the temperature transect along the glider track
            print('Getting RTOFS custom temperature transect')
            targetlon = np.array(glider_df['longitude'])
            targetlat = np.array(glider_df['latitude'])

        # Subset time range
        if m_t0:
            mt0 = m_t0
        else:
            mt0 = gl_t0 - dt.timedelta(hours=6)
        if m_t1:
            mt1 = m_t1
        else:
            mt1 = gl_t1 + dt.timedelta(days=1)

        with rtofs() as ds:
            ds = ds.sel(depth=slice(0, 500))

            date_str = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d %H:%M:%S')
            date_save = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%dT%H%M%SZ')

            mtemp, mdepth, lon_sub, lat_sub = storms.custom_transect(ds, 'temperature', targetlon, targetlat, 'rtofs')
            gl_tm, gl_lon, gl_lat, gl_depth, gl_temp = gld.grid_glider_data(glider_df, 'temperature', 0.5)

            # plot temperature by longitude - model only
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['clab'] = 'Temperature ($^oC$)'
            targs['title'] = f'RTOFS Temperature at {date_str} UTC\n{glider} glider track'
            targs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_rtofs_transect_temp-{date_save}.png')
            targs['levels'] = color_lims['temp']
            targs['ylims'] = ylims
            print('plotting temperature by longitude')
            plot_transect(lon_sub, -mdepth, mtemp['temperature'], **targs)

            # plot temperature by longitude - model and glider
            del targs['title']
            targs['title0'] = f'{glider.split("-")[0]} transect {gl_t0.strftime("%Y-%m-%dT%H:%M")} to ' \
                                f'{gl_t1.strftime("%Y-%m-%dT%H:%M")}'
            targs['title1'] = f'RTOFS Temperature at {date_str} UTC'
            targs['save_file'] = os.path.join(sdir_glider,
                                                f'{glider.split("-")[0]}_rtofs_glider_transect_temp-{date_save}.png')
            plot_transects(gl_lon, -gl_depth, gl_temp, lon_sub, -mdepth, mtemp['temperature'], **targs)

            # get the salinity transect
            print('Getting GOFS custom salinity transect')
            msalt, mdepth, lon_sub, lat_sub = storms.custom_transect(ds, 'salinity', targetlon, targetlat, 'rtofs')
            gl_tm, gl_lon, gl_lat, gl_depth, gl_salt = gld.grid_glider_data(glider_df, 'salinity', 0.5)

            # plot salinity by longitude
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['clab'] = 'Salinity'
            sargs['title'] = f'RTOFS Salinity at {date_str} UTC\n{glider} glider track'
            sargs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_rtofs_transect_salt-{date_save}.png')
            sargs['levels'] = color_lims['salt']
            sargs['ylims'] = ylims
            print('plotting salinity by longitude')
            plot_transect(lon_sub, -mdepth, msalt, **sargs)

            # plot salinity by longitude - model and glider
            del sargs['title']
            sargs['title0'] = f'{glider.split("-")[0]} transect {gl_t0.strftime("%Y-%m-%dT%H:%M")} to ' \
                                f'{gl_t1.strftime("%Y-%m-%dT%H:%M")}'
            sargs['title1'] = f'RTOFS Salinity at {date_str} UTC'
            sargs['save_file'] = os.path.join(sdir_glider,
                                                f'{glider.split("-")[0]}_rtofs_glider_transect_salt-{date_save}.png')
            plot_transects(gl_lon, -gl_depth, gl_salt, lon_sub, -mdepth, msalt, **sargs)


if __name__ == '__main__':
    glider_deployments = ['ru30-20210503T1929']
    sdir = '/Users/mikesmith/Documents/'
    model_t0 = dt.datetime(2021, 5, 9, 0, 0)  # False
    model_t1 = False
    glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
    glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
    line_transect = True  # True or False  # get a straight line transect, rather than a transect along the glider track
    y_limits = [-100, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(9, 16, .5)),
                    salt=dict(shallow=np.arange(31.6, 36.8, .2)))
    main(glider_deployments, sdir, model_t0, model_t1, glider_t0, glider_t1, line_transect, y_limits, c_limits)
