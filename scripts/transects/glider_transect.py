#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/7/2021
Last modified: Lori Garzio on 5/11/2021
Create a transects of user-specified glider(s) temperature and salinity.
"""
import cmocean
import numpy as np
import pandas as pd
import os
import datetime as dt
import src.gliders as gld
from src.gliders_plt import plot_transect
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, g_t0, g_t1, ylims, color_lims):
    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = g_t0
    gargs['time_end'] = g_t1
    gargs['filetype'] = 'dataframe'
    #gargs['filetype'] = 'nc'

    for glider in gliders:
        sdir_glider = os.path.join(save_dir, glider, 'transects')
        os.makedirs(sdir_glider, exist_ok=True)
        glider_df = gld.glider_dataset(glider, **gargs)
        gl_t0 = pd.to_datetime(np.nanmin(glider_df['time']))
        gl_t1 = pd.to_datetime(np.nanmax(glider_df['time']))
        gl_t0str = gl_t0.strftime('%Y-%m-%dT%H:%M')
        gl_t1str = gl_t1.strftime('%Y-%m-%dT%H:%M')
        gl_t0save = gl_t0.strftime('%Y%m%dT%H%M')
        glider_name = glider.split('-')[0]

        gl_tm, gl_lon, gl_lat, gl_depth, gl_temp = gld.grid_glider_data(glider_df, 'temperature', 0.5)

        # plot temperature by time
        targs = {}
        targs['cmap'] = cmocean.cm.thermal
        targs['clab'] = 'Temperature ($^oC$)'
        targs['title'] = f'{glider_name} transect {gl_t0str} to {gl_t0str}'
        targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_transect_temp-{gl_t0save}.png')
        targs['xlab'] = 'Time'
        if ylims:
            targs['ylims'] = ylims
        if color_lims:
            targs['levels'] = color_lims['temp']
        print('plotting temperature by time')
        plot_transect(gl_tm, -gl_depth, gl_temp, **targs)

        # plot temperature by longitude
        targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_transect_temp-lon-{gl_t0save}.png')
        targs['xlab'] = 'Longitude'
        print('plotting temperature by longitude')
        plot_transect(gl_lon, -gl_depth, gl_temp, **targs)

        # grid salinity data
        gl_tm, gl_lon, gl_lat, gl_depth, gl_salt = gld.grid_glider_data(glider_df, 'salinity', 0.5)

        # plot salinity by time
        sargs = {}
        sargs['cmap'] = cmocean.cm.haline
        sargs['clab'] = 'Salinity'
        sargs['title'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_transect_salt-{gl_t0save}.png')
        sargs['xlab'] = 'Time'
        if ylims:
            sargs['ylims'] = ylims
        if color_lims:
            sargs['levels'] = color_lims['salt']
        print('plotting salinity by time')
        plot_transect(gl_tm, -gl_depth, gl_salt, **sargs)

        # plot salinity by longitude
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_transect_salt-lon-{gl_t0save}.png')
        sargs['xlab'] = 'Longitude'
        print('plotting salinity by longitude')
        plot_transect(gl_lon, -gl_depth, gl_salt, **sargs)


if __name__ == '__main__':
    glider_deployments = ['ru30-20210503T1929']
    sdir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
    glider_t0 = dt.datetime(2021, 5, 4, 3, 0)  # False
    glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
    y_limits = [-100, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(9, 15, .5)),
                    salt=dict(shallow=np.arange(31.6, 36.4, .2)))
    # c_limits = None
    main(glider_deployments, sdir, glider_t0, glider_t1, y_limits, c_limits)
