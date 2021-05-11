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

        gl_tm, gl_lon, gl_lat, gl_depth, gl_temp = gld.grid_glider_data(glider_df, 'temperature', 0.5)

        # plot temperature by time
        targs = {}
        targs['cmap'] = cmocean.cm.thermal
        targs['clab'] = 'Temperature ($^oC$)'
        targs['title'] = f'{glider.split("-")[0]} transect {gl_t0.strftime("%Y-%m-%dT%H:%M")} to '\
                         f'{gl_t1.strftime("%Y-%m-%dT%H:%M")}'
        targs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_transect_temp.png')
        targs['xlab'] = 'Time'
        if ylims:
            targs['ylims'] = ylims
        if color_lims:
            targs['levels'] = color_lims['temp']
        print('plotting temperature by time')
        plot_transect(gl_tm, -gl_depth, gl_temp, **targs)

        # plot temperature by longitude
        targs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_transect_temp-lon.png')
        targs['xlab'] = 'Longitude'
        print('plotting temperature by longitude')
        plot_transect(gl_lon, -gl_depth, gl_temp, **targs)

        # grid salinity data
        gl_tm, gl_lon, gl_lat, gl_depth, gl_salt = gld.grid_glider_data(glider_df, 'salinity', 0.5)

        # plot salinity by time
        sargs = {}
        sargs['cmap'] = cmocean.cm.haline
        sargs['clab'] = 'Salinity'
        sargs['title'] = f'{glider.split("-")[0]} transect {gl_t0.strftime("%Y-%m-%dT%H:%M")} to ' \
                         f'{gl_t1.strftime("%Y-%m-%dT%H:%M")}'
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_transect_salt.png')
        sargs['xlab'] = 'Time'
        if ylims:
            sargs['ylims'] = ylims
        if color_lims:
            sargs['levels'] = color_lims['salt']
        print('plotting salinity by time')
        plot_transect(gl_tm, -gl_depth, gl_salt, **sargs)

        # plot salinity by longitude
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider.split("-")[0]}_transect_salt-lon.png')
        sargs['xlab'] = 'Longitude'
        print('plotting salinity by longitude')
        plot_transect(gl_lon, -gl_depth, gl_salt, **sargs)


if __name__ == '__main__':
    glider_deployments = ['ru30-20210503T1929']
    sdir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
    glider_t0 = False  # dt.datetime(2021, 5, 4, 0, 0)
    glider_t1 = dt.datetime(2021, 5, 10, 0, 0)
    y_limits = None  # [-100, 0]
    c_limits = dict(temp=dict(shallow=np.arange(9, 16, .5)),
                    salt=dict(shallow=np.arange(31.6, 36.8, .2)))
    c_limits = None
    main(glider_deployments, sdir, glider_t0, glider_t1, y_limits, c_limits)
