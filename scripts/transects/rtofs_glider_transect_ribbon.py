#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/14/2021
Create transect "ribbons" of RTOFS along user-specified glider(s) tracks. Model transect is in space and time.
"""
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os
import glob
import hurricanes.gliders as gld
from hurricanes.plotting import plot_transect, plot_transects
# from src.plotting import plot_transect, plot_transects
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, g_t0, g_t1, ylims, color_lims):
    # url = '/Users/garzio/Documents/rucool/hurricane_glider_project/RTOFS/RTOFS_6hourly_North_Atlantic/'
    # url = '/home/hurricaneadm/data/rtofs'  # on server
    url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'

    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = g_t0
    gargs['time_end'] = g_t1
    gargs['filetype'] = 'dataframe'

    for glider in gliders:
        sdir_glider = os.path.join(save_dir, glider, 'transects', 'transect-ribbons')
        os.makedirs(sdir_glider, exist_ok=True)
        glider_df = gld.glider_dataset(glider, **gargs)
        gl_t0 = pd.to_datetime(np.nanmin(glider_df['time']))
        gl_t1 = pd.to_datetime(np.nanmax(glider_df['time']))
        gl_t0str = gl_t0.strftime('%Y-%m-%dT%H:%M')
        gl_t1str = gl_t1.strftime('%Y-%m-%dT%H:%M')
        gl_t0save = gl_t0.strftime('%Y%m%dT%H%M')
        glider_name = glider.split('-')[0]

        # Subset time range (add a little extra to the glider time range)
        mt0 = gl_t0 - dt.timedelta(hours=2)
        mt1 = gl_t1 + dt.timedelta(hours=12)

        model_dates = [x.strftime('rtofs.%Y%m%d') for x in pd.date_range(mt0, mt1)]
        rtofs_files = [glob.glob(os.path.join(url, x, '*.nc')) for x in model_dates]
        rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])
        # make sure the file times are within the specified times
        model_time = np.array([], dtype='datetime64[ns]')
        rfiles = []
        for rf in rtofs_files:
            splitter = rf.split('/')
            ymd = splitter[-2].split('.')[-1]
            hr = splitter[-1].split('_')[3][-2:]
            if hr == '24':
                hr = '00'
                td = 1
            else:
                td = 0
            mt = dt.datetime.strptime('T'.join((ymd, hr)), '%Y%m%dT%H') + dt.timedelta(days=td)
            if np.logical_and(mt >= mt0, mt <= gl_t1 + dt.timedelta(hours=1)):
                rfiles.append(rf)
                model_time = np.append(model_time, pd.to_datetime(mt))

        model_t0str = pd.to_datetime(np.nanmin(model_time)).strftime('%Y-%m-%dT%H:%M')
        model_t1str = pd.to_datetime(np.nanmax(model_time)).strftime('%Y-%m-%dT%H:%M')

        # open the first file to get some of the model information
        ds = xr.open_dataset(rfiles[0])
        ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
        ds = ds.sel(depth=slice(0, 1000))
        mlon = ds.lon.values
        mlat = ds.lat.values

        # interpolate glider lat/lon to lat/lon on model time
        sublonm = np.interp(pd.to_datetime(model_time), pd.to_datetime(glider_df.time), glider_df.longitude)
        sublatm = np.interp(pd.to_datetime(model_time), pd.to_datetime(glider_df.time), glider_df.latitude)

        # interpolating transect X and Y to lat and lon
        lonIndex = np.round(np.interp(sublonm, mlon[0, :], np.arange(0, len(mlon[0, :])))).astype(int)
        latIndex = np.round(np.interp(sublatm, mlat[:, 0], np.arange(0, len(mlat[:, 0])))).astype(int)

        # get temperature and salinity data along the glider track (space and time)
        mtemp = np.full([len(ds.depth), len(model_time)], np.nan)
        msalt = np.full([len(ds.depth), len(model_time)], np.nan)
        for i, f in enumerate(rfiles):
            print(f)
            with xr.open_dataset(f) as ds:
                ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
                ds = ds.sel(depth=slice(0, 1000))
                mtemp[:, i] = ds['temperature'][0, :, latIndex[i], lonIndex[i]].values
                msalt[:, i] = ds['salinity'][0, :, latIndex[i], lonIndex[i]].values

        # get the temperature transect from the glider
        gl_tm, gl_lon, gl_lat, gl_depth, gl_temp = gld.grid_glider_data(glider_df, 'temperature', 0.5)

        # plot temperature by time (glider time/location) - model only
        targs = {}
        targs['cmap'] = cmocean.cm.thermal
        targs['clab'] = 'Temperature ($^oC$)'
        targs['title'] = f'RTOFS Temperature along {glider} track\n'\
                         f'Model: {model_t0str} to {model_t1str}\n'\
                         f'Glider: {gl_t0str} to {gl_t1str}'
        targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_rtofs_transect_temp-{gl_t0save}.png')
        targs['levels'] = color_lims['temp']
        targs['ylims'] = ylims
        targs['xlab'] = 'Time'
        plot_transect(model_time, ds.depth.values, mtemp, **targs)

        # plot temperature by time (glider time/location) - model and glider
        del targs['title']
        targs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
        targs['title1'] = f'RTOFS Temperature: {model_t0str} to {model_t1str}'
        targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_rtofs_glider_transect_temp-{gl_t0save}.png')
        plot_transects(gl_tm, gl_depth, gl_temp, model_time, ds.depth.values, mtemp, **targs)

        # get the salinity transect
        gl_tm, gl_lon, gl_lat, gl_depth, gl_salt = gld.grid_glider_data(glider_df, 'salinity', 0.5)

        # plot salinity by time - model only
        sargs = {}
        sargs['cmap'] = cmocean.cm.haline
        sargs['clab'] = 'Salinity'
        sargs['title'] = f'RTOFS Salinity along {glider} track\n' \
                         f'Model: {model_t0str} to {model_t1str}\n' \
                         f'Glider: {gl_t0str} to {gl_t1str}'
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_rtofs_transect_salt-{gl_t0save}.png')
        sargs['levels'] = color_lims['salt']
        sargs['ylims'] = ylims
        sargs['xlab'] = 'Time'
        plot_transect(model_time, ds.depth.values, msalt, **sargs)

        # plot salinity by time (glider time/location) - model and glider
        del sargs['title']
        sargs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
        sargs['title1'] = f'RTOFS Salinity: {model_t0str} to {model_t1str}'
        sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_rtofs_glider_transect_salt-{gl_t0save}.png')
        plot_transects(gl_tm, gl_depth, gl_salt, model_time, ds.depth.values, msalt, **sargs)


if __name__ == '__main__':
    sdir = '/Users/mikesmith/Documents/'
    # glider_deployments = ['ng645-20210613T0000']
    # glider_t0 = dt.datetime(2021, 8, 28, 0, 0)  # False
    # glider_t1 = dt.datetime(2021, 8, 31, 0, 0)
    # y_limits = [100, 0]  # None
    # c_limits = dict(temp=dict(shallow=np.arange(20, 30, 1)),
    #                 salt=dict(shallow=np.arange(34, 37, .25)))
    # glider_deployments = ['ru29-20210908T1943']
    glider_deployments = ['ru29-20210630T1343']
    sdir = '/Users/mikesmith/Documents/'
    glider_t0 = dt.datetime(2021, 6, 1, 0, 0)  # False
    glider_t1 = dt.datetime(2021, 7, 10)
    # glider_t1 = dt.datetime(2021, 9, 25, 12, 0)
    y_limits = [1000, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(4, 30, 1)),
                    salt=dict(shallow=np.arange(34.8, 37.4, .2)))
    # c_limits = None
    main(glider_deployments, sdir, glider_t0, glider_t1, y_limits, c_limits)