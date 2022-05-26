#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/14/2021
Create transect "ribbons" of GOFS along user-specified glider(s) tracks. Model transect is in space and time.
"""
import cmocean
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os
import hurricanes.storms as storms
import hurricanes.gliders as gld
from hurricanes.gliders_plt import plot_transect, plot_transects
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, g_t0, g_t1, ylims, color_lims):
    url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

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

        with xr.open_dataset(url, drop_variables='tau') as gofs:
            gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

            # Subset time range (add a little extra to the glider time range)
            mt0 = gl_t0 - dt.timedelta(hours=1)
            mt1 = gl_t1 + dt.timedelta(hours=1)

            ds = gofs.sel(time=slice(mt0, mt1), depth=slice(0, 1000))
            model_t0str = pd.to_datetime(np.nanmin(ds.time.values)).strftime('%Y-%m-%dT%H:%M')
            model_t1str = pd.to_datetime(np.nanmax(ds.time.values)).strftime('%Y-%m-%dT%H:%M')

            # interpolate glider lat/lon to lat/lon on model time
            sublonm = np.interp(pd.to_datetime(ds.time.values), pd.to_datetime(glider_df.time), glider_df.longitude)
            sublatm = np.interp(pd.to_datetime(ds.time.values), pd.to_datetime(glider_df.time), glider_df.latitude)
            sublonm_GOFS = storms.convert_target_gofs_lon(sublonm)

            # get temperature and salinity data along the glider track (space and time)
            mtemp = np.full([len(ds.depth), len(ds.time)], np.nan)
            msalt = np.full([len(ds.depth), len(ds.time)], np.nan)
            mu = np.full([len(ds.depth), len(ds.time)], np.nan)
            mv = np.full([len(ds.depth), len(ds.time)], np.nan)
            for i, t in enumerate(ds.time):
                tds = ds.sel(time=t, lon=sublonm_GOFS[i], lat=sublatm[i], method='nearest')
                mtemp[:, i] = tds['temperature'].values
                msalt[:, i] = tds['salinity'].values
                mu[:, i] = tds['u'].values
                mv[:, i] = tds['v'].values

            # get the temperature transect from the glider
            gl_tm, gl_lon, gl_lat, gl_depth, gl_temp = gld.grid_glider_data(glider_df, 'temperature', 0.5)
            gl_tm, gl_lon, gl_lat, gl_depth, gl_salt = gld.grid_glider_data(glider_df, 'salinity', 0.5)
            gl_tm, gl_lon, gl_lat, gl_depth, gl_u = gld.grid_glider_data(glider_df, 'u', 0.5)
            gl_tm, gl_lon, gl_lat, gl_depth, gl_v = gld.grid_glider_data(glider_df, 'v', 0.5)


            # plot temperature by time (glider time/location) - model only
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['clab'] = 'Temperature ($^oC$)'
            targs['title'] = f'GOFS Temperature along {glider} track\nModel: {model_t0str} to {model_t1str}  ' \
                             f'Glider: {gl_t0str} to {gl_t1str}'
            targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_transect_temp-{gl_t0save}.png')
            targs['levels'] = color_lims['temp']
            targs['ylims'] = ylims
            targs['xlab'] = 'Time'
            plot_transect(ds.time.values, ds.depth.values, mtemp, **targs)

            # plot temperature by time (glider time/location) - model and glider
            del targs['title']
            targs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
            targs['title1'] = f'GOFS Temperature: {model_t0str} to {model_t1str}'
            targs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_glider_transect_temp-{gl_t0save}.png')
            plot_transects(gl_tm, gl_depth, gl_temp, ds.time.values, ds.depth.values, mtemp, **targs)

            # plot salinity by time - model only
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['clab'] = 'Salinity'
            sargs['title'] = f'GOFS Salinity along {glider} track\nModel: {model_t0str} to {model_t1str}  ' \
                             f'Glider: {gl_t0str} to {gl_t1str}'
            sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_transect_salt-{gl_t0save}.png')
            sargs['levels'] = color_lims['salt']
            sargs['ylims'] = ylims
            sargs['xlab'] = 'Time'
            plot_transect(ds.time.values, ds.depth.values, msalt, **sargs)

            # plot salinity by time (glider time/location) - model and glider
            del sargs['title']
            sargs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
            sargs['title1'] = f'GOFS Salinity: {model_t0str} to {model_t1str}'
            sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_glider_transect_salt-{gl_t0save}.png')
            plot_transects(gl_tm, gl_depth, gl_salt, ds.time.values, ds.depth.values, msalt, **sargs)


            # plot u by time - model only
            sargs = {}
            sargs['cmap'] = cmocean.cm.balance
            sargs['clab'] = 'U'
            sargs['title'] = f'GOFS U Velocity along {glider} track\nModel: {model_t0str} to {model_t1str}  ' \
                             f'Glider: {gl_t0str} to {gl_t1str}'
            sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_transect_u-{gl_t0save}.png')
            sargs['levels'] = color_lims['u']
            sargs['ylims'] = ylims
            sargs['xlab'] = 'Time'
            plot_transect(ds.time.values, ds.depth.values, mu, **sargs)

            # # plot u by time (glider time/location) - model and glider
            # del sargs['title']
            # sargs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
            # sargs['title1'] = f'GOFS U Velocity: {model_t0str} to {model_t1str}'
            # sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_glider_transect_u-{gl_t0save}.png')
            # plot_transects(gl_tm, gl_depth, gl_u, ds.time.values, ds.depth.values, mu, **sargs)

            # plot u by time - model only
            sargs = {}
            sargs['cmap'] = cmocean.cm.balance
            sargs['clab'] = 'U'
            sargs['title'] = f'GOFS V Velocity along {glider} track\nModel: {model_t0str} to {model_t1str}  ' \
                             f'Glider: {gl_t0str} to {gl_t1str}'
            sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_transect_v-{gl_t0save}.png')
            sargs['levels'] = color_lims['v']
            sargs['ylims'] = ylims
            sargs['xlab'] = 'Time'
            plot_transect(ds.time.values, ds.depth.values, mv, **sargs)

            # # plot u by time (glider time/location) - model and glider
            # del sargs['title']
            # sargs['title0'] = f'{glider_name} transect {gl_t0str} to {gl_t1str}'
            # sargs['title1'] = f'GOFS V Velocity: {model_t0str} to {model_t1str}'
            # sargs['save_file'] = os.path.join(sdir_glider, f'{glider_name}_gofs_glider_transect_v-{gl_t0save}.png')
            # plot_transects(gl_tm, gl_depth, gl_v, ds.time.values, ds.depth.values, mv, **sargs)

if __name__ == '__main__':
    sdir = '/Users/mikesmith/Documents/'
    glider_deployments = ['ru29-20200908T1623']
    # glider_t1 = dt.datetime.utcnow()
    # glider_t0 = glider_t1 - dt.timedelta(days=5)  # False
    y_limits = [100, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(26, 31, .5)),
                    salt=dict(shallow=np.arange(34.75, 37, .25)),
                    u=dict(shallow=np.arange(-3, .4, 0.05)),
                    v=dict(shallow=np.arange(-3, .4, 0.05))
                    )
    # glider_deployments = ['maracoos_02-20210716T1814']
    glider_t0 = dt.datetime(2020, 10, 15, 9, 7)  # False
    glider_t1 = dt.datetime(2020, 10, 16, 22, 55)
    # y_limits = [45, 0]  # None
    # c_limits = dict(temp=dict(shallow=np.arange(4, 28, 1)),
                    # salt=dict(shallow=np.arange(30, 34.5, .1)))
    main(glider_deployments, sdir, glider_t0, glider_t1, y_limits, c_limits)