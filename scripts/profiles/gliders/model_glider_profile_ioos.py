#! /usr/bin/env python3

"""
Author: Lori Garzio on 6/14/2021
Last modified: Lori Garzio on 6/14/2021

"""
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import os
from matplotlib import pyplot as plt
import ioos_model_comparisons.storms as storms
import ioos_model_comparisons.gliders as gld
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def main(gliders, save_dir, profile_t0, ylims, x_lims, gwh):
    gofsurl = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
    rtofsurl = '/Users/garzio/Documents/rucool/hurricane_glider_project/RTOFS/RTOFS_6hourly_North_Atlantic/'
    # rtofsurl = '/home/hurricaneadm/data/rtofs'  # on server

    xlabels = {'temperature': 'Temperature ($^oC$)',
               'salinity': 'Salinity'}

    for pt0 in profile_t0:
        print('\nPlotting profile at {}'.format(pt0))
        pt0_str = pt0.strftime('%Y-%m-%dT%H:%M')
        pt0_str_save = pt0.strftime('%Y%m%dT%H%M')

        # initialize keyword arguments for glider functions.
        gargs = dict()
        gargs['time_start'] = pt0 - dt.timedelta(hours=gwh)
        gargs['time_end'] = pt0 + dt.timedelta(hours=gwh)
        gargs['filetype'] = 'dataframe'

        for glider in gliders:
            glider_name = glider.split('-')[0]
            sdir_glider = os.path.join(save_dir, glider, 'profiles')
            os.makedirs(sdir_glider, exist_ok=True)
            glider_ds = gld.glider_dataset(glider, **gargs)
            gl_t0 = pd.to_datetime(np.nanmin(glider_ds['time'])).strftime('%Y-%m-%dT%H:%M')
            gl_t1 = pd.to_datetime(np.nanmax(glider_ds['time'])).strftime('%Y-%m-%dT%H:%M')

            # average glider profile location
            gl_pl = [np.round(np.nanmean(glider_ds.longitude.values), 2),
                     np.round(np.nanmean(glider_ds.latitude.values), 2)]

            # separate glider profiles
            profiletimes, idx = np.unique(glider_ds.time.values, return_index=True)

            targetlon = np.unique(glider_ds.longitude.values)
            targetlat = np.unique(glider_ds.latitude.values)

            targetlon_GOFS = storms.convert_target_gofs_lon(targetlon)

            # get GOFS data
            with xr.open_dataset(gofsurl, drop_variables='tau') as gofs:
                gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})
                gofs_ds = gofs.sel(time=pt0, depth=slice(0, 1000))

                oklonGOFS = np.unique(np.round(np.interp(targetlon_GOFS, gofs_ds.lon, np.arange(0, len(gofs_ds.lon)))).astype(int))
                oklatGOFS = np.unique(np.round(np.interp(targetlat, gofs_ds.lat, np.arange(0, len(gofs_ds.lat)))).astype(int))

                # check that the glider doesn't cross more than one GOFS grid point
                if np.logical_or(len(oklonGOFS) > 1, len(oklatGOFS) > 1):
                    raise ValueError('Glider crosses >1 GOFS grid point. Choose different time range.')

                # get RTOFS data
                if pt0.hour == 0:
                    hr = '{:03d}'.format(24)
                    rtofst0 = pt0 - dt.timedelta(days=1)
                else:
                    hr = '{:03d}'.format(pt0.hour)
                    rtofst0 = pt0
                rfile = f"{rtofsurl}{rtofst0.strftime('rtofs.%Y%m%d')}/rtofs_glo_3dz_f{hr}_6hrly_hvr_US_east.nc"
                rtofs_ds = xr.open_dataset(rfile)
                rtofs_ds = rtofs_ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
                rtofs_ds = rtofs_ds.sel(depth=slice(0, 1000))

                oklonRTOFS = np.unique(np.round(np.interp(targetlon, rtofs_ds.lon[0, :], np.arange(0, len(rtofs_ds.lon[0, :])))).astype(int))
                oklatRTOFS = np.unique(np.round(np.interp(targetlat, rtofs_ds.lat[:, 0], np.arange(0, len(rtofs_ds.lat[:, 0])))).astype(int))

                # check that the glider doesn't cross more than one RTOFS grid point
                if np.logical_or(len(oklonRTOFS) > 1, len(oklatRTOFS) > 1):
                    raise ValueError('Glider crosses >1 RTOFS grid point. Choose different time range: {}'.format(pt0))

                lon_RTOFS = rtofs_ds.lon.values[0, oklonRTOFS]
                lat_RTOFS = rtofs_ds.lat.values[oklatRTOFS, 0]
                RTOFS_pl = [np.round(lon_RTOFS[0], 2), np.round(lat_RTOFS[0], 2)]

                for pv, xl in x_lims.items():
                    fig, ax = plt.subplots(figsize=(8, 9))
                    plt.subplots_adjust(right=0.88, left=0.15)
                    plt.grid()

                    # plot glider profiles
                    for pt in profiletimes:
                        # pt_idx = np.squeeze(np.argwhere(glider_ds.time.values == pt))
                        pt_gl = glider_ds[glider_ds['time'] == pt]
                        # pt_gl = pt_gl[pt_gl.pressure != 0.00]  # get rid of zeros
                        x = pt_gl[pv]
                        y = pt_gl['depth']
                        xmask = ~np.isnan(x)  # get rid of nans so the lines are continuous
                        # ax.plot(x[xmask], y[xmask], lw=3, c='k', label=glider_name)
                        ax.plot(x[xmask], y[xmask], lw=3, c='blue', label=glider_name)

                    # plot GOFS
                    GOFS_targetvar = np.squeeze(gofs_ds[pv][:, oklatGOFS, oklonGOFS])
                    GOFS_lon = storms.convert_gofs_target_lon(GOFS_targetvar.lon.values.tolist())
                    GOFS_pl = [np.round(GOFS_lon[0], 2), np.round(GOFS_targetvar.lat.values, 2)]
                    #ax.plot(GOFS_targetvar.values, GOFS_targetvar.depth.values, lw=3, c='tab:blue', label='GOFS')
                    ax.plot(GOFS_targetvar.values, GOFS_targetvar.depth.values, lw=3, c='red', label='GOFS')

                    # plot RTOFS
                    RTOFS_targetvar = np.squeeze(rtofs_ds[pv][0, :, oklatRTOFS, oklonRTOFS])
                    #ax.plot(RTOFS_targetvar.values, RTOFS_targetvar.depth.values, lw=3, c='tab:orange', label='RTOFS')
                    ax.plot(RTOFS_targetvar.values, RTOFS_targetvar.depth.values, lw=3, c='green', label='RTOFS')

                    if ylims:
                        ax.set_ylim(ylims)
                    else:  # set y limits to glider max depth
                        rounded_depth = np.ceil(np.nanmax(glider_ds['depth']) / 10) * 10
                        ax.set_ylim([0, rounded_depth])
                    if xl:
                        ax.set_xticks(xl)
                    ax.set_xlabel(xlabels[pv])
                    ax.set_ylabel('Depth (m)')

                    ax.invert_yaxis()

                    # get legend handles and only show one set
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    # fig.legend(by_label.values(), by_label.keys(), framealpha=0.5, ncol=3, bbox_to_anchor=(0.89, 0.84), fontsize=12)
                    ax.legend(by_label.values(), by_label.keys(), fontsize=12)

                    ttl = f"GOFS: {pt0_str}, {GOFS_pl}\nRTOFS: {pt0_str}, {RTOFS_pl}\n" \
                          f"Glider: {gl_t0} to {gl_t1}, {gl_pl}"

                    ax.set_title(ttl)

                    savefile = os.path.join(sdir_glider, f'{glider_name}_models_profiles_{pv}-{pt0_str_save}.png')
                    plt.savefig(savefile, dpi=300)
                    plt.close()


if __name__ == '__main__':
    glider_deployments = ['maracoos_02-20210820T1546']
    sdir = '/Users/garzio/Documents/'
    # plot_profile_time = [dt.datetime(2021, 5, 4, 12), dt.datetime(2021, 5, 7, 12), dt.datetime(2021, 5, 9, 18),
    #                      dt.datetime(2021, 5, 14, 6), dt.datetime(2021, 5, 16, 12), dt.datetime(2021, 5, 18, 0)]
    plot_profile_time = [dt.datetime(2021, 8, 26, 0)]
    y_limits = None  # [0, 55]
    # x_limits = dict(temperature=np.arange(8, 14.5, .5), salinity=np.arange(31.6, 34.2, .2))
    x_limits = dict(temperature=None, salinity=None)
    glider_window_hours = 1  # +/- x hours around profile_time for glider data
    main(glider_deployments, sdir, plot_profile_time, y_limits, x_limits, glider_window_hours)
