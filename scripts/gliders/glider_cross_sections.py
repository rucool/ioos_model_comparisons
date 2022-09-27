#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/14/2021
Create transect "ribbons" of RTOFS along user-specified glider(s) tracks. Model transect is in space and time.
"""
import pandas as pd
import os
import ioos_model_comparisons.gliders as gld
import matplotlib.pyplot as plt
import cmocean
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def main(gliders, save_dir, g_t0, g_t1, xlims, ylims):
    # initialize keyword arguments for glider functions
    gargs = dict()
    gargs['time_start'] = g_t0
    gargs['time_end'] = g_t1
    gargs['filetype'] = 'dataframe'
    t0 = g_t0.strftime('%Y-%m-%d')
    t1 = g_t1.strftime('%Y-%m-%d')

    t = np.arange(xlims['temperature'][0], xlims['temperature'][1], xlims['temperature'][2])
    t = np.append(t, xlims['temperature'][1])

    s = np.arange(xlims['salinity'][0], xlims['salinity'][1], xlims['salinity'][2])
    s = np.append(s, xlims['salinity'][1])

    d = np.arange(xlims['density'][0], xlims['density'][1], xlims['density'][2])
    d = np.append(d, xlims['density'][1])

    for glider in gliders:
        sdir_glider = os.path.join(save_dir, glider, 'transects', 'transect-ribbons')
        os.makedirs(sdir_glider, exist_ok=True)
        glider_df = gld.glider_dataset(glider, **gargs)
        # cm = plt.get_cmap('RdYlBu', 10)
        # cmap = cmocean.tools.cmap(cmocean.cm.thermal, N=10)

        cmdict = cmocean.tools.get_dict(cmocean.cm.thermal, N=t.shape[0]-1)
        thermal = LinearSegmentedColormap(cmocean.cm.thermal, segmentdata=cmdict, N=t.shape[0]-1)

        cmdict = cmocean.tools.get_dict(cmocean.cm.haline, N=s.shape[0]-1)
        haline = LinearSegmentedColormap(cmocean.cm.haline, segmentdata=cmdict, N=s.shape[0]-1)

        cmdict = cmocean.tools.get_dict(cmocean.cm.dense, N=d.shape[0]-1)
        dense = LinearSegmentedColormap(cmocean.cm.dense, segmentdata=cmdict, N=d.shape[0]-1)

        # temperature section... time on x axis.. depth on y axis... color-  temperature
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.scatter(glider_df['time'],
                    glider_df['depth'],
                    20,
                    glider_df['temperature'],
                    cmap=thermal.reversed(thermal),
                    vmin=t[0],
                    vmax=t[-1]
                    )
        plt.ylim(ylims)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        cb.set_label('Temperature (°C)', fontsize=16)

        # Add the grid
        plt.grid()
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)
        xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.title(f'Temperature Section for Glider: {glider}\nStart: {t0}, End: {t1}',
                  fontsize=20, fontweight='bold')
        plt.xlabel('Time (GMT)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
        plt.savefig(f'/Users/mikesmith/Documents/{glider}-temperature-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(16, 8))
        plt.scatter(glider_df['time'],
                    glider_df['depth'],
                    20,
                    glider_df['salinity'],
                    cmap=haline.reversed(haline),
                    vmin=s[0],
                    vmax=s[-1]
                    )
        plt.ylim(ylims)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        cb.set_label('Salinity (1)', fontsize=16)
        plt.grid()
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)
        xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.title(f'Salinity Section for Glider: {glider}\nStart: {t0}, End: {t1}', fontsize=20,
                  fontweight='bold')
        plt.xlabel('Time (GMT)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
        plt.savefig(f'/Users/mikesmith/Documents/{glider}-salinity-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        # Density profile cross section
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.scatter(glider_df['time'],
                    glider_df['depth'],
                    20,
                    glider_df['density'],
                    cmap=dense.reversed(dense),
                    vmin=d[0],
                    vmax=d[-1]
                    )
        plt.ylim(ylims)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        cb.set_label('Density (kg m-3)', fontsize=16)
        plt.grid()
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)
        xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.title(f'Density Section for Glider: {glider}\nStart: {t0}, End: {t1}', fontsize=20,
                  fontweight='bold')
        plt.xlabel('Time (GMT)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
        plt.savefig(f'/Users/mikesmith/Documents/{glider}-density-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        # Profile plotting
        grouped = glider_df.groupby([pd.Grouper(freq='1D', key='time')])
        cmap = get_cmap(len(grouped))

        fig, ax = plt.subplots(figsize=(10, 12))
        grouped = list(grouped)

        for i in range(len(grouped)):
            # temperature profile... y axis - depth, x axis. - temperature. color- time
            plt.scatter(grouped[i][1]['temperature'],
                        grouped[i][1]['depth'],
                        12,
                        cmap=cmap(i),
                        label=grouped[i][0].strftime('%Y-%m-%d'))
        plt.ylim(ylims)
        # plt.xlim([t[0], t[-1]])
        plt.xlim([11, 23])


        plt.legend(fontsize=14)

        # Add the grid
        plt.grid()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.title(f'Temperature Profile for Glider: {glider}\nStart: {t0}, End: {t1}',
                  fontsize=20, fontweight='bold')
        plt.xlabel('Temperature (°C)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')

        plt.savefig(f'/Users/mikesmith/Documents/{glider}-temperature-profile-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1,
                    dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 12))
        for i in range(len(grouped)):
            plt.scatter(grouped[i][1]['salinity'],
                        grouped[i][1]['depth'],
                        12,
                        cmap=i,
                        label=grouped[i][0].strftime('%Y-%m-%d'))
        plt.ylim(ylims)
        # plt.xlim([s[0], s[-1]])
        plt.xlim([31, 35])
        plt.legend(fontsize=14)

        # Add the grid
        plt.grid()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f'Salinity Profile for Glider: {glider}\nStart: {t0}, End: {t1}', fontsize=20,
                  fontweight='bold')
        plt.xlabel('Salinity (1)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
        # plt.show()
        plt.savefig(f'/Users/mikesmith/Documents/{glider}-salinity-profile-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1,
                    dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 12))
        for i in range(len(grouped)):
            plt.scatter(grouped[i][1]['density'],
                        grouped[i][1]['depth'],
                        12,
                        cmap=i,
                        label=grouped[i][0].strftime('%Y-%m-%d'))
        plt.ylim(ylims)
        plt.xlim([d[0], d[-1]])
        plt.legend(fontsize=14)

        # Add the grid
        plt.grid()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f'Density Profile for Glider: {glider}\nStart: {t0}, End: {t1}', fontsize=20,
                  fontweight='bold')
        plt.xlabel('Density (kg m-3)', fontsize=18, fontweight='bold')
        plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
        # plt.show()
        plt.savefig(f'/Users/mikesmith/Documents/{glider}-density-profile-{t0}-{t1}', bbox_inches='tight', pad_inches=0.1,
                    dpi=300)
        plt.close()


if __name__ == '__main__':
    # glider_deployments = ['bios_jack-20210709T1945']
    # sdir = '/Users/mikesmith/Documents/'
    # glider_t0 = dt.datetime(2021, 9, 21, 0, 0)  # False
    # glider_t1 = dt.datetime(2021, 9, 28, 9, 0)
    # y_limits = [400, 0]  # None
    # x_limits = dict(temperature=[17, 29, 1],
    #                 salinity=[36.4, 37.3, .1],
    #                 density=[1023, 1029, 1])
    # main(glider_deployments, sdir, glider_t0, glider_t1, x_limits, y_limits)
    glider_deployments = ['ru29-20210630T1343']
    sdir = '/Users/mikesmith/Documents/'
    glider_t0 = dt.datetime(2021, 6, 1, 0, 0)  # False
    glider_t1 = dt.datetime(2021, 7, 15, 0, 0)
    y_limits = [45, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(11, 22, 1)),
                    salt=dict(shallow=np.arange(32, 34, .1)))
    x_limits = dict(temperature=[11, 22, 1],
                    salinity=[32, 34, .1],
                    density=[1021, 1026, 1])
    main(glider_deployments, sdir, glider_t0, glider_t1, x_limits, y_limits)


