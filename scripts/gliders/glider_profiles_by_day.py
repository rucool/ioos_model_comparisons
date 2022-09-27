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
import datetime as dt
import numpy as np

pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def main(gliders, save_dir, g_t0, g_t1):
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

        # Profile plotting
        # grouped = list(glider_df.groupby([pd.Grouper(freq='1D', key='time')]))
        grouped = list(glider_df.groupby([pd.Grouper(freq='24H', key='time', base=3)]))

        fig, ax = plt.subplots(
            1, len(grouped),
            figsize=(16, 9),
            constrained_layout=True
        )
        # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Blues(np.linspace(0, 1, 24))))

        for i in range(len(grouped)):
            newgroup = list(grouped[i][1].groupby('time'))
            ax[i].set_prop_cycle(plt.cycler('color', plt.cm.Reds(np.linspace(0, 1, len(newgroup)))))
            for group in newgroup:
                # temperature profile... y axis - depth, x axis. - temperature. color- time
                h = ax[i].scatter(
                    group[1]['temperature'],
                    group[1]['depth'],
                    label=group[0].strftime('%H:%M:%S'),
                    edgecolor='black'
                )

            ax[i].legend()
            ax[i].set_title(grouped[i][0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

        for axs in ax.flat:
            axs.set_ylim([60, 1])
            axs.set_xlim([21, 30.5])
            axs.grid(True, linestyle='--', linewidth=0.5)
            axs.tick_params(axis='x', labelsize=14)
            axs.tick_params(axis='y', labelsize=14)
            axs.set_xlabel('Temperature (°C)', fontsize=16, fontweight='bold')
            axs.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')


        plt.suptitle('Temperature Profile for Glider: ng645-20210613T0000\nStart: 2021-08-27, End: 2021-08-31',
                  fontsize=20, fontweight='bold')

        plt.savefig('/Users/mikesmith/Documents/ng645-daily_temperature-profile-ida', bbox_inches='tight', pad_inches=0.1,
                    dpi=150)
        plt.close()

        fig, ax = plt.subplots(
            1, len(grouped),
            figsize=(16, 9),
            constrained_layout=True
        )
        # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Blues(np.linspace(0, 1, 24))))

        for i in range(len(grouped)):
            newgroup = list(grouped[i][1].groupby('time'))
            ax[i].set_prop_cycle(plt.cycler('color', plt.cm.Greens(np.linspace(0, 1, len(newgroup)))))
            for group in newgroup:
                # temperature profile... y axis - depth, x axis. - temperature. color- time
                h = ax[i].scatter(
                    group[1]['salinity'],
                    group[1]['depth'],
                    label=group[0].strftime('%H:%M:%S'),
                    edgecolor='black',
                )

            ax[i].legend()
            ax[i].set_title(grouped[i][0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

        for axs in ax.flat:
            axs.set_ylim([60, 1])
            axs.set_xlim([34.5, 37])
            axs.grid(True, linestyle='--', linewidth=0.5)
            axs.tick_params(axis='x', labelsize=14)
            axs.tick_params(axis='y', labelsize=14)
            axs.set_xlabel('Salinity (1)', fontsize=16, fontweight='bold')
            axs.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')

        plt.suptitle('Salinity Profile for Glider: ng645-20210613T0000\nStart: 2021-08-27, End: 2021-08-31',
                     fontsize=20, fontweight='bold')

        plt.savefig('/Users/mikesmith/Documents/ng645-daily_salinity-profile-ida', bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=150)
        plt.close()

        fig, ax = plt.subplots(
            1, len(grouped),
            figsize=(16, 9),
            constrained_layout=True
        )
        # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Blues(np.linspace(0, 1, 24))))
        for i in range(len(grouped)):
            newgroup = list(grouped[i][1].groupby('time'))

            # cmap = mpl.cm.Blues(np.linspace(0, 1, len(newgroup)))
            # cmap = mpl.colors.ListedColormap(cmap[0:10])
            cmap = plt.cm.Blues(np.linspace(0, 1, len(newgroup)))

            ax[i].set_prop_cycle(plt.cycler('color', cmap))
            for group in newgroup:
                # temperature profile... y axis - depth, x axis. - temperature. color- time
                h = ax[i].scatter(
                    group[1]['density'],
                    group[1]['depth'],
                    label=group[0].strftime('%H:%M:%S'),
                    edgecolor='black'
                )

            ax[i].legend()
            ax[i].set_title(grouped[i][0].strftime('%Y-%m-%d'), fontsize=18, fontweight='bold')

        for axs in ax.flat:
            axs.set_ylim([60, 1])
            axs.set_xlim([1021, 1026])
            axs.grid(True, linestyle='--', linewidth=0.5)
            axs.tick_params(axis='x', labelsize=14)
            axs.tick_params(axis='y', labelsize=14)
            axs.set_xlabel('density (kg m-3)', fontsize=16, fontweight='bold')
            axs.set_ylabel('Depth (m)', fontsize=16, fontweight='bold')

        plt.suptitle('Density Profile for Glider: ng645-20210613T0000\nStart: 2021-08-27, End: 2021-08-31',
                     fontsize=20, fontweight='bold')

        plt.savefig('/Users/mikesmith/Documents/ng645-daily_density-profile-ida', bbox_inches='tight',
                    pad_inches=0.1,
                    dpi=150)
        plt.close()



if __name__ == '__main__':
    glider_deployments = ['ng645-20210613T0000']
    sdir = '/Users/mikesmith/Documents/'
    glider_t0 = dt.datetime(2021, 8, 28, 3, 0)  # False
    glider_t1 = dt.datetime(2021, 8, 31, 3, 0)
    y_limits = [-20, 0]  # None
    c_limits = dict(temp=dict(shallow=np.arange(25, 30, .25)),
                    salt=dict(shallow=np.arange(34, 37, .25)))
    main(glider_deployments, sdir, glider_t0, glider_t1)
