#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:13:12 2019

@author: aristizabal
"""
import os
import datetime as dt
import cmocean
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from erddapy import ERDDAP

import hurricanes.configs as configs
from hurricanes.models import amseas
from hurricanes.platforms import get_bathymetry, get_active_gliders
import time
import matplotlib
from pathlib import Path

# Do not produce figures on screen
plt.switch_backend('agg')

extent = configs.extent_gliders

startTime = time.time()
matplotlib.use('agg')

# Set path to save plots
path_save = (configs.path_plots / "profiles" / "gliders")

# # Increase fontsize of labels globally
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

# Get boundaries for todays date
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1) 

# Create folder name and make sure it exists
folder = path_save / today.strftime("%Y/%m-%d") / 'amseas'
os.makedirs(folder, exist_ok=True)


# Get glider data from erddap
gliders = get_active_gliders(extent, today, tomorrow, 
                             variables=configs.variables_gliders)

# Read AMSEAS output
ds = amseas(rename=True).sel(time=slice(today, tomorrow))

# Reading bathymetry data
bathy = get_bathymetry(extent)

# Looping through all gliders
for glider, df in gliders.groupby(level=0):
    print(f"Glider: {glider}")

    # Coverting glider vectors into arrays
    latg = df['lat']
    long = df['lon']
    dg = df['depth']
    tg = df['temperature']
    sg = df['salinity']

    delta_z = 0.3
    zn = int(np.round(np.max(dg) / delta_z))

    # Loop through each profile for the glider
    for profile_time, profile in df.groupby(level=1):
        print()
    

    # interpolating glider lon and lat to lat and lon on AMSEAS time
    sublon_amseas = np.interp(tstamp_amseas[oktime_amseas], tstamp_glider, target_lon_amseas)
    sublat_amseas = np.interp(tstamp_amseas[oktime_amseas], tstamp_glider, target_lat_amseas)

    # getting the model grid positions for sublonm and sublatm
    oklon_amseas = np.round(np.interp(sublon_amseas, lonamseas, np.arange(len(lonamseas)))).astype(int)
    oklat_amseas = np.round(np.interp(sublat_amseas, latamseas, np.arange(len(latamseas)))).astype(int)

    # Getting glider transect from AMSEAS
    print('Getting glider transect from AMSEAS')
    if len(oktime_amseas) == 0:
        target_temp_amseas = np.empty((len(depthamseas), 1))
        target_temp_amseas[:] = np.nan
        target_salt_amseas = np.empty((len(depthamseas), 1))
        target_salt_amseas[:] = np.nan
    else:
        target_temp_amseas = np.empty((len(depthamseas), len(oktime_amseas)))
        target_temp_amseas[:] = np.nan
        target_salt_amseas = np.empty((len(depthamseas), len(oktime_amseas)))
        target_salt_amseas[:] = np.nan

        target_temp_amseas = np.empty((len(depthamseas), len(oktime_amseas)))
        target_temp_amseas[:] = np.nan
        target_salt_amseas = np.empty((len(depthamseas), len(oktime_amseas)))
        target_salt_amseas[:] = np.nan
        for i, t in enumerate(oktime_amseas):
            target_temp_amseas[:, i] = amseas.variables['water_temp'][oktime_amseas[i], :, oklat_amseas[i],
                                        oklon_amseas[i]]
            target_salt_amseas[:, i] = amseas.variables['salinity'][oktime_amseas[i], :, oklat_amseas[i],
                                        oklon_amseas[i]]

    # Convertion from GOFS to glider convention
    lonamseasg = np.empty((len(lonamseas),))
    lonamseasg[:] = np.nan
    for i, ii in enumerate(lonamseas):
        if ii >= 180.0:
            lonamseasg[i] = ii - 360
        else:
            lonamseasg[i] = ii
    latamseasg = latamseas

    meshlon_amseasg = np.meshgrid(lonamseasg, latamseasg)[0]
    meshlat_amseasg = np.meshgrid(latamseasg, lonamseasg)[0]

    # min and max values for plotting
    min_temp = np.floor(np.nanmin([np.nanmin(df[df.columns[3]]), np.nanmin(target_temp_amseas)]))
    max_temp = np.ceil(np.nanmax([np.nanmax(df[df.columns[3]]), np.nanmax(target_temp_amseas)]))

    min_salt = np.floor(np.nanmin([np.nanmin(df[df.columns[4]]), np.nanmin(target_salt_amseas)]))
    max_salt = np.ceil(np.nanmax([np.nanmax(df[df.columns[4]]), np.nanmax(target_salt_amseas)]))

    # Temperature profile
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

    ax = plt.subplot(grid[:, 0])
    plt.plot(tempg, -depthg, '.', color='cyan', label='_nolegend_')
    plt.plot(np.nanmean(tempg_gridded, axis=1), -depthg_gridded, '.-b', \
                label=id[:-14] + ' ' + str(timeg[0])[0:4] + ' ' + '[' + str(timeg[0])[5:19] + ',' + str(timeg[-1])[
                                                                                                    5:19] + ']')

    plt.plot(target_temp_amseas, -depthamseas, '.-', color='lightgreen', label='_nolegend_')
    if len(oktime_amseas) != 0:
        plt.plot(np.nanmean(target_temp_amseas, axis=1), -depthamseas, '.-', color='darkolivegreen', markersize=12,
                    linewidth=2, \
                    label='AMSEAS' + ' ' + str(time_amseas[0].year) + ' ' + '[' + str(time_amseas[0])[
                                                                                5:13] + ',' + str(time_amseas[-1])[
                                                                                                5:13] + ']')
    plt.ylabel('Depth (m)', fontsize=20)
    plt.xlabel('Temperature ($^oC$)', fontsize=20)
    plt.title('Temperature Profile ' + id, fontsize=20)
    if np.nanmax(depthg) <= 100:
        plt.ylim([-np.nanmax(depthg) - 30, 0.1])
    else:
        plt.ylim([-np.nanmax(depthg) - 100, 0.1])
    plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0.0), fontsize=14)
    plt.grid('on')

    # lat and lon bounds of box to draw
    if len(oklat_amseas) != 0:
        minlonamseas = np.min(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                                oklon_amseas.min() - 2:oklon_amseas.max() + 2])
        maxlonamseas = np.max(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                                oklon_amseas.min() - 2:oklon_amseas.max() + 2])
        minlatamseas = np.min(meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                                oklat_amseas.min() - 2:oklat_amseas.max() + 2])
        maxlatamseas = np.max(meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                                oklat_amseas.min() - 2:oklat_amseas.max() + 2])

    # Getting subdomain for plotting glider track on bathymetry
    oklatbath = np.logical_and(bath_lat >= np.min(latg) - 5, bath_lat <= np.max(latg) + 5)
    oklonbath = np.logical_and(bath_lon >= np.min(long) - 5, bath_lon <= np.max(long) + 5)

    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath, :]
    bath_elevsub = bath_elevs[:, oklonbath]

    ax = plt.subplot(grid[0:2, 1])
    lev = np.arange(-9000, 9100, 100)
    plt.contourf(bath_lonsub, bath_latsub, bath_elevsub, lev, cmap=cmocean.cm.topo)
    # plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    # plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    # plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    plt.axis([np.min(long) - 5, np.max(long) + 5, np.min(latg) - 5, np.max(latg) + 5])
    plt.plot(long, latg, '.-', color='orange', label='Glider track',
                markersize=3)
    if len(oklat_amseas) != 0:
        rect = patches.Rectangle((minlonamseas, minlatamseas),
                                    maxlonamseas - minlonamseas, maxlatamseas - minlatamseas,
                                    linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.title('Glider track and model grid positions', fontsize=20)
    plt.axis('scaled')
    plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1))

    ax = plt.subplot(grid[2, 1])
    ax.plot(long, latg, '.-', color='orange', label='Glider track')
    ax.plot(long[0], latg[0], 'sc', label='Initial profile time ' + str(timeg[0])[0:16])
    ax.plot(long[-1], latg[-1], 'sb', label='final profile time ' + str(timeg[-1])[0:16])
    if len(oklat_amseas) != 0:
        ax.plot(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                oklon_amseas.min() - 2:oklon_amseas.max() + 2],
                meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                oklat_amseas.min() - 2:oklat_amseas.max() + 2].T,
                '.', color='darkolivegreen')
        ax.plot(lonamseasg[oklon_amseas], latamseasg[oklat_amseas], 'o',
                color='darkolivegreen', markersize=8, markeredgecolor='k',
                label='AMSEAS grid points \n nx = ' + str(oklon_amseas)
                        + '\n ny = ' + str(oklat_amseas)
                        + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in time_amseas]))
                        + str([str(i)[11:13] for i in time_amseas]))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

    file = os.path.join(folder, f'{id}_temp_profile_{tini.strftime("%Y%m%d")}_to_{tend.strftime("%Y%m%d")}')
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1)

    # Salinity profile
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

    ax = plt.subplot(grid[:, 0])
    plt.plot(saltg, -depthg, '.', color='cyan')
    plt.plot(np.nanmean(saltg_gridded, axis=1), -depthg_gridded, '.-b', \
                label=id[:-14] + ' ' + str(timeg[0])[0:4] + ' ' + '[' + str(timeg[0])[5:19] + ',' + str(timeg[-1])[
                                                                                                    5:19] + ']')

    plt.plot(target_salt_amseas, -depthamseas, '.-', color='lightgreen', label='_nolegend_')
    if len(oktime_amseas) != 0:
        plt.plot(np.nanmean(target_salt_amseas, axis=1), -depthamseas, '.-', color='darkolivegreen', markersize=12,
                    linewidth=2, \
                    label='AMSEAS' + ' ' + str(time_amseas[0].year) + ' ' + '[' + str(time_amseas[0])[
                                                                                5:13] + ',' + str(time_amseas[-1])[
                                                                                                5:13] + ']')
    plt.ylabel('Depth (m)', fontsize=20)
    plt.xlabel('Salinity', fontsize=20)
    plt.title('Salinity Profile ' + id, fontsize=20)
    if np.nanmax(depthg) <= 100:
        plt.ylim([-np.nanmax(depthg) - 30, 0.1])
    else:
        plt.ylim([-np.nanmax(depthg) - 100, 0.1])
    plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0.0), fontsize=14)
    plt.grid('on')

    # lat and lon bounds of box to draw
    if len(oklat_amseas) != 0:
        minlonamseas = np.min(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                                oklon_amseas.min() - 2:oklon_amseas.max() + 2])
        maxlonamseas = np.max(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                                oklon_amseas.min() - 2:oklon_amseas.max() + 2])
        minlatamseas = np.min(meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                                oklat_amseas.min() - 2:oklat_amseas.max() + 2])
        maxlatamseas = np.max(meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                                oklat_amseas.min() - 2:oklat_amseas.max() + 2])

    # Getting subdomain for plotting glider track on bathymetry
    oklatbath = np.logical_and(bath_lat >= np.min(latg) - 5, bath_lat <= np.max(latg) + 5)
    oklonbath = np.logical_and(bath_lon >= np.min(long) - 5, bath_lon <= np.max(long) + 5)

    ax = plt.subplot(grid[0:2, 1])
    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath, :]
    bath_elevsub = bath_elevs[:, oklonbath]

    lev = np.arange(-9000, 9100, 100)
    plt.contourf(bath_lonsub, bath_latsub, bath_elevsub, lev, cmap=cmocean.cm.topo)
    # plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    # plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    # plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    plt.axis([np.min(long) - 5, np.max(long) + 5, np.min(latg) - 5, np.max(latg) + 5])
    plt.plot(long, latg, '.-', color='orange', label='Glider track',
                markersize=3)
    if len(oklat_amseas) != 0:
        rect = patches.Rectangle((minlonamseas, minlatamseas),
                                    maxlonamseas - minlonamseas, maxlatamseas - minlatamseas,
                                    linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.title('Glider track and model grid positions', fontsize=20)
    plt.axis('scaled')
    plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1))

    ax = plt.subplot(grid[2, 1])
    ax.plot(long, latg, '.-', color='orange', label='Glider track')
    ax.plot(long[0], latg[0], 'sc', label='Initial profile time ' + str(timeg[0])[0:16])
    ax.plot(long[-1], latg[-1], 'sb', label='final profile time ' + str(timeg[-1])[0:16])
    if len(oklat_amseas) != 0:
        ax.plot(meshlon_amseasg[oklat_amseas.min() - 2:oklat_amseas.max() + 2,
                oklon_amseas.min() - 2:oklon_amseas.max() + 2],
                meshlat_amseasg[oklon_amseas.min() - 2:oklon_amseas.max() + 2,
                oklat_amseas.min() - 2:oklat_amseas.max() + 2].T,
                '.', color='darkolivegreen')
        ax.plot(lonamseasg[oklon_amseas], latamseasg[oklat_amseas], 'o',
                color='darkolivegreen', markersize=8, markeredgecolor='k',
                label='AMSEAS grid points \n nx = ' + str(oklon_amseas)
                        + '\n ny = ' + str(oklat_amseas)
                        + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in time_amseas]))
                        + str([str(i)[11:13] for i in time_amseas]))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

    file = os.path.join(folder, f'{id}_salt_profile_{tini.strftime("%Y%m%d")}_to_{tend.strftime("%Y%m%d")}')
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1)
