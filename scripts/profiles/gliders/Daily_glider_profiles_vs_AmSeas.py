#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:13:12 2019

@author: aristizabal
"""
import os
from datetime import datetime, timedelta
import cmocean
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from erddapy import ERDDAP

# lat and lon bounds
lon_lim = [-98.0, -55.0]
lat_lim = [5.0, 32.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'
url_amseas = 'https://www.ncei.noaa.gov/thredds-coastal/dodsC/ncom_amseas_agg/AmSeas_Dec_17_2020_to_Current_best.ncd'

# Realtime Server Inputs
save_dir = '/www/web/rucool/'
bath_file = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# # Local Testing Inputs
# save_dir = '/Users/mikesmith/Documents/'
# bath_file = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Do not produce figures on screen
plt.switch_backend('agg')

# Increase fontsize of labels globally
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

# %% Get time bounds for current day
ti = datetime.today() - timedelta(hours=6)
tini = datetime(ti.year, ti.month, ti.day)
te = ti + timedelta(1)
tend = datetime(te.year, te.month, te.day)

folder = os.path.join(f"{save_dir}/hurricane/Hurricane_season_{ti.strftime('%Y')}", ti.strftime('%m-%d'), 'AmSeas')
os.makedirs(folder, exist_ok=True)

# %% Look for datasets in IOOS glider dac
print('Looking for glider data sets')
e = ERDDAP(server=url_glider)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': tini.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'max_time': tend.strftime('%Y-%m-%dT%H:%M:%SZ'),
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
    'time>=': str(tini),
    'time<=': str(tend),
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[1],
}

variables = [
    'depth',
    'latitude',
    'longitude',
    'time',
    'temperature',
    'salinity'
]

e = ERDDAP(
    server=url_glider,
    protocol='tabledap',
    response='nc'
)

# %% Read AMSEAS output
print('Retrieving coordinates from AMSEAS')
amseas = xr.open_dataset(url_amseas, decode_times=False)

latamseas = np.asarray(amseas.lat[:])
lonamseas = np.asarray(amseas.lon[:])
depthamseas = np.asarray(amseas.depth[:])

tt_amseas = amseas.time
tamseas = netCDF4.num2date(tt_amseas[:], tt_amseas.units)

# %% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

# %% Looping through all gliders found
for id in gliders:
    print('Reading ' + id)
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    # checking data frame is not empty
    df = e.to_pandas()
    if np.logical_and(len(df.dropna()) != 0, df.columns[0] != 'Error {'):

        # Converting glider data to data frame
        df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
        ).dropna()

        # Coverting glider vectors into arrays
        timeg, ind = np.unique(df.index.values, return_index=True)
        latg = df['latitude (degrees_north)'].values[ind]
        long = df['longitude (degrees_east)'].values[ind]

        dg = df['depth (m)'].values
        # vg = df['temperature (degree_Celsius)'].values
        tg = df[df.columns[3]].values
        sg = df[df.columns[4]].values

        delta_z = 0.3
        zn = int(np.round(np.max(dg) / delta_z))

        depthg = np.empty((zn, len(timeg)))
        depthg[:] = np.nan
        tempg = np.empty((zn, len(timeg)))
        tempg[:] = np.nan
        saltg = np.empty((zn, len(timeg)))
        saltg[:] = np.nan

        # Grid variables
        depthg_gridded = np.arange(0, np.nanmax(dg), delta_z)
        tempg_gridded = np.empty((len(depthg_gridded), len(timeg)))
        tempg_gridded[:] = np.nan
        saltg_gridded = np.empty((len(depthg_gridded), len(timeg)))
        saltg_gridded[:] = np.nan

        for i, ii in enumerate(ind):
            if i < len(timeg) - 1:
                depthg[0:len(dg[ind[i]:ind[i + 1]]), i] = dg[ind[i]:ind[i + 1]]
                tempg[0:len(tg[ind[i]:ind[i + 1]]), i] = tg[ind[i]:ind[i + 1]]
                saltg[0:len(sg[ind[i]:ind[i + 1]]), i] = sg[ind[i]:ind[i + 1]]
            else:
                depthg[0:len(dg[ind[i]:len(dg)]), i] = dg[ind[i]:len(dg)]
                tempg[0:len(tg[ind[i]:len(tg)]), i] = tg[ind[i]:len(tg)]
                saltg[0:len(sg[ind[i]:len(sg)]), i] = sg[ind[i]:len(sg)]

        for t, tt in enumerate(timeg):
            depthu, oku = np.unique(depthg[:, t], return_index=True)
            tempu = tempg[oku, t]
            saltu = saltg[oku, t]
            okdd = np.isfinite(depthu)
            depthf = depthu[okdd]
            tempf = tempu[okdd]
            saltf = saltu[okdd]

            okt = np.isfinite(tempf)
            if np.sum(okt) < 3:
                tempg_gridded[:, t] = np.nan
            else:
                okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),
                                     depthg_gridded < np.max(depthf[okt]))
                tempg_gridded[okd, t] = np.interp(depthg_gridded[okd], depthf[okt], tempf[okt])

            oks = np.isfinite(saltf)
            if np.sum(oks) < 3:
                saltg_gridded[:, t] = np.nan
            else:
                okd = np.logical_and(depthg_gridded >= np.min(depthf[oks]),
                                     depthg_gridded < np.max(depthf[oks]))
                saltg_gridded[okd, t] = np.interp(depthg_gridded[okd], depthf[oks], saltf[oks])

        # Conversion from glider longitude and latitude to AMSEAS convention
        target_lon_amseas = np.empty((len(long),))
        target_lon_amseas[:] = np.nan
        for i, ii in enumerate(long):
            if ii < 0:
                target_lon_amseas[i] = 360 + ii
            else:
                target_lon_amseas[i] = ii
        target_lat_amseas = latg

        # Narrowing time window of AMSEAS to coincide with glider time window
        tmin = mdates.num2date(mdates.date2num(timeg[0]))
        tmax = mdates.num2date(mdates.date2num(timeg[-1]))

        if isinstance(amseas, float):
            tstamp_amseas = np.nan
        else:
            tt_amseas = np.asarray([datetime(tamseas[i].year, tamseas[i].month, tamseas[i].day,
                                             tamseas[i].hour) for i in np.arange(len(tamseas))])
            tstamp_amseas = np.asarray([mdates.date2num(tt_amseas[i]) for i in np.arange(len(tt_amseas))])

        # Changing times to timestamp
        tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
        oktime_amseas = np.unique(
            np.round(np.interp(tstamp_glider, tstamp_amseas, np.arange(len(tstamp_amseas)))).astype(int))
        time_amseas = tt_amseas[oktime_amseas]

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
