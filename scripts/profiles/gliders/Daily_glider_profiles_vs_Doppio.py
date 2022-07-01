#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:52:48 2019

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
lon_lim = [-77.5, -59.69]
lat_lim = [32.24, 46.61]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'
url_doppio = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'

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

# %% Get time bounds for the current day
ti = datetime.today() - timedelta(hours=6)
tini = datetime(ti.year, ti.month, ti.day)
te = ti + timedelta(1)
tend = datetime(te.year, te.month, te.day)

folder = os.path.join(f"{save_dir}/hurricane/Hurricane_season_{ti.strftime('%Y')}", ti.strftime('%m-%d'), 'Doppio')
os.makedirs(folder, exist_ok=True)

# Look for datasets in IOOS glider dac
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

# %% Read Doppio time, lat and lon
print('Retrieving coordinates and time from Doppio ')
doppio = xr.open_dataset(url_doppio, decode_times=False)

latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
srhodoppio = np.asarray(doppio.variables['s_rho'][:])
ttdoppio = doppio.variables['time'][:]
tdoppio = netCDF4.num2date(ttdoppio[:], ttdoppio.attrs['units'])

# %% Read Doppio S-coordinate parameters
Vtransf = np.asarray(doppio.variables['Vtransform'])
Vstrect = np.asarray(doppio.variables['Vstretching'])
Cs_r = np.asarray(doppio.variables['Cs_r'])
Cs_w = np.asarray(doppio.variables['Cs_w'])
sc_r = np.asarray(doppio.variables['s_rho'])
sc_w = np.asarray(doppio.variables['s_w'])

# depth
h = np.asarray(doppio.variables['h'])

# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

igrid = 1

# %% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

# Looping through all gliders found
for id in gliders:
    print('Reading ' + id)
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    # checking data frame is not empty
    df = e.to_pandas()
    # if len(df.dropna()) != 0:
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

        # Narrowing time window of Doppio to coincide with glider time window
        tmin = mdates.num2date(mdates.date2num(timeg[0]))
        tmax = mdates.num2date(mdates.date2num(timeg[-1]))

        # Changing times to timestamp
        ttdoppio = [datetime(tdoppio[i].year, tdoppio[i].month, tdoppio[i].day,
                             tdoppio[i].hour) for i in np.arange(len(tdoppio))]
        ttstamp_doppio = [mdates.date2num(ttdoppio[i]) for i in np.arange(len(ttdoppio))]
        tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

        # oktime_doppio = np.where(np.logical_and(tdoppio >= tmin,tdoppio <= tmax))
        oktime_doppio = np.unique(
            np.round(np.interp(tstamp_glider, ttstamp_doppio, np.arange(len(ttstamp_doppio)))).astype(int))
        timedoppio = tdoppio[oktime_doppio]

        # Changing times to timestamp
        timedoppio = [datetime(timedoppio[i].year, timedoppio[i].month, timedoppio[i].day,
                               timedoppio[i].hour) for i in np.arange(len(timedoppio))]
        tstamp_doppio = [mdates.date2num(timedoppio[i]) for i in np.arange(len(timedoppio))]

        # interpolating glider lon and lat to lat and lon on doppio time
        sublondoppio = np.interp(tstamp_doppio, tstamp_glider, long)
        sublatdoppio = np.interp(tstamp_doppio, tstamp_glider, latg)

        # getting the model grid positions for sublonm and sublatm
        oklatdoppio = np.empty((len(oktime_doppio)))
        oklatdoppio[:] = np.nan
        oklondoppio = np.empty((len(oktime_doppio)))
        oklondoppio[:] = np.nan
        for t, tt in enumerate(oktime_doppio):

            # search in xi_rho direction
            oklatmm = []
            oklonmm = []
            for pos_xi in np.arange(latrhodoppio.shape[1]):
                pos_eta = np.round(
                    np.interp(sublatdoppio[t], latrhodoppio[:, pos_xi], np.arange(len(latrhodoppio[:, pos_xi])),
                              left=np.nan, right=np.nan))
                if np.isfinite(pos_eta):
                    oklatmm.append((pos_eta).astype(int))
                    oklonmm.append(pos_xi)

            pos = np.round(np.interp(sublondoppio[t], lonrhodoppio[oklatmm, oklonmm],
                                     np.arange(len(lonrhodoppio[oklatmm, oklonmm])))).astype(int)
            oklatdoppio1 = oklatmm[pos]
            oklondoppio1 = oklonmm[pos]

            # search in eta-rho direction
            oklatmm = []
            oklonmm = []
            for pos_eta in np.arange(latrhodoppio.shape[0]):
                pos_xi = np.round(
                    np.interp(sublondoppio[t], lonrhodoppio[pos_eta, :], np.arange(len(lonrhodoppio[pos_eta, :])),
                              left=np.nan, right=np.nan))
                if np.isfinite(pos_xi):
                    oklatmm.append(pos_eta)
                    oklonmm.append(pos_xi.astype(int))

            pos_lat = np.round(np.interp(sublatdoppio[t], latrhodoppio[oklatmm, oklonmm],
                                         np.arange(len(latrhodoppio[oklatmm, oklonmm])))).astype(int)
            oklatdoppio2 = oklatmm[pos_lat]
            oklondoppio2 = oklonmm[pos_lat]

            # check for minimum distance
            dist1 = np.sqrt((oklondoppio1 - sublondoppio[t]) ** 2 + (oklatdoppio1 - sublatdoppio[t]) ** 2)
            dist2 = np.sqrt((oklondoppio2 - sublondoppio[t]) ** 2 + (oklatdoppio2 - sublatdoppio[t]) ** 2)
            if dist1 >= dist2:
                oklatdoppio[t] = oklatdoppio1
                oklondoppio[t] = oklondoppio1
            else:
                oklatdoppio[t] = oklatdoppio2
                oklondoppio[t] = oklondoppio2

        oklatdoppio = oklatdoppio.astype(int)
        oklondoppio = oklondoppio.astype(int)

        if np.logical_or(oklondoppio.max() == 0, oklatdoppio.max() == 0):
            continue
        if np.logical_or(oklondoppio.min() <= 1, oklatdoppio.min() <= 1):
            continue

            # Getting glider transect from doppio
        print('Getting glider transect from Doppio')
        target_tempdoppio = np.empty((len(srhodoppio), len(oktime_doppio)))
        target_tempdoppio[:] = np.nan
        target_saltdoppio = np.empty((len(srhodoppio), len(oktime_doppio)))
        target_saltdoppio[:] = np.nan
        target_zdoppio = np.empty((len(srhodoppio), len(oktime_doppio)))
        target_zdoppio[:] = np.nan
        for i in range(len(oktime_doppio)):
            print(len(oktime_doppio), ' ', i)
            target_tempdoppio[:, i] = doppio.variables['temp'][oktime_doppio[i], :, oklatdoppio[i], oklondoppio[i]]
            target_saltdoppio[:, i] = doppio.variables['salt'][oktime_doppio[i], :, oklatdoppio[i], oklondoppio[i]]
            h = np.asarray(doppio.variables['h'][oklatdoppio[i], oklondoppio[i]])
            zeta = np.asarray(doppio.variables['zeta'][oktime_doppio[i], oklatdoppio[i], oklondoppio[i]])

            # Calculate doppio depth as a function of time
            if Vtransf == 1:
                if igrid == 1:
                    for k in np.arange(sc_r.shape[0]):
                        z0 = (sc_r[k] - Cs_r[k]) * hc + Cs_r[k] * h
                        target_zdoppio[k, i] = z0 + zeta * (1.0 + z0 / h)

            if Vtransf == 2:
                if igrid == 1:
                    for k in np.arange(sc_r.shape[0]):
                        z0 = (hc * sc_r[k] + Cs_r[k] * h) / (hc + h)
                        target_zdoppio[k, i] = zeta + (zeta + h) * z0

        # change time vector to matrix
        target_timedoppio = np.tile(timedoppio, (len(srhodoppio), 1))

        # min and max values for plotting
        min_temp = np.floor(np.min([np.nanmin(df[df.columns[3]]), np.nanmin(target_tempdoppio)]))
        max_temp = np.ceil(np.max([np.nanmax(df[df.columns[3]]), np.nanmax(target_tempdoppio)]))

        min_salt = np.floor(np.min([np.nanmin(df[df.columns[4]]), np.nanmin(target_saltdoppio)]))
        max_salt = np.ceil(np.max([np.nanmax(df[df.columns[4]]), np.nanmax(target_saltdoppio)]))

        # Temperature profile
        fig, ax = plt.subplots(figsize=(14, 12))
        grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

        ax = plt.subplot(grid[:, 0])
        plt.plot(tempg, -depthg, '.', color='cyan', label='_nolegend_')
        plt.plot(np.nanmean(tempg_gridded, axis=1), -depthg_gridded, '.-b',
                 label=f'{id[:-14]} {str(timeg[0])[0:4]} [{str(timeg[0])[5:19]}, {str(timeg[-1])[5:19]}]'
                 )

        plt.plot(target_tempdoppio[::3], target_zdoppio[::3], '.-', color='peachpuff', label='_nolegend_')
        plt.plot(np.nanmean(target_tempdoppio, axis=1), np.nanmean(target_zdoppio, axis=1), '.-', color='maroon',
                 markersize=12, linewidth=2,
                 label='Doppio' + ' ' + str(timedoppio[0].year) + ' ' + '[' + str(timedoppio[0])[5:13] + ',' + str(
                     timedoppio[-1])[5:13] + ']')
        plt.xlabel('Temperature ($^oC$)', fontsize=20)
        plt.ylabel('Depth (m)', fontsize=20)
        plt.title('Temperature Profile ' + id, fontsize=20)
        plt.ylim([-np.nanmax(depthg) + 100, 0])
        if np.nanmax(depthg) <= 100:
            plt.ylim([-np.nanmax(depthg) - 30, 0.1])
        else:
            plt.ylim([-np.nanmax(depthg) - 100, 0.1])
        plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0.0), fontsize=14)
        plt.grid('on')

        # lat and lon bounds of box to draw
        minlondoppio = np.min(
            lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        maxlondoppio = np.max(
            lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        minlatdoppio = np.min(
            latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        maxlatdoppio = np.max(
            latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])

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

        plt.plot(long, latg, '.-', color='orange', label='Glider track',
                 markersize=3)
        rect = patches.Rectangle((minlondoppio, minlatdoppio),
                                 maxlondoppio - minlondoppio, maxlatdoppio - minlatdoppio,
                                 linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        plt.title('Glider track and model grid positions', fontsize=20)
        plt.axis('scaled')
        plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1))

        ax = plt.subplot(grid[2, 1])
        ax.plot(long, latg, '.-', color='orange', label='Glider track')
        ax.plot(long[0], latg[0], 'sc', label='Initial profile time ' + str(timeg[0])[0:16])
        ax.plot(long[-1], latg[-1], 'sb', label='final profile time ' + str(timeg[-1])[0:16])
        ax.plot(lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2],
                latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2],
                '.', color='maroon')
        ax.plot(lonrhodoppio[oklatdoppio, oklondoppio], latrhodoppio[oklatdoppio, oklondoppio], 'o',
                color='maroon', markersize=8, markeredgecolor='k',
                label='Doppio grid points \n nx = ' + str(oklondoppio[::3])
                      + '\n ny = ' + str(oklatdoppio[::3])
                      + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in timedoppio[::3]]))
                      + str([str(i)[11:13] for i in timedoppio[::3]]))
        plt.axis('scaled')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

        file = os.path.join(folder, f'{id}_temp_profile_{tini.strftime("%Y%m%d")}_to_{tend.strftime("%Y%m%d")}')
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1)

        # Salinity profile
        fig, ax = plt.subplots(figsize=(14, 12))
        grid = plt.GridSpec(5, 2, wspace=0.4, hspace=0.3)

        ax = plt.subplot(grid[:, 0])
        plt.plot(saltg, -depthg, '.', color='cyan')
        plt.plot(np.nanmean(saltg_gridded, axis=1), -depthg_gridded, '.-b',
                 label=id[:-14] + ' ' + str(timeg[0])[0:4] + ' ' + '[' + str(timeg[0])[5:19] + ',' + str(timeg[-1])[
                                                                                                     5:19] + ']')
        plt.plot(target_saltdoppio, target_zdoppio, '.-', color='peachpuff', label='_nolegend_')
        plt.plot(np.nanmean(target_saltdoppio, axis=1), np.nanmean(target_zdoppio, axis=1), '.-', color='maroon',
                 markersize=12, linewidth=2,
                 label='Doppio' + ' ' + str(timedoppio[0].year) + ' ' + '[' + str(timedoppio[0])[5:13] + ',' + str(
                     timedoppio[-1])[5:13] + ']')
        plt.ylabel('Depth (m)', fontsize=20)
        plt.xlabel('Salinity', fontsize=20)
        plt.title('Salinity Profile ' + id, fontsize=20)
        plt.ylim([-np.nanmax(depthg) + 100, 0])
        if np.nanmax(depthg) <= 100:
            plt.ylim([-np.nanmax(depthg) - 30, 0.1])
        else:
            plt.ylim([-np.nanmax(depthg) - 100, 0.1])
        plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0.0), fontsize=14)
        plt.grid('on')

        # lat and lon bounds of box to draw
        minlondoppio = np.min(
            lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        maxlondoppio = np.max(
            lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        minlatdoppio = np.min(
            latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])
        maxlatdoppio = np.max(
            latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2])

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

        plt.plot(long, latg, '.-', color='orange', label='Glider track',
                 markersize=3)
        rect = patches.Rectangle((minlondoppio, minlatdoppio),
                                 maxlondoppio - minlondoppio, maxlatdoppio - minlatdoppio,
                                 linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        plt.title('Glider track and model grid positions', fontsize=20)
        plt.axis('scaled')
        plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1))

        ax = plt.subplot(grid[2, 1])
        ax.plot(long, latg, '.-', color='orange', label='Glider track')
        ax.plot(long[0], latg[0], 'sc', label='Initial profile time ' + str(timeg[0])[0:16])
        ax.plot(long[-1], latg[-1], 'sb', label='final profile time ' + str(timeg[-1])[0:16])
        ax.plot(lonrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2],
                latrhodoppio[oklatdoppio.min() - 2:oklatdoppio.max() + 2, oklondoppio.min() - 2:oklondoppio.max() + 2],
                '.', color='maroon')
        ax.plot(lonrhodoppio[oklatdoppio, oklondoppio], latrhodoppio[oklatdoppio, oklondoppio], 'o',
                color='maroon', markersize=8, markeredgecolor='k',
                label='Doppio grid points \n nx = ' + str(oklondoppio[::3])
                      + '\n ny = ' + str(oklatdoppio[::3])
                      + '\n [day][hour] = ' + str(np.unique([str(i)[8:10] for i in timedoppio[::3]]))
                      + str([str(i)[11:13] for i in timedoppio[::3]]))
        plt.axis('scaled')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

        file = os.path.join(folder, f'{id}_salt_profile_{tini.strftime("%Y%m%d")}_to_{tend.strftime("%Y%m%d")}')
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1)
