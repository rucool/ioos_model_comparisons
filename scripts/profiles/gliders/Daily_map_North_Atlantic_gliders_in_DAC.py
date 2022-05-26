#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:00:04 2019

@author: aristizabal
"""
import glob
import os
import urllib.request
from datetime import datetime, timedelta
from zipfile import ZipFile

import cartopy
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup
from cartopy.io.shapereader import Reader
from erddapy import ERDDAP

# Do not produce figures on screen
plt.switch_backend('agg')

# %% User input
# lat and lon bounds
lon_lim = [-110.0, -10.0]
lat_lim = [15.0, 45.0]

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'
url_nhc = 'https://www.nhc.noaa.gov/gis/'

# Bathymetry file
bath_file = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# EEZs file
file_EEZs = '/home/hurricaneadm/World_EEZ_v11_20191118/eez_boundaries_v11.shp'

# Increase fontsize of labels globally
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

# %% Get time bounds for the current day
ti = datetime.today() - timedelta(hours=6)
tini = datetime(ti.year, ti.month, ti.day)
te = ti + timedelta(1)
tend = datetime(te.year, te.month, te.day)

folder = os.path.join(f"/www/web/rucool/hurricane/hurricane_season_{ti.strftime('%Y')}", ti.strftime('%m-%d'))
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
# print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# get entire deployment (lat and lon) during hurricane season

# Time bounds
min_time2 = str(tini.year) + '-01-01T00:00:00Z'
max_time2 = str(tini.year) + '-12-31T00:00:00Z'

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': min_time2,
    'max_time': max_time2,
}

search_url = e.get_search_url(response='csv', **kw)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders_all = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders_all), '\n'.join(gliders_all)))

# Setting constraints
constraints = {
    'time>=': min_time2,
    'time<=': max_time2,
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[1],
}

variables = [
    'depth',
    'latitude',
    'longitude',
    'time'
]

e = ERDDAP(
    server=url_glider,
    protocol='tabledap',
    response='nc'
)

# %% Reading bathymetry data
ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0], bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0], bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath, :]
bath_elevsub = bath_elevs[:, oklonbath]

# %% Download kmz files
os.system('rm -rf *best_track*')
os.system('rm -rf *TRACK_latest*')
os.system('rm -rf *CONE_latest*')

r = requests.get(url_nhc)
data = r.text

soup = BeautifulSoup(data, "html.parser")

for i, s in enumerate(soup.find_all("a")):
    ff = s.get('href')
    if type(ff) == str:
        if np.logical_and('kmz' in ff, str(tini.year) in ff):
            if 'CONE_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff, file_name)
            if 'TRACK_latest' in ff:
                file_name = ff.split('/')[3]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc[:-4] + ff, file_name)
            if 'best_track' in ff:
                file_name = ff.split('/')[1]
                print(ff, file_name)
                urllib.request.urlretrieve(url_nhc + ff, file_name)

# %%
kmz_files = glob.glob('*.kmz')

# NOTE: UNTAR  the .kmz FILES AND THEN RUN FOLLOWING CODE
for f in kmz_files:
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')
    os.system('unzip -o ' + f + ' -d ' + f[:-4])

# %% get names zip and kml files
zip_files = glob.glob('*.zip')
zip_files = [f for f in zip_files if np.logical_or('al' in f, 'AL' in f)]

# %% Map of North Atlantic with glider tracks
col = ['red', 'darkcyan', 'gold', 'm', 'darkorange', 'crimson', 'lime',
       'darkorchid', 'brown', 'sienna', 'yellow', 'orchid', 'gray',
       'darkcyan', 'gold', 'm', 'darkorange', 'crimson', 'lime', 'red',
       'darkorchid', 'brown', 'sienna', 'yellow', 'orchid', 'gray',
       'rosybrown', 'peru', 'olive', 'springgreen', 'deepskyblue',
       'midnightblue', 'slateblue', 'indigo', 'tan', 'coral']

mark = ['o', '*', 'p', '^', 'D', 'X', 'o', '*', 'p', '^', 'D', '*',
        'o', '*', 'p', '^', 'D', 'X', 'o', '*', 'p', '^', 'D', 'o',
        'o', '*', 'p', '^', 'D', 'X', 'o', '*', 'p', '^', 'D', 'X']

lev = np.arange(-9000, 9100, 100)
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
plt.contourf(bath_lon, bath_lat, bath_elev, lev, cmap=cmocean.cm.topo)
plt.yticks([])
plt.xticks([])
plt.axis([-100, -10, 0, 50])
ax.set_aspect(1)
plt.title('Active Glider Deployments on ' + str(tini)[0:10], fontsize=20)

coast = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
ax.add_feature(coast, edgecolor='black', facecolor='none')
ax.add_feature(cfeature.BORDERS)  # adds country borders  
ax.add_feature(cfeature.STATES)
shape_feature = cfeature.ShapelyFeature(Reader(file_EEZs).geometries(),
                                        cartopy.crs.PlateCarree(), edgecolor='grey', facecolor='none')
ax.add_feature(shape_feature, zorder=1)

for i, f in enumerate(zip_files):
    kmz = ZipFile(f, 'r')
    if 'TRACK' in f:
        kml_f = glob.glob(f[:-4] + '/*.kml')
        kml_track = kmz.open(kml_f[0].split('/')[1], 'r').read()

        # %% Get TRACK coordinates
        soup = BeautifulSoup(kml_track, 'html.parser')

        lon_forec_track = np.empty(len(soup.find_all("point")))
        lon_forec_track[:] = np.nan
        lat_forec_track = np.empty(len(soup.find_all("point")))
        lat_forec_track[:] = np.nan
        for i, s in enumerate(soup.find_all("point")):
            print(s.get_text("coordinates"))
            lon_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
            lat_forec_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])

        plt.plot(lon_forec_track, lat_forec_track, '.-', color='gold')

    else:
        if 'CONE' in f:
            kml_f = glob.glob(f[:-4] + '/*.kml')
            kml_cone = kmz.open(kml_f[0].split('/')[1], 'r').read()

            # %% CONE coordinates
            soup = BeautifulSoup(kml_cone, 'html.parser')

            lon_forec_cone = []
            lat_forec_cone = []
            for i, s in enumerate(soup.find_all("coordinates")):
                coor = s.get_text('coordinates').split(',0')
                for st in coor[1:-1]:
                    lon_forec_cone.append(st.split(',')[0])
                    lat_forec_cone.append(st.split(',')[1])

            lon_forec_cone = np.asarray(lon_forec_cone).astype(float)
            lat_forec_cone = np.asarray(lat_forec_cone).astype(float)

            plt.plot(lon_forec_cone, lat_forec_cone, '.-b', markersize=1)

        else:
            kml_f = glob.glob(f[:-4] + '/*.kml')
            kml_best_track = kmz.open(kml_f[0].split('/')[1], 'r').read()

            # %% best track coordinates
            soup = BeautifulSoup(kml_best_track, 'html.parser')

            lon_best_track = np.empty(len(soup.find_all("point")))
            lon_best_track[:] = np.nan
            lat_best_track = np.empty(len(soup.find_all("point")))
            lat_best_track[:] = np.nan
            for i, s in enumerate(soup.find_all("point")):
                print(s.get_text("coordinates"))
                lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
                lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])

            # get name
            for f in soup.find_all('name'):
                if 'AL' in f.get_text('name'):
                    name = f.get_text('name')

            plt.plot(lon_best_track, lat_best_track, 'or', markersize=3)
            plt.text(np.mean(lon_best_track), np.mean(lat_best_track), name.split(' ')[-1], fontsize=14,
                     fontweight='bold', bbox=dict(facecolor='white', alpha=0.3))

i = 0
for j, id_all in enumerate(gliders_all):
    id = [id for id in gliders if id == id_all]
    if len(id) != 0:
        i += 1
        print(i - 1)
        print(id[0])
        e.dataset_id = id[0]
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas()
        if len(df.index) != 0:
            df = e.to_pandas(
                index_col='time (UTC)',
                parse_dates=True,
                skiprows=(1,)  # units information can be dropped.
            ).dropna()

            print(len(df))

            timeg, ind = np.unique(df.index.values, return_index=True)
            latg = df['latitude (degrees_north)'].values[ind]
            long = df['longitude (degrees_east)'].values[ind]
            ax.plot(long, latg, '.-',
                    color='darkorange',
                    markersize=0.4)
            ax.plot(long[-1], latg[-1],
                    color=col[i - 1],
                    marker=mark[i - 1],
                    markeredgecolor='k',
                    markersize=7,
                    label=id[0].split('-')[0])

plt.legend(loc='upper center', bbox_to_anchor=(1.05, 1))
file = os.path.join(folder, 'Daily_map_North_Atlantic_gliders_in_DAC_' + str(tini).split()[0] + '_' + str(tend).split()[0] + '.png')
plt.savefig(file, bbox_inches='tight', pad_inches=0.1)

# %%
os.system('rm -rf *best_track*')
os.system('rm -rf *TRACK_latest*')
os.system('rm -rf *CONE_latest*')
