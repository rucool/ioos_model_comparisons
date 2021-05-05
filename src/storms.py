#! /usr/bin/env

"""
Author: Lori Garzio on 2/19/2021
Last modified: 4/27/2021
Tools for analyzing specific storms
"""
import pandas as pd
import numpy as np
import xarray as xr
import cftime
from collections import namedtuple
from src.common import limits
from src.platforms import active_argo_floats, active_gliders


def convert_target_gofs_lon(target_lon):
    target_convert = np.array([])
    if np.logical_or(isinstance(target_lon, float), isinstance(target_lon, int)):
        target_lon = np.array([target_lon])
    for tc in target_lon:
        if tc < 0:
            target_convert = np.append(target_convert, 360 + tc)
        else:
            target_convert = np.append(target_convert, tc)
    return target_convert


def convert_gofs_target_lon(gofs_lon):
    gofslon_convert = np.array([])
    if np.logical_or(isinstance(gofs_lon, float), isinstance(gofs_lon, int)):
        gofs_lon = np.array([gofs_lon])
    for gl in gofs_lon:
        if gl > 180:
            gofslon_convert = np.append(gofslon_convert, gl - 360)
        else:
            gofslon_convert = np.append(gofslon_convert, gl)
    return gofslon_convert


def custom_transect(ds, varname, target_lons, target_lats, model):
    lat = ds.lat.values
    lon = ds.lon.values

    # find the lat/lon indicies closest to the lats/lons provided
    if model in ['gofs', 'cmems']:
        lon_idx = np.round(np.interp(target_lons, lon, np.arange(0, len(lon)))).astype(int)
        lat_idx = np.round(np.interp(target_lats, lat, np.arange(0, len(lat)))).astype(int)
        lon_subset = lon[lon_idx]
        lat_subset = lat[lat_idx]
        if model == 'gofs':
            lon_subset = convert_gofs_target_lon(lon_subset)
    elif model == 'rtofs':
        lon_idx = np.round(np.interp(target_lons, lon[0, :], np.arange(0, len(lon[0, :])))).astype(int)
        lat_idx = np.round(np.interp(target_lats, lat[:, 0], np.arange(0, len(lat[:, 0])))).astype(int)
        lon_subset = lon[0, lon_idx]
        lat_subset = lat[lat_idx, 0]

    depth = ds.depth.values
    target_var = np.empty((len(depth), len(lon_idx)))
    target_var[:] = np.nan

    for pos in range(len(lon_idx)):
        print(len(lon_idx), pos)
        if model in ['gofs', 'cmems']:
            target_var[:, pos] = ds.variables[varname][:, lat_idx[pos], lon_idx[pos]]
        elif model == 'rtofs':
            target_var[:, pos] = ds.variables[varname][0, :, lat_idx[pos], lon_idx[pos]]

    return target_var, depth, lon_subset, lat_subset


def forecast_storm_region(forecast_track):
    regions = limits(regions=['mab', 'gom', 'carib', 'sab'])

    # add shortened codes for the regions
    regions['Gulf of Mexico'].update(code='gom')
    regions['South Atlantic Bight'].update(code='sab')
    regions['Mid Atlantic Bight'].update(code='mab')
    regions['Caribbean'].update(code='carib')

    minlon = np.min(forecast_track['lon'])
    minlat = np.min(forecast_track['lat'])
    storm_region = dict()
    for name, region in regions.items():
        if np.logical_and(minlon >= region['lonlat'][0], minlon <= region['lonlat'][1]):
            if np.logical_and(minlat >= region['lonlat'][2], minlat <= region['lonlat'][3]):
                storm_region[name] = region
    if len(storm_region) < 1:
        if np.max(forecast_track['lon']) > -52:
            # Atlantic Ocean limits
            extent = [-60, -10, 8, 45]
            salinity = [dict(depth=0, limits=[34, 37])]
            sea_water_temperature = [dict(depth=0, limits=[22, 29])]

            storm_region['Atlantic'] = dict()
            atl = storm_region['Atlantic']
            atl.update(code='atl')
            atl.update(lonlat=extent)
            atl.update(salinity=salinity)
            atl.update(temperature=sea_water_temperature)

    return storm_region


def get_argo_data(extent, t0, t1):
    # Written by Mike Smith
    Argo = namedtuple('Argo', ['name', 'time', 'lon', 'lat'])
    argo_floats = []
    data = active_argo_floats(extent, t0, t1)

    if not data.empty:
        most_recent = data.loc[data.groupby('platform_number')['time (UTC)'].idxmax()]

        for float in most_recent.itertuples():
            A = Argo(float.platform_number, pd.to_datetime(float._2).strftime('%Y-%m-%dT%H:%M:%S'), float._4, float._5)
            argo_floats.append(A)

    return argo_floats


def return_ibtracs_storm(fname, storm_idx, variables):
    ibnc = xr.open_dataset(fname, mask_and_scale=False)
    nc = ibnc.sel(storm=storm_idx)

    # remove fill values and append data to dictionary
    d = dict()
    for v in variables:
        vv = nc[v]
        if v == 'time':
            fv = cftime.DatetimeGregorian(-25518, 1, 28, 0, 0, 0, 0)
        else:
            fv = vv._FillValue
        data = vv.values[vv != fv]
        if v == 'landfall':  # there is always one less landfall value, replace with last value
            data = np.append(data, data[-1])
        d[v] = data
    return d


def return_target_transect(target_lons, target_lats):
    # return a more dense target transect than provided by storm forecast or IBTrACS coordinates
    targetlon = np.array([])
    targetlat = np.array([])
    for ii, tl in enumerate(target_lons):
        if ii > 0:
            x1 = tl
            x2 = target_lons[ii - 1]
            y1 = target_lats[ii]
            y2 = target_lats[ii - 1]
            m = (y1 - y2) / (x1 - x2)
            b = y1 - m * x1
            #X = np.arange(x1, x2, 0.1)
            X = np.arange(x1, x2, 0.2)
            Y = b + m * X
            if ii == 1:
                targetlon = np.append(targetlon, x2)
                targetlat = np.append(targetlat, y2)
            targetlon = np.append(targetlon, X[::-1])
            targetlat = np.append(targetlat, Y[::-1])
    return targetlon, targetlat
