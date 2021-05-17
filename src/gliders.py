#! /usr/bin/env

"""
Author: Lori Garzio on 4/27/2021
Last modified: 5/7/2021
Tools for grabbing glider data
"""
import pandas as pd
import numpy as np
import datetime as dt
from erddapy import ERDDAP
from src.common import limits


ioos_url = 'https://data.ioos.us/gliders/erddap'


def grid_glider_data(df, varname, delta_z=.3):
    """
    Written by aristizabal. Returns a gridded glider dataset by depth and time
    """
    df.dropna(inplace=True)
    df.drop(df[df['depth'] < .1].index, inplace=True)  # drop rows where depth is <1
    df.drop(df[df[varname] == 0].index, inplace=True)  # drop rows where the variable equals zero
    df.sort_values(by=['time', 'depth'], inplace=True)

    # find unique times and coordinates
    timeg, ind = np.unique(df.time.values, return_index=True)
    latg = df['latitude'].values[ind]
    long = df['longitude'].values[ind]
    dg = df['depth'].values
    vg = df[varname].values
    zn = np.int(np.max(np.diff(np.hstack([ind, len(dg)]))))

    depthg = np.empty((zn, len(timeg)))
    depthg[:] = np.nan
    varg = np.empty((zn, len(timeg)))
    varg[:] = np.nan

    for i, ii in enumerate(ind):
        if i < len(timeg) - 1:
            i_f = ind[i + 1]
        else:
            i_f = len(dg)
        depthi = dg[ind[i]:i_f]
        vari = vg[ind[i]:i_f]
        depthg[0:len(dg[ind[i]:i_f]), i] = depthi
        varg[0:len(vg[ind[i]:i_f]), i] = vari

    # sort time variable
    okt = np.argsort(timeg)
    timegg = timeg[okt]
    depthgg = depthg[:, okt]
    vargg = varg[:, okt]

    # Grid variables
    depthg_gridded = np.arange(0, np.nanmax(depthgg), delta_z)
    varg_gridded = np.empty((len(depthg_gridded), len(timegg)))
    varg_gridded[:] = np.nan

    for t, tt in enumerate(timegg):
        depthu, oku = np.unique(depthgg[:, t], return_index=True)
        varu = vargg[oku, t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        varf = varu[okdd]
        ok = np.asarray(np.isfinite(varf))
        if np.sum(ok) < 3:
            varg_gridded[:, t] = np.nan
        else:
            okd = np.logical_and(depthg_gridded >= np.min(depthf[ok]), depthg_gridded < np.max(depthf[ok]))
            varg_gridded[okd, t] = np.interp(depthg_gridded[okd], depthf[ok], varf[ok])

    return timegg, long, latg, depthg_gridded, varg_gridded


def custom_gliderline_transects():
    """
    Define specific straight line transects for glider deployments
    """
    custom_transects = dict()
    custom_transects['maracoos_02-20210503T1937'] = dict()
    custom_transects['maracoos_02-20210503T1937']['extent'] = [-74.24, 39.43, -72.5, 38.8]

    custom_transects['ru30-20210503T1929'] = dict()
    custom_transects['ru30-20210503T1929']['extent'] = [-74.22, 39.37, -72.92, 38.83]

    return custom_transects


def get_erddap_dataset(ds_id, variables=None, constraints=None, filetype=None):
    """
    Returns a netcdf dataset for a specified dataset ID (or dataframe if dataset cannot be converted to xarray)
    :param ds_id: dataset ID e.g. ng314-20200806T2040
    :param variables: optional list of variables
    :param constraints: optional list of constraints
    :param filetype: optional filetype to return, 'nc' (default) or 'dataframe'
    :return: netcdf dataset
    """
    variables = variables or None
    constraints = constraints or None
    filetype = filetype or 'nc'

    e = ERDDAP(server=ioos_url,
               protocol='tabledap',
               response='nc')
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if variables:
        e.variables = variables
    if filetype == 'nc':
        try:
            ds = e.to_xarray()
            ds = ds.sortby(ds.time)
        except OSError:
            print('No dataset available for specified constraints: {}'.format(ds_id))
            ds = []
        except TypeError:
            print('Cannot convert to xarray, providing dataframe: {}'.format(ds_id))
            ds = e.to_pandas().dropna()
    elif filetype == 'dataframe':
        ds = e.to_pandas().dropna()
    else:
        print('Unrecognized filetype: {}. Needs to  be "nc" or "dataframe"'.format(filetype))

    return ds


def return_glider_ids(kwargs):
    """
    Searches an ERDDAP server for datasets and returns dataset IDs
    :param kwargs: dictionary containing coordinate and time limits
    :return: array containing dataset IDs
    """
    e = ERDDAP(server=ioos_url)
    search_url = e.get_search_url(response='csv', **kwargs)
    try:
        search = pd.read_csv(search_url)
        ds_ids = search['Dataset ID'].values
    except:
        ds_ids = np.array([])

    return ds_ids


def glider_data(bbox=None, time_start=None, time_end=None):
    """
    Return data from all gliders found within specified region and times
    """
    bbox = bbox or [-100, -40, 18, 60]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))
    t0 = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Search constraints
    kw = {
        'min_lon': bbox[0],
        'max_lon': bbox[1],
        'min_lat': bbox[2],
        'max_lat': bbox[3],
        'min_time': t0,
        'max_time': t1,
    }

    gliders = return_glider_ids(kw)

    msg = 'Found {} Glider Datasets:\n\n{}'.format
    print(msg(len(gliders), '\n'.join(gliders)))

    # Setting constraints
    constraints = {
        'time>=': t0,
        'time<=': t1,
        'longitude>=': bbox[0],
        'longitude<=': bbox[1],
        'latitude>=': bbox[2],
        'latitude<=': bbox[3],
    }

    variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity',
    ]

    kwargs = dict()
    kwargs['variables'] = variables
    kwargs['constraints'] = constraints
    gl_data = dict()
    for glid in gliders:
        ds = get_erddap_dataset(glid, **kwargs)
        if len(ds) > 0:
            print('Reading ' + glid)
            gl_data[glid] = dict()
            if isinstance(ds, pd.core.frame.DataFrame):
                for col in ds.columns:
                    ds.rename(columns={col: col.split(' ')[0]}, inplace=True)
            for v in variables:
                gl_data[glid][v] = np.array(ds[v])

    return gl_data


def glider_dataset(gliderid, time_start=None, time_end=None, variables=None, filetype=None):
    """
    Return data from a specific glider
    """
    print('Retrieving glider dataset: {}'.format(gliderid))
    time_start = time_start or None
    time_end = time_end or None
    variables = variables or [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity',
    ]
    filetype = filetype or 'nc'

    constraints = dict()
    if time_start:
        constraints['time>='] = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    if time_end:
        constraints['time<='] = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if len(constraints) < 1:
        constraints = None

    kwargs = dict()
    kwargs['variables'] = variables
    kwargs['constraints'] = constraints
    kwargs['filetype'] = filetype

    ds = get_erddap_dataset(gliderid, **kwargs)
    if isinstance(ds, pd.core.frame.DataFrame):
        for col in ds.columns:
            ds.rename(columns={col: col.split(' ')[0]}, inplace=True)  # get rid of units in column names
        ds['time'] = ds['time'].apply(lambda t: dt.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ'))

    return ds


def glider_region(ds):
    regions = limits(regions=['mab', 'gom', 'carib', 'sab'])

    # add shortened codes for the regions
    regions['Gulf of Mexico'].update(code='gom')
    regions['South Atlantic Bight'].update(code='sab')
    regions['Mid Atlantic Bight'].update(code='mab')
    regions['Caribbean'].update(code='carib')

    minlon = np.nanmin(ds.longitude.values)
    minlat = np.nanmin(ds.latitude.values)
    glider_region = dict()
    for name, region in regions.items():
        if np.logical_and(minlon >= region['lonlat'][0], minlon <= region['lonlat'][1]):
            if np.logical_and(minlat >= region['lonlat'][2], minlat <= region['lonlat'][3]):
                glider_region[name] = region
    if len(glider_region) < 1:
        if np.nanmax(ds.longitude.values) > -52:
            # Atlantic Ocean limits
            extent = [-60, -10, 8, 45]
            salinity = [dict(depth=0, limits=[34, 37, 0.1])]
            sea_water_temperature = [dict(depth=0, limits=[22, 29, 0.5])]

            glider_region['Atlantic'] = dict()
            atl = glider_region['Atlantic']
            atl.update(code='atl')
            atl.update(lonlat=extent)
            atl.update(salinity=salinity)
            atl.update(temperature=sea_water_temperature)

    # add zoomed glider region
    extent = [np.nanmin(ds.longitude.values) - 2.5, np.nanmax(ds.longitude.values) + 2.5,
              np.nanmin(ds.latitude.values) - 2, np.nanmax(ds.latitude.values) + 2]
    salinity = [dict(depth=0, limits=[30, 36, 0.1])]
    sea_water_temperature = [dict(depth=0, limits=[8, 18, 0.5])]

    glider_region['zoomed'] = dict()
    zoom = glider_region['zoomed']
    zoom.update(code='zoom')
    zoom.update(lonlat=extent)
    zoom.update(salinity=salinity)
    zoom.update(temperature=sea_water_temperature)

    return glider_region


def glider_summary(gldata, savefile):
    """
    Save a summary of glider datasets
    """
    rows = []
    for glid, values in gldata.items():
        t0 = pd.to_datetime(np.min(values['time'])).strftime('%Y-%m-%dT%H:%M:%S')
        tf = pd.to_datetime(np.max(values['time'])).strftime('%Y-%m-%dT%H:%M:%S')
        lat_lims = [np.round(np.min(values['latitude']), 2), np.round(np.max(values['latitude']), 2)]
        lon_lims = [np.round(np.min(values['longitude']), 2), np.round(np.max(values['longitude']), 2)]
        rows.append([glid, t0, tf, lat_lims, lon_lims])
    df = pd.DataFrame(rows, columns=['dataset_id', 't0', 'tf', 'lat_lims', 'lon_lims'])
    df.to_csv(savefile, index=False)
