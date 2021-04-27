#! /usr/bin/env

"""
Author: Lori Garzio on 4/27/2021
Last modified: 4/27/2021
Tools for grabbing glider data
"""
import pandas as pd
import numpy as np
import datetime as dt
from erddapy import ERDDAP


ioos_url = 'https://data.ioos.us/gliders/erddap'


def get_erddap_nc(ds_id, var_list=None, constraints=None):
    """
    Returns a netcdf dataset for a specified dataset ID
    :param ds_id: dataset ID e.g. ng314-20200806T2040
    :param var_list: optional list of variables
    :param constraints: optional list of constraints
    :return: netcdf dataset
    """
    e = ERDDAP(server=ioos_url,
               protocol='tabledap',
               response='nc')
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if var_list:
        e.variables = var_list
    try:
        ds = e.to_xarray()
        ds = ds.sortby(ds.time)
    except OSError:
        print('No dataset available for specified constraints: {}'.format(ds_id))
        ds = None

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

    gl_data = dict()
    for glid in gliders:
        ds = get_erddap_nc(glid, variables, constraints)
        if ds:
            print('Reading ' + glid)
            gl_data[glid] = dict()
            for v in variables:
                gl_data[glid][v] = ds[v].data

    return gl_data


def glider_summary(gldata, savefile):
    rows = []
    for glid, values in gldata.items():
        t0 = pd.to_datetime(np.min(values['time'])).strftime('%Y-%m-%dT%H:%M:%S')
        tf = pd.to_datetime(np.max(values['time'])).strftime('%Y-%m-%dT%H:%M:%S')
        lat_lims = [np.round(np.min(values['latitude']), 2), np.round(np.max(values['latitude']), 2)]
        lon_lims = [np.round(np.min(values['longitude']), 2), np.round(np.max(values['longitude']), 2)]
        rows.append([glid, t0, tf, lat_lims, lon_lims])
    df = pd.DataFrame(rows, columns=['dataset_id', 't0', 'tf', 'lat_lims', 'lon_lims'])
    df.to_csv(savefile, index=False)
