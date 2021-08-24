import os
import glob
import datetime as dt
import logging
import pandas as pd
import numpy as np


def list_files(main_dir):
    """
    :param types: file extension that you want to find
    :param main_dir: main directory that you want to recursively search for files
    :return:  file list
    """
    file_list = []  # create empty list for finding files

    sub_dirs = [os.path.join(main_dir, o) for o in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, o)) ]

    for sub in sub_dirs:
        file_list.extend(glob.glob(os.path.join(sub, '*.nc')))
    file_list = sorted(file_list)
    return file_list


def list_to_dataframe(file_list):
    df = pd.DataFrame(sorted(file_list), columns=['file'])
    try:
        df['date'] = df['file'].str.extract(r'(\d{4}\d{2}\d{2})')
        df['hour'] = df['file'].str.extract(r'(f\d{3})')
        df['hour'] = df['hour'].str.replace('f', '')
        df['time'] = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y%m%d'))
        df['time'] += pd.to_timedelta(df['hour'].astype(int), unit='h')
        df = df.set_index(['time']).sort_index()
    except ValueError:
        logging.error('Cannot pass empty file_list to function. Returning empty dataframe.')
    return df


def rename_model_variables(ds, model):
    """

    :param ds: xarray dataset
    :param model: <str> model name either 'gofs' or 'rtofs'
    :return: xarray dataset with standardized variable and dimension names
    """

    if model == 'rtofs':
        rename = {'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'}
    elif model == 'gofs':
        rename = {'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v', 'surf_el': 'height'}

    return ds.rename(rename)

