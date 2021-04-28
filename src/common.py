import os
import glob
import datetime as dt
import logging
import pandas as pd
import numpy as np


def transects():
    # transect coordinates and variable limits
    # mesoscale features (mostly brought in by altimetry)
    transects = dict(
        loop_current_eddy=dict(extent=[-90, 26, -85, 26],
                               limits=dict(temperature=dict(deep=np.arange(4, 28), shallow=np.arange(14, 28)),
                                           salinity=dict(deep=np.arange(34.8, 37, 0.1), shallow=np.arange(34.8, 37, 0.1) ))
                               ),
        green_blob=dict(extent=[-93.5, 26.25, -91.5, 26.25],
                               limits=dict(temperature=dict(deep=np.arange(4, 26), shallow=np.arange(10, 26)),
                                           salinity=dict(deep=np.arange(34.7, 37, 0.1), shallow=np.arange(35.4, 37, 0.1) ))
                               ),
        cold_water_arm_western_gulf=dict(extent=[-94, 27, -92.5, 27],
                               limits=dict(temperature=dict(deep=np.arange(4, 26), shallow=np.arange(11, 26)),
                                           salinity=dict(deep=np.arange(34.8, 36.5, 0.1), shallow=np.arange(34.8, 36.5, 0.1) ))
                               ),
        two_eddies_in_western_gulf=dict(extent=[-96.25, 26.25, -91.25, 26.25],
                                        limits=dict(temperature=dict(deep=np.arange(4, 26), shallow=np.arange(10, 26)),
                                                    salinity=dict(deep=np.arange(34.7, 37, 0.1), shallow=np.arange(35.4, 37, 0.1)))
                                        ),
    )
    return transects


def limits(model=None):
    """
    return extent and other variable limits of certain regions for rtofs or gofs
    :param model: rtofs or gofs
    :return: dictionary containing limits
    """

    model = model or 'rtofs'

    # Specify common variable and region limits for both gofs and rtofs
    # To add different depths for each variable, append to the specific variable list the following format:
    # dict(depth=n, limits=[min, max, stride])

    # Gulf of Mexico Limits
    gom_extent = [-100, -80, 18, 32]
    gom_sea_water_temperature = [dict(depth=0, limits=[20, 28, .5]), dict(depth=200, limits=[12, 24, .5])]
    gom_salinity = [dict(depth=0, limits=[34, 37, .1])]
    gom_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]

    # South Atlantic Bight Limits
    sab_extent = [-82, -64, 25, 36]
    sab_sea_water_temperature = [dict(depth=0, limits=[19, 26, .5])]
    sab_salinity = [dict(depth=0, limits=[32, 38, .1])]
    sab_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]

    # Mid Atlantic Bight Limits
    mab_extent = [-77, -68, 35, 43]
    mab_sea_water_temperature = [dict(depth=0, limits=[5, 26, .5])]
    mab_salinity = [dict(depth=0, limits=[30, 38, .1])]
    mab_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]

    # Caribbean Bight Limits
    carib_extent = [-90, -55, 6, 24]
    carib_sea_water_temperature = [dict(depth=0, limits=[22, 29, .5])]
    carib_salinity = [dict(depth=0, limits=[34, 37, .1])]
    carib_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]

    # Windward Islands
    wind_extent = [-68.2, -56.4, 9.25, 19.75]
    # 19.651243, -68.143543
    # 9.221479, -56.515488
    wind_sea_water_temperature = [dict(depth=0, limits=[25, 28, .25])]
    wind_salinity = [dict(depth=0, limits=[34.75, 37, .1])]
    wind_sea_surface_height = [dict(depth=0, limits=[-.6, .7, .1])]

    # Create new dictionary for selected model. Needs to be done because the variable names are different in each model
    # initialize empty dictionary for limits
    limits = dict()

    limits['Gulf of Mexico'] = dict()
    gom = limits['Gulf of Mexico']
    gom.update(lonlat=gom_extent)

    limits['South Atlantic Bight'] = dict()
    sab = limits['South Atlantic Bight']
    sab.update(lonlat=sab_extent)

    limits['Mid Atlantic Bight'] = dict()
    mab = limits['Mid Atlantic Bight']
    mab.update(lonlat=mab_extent)

    limits['Caribbean'] = dict()
    carib = limits['Caribbean']
    carib.update(lonlat=carib_extent)

    limits['Windward Islands'] = dict()
    wind = limits['Windward Islands']
    wind.update(lonlat=wind_extent)


    gom.update(salinity=gom_salinity)
    gom.update(temperature=gom_sea_water_temperature)
    sab.update(salinity=sab_salinity)
    sab.update(temperature=sab_sea_water_temperature)
    mab.update(salinity=mab_salinity)
    mab.update(temperature=mab_sea_water_temperature)
    carib.update(salinity=carib_salinity)
    carib.update(temperature=carib_sea_water_temperature)
    wind.update(salinity=wind_salinity)
    wind.update(temperature=wind_sea_water_temperature)

    # GOFS has sea surface height
    if model == 'gofs':
        gom.update(sea_surface_height=gom_sea_surface_height)
        sab.update(sea_surface_height=sab_sea_surface_height)
        mab.update(sea_surface_height=mab_sea_surface_height)
        carib.update(sea_surface_height=carib_sea_surface_height)
        wind.update(sea_surface_height=wind_sea_surface_height)
    return limits


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

