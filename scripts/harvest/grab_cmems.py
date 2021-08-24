#! /usr/bin/env python3

"""
Author: aristizabal
Last modified: Lori Garzio on 2/22/2021
"""
import datetime as dt
import os
import xarray as xr


def download_ds(out_dir, out_name, st, et, coordlims, depth_max, user, pwd):
    url = 'http://nrt.cmems-du.eu/motu-web/Motu'
    service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
    product_id = 'global-analysis-forecast-phy-001-024'

    os.makedirs(out_dir, exist_ok=True)

    motuc = 'python -m motuclient --motu ' + url + \
            ' --service-id ' + service_id + \
            ' --product-id ' + product_id + \
            ' --longitude-min ' + str(coordlims[0] - 1/6) + \
            ' --longitude-max ' + str(coordlims[1] + 1/6) + \
            ' --latitude-min ' + str(coordlims[2] - 1/6) + \
            ' --latitude-max ' + str(coordlims[3] + 1/6) + \
            ' --date-min ' + str(st - dt.timedelta(0.5)) + '"' + \
            ' --date-max ' + str(et + dt.timedelta(0.5)) + '"' + \
            ' --depth-min ' + '0.493' + \
            ' --depth-max ' + str(depth_max) + \
            ' --variable ' + 'thetao' + ' ' + \
            ' --variable ' + 'so' + ' ' + \
            ' --out-dir ' + out_dir + \
            ' --out-name ' + out_name + ' ' + \
            ' --user ' + user + ' ' + \
            ' --pwd ' + pwd

    os.system(motuc)
    print('\nCMEMS file downloaded to: {}'.format(os.path.join(out_dir, out_name)))
    return os.path.join(out_dir, out_name)


def copernicusmarine_datastore(dataset, username, password):
    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    return xr.open_dataset(data_store)
