from erddapy import ERDDAP
import datetime as dt
import pandas as pd
from requests.exceptions import HTTPError
from collections import namedtuple

Argo = namedtuple('Argo', ['name', 'lon', 'lat'])
Glider = namedtuple('Glider', ['name', 'lon', 'lat'])
time_formatter = '%Y-%m-%dT%H:%M:%SZ'


def active_argo_floats(bbox=None, time_start=None, time_end=None, floats=None):
    """

    :param lon_lims: list containing westernmost longitude and easternmost latitude
    :param lat_lims: list containing southernmost latitude and northernmost longitude
    :param time_start: time to start looking for floats
    :param time_end: time to end looking for floats
    :return:
    """

    bbox = bbox or [-100, -45, 5, 46]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))
    floats = floats or False

    constraints = {
        'time>=': str(time_start),
        'time<=': str(time_end),
    }

    if bbox:
        constraints['longitude>='] = bbox[0]
        constraints['longitude<='] = bbox[1]
        constraints['latitude>='] = bbox[2]
        constraints['latitude<='] = bbox[3]

    if floats:
        constraints['platform_number='] = floats

    variables = [
        'platform_number',
        'time',
        'pres',
        'longitude',
        'latitude',
        'temp',
        'psal',
    ]

    e = ERDDAP(
        server='IFREMER',
        protocol='tabledap',
        response='nc'
    )

    e.dataset_id = 'ArgoFloats'
    e.constraints = constraints
    e.variables = variables

    try:
        df = e.to_pandas(
            parse_dates=['time (UTC)'],
            skiprows=(1,)  # units information can be dropped.
        ).dropna()
    except HTTPError:
        df = pd.DataFrame()

    return df


def active_gliders(bbox=None, time_start=None, time_end=dt.date.today(), glider_id=None):
    bbox = bbox or [-100, -40, 18, 60]
    time_start = time_start or (time_end - dt.timedelta(days=1))
    t0 = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')
    glider_id = glider_id or None

    e = ERDDAP(server='NGDAC')

    # Grab every dataset available
    # datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

    # Search constraints
    kw = dict()
    kw['min_time'] = t0
    kw['max_time'] = t1

    if bbox:
        kw['min_lon'] = bbox[0]
        kw['max_lon'] = bbox[1]
        kw['min_lat'] = bbox[2]
        kw['max_lat'] = bbox[3]

    if glider_id:
        search = glider_id
    else:
        search = None

    search_url = e.get_search_url(search_for=search, response='csv', **kw)

    try:
        # Grab the results
        search = pd.read_csv(search_url)
    except:
        # return empty dataframe if there are no results
        return pd.DataFrame()

    # Extract the IDs
    gliders = search['Dataset ID'].values

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

    e = ERDDAP(
            server='NGDAC',
            protocol='tabledap',
            response='nc'
    )

    glider_dfs = []

    for id in gliders:
        # print('Reading ' + id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables

        # checking data frame is not empty
        try:
            df = e.to_pandas(
                index_col='time (UTC)',
                parse_dates=True,
                skiprows=(1,)  # units information can be dropped.
            ).dropna()
        except:
            continue
        df = df.reset_index()
        df['dataset_id'] = id
        df = df.set_index(['dataset_id', 'time (UTC)'])
        glider_dfs.append(df)

    try:
        ndf = pd.concat(glider_dfs)
    except ValueError:
        return pd.DataFrame()

    return ndf


def active_drifters(bbox=None, time_start=None, time_end=None):
    bbox = bbox or [-100, -40, 18, 60]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))
    t0 = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    e = ERDDAP(server='OSMC', protocol="tabledap")
    e.dataset_id = "gdp_interpolated_drifter"

    # Setting constraints
    e.constraints = {
        "time>=": t0,
        "time<=": t1,
        'longitude>=': bbox[0],
        'longitude<=': bbox[1],
        'latitude>=': bbox[2],
        'latitude<=': bbox[3],
    }

    # e.variables = [
    #     "WMO",
    #     "latitude",
    #     "longitude",
    #     "time",
    # ]

    try:
        df = e.to_pandas()
    except ValueError:
        return pd.DataFrame()

    return df


def get_ndbc(bbox=None, time_start=None, time_end=None, buoy=None):
    bbox = bbox or [-100, -45, 5, 46]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))
    buoy = buoy or False
    time_formatter = '%Y-%m-%dT%H:%M:%SZ'

    e = ERDDAP(
        server='CSWC',
        protocol='tabledap',
        response='csv'
    )

    e.dataset_id = 'cwwcNDBCMet'
    e.constraints = {
        'time>=': time_start.strftime(time_formatter),
        'time<=': time_end.strftime(time_formatter),
    }

    if bbox:
        e.constraints['longitude>='] = bbox[0]
        e.constraints['longitude<='] = bbox[1]
        e.constraints['latitude>='] = bbox[2]
        e.constraints['latitude<='] = bbox[3]

    e.variables = [
        "station",
        "latitude",
        "longitude",
        "time"
    ]

    if buoy:
        e.constraints['station='] = buoy

    df = e.to_pandas(
        parse_dates=['time (UTC)'],
        skiprows=(1,)  # units information can be dropped.
    ).dropna()

    stations = df.station.unique()

    # e.variables = [
    #     "station",
    #     "latitude",
    #     "longitude",
    #     "wd",
    #     "wspd",
    #     "gst",
    #     "wvht",
    #     "dpd",
    #     "apd",
    #     "mwd",
    #     "bar",
    #     "atmp",
    #     "wtmp",
    #     "dewp",
    #     # "vis",
    #     # "ptdy",
    #     # "tide",
    #     "wspu",
    #     "wspv",
    #     "time",
    # ]

    try:
        df = e.to_pandas(
            parse_dates=['time (UTC)'],
            skiprows=(1,)  # units information can be dropped.
        ).dropna()
    except HTTPError:
        df = pd.DataFrame()

    return df