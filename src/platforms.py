from erddapy import ERDDAP
import datetime as dt
import pandas as pd
from requests.exceptions import HTTPError


def active_argo_floats(bbox= None, time_start=None, time_end=None):
    """

    :param lon_lims: list containing westernmost longitude and easternmost latitude
    :param lat_lims: list containing southernmost latitude and northernmost longitude
    :param time_start: time to start looking for floats
    :param time_end: time to end looking for floats
    :return:
    """

    url_Argo = 'http://www.ifremer.fr/erddap'

    bbox = bbox or [-100, -80, 18, 32]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))

    constraints = {
        'time>=': str(time_start),
        'time<=': str(time_end),
        'longitude>=': bbox[0],
        'longitude<=': bbox[1],
        'latitude>=': bbox[2],
        'latitude<=': bbox[3],
    }

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
        server=url_Argo,
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


def active_gliders(bbox=None, time_start=None, time_end=None):
    url = 'https://data.ioos.us/gliders/erddap'

    bbox = bbox or [-100, -40, 18, 60]
    time_end = time_end or dt.date.today()
    time_start = time_start or (time_end - dt.timedelta(days=1))
    t0 = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1 = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    e = ERDDAP(server=url)

    # Grab every dataset available
    # datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

    # Search constraints
    kw = {
        'min_lon': bbox[0],
        'max_lon': bbox[1],
        'min_lat': bbox[2],
        'max_lat': bbox[3],
        'min_time': t0,
        'max_time': t1,
    }

    search_url = e.get_search_url(response='csv', **kw)

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
            server=url,
            protocol='tabledap',
            response='nc'
    )

    glider_dfs = []

    for id in gliders:
        print('Reading ' + id)
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

