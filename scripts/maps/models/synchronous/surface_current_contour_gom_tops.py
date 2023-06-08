import datetime as dt

import ioos_model_comparisons.configs as conf
import numpy as np
import pandas as pd
from ioos_model_comparisons.platforms import (get_active_gliders, 
                                  get_argo_floats_by_time,
                                  get_bathymetry)
from ioos_model_comparisons.plotting_ugos import surface_current_fronts_single

from ioos_model_comparisons.regions import region_config
import matplotlib
import time
from pathlib import Path
import xarray as xr

# Directory where the file should be downloaded
# ddir = Path('/Users/mikesmith/data/tops/')
ddir = Path('/home/hurricaneadm/data/tops/')

startTime = time.time() # Start time to see how long the script took
matplotlib.use('agg')

# Set path to save plots
path_save = (conf.path_plots / "maps")

# initialize keyword arguments for map plots
kwargs = dict()
kwargs['transform'] = conf.projection
kwargs['dpi'] = conf.dpi
kwargs['overwrite'] = True

# Get yesterday dates
now = dt.datetime.now()

# Subtract one day from the current date to get yesterday's date
yesterday = now - dt.timedelta(days=1)
tomorrow = now + dt.timedelta(days=1)

for d in [yesterday, now, tomorrow]:
    
    # Format the date
    formatted_date_1 = yesterday.strftime('%Y%m%d')
    formatted_date_2 = d.strftime('%Y%m%d')

    # Formatter for time
    tstr = '%Y-%m-%d %H:%M:%S'

    # This is the initial time to start the search for argo/gliders
    search_start = d - dt.timedelta(hours=conf.search_hours)

    region = region_config('gom') #gom, loop_current, yucatan
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')
    kwargs['path_save'] = path_save / region['folder']
    if conf.argo:
        argo_data = get_argo_floats_by_time(extent, search_start, d)
    else:
        argo_data = pd.DataFrame()

    if conf.gliders:
        glider_data = get_active_gliders(extent, search_start, d, parallel=False)
    else:
        glider_data = pd.DataFrame()

    if conf.bathy:
        bathy_data = get_bathymetry(extent)

    # The name we want to give to the downloaded file
    file_name = f'tops_compositem_{formatted_date_1}_{formatted_date_2}.nc'

    fname = ddir / yesterday.strftime('%Y/%m') / file_name

    # Load TOPS
    tops = xr.open_dataset(fname).rename({'uvel': 'u', 'vvel': 'v'})
    tops.attrs['model'] = 'TOPS'

    # Deal with time related variables
    ctime = yesterday
    search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
    search_window_t1 = ctime.strftime(tstr) 

    try:
        topst = tops.sel(depth=0)
        print(f"TOPS: True")
        tops_flag = True
    except KeyError as error:
        print(f"TOPS: False - {error}")
        tops_flag = False
        
    print("\n")

    if 'eez' in region:
        kwargs["eez"] = region["eez"]

    if 'figure' in region:
        if 'legend' in region['figure']:
            kwargs['cols'] = region['figure']['legend']['columns']

        if 'figsize' in region['figure']:
            kwargs['figsize'] = region['figure']['figsize']

    try:
        kwargs['bathy'] = bathy_data.sel(
            longitude=slice(extent[0] - 1, extent[1] + 1),
            latitude=slice(extent[2] - 1, extent[3] + 1)
        )
    except NameError:
        pass
            
    extended = np.add(extent, [-1, 1, -1, 1]).tolist()

    tops_sub = topst.sel(
        lon=slice(extended[0], extended[1]),
        lat=slice(extended[2], extended[3])
    ).set_coords(['u', 'v'])

    # Check if any asset data exists and subset to appropriate region and time
    # Was any argo data downloaded?
    if not argo_data.empty:
        argo_lon = argo_data['lon']
        argo_lat = argo_data['lat']
        argo_region = argo_data[
            (extended[0] <= argo_lon) & (argo_lon <= extended[1]) & (extended[2] <= argo_lat) & (argo_lat <= extended[3])
        ]
        argo_region.sort_index(inplace=True)
        idx = pd.IndexSlice
        kwargs['argo'] = argo_region.loc[idx[:, search_window_t0:search_window_t1], :]

    # Was any glider data downloaded?
    if not glider_data.empty:
        glider_lon = glider_data['lon']
        glider_lat = glider_data['lat']
        glider_region = glider_data[
            (extended[0] <= glider_lon) & (glider_lon <= extended[1]) & (extended[2] <= glider_lat) & (glider_lat <= extended[3])
            ]
        glider_region = glider_region[
            (search_window_t0 <= glider_region.index.get_level_values('time'))
            &
            (glider_region.index.get_level_values('time') <= search_window_t1)
            ]
        kwargs['gliders'] = glider_region

    try:
        surface_current_fronts_single(tops_sub.squeeze(), region, **kwargs)
    except Exception as e:
        print(f"Failed to process TOPS at {ctime}")
        print(f"Error: {e}")