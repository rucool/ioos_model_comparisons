import datetime as dt
from pathlib import Path

import hurricanes.configs as configs
import numpy as np
import pandas as pd
from hurricanes.calc import lon360to180, lon180to360
from hurricanes.models import gofs
from hurricanes.platforms import (get_active_gliders, get_argo_floats_by_time,
                                  get_bathymetry)
from hurricanes.plotting import plot_model_region
from hurricanes.regions import region_config

# Get path information about this script
script_name = Path(__file__).name

# Set main path of data and plot location
path_save = (configs.path_plots / "maps")

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'gofs'
kwargs['transform'] = configs.projection
kwargs['path_save'] = path_save
kwargs['dpi'] = configs.dpi

# Get today and yesterday dates
today = dt.date.today()
tomorrow = today + dt.timedelta(days=1)
past = today - dt.timedelta(days=configs.days)

# Create dates that we want to plot
date_list = pd.date_range(past, tomorrow, freq='6H')
start = date_list[0] - dt.timedelta(hours=configs.search_hours)

with gofs(rename=True) as ds:
    # Get all the datetimes (match with rtofs dates of every 6 hours)
    tds = ds.sel(
        time=date_list[date_list < pd.to_datetime(ds.time.max().data)]
        )

    # Loop through regions
    for item in configs.regions:
        region = region_config(item, model=kwargs["model"])
        extent = region['extent']
        extent = np.add(extent, [-1, 1, -1, 1]).tolist()
        
        print(f'Region: {region["name"]}, Extent: {extent}')

        if configs.argo:
            kwargs['argo'] = get_argo_floats_by_time(extent, start, tomorrow)

        if configs.gliders:
            kwargs['gliders'] = get_active_gliders(extent, start, tomorrow, 
                                                   parallel=False)

        if configs.bathy:
            kwargs['bathy'] = get_bathymetry(extent)
        
        if region['currents']['bool']:
                kwargs['currents'] = region['currents']

        # subset dataset to the proper extents for each region
        lon360 = lon180to360(extent[:2]) # convert from 360 to 180 lon
        sub = tds.sel(
            lon=slice(lon360[0], lon360[1]),
            lat=slice(extent[2], extent[3])
            ).set_coords(['u', 'v'])
    
        # Convert from 0,360 lon to -180,180
        sub['lon'] = lon360to180(sub['lon'])
        
        for time in date_list:
            t0 = pd.to_datetime(time - np.timedelta64(configs.search_hours, 'h'))
            t1 = pd.to_datetime(time)
            kwargs['t0'] = t0
            try:
                plot_model_region(sub.sel(time=time), region, **kwargs)
            except KeyError as e:
                print(e)
                continue
