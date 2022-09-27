import datetime as dt
from pathlib import Path

import ioos_model_comparisons.configs as configs
import numpy as np
import pandas as pd
from ioos_model_comparisons.models import rtofs
from ioos_model_comparisons.platforms import (get_active_gliders, get_argo_floats_by_time,
                                  get_bathymetry)
from ioos_model_comparisons.plotting import plot_model_region
from ioos_model_comparisons.regions import region_config

# Get path information about this script
script_name = Path(__file__).name

# Set main path of data and plot location
path_save = (configs.path_plots / "maps")

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
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

with rtofs() as ds:
    # Save rtofs lon and lat as variables to speed up indexing calculation
    grid_lons = ds.lon.values[0,:]
    grid_lats = ds.lat.values[:,0]
    grid_x = ds.x.values
    grid_y = ds.y.values

    # Loop through regions
    for item in configs.regions:
        region = region_config(item, model=kwargs["model"])
        extent = region['extent']
        extent = np.add(extent, [-1, 1, -1, 1]).tolist()

        if configs.argo:
            kwargs["argo"] = get_argo_floats_by_time(extent, start, tomorrow)

        if configs.gliders:
            kwargs["gliders"] = get_active_gliders(extent, start, tomorrow, 
                                                   parallel=False)

        if configs.bathy:
            kwargs['bathy'] = get_bathymetry(extent)

        if region['currents']['bool']:
            kwargs['currents'] = region['currents']

        print(f'Region: {region["name"]}, Extent: {extent}')

        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp(extent[:2], grid_lons, grid_x)
        lats_ind = np.interp(extent[2:], grid_lats, grid_y)

        # Use np.floor on the first index and np.ceiling on  the second index
        # of each slice in order to widen the area of the extent slightly.
        extent = [
            np.floor(lons_ind[0]).astype(int),
            np.ceil(lons_ind[1]).astype(int),
            np.floor(lats_ind[0]).astype(int),
            np.ceil(lats_ind[1]).astype(int)
        ]

        # Use the xarray .isel selector on x/y 
        # since we know the exact indexes we want to slice
        tds = ds.isel(
            x=slice(extent[0], extent[1]),
            y=slice(extent[2], extent[3])
            ).set_coords(['u', 'v'])

        for time in date_list:
            t0 = pd.to_datetime(time - np.timedelta64(configs.search_hours, 'h'))
            t1 = pd.to_datetime(time)
            kwargs['t0'] = t0
            plot_model_region(tds.sel(time=time), region, **kwargs)
