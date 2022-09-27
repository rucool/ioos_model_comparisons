import os
from glob import glob
from unicodedata import name
import geopandas
import pandas as pd
import lxml.html
import sys
import warnings
from dateutil import parser
from pytz import timezone
import matplotlib.pyplot as plt
from ioos_model_comparisons.plotting import map_add_ticks, map_add_features, map_add_gliders, map_add_legend
import cartopy.crs as ccrs
from ioos_model_comparisons.platforms import active_gliders
import xarray as xr

extent = [-80, -63, 32, 45]

storms = dict(
    # ida=2281, 
    # sandy=2130,
    # irene=2101,
    isaias=2248
    )

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

colors_map = {
    "nan": "white",
    "-5.0": "cyan",
    "-4.0":  "cyan",
    "-3.0":  "cyan",
    "-2.0":  "cyan",
    "-1.0":  "cyan",
    "0.0": "cyan",
    "1.0": "yellow",
    "2.0": "gold",
    "3.0": "navajowhite",
    "4.0": "darkorange",
    "5.0": "red",
}

size_map = {
    "nan": 1,
    "-5.0": 5,
    "-4.0": 5,
    "-3.0": 5,
    "-2.0": 5,
    "-1.0": 5,
    "0.0": 5,
    "1.0": 6,
    "2.0": 7,
    "3.0": 8,
    "4.0": 9,
    "5.0": 10,
}

ds = xr.open_dataset('IBTrACS.NA.v04r00.nc')

for storm, number in storms.items():
    time = ds.time[number,:]
    dt = pd.to_datetime(time.data[0])
    # name = ds.name[number]
    # ida_wind_ib = ds.usa_wind[ida_storm_number,:]
    # ida_press_ib = ds.usa_pres[ida_storm_number,:]
    # ida_rmw_ib = ds.usa_rmw[ida_storm_number,:]
    lon = ds.lon[number,:]
    lat = ds.lat[number,:]
    cat = ds.usa_sshs[number,:]

    fig, ax = plt.subplots(
            figsize=(11, 8),
            subplot_kw=dict(projection=map_projection)
        )
        
    map_add_features(ax, extent)

    # Plot the hurricane track
    track = ax.plot(lon, lat, 'r-', linewidth=7, transform=data_projection, zorder=20)
    # ax.text(lon, lat, time,  fontsize=10, fontweight='bold', transform=ccrs.PlateCarree(), zorder=20)
    # ax.text(t[1].lon-2, t[1].lat-.05, t[1].time_convert.strftime('%Y-%m-%dT%H:%M:%SZ'), fontsize=10, fontweight='bold', transform=projection, zorder=20)

    # wrangle hurricane category data so that we can map it to color and size
    df = cat.to_dataframe()
    df['usa_sshs'] = df['usa_sshs'].astype(str)
    temp = df['usa_sshs'].reset_index().drop('date_time', axis=1)
    colors = temp['usa_sshs'].map(colors_map)
    size = temp['usa_sshs'].map(size_map)*25

    # Plot hurricane markers
    markers = ax.scatter(lon, lat, c=colors, s=size, transform=data_projection, marker='o', zorder=20)

    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], marker='o', linestyle='None', label='Tropical Storm', color='cyan', markersize=12),
        Line2D([0], [0], marker='o', linestyle='None', label='Category 1', color='yellow', markersize=12),
        Line2D([0], [0], marker='o', linestyle='None', label='Category 2', color='gold', markersize=12),
        Line2D([0], [0], marker='o', linestyle='None', label='Category 3', color='orange', markersize=12),
        Line2D([0], [0], marker='o', linestyle='None', label='Category 4', color='darkorange', markersize=12),
        Line2D([0], [0], marker='o', linestyle='None', label='Category 5', color='red', markersize=12),
    ]

    ax.legend(handles=custom_lines, loc='lower right', scatterpoints=1)

    # Plot title
    # map_add_bathymetry(ax, bathy, data_projection)
    map_add_ticks(ax, extent)

    # Path.clip_to_bbox(extent)
    ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    ax.set_title(f'Hurricane {storm.title()} ({dt.strftime("%Y")})\n', fontsize=18, fontweight='bold')
    plt.savefig(f'/Users/mikesmith/Documents/{storm}-ibtracs-times-path.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()