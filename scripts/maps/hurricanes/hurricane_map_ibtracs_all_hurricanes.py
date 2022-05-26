# import os
# from glob import glob
# from unicodedata import name
# import geopandas
import pandas as pd
# import lxml.html
# import sys
# import warnings
# from dateutil import parser
# from pytz import timezone
import matplotlib.pyplot as plt
# from hurricanes.plotting import map_add_ticks, map_add_features
import cartopy.crs as ccrs
# from hurricanes.platforms import active_gliders
import xarray as xr
# from itertools import cycle

extent = [-80, -63, 32, 45]

storms = dict(
    irene=2101,
    sandy=2130,
    isaias=2248,
    ida=2281,
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

# rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 10)))

ds = xr.open_dataset('~/Downloads/IBTrACS.NA.v04r00.nc')

# plt.plot(lon, lat, color=colors[point["TCDVLP"]], transform=ccrs.PlateCarree(), zorder=20)
fig, ax = plt.subplots(
            figsize=(11, 8),
            subplot_kw=dict(projection=map_projection)
        )
# map_add_features(ax, extent)


for storm, number in storms.items():
    time = ds.time[number, :]
    dt = pd.to_datetime(time.data[0])
    name = ds.name[number]
    lon = ds.lon[number, :]
    lat = ds.lat[number, :]
    cat = ds.usa_sshs[number, :]

    # Plot the hurricane track
    ax.plot(lon, lat, linewidth=2, transform=data_projection, zorder=20,
            label=f"{storm.title()} ({dt.strftime('%Y')})")

    # wrangle hurricane category data so that we can map it to color and size
    df = cat.to_dataframe()
    df['usa_sshs'] = df['usa_sshs'].astype(str)
    temp = df['usa_sshs'].reset_index().drop('date_time', axis=1)
    colors = temp['usa_sshs'].map(colors_map)
    size = temp['usa_sshs'].map(size_map)*8

    # Plot hurricane markers
    ax.scatter(lon, lat, 
               c=colors, s=size, marker='o',
               transform=data_projection, zorder=20)

l1 = ax.legend()

# Plot title
# map_add_bathymetry(ax, bathy, data_projection)
# map_add_ticks(ax, extent)

# Path.clip_to_bbox(extent)
ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

# from matplotlib.lines import Line2D

# custom_lines = [
#     Line2D([0], [0], marker='o', color='w', label='Tropical Depression/Storm', markerfacecolor='cyan', markersize=15),
#     Line2D([0], [0], marker='o', color='w', label='Category 1', markerfacecolor='yellow', markersize=15),
#     Line2D([0], [0], marker='o', color='w', label='Category 2', markerfacecolor='gold', markersize=15),
#     Line2D([0], [0], marker='o', color='w', label='Category 3', markerfacecolor='orange', markersize=15),
#     Line2D([0], [0], marker='o', color='w', label='Category 4', markerfacecolor='darkorange', markersize=15),
#     Line2D([0], [0], marker='o', color='w', label='Category 5', markerfacecolor='red', markersize=15),
# ]

# fig, ax = plt.subplots()
# ax.legend(handles=custom_lines, loc='lower right', scatterpoints=1)
# plt.gca().add_artist(l1)

ax.set_title('Mid-Atlantic Bight Hurricanes\n', fontsize=18, fontweight='bold')
plt.savefig('/Users/mikesmith/Documents/all-ibtracs-path.png',
            bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
plt.close()
