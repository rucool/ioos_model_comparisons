
import pandas as pd
import matplotlib.pyplot as plt
# from ioos_model_comparisons.plotting import map_add_ticks, map_add_features
import cartopy.crs as ccrs
import xarray as xr
import datetime as dt
import numpy as np
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
import cool_maps.plot as cplt
from shapely.geometry.polygon import LinearRing

# extent = [-80, -61, 32, 46] # GoM


y0 = 2000
y1 = 2023

# Caribbean
# extent = [-98.5, -40, 5, 33] 
# lons = [-89, -89, -60, -60]
# lats = [23.25, 8, 8, 23.25]
# ds = xr.open_dataset('~/Downloads/IBTrACS.NA.v04r00.nc')
# title_str = f'Caribbean - {y0} to {y1}'
# save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-caribbean-transect.png'
# types = 'Hurricanes'

# Bay of Bengal
# extent = [65, 105, 0, 30]
# lons = [77, 77, 99, 99]
# lats = [23, 0, 0, 23]
# ds = xr.open_dataset('~/Downloads/IBTrACS.NI.v04r00.nc')
# title_str = f'Bay of Bengal - {y0} to {y1}'
# save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-bengal-transect.png'
# types = 'Tropical Cyclones'

# # NPOMS
# extent = [95, 160, 0, 55]
# lons = [100, 100, 150, 150]
# lats = [45, 0, 0, 45]
# title_str = f'NPOMS Region - {y0} to {y1}'
# save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-npoms-transect.png'
# types = 'Typhoons'
# ds = xr.open_dataset('~/Downloads/IBTrACS.WP.v04r00.nc')

# # [-98, -81, 18, 30.5]
# lons = [-98, -98, -81, -81]
# lats = [30.5, 18, 18, 30.5]

# # Africa
# extent = [10, 120, -45, -0]
# lons = [20, 20, 110, 110]
# lats = [-40, -5, -5, -40]
# title_str = f'Indian Ocean - {y0} to {y1}'
# save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-indian_ocean-transect.png'
# types = 'Tropical Cyclones'
# ds = xr.open_dataset('~/Downloads/IBTrACS.SI.v04r00.nc')

# freddy = pd.read_csv('/Users/mikesmith/Documents/freddy_track.csv', )
# freddy['time'] = pd.to_datetime(freddy['time'])
# freddy.set_index('time', inplace=True)
# freddy = freddy.resample('6H').ffill()

# West Coast - Version 1
extent = [-180, -90, 5, 35]
lons = [-179.9, -179.9, -90, -90]
lats = [35, 5, 5, 35]
title_str = f'Eastern Pacific Ocean - {y0} to {y1}'
save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-eastern_pacific_ocean-transect.png'
types = 'Hurricanes'
ds = xr.open_dataset('~/Downloads/IBTrACS.EP.v04r00.nc')
# # ds = xr.open_dataset('~/Downloads/IBTrACS.ALL.v04r00.nc')

# West Coast - Version 2
# extent = [-180, -90, 5, 35]
# lons = [-179.9, -179.9, -150, -150]
# lats = [30, 15, 15, 30]
# title_str = f'Eastern Pacific Ocean - {y0} to {y1}'
# save_str = f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}-eastern_pacific_ocean-transect.png'
# types = 'Hurricanes'
# ds = xr.open_dataset('~/Downloads/IBTrACS.EP.v04r00.nc')
# ds = xr.open_dataset('~/Downloads/IBTrACS.ALL.v04r00.nc')

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

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

colors_map = {
    "nan": 'black',
    "-5.0": 'black',
    "-4.0": 'black',
    "-3.0": 'black',
    "-2.0": 'black',
    "-1.0": 'black',
    "0.0": 'black',
    "1.0": 'yellow',
    "2.0": 'gold',
    "3.0": 'orange',
    "4.0": 'darkorange',
    "5.0": 'red',
}
# 'yellow', 'gold', 'orange', 'darkorange', 'red'

ds = ds.where(ds.season >= y0, drop=True)

levels = [-8000, -1000, -100, 0]  # Contour levels (depths)
colors = ['cornflowerblue', cfeature.COLORS['water'],
          'lightsteelblue',]  # contour colors

# Get bathymetry from srtm15
# bs = cplt.get_bathymetry(extent)
bs = xr.open_dataset('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc')
bs = bs.sel(lon=slice(-180, -90), lat=slice(5, 35))
bs = bs.rename({'lon': 'longitude', 'lat': 'latitude'})

# Create map using cool_maps
fig, ax = cplt.create(extent, figsize=(16, 9), gridlines=True, labelsize=16)


ring = LinearRing(list(zip(lons, lats)))
ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none',
                  edgecolor='red', zorder=1000, linewidth=5)

# add filled contour to map
cs = ax.contourf(bs['longitude'], bs['latitude'], bs['z'], levels,
                 colors=colors, transform=ccrs.PlateCarree(), ticks=False)

# Create legend for contours
proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
         for pc in cs.collections]
proxy.reverse()
plt.legend(proxy, ["0-100m", "100-1000m", "1000+m"],
           loc='upper right').set_zorder(10000)

n = 0
for storm in ds.storm:
    # Select storm
    tds = ds.sel(storm=storm)

    # Create temporary variables
    time = pd.to_datetime(tds.time)
    lon = tds.lon.values
    lat = tds.lat.values
    name = str(tds.name.values).strip('b').strip("'").lower().title()
    cat = tds.usa_sshs
    if cat[cat >= 1].any():
        # wrangle hurricane category data so that we can map it to color and size
        df = cat.to_dataframe()
        df['usa_sshs'] = df['usa_sshs'].astype(str)
        temp = df['usa_sshs'].reset_index().drop('date_time', axis=1)
        colors = temp['usa_sshs'].map(colors_map)
        size = temp['usa_sshs'].map(size_map)*8

        # [-99, -79, 18, 31]
        # Gulf of Mexico
        # lat_check = np.logical_and(lat > 18, lat < 30.5)
        # lon_check = np.logical_and(lon < -81, lon > -98)

        # Caribbean
        lat_check = np.logical_and(lat > extent[2], lat < extent[3])
        lon_check = np.logical_and(lon < extent[1], lon > extent[0])

        # Bay of Bengal
        # lat_check = np.logical_and(lat < lats[-2], lat > lats[-1])
        # lon_check = np.logical_and(lon < lons[-2], lon > lons[-3])

        # Hawaii
        # lat_check = np.logical_and(lat > 15, lat < 30)
        # lon_check = np.logical_and(lon < -150, lon > -180)

        if np.logical_and(lon_check, lat_check).any():
            n = n + 1

        # if np.logical_and(lat > 20, lon > -60).any():
            # color='red'
        # else:
            color = 'black'

            # Plot the hurricane track
            h = ax.plot(lon, lat,
                        color=color,
                        # color=colors[cat.argmax().values],
                        # '-o',
                        linewidth=2,
                        # markersize=1,
                        transform=data_projection, zorder=20,
                        label=f"{name} ({time[0].strftime('%Y')})")

            # # Plot hurricane markers
            # ax.scatter(lon, lat,
            #         c=colors, s=1, marker='o',
            #         transform=data_projection, zorder=20)

# l1 = ax.legend()
# h1 = ax.plot(freddy['lon'], freddy['lat'],
#             color='red',
#             # color=colors[cat.argmax().values],
#             # '-o',
#             linewidth=2,
#             # markersize=1,
#             transform=data_projection, zorder=20,
#             )

# Plot title
# map_add_bathymetry(ax, bathy, data_projection)
# map_add_features(ax, extent)
# map_add_ticks(ax, extent)

# Path.clip_to_bbox(extent)
# ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
# ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')

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
ax.set_title(f"{title_str} - {n} {types}", fontsize=18, fontweight='bold')
plt.savefig(save_str,
            bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()
