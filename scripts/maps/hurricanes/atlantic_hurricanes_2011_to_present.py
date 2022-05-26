
import pandas as pd
import matplotlib.pyplot as plt
from hurricanes.plotting import map_add_ticks, map_add_features
import cartopy.crs as ccrs
import xarray as xr
import datetime as dt

extent = [-80, -61, 32, 46]
y0 = 2011

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

# rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 10)))

ds = xr.open_dataset('~/Downloads/IBTrACS.NA.v04r00.nc')

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

ds = ds.where(ds.season >= y0, drop=True)

# plt.plot(lon, lat, color=colors[point["TCDVLP"]],
# transform=ccrs.PlateCarree(), zorder=20)
fig, ax = plt.subplots(
            figsize=(11, 8),
            subplot_kw=dict(projection=map_projection)
        )

for storm in ds.storm:
    # Select storm
    tds = ds.sel(storm=storm)

    # Create temporary variables
    time = pd.to_datetime(tds.time)
    lon = tds.lon.values
    lat = tds.lat.values
    name = str(tds.name.values).strip('b').strip("'").lower().title()
    cat = tds.usa_sshs

    # Plot the hurricane track
    h = ax.plot(lon, lat,
                # '-o',
                linewidth=2,
                # markersize=1,
                transform=data_projection, zorder=20,
                label=f"{name} ({time[0].strftime('%Y')})")

    # wrangle hurricane category data so that we can map it to color and size
    df = cat.to_dataframe()
    df['usa_sshs'] = df['usa_sshs'].astype(str)
    temp = df['usa_sshs'].reset_index().drop('date_time', axis=1)
    # colors = temp['usa_sshs'].map(colors_map)
    size = temp['usa_sshs'].map(size_map)*8

    # Plot hurricane markers
    ax.scatter(lon, lat,
               c=h[0].get_color(), s=size, marker='o',
               transform=data_projection, zorder=20)

# l1 = ax.legend()

# Plot title
# map_add_bathymetry(ax, bathy, data_projection)
map_add_features(ax, extent)
map_add_ticks(ax, extent)

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
y1 = dt.datetime.now().strftime("%Y")
ax.set_title(f'Mid-Atlantic Bight Hurricanes\n{y0} thru {y1}', 
             fontsize=18, fontweight='bold')
plt.savefig(f'/Users/mikesmith/Documents/all-ibtracs-path_{y0}_to_{y1}.png',
            bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()
