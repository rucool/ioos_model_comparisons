import xarray as xr
import os
from ioos_model_comparisons.limits import limits_regions
import datetime as dt
import numpy as np
from ioos_model_comparisons.platforms import active_argo_floats, active_gliders
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader


url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '/Users/mikesmith/Documents/'

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
eez = '/Users/mikesmith/Documents/github/rucool/Daily_glider_models_comparisons/World_EEZ_v11_20191118/eez_boundaries_v11.shp'
days = 365
dpi = 150

regions = limits_regions('rtofs', ['gom'])

now = dt.date.today()
t0 = pd.to_datetime(now - pd.Timedelta(days, 'd'))
t1 = pd.to_datetime(now)

LAND = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m',
    edgecolor='face',
    facecolor='tan'
)

state_lines = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none'
)

def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"

bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
bathy = xr.open_dataset(bathymetry)

# Loop through regions
for region in regions.items():
    extent = region[1]['lonlat']
    print(f'Region: {region[0]}, Extent: {extent}')

    temp_save_dir = os.path.join(save_dir, '_'.join(region[0].split(' ')).lower())
    os.makedirs(temp_save_dir, exist_ok=True)

    floats = active_argo_floats(extent, t0, t1)
    grouped = floats.groupby(['time (UTC)', 'platform_number', ])
    argo_df = grouped.mean()

    gliders = active_gliders(extent, t0, t1)

    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )
    # Plot title
    plt.title(f'Argo Floats\n Active: {t0.strftime("%Y-%m-%d")} to {t1.strftime("%Y-%m-%d")}')
    # Axes properties and features
    ax.set_extent(extent)
    ax.add_feature(LAND, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_lines, zorder=11, edgecolor='black')
    shape_feature = cfeature.ShapelyFeature(Reader(eez).geometries(),
                                            ccrs.PlateCarree(), edgecolor='grey', facecolor='none')
    ax.add_feature(shape_feature, zorder=1)

    levels = np.arange(-200, 0, 200)
    levels = np.insert(levels, 0, [-1500, -1000])
    bath_lat = bathy.variables['lat'][:]
    bath_lon = bathy.variables['lon'][:]
    bath_elev = bathy.variables['elevation'][:]

    CS = ax.contour(bath_lon, bath_lat, bath_elev, levels, linewidths=.75, alpha=.5, colors='k',
                     transform=ccrs.PlateCarree())
    ax.clabel(CS, [-200, -1000, -1500], inline=True, fontsize=6, fmt=fmt, manual=True)

    for index, row in argo_df.iterrows():
        ax.plot(row['longitude (degrees_east)'], row['latitude (degrees_north)'], markersize=4,
                marker='o', color='k', transform=ccrs.PlateCarree())

    if not gliders.empty:
        for g, new_df in gliders.groupby(level=0):
            q = new_df.iloc[-1]
            ax.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'], color='red',
                     linewidth=1.5, transform=ccrs.PlateCarree())
            ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker='^',
                     markeredgecolor='black',
                     markersize=8.5, label=g, transform=ccrs.PlateCarree())
            # ax.legend(loc='upper right', fontsize=6)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,)

    # Gridlines and grid labels
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=.5,
        color='black',
        alpha=0.25,
        linestyle='--'
    )

    gl.xlabels_top = gl.ylabels_right = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    save_file = os.path.join(save_dir, f'argo_floats-{"_".join(region[0].split(" ")).lower()}-{t0.strftime("%Y-%m-%d")}-to-{t1.strftime("%Y-%m-%d")}')
    plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
