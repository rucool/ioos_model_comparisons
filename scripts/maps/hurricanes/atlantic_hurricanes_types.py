
import pandas as pd
import matplotlib.pyplot as plt
from hurricanes.plotting import map_add_ticks, map_add_features, map_add_bathymetry
import cartopy.crs as ccrs
import xarray as xr
import datetime as dt
import numpy as np


extent = [-80, -61, 32, 46]
y0 = 2000

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

# rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, 10)))

ds = xr.open_dataset('~/Documents/data/ibtracs/IBTrACS.NA.v04r00.nc')

hurricanes = dict(
    offshore=dict(
        henri=b'2021226N38295',
        hermine_2016=b'2016242N24279',
        sandy=b'2012296N14283',
        beryl=b'2006200N32287',
        twenty_two=b'2005281N26303',
        hermine_2004=b'2004241N29295',
    ),
    inland=dict(
        ida=b'2021236N12296',
        claudette=b'2021166N20265',
        zeta=b'2020299N18277',
        nestor=b'2019291N22264',
        michael=b'2018280N18273',
        cindy=b'2017171N24271',
        jeanne=b'2004258N16300',
        ivan=b'2004247N10332',
        helene=b'2000260N15308',
        gordon=b'2000259N20273',
    ),
    coastal=dict(
        elsa=b'2021182N09317',
        kyle_2020=b'2020228N37286',
        isaias=b'2020211N13306',
        fay=b'2020188N28271',
        ana=b'2015126N27281',
        arthur=b'2014180N32282',
        andrea=b'2013157N25273',
        irene=b'2011233N15301',
        gabrielle=b'2007251N30288',
        barry=b'2007151N18273',
        ernesto=b'2006237N13298',
        gaston=b'2004241N32282',
        charley=b'2004223N11301',
        bonnie=b'2004217N13306',
        kyle_2002=b'2002264N28308',
        allison=b'2001157N28265',
    )
)

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

bathymetry = '/Users/mikesmith/Documents/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
bathy = xr.open_dataset(bathymetry)
bathy = bathy.sel(
    lon=slice(extent[0] - 1, extent[1] + 1),
    lat=slice(extent[2] - 1, extent[3] + 1)
)

for type in hurricanes.keys():
    # plt.plot(lon, lat, color=colors[point["TCDVLP"]],
    # transform=ccrs.PlateCarree(), zorder=20)
    fig, ax = plt.subplots(
                figsize=(11, 8),
                subplot_kw=dict(projection=map_projection)
            )
    i = 0
    for storm, id in hurricanes[type].items():
        storm_loc = (ds.sid == id).argmax().item()
        tds = ds.sel(storm=storm_loc)

        # Create temporary variables
        time = pd.to_datetime(tds.time)
        lon = tds.lon.values
        lat = tds.lat.values
        name = str(tds.name.values).strip('b').strip("'").lower().title()
        cat = tds.usa_sshs

        # Plot the hurricane track
        if i < 10:
            marker = 'o-'
        elif (i >= 10) & (i < 20):
            marker = '^-'
        elif (i >= 20) & (i < 30):
            marker = 's-'
         
        h = ax.plot(lon, lat,
                    marker,
                    linewidth=2,
                    # markersize=1,
                    transform=data_projection, zorder=20,
                    label=f"{name.title()} ({time[0].strftime('%Y')})")
        i = i + 1

        # # wrangle hurricane category data
        # df = cat.to_dataframe()
        # df['usa_sshs'] = df['usa_sshs'].astype(str)
        # temp = df['usa_sshs'].reset_index().drop('date_time', axis=1)
        # # colors = temp['usa_sshs'].map(colors_map)
        # size = temp['usa_sshs'].map(size_map)*8

        # # Plot hurricane markers
        # ax.scatter(lon, lat,
        #            c=h[0].get_color(), s=size, marker='o',
        #            transform=data_projection, zorder=20)
    l1 = ax.legend().set_zorder(100)
    # # Plot title
    map_add_bathymetry(ax, bathy, data_projection, np.array([-200]))
    map_add_features(ax, extent)
    map_add_ticks(ax, extent)
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
    #     ] 
    # fig, ax = plt.subplots()
    # ax.legend(handles=custom_lines, loc='lower right', scatterpoints=1)
    # plt.gca().add_artist(l1)

    y1 = dt.datetime.now().strftime("%Y")
    ax.set_title(f'Mid-Atlantic Bight\n{type.title()} Hurricanes\n{y0} thru 2021',
                 fontsize=18, fontweight='bold')
    plt.savefig(
        f'/Users/mikesmith/Documents/all-ibtracs-path_{type}_{y0}_to_2021.png',
        bbox_inches='tight', pad_inches=0.1, dpi=300
        )
    plt.close()


