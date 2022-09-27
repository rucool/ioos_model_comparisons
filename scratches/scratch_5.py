#!/usr/bin/env python

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import os
from ioos_model_comparisons.limits import limits_regions
import datetime as dt
import numpy as np
from ioos_model_comparisons.platforms import active_gliders, active_argo_floats
import pandas as pd
from glob import glob
import geopandas
import lxml.html
import sys
import warnings
from dateutil import parser
from pytz import timezone
from ioos_model_comparisons.plotting import map_add_ticks, cmaps, map_add_features, map_add_bathymetry
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

pdf = pd.read_csv('/Users/mikesmith/Documents/whoi_argo_complete_profiles.csv', parse_dates=['time'])
pdf['date'] = pd.to_datetime(pdf['time'].dt.strftime('%Y-%m-%d'))

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

try:
    from urllib.request import urlopen, urlretrieve
except Exception:
    from urllib import urlopen, urlretrieve

glider = 'ng645-20210613T0000'
save_dir = '/Users/mikesmith/Documents/'
t0 = dt.datetime(2021, 8, 27, 0, 0)
t1 =  dt.datetime(2021, 8, 31, 0, 0)
model = 'rtofs'

url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps/'
bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

days = 2
projection = dict(map=ccrs.Mercator(), data=ccrs.PlateCarree())
argo = True
gliders = True
dpi = 150
search_hours = 24*10  #Hours back from timestamp to search for drifters/gliders=
#
regions = limits_regions('rtofs', ['gom', 'nola'])
#
# # initialize keyword arguments for glider functions
# gargs = dict()
# gargs['time_start'] = t0  # False
# gargs['time_end'] =t1
# gargs['filetype'] = 'dataframe'
#
# glider_df = glider_dataset(glider, **gargs)
#
# grouped = glider_df.groupby('time')
# fig, ax = plt.subplots(figsize=(12, 6))
#
# surface_temp = []
# times = []
# for name, group in grouped:
#     group.sort_values('depth', inplace=True)
#     surface_temp.append(group.iloc[0].temperature)
#     times.append(name)


# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['transform'] = projection
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# f = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/rtofs.20210829/rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc'
files = [
    '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/rtofs.20210827/rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',
    '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/rtofs.20210829/rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc'
]

for f in files:
    with xr.open_dataset(f) as ds:
        ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
        lat = ds.lat.data
        lon = ds.lon.data

        t0 = pd.to_datetime(ds.time.data[0] - np.timedelta64(search_hours, 'h'))
        t0_glider = pd.to_datetime(ds.time.data[0] - np.timedelta64(10000, 'h'))
        t1 = pd.to_datetime(ds.time.data[0])
        kwargs['t0'] = t0

        # Loop through regions
        for region in regions.items():
            extent = region[1]['lonlat']

            # argo = active_argo_floats(bbox=extent, time_start=t0, time_end=t1)

            glider = active_gliders(bbox=extent, time_start=t0_glider, time_end=t1, glider_id=glider,)
            glider.reset_index(inplace=True)
            glider['datetime'] = pd.to_datetime(glider['time (UTC)'].dt.strftime('%Y-%m-%d %H:%M:%S'))
            glider.set_index('datetime', inplace=True)

            bathy = bathy.sel(
                lon=slice(extent[0]-1, extent[1]+1),
                lat=slice(extent[2]-1, extent[3]+1)
            )


            extent = np.add(extent, [-1, 1, -1, 1]).tolist()
            print(f'Region: {region[0]}, Extent: {extent}')

            # interpolating transect X and Y to lat and lon
            lonIndex = np.round(np.interp(extent[:2], lon[0, :], np.arange(0, len(lon[0, :])))).astype(int)
            latIndex = np.round(np.interp(extent[2:], lat[:, 0], np.arange(0, len(lat[:, 0])))).astype(int)
            sub = ds.sel(
                X=slice(lonIndex[0], lonIndex[1]),
                Y=slice(latIndex[0], latIndex[1])
            )
            region_name = region[0]
            limits = region[1]
            extent = limits['lonlat']

            region_file_str = ('_').join(region_name.lower().split(' '))
            save_dir_region = os.path.join(save_dir, 'regions', region_file_str)


            for k, v in limits.items():
                if k == 'lonlat':
                    continue
                elif k == 'currents':
                    continue

                var_str = ' '.join(sub[k].standard_name.split('_')).title()
                save_dir_var = os.path.join(save_dir_region, k)

                for item in v:
                    depth = item['depth']
                    dsd = sub.sel(depth=depth)

                    save_dir_depth = os.path.join(save_dir_var, f'{depth}m')

                    title = f'Region: {region_name.title()}, Variable: {var_str} @ {depth}m\n' \
                            f'Time: {str(t1)} UTC, Model: {model.upper()}\n'

                    # if not gliders.empty or not argo.empty:
                    title += f'Glider Window {str(t0)} to {str(t1)}'
                    sname = f'{model}-{k}-{t1.strftime("%Y-%m-%dT%H%M%SZ")}'

                    save_dir_final = os.path.join(save_dir_depth, t1.strftime('%Y/%m'))
                    os.makedirs(save_dir_final, exist_ok=True)
                    save_file = os.path.join(save_dir_final, sname)
                    save_file = save_file + '.png'

                    vargs = {}
                    vmin = item['limits'][0]
                    vmax = item['limits'][1]
                    cmap = cmaps(sub[k].name)

                    if k == 'sea_surface_height':
                        levels = np.arange(vmin, vmax, item['limits'][2])
                    elif k == 'salinity':
                        levels= np.arange(vmin, vmax, item['limits'][2])
                    elif k == 'temperature':
                        levels = np.arange(vmin, vmax, item['limits'][2])

                    try:
                        vargs.pop('vmin'), vargs.pop('vmax')
                    except KeyError:
                        pass

                    fig, ax = plt.subplots(
                        figsize=(11, 8),
                        subplot_kw=dict(projection=projection['map'])
                    )

                    # Plot title
                    plt.title(title)

                    # ax = region_subplot(fig, ax, extent, dsd[k].squeeze(), title, **vargs)
                    # ax.plot(track['lon'], track['lat'], 'k-', linewidth=6, transform=vargs['transform'])
                    # if currents['bool']:
                    #     cwargs = copy.deepcopy(currents)
                    #     cwargs.pop('bool')
                    #     map_add_currents(ax, dsd, **cwargs)

                    # bathy = bathy or None
                    # transects = transects or pd.DataFrame()

                    cargs = {}
                    cargs['vmin'] = vmin
                    cargs['vmax'] = vmax
                    cargs['levels'] = levels
                    cargs['transform'] = projection['data']
                    cargs['cmap'] = cmap
                    cargs['extend'] = 'both'
                    h = ax.contourf(dsd['lon'], dsd['lat'], dsd[k].squeeze(), **cargs)

                    map_add_features(ax, extent)

                    # map_add_gliders(ax, gliders, transform)
                    q = glider.iloc[-1]

                    # plot entire track
                    ax.plot(glider['longitude (degrees_east)'], glider['latitude (degrees_north)'], color='white',
                            linewidth=4, transform=projection['data'])

                    # # plot track during ida
                    # ax.plot(glider[t0:t1]['longitude (degrees_east)'], glider[t0:t1]['latitude (degrees_north)'], color='white',
                    #         linewidth=4, transform=projection['data'])

                    # # plot track during ida
                    # ax.plot(glider[dt.datetime(2021, 8, 28):dt.datetime(2021, 8, 30)]['longitude (degrees_east)'],
                    #         glider[dt.datetime(2021, 8, 28):dt.datetime(2021, 8, 30)]['latitude (degrees_north)'], color='red',
                    #         linewidth=4, transform=projection['data'])

                    # plot most recent lat/lon as a marker
                    ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker='^',
                            markeredgecolor='black',
                            markersize=11, transform=projection['data'])

                    # rapid_floats = np.intersect1d(argo.platform_number.unique(), pdf.wmo.unique())
                    # normal_floats = np.setdiff1d(argo.platform_number.unique(), rapid_floats)

                    # most_recent = argo.loc[argo.groupby('platform_number')['time (UTC)'].idxmax()]

                    n = 0
                    # for float in argo.itertuples():
                    #     if float.platform_number in rapid_floats:
                    #         color = 'green'
                    #         # label = 'Rapid Profiling Argo'
                    #     elif float.platform_number in normal_floats:
                    #         color = 'yellow'
                    #         # label = 'Standard'
                    #     ax.plot(float._4, float._5, marker='s', markersize=9, markeredgecolor='black',
                    #             color=color,
                    #             # label=float.platform_number,
                    #             transform=projection['data'])
                    #     # map_add_legend(ax)
                    #     n = n + 1
                    ax.plot(-88.237, 29.207, marker='*', markersize=16, markeredgecolor='black', color='red',
                            label='NDBC-42040', transform=projection['data'])
                    # ax.plot(-89.649, 28.988, marker='*', markersize=16, markeredgecolor='black', color='red',
                            # label='NDBC-42084', transform=projection['data'])

                    # map_add_legend(ax)

                    map_add_bathymetry(ax, bathy, projection['data'])

                    ax.set_title(title, fontsize=18, fontweight='bold')
                    ax.set_xlabel('Longitude', fontsize=16)
                    ax.set_ylabel('Latitude', fontsize=16)

                    axins = inset_axes(ax,  # here using axis of the lowest plot
                                       width="2.5%",  # width = 5% of parent_bbox width
                                       height="100%",  # height : 340% good for a (4x4) Grid
                                       loc='lower left',
                                       bbox_to_anchor=(1.05, 0., 1, 1),
                                       bbox_transform=ax.transAxes,
                                       borderpad=0,
                                       )

                    cb = fig.colorbar(h, cax=axins)
                    cb.ax.tick_params(labelsize=14)
                    cb.set_label(f'{dsd[k].name.title()} ({dsd[k].units})', fontsize=13)

                    map_add_ticks(ax, extent)

                    # h, l = ax.get_legend_handles_labels()  # get labels and handles from ax1

                    # Shrink current axis's height by 10% on the bottom
                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    #                  box.width, box.height * 0.9])
                    #
                    # Put a legend below current axis
                    # ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    #           fancybox=True, shadow=True, ncol=5)

                    os.environ["CPL_ZIP_ENCODING"] = "UTF-8"
                    os.environ["TZ"] = "GMT0"


                    def url_lister(url):
                        urls = []
                        connection = urlopen(url)
                        dom = lxml.html.fromstring(connection.read())
                        for link in dom.xpath("//a/@href"):
                            urls.append(link)
                        return urls


                    def download(url, path):
                        sys.stdout.write(fname + "\n")
                        if not os.path.isfile(path):
                            urlretrieve(url, filename=path, reporthook=progress_hook(sys.stdout))
                            sys.stdout.write("\n")
                            sys.stdout.flush()


                    def progress_hook(out):
                        """
                        Return a progress hook function, suitable for passing to
                        urllib.retrieve, that writes to the file object *out*.
                        """

                        def it(n, bs, ts):
                            got = n * bs
                            if ts < 0:
                                outof = ""
                            else:
                                # On the last block n*bs can exceed ts, so we clamp it
                                # to avoid awkward questions.
                                got = min(got, ts)
                                outof = "/%d [%d%%]" % (ts, 100 * got // ts)
                            out.write("\r  %d%s" % (got, outof))
                            out.flush()

                        return it


                    code = "al092021"

                    hurricane = "{}_5day".format(code)

                    nhc = "http://www.nhc.noaa.gov/gis/forecast/archive/"

                    # # We don't need the latest file b/c that is redundant to the latest number.
                    # fnames = [
                    #     fname
                    #     for fname in url_lister(nhc)
                    #     if fname.startswith(hurricane) and "latest" not in fname
                    # ]

                    base = os.path.abspath(os.path.join(os.path.curdir, "data", hurricane))

                    if not os.path.exists(base):
                        os.makedirs(base)

                    # for fname in fnames:
                    #     url = "{}/{}".format(nhc, fname)
                    #     path = os.path.join(base, fname)
                    #     download(url, path)

                    cones, points = [], []
                    for fname in sorted(glob(os.path.join(base, "{}_*.zip".format(hurricane)))):
                        number = os.path.splitext(os.path.split(fname)[-1])[0].split("_")[-1]
                        pgn = geopandas.read_file(
                            "/{}-{}_5day_pgn.shp".format(code, number), vfs="zip://{}".format(fname)
                        )
                        cones.append(pgn)

                        pts = geopandas.read_file(
                            "/{}-{}_5day_pts.shp".format(code, number), vfs="zip://{}".format(fname)
                        )
                        # Only the first "obsevartion."
                        points.append(pts.iloc[0])

                    colors = {
                        "Subtropical Depression": "yellow",
                        "Tropical Depression": "yellow",
                        "Tropical Storm": "orange",
                        "Subtropical Storm": "orange",
                        "Hurricane": "coral",
                        "Major Hurricane": "crimson",
                    }

                    size = {
                        "Subtropical Depression": 5,
                        "Tropical Depression": 6,
                        "Tropical Storm": 3,
                        "Subtropical Storm": 6,
                        "Hurricane": 9,
                        "Major Hurricane": 16,
                    }

                    border = {
                        "Subtropical Depression": 'black',
                        "Tropical Depression": 'black',
                        "Tropical Storm": 'black',
                        "Subtropical Storm": 'black',
                        "Hurricane": 'black',
                        "Major Hurricane": 'black',
                    }

                    time_orig = []
                    time_convert = []
                    lat = []
                    lon = []
                    strength = []

                    # All the points along the track.
                    for point in points:
                        if 'CDT' in point["FLDATELBL"]:
                            tdt = parser.parse(point['FLDATELBL'].replace('CDT', '-05:00'))
                            cdt = tdt.astimezone(timezone('UTC'))
                        elif 'EDT' in point['FLDATELBL']:
                            tdt = parser.parse(point['FLDATELBL'].replace('EDT', '-04:00'))
                            cdt = tdt.astimezone(timezone('UTC'))

                        time_orig.append(tdt)
                        time_convert.append(cdt)
                        strength.append(point["TCDVLP"])
                        lat.append(point["LAT"])
                        lon.append(point["LON"])
                        plt.plot(point['LON'], point["LAT"], color=colors[point["TCDVLP"]])

                    test = pd.DataFrame(
                        {'time_orig': time_orig, 'time_convert': time_convert, 'lon': lon, 'lat': lat,
                         'strength': strength})

                    test = test[
                        (test.lon >= extent[0]) & (test.lon <= extent[1])
                        &
                        (test.lat >= extent[2]) & (test.lat <= extent[3])
                        ]
                    ax.plot(test.lon, test.lat, 'k-', linewidth=1, transform=projection['data'])
                    ax.scatter(test.lon, test.lat,
                               c=test['strength'].map(colors),
                               s=test['strength'].map(size) * 10,
                               edgecolors=test['strength'].map(border),
                               transform=projection['data'], zorder=12)
                    # for t in test.iterrows():
                    #     ax.text(t[1].lon - 1, t[1].lat , t[1].time_convert.strftime('%Y-%m-%dT%H:%M:%SZ'), fontsize=8,
                    #             fontweight='bold', color='white', transform=transform, zorder=20)

                    plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                    # plt.show()