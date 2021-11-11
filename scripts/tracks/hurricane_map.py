import os
from glob import glob
import geopandas
import pandas as pd
import lxml.html
import sys
import warnings
from dateutil import parser
from pytz import timezone
import matplotlib.pyplot as plt
from hurricanes.plotting import map_add_ticks, map_add_features, map_add_gliders, map_add_legend
import cartopy.crs as ccrs
from hurricanes.platforms import active_gliders

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

try:
    from urllib.request import urlopen, urlretrieve
except Exception:
    from urllib import urlopen, urlretrieve


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
extent = [-93, -87, 26, 31]
projection = ccrs.PlateCarree()

sos_name = "waves"

hurricane = "{}_5day".format(code)

nhc = "http://www.nhc.noaa.gov/gis/forecast/archive/"

# We don't need the latest file b/c that is redundant to the latest number.
fnames = [
    fname
    for fname in url_lister(nhc)
    if fname.startswith(hurricane) and "latest" not in fname
]

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
    "Hurricane": "red",
    "Major Hurricane": "crimson",
}

size = {
    "Subtropical Depression": 5,
    "Tropical Depression": 6,
    "Tropical Storm": 7,
    "Subtropical Storm": 8,
    "Hurricane": 9,
    "Major Hurricane": 10,
}


time_orig = []
time_convert = []
lat = []
lon = []
strength = []

fig, ax = plt.subplots(
                figsize=(11, 8),
                subplot_kw=dict(projection=projection)
            )

map_add_features(ax, extent)

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
    plt.plot(point['LON'], point["LAT"], color=colors[point["TCDVLP"]], transform=projection, zorder=20)

test = pd.DataFrame({'time_orig': time_orig, 'time_convert': time_convert, 'lon': lon, 'lat': lat, 'strength': strength})
# fig, ax = plt.subplots()
t0 = pd.Timestamp(2021, 8, 28, 18, 0, 0)
t1 = pd.Timestamp(2021, 8, 30)
gliders = active_gliders(extent, t0, t1)

ax.plot(test.lon, test.lat, 'k-', linewidth=1, transform=projection, zorder=20)
ax.scatter(test.lon, test.lat, c=test['strength'].map(colors), s=test['strength'].map(size)*4, transform=projection, zorder=20)

test.set_index('time_convert', inplace=True)
test = test[t0:t1]
test.reset_index(inplace=True)

for t in test.iterrows():
    ax.text(t[1].lon-2, t[1].lat-.05, t[1].time_convert.strftime('%Y-%m-%dT%H:%M:%SZ'), fontsize=10, fontweight='bold', transform=projection, zorder=20)

map_add_gliders(ax, gliders, transform=projection)
map_add_legend(ax)

# Plot title

# map_add_bathymetry(ax, bathy, transform)
map_add_ticks(ax, extent)

for cone in cones:
    ax.add_geometries(cone.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k')
# from matplotlib.path import Path

# Path.clip_to_bbox(extent)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

ax.set_title('Hurricane Ida Path\n2021-08-27 to 2021-08-30', fontsize=18, fontweight='bold')
plt.savefig('/Users/mikesmith/Documents/ida-path.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
