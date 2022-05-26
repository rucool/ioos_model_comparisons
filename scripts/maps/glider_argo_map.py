import pandas as pd
import matplotlib.pyplot as plt
from hurricanes.plotting import map_add_ticks, map_add_features, map_add_bathymetry
from hurricanes.platforms import active_argo_floats, active_gliders
import cartopy.crs as ccrs
import xarray as xr

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

t0 = pd.Timestamp(2020, 6, 1, 0, 0, 0)
t1 = pd.Timestamp(2020, 12, 1)
extent = [-100, -46, 6.75, 45.25]

def map_add_all_argo(ax, df, transform):
    grouped = df.groupby(['longitude (degrees_east)', 'latitude (degrees_north)'])
    for i, x in grouped:
        ax.plot(i[0], i[1], marker='o', markersize=3, color='blue', transform=transform)
    
    return ax


def map_add_all_gliders(ax, df, transform):
    grouped = df.groupby(['dataset_id'])
    for i, x in grouped:
        ax.plot(x['longitude (degrees_east)'], x['latitude (degrees_north)'], linewidth=2, color='red', transform=transform)
    return ax

# import matplotlib.ticker as mticker

# # function to define major and minor tick locations and major tick labels
# def get_ticks(bounds, dirs, otherbounds):
#     dirs = dirs.lower()
#     l0 = np.float(bounds[0])
#     l1 = np.float(bounds[1])
#     r = np.max([l1 - l0, np.float(otherbounds[1]) - np.float(otherbounds[0])])
#     if r <= 1.5:
#         # <1.5 degrees: 15' major ticks, 5' minor ticks
#         minor_int = 1.0 / 12.0
#         major_int = 1.0 / 4.0
#     elif r <= 3.0:
#         # <3 degrees: 30' major ticks, 10' minor ticks
#         minor_int = 1.0 / 6.0
#         major_int = 0.5
#     elif r <= 7.0:
#         # <7 degrees: 1d major ticks, 15' minor ticks
#         minor_int = 0.25
#         major_int = np.float(1)
#     elif r <= 15:
#         # <15 degrees: 2d major ticks, 30' minor ticks
#         minor_int = 0.5
#         major_int = np.float(2)
#     elif r <= 30:
#         # <30 degrees: 3d major ticks, 1d minor ticks
#         minor_int = np.float(1)
#         major_int = np.float(3)
#     else:
#         # >=30 degrees: 5d major ticks, 1d minor ticks
#         minor_int = np.float(1)
#         major_int = np.float(5)

#     minor_ticks = np.arange(np.ceil(l0 / minor_int) * minor_int, np.ceil(l1 / minor_int) * minor_int + minor_int,
#                             minor_int)
#     minor_ticks = minor_ticks[minor_ticks <= l1]
#     major_ticks = np.arange(np.ceil(l0 / major_int) * major_int, np.ceil(l1 / major_int) * major_int + major_int,
#                             major_int)
#     major_ticks = major_ticks[major_ticks <= l1]

#     if major_int < 1:
#         d, m, s = dd2dms(np.array(major_ticks))
#         if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
#             n = 'W' * sum(d < 0)
#             p = 'E' * sum(d >= 0)
#             dir = n + p
#             major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
#                                  range(len(d))]
#         elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
#             n = 'S' * sum(d < 0)
#             p = 'N' * sum(d >= 0)
#             dir = n + p
#             major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
#                                  range(len(d))]
#         else:
#             major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" for i in range(len(d))]
#     else:
#         d = major_ticks
#         if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
#             n = 'W' * sum(d < 0)
#             p = 'E' * sum(d >= 0)
#             dir = n + p
#             major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
#         elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
#             n = 'S' * sum(d < 0)
#             p = 'N' * sum(d >= 0)
#             dir = n + p
#             major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
#         else:
#             major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" for i in range(len(d))]

#     return minor_ticks, major_ticks, major_tick_labels

# def map_add_ticks(ax, extent, fontsize=13):
#     xl = [extent[0], extent[1]]
#     yl = [extent[2], extent[3]]

#     tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
#     ax.set_xticks(tick0x, minor=True, crs=ccrs.PlateCarree())
#     ax.set_xticks(tick1, crs=ccrs.PlateCarree())
#     ax.set_xticklabels(ticklab, fontsize=fontsize)

#     # get and add latitude ticks/labels
#     tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
#     ax.set_yticks(tick0y, minor=True, crs=ccrs.PlateCarree())
#     ax.set_yticks(tick1, crs=ccrs.PlateCarree())
#     ax.set_yticklabels(ticklab, fontsize=fontsize)

#     gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree())
#     gl.xlocator = mticker.FixedLocator(tick0x)
#     gl.ylocator = mticker.FixedLocator(tick0y)

#     ax.tick_params(which='major',
#                    direction='out',
#                    bottom=True, top=True,
#                    labelbottom=True, labeltop=False,
#                    left=True, right=True,
#                    labelleft=True, labelright=False,
#                    length=5, width=2)

#     ax.tick_params(which='minor',
#                    direction='out',
#                    bottom=True, top=True,
#                    labelbottom=True, labeltop=False,
#                    left=True, right=True,
#                    labelleft=True, labelright=False,
#                    width=1)
#     return ax

fig, ax = plt.subplots(
                figsize=(11, 8),
                subplot_kw=dict(projection=map_projection)
            )

map_add_features(ax, extent)


# gliders = active_gliders(extent, t0, t1)
# gliders.to_csv('/Users/mikesmith/Documents/gliders_2020.csv')

# argo = active_argo_floats(extent, t0, t1)
# argo.to_csv('/Users/mikesmith/Documents/argo_2020.csv')

# ax.plot(test.lon, test.lat, 'k-', linewidth=1, transform=projection, zorder=20)
# ax.scatter(test.lon, test.lat, c=test['strength'].map(colors), s=test['strength'].map(size)*4, transform=projection, zorder=20)


# map_add_gliders(ax, gliders, transform=data_projection)
argo = pd.read_csv('/Users/mikesmith/Documents/argo_2020.csv')
map_add_all_argo(ax, argo, transform=data_projection)

gliders = pd.read_csv('/Users/mikesmith/Documents/gliders_2020.csv')
map_add_all_gliders(ax, gliders, transform=data_projection)

# map_add_legend(ax)
bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
bathy = xr.open_dataset(bathymetry)
bathy = bathy.sel(
    lon=slice(extent[0] - 1, extent[1] + 1),
    lat=slice(extent[2] - 1, extent[3] + 1)
)
# Plot title

map_add_ticks(ax, extent)
map_add_bathymetry(ax, bathy, data_projection)

# Path.clip_to_bbox(extent)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

ax.set_title('2020 Hurricane Season\n2020-06-01 to 2020-11-30', fontsize=18, fontweight='bold')
plt.savefig('/Users/mikesmith/Documents/hurricane_season_assets_2020.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
plt.close()