from ioos_model_comparisons.src import limits_regions
import datetime as dt
from ioos_model_comparisons.src import active_drifters
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from ioos_model_comparisons.src import map_add_features, map_add_ticks

regions = limits_regions('rtofs', ['carib'])
transform = ccrs.PlateCarree()
now = dt.datetime.utcnow()
then = dt.datetime(2018, 1, 1)

extent = regions['Caribbean']['lonlat']


######################
# Argo Data
######################

# argo_data = active_argo_floats(extent, then, now)
# argo_data = pickle.load(open("/Users/mikesmith/Documents/argo-2018-2021.pkl", "rb"))
#
# test = argo_data.groupby(['platform_number', 'time (UTC)']).agg({'longitude (degrees_east)': max,  'latitude (degrees_north)': max})
# test = test.reset_index()
# unique_argo = test.platform_number.unique().shape[0]
#
# transform = ccrs.PlateCarree()
#
# fig, ax = plt.subplots(
#     figsize=(11, 8),
#     subplot_kw=dict(projection=transform)
# )
#
# vargs = {}
#
# map_add_features(ax, extent)
# # map_add_bathymetry(ax1, bathy, transform)
# map_add_ticks(ax, extent)
#
# ax.scatter(test['longitude (degrees_east)'], test['latitude (degrees_north)'], 4,
#            edgecolors='black',
#            transform=transform)
# ax.set_title(f'Caribbean Argo Float Surfacings - 2018-01-01 thru 2021-09-20\n{unique_argo} unique floats, {test.shape[0]} surfacings', fontsize=18, fontweight='bold')
# ax.set_xlabel('Longitude', fontsize=14)
# ax.set_ylabel('Latitude', fontsize=14)
#
# plt.savefig('/Users/mikesmith/Documents/argo_floats_record.png', dpi=150)

######################
# Glider Data
######################
# glider_data = active_gliders(regions['Caribbean']['lonlat'], then, now)
# glider_data = pickle.load(open("/Users/mikesmith/Documents/glider-2018-2021.pkl", "rb"))
#
# fig, ax = plt.subplots(
#     figsize=(11, 8),
#     subplot_kw=dict(projection=transform)
# )
#
# vargs = {}
#
# map_add_features(ax, extent)
# # map_add_bathymetry(ax1, bathy, transform)
# map_add_ticks(ax, extent)
# custom_cmap = categorical_cmap(10, 6, cmap="tab10")
# marker = cycle(['^', 'o', 'h', 'p', 'D'])
#
# n = 0
# for g, new_df in glider_data.groupby(level=0):
#     q = new_df.iloc[-1]
#     ax.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'], 'k--',
#             linewidth=1.5, transform=ccrs.PlateCarree())
#     ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker=next(marker), markeredgecolor='black',
#             markersize=7, label=g, color=custom_cmap.colors[n], transform=transform, zorder=20)
#     n = n + 1
# num_gliders = glider_data.reset_index()['dataset_id'].unique().shape
# ax.set_title(f'Caribbean Gliders - 2018-01-01 thru 2021-09-20\n{num_gliders[0]} deployments', fontsize=18, fontweight='bold')
# ax.set_xlabel('Longitude', fontsize=14)
# ax.set_ylabel('Latitude', fontsize=14)
#
# # Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
#
# # Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
#           fancybox=True, shadow=True, ncol=6,  prop={"size": 6})
# # ax.legend(ncol=10, loc='center', fontsize=8)
#
# plt.savefig('/Users/mikesmith/Documents/glider_record.png', dpi=150)


######################
# Drifter Data
######################
drifter_data = active_drifters(extent, then, now)
# drifter = pickle.load(open("/Users/mikesmith/Documents/glider-2018-2021.pkl", "rb"))

fig, ax = plt.subplots(
    figsize=(11, 8),
    subplot_kw=dict(projection=transform)
)

vargs = {}

map_add_features(ax, extent)
# map_add_bathymetry(ax1, bathy, transform)
map_add_ticks(ax, extent)

# custom_cmap = categorical_cmap(10, 6, cmap="tab10")
# marker = cycle(['^', 'o', 'h', 'p', 'D'])

n = 0
for g, new_df in drifter_data.groupby('WMO'):
    q = new_df.iloc[-1]
    ax.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'],
            linewidth=1.5, transform=ccrs.PlateCarree())
    # ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker=next(marker), markeredgecolor='black',
    #         markersize=7, label=g, color=custom_cmap.colors[n], transform=transform, zorder=20)
    n = n + 1

num_drifters = drifter_data.reset_index()['WMO'].unique().shape
ax.set_title(f'Caribbean Drifters - 2021-01-01 thru 2021-09-20\n{num_drifters[0]} drifters', fontsize=18, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

# Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),
#           fancybox=True, shadow=True, ncol=6,  prop={"size": 6})
# ax.legend(ncol=10, loc='center', fontsize=8)

plt.savefig('/Users/mikesmith/Documents/drifter_record.png', dpi=150)


fig, ax = plt.subplots(
    figsize=(11, 8),
    subplot_kw=dict(projection=transform)
)

vargs = {}

map_add_features(ax, extent)
# map_add_bathymetry(ax1, bathy, transform)
map_add_ticks(ax, extent)
plt.hexbin(drifter_data['longitude (degrees_east)'], drifter_data['latitude (degrees_north)'], transform=transform, gridsize=50)
plt.show()