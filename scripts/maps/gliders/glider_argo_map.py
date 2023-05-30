import pandas as pd
import matplotlib.pyplot as plt
from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
import cartopy.crs as ccrs
import cool_maps.plot as cplt
import numpy as np

map_projection = ccrs.Mercator()
data_projection = ccrs.PlateCarree()

t0 = pd.Timestamp(2023, 4, 25, 0, 0, 0)
t1 = pd.Timestamp(2023, 5, 2, 0, 0, 0)

# t0 = pd.Timestamp(2022, 6, 1, 0, 0, 0)
# t1 = pd.Timestamp(2022, 12, 1)
# extent = [-100, -46, 6.75, 45.25]

# extent = [-99, -79, 18, 31] #GoM
# rstr = 'gom'

# extent = [-89, -58, 7, 23] # caribbean
# rstr = 'carib'

# extent = [-82, -70, 24, 35]
# rstr = 'sab'

# extent = [-77, -67, 35, 43]
# rstr = 'mab'

extent = [-180, -90, 5, 35]
rstr = 'pacific'

extent = [-10, 15, -40, -20]
rstr = 'esaw'

# extent = [-50, -25, -40, -20]
# rstr = 'wsaw'

# variables to download
var_list = ["project_name", "pi_name", "platform_type", 'temp', 'psal', 'pressure']

# Get glider and argo data.. Save as CSV
# gliders = get_active_gliders(extent, t0, t1, variables=['time', 'latitude', 'longitude', 'profile_id'])
# gliders.to_csv(f'/Users/mikesmith/Documents/gliders_2023_{rstr}.csv')

# argo = get_argo_floats_by_time(extent, t0, t1, variables=["project_name", "pi_name", "platform_type", 'temp', 'psal', 'pres'])
# argo.to_csv(f'/Users/mikesmith/Documents/argo_2023_{rstr}.csv')

# # # # ax.plot(test.lon, test.lat, 'k-', linewidth=1, transform=projection, zorder=20)
# # # # ax.scatter(test.lon, test.lat, c=test['strength'].map(colors), s=test['strength'].map(size)*4, transform=projection, zorder=20)
title_str = f'{t0} to {t1}'


def map_add_all_argo(ax, df, color='green', transform=data_projection):
    grouped = df.groupby(['lon', 'lat'])
    for i, x in grouped:
        h = ax.plot(i[0], i[1],
                    marker='o', 
                    markersize=4,
                    markeredgecolor='black',
                    linestyle='None',
                    color=color, 
                    transform=transform,
                    zorder=100)
    return h
        

def map_add_all_gliders(ax, df, transform):
    grouped = df.groupby(['glider'])
    for i, x in grouped:
        h = ax.plot(x['lon'], x['lat'], linewidth=2, color='red', transform=transform, zorder=100)
    return h


argo = pd.read_csv(f'/Users/mikesmith/Documents/argo_2023_{rstr}.csv')
gliders = pd.read_csv(f'/Users/mikesmith/Documents/gliders_2023_{rstr}.csv')

# GoM
# argo = argo.loc[argo.lon <= -80.9]
# gliders = gliders.loc[gliders.lon <= -80.9]
# argo = argo.loc[argo.lat >= 19]

# Caribbean
# argo = argo.loc[(argo.lon <= -58) & (argo.lon >= -89) & (argo.lat >= 10)]
# argo = argo.loc[(argo.lon >= -87)]
# argo = argo.loc[(argo.lat >= 9)]
# argo = argo.loc[~((argo.lat > 22) & (argo.lon <=-83))]

# SaB
# argo = argo.loc[argo.lat <= 35]
# gliders = gliders.loc[gliders.lat <= 35]

# MaB
# argo = argo.loc[(argo.lat <= 42.5) & (argo.lat >= 35)]
# gliders = gliders.loc[(gliders.lat <= 42.5) & (gliders.lat >= 35)]

# Pacific
# argo = argo[~np.logical_and(argo.lon >= -97, argo.lat > 17)]
# gliders = gliders[~np.logical_and(gliders.lon >= -97, gliders.lat > 17)]

# if np.logical_and(lon_check, lat_check).any():
    # n = n + 1

# ugos_projects = [x for x in argo["project_name"].unique() if 'UGOS' in x]
# other_projects = [x for x in argo["project_name"].unique() if 'UGOS' not in x]
# ugos_projects = [x for x in argo["pi_name"].unique() if 'BOWER' in x]
# other_projects = [x for x in argo["pi_name"].unique() if 'BOWER' not in x]

# ugos_df = pd.concat([argo.loc[argo['pi_name'] == x] for x in ugos_projects], axis=0)
# other_df = pd.concat([argo.loc[argo['pi_name'] == x] for x in other_projects], axis=0)

unique_floats = argo.argo.unique().shape[0]
# ugos_floats = ugos_df.argo.unique().shape[0]
# other_floats = other_df.argo.unique().shape[0]

argo_profiles_all = argo.time.unique().shape[0]
# argo_profiles_ugos = ugos_df.time.unique().shape[0]
# argo_profiles_other = other_df.time.unique().shape[0]

# unique_gliders = gliders.glider.unique().shape[0]
# glider_profiles = gliders.profile_id.unique().shape[0]

from scipy.io import loadmat
import cartopy.feature as cfeature
import xarray as xr
# fname = '/Users/mikesmith/Downloads/GMOG_CICESE_Trajectories_0024_0025_0026.mat'
# data = loadmat(fname)

# lat1 = data['lat_0024']
# lon1 = data['lon_0024']

# lat2 = data['lat_0025']
# lon2 = data['lon_0025']

# lat3 = data['lat_0026']
# lon3 = data['lon_0026']

fig, ax = cplt.create(extent, bathymetry=False, gridlines=True, labelsize=16)

levels = [-8000, -1000, -100, 0]  # Contour levels (depths)
colors = ['cornflowerblue', cfeature.COLORS['water'],
          'lightsteelblue',]  # contour colors

# Get bathymetry from srtm15
# bs = cplt.get_bathymetry(extent)
bs = xr.open_dataset('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc')
bs = bs.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
bs = bs.rename({'lon': 'longitude', 'lat': 'latitude'})

# add filled contour to map
cs = ax.contourf(bs['longitude'], bs['latitude'], bs['z'], levels,
                 colors=colors, transform=ccrs.PlateCarree(), ticks=False)

# h1 = map_add_all_gliders(ax, gliders, transform=data_projection)

# h = ax.plot(lon1, lat1, linewidth=2, color='maroon', transform=data_projection, zorder=100)
# h = ax.plot(lon2, lat2, linewidth=2, color='maroon', transform=data_projection, zorder=100)
# h = ax.plot(lon3, lat3, linewidth=2, color='maroon', transform=data_projection, zorder=100)

# h2 = map_add_all_argo(ax, ugos_df, transform=data_projection, color='blue')
# h3 = map_add_all_argo(ax, other_df, transform=data_projection, color='green')
# argo = argo.groupby('argo').first()

h2 = map_add_all_argo(ax, argo, transform=data_projection, color='green')

from matplotlib.patches import Rectangle

h_n = []
# h_n.append(h1[0])
# h_n.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))

# h_n.append(h[0])
# h_n.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))

h_n.append(h2[0])
h_n.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))

# h_n.append(h3[0])
# h_n.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))

l_n = []

# l_n.append(f'Gliders (#): {unique_gliders}')
# l_n.append(f'Profiles: {glider_profiles}')

# l_n.append(f'Gliders/CICESE (#): {3}')
# l_n.append(f'Profiles: {1083}')
l_n.append(f'Argo (#): {unique_floats}')
l_n.append(f'Profiles: {argo_profiles_all}')

# l_n.append(f'Argo/UGOS (#): {ugos_floats}')
# l_n.append(f'Profiles: {argo_profiles_ugos}')

# l_n.append(f'Argo/Other (#): {other_floats}')
# l_n.append(f'Profiles: {argo_profiles_other}')


# ax.legend(legend_1, ["Rectangle", "mlines"], loc='lower right') 
leg = ax.legend(h_n, l_n, loc='upper right', fontsize=7, 
                title="Profile Statistics",
                # title_fontsize=8,
                title_fontproperties={'size': 7, 'weight':'bold'}
                ).set_zorder(1000)
# plt.setp(leg.get_title(), fontsize=7, fontweight="bold")

# plt.legend()

ax.set_title(title_str, fontsize=18, fontweight='bold')
# plt.gcf().text(0.08, 0.0, 'Sources\nGlider: IOOS National Glider DAC\nArgo: Global Data Assembly Centre (Argo GDAC)', fontsize=8, fontstyle='italic')

plt.savefig(f'/Users/mikesmith/Documents/hurricane_season_assets_2022_{rstr}_pacific.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()
plt.close()
 