import matplotlib.pyplot as plt
from hurricanes.src import map_add_ticks, map_add_features
import cartopy.crs as ccrs
import pandas as pd

df = pd.read_csv('/Users/mikesmith/Documents/whoi_argo_hurricane-season-2021-complete_profiles.csv', parse_dates=[1], infer_datetime_format=True)
df = df[df.time > '2021-09-01']
tdf = df[df.surfacing_interval < '3 days']
extent = [-100, -60, 10, 31]
fig, ax = plt.subplots(
    figsize=(11, 8),
    subplot_kw=dict(projection=ccrs.Mercator()))

for name, group, in tdf.groupby('wmo'):
    ax.scatter(group.lon, group.lat, s=12, transform=ccrs.PlateCarree(), zorder=20)
map_add_features(ax, extent)
map_add_ticks(ax, extent, fontsize=18)
plt.title('Argo Floats - Rapid Sampling\nAugust 1, 2021 to October 1, 2021', fontweight='bold', fontsize=24)
plt.savefig('rapid_sampling_argo_floats_augsept2021.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
# plt.show()