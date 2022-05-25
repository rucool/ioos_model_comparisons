import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from hurricanes.gliders import glider_dataset
from matplotlib.ticker import AutoMinorLocator
from hurricanes.platforms import active_argo_floats

glider = 'ng645-20210613T0000'
save_dir = '/Users/mikesmith/Documents/'
t0 = dt.datetime(2021, 8, 27, 0, 0)
t1 =  dt.datetime(2021, 8, 31, 0, 0)

from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = t0  # False
gargs['time_end'] =t1
gargs['filetype'] = 'dataframe'

glider_df = glider_dataset(glider, **gargs)
argo = active_argo_floats(time_start=t0, time_end=t1, floats='4903356')
grouped_argo = argo.groupby('time (UTC)')


grouped = glider_df.groupby('time')
fig, ax = plt.subplots(figsize=(12, 6))

surface_temp = []
surface_salinity = []
times = []
for name, group in grouped:
    group.sort_values('depth', inplace=True)
    surface_temp.append(group.iloc[0].temperature)
    surface_salinity.append(group.iloc[0].salinity)
    times.append(name)

# Temperature Profiles (Panel 1)
h = ax.plot(
    times,
    surface_temp,
    'k-o',
)

for name, group in grouped_argo:
    ax.plot(name, group.iloc[0]['temp (degree_Celsius)'], 'ro')


ax.axvline(dt.datetime(2021, 8, 29, 3, 0, 0))

ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax.tick_params(direction='out', labelsize=20)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator(12))
plt.xlabel('Date/Time (GMT)', fontsize=20, fontweight='bold')
plt.ylabel('Temperature (°C)', fontsize=20, fontweight='bold')

plt.xticks(rotation=30)
title_str = f'{glider} Surface Temperature'
ttl = plt.title(title_str, fontsize=28, fontweight='bold')
# ttl.set_position([.5, 1.05])/
plt.tight_layout()
plt.savefig(f'/Users/mikesmith/Documents/{glider}-surface_temperatures.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()


# Salinity Profiles (Panel 1)
fig, ax = plt.subplots(figsize=(12, 6))

h = ax.plot(
    times,
    surface_salinity,
    'k-o',
)

for name, group in grouped_argo:
    ax.plot(name, group.iloc[0]['psal (PSU)'], 'ro')

ax.axvline(dt.datetime(2021, 8, 29, 3, 0, 0))

ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax.tick_params(direction='out', labelsize=20)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_minor_locator(AutoMinorLocator(12))
plt.xlabel('Date/Time (GMT)', fontsize=20, fontweight='bold')
plt.ylabel('Salinity', fontsize=20, fontweight='bold')

plt.xticks(rotation=30)
title_str = f'{glider} Surface Salinity'
ttl = plt.title(title_str, fontsize=28, fontweight='bold')
# ttl.set_position([.5, 1.05])
plt.tight_layout()
plt.savefig(f'/Users/mikesmith/Documents/{glider}-surface_salinity.png', bbox_inches='tight', pad_inches=0.1, dpi=300)