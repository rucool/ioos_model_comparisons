import pandas as pd
import matplotlib.pyplot as plt
from siphon.simplewebservice.ndbc import NDBC
import numpy as np
import math
from oceans.plotting import stick_plot
from oceans.ocfis import spdir2uv
from matplotlib.ticker import AutoMinorLocator

# hurricane = 'Henri'
#ndbc = ['44017', '44097']  # henri
# time = ['2021-08-19', '2021-08-24']

hurricane = 'Ida'
ndbc = ['42040', '42084']  #ida
time = ['2021-08-27', '2021-08-31']

df = NDBC.realtime_observations(ndbc[0])
df2 = NDBC.realtime_observations(ndbc[1])

df['datetime'] = pd.to_datetime(df['time'].dt.strftime('%Y-%m-%d %H:%M:%S'))
df.sort_values(['datetime'], inplace=True)
df.set_index('datetime', inplace=True)

df2['datetime'] = pd.to_datetime(df2['time'].dt.strftime('%Y-%m-%d %H:%M:%S'))
df2.sort_values(['datetime'], inplace=True)
df2.set_index('datetime', inplace=True)

df = df.tz_localize(None)
df2 = df2.tz_localize(None)
df = df.loc[time[0]:time[1]]
df2 = df2.loc[time[0]:time[1]]

df = df.resample('H').mean()
df2 = df2.resample('H').mean()

df = df.reset_index()
df2 = df2.reset_index()

diff = df['water_temperature'] - df['air_temperature']
diff2 = df2['water_temperature'] - df2['air_temperature']

# Get low pressure time
low = df[df.pressure == df.pressure.min()]['datetime'].to_list()[0]

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1,
                                                        figsize=(24, 24),
                                                        constrained_layout=True,
                                                        )

# Pressure
ax1.plot(df['datetime'], df['pressure'], color='tab:red', linestyle='--', label=ndbc[0])
ax1.plot(df2['datetime'], df2['pressure'], color='tab:blue', linestyle='-', label=ndbc[1])
ax1.set_ylabel('Pressure [hPa]', fontsize=18, fontweight='bold')
ax1.legend(fontsize=16)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=18)
ax1.xaxis.set_minor_locator(AutoMinorLocator(12))

plim = ax1.yaxis.get_data_interval()

temp = []

# Water temperature
ax2.plot(df['datetime'], df['air_temperature'], color='tab:red', linestyle='--', label=ndbc[0])
ax2.plot(df2['datetime'], df2['air_temperature'], color='tab:blue', linestyle='-', label=ndbc[1])
ax2.set_ylabel('Air Temp [°C]', fontsize=18, fontweight='bold')
ax2.legend(fontsize=16)
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=18)
ax2.xaxis.set_minor_locator(AutoMinorLocator(12))

lim = ax2.yaxis.get_data_interval()
temp.append(lim)

# Water temperature
ax3.plot(df['datetime'], df['water_temperature'], color='tab:red', linestyle='--', label=ndbc[0])
ax3.plot(df2['datetime'], df2['water_temperature'], color='tab:blue', linestyle='-', label=ndbc[1])
ax3.set_ylabel('Water Temp [°C]', fontsize=18, fontweight='bold')
ax3.legend(fontsize=16)
ax3.grid(True, linestyle='--', linewidth=0.5)
ax3.tick_params(axis='both', labelsize=18)
ax3.xaxis.set_minor_locator(AutoMinorLocator(12))
lim = ax3.yaxis.get_data_interval()
temp.append(lim)

ax4.plot(df['datetime'], diff, color='tab:red', linestyle='--', label=ndbc[0])
ax4.plot(df2['datetime'], diff2, color='tab:blue', linestyle='-', label=ndbc[1])
ax4.set_ylabel('Δ Temp [°C]', fontsize=18, fontweight='bold')
ax4.legend(fontsize=16)
ax4.grid(True, linestyle='--', linewidth=0.5)
ax4.tick_params(axis='both', labelsize=18)
ax4.xaxis.set_minor_locator(AutoMinorLocator(12))
ax4.set_ylim([-2, 6])

# Water temperature
ax5.plot(df['datetime'], df['wave_height'], color='tab:red', linestyle='--', label=ndbc[0])
ax5.plot(df2['datetime'], df2['wave_height'], color='tab:blue', linestyle='-', label=ndbc[1])
ax5.set_ylabel('Wave Height [m]', fontsize=18, fontweight='bold')
ax5.legend(fontsize=16)
ax5.grid(True, linestyle='--', linewidth=0.5)
ax5.tick_params(axis='both', labelsize=18)
ax5.xaxis.set_minor_locator(AutoMinorLocator(12))

# Wind speed
# ax6b = ax6.twinx()
ax6.plot(df['datetime'], df['wind_speed'], color='tab:red',  linestyle='--', label=ndbc[0])
ax6.plot(df2['datetime'], df2['wind_speed'], color='tab:blue',  label=ndbc[1])

# ax6.plot(df['datetime'], df['wind_gust'], color='tab:red', linestyle='--', alpha=0.3, label=ndbc[0])
# ax6b.plot(df['datetime'], df['wind_direction'], color='tab:red', linestyle='-', alpha=0.3, label=ndbc[0])
# ax6b.plot(df2['datetime'], df2['wind_direction'], color='tab:blue', linestyle='-', alpha=0.3, label=ndbc[1])

ax6.set_ylabel('Wind Speed [m/s]', fontsize=18, fontweight='bold')
# ax6b.set_ylabel('Wind Direction', fontsize=10, fontweight='bold')
ax6.legend(fontsize=16)
ax6.grid(True, linestyle='--', linewidth=0.5)
ax6.tick_params(axis='both', labelsize=18)
ax6.xaxis.set_minor_locator(AutoMinorLocator(12))

# Stick plot
u1, v1 = spdir2uv(df['wind_speed'], df['wind_direction'], deg=True)
u2, v2 = spdir2uv(df2['wind_speed'], df2['wind_direction'], deg=True)

q = stick_plot(df['datetime'], -u1, -v1, ax=ax7)

ref = 10
qk = plt.quiverkey(
    q,
    0.1,
    0.85,
    ref,
    "{} {}".format(ref, 'm/s'),
    labelpos="N",
    coordinates="axes",
)


ax7.set_ylabel('Stick Plot', fontsize=18, fontweight='bold')
ax7.tick_params(axis='both', labelsize=18)
ax7.xaxis.set_minor_locator(AutoMinorLocator(12))
qlim = ax7.yaxis.get_data_interval()


plt.suptitle(f'Hurricane {hurricane} - NDBC Buoy Observations', fontsize=20, fontweight='bold')

temp = pd.DataFrame(np.array(temp), columns=['min', 'max'])
temp_min = math.floor(temp['min'].min() * 4) / 4
temp_max = math.ceil(temp['max'].max() * 4) / 4
ax2.set_ylim([temp_min, temp_max])
ax3.set_ylim([temp_min, temp_max])

# lim = ax1.yaxis.get_data_interval()
ax1.vlines(low, plim[0], plim[1], 'k', linestyles='dashdot')
ax2.vlines(low, temp_min, temp_max, 'k', linestyles='dashdot')
ax3.vlines(low, temp_min, temp_max, 'k', linestyles='dashdot')
ax4.vlines(low, -2, 6, 'k', linestyles='dashdot')
ax5.vlines(low, -2, 10, 'k', linestyles='dashdot')
ax6.vlines(low, -2, 22, 'k', linestyles='dashdot')
ax6.vlines(low, qlim[0], qlim[1], 'k', linestyles='dashdot')

# plt.show()
plt.savefig(f"/Users/mikesmith/Documents/hurricane_{hurricane}_ndbc_observations.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()


diff = df['water_temperature'] - df['air_temperature']
diff2 = df2['water_temperature'] - df2['air_temperature']


fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True,)