import xarray as xr
import os
from glob import glob
from hurricanes.limits import limits_regions
import datetime as dt
import numpy as np
from hurricanes.platforms import active_argo_floats
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D
import math

url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '../../plots/argo_profile_model_comparisons/'
# user = 'maristizabalvar'
# pw='MariaCMEMS2018'

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

# floats = ['4903248', '4903253', '4903259', '6902855', '4902916']
floats = ['4903356']
t1 = dt.datetime(2021, 9, 17)
days_to_check_for_argo_surfacing = 15
days_pre_surfacing = 1
days_post_surfacing = 2
depth = 400
dpi = 150
figsize = (20, 14)

regions = limits_regions('rtofs', ['gom'])

argo_search_end = t1 + dt.timedelta(days=1)
argo_search_start = argo_search_end - dt.timedelta(days=days_to_check_for_argo_surfacing)

date_list = [argo_search_end - dt.timedelta(days=x+1) for x in range(days_to_check_for_argo_surfacing + 20)]
date_list.insert(0, argo_search_end)
date_list.insert(0, argo_search_end + dt.timedelta(days=1))
date_list.reverse()

line = ('-', '--', '-.', ':')
alpha = (1, .75, .5, .25)

legend_elements = [
    Line2D([0], [0], color='r', lw=2, label='GOFS'),
    Line2D([0], [0], color='g', lw=2, label='RTOFS'),
    Line2D([0], [0], color='b', lw=2, label='ARGO'),
    # Line2D([0], [0], color='m', lw=2, label='CMEMS')]
    ]

gofs = xr.open_dataset(gofs_url, drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})
gofs = gofs.sel(depth=slice(0, depth))

rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])
rtofs = xr.open_mfdataset(rtofs_files)
rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
rtofs = rtofs.sel(depth=slice(0, depth))

# cmems_salinity = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-so', user, pw)
# cmems_temp = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-thetao', user, pw)
# cmems = xr.merge([cmems_salinity, cmems_temp])
# cmems = cmems.sel(depth=slice(0, 400))

# floats = ['4902350', '6902851', '4902915', '4901719', '4903356', '4903238', '4903238', '4901719']
# floats = ['4901719']
# floats = ['4901716', '6902851', '4902351', '4902915', '4902915', '4902114', '4902441', '4902928']
# floats = ['4903248', '4903253', '4903259', '6902855', '4902916']


for f in floats:
    argos = active_argo_floats(time_start=argo_search_start, time_end=argo_search_end, floats=f)
    if not argos.empty:
        for argo in argos.platform_number.unique():
            temp = argos[argos['platform_number'] == argo]

            for t_float in temp['time (UTC)'].unique():
                temp_float_time = temp[temp['time (UTC)'] == t_float]
                filtered = temp_float_time[(np.abs(stats.zscore(temp_float_time['psal (PSU)'])) < 3)]  # filter salinity
                filtered = filtered[filtered['pres (decibar)'] <= 400]

                x = filtered['longitude (degrees_east)'].unique()[-1]
                y = filtered['latitude (degrees_north)'].unique()[-1]

                t0 = dt.datetime(*[int(x) for x in t_float.strftime('%Y-%m-%d').split('-')])

                # Get today and yesterday dates
                pre = [t0 - dt.timedelta(days=x+1) for x in range(days_pre_surfacing)]
                post = [t0 + dt.timedelta(days=x+1) for x in range(days_post_surfacing + 1)]
                temp_date_list = pre + [t0] + post

                ranges = pd.date_range(temp_date_list[0], temp_date_list[-1], freq='6H', closed='right')

                # Conversion from longitude to GOFS convention
                if x < 0:
                    x_gofs = 360 + x
                else:
                    x_gofs = x

                gofs_temp = gofs.sel(
                    lon=x_gofs,
                    lat=y,
                    method='nearest')

                gofs_temp = gofs_temp.squeeze()

                # cmems_sub = cmems.sel(longitude=x,
                #                       latitude=y,
                #                       method='nearest')

                # interpolating transect X and Y to lat and lon
                rtofslonIndex = np.round(np.interp(x, rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
                rtofslatIndex = np.round(np.interp(y, rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)

                rtofs_sub = rtofs.sel(
                    X=rtofslonIndex,
                    Y=rtofslatIndex,
                    method='nearest'
                )
                rtofs_sub = rtofs_sub.squeeze()

                fig, ax = plt.subplots(
                    2, 4,
                    figsize=figsize,
                    constrained_layout=True
                )

                temp_x = []
                salt_x = []

                for i, t_array in enumerate(np.split(ranges, 4)):

                    n = 0

                    for t in t_array:
                        try:
                            g = gofs_temp.sel(time=t)
                            r = rtofs_sub.sel(time=t)
                            # c = cmems_sub.sel(time=t)
                        except KeyError:
                            continue

                        # Temperature
                        ax[0, i].plot(filtered['temp (degree_Celsius)'], filtered['pres (decibar)'], 'b-', )
                        ax[0, i].plot(g['temperature'].squeeze(), g['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                        ax[0, i].plot(r['temperature'].squeeze(), r['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])
                        # ax[0, i].plot(c['thetao'].squeeze(), c['depth'].squeeze(), f'm{line[n]}', alpha=alpha[n])
                        temp_x.append(ax[0, i].xaxis.get_data_interval())
                        ax[0, i].set_xlabel('Temperature (˚C)', fontsize=14)

                        # Salinity
                        ax[1, i].plot(filtered['psal (PSU)'], filtered['pres (decibar)'], 'b-', )
                        ax[1, i].plot(g['salinity'].squeeze(), g['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                        ax[1, i].plot(r['salinity'].squeeze(), r['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])
                        # ax[1, i].plot(c['so'].squeeze(), c['depth'].squeeze(), f'm{line[n]}', alpha=alpha[n])
                        salt_x.append(ax[1, i].xaxis.get_data_interval())
                        ax[1, i].set_xlabel('Salinity (psu)', fontsize=14)

                        n = n + 1

                temp_x = pd.DataFrame(np.array(temp_x), columns=['min', 'max'])
                salt_x = pd.DataFrame(np.array(salt_x), columns=['min', 'max'])
                temp_x_min = math.floor(temp_x['min'].min() * 4) / 4
                temp_x_max = math.ceil(temp_x['max'].max() * 4) / 4
                salt_x_min = math.floor(salt_x['min'].min() * 4) / 4
                salt_x_max = math.ceil(salt_x['max'].max() * 4) / 4

                ax[0, 0].set_xlim([temp_x_min, temp_x_max])
                ax[0, 1].set_xlim([temp_x_min, temp_x_max])
                ax[0, 2].set_xlim([temp_x_min, temp_x_max])
                ax[0, 3].set_xlim([temp_x_min, temp_x_max])
                ax[1, 0].set_xlim([salt_x_min, salt_x_max])
                ax[1, 1].set_xlim([salt_x_min, salt_x_max])
                ax[1, 2].set_xlim([salt_x_min, salt_x_max])
                ax[1, 3].set_xlim([salt_x_min, salt_x_max])

                for axs in ax.flat:
                    axs.set_ylim([400, 1])
                    axs.grid(True, linestyle='--', linewidth=0.5)
                    axs.tick_params(axis='both', labelsize=10)
                    axs.set_ylabel('Depth (m)', fontsize=12)
                    axs.legend(handles=legend_elements, loc='best', fontsize=10)

                ax[0, 0].set_title(f'Model Run\n{temp_date_list[0].strftime("%Y-%m-%d")}\n-1 Day', fontsize=14)
                ax[0, 1].set_title(f'Model Run\n{temp_date_list[1].strftime("%Y-%m-%d")}\nFloat Surfacing', fontsize=14)
                ax[0, 2].set_title(f'Model Run\n{temp_date_list[2].strftime("%Y-%m-%d")}\n+1 Day', fontsize=14)
                ax[0, 3].set_title(f'Model Run\n{temp_date_list[3].strftime("%Y-%m-%d")}\n+2 Day', fontsize=14)

                plt.suptitle(f'Model Profile Comparisons\n'
                             f'Argo {argo} Surfacing: {t0.strftime("%Y-%m-%d")}\n',
                             fontsize=16,
                             fontweight='bold',
                             y=0.98)

                plt.tight_layout()

                save_str = f'argo_{argo}-{t0.strftime("%Y-%m-%d")}-{len(temp_date_list)-1}day-model-comparison.png'
                full_file = os.path.join(save_dir, save_str)

                plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                plt.close()

gofs.close()
rtofs.close()
# cmems.close()
