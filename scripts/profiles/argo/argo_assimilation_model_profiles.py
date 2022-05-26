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

url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/argo_profile_model_comparisons/'

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

days_to_check_for_argo_surfacing = 15
days_pre_surfacing = 1
days_post_surfacing = 2
dpi = 150
figsize = (16, 12)

regions = limits_regions('rtofs', ['gom', 'carib'])

argo_search_end = dt.date.today()
argo_search_start = argo_search_end - dt.timedelta(days=days_to_check_for_argo_surfacing)

date_list = [argo_search_end - dt.timedelta(days=x+1) for x in range(days_to_check_for_argo_surfacing + 15)]
date_list.insert(0, argo_search_end)
date_list.insert(0, argo_search_end + dt.timedelta(days=1))
date_list.reverse()

line = ('-', '--', '-.', ':')
alpha = (1, .75, .5, .25)

legend_elements = [Line2D([0], [0], color='r', lw=2, label='GOFS'),
                   Line2D([0], [0], color='g', lw=2, label='RTOFS'),
                   Line2D([0], [0], color='b', lw=2, label='ARGO')]


gofs = xr.open_dataset(gofs_url, drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])
rtofs = xr.open_mfdataset(rtofs_files)
rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})

# for r in regions:
#     extent = regions[r]['lonlat']

floats = ['4903354', '4902350']
extent = regions['Gulf of Mexico']['lonlat']

floats = active_argo_floats(bbox=extent, time_start=argo_search_start, time_end=argo_search_end, floats=floats[1])

for argo in floats.platform_number.unique():
    temp = floats[floats['platform_number'] == argo]

    for t_float in temp['time (UTC)'].unique():
        temp_float_time = temp[temp['time (UTC)'] == t_float]
        filtered = temp_float_time[(np.abs(stats.zscore(temp_float_time['psal (PSU)'])) < 3)]  # filter salinity

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
            time=ranges,
            lon=x_gofs,
            lat=y,
            method='nearest')

        gofs_temp = gofs_temp.squeeze()

        # interpolating transect X and Y to lat and lon
        rtofslonIndex = np.round(np.interp(x, rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
        rtofslatIndex = np.round(np.interp(y, rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)

        rtofs_sub = rtofs.sel(
            time=ranges,
            X=rtofslonIndex,
            Y=rtofslatIndex,
            method='nearest'
        )
        rtofs_sub = rtofs_sub.squeeze()
        rtofs_sub.load()

        gofs_lon = str(round(gofs_temp.lon.data - 360, 2))
        gofs_lat = str(round(gofs_temp.lat.data - 0, 2))
        rtofs_lon = str(round(np.float64(rtofs_sub.lon.data), 2))
        rtofs_lat = str(round(np.float64(rtofs_sub.lat.data), 2))

        grouped_gofs = list(gofs_temp.groupby('time.day'))

        fig, ax = plt.subplots(
            2, 4,
            sharey=True,
            figsize=figsize,
            # constrained_layout=True
        )

        for i, tds in enumerate(grouped_gofs):
            gds = grouped_gofs[i][1]

            n = 0

            for t in gds.time:
                g = gds.sel(time=t)
                try:
                    r = rtofs_sub.sel(time=t)
                except KeyError:
                    continue

                # Temperature
                ax[0, i].plot(filtered['temp (degree_Celsius)'], filtered['pres (decibar)'], 'b-', )
                ax[0, i].plot(g['temperature'].squeeze(), g['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                ax[0, i].plot(r['temperature'].squeeze(), r['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])

                ax[0, i].set_xlabel('Temperature (˚C)', fontsize=8)
                ax[0, i].set_xlim([8, 32])

                # Salinity
                ax[1, i].plot(filtered['psal (PSU)'], filtered['pres (decibar)'], 'b-', )
                ax[1, i].plot(g['salinity'].squeeze(), g['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                ax[1, i].plot(r['salinity'].squeeze(), g['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])

                ax[1, i].set_xlabel('Salinity (psu)', fontsize=8)
                ax[1, i].set_xlim([34, 38])

                n = n + 1

        for axs in ax.flat:
            axs.set_ylim([400, 1])
            axs.grid(True, linestyle='--', linewidth=0.5)
            axs.tick_params(axis='x', labelsize=7)
            axs.tick_params(axis='y', labelsize=7)
            axs.set_ylabel('Depth (m)', fontsize=8)
            axs.legend(handles=legend_elements, loc='best', fontsize=7)

        ax[0, 0].set_title(f'{date_list[0].strftime("%Y-%m-%d")}', fontsize=9)
        ax[0, 1].set_title(f'{date_list[1].strftime("%Y-%m-%d")}', fontsize=9)
        ax[0, 2].set_title(f'{date_list[2].strftime("%Y-%m-%d")}', fontsize=9)
        ax[0, 3].set_title(f'{date_list[3].strftime("%Y-%m-%d")}', fontsize=9)

        plt.suptitle(f'Model Profile Comparisons\n'
                     f'Argo {argo} Surfacing: {t0.strftime("%Y-%m-%d")}\n',
                     fontsize=10,
                     y=0.98)
        plt.tight_layout()

        save_str = f'argo_{argo}-{t0.strftime("%Y-%m-%d")}-{len(date_list)}day-model-comparison.png'
        full_file = os.path.join('/Users/mikesmith/Documents/', save_str)

        plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

gofs.close()
rtofs.close()
