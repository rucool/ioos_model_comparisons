import xarray as xr
import os
import datetime as dt
import numpy as np
from ioos_model_comparisons.platforms import get_argo_floats_by_time
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D
import math
import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.models import gofs as g, rtofs as r, cmems as c
from ioos_model_comparisons.calc import lon180to360, lon360to180

save_dir = conf.path_plots / 'profiles' / 'argo' / 'assimilation'
os.makedirs(save_dir, exist_ok=True)

# argo_id = ["4903250", "4901720"]
# argo_times = [dt.datetime(2022, 7, 18), dt.datetime(2022, 7, 5)]

# argo_id = ["4901720",]
# argo_times = [dt.datetime(2022, 7, 21),]

argo_dict = {
    '4903227': dt.datetime(2022, 10, 3),
}

days_to_check_for_argo_surfacing = 2
days_pre_surfacing = 1
days_post_surfacing = 2
depth = 400
interp = False # Interpolate models to exact point of argo float

# Plot configurations
dpi = 150
figsize = (16, 10) 
line = ('-', '--', '-.', ':') # 0600, 1200, 1800, 2400
labels = ['+0600', '+1200', '+1800', '+2400']
lines = [Line2D([0], [0], color='k', linewidth=3, linestyle=l) for l in line]
alpha = (1, .75, .5, .25)

# Models to include
rtofs_include = True
gofs_include = True
cmems_include = True

legend_elements = [Line2D([0], [0], color='b', lw=2, label='ARGO')]

# Open connection to each model
if gofs_include:
    gofs = g(rename=True).sel(depth=slice(0,400))[['temperature', 'salinity']]
    
    legend_elements.append(Line2D([0], [0], color='g', lw=2, label='GOFS'))

if rtofs_include:
    rtofs = r().sel(depth=slice(0,400))[['temperature', 'salinity']]
    
    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlon = rtofs.lon.data[0, :]
    rlat = rtofs.lat.data[:, 0]
    rx = rtofs.x.data
    ry = rtofs.y.data
    
    legend_elements.append(Line2D([0], [0], color='r', lw=2, label='RTOFS'))

if cmems_include:
    cmems = c(rename=True).sel(depth=slice(0,400))[['temperature', 'salinity']]
    legend_elements.append(Line2D([0], [0], color='m', lw=2, label='CMEMS'))

# Iterate through Argo dictionary
for key, value in argo_dict.items():
    time_end = value + dt.timedelta(days=1)
    time_start = time_end - dt.timedelta(days=days_to_check_for_argo_surfacing)

    argo_data = get_argo_floats_by_time(time_start=time_start,
                                        time_end=time_end,
                                        wmo_id=key,
                                        variables=[
                                            'pres', 
                                            'temp', 
                                            'psal']
                                        )

    if not argo_data.empty:
        # Iterate through Argo float profiles (there might be more than 1)
        for t_float, df in argo_data.groupby(level=1):
            # Create masks for dataframe
            mask_salinity = (np.abs(stats.zscore(df['psal (PSU)'])) < 3)
            mask_depth = df['pres (decibar)'] <= 400

            # Mask out the data
            df = df[mask_salinity & mask_depth] 

            lon = df['lon'].unique()[0]
            lat = df['lat'].unique()[0]

            ctime = t_float.date()

            # Get today and yesterday dates
            pre = ctime - dt.timedelta(days=days_pre_surfacing)
            post = ctime + dt.timedelta(days=days_post_surfacing+1)
            date_ranges = pd.date_range(pre, post, freq='24H', inclusive='left')
            time_ranges = pd.date_range(pre, post, freq='6H', inclusive='right')

            if rtofs_include:
                # interpolating lon and lat to x and y index of the rtofs grid
                rlonI = np.interp(lon, rlon, rx)
                rlatI = np.interp(lat, rlat, ry)

                rsub = rtofs.sel(time=time_ranges)
                if interp:
                    rsub = rsub.interp(
                        x=rlonI,
                        y=rlat
                    )
                else:
                    rsub = rsub.sel(
                        x=int(rlonI),
                        y=int(rlatI),
                        method='nearest'
                        )
                rsub.load()

            if gofs_include:
                gsub = gofs.sel(time=time_ranges)
                if interp:
                    gsub = gsub.interp(
                        lon=lon180to360(lon),
                        lat=lat,
                    )
                else:
                    gsub = gsub.sel(
                        lon=lon180to360(lon),
                        lat=lat,
                        method='nearest'
                        )
                gsub.load()
                
            if cmems_include:
                csub = cmems.sel(time=slice(pre, post))
                if interp:
                    csub = csub.interp(
                        lon=lon,
                        lat=lat
                    )
                else:
                    csub = csub.sel(
                        lon=lon,
                        lat=lat,
                        method='nearest'
                        )
                csub.load()

            # Initialize plot 
            fig, ax = plt.subplots(
                2, 4,
                figsize=figsize,
                constrained_layout=True,
                sharey=True
            )

            # Create empty list for temperature and salinity for calculating
            # x axis limits to be equal across all subplots
            temp_x = []
            salt_x = []

            # Iterate through each day
            for i, t_array in enumerate(np.split(time_ranges, 4)):
                # if gofs_include:
                #     gsub_t = gsub.sel(time=t_array, method='nearest')
                #     # gsub_t.load()

                # if rtofs_include:
                #     rsub_t = rsub.sel(time=t_array, method='nearest')
                #     # rsub_t.load()
                    
                n = 0
                day_str = t_array.strftime('%Y-%m-%d')

                for t in t_array:
                    # Argo Float
                    ax[0, i].plot(df['temp (degree_Celsius)'], df['pres (decibar)'], 'b-', )
                    ax[1, i].plot(df['psal (PSU)'], df['pres (decibar)'], 'b-', )
                    
                    if gofs_include:
                        g = gsub.sel(time=t, method='nearest')
                        ax[0, i].plot(g['temperature'].squeeze(), g['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])
                        ax[1, i].plot(g['salinity'].squeeze(), g['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])

                    if rtofs_include:
                        r = rsub.sel(time=t, method='nearest')
                        ax[0, i].plot(r['temperature'].squeeze(), r['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                        ax[1, i].plot(r['salinity'].squeeze(), r['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])

                    if cmems_include:
                        c = csub.sel(time=day_str[0], method='nearest')
                        ax[0, i].plot(c['temperature'].squeeze(), c['depth'].squeeze(), f'm{line[1]}', alpha=alpha[n])
                        ax[1, i].plot(c['salinity'].squeeze(), c['depth'].squeeze(), f'm{line[1]}', alpha=alpha[n])

                    # Temperature
                    temp_x.append(ax[0, i].xaxis.get_data_interval())
                    ax[0, i].set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')

                    # Salinity
                    salt_x.append(ax[1, i].xaxis.get_data_interval())
                    ax[1, i].set_xlabel('Salinity', fontsize=14, fontweight='bold')

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
                axs.set_ylim([400, 0])
                axs.grid(True, linestyle='--', linewidth=0.5)
                axs.tick_params(axis='both', labelsize=10)
                axs.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
                lh = axs.legend(lines, labels, loc='lower right', title="Hours", fontsize=8)
                axs.add_artist(lh)
                axs.legend(handles=legend_elements, loc='upper left', fontsize=10)
                for legobj in lh.legendHandles:
                    legobj.set_linewidth(1.0)

            ax[0, 0].set_title(f'{date_ranges[0]}\n-1 Day', fontsize=15, fontweight='bold')
            ax[0, 1].set_title(f'{date_ranges[1]}\n+0 Day', fontsize=15, fontweight='bold')
            ax[0, 2].set_title(f'{date_ranges[2]}\n+1 Day', fontsize=15, fontweight='bold')
            ax[0, 3].set_title(f'{date_ranges[3]}\n+2 Day', fontsize=15, fontweight='bold')

            plt.suptitle(f'Profile Comparisons\n'
                        f'Argo {key} Surfacing: {ctime}\n',
                        fontsize=18,
                        fontweight='bold',
                        y=0.98)

            plt.tight_layout()
            
            if interp:
                loc = 'interp'
            else:
                loc = 'nearest'
            save_str = f'argo_{key}-{ctime}-4day-profile-comparisons-{loc}.png'
            full_file = os.path.join(save_dir, save_str)

            plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()
