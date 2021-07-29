import xarray as xr
import os
from glob import glob
from src.common import limits
import datetime as dt
import numpy as np
from src.platforms import active_argo_floats
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats

url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/realtime/argo_profile_to_model_comparisons/'

# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/argo_profile_model_comparisons/'

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

days = 10
dpi = 150

regions = limits('rtofs', ['mab', 'gom', 'carib', 'wind', 'sab'])

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

# Get today and yesterday dates
date_list = [dt.date.today() - dt.timedelta(days=x+1) for x in range(days)]
date_list.insert(0, dt.date.today())
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])
rtofs = xr.open_mfdataset(rtofs_files)
rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})

gofs = xr.open_dataset(gofs_url, drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

now = pd.Timestamp.utcnow()
t0 = pd.to_datetime(now - pd.Timedelta(days, 'd'))
t1 = pd.to_datetime(now)

LAND = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m',
    edgecolor='face',
    facecolor='tan'
)

state_lines = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none'
)

# Loop through regions
for region in regions.items():
    extent = region[1]['lonlat']
    print(f'Region: {region[0]}, Extent: {extent}')

    temp_save_dir = os.path.join(save_dir, '_'.join(region[0].split(' ')).lower())
    os.makedirs(temp_save_dir, exist_ok=True)

    floats = active_argo_floats(extent, t0, t1)

    temp_df = floats.loc[floats.groupby('platform_number')['time (UTC)'].idxmax()]

    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )
    # Plot title
    plt.title(f'Argo Floats\n Active: {t0.strftime("%Y-%m-%dT%H:%M:%SZ")} to {t1.strftime("%Y-%m-%dT%H:%M:%SZ")}')
    # Axes properties and features
    ax.set_extent(extent)
    ax.add_feature(LAND, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_lines, zorder=11, edgecolor='black')

    for index, row in temp_df.iterrows():
        ax.plot(row['longitude (degrees_east)'], row['latitude (degrees_north)'],
                marker='o', label=row['platform_number'], transform=ccrs.PlateCarree())
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,)

    # Gridlines and grid labels
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=.5,
        color='black',
        alpha=0.25,
        linestyle='--'
    )

    gl.xlabels_top = gl.ylabels_right = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    save_file = os.path.join(save_dir, f'argo_floats-{"_".join(region[0].split(" ")).lower()}-{t0.strftime("%Y-%m-%dT%H%M%SZ")}-to-{t1.strftime("%Y-%m-%dT%H%M%SZ")}')
    plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    for float in floats.platform_number.unique():

        temp = floats[floats['platform_number'] == float]

        for t_float in temp['time (UTC)'].unique():
            temp_float_time = temp[temp['time (UTC)'] == t_float]
            filtered = temp_float_time[(np.abs(stats.zscore(temp_float_time['psal (PSU)'])) < 3)]  # filter salinity

            # temp = temp.sort_values(by=['pres (decibar)'])
            x = filtered['longitude (degrees_east)'].unique()[-1]
            y = filtered['latitude (degrees_north)'].unique()[-1]

            # Conversion from longitude to GOFS convention
            if x < 0:
                x_gofs = 360 + x
            else:
                x_gofs = x

            gofs_temp = gofs.sel(
                time=t_float,
                lon=x_gofs,
                lat=y,
                method='nearest')

            gofs_temp = gofs_temp.squeeze()

            # interpolating transect X and Y to lat and lon
            rtofslonIndex = np.round(np.interp(x, rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
            rtofslatIndex = np.round(np.interp(y, rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)

            rtofs_sub = rtofs.sel(
                time=t_float,
                X=rtofslonIndex,
                Y=rtofslatIndex,
                method='nearest'
            )
            rtofs_sub = rtofs_sub.squeeze()
            rtofs_sub.load()

            fig, ax = plt.subplots(
                1, 2,
                sharey=True
            )

            gofs_lon = str(round(gofs_temp.lon.data - 360, 2))
            gofs_lat = str(round(gofs_temp.lat.data - 0, 2))
            rtofs_lon = str(round(np.float64(rtofs_sub.lon.data), 2))
            rtofs_lat = str(round(np.float64(rtofs_sub.lat.data), 2))

            # Temperature
            ax[0].plot(filtered['temp (degree_Celsius)'], filtered['pres (decibar)'],
                       'b-',
                       label=f'{float} [{str(round(x, 2))}, {str(round(y, 2))}]')
            ax[0].plot(gofs_temp['temperature'].squeeze(), gofs_temp['depth'].squeeze(),
                       'r-',
                       label=f'GOFS [{ gofs_lon }, { gofs_lat }]')
            ax[0].plot(rtofs_sub['temperature'].squeeze(), rtofs_sub['depth'].squeeze(),
                       'g-',
                       label=f'RTOFS [{ rtofs_lon }, { rtofs_lat }]')
            ax[0].set_ylim([400, 1])
            ax[0].set_xlim([8, 30])
            ax[0].grid(True, linestyle='--', linewidth=.5)
            ax[0].legend(loc='upper left', fontsize=6)
            plt.setp(ax[0], ylabel='Depth (m)', xlabel='Temperature (˚C)')

            # Salinity
            ax[1].plot(filtered['psal (PSU)'], filtered['pres (decibar)'],
                       'b-',
                       label=f'{float} [{str(round(x, 2))}, {str(round(y, 2))}]')
            ax[1].plot(gofs_temp['salinity'].squeeze(), gofs_temp['depth'].squeeze(),
                       'r-',
                       label=f'GOFS [{ gofs_lon }, { gofs_lat }]')
            ax[1].plot(rtofs_sub['salinity'].squeeze(), rtofs_sub['depth'].squeeze(),
                       'g-',
                       label=f'RTOFS [{ rtofs_lon }, { rtofs_lat }]')
            ax[1].set_ylim([400, 1])
            ax[1].grid(True, linestyle='--', linewidth=.5)
            ax[1].legend(loc='upper left', fontsize=6)
            plt.setp(ax[1], ylabel='Depth (m)', xlabel='Salinity (psu)')

            t_str = t_float.strftime('%Y-%m-%d %H:%M:%SZ')
            plt.suptitle(f'Argo #{float} Profile Comparisons\n'
                         f'ARGO: { t_float.strftime("%Y-%m-%d %H:%M:%SZ") }\n'
                         f'RTOFS: {pd.to_datetime(rtofs_sub.time.data)}\n'
                         f'GOFS: {pd.to_datetime(gofs_temp.time.data)}', fontsize=8)
            save_str = f'{t_float.strftime("%Y-%m-%dT%H%M%SZ")}-argo_{float}-model-comparison.png'

            full_file = os.path.join(temp_save_dir, save_str)
            plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()

gofs.close()
rtofs.close()