import xarray as xr
import os
from glob import glob
from hurricanes.limits import limits_regions
import datetime as dt
import numpy as np
from hurricanes.platforms import active_argo_floats
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.stats as stats
from hurricanes.plotting import plot_region, region_subplot
import cmocean
import matplotlib.patheffects as path_effects
# from scripts.harvest.grab_cmems import copernicusmarine_datastore as grab_cmems

url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/realtime/argo_profile_to_model_comparisons/'

# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/argo_profile_model_comparisons/'
# user = 'user'
# pw = 'password'

days = 8
dpi = 150

regions = limits_regions('rtofs', ['mab', 'gom', 'carib', 'wind', 'sab'])

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
rtofs = rtofs.sel(depth=slice(0, 400))

gofs = xr.open_dataset('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0', drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

now = pd.Timestamp.utcnow()
t0 = pd.to_datetime(now - pd.Timedelta(days, 'd'))
t1 = pd.to_datetime(now)

# cmems_salinity = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-so', user, pw)
# cmems_temp = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-thetao', user, pw)
# cmems = xr.merge([cmems_salinity, cmems_temp])
# cmems = cmems.sel(depth=slice(0, 400), time=slice(t0, t1),)

# Loop through regions
for region, values in regions.items():
    extent = values['lonlat']
    print(f'Region: {region}, Extent: {extent}')

    temp_save_dir = os.path.join(save_dir, '_'.join(region.split(' ')).lower())
    os.makedirs(temp_save_dir, exist_ok=True)

    floats = active_argo_floats(extent, t0, t1)

    temp_df = floats.loc[floats.groupby('platform_number')['time (UTC)'].idxmax()]

    region_limits = regions[region]

    vargs = {}
    vargs['vmin'] = values['temperature'][0]['limits'][0]
    vargs['vmax'] = values['temperature'][0]['limits'][1]
    vargs['transform'] = ccrs.PlateCarree()
    vargs['cmap'] = cmocean.cm.thermal
    vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], values['temperature'][0]['limits'][2])
    vargs['argo'] = temp_df

    try:
        vargs.pop('vmin'), vargs.pop('vmax')
    except KeyError:
        pass

    save_file = os.path.join(save_dir, "_".join(region.split(" ")).lower() ,f'argo_floats-{"_".join(region.split(" ")).lower()}-{t0.strftime("%Y-%m-%dT%H%M%SZ")}-to-{t1.strftime("%Y-%m-%dT%H%M%SZ")}')
    plot_region(rtofs['temperature'].isel(depth=0, time=0),
                extent,
                f'Argo Floats\n Active: {t0.strftime("%Y-%m-%dT%H:%M:%SZ")} to {t1.strftime("%Y-%m-%dT%H:%M:%SZ")}',
                save_file,
                **vargs
                )

    for float in floats.platform_number.unique():

        temp = floats[floats['platform_number'] == float]

        for t_float in temp['time (UTC)'].unique():
            temp_float_time = temp[temp['time (UTC)'] == t_float]
            filtered = temp_float_time[(np.abs(stats.zscore(temp_float_time['psal (PSU)'])) < 3)]  # filter salinity
            filtered = filtered[filtered['pres (decibar)'] <= 400]

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
            gofs_temp = gofs_temp.sel(depth=slice(0, 400))
            gofs_temp = gofs_temp.squeeze()

            # interpolating transect X and Y to lat and lon
            rtofslonIndex = np.round(np.interp(x, rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
            rtofslatIndex = np.round(np.interp(y, rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)

            rtofs_sub = rtofs.sel(
                time=t_float,
                X=rtofslonIndex,
                Y=rtofslatIndex,
                method='nearest',
            ).squeeze()
            rtofs_sub.load()

            # cmems_sub = cmems.sel(longitude=x, latitude=y, time=t_float, method='nearest')

            fig = plt.figure(figsize=(14, 8))
            plt.rcParams['figure.constrained_layout.use'] = True
            grid = plt.GridSpec(6, 10, hspace=0.2, wspace=0.2, figure=fig)
            ax1 = plt.subplot(grid[:, :3])  # Temperature
            ax2 = plt.subplot(grid[:, 4:7])  # Salinity
            ax3 = plt.subplot(grid[0, 7:])  # Plot Title
            ax4 = plt.subplot(grid[2:4, 7:], projection=vargs['transform'])  # Map
            ax5 = plt.subplot(grid[5, 7:])  # Legend for profile plots

            gofs_lon = str(round(gofs_temp.lon.data - 360, 2))
            gofs_lat = str(round(gofs_temp.lat.data - 0, 2))
            rtofs_lon = str(round(np.float64(rtofs_sub.lon.data), 2))
            rtofs_lat = str(round(np.float64(rtofs_sub.lat.data), 2))

            # Temperature
            ax1.plot(filtered['temp (degree_Celsius)'], filtered['pres (decibar)'],
                       'b-o',
                       label=f'{float} [{str(round(x, 2))}, {str(round(y, 2))}]')
            ax1.plot(gofs_temp['temperature'].squeeze(), gofs_temp['depth'].squeeze(),
                       'r-o',
                       label=f'GOFS [{ gofs_lon }, { gofs_lat }]')
            ax1.plot(rtofs_sub['temperature'].squeeze(), rtofs_sub['depth'].squeeze(),
                       'g-o',
                       label=f'RTOFS [{ rtofs_lon }, { rtofs_lat }]')
            # ax1.plot(cmems_sub['thetao'].squeeze(), cmems_sub['depth'].squeeze(),
            #            'm-o',
            #            label=f'CMEMS [{ cmems_sub.longitude.data.round(2) }, { cmems_sub.latitude.data.round(2) }]')

            ax1.set_ylim([400, 1])
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14)
            ax1.set_ylabel('Depth (m)', fontsize=14)

            # Salinity
            ax2.plot(filtered['psal (PSU)'], filtered['pres (decibar)'],
                       'b-o',
                       label=f'{float} [{str(round(x, 2))}, {str(round(y, 2))}]')
            ax2.plot(gofs_temp['salinity'].squeeze(), gofs_temp['depth'].squeeze(),
                       'r-o',
                       label=f'GOFS [{ gofs_lon }, { gofs_lat }]')
            ax2.plot(rtofs_sub['salinity'].squeeze(), rtofs_sub['depth'].squeeze(),
                       'g-o',
                       label=f'RTOFS [{ rtofs_lon }, { rtofs_lat }]')
            # ax2.plot(cmems_sub['so'].squeeze(), cmems_sub['depth'].squeeze(),
            #            'm-o',
            #            label=f'CMEMS [{ cmems_sub.longitude.data.round(2) }, { cmems_sub.latitude.data.round(2) }]')
            ax2.set_ylim([400, 1])
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14)
            ax2.set_ylabel('Depth (m)', fontsize=14)

            text = ax3.text(-0.125, 1.0, f'Argo #{float} Profile Comparisons\n'
                         f'ARGO: { t_float.strftime("%Y-%m-%d %H:%M:%SZ") }\n'
                         f'RTOFS: {pd.to_datetime(rtofs_sub.time.data)}\n'
                         f'GOFS: {pd.to_datetime(gofs_temp.time.data)}',
                            ha='left', va='top', size=18, fontweight='bold')

            text.set_path_effects([path_effects.Normal()])
            ax3.set_axis_off()

            dx = dy = 2.25  # Area around the point of interest.
            ax4 = region_subplot(fig, ax4, extent,
                                 transform=vargs['transform'],
                                 argo=filtered,
                                 ticks=None,
                                 colorbar=False)

            h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

            ax5.legend(h, l, ncol=1, loc='center', fontsize=12)
            ax5.set_axis_off()

            from hurricanes.plotting import map_add_ticks
            map_add_ticks(ax4, extent, fontsize=10)

            # t_str = t_float.strftime('%Y-%m-%d %H:%M:%SZ')

            save_str = f'{t_float.strftime("%Y-%m-%dT%H%M%SZ")}-argo_{float}-model-comparison.png'
            full_file = os.path.join(temp_save_dir, save_str)

            plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()

gofs.close()
rtofs.close()
