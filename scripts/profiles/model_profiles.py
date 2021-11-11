from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt
from hurricanes.common import list_to_dataframe

# url = '/home/hurricaneadm/data/rtofs/'
# save_dir = '/www/web/rucool/hurricane/model_comparisons/profiles/'

url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/profiles/'

days = 2

# adjust variable limits
temp_limits = [5, 30]
salinity_limits = [33, 37.1]


# set lon, lat for profile
profiles = dict(
    loop_current_eddy=dict(x=-87.25, y=26),
    cold_water_arm_in_western_gulf=dict(x=-93.5, y=27),
    green_blob=dict(x=-92.4, y=26.4)
)

# subset model to this area
extent = [-100, -80, 18, 32]

os.makedirs(save_dir, exist_ok=True)

# Get today and yesterday dates
date_list = [dt.datetime.today() - dt.timedelta(days=x) for x in range(days)]

rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = list_to_dataframe([inner for outer in rtofs_files for inner in outer])

# Open GOFS (don't need to reopen this for every single time through the loop)
gofs = xr.open_dataset('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0', drop_variables='tau')  # Open GOFS
gofs = gofs.sel(lon=slice(extent[0] + 360, extent[1] + 360), lat=slice(extent[2] - 1, extent[3] + 1))

# Iterate through times
for t in rtofs_files.itertuples():
    rtofs = xr.open_dataset(t.file)  # Open RTOFS at specified timestamp

    latRTOFS = np.asarray(rtofs.Latitude[:])
    lonRTOFS = np.asarray(rtofs.Longitude[:])

    latGOFS = np.asarray(gofs['lat'][:])
    lonGOFS = np.asarray(gofs['lon'][:])
    depthGOFS = np.asarray(gofs['depth'][:])

    for key in profiles.keys():
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        x = profiles[key]['x']
        y = profiles[key]['y']

        # Conversion from longitude to GOFS convention
        if x < 0:
            x_gofs = 360 + x
        else:
            x_gofs = x

        # RTOFS
        # interpolating transect X and Y to lat and lon
        oklonRTOFS = np.round(np.interp(x, lonRTOFS[0, :], np.arange(0, len(lonRTOFS[0, :])))).astype(int)
        oklatRTOFS = np.round(np.interp(y, latRTOFS[:, 0], np.arange(0, len(latRTOFS[:, 0])))).astype(int)

        lon_RTOFS = lonRTOFS[0, oklonRTOFS]
        lat_RTOFS = latRTOFS[oklatRTOFS, 0]

        # preallocate arrays for temperature and salinity
        target_tempRTOFS = np.full([len(rtofs['Depth']), 1], np.nan)
        target_saltRTOFS = np.full([len(rtofs['Depth']), 1], np.nan)

        target_tempRTOFS = rtofs['temperature'][0, :, oklatRTOFS, oklonRTOFS]
        target_saltRTOFS = rtofs['salinity'][0, :, oklatRTOFS, oklonRTOFS]

        # GOFS
        try:
            dst = gofs.sel(time=t.Index)
        except KeyError:
            continue

        # interpolating transect X and Y to lat and lon
        oklonGOFS = np.round(np.interp(x_gofs, lonGOFS, np.arange(0, len(lonGOFS)))).astype(int)
        oklatGOFS = np.round(np.interp(y, latGOFS, np.arange(0, len(latGOFS)))).astype(int)

        target_tempGOFS = np.full([len(depthGOFS), 1], np.nan)
        target_saltGOFS = np.full([len(depthGOFS), 1], np.nan)

        target_tempGOFS = dst['water_temp'][:, oklatGOFS, oklonGOFS]
        target_saltGOFS = dst['salinity'][:, oklatGOFS, oklonGOFS]

        # Plot RTOFS
        fig, axs = plt.subplots(1, 2, sharey=True)

        axs[0].plot(target_tempRTOFS, target_tempRTOFS['Depth'])
        axs[0].set_title('Temperature', size=10)
        axs[0].grid(True, linestyle='--', linewidth=.5)
        axs[0].set_xlim(temp_limits)
        plt.setp(axs[0], ylabel='Depth (m)', xlabel='Temperature (˚C)')

        axs[1].plot(target_saltRTOFS, target_saltRTOFS['Depth'])
        axs[1].set_title('Salinity', size=10)
        axs[1].grid(True, linestyle='--', linewidth=.5)
        axs[1].set_xlim(salinity_limits)
        plt.setp(axs[1], ylabel='Depth (m)', xlabel='Salinity (psu)')

        plt.ylim([300, 1])
        rtofs_time = target_tempRTOFS.MT.dt.strftime("%Y-%m-%dT%H%M%SZ").data
        plt.suptitle(f'RTOFS Profile ({rtofs_time}) @ {y}N, {x}W', size=12)
        sname = f'{key}_rtofs_{rtofs_time}_profile.png'
        plt.savefig(os.path.join(save_dir, key, sname), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        # Plot GOFS
        fig, axs = plt.subplots(1, 2, sharey=True)

        axs[0].plot(target_tempGOFS, target_tempGOFS['depth'])
        axs[0].set_title('Temperature', size=10)
        axs[0].grid(True, linestyle='--', linewidth=.5)
        axs[0].set_xlim(temp_limits)
        plt.setp(axs[0], ylabel='Depth (m)', xlabel='Temperature (˚C)')

        axs[1].plot(target_saltGOFS, target_saltGOFS['depth'])
        axs[1].set_title('Salinity', size=10)
        axs[1].grid(True, linestyle='--', linewidth=.5)
        axs[1].set_xlim(salinity_limits)
        plt.setp(axs[1], ylabel='Depth (m)', xlabel='Salinity (psu)')

        plt.ylim([300, 1])
        gofs_time = dst["time"].dt.strftime("%Y-%m-%dT%H%M%SZ").data
        plt.suptitle(f'GOFS Profile ({gofs_time}) @ {y}N, {x}W', size=12)
        sname = f'{key}_gofs-{gofs_time}_profile.png'
        plt.savefig(os.path.join(save_dir, key, sname), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()


        # Compare the two model profiles
        fig, axs = plt.subplots(1, 2, sharey=True)

        # Interpolate ROTFS Temperature to GOFS Depths
        tds = target_tempRTOFS.interp(Depth=target_tempGOFS['depth'].data)
        tstr = target_tempRTOFS["MT"].dt.strftime("%Y-%m-%dT%H%M%S").data

        # Plot title
        plt.suptitle(f'GOFS vs RTOFS - Sea Water Temperature\n ({tstr}) @ {y}N, {x}W')
        axs[0].plot(target_tempGOFS, target_tempGOFS['depth'], label='GOFS')
        axs[0].plot(target_tempRTOFS, target_tempRTOFS['Depth'], label='RTOFS')
        # axs[0].plot((tds.data - target_tempGOFS.data), target_tempGOFS['depth'], label='Difference')
        axs[0].grid(True, linestyle='--', linewidth=.5)
        axs[0].legend()
        axs[0].set_xlim([10, 32])

        plt.setp(axs[0], ylabel='Depth (m)', xlabel='Temperature (˚C )')

        axs[1].plot((tds.data - target_tempGOFS.data), target_tempGOFS['depth'], label='Difference')
        axs[1].grid(True, linestyle='--', linewidth=.5)
        axs[1].set_xlim([-3, 3])
        plt.setp(axs[1], xlabel='Difference (˚C)')

        plt.ylim([300, 1])

        sname = f'{key}_gofs_vs_rtofs_water_temperature_profile-{tstr}.png'
        plt.savefig(os.path.join(save_dir, key, sname), dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        rtofs.close()
