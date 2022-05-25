import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime as dt


url = '/Users/mikesmith/Desktop/test/rtofs.20210304/rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/profiles/'

temp = [6, 31]
salinity = [33, 37.1]

x = -87.25
y = 26

extent = [-100, -80, 18, 32]

os.makedirs(save_dir, exist_ok=True)

# for f in glob.glob(os.path.join(url, '*.nc')):
# try:
with xr.open_dataset(url) as ds:
    latRTOFS = np.asarray(ds.Latitude[:])
    lonRTOFS = np.asarray(ds.Longitude[:])

    # interpolating transect X and Y to lat and lon
    oklonRTOFS = np.round(np.interp(x, lonRTOFS[0, :], np.arange(0, len(lonRTOFS[0, :])))).astype(int)
    oklatRTOFS = np.round(np.interp(y, latRTOFS[:, 0], np.arange(0, len(latRTOFS[:, 0])))).astype(int)

    lon_RTOFS = lonRTOFS[0, oklonRTOFS]
    lat_RTOFS = latRTOFS[oklatRTOFS, 0]
    sst_RTOFS = np.asarray(ds.variables['temperature'][0, 0, oklatRTOFS, oklonRTOFS])
    sss_RTOFS = np.asarray(ds.variables['salinity'][0, 0, oklatRTOFS, oklonRTOFS])

    target_tempRTOFS = np.empty((len(ds['Depth']), 1))
    target_tempRTOFS[:] = np.nan
    target_saltRTOFS = np.empty((len(ds['Depth']), 1))
    target_saltRTOFS[:] = np.nan

    target_tempRTOFS = ds['temperature'][0, :, oklatRTOFS, oklonRTOFS]
    target_saltRTOFS = ds['salinity'][0, :, oklatRTOFS, oklonRTOFS]

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].plot(target_tempRTOFS, target_tempRTOFS['Depth'])
    axs[0].set_title('Temperature', size=10)
    axs[0].grid(True, linestyle='--', linewidth=.5)
    axs[0].set_xlim([5, 30])

    axs[1].plot(target_saltRTOFS, target_saltRTOFS['Depth'])
    axs[1].set_title('Salinity', size=10)
    axs[1].grid(True, linestyle='--', linewidth=.5)
    axs[1].set_xlim([34.5, 37])

    plt.ylim([1000,1])
    plt.suptitle(f'RTOFS Profile ({ds["MT"].dt.strftime("%Y-%m-%dT%H%M%SZ").data[0]}) @ {y}N, -{x}W', size=12)
    sname = f'rtofs-{ds["MT"].dt.strftime("%Y-%m-%dT%H%M%SZ").data[0]}_profile.png'
    plt.savefig(os.path.join(save_dir, sname), bbox_inches='tight', pad_inches=0.1, dpi=300)
# except OSError:
#     continue


# Conversion from glider longitude and latitude to GOFS convention
if x < 0:
    x_gofs = 360 + x
else:
    x_gofs = x

date = dt.date.today()
date = date.strftime('%Y-%m-%dT%H:%M:%S')


with xr.open_dataset('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0', drop_variables='tau') as ds:
    dst = ds.sel(time=date, lon=slice(extent[0]+360, extent[1]+360), lat=slice(extent[2]-1, extent[3]+1))  # dst = ds[variables].sel(time=slice('2021-02-01', '2021-02-18'))

    latGOFS = np.asarray(dst['lat'][:])
    lonGOFS = np.asarray(dst['lon'][:])
    depthGOFS = np.asarray(dst['depth'][:])

    # interpolating transect X and Y to lat and lon
    oklonGOFS = np.round(np.interp(x_gofs, lonGOFS, np.arange(0, len(lonGOFS)))).astype(int)
    oklatGOFS = np.round(np.interp(y, latGOFS, np.arange(0, len(latGOFS)))).astype(int)

    # GOFS = dst.sel(time=t)  # Select the latest time

    target_tempGOFS = np.empty((len(depthGOFS), 1))
    target_tempGOFS[:] = np.nan
    target_saltGOFS = np.empty((len(depthGOFS), 1))
    target_saltGOFS[:] = np.nan

    target_tempGOFS = dst['water_temp'][:, oklatGOFS, oklonGOFS]
    target_saltGOFS = dst['salinity'][:, oklatGOFS, oklonGOFS]

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].plot(target_tempGOFS, target_tempGOFS['depth'])
    axs[0].set_title('Temperature', size=10)
    axs[0].grid(True, linestyle='--', linewidth=.5)
    axs[0].set_xlim([5, 30])

    axs[1].plot(target_saltGOFS, target_saltGOFS['depth'])
    axs[1].set_title('Salinity', size=10)
    axs[1].grid(True, linestyle='--', linewidth=.5)
    axs[1].set_xlim([34.5, 37])

    plt.ylim([1000, 1])
    plt.suptitle(f'GOFS Profile ({dst["time"].dt.strftime("%Y-%m-%dT%H%M%SZ").data}) @ {y}N, {x}W', size=12)
    sname = f'gofs-{dst["time"].dt.strftime("%Y-%m-%dT%H%M%SZ").data}_profile.png'
    plt.savefig(os.path.join(save_dir, sname), bbox_inches='tight', pad_inches=0.1, dpi=300)

fig, ax = plt.subplots(
    figsize=(4, 6)
)
tstr = target_tempRTOFS["MT"].dt.strftime("%Y-%m-%dT%H%M%S").data

# Interpolate ROTFS Temperature to GOFS Depths
tds = target_tempRTOFS.interp(Depth=target_tempGOFS['depth'].data)

# Plot title
plt.title(f'GOFS vs RTOFS - Sea Water Temperature\n {tstr} @ {y}N, {x}W')
plt.plot(target_tempGOFS, target_tempGOFS['depth'], label='GOFS')
plt.plot(tds, target_tempGOFS['depth'], label='RTOFS')
plt.plot((tds.data - target_tempGOFS.data), target_tempGOFS['depth'], label='Difference')
plt.ylim([1000,1])
plt.legend()
plt.xlabel('Temperature (degrees C)')
plt.ylabel('Depth (m)')
plt.grid(linestyle='--', linewidth=.5)

sname = f'gofs_vs_rtofs_water_temperature_profile-{tstr}.png'
plt.savefig(os.path.join(save_dir, sname), dpi=300, bbox_inches='tight', pad_inches=0.1)