#%%
import datetime as dt

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ioos_model_comparisons.calc import depth_bin
from ioos_model_comparisons.models import rtofs
from ioos_model_comparisons.platforms import get_glider_by_id
from pathlib import Path

#%%
# Standard settings
id = 'ru30-20221011T1527'
depth_min = 0
depth_max = 120

# id = 'ru23-20221011T1759'
# depth_min = 0
# depth_max = 100

# id = 'sbu01-20221011T1852'
# depth_min = 0
# depth_max = 180

sdir = Path("/Users/mikesmith/Documents/")
dpi = 150
figsize = (16,9)

# Interpolation settings
depth_freq = 1
time_freq = '15Min'

# Colormaps
cmap_temp = cmocean.cm.thermal
cmap_salt = cmocean.cm.haline

# Contour settings
levels_thermal = np.arange(12, 22, 1) # Set to None for automatic contours
levels_haline = np.arange(32.3, 35.1, .1) # Set to None for automatic contours

# Grab glider by id from NGDAC
df = get_glider_by_id(id).reset_index()

# Rename all the variables
df = df.rename({
    "time (UTC)": "time",
    "longitude (degrees_east)": "lon",
    "latitude (degrees_north)": "lat",
    "pressure (decibar)": "pressure",
    "temperature (degrees_C)": "temperature",
    "depth (m)": "depth",
    "salinity (1)": "salinity",
    "conductivity (S m-1)": "conductivity",
    "density (kg m-3)": "density",
}, axis=1)
df.drop(['pressure', 'conductivity'], axis=1, inplace=True)
# df.head()

#%%
# Groupby time and resample downsample to an hourly frequency (for track)
tdf = df.groupby(['time']).mean()
tdf = tdf.resample(time_freq).mean().interpolate('linear')
# tdf.head()

#%%
# Save glider time, lon, and lat to their own variables
gtime = tdf.index
glon = tdf.lon
glat = tdf.lat

#%% 
# Convert glider dataframe to xarray dataset

# Groupby 'time' and apply the your choice of depth gridding: depth_bin or 
# depth_interpolate. 
df_interp = df.groupby(['time']).apply(depth_bin, 
                                       depth_min=depth_min,
                                       depth_max=depth_max, 
                                       stride=depth_freq)

# The df.to_xarray() method will automatically convert all the multi-indexes to
# the appropriate dimensions in your xarray dataset
gds = df_interp.drop(['depth'], axis=1).to_xarray()
# gds = gds.resample(time='15Min').interpolate().interpolate_na(dim='depth')

# #%%
# # Save glider time, lon, and lat to their own variables
# gtime = tdf.index
# glon = tdf.lon
# glat = tdf.lat

#%%
# Load RTOFS model
ds = rtofs().sel(depth=slice(depth_min, depth_max))

# Grab lons, lats, x, and y dimensions as separate variables. Saves time later
lons = ds.lon.data[0, :]
lats = ds.lat.data[:, 0] 
x = ds.x.data
y = ds.y.data

#%%
# Interpolate glider points to x and y points in rtofs
lonI = np.interp(glon, lons, x)
latI = np.interp(glat, lats, y)

depths = np.arange(0, 101, 1)

#%%
# Select nearest points for the track in space and time
ds_track = ds.interp(
    time=xr.DataArray(gtime, dims='point'),
    x=xr.DataArray(lonI, dims='point'),
    y=xr.DataArray(latI, dims='point'),
    depth=xr.DataArray(depths, dims='depth')
)
# ds_track

#%%

# Plot the temperature transect comparisons
fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

# Plot the glider cross section
gds['temperature'].plot.contourf(x='time', y='depth', cmap=cmap_temp, levels=levels_thermal, extend='both', ax=ax[0])
ax[0].set_ylim([100, 0])
ax[0].set_title(f'Glider ID: {id}', fontweight='bold', fontsize=18)
ax[0].set_ylabel('Depth (m)', fontweight='bold', fontsize=14)
ax[0].set_xlabel('')

# Plot the model cross section
ds_track['temperature'].plot.contourf(x='time', y='depth', cmap=cmap_temp, levels=levels_thermal, extend='both', ax=ax[1])
ax[1].set_ylim([100, 0])
ax[1].set_title('RTOFS', fontweight='bold', fontsize=18)
ax[1].set_ylabel('Depth (m)', fontweight='bold', fontsize=14)
ax[1].set_xlabel('')

sname = sdir / f"{id}-synoptic_timeseries-temperature_rtofs.png"
plt.savefig(sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
plt.close()

# Plot the salinity transect comparisons
fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

# Plot the glider cross section
gds['salinity'].plot.contourf(x='time', y='depth', cmap=cmap_salt, levels=levels_haline, extend='both', ax=ax[0])
ax[0].set_ylim([100, 0])
ax[0].set_title(f'Glider ID: {id}', fontweight='bold', fontsize=18)
ax[0].set_ylabel('Depth (m)', fontweight='bold', fontsize=14)
ax[0].set_xlabel('')

# Plot the model cross section
ds_track['salinity'].plot.contourf(x='time', y='depth', cmap=cmap_salt, levels=levels_haline, extend='both', ax=ax[1])
ax[1].set_ylim([100, 0])
ax[1].set_title('RTOFS', fontweight='bold', fontsize=18)
ax[1].set_ylabel('Depth (m)', fontweight='bold', fontsize=14)
ax[1].set_xlabel('')

sname = sdir / f"{id}-synoptic_timeseries-salinity_rtofs.png"
plt.savefig(sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
plt.close()