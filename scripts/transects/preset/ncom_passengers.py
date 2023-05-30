# %%
import xarray as xr
import datetime as dt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Time configurations
days = 5 # how many days we should plot in the past
freq = '24H' # how many hours should plots be spaced

# Transect Info
start = -63 - 9.694/60, 36
end = -63 - 9.694/60, 41
points = 2000 

# Depth Info (for vertical interpolation)
depth_min = 0 
depth_max = 1000
depth_spacing = 5 # meters

# Colorbar spacing
speed_min = -1.5
speed_max = 1.5
speed_interval = .1

# Save file path
sdir = Path('/Users/mikesmith/Documents/')
sname = 'ncom_transect'

# Path to model data
mdir = Path('/Volumes/home/coolgroup/passengers_2022/model_runs')
mdir = mdir / '*' / '*.nc' # Account for subdirectories in glob search

# Create dates that we want to plot
today = dt.date.today()
# today = dt.datetime(*dt.date.today().timetuple()[:3]) + dt.timedelta(hours=12)
date_end = today + dt.timedelta(days=2)
date_start = today - dt.timedelta(days=days)
date_list = pd.date_range(date_start, date_end, freq=freq)

# Calculate colorbar levels
levels = np.arange(speed_min, speed_max+speed_interval, speed_interval)

# Lazily open all the netcdfs into an xarray dataset
ds = xr.open_mfdataset(str(mdir))

# Calculate a line between start (lon, lat) and end (lon, lat) with n intervals in between
def calculate_transect(start, end, npts):
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    pts = g.inv_intermediate(start[0], start[1], end[0], end[1], npts) 
    return np.column_stack([pts.lons, pts.lats])

# Convert 360 lons to 180 lons
def lon360to180(array):
    array = np.array(array)
    return np.mod(array+180, 360)-180

# Calculate a line between the start and end points with n points in between (increase n for higher resolution interpolation)
pts = calculate_transect(start, end, points)
    
for t in date_list:
    # Select the time in the settings above
    tds = ds.sel(time=t)
    tds['lon'] = lon360to180(tds['lon'])

    # Interpolate model to transect
    ds_slice = tds.interp(
        lon=xr.DataArray(pts[:,0], dims="point"),
        lat=xr.DataArray(pts[:,1], dims="point"),
        depth=xr.DataArray(np.arange(depth_min, depth_max+depth_spacing, depth_spacing), dims="depth")
        ).load()

    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=(16,9))

    # Plot the u velocity
    ds_slice['water_u'].plot.contourf(x='lat', y='depth', extend='both', levels=levels, ax=ax[0])
    ax[0].set_title('Eastwater Water Velocity', fontweight='bold')
    ax[0].set_ylim([1000, 0])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Depth (m)', fontweight='bold')
    ax[0].locator_params(axis='x', nbins=12)

    # Plot the v velocity
    ds_slice['water_v'].plot.contourf(x='lat', y='depth', extend='both', levels=levels, ax=ax[1])
    ax[1].set_title('Northward Water Velocity', fontweight='bold')
    ax[1].set_ylim([1000, 0])
    ax[1].set_xlabel('Latitude', fontweight='bold')
    ax[1].set_ylabel('Depth (m)', fontweight='bold')
    ax[1].locator_params(axis='x', nbins=12)

    plt.suptitle(f"NCOM - {t.strftime('%Y-%m-%dT%H%M%SZ')}\nTransect: {np.round(start[0], 4)} W, {np.round(start[1], 4)} N to {np.round(end[0], 4)} W, {np.round(end[1], 4)} N", fontweight='bold', fontsize=20)

    # Save the figure
    plt.savefig(sdir / f"{sname}_{t.strftime('%Y-%m-%dT%H%M%SZ')}.png", dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor='white')
