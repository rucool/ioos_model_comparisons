import cartopy.crs as ccrs
import xarray as xr
import os
from glob import glob
from hurricanes.plotting import plot_model_region_comparison
from hurricanes.limits import limits_regions
import datetime as dt
import numpy as np
from hurricanes.platforms import active_gliders, active_argo_floats
import pandas as pd

# Realtime Server Inputs
url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/realtime/surface_maps_comparison/'
bathymetry = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# Testing Inputs
# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps_comparison'
# bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

days = 0
projection = dict(map=ccrs.Mercator(), data=ccrs.PlateCarree())
argo = True
gliders = True
dpi = 150
search_hours = 24*5  #Hours back from timestamp to search for drifters/gliders

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

regions = limits_regions('gofs', ['yucatan', 'usvi', 'mab', 'gom', 'carib', 'wind', 'sab'])

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['transform'] = projection
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# Get today and yesterday dates
today = dt.date.today()

date_list = [today - dt.timedelta(days=x) for x in range(days+1)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

ranges = pd.date_range(today - dt.timedelta(days=days), today + dt.timedelta(days=1), freq='6H', closed='right')

gofs = xr.open_dataset(gofs_url, drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})
rtofs = xr.open_mfdataset(rtofs_files)
rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})

extent_list = []
for region in regions.keys():
    extent_list.append(regions[region]['lonlat'])

extent_df = pd.DataFrame(np.array(extent_list), columns=['lonmin', 'lonmax', 'latmin', 'latmax'])
global_extent = [extent_df.lonmin.min(), extent_df.lonmax.max(), extent_df.latmin.min(), extent_df.latmax.max()]

search_start = ranges[0] - dt.timedelta(hours=search_hours)
search_end = ranges[-1]

if argo:
    print(f'Download ARGO Data within {global_extent} between {search_start} and {search_end}')
    argo_data = active_argo_floats(global_extent, search_start, search_end)
else:
    argo_data = pd.DataFrame()

if gliders:
    print(f'Downloading glider data within {global_extent} between {search_start} and {search_end}')
    glider_data = active_gliders(global_extent, search_start, search_end)
else:
    glider_data = pd.DataFrame()

# Loop through regions
for region in regions.items():
    extent = region[1]['lonlat']
    print(f'Region: {region[0]}, Extent: {extent}')
    kwargs['colorbar'] = True

    if region[1]['currents']['bool']:
        kwargs['currents'] = region[1]['currents']

    if bathymetry:
        kwargs['bathy'] = bathy.sel(
            lon=slice(extent[0] - 1, extent[1] + 1),
            lat=slice(extent[2] - 1, extent[3] + 1)
        )
    extent = np.add(extent, [-1, 1, -1, 1]).tolist()

    # interpolating transect X and Y to lat and lon
    rtofslonIndex = np.round(np.interp(extent[:2], rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
    rtofslatIndex = np.round(np.interp(extent[2:], rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)
    rtofs_sub = rtofs.sel(
        X=slice(rtofslonIndex[0], rtofslonIndex[1]),
        Y=slice(rtofslatIndex[0], rtofslatIndex[1])
    )

    # subset dataset to the proper extents for each region
    gofs_sub = gofs.sel(
        lon=slice(extent[0] + 359, extent[1] + 361),
        lat=slice(extent[2] - 1, extent[3] + 1)
    )
    gofs_sub['lon'] = gofs_sub['lon'] - 360  # Convert model lon to glider lon

    for t in ranges:
        search_window_t0 = (t - dt.timedelta(hours=search_hours)).strftime('%Y-%m-%d %H:%M:%S')
        kwargs['t0'] = search_window_t0
        search_window_t1 = t.strftime('%Y-%m-%d %H:%M:%S')
        if not argo_data.empty:
            argo_lon = argo_data['longitude (degrees_east)']
            argo_lat = argo_data['latitude (degrees_north)']
            argo_region = argo_data[
                (extent[0] < argo_lon) & (argo_lon < extent[1]) & (extent[2] < argo_lat) & (argo_lat < extent[3])
            ]
            argo_region.sort_values(by=['time (UTC)'], inplace=True)
            argo_region.set_index('time (UTC)', inplace=True)
            argo_region = argo_region.loc[search_window_t0:search_window_t1]
            kwargs['argo'] = argo_region.reset_index()

        if not glider_data.empty:
            glider_lon = glider_data['longitude (degrees_east)']
            glider_lat = glider_data['latitude (degrees_north)']
            glider_region = glider_data[
                (extent[0] < glider_lon) & (glider_lon < extent[1]) & (extent[2] < glider_lat) & (glider_lat < extent[3])
                ]
            # glider_region.set_index('time (UTC)', inplace=True)
            glider_region = glider_region[
                (search_window_t0 < glider_region.index.get_level_values('time (UTC)'))
                &
                (glider_region.index.get_level_values('time (UTC)') < search_window_t1)
                ]
            kwargs['gliders'] = glider_region

        # kwargs['t0'] = t0
        try:
            plot_model_region_comparison(rtofs_sub.sel(time=t), gofs_sub.sel(time=t), region, t, **kwargs)
        except KeyError:
            continue