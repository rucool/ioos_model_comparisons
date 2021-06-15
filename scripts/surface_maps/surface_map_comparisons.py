import cartopy.crs as ccrs
import xarray as xr
import os
from glob import glob
from src.plotting import plot_model_regions_comparison
from src.common import limits
import datetime as dt
import numpy as np
from src.plotting import active_gliders, active_argo_floats
import pandas as pd

# Realtime Server Inputs
url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/surface_maps_comparison/'
bathymetry = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# Testing Inputs
# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps_comparison'
# bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

days = 2
map_projection = ccrs.PlateCarree()
argo = True
gliders = True
dpi = 150
search_hours = 24  #Hours back from timestamp to search for drifters/gliders

gofs_url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

regions = limits('rtofs', ['mab', 'gom', 'carib', 'wind', 'sab'])

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['transform'] = map_projection
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# Get today and yesterday dates
date_list = [dt.date.today() - dt.timedelta(days=x) for x in range(days)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

gofs = xr.open_dataset(gofs_url, drop_variables='tau')
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

for f in rtofs_files:
    print(f)
    try:
        with xr.open_dataset(f) as rtofs:
            rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})

            t0 = pd.to_datetime(rtofs.time.data[0] - np.timedelta64(search_hours, 'h'))
            t1 = pd.to_datetime(rtofs.time.data[0])

            # Loop through regions
            for region in regions.items():
                extent = region[1]['lonlat']
                print(f'Region: {region[0]}, Extent: {extent}')

                if argo:
                    kwargs['argo'] = active_argo_floats(extent, t0, t1)

                if gliders:
                    kwargs['gliders'] = active_gliders(extent, t0, t1)

                if bathy:
                    kwargs['bathy'] = bathy.sel(
                        lon=slice(extent[0] - 1, extent[1] + 1),
                        lat=slice(extent[2] - 1, extent[3] + 1)
                    )
                extent = np.add(extent, [-1, 1, -1, 1]).tolist()
                print(f'Region: {region[0]}, Extent: {extent}')

                # interpolating transect X and Y to lat and lon
                rtofslonIndex = np.round(np.interp(extent[:2], rtofs.lon.data[0, :], np.arange(0, len(rtofs.lon.data[0, :])))).astype(int)
                rtofslatIndex = np.round(np.interp(extent[2:], rtofs.lat.data[:, 0], np.arange(0, len(rtofs.lat.data[:, 0])))).astype(int)
                rtofs_sub = rtofs.sel(
                    X=slice(rtofslonIndex[0], rtofslonIndex[1]),
                    Y=slice(rtofslatIndex[0], rtofslatIndex[1])
                )

                # subset dataset to the proper extents for each region
                gofs_sub = gofs.sel(
                    time=t1,
                    lon=slice(extent[0] + 359, extent[1] + 361),
                    lat=slice(extent[2] - 1, extent[3] + 1)
                )
                gofs_sub['lon'] = gofs_sub['lon'] - 360  # Convert model lon to glider lon

                plot_model_regions_comparison(rtofs_sub, gofs_sub, region, t1, **kwargs)
    except OSError:
        continue
