import cartopy.crs as ccrs
import xarray as xr
import os
from glob import glob
from hurricanes.plotting import plot_model_region
from hurricanes.limits import limits_regions
import datetime as dt
import numpy as np
from hurricanes.platforms import active_gliders, active_argo_floats
import pandas as pd

# Figures
# Surface fields: sst (rtofs, gofs, copernicus), ssh (gofs, copernicus), sss (rtofs, gofs, copernicus)
# Region: MAB, SAB, Caribbean, and GOM
# Models: GOFS, RTOFS, Copernicus
# Glider (active gliders - Whole Track), Argo (Last month), Drifters
# surface... 100-1000m range cross-section at 26N eddy

# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps/'
# bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/realtime/surface_maps/'
bathymetry = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

days = 2
projection = dict(map=ccrs.Mercator(), data=ccrs.PlateCarree())
argo = False
gliders = False
dpi = 150
search_hours = 24*5  #Hours back from timestamp to search for drifters/gliders=

regions = limits_regions('rtofs', ['north_atlantic'])

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['transform'] = projection
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# Get today and yesterday dates
today = dt.date.today()
date_list = [today - dt.timedelta(days=x) for x in range(days)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

for f in rtofs_files:
    print(f)
    try:
        with xr.open_dataset(f) as ds:
            ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat', 'MT': 'time', 'Depth': 'depth'})
            lat = ds.lat.data
            lon = ds.lon.data

            t0 = pd.to_datetime(ds.time.data[0] - np.timedelta64(search_hours, 'h'))
            t1 = pd.to_datetime(ds.time.data[0])
            kwargs['t0'] = t0

            # Loop through regions
            for region in regions.items():
                extent = region[1]['lonlat']

                if argo:
                    kwargs['argo'] = active_argo_floats(extent, t0, t1)

                if gliders:
                    kwargs['gliders'] = active_gliders(extent, t0, t1)

                if bathy:
                    kwargs['bathy'] = bathy.sel(
                        lon=slice(extent[0]-1, extent[1]+1),
                        lat=slice(extent[2]-1, extent[3]+1)
                    )

                if region[1]['currents']['bool']:
                    kwargs['currents'] = region[1]['currents']

                # kwargs['transform'] = projection

                extent = np.add(extent, [-1, 1, -1, 1]).tolist()
                print(f'Region: {region[0]}, Extent: {extent}')

                # interpolating transect X and Y to lat and lon
                lonIndex = np.round(np.interp(extent[:2], lon[0, :], np.arange(0, len(lon[0, :])))).astype(int)
                latIndex = np.round(np.interp(extent[2:], lat[:, 0], np.arange(0, len(lat[:, 0])))).astype(int)
                sub = ds.sel(
                    X=slice(lonIndex[0], lonIndex[1]),
                    Y=slice(latIndex[0], latIndex[1])
                )
                plot_model_region(sub, region, t1, **kwargs)
    except OSError:
        continue