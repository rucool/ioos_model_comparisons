import xarray as xr
import cartopy.crs as ccrs
import datetime as dt
from src.plotting import plot_model_region
from src.common import limits
import pandas as pd
import numpy as np
from src.platforms import active_gliders, active_argo_floats

# Figures
# Surface fields: sst (rtofs, gofs, copernicus), ssh (gofs, copernicus), sss (rtofs, gofs, copernicus)
# Region: MAB, SAB, Caribbean, and GOM
# Models: GOFS, RTOFS, Copernicus
# Glider (active gliders - Whole Track), Argo (Last month), Drifters
# surface... 100-1000m range cross-section at 26N eddy

url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'

# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps/'
# bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

save_dir = '/www/web/rucool/hurricane/model_comparisons/surface_maps/'
bathymetry = '/home/hurricaneadm/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

days = 0
map_projection = ccrs.PlateCarree()
argo = True
gliders = True
dpi = 150
search_hours = 24

regions = limits('gofs', ['mab', 'gom', 'carib', 'wind', 'sab'])

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'gofs'
kwargs['transform'] = map_projection
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# Get today and yesterday dates
today = dt.date.today()
time_start = today - dt.timedelta(days=days)
time_end = today + dt.timedelta(days=1)
ranges = pd.date_range(time_start, time_end, freq='6H')


with xr.open_dataset(url, drop_variables='tau') as gofs:
    gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

    # Get all the datetimes (match with rtofs dates of every 6 hours)
    ds = gofs.sel(time=ranges[ranges < pd.to_datetime(gofs.time.max().data)])

    for t in ds.time:
        print(f'Accessing GOFS: {str(t.dt.strftime("%Y-%m-%d %H:%M:%S").data)}')
        tds = ds.sel(time=t)  # Select the latest time

        t0 = pd.to_datetime(tds.time.data - np.timedelta64(search_hours, 'h'))
        t1 = pd.to_datetime(tds.time.data)

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

            # extent = np.add(extent, [-1, 1, -1, 1]).tolist()
            # print(f'Region: {region[0]}, Extent: {extent}')

            # subset dataset to the proper extents for each region
            sub = tds.sel(
                lon=slice(extent[0] + 359, extent[1] + 361),
                lat=slice(extent[2] - 1, extent[3] + 1)
            )
            sub['lon'] = sub['lon'] - 360  # Convert model lon to glider lon
            plot_model_region(sub, region, t1, **kwargs)
