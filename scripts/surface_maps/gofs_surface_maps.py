import xarray as xr
import cartopy.crs as ccrs
import datetime as dt
import os
from src.plotting import plot_model_region
from src.common import limits
import pandas as pd


# Figures
# Surface fields: sst (rtofs, gofs, copernicus), ssh (gofs, copernicus), sss (rtofs, gofs, copernicus)
# Region: MAB, SAB, Caribbean, and GOM
# Models: GOFS, RTOFS, Copernicus
# Glider (active gliders - Whole Track), Argo (Last month), Drifters
# surface... 100-1000m range cross-section at 26N eddy

url = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/surface_maps/gofs/'
bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
days = 5
map_projection = ccrs.PlateCarree()
argo = True
gliders = True
dpi = 150

regions = limits('gofs')

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'gofs'
kwargs['transform'] = map_projection
kwargs['argo'] = argo
kwargs['gliders'] = gliders
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

if bathymetry:
    kwargs['bathy'] = xr.open_dataset(bathymetry)

# Get today and yesterday dates
today = dt.date.today()
yesterday = today - dt.timedelta(days=days)
tomorrow = today + dt.timedelta(days=1)
ranges = pd.date_range(yesterday, tomorrow, freq='6H')

with xr.open_dataset(url, drop_variables='tau') as gofs:
    gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

    # Get all the datetimes (match with rtofs dates of every 6 hours)
    ds = gofs.sel(time=ranges[ranges < pd.to_datetime(gofs.time.max().data)])

    for t in ds.time:
        print(f'Accessing GOFS: {str(t.dt.strftime("%Y-%m-%d %H:%M:%S").data)}')
        tds = ds.sel(time=t)  # Select the latest time

        # Loop through regions
        for region in regions.items():
            extent = region[1]['lonlat']
            print(f'Region: {region[0]}, Extent: {extent}')

            # subset dataset to the proper extents for each region
            sub = tds.sel(lon=slice(extent[0] + 359, extent[1] + 361), lat=slice(extent[2] - 1, extent[3] + 1))
            sub['lon'] = sub['lon'] - 360  # Convert model lon to glider lon
            plot_model_region(sub, region, **kwargs)
