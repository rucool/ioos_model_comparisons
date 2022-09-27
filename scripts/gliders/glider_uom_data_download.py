#!/usr/bin/env python

# # import matplotlib.dates as mdates
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from erddapy import ERDDAP
from upper_ocean_metrics.upper_ocean_metrics import ohc_from_profile, mld_temp_crit, mld_dens_crit, temp_average_depth, potential_energy_anomaly100
from ioos_model_comparisons.platforms import get_glider_by_id
import seawater
from glob import glob
import os
from pathlib import Path

# urls
# glider = "ng230-20210928T0000"
glider = "ng278-20210928T0000"
glider = "ng347-20210928T0000"
glider = "ng447-20210928T0000"

gextent = [-95, -91, 25, 28]


url_glider = "https://data.ioos.us/gliders/erddap"
url_gofs = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
url_rtofs = "/Users/mikesmith/Documents/data/rtofs/"
data_dir = Path("/Users/mikesmith/Documents/data/").resolve()
glider_dir = data_dir / "gliders"
impact_dir = data_dir / "impact_metrics"
impact_calculated_dir = impact_dir / "calculated"
impact_model_dir = impact_dir / "models"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(glider_dir, exist_ok=True)
os.makedirs(impact_dir, exist_ok=True)
os.makedirs(impact_model_dir, exist_ok=True)
os.makedirs(impact_calculated_dir, exist_ok=True)


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array.flat[idx], idx


def geo_coord_to_GOFS_coord(lon):
    """Function that converts geographic coordinates to GOFS3.1 coordinates

    Args:
        long (_type_): longitude
        latg (_type_): latitude

    Returns:
        _type_: _description_
    """
    lon = np.asarray(lon)
    if np.ndim(lon) > 0:
        lon = [360+x if x<0 else x for x in lon]
    else:
        lon = [360 + lon if lon<0 else lon][0]
    return lon


# Read glider dataframe output from erddap
glider_pickle = glider_dir / f"{glider}_data.pkl"

try:
    df = pd.read_pickle(glider_pickle)
except FileNotFoundError:
    # Download glider data from erddap with dataset id
    df = get_glider_by_id(glider)
    df.to_pickle(glider_pickle) # Save glider data to pickle file
    
t0 = df.index.min().strftime("%Y-%m-%d")
t1 = df.index.max().strftime("%Y-%m-%d")
df.reset_index(inplace=True)

# Glider - Iterate grouped glider times (each time is a profile)
glider_time = []
glider_lon = []
glider_lat = []

for time, group in df.groupby('time (UTC)'):
    glider_time.append(time)
    glider_lon.append(group['longitude (degrees_east)'].unique()[0])
    glider_lat.append(group['latitude (degrees_north)'].unique()[0])

# Convert from lon, lat to gofs lon, lat 
glon = geo_coord_to_GOFS_coord(glider_lon)

# Convert from lon, lat to gofs lon, lat 
elon = geo_coord_to_GOFS_coord(np.array([gextent[0], gextent[1]]))

# Create dataarrays for pointwise indexing
# https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray
times = xr.DataArray(glider_time, dims='point')
lons = xr.DataArray(glider_lon, dims='point')
lats = xr.DataArray(glider_lat, dims='point')

# RTOFS
# Load in RTOFS files locally
rtofs_files = []
for date in pd.date_range(t0, t1).to_list():
    files = glob(os.path.join(url_rtofs, date.strftime('rtofs.%Y%m%d'), '*.nc'))
    for f in files:
        if f == '':
            continue
        else:
            rtofs_files.append(f)

rtofs = xr.open_mfdataset(sorted(rtofs_files),
                          parallel=True,
                          chunks={
                              'MT': 380,
                              "Depth": 40,
                              "Y": 1710,
                              "X": 742
                              }
                          )
rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat',
                      'MT': 'time', 'Depth': 'depth'})

import zarr
rtofs.to_zarr()

# # Save rtofs lon and lat as variables to speed up indexing calculation
# rtofs_time = rtofs.time.values
# rtofs_lon = rtofs.lon.values
# rtofs_lat = rtofs.lat.values

# # Find index of nearest lon and lat points
# idx_lon = []
# for lon in glider_lon:
#     val, ind = find_nearest(rtofs_lon[0, :], lon)
#     idx_lon.append(ind)
    
# idx_lat = []
# for lat in glider_lat:
#     val, ind = find_nearest(rtofs_lat[:, 0], lat)
#     idx_lat.append(ind)

# # Create dataarrays for pointwise indexing
# # https://stackoverflow.com/questions/40544846/
# # read-multiple-coordinates-with-xarray
# lons_rtofs = xr.DataArray(idx_lon, dims='point')
# lats_rtofs = xr.DataArray(idx_lat, dims='point')
# # trtofs = rtofs.sel(time=slice(t0, t1))
# trtofs = rtofs[['temperature', 'salinity']].sel(time=times, Y=lats_rtofs, X=lons_rtofs, method='nearest')
# # trtofs.to_netcdf(impact_model_dir / f"{glider}_rtofs_data.nc")

from dask.diagnostics import ProgressBar

# or distributed.progress when using the distributed scheduler
# delayed_obj = trtofs.to_netcdf(impact_model_dir / f"{glider}_rtofs_data.nc", compute=False)

# with ProgressBar():
#     results = delayed_obj.compute()

# GOFS
gofs = xr.open_dataset(url_gofs, drop_variables='tau') # Load GOFS data
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})
sgofs = gofs.sel(lon=slice(elon[0], elon[1]), lat=slice(gextent[2], gextent[3]))

# GOFS: Select the time, lon, and lat nearest the glider
# tgofs = gofs.sel(time=slice(t0, t1))
tgofs = sgofs[['temperature', 'salinity']].sel(time=times, lon=lons, lat=lats, method='nearest')

delayed_obj = tgofs.to_netcdf(impact_model_dir / f"{glider}_gofs_data.nc", compute=False)
with ProgressBar():
    results = delayed_obj.compute()

# tgofs.to_netcdf(impact_model_dir / f"{glider}_gofs_data.nc")


# # Load Copernicus data
# def copernicusmarine_datastore(dataset, username, password):
#     from pydap.client import open_url
#     from pydap.cas.get_cookies import setup_session
#     cas_url = 'https://cmems-cas.cls.fr/cas/login'
#     session = setup_session(cas_url, username, password)
#     session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
#     database = ['my', 'nrt']
#     url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
#     try:
#         data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
#     except:
#         url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
#         data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
#     return data_store


# USERNAME = 'maristizabalvar'
# PASSWORD = 'MariaCMEMS2018'
# DATASET_ID = 'global-analysis-forecast-phy-001-024'

# data_store = copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD)

# # Downloading and reading Copernicus grid
# ds = xr.open_dataset(data_store, drop_variables='tau')
# ds = ds.rename(
#     {'thetao': 'temperature', 
#      'so': 'salinity',
#      'latitude': 'lat',
#      'longitude': 'lon'
#      }
#     )
# tds = ds[['salinity', 'temperature']].sel(
#     time=glider_time, 
#     lon=glider_lon,
#     lat=glider_lat,
#     method='nearest')

# # Create dataarrays for pointwise indexing
# # https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray
# # Copernicus: Select the time, lon, and lat nearest the glider
# tds = ds[['salinity', 'temperature']].sel(
#     time=xr.DataArray(glider_time, dims='point'), 
#     lon=xr.DataArray(glider_lon, dims='point'),
#     lat=xr.DataArray(glider_lat, dims='point'),
#     method='nearest')
# # tds.to_netcdf('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.nc')
# # tds = xr.open_dataset('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.nc')

# df = pd.DataFrame()
# # Calculae upper ocean metrics from gofs and add to DataFrame
# for t in tds.point:
#     temp = tds.sel(point=t)
#     try:
#         gdf = calculate_upper_ocean_metrics(
#             pd.to_datetime(temp.time.values), 
#             temp.temperature.values, 
#             temp.salinity.values, 
#             temp.depth.values, 
#             np.full(temp.temperature.values.shape, temp.lat.values),
#             )
#         df = pd.concat([df, gdf])
#         # df.to_csv('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.csv')
#     except ValueError:
#         continue
# df.to_pickle('/Users/mikesmith/Documents/uom/calculated_copernicus_ng645-20210613T0000_data.pkl')