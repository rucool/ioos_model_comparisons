#!/usr/bin/env python

# # import matplotlib.dates as mdates
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from erddapy import ERDDAP
from upper_ocean_metrics.uom_functions import calculate_upper_ocean_metrics
from ioos_model_comparisons.platforms import get_glider_by_id
from ioos_model_comparisons.calc import convert_lon_180_to_360, find_nearest
import seawater
from glob import glob
import os
from pathlib import Path

# urls
# glider = "ng230-20210928T0000"
# glider = "ng278-20210928T0000"
# glider = "ng347-20210928T0000"
# glider = "ng447-20210928T0000"
glider = "ng645-20210613T0000"


# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_rtofs = (path_data / "rtofs")
path_gofs = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
path_gliders = (path_data / "gliders")
path_impact = (path_data / "impact_metrics")
path_impact_calculated = path_impact / "calculated"
path_impact_model = path_impact / "models"


def calculate_density(temperature, salinity, depth, lat):
    pressure = seawater.eos80.pres(depth, lat)
    density = seawater.eos80.dens(salinity, temperature, pressure)
    return density


# Read glider dataframe output from erddap
glider_pickle = path_gliders / f"{glider}_data.pkl"

try:
    df = pd.read_pickle(glider_pickle)
except FileNotFoundError:
    # Download glider data from erddap with dataset id
    df = get_glider_by_id(glider)
    df.to_pickle(glider_pickle) # Save glider data to pickle file
    
df = df.reset_index()
df = df.rename({
    "time (UTC)": "time",
    "longitude (degrees_east)": "lon",
    "latitude (degrees_north)": "lat",
    "pressure (decibar)": "pressure",
    "temperature (degrees_C)": "temp",
    "depth (m)": "depth",
    "salinity (1)": "salinity",
    "conductivity (mS cm-1)": "conductivity",
    "density (kg m-3)": "density",
}, axis=1)

df = df.set_index("time").sort_index()
t0 = df.index.min().strftime("%Y-%m-%d")
t1 = df.index.max().strftime("%Y-%m-%d")

# For profile diagnostic purposes
# df = df.loc[pd.IndexSlice["2021-08-18":"2021-08-22"], :]

# # Glider - Iterate grouped glider times (each time is a profile)
# glider_df = pd.DataFrame()

# for time, group in df.groupby(level=0):
#     print(time)
#     gldf = calculate_upper_ocean_metrics(
#         time,
#         group['temp'].to_numpy(), 
#         group['salinity'].to_numpy(), 
#         group['depth'].to_numpy(), 
#         group['lat'].to_numpy(),
#         group['density'].to_numpy(),
#         )
#     glider_df = pd.concat([glider_df, gldf])
#     glider_df.to_csv(path_impact_calculated / f"{glider}_calculated_glider_data.csv")
# glider_df.to_pickle(path_impact_calculated / f"{glider}_calculated_glider_data.pkl")


tdf = df.groupby(level=0).first()
glon = tdf["lon"]
glat = tdf["lat"]

# # Create dataarrays for pointwise indexing
# # https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray
times = xr.DataArray(tdf.index.tolist(), dims='point')
lons = xr.DataArray(glon.values, dims='point')
lons_gofs = xr.DataArray(convert_lon_180_to_360(glon), dims="point")
lats = xr.DataArray(glat.values, dims='point')

# GOFS
gofs = xr.open_dataset(path_gofs, drop_variables='tau') # Load GOFS data
gofs = gofs.rename({'surf_el': 'sea_surface_height', 'water_temp': 'temperature', 'water_u': 'u', 'water_v': 'v'})

# GOFS: Select the time, lon, and lat nearest the glider
# tgofs = gofs[['temperature', 'salinity']].sel(time=slice("2021-08-18","2021-08-22"))
# tgofs = tgofs.sel(lon=lons, lat=lats, method='nearest')
tgofs = gofs[['temperature', 'salinity']].sel(time=times, lon=lons_gofs, lat=lats, method='nearest')
# tgofs.load()
# tgofs.to_netcdf(impact_model_dir / f"{glider}_gofs_data.nc")

# Calculate upper ocean metrics from gofs and add to DataFrame
gofs_df = pd.DataFrame()
for t in tgofs.point:
    temp = tgofs.sel(point=t)
    temp.load()
    # try:
    gdf = calculate_upper_ocean_metrics(
        pd.to_datetime(temp.time.values), 
        temp.temperature.values, 
        temp.salinity.values, 
        temp.depth.values, 
        np.full(temp.temperature.values.shape, temp.lat.values),
        )
    gofs_df = pd.concat([gofs_df, gdf]) 
    gofs_df.to_csv(path_impact_calculated / f"{glider}_calculated_gofs_data.csv")
gofs_df.to_pickle(path_impact_calculated / f"{glider}_calculated_gofs_data.pkl")

# # RTOFS
# # Load in RTOFS files locally
# rtofs_files = []
# for date in pd.date_range(t0, t1).to_list():
#     files = glob(os.path.join(path_rtofs, date.strftime('rtofs.%Y%m%d'), '*.nc'))
#     for f in files:
#         if f == '':
#             continue
#         else:
#             rtofs_files.append(f)

# # rtofs = xr.open_mfdataset(sorted(rtofs_files), parallel=True)
# rtofs = xr.open_mfdataset(
#     sorted(rtofs_files),
#     concat_dim="MT",
#     combine="nested",
#     data_vars='minimal',
#     coords='minimal',
#     compat='override',
#     parallel=True
#     )

# rtofs = rtofs.rename({'Longitude': 'lon', 'Latitude': 'lat',
#                       'MT': 'time', 'Depth': 'depth'})

# # Save rtofs lon and lat as variables to speed up indexing calculation
# rtofs_lon = rtofs.lon.values
# rtofs_lat = rtofs.lat.values

# # Find index of nearest lon and lat points
# calc_lon = np.array([find_nearest(rtofs_lon[0, :], lon) for lon in glon.values])
# calc_lat = np.array([find_nearest(rtofs_lat[:, 0], lat) for lat in glat.values])

# # Create dataarrays for pointwise indexing
# # https://stackoverflow.com/questions/40544846/
# # read-multiple-coordinates-with-xarray
# lons = xr.DataArray(calc_lon[:,1], dims='point')
# lats = xr.DataArray(calc_lat[:,1], dims='point')
# # trtofs = rtofs[['temperature', 'salinity']].sel(time=slice("2021-08-18","2021-08-22")) 
# grtofs = rtofs[['temperature', 'salinity']].sel(time=times, Y=lats, X=lons, method="nearest")
# trtofs = rtofs.sel(time=times, method="nearest")
# trtofs = trtofs.sel(Y=lats, X=lons)
# # trtofs.load()
# trtofs.to_netcdf(path_impact_model / f"{glider}_rtofs_data.nc")
# # trtofs = xr.open_dataset(f'/Users/mikesmith/Documents/calculated_rtofs_ng645-20210613T0000_data.nc')

# # Calculate upper ocean metrics from rtofs and add to DataFrame
# rtofs_df = pd.DataFrame()
# for t in trtofs.point:
#     temp = trtofs.sel(point=t)
#     rdf = calculate_upper_ocean_metrics(
#         pd.to_datetime(temp.time.values),
#         temp.temperature.values,
#         temp.salinity.values,
#         temp.depth.values,
#         np.full(temp.temperature.values.shape, temp.lat.values),
#         )
#     rtofs_df = pd.concat([rtofs_df, rdf])
#     rtofs_df.to_csv(impact_calculated_dir / f"{glider}_calculated_rtofs_data.csv")
# rtofs_df.to_pickle(impact_calculated_dir / f"{glider}_calculated_rtofs_data.pkl")


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