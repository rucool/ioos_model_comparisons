#!/usr/bin/env python

# # import matplotlib.dates as mdates
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
# from glider_uom import calculate_upper_ocean_metrics
# from erddapy import ERDDAP
# from upper_ocean_metrics.uom_functions import calculate_upper_ocean_metrics
from hurricanes.platforms import get_glider_by_id, Argo
from hurricanes.calc import convert_lon_180_to_360, find_nearest
import os
from pathlib import Path

glider = "ng645-20210613T0000"
days_offset = 0
gofs = False
rtofs = False
copernicus = False
region = [-98, -80, 18, 31]
# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_rtofs = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
# path_rtofs = (path_data / "rtofs")
path_gofs = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
path_gliders = (path_data / "gliders")
path_impact = (path_data / "impact_metrics")
path_impact_calculated = path_impact / "calculated"
path_impact_model = path_impact / "models"

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
dstr = "%Y-%m-%d"
t0 = df.index.min().strftime(dstr)
t1 = df.index.max().strftime(dstr)
tdf = df.groupby(level=0).first()
glon = tdf["lon"]
glat = tdf["lat"]
region = [
    convert_lon_180_to_360(-98),
    convert_lon_180_to_360(-80),
    18,
    31
    ]

if days_offset:
    gtime = tdf.index.shift(days_offset, freq='D')
else:
    gtime = tdf.index 

if gofs:
    # GOFS
    gofs = xr.open_dataset(
        path_gofs, 
        drop_variables=['tau', 'water_temp_bottom', 'salinity_bottom', 'water_u_bottom', 'water_v_bottom']
        )
    gofs = gofs.rename(
        {'surf_el': 'sea_surface_height',
         'water_temp': 'temperature',
         'water_u': 'u',
         'water_v': 'v'
         }
        )
    
    gofs = gofs.sel(
        time=slice(pd.to_datetime(t0), pd.to_datetime(t1)),
        depth=slice(0, 1000), 
        lon=slice(region[0], region[1]), 
        lat=slice(region[2], region[3])
        )
    
    gofs = gofs[['temperature', 'salinity']].chunk({"time": 100, 'lon': 113, 'lat': 163})
    clon = convert_lon_180_to_360(glon.values)
    gldf = []
    for ind, t in enumerate(gtime.to_list()):
        # GOFS: Select the time, lon, and lat nearest the glider
        tgofs = gofs.sel(
            time=t,
            lon=clon[ind],
            lat=glat[ind],
            method='nearest'
            )
        gldf.append(tgofs)

    # # GOFS: Select the time, lon, and lat nearest the glider
    # tgofs = gofs[['temperature', 'salinity']].sel(
    #     time=xr.DataArray(gtime, dims='point'),
    #     lon=xr.DataArray(convert_lon_180_to_360(glon.values), dims="point"),
    #     lat=xr.DataArray(glat.values, dims='point'),
    #     method='nearest'
    #     )
    # gldf = [tgofs.sel(point=t) for t in tgofs.point]
    gofs_transect = xr.concat(gldf, dim="point")

    save_file = path_impact_model / f"{glider}_gofs_{days_offset}day_offset_data.nc"
    try:
        gofs.to_netcdf(save_file)
    except PermissionError:
        os.remove(save_file)
        gofs.to_netcdf(save_file)

if rtofs:
    # rtofs = xr.open_dataset(path_rtofs)
    # Load in RTOFS files locally
    rtofs = xr.open_zarr("/Users/mikesmith/Documents/data/rtofs_2021.zarr")
    # rtofs[["lon", "lat"]].load()

    # Save rtofs lon and lat as variables to speed up indexing calculation
    rtofs_lon = rtofs.lon.values
    rtofs_lat = rtofs.lat.values
    
    # Find index of nearest lon and lat points
    calc_lon = np.array([find_nearest(rtofs_lon[0, :], lon) for lon in glon.values])
    calc_lat = np.array([find_nearest(rtofs_lat[:, 0], lat) for lat in glat.values])

    # Create dataarrays for pointwise indexing
    # https://stackoverflow.com/questions/40544846/
    # read-multiple-coordinates-with-xarray
    trtofs = rtofs[['salinity', 'temperature']].sel(
        time=xr.DataArray(gtime, dims='point'), 
        Y=xr.DataArray(calc_lat[:,1], dims='point'), 
        X=xr.DataArray(calc_lon[:,1], dims='point'),
        method="nearest"
        )

    # Calculate upper ocean metrics from rtofs and add to DataFrame
    from joblib import Parallel, delayed
    import multiprocessing

    def request_point(point):
        return trtofs.sel(point=point)

    num_cores = multiprocessing.cpu_count() - 2   
    rdfs = Parallel(n_jobs=6)(delayed(request_point)(t) for t in trtofs.point)
    # dfs = {glider: df for (glider, df) in downloads}
    
    # rdfs = [trtofs.sel(point=t).load() for t in trtofs.point[:10]]
    rtofs_transect = xr.concat(rdfs, dim="point")

    save_file = path_impact_model / f"{glider}_rtofs_{days_offset}day_offset_data.nc"
    try:
        trtofs.to_netcdf(save_file)
    except PermissionError:
        os.remove(save_file)
        trtofs.to_netcdf(save_file)


if copernicus:
    def copernicusmarine_datastore(dataset, username, password):
        from pydap.client import open_url
        from pydap.cas.get_cookies import setup_session
        cas_url = 'https://cmems-cas.cls.fr/cas/login'
        session = setup_session(cas_url, username, password)
        session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
        database = ['my', 'nrt']
        url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
        try:
            data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
        except:
            url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
            data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
        return data_store


    USERNAME = 'maristizabalvar'
    PASSWORD = 'MariaCMEMS2018'
    DATASET_ID = 'global-analysis-forecast-phy-001-024'

    data_store = copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD)

    # Downloading and reading Copernicus grid
    ds = xr.open_dataset(data_store, drop_variables='tau')
    ds = ds.rename(
        {
            'thetao': 'temperature', 
            'so': 'salinity',
            'latitude': 'lat',
            'longitude': 'lon'
            }
        )

    # Create dataarrays for pointwise indexing
    # https://stackoverflow.com/questions/40544846/read-multiple-coordinates-with-xarray
    # Copernicus: Select the time, lon, and lat nearest the glider
    tds = ds[['salinity', 'temperature']].sel(
        time=xr.DataArray(xr.DataArray(gtime, dims='point'), dims='point'), 
        lon=xr.DataArray(xr.DataArray(glon.values, dims='point'), dims='point'),
        lat=xr.DataArray(xr.DataArray(glat.values, dims='point'), dims='point'),
        method='nearest')
    # tds.to_netcdf('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.nc')
    # tds = xr.open_dataset('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.nc')



