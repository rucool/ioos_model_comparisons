import pandas as pd
from  upper_ocean_metrics.uom_functions import calculate_upper_ocean_metrics
from pathlib import Path
from hurricanes.platforms import get_glider_by_id
import xarray as xr
import numpy as np
import os

glider = None
# glider = "ng645-20210613T0000"
# rtofs = None
rtofs = "ng645-20210613T0000_rtofs_0day_offset_data.nc"
# rtofs=None
gofs = False
copernicus = False

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_gliders = (path_data / "gliders")
path_impact = (path_data / "impact_metrics")
path_impact_model = path_impact / "models"
path_impact_calculated = path_impact / "calculated"

if glider:
    # Read glider dataframe output from erddap
    glider_pickle = path_gliders / f"{glider}_data.pkl"
    sname = os.path.splitext(glider_pickle)[0] + '_computed'
    try:
        df = pd.read_pickle(glider_pickle)
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
    except FileNotFoundError:
        # Download glider data from erddap with dataset id
        df = get_glider_by_id(glider)
        df.to_pickle(glider_pickle) # Save glider data to pickle file
        
    # Glider - Iterate grouped glider times (each time is a profile)
    glider_df = pd.DataFrame()
    glider_list = []
    for time, group in df.groupby(level=0):
        print(time)
        gldf = calculate_upper_ocean_metrics(
            time,
            group['temp'].to_numpy(), 
            group['salinity'].to_numpy(), 
            group['depth'].to_numpy(), 
            group['lat'].to_numpy(),
            group['density'].to_numpy(),
            )
        glider_list.append(gldf)
    glider_df = pd.concat(glider_list)
    # glider_df.to_csv(path_impact_calculated / f"{sname}.csv")
    glider_df.to_pickle(path_impact_calculated / f"{sname}_.pkl")

if rtofs:
    try:
        sname = os.path.splitext(rtofs)[0] + '_computed'
        with xr.open_dataset(path_impact_model / rtofs) as ds:
            rtofs_list = []
            for t in ds.point:
                temp = ds.sel(point=t)
                rdf = calculate_upper_ocean_metrics(
                    pd.to_datetime(temp.time.values),
                    temp.temperature.values,
                    temp.salinity.values,
                    temp.depth.values,
                    np.full(temp.temperature.values.shape, temp.lat.values),
                    )
                rtofs_list.append(rdf)
            rtofs_df = pd.concat(rtofs_list)
            # rtofs_df.to_csv(path_impact_calculated / f"{sname}.csv")
            rtofs_df.to_pickle(path_impact_calculated / f"{sname}.pkl")
    except FileNotFoundError:
        print("Corresponding RTOFS model data not found. Please run Step 1.")

if gofs:
    try: 
        with xr.open_dataset(path_impact_model / gofs) as ds:
            # Calculate upper ocean metrics from gofs and add to DataFrame
            gofs_df = pd.DataFrame()
            for t in ds.point:
                temp = ds.sel(point=t)
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
                # gofs_df.to_csv(path_impact_calculated / f"{glider}_calculated_gofs_data.csv")
            gofs_df.to_pickle(path_impact_calculated / f"{glider}_calculated_gofs_data.pkl")
    except FileNotFoundError:
        print("Corresponding GOFS model data not found. Please run Step 1.")

if copernicus:
    try:
        with xr.open_dataset(path_impact_model / copernicus) as ds:
            df = pd.DataFrame()
            # Calculae upper ocean metrics from gofs and add to DataFrame
            for t in ds.point:
                temp = ds.sel(point=t)
                try:
                    gdf = calculate_upper_ocean_metrics(
                        pd.to_datetime(temp.time.values), 
                        temp.temperature.values, 
                        temp.salinity.values, 
                        temp.depth.values, 
                        np.full(temp.temperature.values.shape, temp.lat.values),
                        )
                    df = pd.concat([df, gdf])
                    # df.to_csv('/Users/mikesmith/Documents/calculated_copernicus_ng645-20210613T0000_data.csv')
                except ValueError:
                    continue
            df.to_pickle('/Users/mikesmith/Documents/uom/calculated_copernicus_ng645-20210613T0000_data.pkl')
    except:
        pass



        glider = glider.rename({
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