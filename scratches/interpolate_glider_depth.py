from ioos_model_comparisons.calc import depth_bin, depth_interpolate
import numpy as np
import pandas as pd
import xarray as xr
from ioos_model_comparisons.platforms import get_glider_by_id
from ioos_model_comparisons.calc import convert_lon_180_to_360, find_nearest
import seawater
from glob import glob
import os
from pathlib import Path

glider = "ng645-20210613T0000"
days_offset = 2

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_rtofs = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
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
t0 = df.index.min().strftime("%Y-%m-%d")
t1 = df.index.max().strftime("%Y-%m-%d")
tdf = df.reset_index()
interp = tdf.groupby(['time']).apply(depth_interpolate)
# interp
ds = interp.to_xarray()
tds = ds.resample(time='6H').interpolate('linear')
tds