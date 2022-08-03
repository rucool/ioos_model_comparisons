import xarray as xr
import pandas as pd
from pathlib import Path
import os
from glob import glob

# Get path information about this script
current_dir = Path(__file__).parent #__file__ gets the location of this file.
script_name = Path(__file__).name

# Set main path of data and plot location
root_dir = Path.home() / "Documents"

# Paths to data sources
path_data = (root_dir / "data") # create data path
path_rtofs = (path_data / "rtofs")

t0 = pd.Timestamp(2021, 5, 1)
t1 = pd.Timestamp(2022, 2, 1)

# rtofs_file_dates = []
# rtofs_file_paths = []
# for date in pd.date_range(t0, t1).to_list():
#     tstr = date.strftime('rtofs.%Y%m%d')
#     files = sorted(glob(os.path.join(path_rtofs, tstr, '*.nc')))
#     for f in files:
#         date_list = f.split('rtofs/rtofs.')[1].split('/')
#         rtofs_file_dates.append(pd.to_datetime(date_list[0]) + dt.timedelta(hours=int(date_list[1].split('_')[3].strip('f'))))
#         rtofs_file_paths.append(f)
rtofs_files = [glob(os.path.join(path_rtofs, date.strftime("rtofs.%Y%m%d"), '*.nc')) for date in pd.date_range(t0, t1).to_list()]
rtofs_files = sorted(sum(rtofs_files, [])) #use builtin to collapse list of lists

ds = xr.open_mfdataset(
    rtofs_files,
    concat_dim="MT",
    combine="nested",
    data_vars='minimal',
    coords='minimal',
    compat='override',
    parallel=True)
ds = ds.rename({'Longitude': 'lon', 'Latitude': 'lat',
                'MT': 'time', 'Depth': 'depth'})
ds = ds.drop_vars('Date')
ds[["lon", "lat"]].load()


