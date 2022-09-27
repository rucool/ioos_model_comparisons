# %%
import datetime as dt
import numpy as np
from ioos_model_comparisons.models import gofs, rtofs
from ioos_model_comparisons.regions import region_config
import seaborn as sns

sns.set_theme(style='whitegrid')

# %%
regions = ['gom'] #'yucatan', 'usvi', 'mab', 'gom', 'carib', 'wind', 'sab']
depths = [0, 50, 100, 150, 200]

today = dt.date.today()
week_ago = today - dt.timedelta(days=7)

def percentile(da, min=2, max=98):
    vmin = np.floor(np.nanpercentile(da, min))
    vmax = np.ceil(np.nanpercentile(da, max))
    return vmin, vmax

# %% 
with rtofs() as ds:
    # Save rtofs lon and lat as variables to speed up indexing calculation
    grid_lons = ds.lon.values[0,:]
    grid_lats = ds.lat.values[:,0]
    grid_x = ds.x.values
    grid_y = ds.y.values

    # Loop through regions
    for item in regions:
        region = region_config(item, model="rtofs")
        extent = region['extent']

        print(f'Region: {region["name"]}, Extent: {extent}')

        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp(extent[:2], grid_lons, grid_x)
        lats_ind = np.interp(extent[2:], grid_lats, grid_y)

        # We use np.floor on the first index of each slice and np.ceiling on the second index of each slice 
        # in order to widen the area of the extent slightly around each index. 
        # This returns a float so we have to broadcast the datatype to an integer in order to use .isel
        extent = np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int), np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int)

        # Use the xarray .isel selector on x/y since we know the exact indexes we want to slice
        tds = ds.sel(
            time=slice(week_ago, today),
            depth=depths
            ).isel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))

        mds = tds.mean('time')
        df = mds.to_dataframe().reset_index()

        for var in ["temperature", "salinity"]:
            sns.boxplot(x='depth', y=var, data=df)
            vmin, vmax = percentile(mds[var])
        # np.linspace(vmin, vmax, 1+np.int(vmax-vmin)*2)
