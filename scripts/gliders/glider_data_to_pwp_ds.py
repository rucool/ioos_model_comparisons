import datetime as dt
import numpy as np
import xarray as xr
import pandas as pd
# from src.gliders import glider_dataset
from erddapy import ERDDAP

glider = 'ng645-20210613T0000'
save_dir = '/Users/mikesmith/Documents/'

# This will only get the profile between 12 and 13 GMT. Otherwise, if you expand this, you would have to loop through
# the data or filter to the correct time in the DataFrame
t0 = dt.datetime(2021, 8, 28, 12, 0)
t1 =  dt.datetime(2021, 8, 28, 13, 0)

target_depth = 19.864616 # I picked this exactly out in the data.

ylims = [200, 0]

# initialize keyword arguments for glider functions
gargs = dict()
gargs['time_start'] = t0  # False
gargs['time_end'] = t1
gargs['filetype'] = 'dataframe'


def get_erddap_dataset(ds_id, variables=None, constraints=None, filetype=None):
    """
    Returns a netcdf dataset for a specified dataset ID (or dataframe if dataset cannot be converted to xarray)
    :param ds_id: dataset ID e.g. ng314-20200806T2040
    :param variables: optional list of variables
    :param constraints: optional list of constraints
    :param filetype: optional filetype to return, 'nc' (default) or 'dataframe'
    :return: netcdf dataset
    """
    variables = variables or None
    constraints = constraints or None
    filetype = filetype or 'nc'

    e = ERDDAP(server='NGDAC',
               protocol='tabledap',
               response='nc')
    e.dataset_id = ds_id
    if constraints:
        e.constraints = constraints
    if variables:
        e.variables = variables
    if filetype == 'nc':
        try:
            ds = e.to_xarray()
            ds = ds.sortby(ds.time)
        except OSError:
            print('No dataset available for specified constraints: {}'.format(ds_id))
            ds = []
        except TypeError:
            print('Cannot convert to xarray, providing dataframe: {}'.format(ds_id))
            ds = e.to_pandas().dropna()
    elif filetype == 'dataframe':
        ds = e.to_pandas().dropna()
    else:
        print('Unrecognized filetype: {}. Needs to  be "nc" or "dataframe"'.format(filetype))

    return ds

def glider_dataset(gliderid, time_start=None, time_end=None, variables=None, filetype=None):
    """
    Return data from a specific glider
    """
    print('Retrieving glider dataset: {}'.format(gliderid))
    time_start = time_start or None
    time_end = time_end or None
    variables = variables or [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity',
        'density',
        'profile_id'
    ]
    filetype = filetype or 'nc'

    constraints = dict()
    if time_start:
        constraints['time>='] = time_start.strftime('%Y-%m-%dT%H:%M:%SZ')
    if time_end:
        constraints['time<='] = time_end.strftime('%Y-%m-%dT%H:%M:%SZ')

    if len(constraints) < 1:
        constraints = None

    kwargs = dict()
    kwargs['variables'] = variables
    kwargs['constraints'] = constraints
    kwargs['filetype'] = filetype

    ds = get_erddap_dataset(gliderid, **kwargs)
    if isinstance(ds, pd.core.frame.DataFrame):
        for col in ds.columns:
            ds.rename(columns={col: col.split(' ')[0]}, inplace=True)  # get rid of units in column names
        ds['time'] = ds['time'].apply(lambda t: dt.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ'))

    return ds


# Grab the glider data back from erddap
glider_df = glider_dataset(glider, **gargs)
from hurricanes.src import depth_interpolate

tdf = depth_interpolate(glider_df)

# Output the original data to a netCDF
# This is a dictionary that Sam sent that needs to be filled in with the profile data
IC_data = {'z': {'dims': 'z', 'data': glider_df['depth']}, # z is positive down in pwp
             't': {'dims': 'z', 'data': glider_df['temperature']},
             's': {'dims': 'z', 'data': glider_df['salinity']},
             'lat': {'dims': 'lat','data': np.array([glider_df['latitude'][0]])}} # static latitude

# Create an xarray dataset from the dict above
IC_ds = xr.Dataset.from_dict(IC_data)

# Output to a netCDF so we can load it in easier
IC_ds.to_netcdf(f'/Users/mikesmith/Documents/{glider}_20210828121611Z_original.nc')

# # Output the altered data to a netCDF
# # Set any salinity that is less than the target depth equal to the salinity at the target depth.
# glider_df['fake_salinity'] = glider_df['salinity']
# glider_df['fake_salinity'][glider_df.depth < target_depth] = glider_df[glider_df['depth'] == target_depth]['salinity'].iloc[0]

# This is a dictionary that Sam sent that needs to be filled in with the profile data
IC_data = {'z': {'dims': 'z', 'data': glider_df['depth']}, # z is positive down in pwp
             't': {'dims': 'z', 'data': glider_df['temperature']},
             's': {'dims': 'z', 'data': glider_df['salinity']},
             'lat': {'dims': 'lat','data': np.array([glider_df['latitude'][0]])}} # static latitude

# Create an xarray dataset from the dict above
IC_ds = xr.Dataset.from_dict(IC_data)

# Output to a netCDF so we can load it in easier
IC_ds.to_netcdf(f'/Users/mikesmith/Documents/{glider}_20210828121611Z_altered.nc')

# import matplotlib.pyplot as plt
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 8))
#
# # Temperature Profiles (Panel 1)
# h = ax1.scatter(
#     glider_df['temperature'],
#     glider_df['depth'],
#     c='red',
#     s=100,
#     edgecolor='black'
# )
#
# ax1.set_ylim(ylims)
# ax1.grid(True, linestyle='--', linewidth=0.5)
# ax1.tick_params(axis='x', labelsize=20)
# ax1.tick_params(axis='y', labelsize=20)
# ax1.set_xlabel('Temperature (c)', fontsize=22, fontweight='bold')
# ax1.set_ylabel('Depth (m)', fontsize=22, fontweight='bold')
#
#
# # Salinity Profiles (Panel 2)
# h = ax2.plot(
#     glider_df['salinity'],
#     glider_df['depth'],
#     'g',
#     glider_df['fake_salinity'],
#     glider_df['depth'],
#     'r',
# )
#
# ax2.set_ylim(ylims)
# # ax2.set_xlim([-.4, .8])
# ax2.grid(True, linestyle='--', linewidth=0.5)
# ax2.tick_params(axis='x', labelsize=20)
# ax2.tick_params(axis='y', labelsize=20)
# ax2.set_xlabel('Salinity', fontsize=22, fontweight='bold')
# ax2.set_ylabel('Depth (m)', fontsize=2, fontweight='bold')
#
# title_str = f'{glider} Profiles\n'
# plt.suptitle(title_str, fontsize=30, fontweight='bold')
#
# # cb.set_label('Density[kg m$^{-3}$]')
# # plt.tight_layout()
# # plt.savefig(f'/Users/mikesmith/Documents/{glider}-profiles_dddz-{tstr}_{d}-dist.png')
# plt.show()
#
#

