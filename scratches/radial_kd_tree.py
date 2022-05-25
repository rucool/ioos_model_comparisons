import matplotlib.pyplot as plt
from hfradar.src.radials import Radial
import glob
import xarray as xr
import os


def concatenate_radials(radial_list, type=None, enhance=False):
    """
    This function takes a list of Radial objects or radial file paths and
    combines them along the time dimension using xarrays built-in concatenation
    routines.
    :param radial_list: list of radial files or Radial objects that you want to concatenate
    :return: radials concatenated into an xarray dataset by range, bearing, and time
    """
    type = type or 'multidimensional'

    radial_dict = {}
    for radial in radial_list:

        if not isinstance(radial, Radial):
            radial = Radial(radial)

        if type == 'multidimensional':
            radial_dict[radial.file_name] = radial.to_xarray_multidimensional(enhance=enhance)
        elif type == 'tabular':
            radial_dict[radial.file_name] = radial.to_xarray_tabular(enhance=enhance)

    ds = xr.concat(radial_dict.values(), 'time')
    return ds.sortby('time')


# radial_dir = '/Users/mikesmith/Documents/work/ugos/new/raw/'
radial_dir = '/Users/mikesmith/Documents/work/ugos/radials/raw/first/MARA/'

files = glob.glob(os.path.join(radial_dir, 'RDLm_MARA_2020_01*.ruv'))

ds = concatenate_radials(sorted(files), type='multidimensional', enhance=True)
ds = ds.reset_coords()

ds = ds.mean(dim=('time'))
ds = ds.assign_coords({'lon': ds.lon, 'lat': ds.lat})

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from hfradar.src.calc import KDTreeIndex

tree = KDTreeIndex(ds.VELO)

# Points query
index = tree.query([(24.18, -81.48), (24.23, -80.26)])
ax = plt.subplot(projection=ccrs.PlateCarree())
ds.VELO.plot.pcolormesh('lon', 'lat', ax=ax, infer_intervals=True)
ax.scatter(ds.VELO.lon[index], ds.VELO.lat[index], marker='x', color='g', transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()

# Radius query
index = tree.query_ball_point((24.18, -81.48), 5)
ax = plt.subplot(projection=ccrs.PlateCarree())
ds.VELO.plot.pcolormesh('lon', 'lat', ax=ax, infer_intervals=True)
ax.scatter(ds.VELO.lon[index], ds.VELO.lat[index], marker='x', color='g', transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()


tds = ds.mean(dim=('time'))
plt.quiver(tds.lon, tds.lat, tds.VELU, tds.VELV)
plt.show()