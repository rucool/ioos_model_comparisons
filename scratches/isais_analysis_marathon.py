from hfradar.src.radials import Radial
import glob
import os
import xarray as xr
import matplotlib.pyplot as plt


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


radial_dir = '/Users/mikesmith/Documents/work/ugos/new/'

files = glob.glob(os.path.join(radial_dir, '*.ruv'))

ds = concatenate_radials(sorted(files), type='multidimensional', enhance=True)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

# Isais
# dsb = ds.sel(bearing=87, range=64.07, method='nearest')
test = ds.sel(bearing=slice(177, 180), range=slice(85, 100))

# Mean of area
mean = test.mean(dim=('bearing', 'range'))
mean.VELO.plot()
plt.grid()
# axes[0].set_ylim([-7, 7])
plt.xlabel('')
plt.ylabel('Velocity (cm/s)')
plt.suptitle('Hurricane Isaias\n2020-07-28 to 2020-08-05')
plt.show()
# plt.savefig('/Users/mikesmith/Desktop/isaias-velo.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


# Mean of area
mean = test.mean(dim=('bearing', 'range'))

fig, axes = plt.subplots(nrows=2)
plt.suptitle('Hurricane Isaias\n2020-07-28 to 2020-08-05')

mean.VELU.plot(ax=axes[0])
axes[0].grid()
# axes[0].set_ylim([-7, 7])
axes[0].set_xlabel('')
axes[0].set_ylabel('Velocity (cm/s)')

mean.VELV.plot(ax=axes[1])
axes[1].grid()
axes[1].set_ylim([-80, 80])
axes[1].set_ylabel('v (cm/s)')
axes[1].set_xlabel('')

plt.tight_layout()
plt.show()
# plt.savefig('/Users/mikesmith/Desktop/isaias-mean.png', dpi=300, bbox_inches='tight', pad_inches=0.1)


# Single bearing bin, multiple ranges
bearing = test.isel(bearing=0)

fig, axes = plt.subplots(nrows=2, sharex=True)
bearing.VELU.plot.line(x='time', ax=axes[0], add_legend=False)
axes[0].grid()
# axes[0].set_ylim([-7, 7])
axes[0].set_ylabel('u (cm/s)')
axes[0].set_xlabel('')

bearing.VELV.plot.line(x='time', ax=axes[1], add_legend=False)
axes[1].grid()
axes[1].set_ylim([-80, 80])
axes[1].set_ylabel('v (cm/s)')
axes[1].set_xlabel('')
# axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
plt.suptitle(f'Hurricane Isaias\n2020-07-28 to 2020-08-05')

plt.tight_layout()
plt.show()
# plt.savefig('/Users/mikesmith/Desktop/isaias-split.png', dpi=300, bbox_inches='tight', pad_inches=0.1)



