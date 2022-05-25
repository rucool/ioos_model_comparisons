from hfradar.plotting.plot_nc import plot_radials as ncradialsplot
import glob
import os
import xarray as xr
import re

# fdir = '/Users/mikesmith/Documents/Work/codar/ugos/radials/qc/MARA/nc'
fdir = '/Users/mikesmith/Documents/work/ugos/radials/qc/MARA/nc/new'
f_ext =  '*.nc'
# save_dir = '/Users/mikesmith/Documents/mara_qc/postprocessed/'
save_dir = '/Users/mikesmith/Documents/ugos_mara_second_half/nc/pass_or_fail'

extent = [-83.25, -78.75, 22.75, 25.25]
lat_ticks = [22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26]
lon_ticks = [-84,  -83, -82, -81, -80, -79, -78]
title = 'UGOS - Marathon, FL - Radial Velocities - Quality Controlled (Primary Filter)'

kwargs = dict(
    extent = extent,
    lat_ticks = lat_ticks,
    lon_ticks = lon_ticks,
    title = title,
    sub = 1
)

# Radial Velocity - Red vs Blue
# kwargs['plot_type'] = 'velocity'
# kwargs['prim_filter'] = True
# kwargs['velocity_min'] = -80
# kwargs['velocity_max'] = 80
# kwargs['cbar_step'] = 20

# # Good (green) vs Bad (red)
kwargs['plot_type'] = 'qc_pass_fail'

# # Motion - Towards () or Away from radar
# kwargs['plot_type'] = 'motion'

os.makedirs(save_dir, exist_ok=True)

radials = sorted(glob.glob(os.path.join(fdir, f_ext)))

for r in radials:
    save_file = os.path.basename(r) + '.png'
    kwargs['output_file'] = os.path.join(save_dir, kwargs['plot_type'], save_file)

    with xr.open_dataset(r) as f:
        origin = re.findall(r"[-+]?\d*\.\d+|\d+", f.site.Origin)
        kwargs['markers'] = [[float(origin[1]), float(origin[0]), dict(marker='o', markersize=4, color='r')]]
        ncradialsplot(f, **kwargs)