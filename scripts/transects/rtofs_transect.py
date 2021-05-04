from glob import glob
import os
import cmocean
import numpy as np
import xarray as xr
import datetime as dt
from src.calc import calculate_transect, convert_ll_to_model_ll
from src.plotting import plot_transect, plot_transects
from src.common import transects

# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/transects/rtofs/'

url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/home/michaesm/public_html/hurricanes/plots/transects/rtofs/'

days = 2

# transect coordinates and variable limits
transects = transects()

# Get today and yesterday dates
date_list = [dt.datetime.today() - dt.timedelta(days=x) for x in range(days)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])

for f in rtofs_files:
    try:
        with xr.open_dataset(f) as ds:
            date_str = ds["MT"].dt.strftime("%Y-%m-%dT%H%M%SZ").data[0]
            lat = np.asarray(ds.Latitude[:])
            lon = np.asarray(ds.Longitude[:])
            depth = np.asarray(ds.Depth[:])

            for transect in transects.keys():
                transect_str = " ".join(transect.split("_")).title()
                x1 = transects[transect]['extent'][0]
                y1 = transects[transect]['extent'][1]
                x2 = transects[transect]['extent'][2]
                y2 = transects[transect]['extent'][3]

                # calculate longitude and latitude of transect lines
                X, Y, _ = calculate_transect(x1, y1, x2, y2)

                # Conversion from glider longitude and latitude to RTOFS convention
                target_lon, target_lat = convert_ll_to_model_ll(X, Y, 'rtofs')

                # interpolating transect X and Y to lat and lon
                lonIndex = np.round(np.interp(target_lon, lon[0, :], np.arange(0, len(lon[0, :])))).astype(int)
                latIndex = np.round(np.interp(target_lat, lat[:, 0], np.arange(0, len(lat[:, 0])))).astype(int)

                target_temp = np.full([len(depth), len(target_lon)], np.nan)
                target_salt = np.full([len(depth), len(target_lon)], np.nan)

                for pos in range(len(lonIndex)):
                    print(len(lonIndex), pos)
                    target_temp[:, pos] = ds.variables['temperature'][0, :, latIndex[pos], lonIndex[pos]]
                    target_salt[:, pos] = ds.variables['salinity'][0, :, latIndex[pos], lonIndex[pos]]

                # Make directories
                save_dir_transect_temp = os.path.join(save_dir, transect, 'rtofs', 'temperature')
                os.makedirs(save_dir_transect_temp, exist_ok=True)

                save_dir_transect_salinity = os.path.join(save_dir, transect, 'rtofs', 'salinity')
                os.makedirs(save_dir_transect_salinity, exist_ok=True)

                # Contour argument inputs (Temperature)
                targs = {}
                targs['cmap'] = cmocean.cm.thermal
                targs['title'] = f'RTOFS Temperature Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
                targs['save_file'] = os.path.join(save_dir_transect_temp, f'rtofs_{transect}_transect_temperature-{date_str}.png')
                targs['levels'] = dict(
                    deep=transects[transect]['limits']['temperature']['deep'],
                    shallow=transects[transect]['limits']['temperature']['shallow']
                )
                plot_transects(X, -depth, target_temp, **targs)

                # Contour argument inputs (Salinity)
                sargs = {}
                sargs['cmap'] = cmocean.cm.haline
                sargs['title'] = f'RTOFS Salinity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
                sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'rtofs_{transect}_transect_salinity-{date_str}.png')
                sargs['levels'] = dict(
                    deep=transects[transect]['limits']['salinity']['deep'],
                    shallow=transects[transect]['limits']['salinity']['shallow']
                )
                plot_transects(X, -depth, target_salt, **sargs)

                # Contour argument inputs (Temperature)
                targs = {}
                targs['cmap'] = cmocean.cm.thermal
                targs['title'] = f'RTOFS Temperature Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
                targs['save_file'] = os.path.join(save_dir_transect_temp, f'rtofs_{transect}_transect_temperature-300m-{date_str}.png')
                targs['levels'] = dict(
                    deep=transects[transect]['limits']['temperature']['deep'],
                    shallow=transects[transect]['limits']['temperature']['shallow']
                )
                plot_transect(X, -depth, target_temp, **targs)

                # Contour argument inputs (Salinity)
                sargs = {}
                sargs['cmap'] = cmocean.cm.haline
                sargs['title'] = f'RTOFS Salinity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
                sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'rtofs_{transect}_transect_salinity-300m-{date_str}.png')
                sargs['levels'] = dict(
                    deep=transects[transect]['limits']['salinity']['deep'],
                    shallow=transects[transect]['limits']['salinity']['shallow']
                )
                plot_transect(X, -depth, target_salt, **sargs)

    except OSError:
        continue
