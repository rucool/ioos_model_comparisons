from glob import glob
import os
import cmocean
import xarray as xr
import datetime as dt
from src.calc import calculate_transect, convert_ll_to_model_ll
from src.plotting import plot_transects, plot_region
from src.common import rename_model_variables
from src.storms import custom_transect
from src.limits import transects, limits_regions
import numpy as np
import cartopy.crs as ccrs
import pandas as pd


# url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/transects/'
model = 'rtofs'
transect_spacing = 0.05/20
url = '/home/hurricaneadm/data/rtofs/'
save_dir = '/www/web/rucool/hurricane/model_comparisons/transects/rtofs/'

days = 4

# transect coordinates and variable limits
transects = transects()
limits = limits_regions()

# Get today and yesterday dates
date_list = [dt.datetime.today() - dt.timedelta(days=x) for x in range(days)]
rtofs_files = [glob(os.path.join(url, x.strftime('rtofs.%Y%m%d'), '*.nc')) for x in date_list]
rtofs_files = sorted([inner for outer in rtofs_files for inner in outer])


for f in rtofs_files:
    with xr.open_dataset(f) as rtofs:
        rtofs = rename_model_variables(rtofs, model)
        for transect in transects.keys():
            transect_str = " ".join(transect.split("_")).title()
            x1 = transects[transect]['extent'][0]
            y1 = transects[transect]['extent'][1]
            x2 = transects[transect]['extent'][2]
            y2 = transects[transect]['extent'][3]

            # Make directories
            save_dir_transect_temp = os.path.join(save_dir, transect, model, 'temperature')
            os.makedirs(save_dir_transect_temp, exist_ok=True)

            save_dir_transect_salinity = os.path.join(save_dir, transect, model, 'salinity')
            os.makedirs(save_dir_transect_salinity, exist_ok=True)

            save_dir_transect = os.path.join(save_dir, transect)
            os.makedirs(save_dir_transect, exist_ok=True)

            region_limits = limits[transects[transect]['region']]

            # calculate longitude and latitude of transect lines
            X, Y, _ = calculate_transect(x1, y1, x2, y2, transect_spacing)

            # Conversion from glider longitude and latitude to RTOFS convention
            target_lon, target_lat = convert_ll_to_model_ll(X, Y, model)

            date_str = pd.to_datetime(rtofs.time.data[0]).strftime("%Y-%m-%dT%H%M%SZ")

            var_dict, depth, lon, lat = custom_transect(rtofs, ['temperature', 'salinity'], target_lon, target_lat, model)

            vargs = {}
            vargs['vmin'] = region_limits['temperature'][0]['limits'][0]
            vargs['vmax'] = region_limits['temperature'][0]['limits'][1]
            vargs['transform'] = ccrs.PlateCarree()
            vargs['cmap'] = cmocean.cm.thermal
            vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], region_limits['temperature'][0]['limits'][2])

            try:
                vargs.pop('vmin'), vargs.pop('vmax')
            except KeyError:
                pass

            vargs['transects'] = pd.DataFrame(dict(lon=target_lon, lat=target_lat))

            save_map_file = os.path.join(save_dir_transect, f'{transect}_map.png')
            plot_region(rtofs['temperature'].sel(depth=0),
                        limits[transects[transect]['region']]['lonlat'],
                        f'Transect - [{x1}W, {y1}N, {x2}W, {y2}N]\n{date_str}',
                        save_map_file,
                        **vargs)

            # Contour argument inputs (Temperature)
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['title'] = f'{model.upper()} Temperature Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            targs['save_file'] = os.path.join(save_dir_transect_temp, f'{model}_{transect}_transect_temperature-{date_str}.png')
            targs['levels'] = dict(
                deep=transects[transect]['limits']['temperature']['deep'],
                shallow=transects[transect]['limits']['temperature']['shallow']
            )
            targs['isobath'] = transects[transect]['limits']['temperature']['isobath']

            if transects[transect]['xaxis'] == 'longitude':
                plot_transects(lon, depth, var_dict['temperature'], 'Longitude', **targs)
            elif transects[transect]['xaxis'] == 'latitude':
                plot_transects(lat, depth, var_dict['temperature'], 'Latitude', **targs)

            # Contour argument inputs (Salinity)
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['title'] = f'{model.upper()} Salinity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'{model}_{transect}_transect_salinity--{date_str}.png')
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['salinity']['deep'],
                shallow=transects[transect]['limits']['salinity']['shallow']
            )
            if transects[transect]['xaxis'] == 'longitude':
                plot_transects(lon, depth, var_dict['salinity'], 'Longitude', **sargs)
            elif transects[transect]['xaxis'] == 'latitude':
                plot_transects(lat, depth, var_dict['salinity'], 'Latitude', **sargs)