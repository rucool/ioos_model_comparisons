import cmocean
import xarray as xr
import datetime as dt
import os
from ioos_model_comparisons.calc import calculate_transect, convert_ll_to_model_ll
from ioos_model_comparisons.plotting import plot_transects
from ioos_model_comparisons.common import rename_model_variables
from ioos_model_comparisons.storms import custom_transect
from ioos_model_comparisons.limits import transects, limits_regions
import pandas as pd

save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/transects/'
# save_dir = '/www/web/rucool/hurricane/model_comparisons/transects/gofs/'
model = 'gofs'
transect_spacing = 0.05/20

days = 3

# Get today and yesterdays date
# today = dt.date.today()
today = dt.date(2022, 1, 26)
ranges = pd.date_range(today - dt.timedelta(days=days), today + dt.timedelta(days=1), freq='6H')
# pd.date_range(yesterday, today, periods=4)

transects = transects()
limits = limits_regions()

with xr.open_dataset('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0', drop_variables='tau') as ds:
    ds = rename_model_variables(ds, model)
    dst = ds.sel(time=ranges[ranges < pd.to_datetime(ds.time.max().data)])

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

        save_dir_transect_u = os.path.join(save_dir, transect, model, 'u')
        os.makedirs(save_dir_transect_u, exist_ok=True)

        save_dir_transect_v = os.path.join(save_dir, transect, model, 'v')
        os.makedirs(save_dir_transect_v, exist_ok=True)

        # calculate longitude and latitude of transect lines
        X, Y, _ = calculate_transect(x1, y1, x2, y2, transect_spacing)

        # Conversion from glider longitude and latitude to RTOFS convention
        target_lon, target_lat = convert_ll_to_model_ll(X, Y, model)

        # lon_index, lat_index = find_lonlat_index(dst, target_lon, target_lat, 'gofs')

        for t in dst.time:
        # for t in [dt.datetime(2021, 12, 16, 6)]:
            print(t)
            GOFS = dst.sel(time=t)  # Select the latest time
            date_str = GOFS["time"].dt.strftime("%Y-%m-%dT%H%M%SZ").data

            temp_save_file = os.path.join(save_dir_transect_temp, f'{model}_{transect}_transect_temp-{date_str}.png')
            salt_save_file = os.path.join(save_dir_transect_salinity, f'{model}_{transect}_transect_salinity-{date_str}.png')

            # if os.path.isfile(temp_save_file):
            #     continue

            # if os.path.isfile(salt_save_file):
            #     continue

            # # if transect has a straight line, if it isnt found exactly on the grid, you will not get data back
            # # the following ensures that you get the nearest neighbor to each lon and lat min and max
            # closest_to_lon_min = GOFS.sel(lon=target_lon.min(), method='nearest')
            # closest_to_lon_max = GOFS.sel(lon=target_lon.max(), method='nearest')
            # closest_to_lat_min = GOFS.sel(lat=target_lat.min(), method='nearest')
            # closest_to_lat_max = GOFS.sel(lat=target_lat.max(), method='nearest')
            # c = GOFS.sel(lon=slice(closest_to_lon_min.lon.data, closest_to_lon_max.lon.data), lat=slice(closest_to_lat_min.lat.data, closest_to_lat_max.lat.data)).squeeze()
            #

            var_dict, depth, lon, lat = custom_transect(GOFS, ['temperature', 'salinity', 'u', 'v'], target_lon, target_lat, 'gofs')

            # Contour argument inputs (Temperature)
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['title'] = f'{model.upper()} Temperature Transect\n Feature: {transect_str} {[x1, y1, x2, y2]}\n{date_str}'
            targs['save_file'] = temp_save_file
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
            sargs['title'] = f'{model.upper()} Salinity Transect\n Feature: {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = salt_save_file
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['salinity']['deep'],
                shallow=transects[transect]['limits']['salinity']['shallow']
            )
            if transects[transect]['xaxis'] == 'longitude':
                plot_transects(lon, depth, var_dict['salinity'], 'Longitude', **sargs)
            elif transects[transect]['xaxis'] == 'latitude':
                plot_transects(lat, depth, var_dict['salinity'], 'Latitude', **sargs)

            # Contour argument inputs (u)
            sargs = {}
            sargs['cmap'] = cmocean.cm.balance
            sargs['title'] = f'{model.upper()} Eastward (u) Sea Water Velocity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = os.path.join(save_dir_transect_u, f'{model}_{transect}_transect_u--{date_str}.png')
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['u']['deep'],
                shallow=transects[transect]['limits']['u']['shallow']
            )
            if transects[transect]['xaxis'] == 'longitude':
                plot_transects(lon, depth, var_dict['u'], 'Longitude', **sargs)
            elif transects[transect]['xaxis'] == 'latitude':
                plot_transects(lat, depth, var_dict['u'], 'Latitude', **sargs)

            # Contour argument inputs (v)
            sargs = {}
            sargs['cmap'] = cmocean.cm.balance
            sargs['title'] = f'{model.upper()} Northward (v) Sea Water Velocity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = os.path.join(save_dir_transect_v, f'{model}_{transect}_transect_v--{date_str}.png')
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['v']['deep'],
                shallow=transects[transect]['limits']['v']['shallow']
            )
            if transects[transect]['xaxis'] == 'longitude':
                plot_transects(lon, depth, var_dict['v'], 'Longitude', **sargs)
            elif transects[transect]['xaxis'] == 'latitude':
                plot_transects(lat, depth, var_dict['v'], 'Latitude', **sargs)