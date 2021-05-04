import cmocean
import xarray as xr
import datetime as dt
import os
from src.calc import calculate_transect, convert_ll_to_model_ll
from src.plotting import plot_transect, plot_transects
from src.common import transects

# save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/transects/gofs/'
save_dir = '/www/home/michaesm/public_html/hurricanes/plots/transects/gofs/'

days = 2

# Get today and yesterdays date
today = dt.date.today()
yesterday = today - dt.timedelta(days=days)
# pd.date_range(yesterday, today, periods=4)
extent = [-100+360, -80+360, 18, 32]

transects = transects()

with xr.open_dataset('https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0', drop_variables='tau') as ds:
    dst = ds.sel(time=slice(yesterday, today))

    for t in dst.time:
        GOFS = dst.sel(time=t)  # Select the latest time
        date_str = GOFS["time"].dt.strftime("%Y-%m-%dT%H%M%SZ").data

        for transect in transects.keys():
            transect_str = " ".join(transect.split("_")).title()
            x1 = transects[transect]['extent'][0]
            y1 = transects[transect]['extent'][1]
            x2 = transects[transect]['extent'][2]
            y2 = transects[transect]['extent'][3]

            # calculate longitude and latitude of transect lines
            X, Y, _ = calculate_transect(x1, y1, x2, y2)

            # Conversion from glider longitude and latitude to RTOFS convention
            target_lon, target_lat = convert_ll_to_model_ll(X, Y, 'gofs')

            # if transect has a straight line, if it isnt found exactly on the grid, you will not get data back
            # the following ensures that you get the nearest neighbor to each lon and lat min and max
            closest_to_lon_min = GOFS.sel(lon=target_lon.min(), method='nearest')
            closest_to_lon_max = GOFS.sel(lon=target_lon.max(), method='nearest')
            closest_to_lat_min = GOFS.sel(lat=target_lat.min(), method='nearest')
            closest_to_lat_max = GOFS.sel(lat=target_lat.max(), method='nearest')

            c = GOFS.sel(lon=slice(closest_to_lon_min.lon.data, closest_to_lon_max.lon.data),
                         lat=slice(closest_to_lat_min.lat.data, closest_to_lat_max.lat.data)).squeeze()

            # Make directories
            save_dir_transect_temp = os.path.join(save_dir, transect, 'gofs', 'temperature')
            os.makedirs(save_dir_transect_temp, exist_ok=True)

            save_dir_transect_salinity = os.path.join(save_dir, transect, 'gofs', 'salinity')
            os.makedirs(save_dir_transect_salinity, exist_ok=True)

            # Contour argument inputs (Temperature)
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['title'] = f'GOFS Temperature Transect\n Feature: {transect_str} {[x1, y1, x2, y2]}\n{date_str}'
            targs['save_file'] = os.path.join(save_dir_transect_temp, f'gofs_{transect}_transect_temp-{date_str}.png')
            targs['levels'] = dict(
                deep=transects[transect]['limits']['temperature']['deep'],
                shallow=transects[transect]['limits']['temperature']['shallow']
            )
            # plot_transects(X, -depth, target_tempGOFS, **targs)
            plot_transects(c.lon, -c.depth, c.water_temp.squeeze(), **targs)


            # Contour argument inputs (Salinity)
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['title'] = f'GOFS Salinity Transect\n Feature: {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'gofs_{transect}_transect_salinity-{date_str}.png')
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['salinity']['deep'],
                shallow=transects[transect]['limits']['salinity']['shallow']
            )
            # plot_transects(X, -depth, target_saltGOFS, **sargs)
            plot_transects(c.lon, -c.depth, c.salinity.squeeze(), **sargs)


            # Contour argument inputs (Temperature)
            targs = {}
            targs['cmap'] = cmocean.cm.thermal
            targs['title'] = f'GOFS Temperature Transect\n Feature: {transect_str} {[x1, y1, x2, y2]}\n{date_str}'
            targs['save_file'] = os.path.join(save_dir_transect_temp, f'gofs_{transect}_transect_temp-300m-{date_str}.png')
            targs['levels'] = dict(
                deep=transects[transect]['limits']['temperature']['deep'],
                shallow=transects[transect]['limits']['temperature']['shallow']
            )
            # plot_transect(X, -depth, target_tempGOFS, **targs)
            plot_transect(c.lon, -c.depth, c.water_temp.squeeze(), **targs)


            # Contour argument inputs (Salinity)
            sargs = {}
            sargs['cmap'] = cmocean.cm.haline
            sargs['title'] = f'GOFS Salinity Transect\n Feature: {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'gofs_{transect}_transect_salinity-300m-{date_str}.png')
            sargs['levels'] = dict(
                deep=transects[transect]['limits']['salinity']['deep'],
                shallow=transects[transect]['limits']['salinity']['shallow']
            )
            # plot_transect(X, -depth, target_saltGOFS, **sargs)
            plot_transect(c.lon, -c.depth, c.salinity.squeeze(), **sargs)