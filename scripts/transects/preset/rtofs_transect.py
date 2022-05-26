from glob import glob
import os
import cmocean
import xarray as xr
import datetime as dt
from hurricanes.calc import calculate_transect, convert_ll_to_model_ll
from hurricanes.plotting import plot_transects, region_subplot, plot_region_quiver, plot_transect
from hurricanes.common import rename_model_variables
from hurricanes.storms import custom_transect
from hurricanes.limits import transects, limits_regions
import numpy as np
import cartopy.crs as ccrs
import pandas as pd


url = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'
save_dir = '/Users/mikesmith/Documents/github/rucool/hurricanes/plots/transects/'
model = 'rtofs'
transect_spacing = 0.01/20
# url = '/home/hurricaneadm/data/rtofs/'
# save_dir = '/www/web/rucool/hurricane/model_comparisons/transects/rtofs/'

time_end = dt.datetime.today()
# time_end = dt.datetime(2022, 3, 7)
days = 4

# transect coordinates and variable limits
transects = transects()
limits = limits_regions()

bathymetry = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/bathymetry/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'
if bathymetry:
    bathy = xr.open_dataset(bathymetry)

# Get today and yesterday dates
date_list = [time_end - dt.timedelta(days=x) for x in range(days)]
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

            save_dir_transect_u = os.path.join(save_dir, transect, model, 'u')
            os.makedirs(save_dir_transect_u, exist_ok=True)

            save_dir_transect_v = os.path.join(save_dir, transect, model, 'v')
            os.makedirs(save_dir_transect_v, exist_ok=True)

            save_dir_transect = os.path.join(save_dir, transect)
            os.makedirs(save_dir_transect, exist_ok=True)

            region_limits = limits[transects[transect]['region']]
            extent = region_limits['lonlat']

            # calculate longitude and latitude of transect lines
            X, Y, _ = calculate_transect(x1, y1, x2, y2, transect_spacing)

            # Conversion from glider longitude and latitude to RTOFS convention
            target_lon, target_lat = convert_ll_to_model_ll(X, Y, model)

            date_str = pd.to_datetime(rtofs.time.data[0]).strftime("%Y-%m-%dT%H%M%SZ")

            var_dict, depth, lon, lat = custom_transect(rtofs, ['temperature', 'salinity', 'u', 'v'], target_lon, target_lat, model)

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

            # Make a plot of the line 

            save_map_file = os.path.join(save_dir_transect, f'{transect}_{date_str}_map.png')
            import matplotlib.pyplot as plt
            # plot_region(
            #     rtofs['temperature'].sel(depth=0),
            #     limits[transects[transect]['region']]['lonlat'],
            #     f'{transects[transect]["region"]}\nTransect - [{x1}W, {y1}N, {x2}W, {y2}N]\n{date_str}',
            #     save_map_file,
            #     **vargs
            #     )
            
            # vargs = {}
            # # if item['limits'][0]:
            # vargs['vmin'] = item['limits'][0]
            # vargs['vmax'] = item['limits'][1]
            # vargs['transform'] = transform['data']
            # vargs['cmap'] = cmaps(rtofs[k].name)
            # vargs['ticks'] = ticks

            # if k == 'sea_surface_height':
            #     continue
            # elif k == 'salinity':
            #     vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            # elif k == 'temperature':
            #     vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            fig, ax = plt.subplots(
                figsize=(12, 8),
                subplot_kw=dict(projection=ccrs.Mercator()),
                )
            ax.set_xlabel('Longitude', fontsize=14)
            ax.set_ylabel('Latitude', fontsize=14)
            
            ax = region_subplot(fig, ax, extent, da=rtofs['temperature'].sel(depth=0), 
                                title='RTOFS',
                                bathy=bathy,
                                **vargs)
            plt.savefig(save_map_file, dpi=150, bbox_inches='tight', pad_inches=0.1)

            qargs = {}
            qargs['transform'] = ccrs.PlateCarree()
            qargs['cmap'] = cmocean.cm.speed
            
            # qargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], region_limits['temperature'][0]['limits'][2])
            if bathy:
                vargs['bathy'] = bathy.sel(
                lon=slice(extent[0]-1, extent[1]+1),
                lat=slice(extent[2]-1, extent[3]+1)
    )
            save_map_file = os.path.join(save_dir_transect, f'{transect}_{date_str}_quiver_map.png')
            
            plot_region_quiver(rtofs.sel(depth=0),
                        limits[transects[transect]['region']]['lonlat'],
                        f'{transects[transect]["region"]}\n{date_str}',
                        save_map_file,
                        **vargs)

            # # Contour argument inputs (Temperature)
            # targs = {}
            # targs['cmap'] = cmocean.cm.thermal
            # targs['title'] = f'{model.upper()} Temperature Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            # targs['save_file'] = os.path.join(save_dir_transect_temp, f'{model}_{transect}_transect_temperature-{date_str}.png')
            # targs['levels'] = dict(
            #     deep=transects[transect]['limits']['temperature']['deep'],
            #     shallow=transects[transect]['limits']['temperature']['shallow']
            # )
            # targs['isobath'] = transects[transect]['limits']['temperature']['isobath']

            # if transects[transect]['xaxis'] == 'longitude':
            #     plot_transects(lon, depth, var_dict['temperature'], 'Longitude', **targs)
            # elif transects[transect]['xaxis'] == 'latitude':
            #     plot_transects(lat, depth, var_dict['temperature'], 'Latitude', **targs)

            # # Contour argument inputs (Salinity)
            # sargs = {}
            # sargs['cmap'] = cmocean.cm.haline
            # sargs['title'] = f'{model.upper()} Salinity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            # sargs['save_file'] = os.path.join(save_dir_transect_salinity, f'{model}_{transect}_transect_salinity--{date_str}.png')
            # sargs['levels'] = dict(
            #     deep=transects[transect]['limits']['salinity']['deep'],
            #     shallow=transects[transect]['limits']['salinity']['shallow']
            # )
            # if transects[transect]['xaxis'] == 'longitude':
            #     plot_transects(lon, depth, var_dict['salinity'], 'Longitude', **sargs)
            # elif transects[transect]['xaxis'] == 'latitude':
            #     plot_transects(lat, depth, var_dict['salinity'], 'Latitude', **sargs)

            # # Contour argument inputs (u)
            # sargs = {}
            # sargs['cmap'] = cmocean.cm.balance
            # sargs['title'] = f'{model.upper()} Eastward (u) Sea Water Velocity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            # sargs['save_file'] = os.path.join(save_dir_transect_u, f'{model}_{transect}_transect_u--{date_str}.png')
            # sargs['levels'] = dict(
            #     deep=transects[transect]['limits']['u']['deep'],
            #     # shallow=transects[transect]['limits']['u']['shallow']
            # )
            
            # if transects[transect]['xaxis'] == 'longitude':
            #     # plot_transects(lon, depth, var_dict['u'], 'Longitude', **sargs)
            #     plot_transect(lon, depth, var_dict['u'], 'Longitude', **sargs)
            # elif transects[transect]['xaxis'] == 'latitude':
            #     # plot_transects(lat, depth, var_dict['u'], 'Latitude', **sargs)
            #     plot_transect(lon, depth, var_dict['u'], 'Latitude', **sargs)

            # # Contour argument inputs (v)
            # sargs = {}
            # sargs['cmap'] = cmocean.cm.balance
            # sargs['title'] = f'{model.upper()} Northward (v) Sea Water Velocity Transect\n {transect_str} @ {[x1, y1, x2, y2]}\n{date_str}'
            # sargs['save_file'] = os.path.join(save_dir_transect_v, f'{model}_{transect}_transect_v--{date_str}.png')
            # sargs['levels'] = dict(
            #     deep=transects[transect]['limits']['v']['deep'],
            #     # shallow=transects[transect]['limits']['v']['shallow']
            # )
            # if transects[transect]['xaxis'] == 'longitude':
            #     # plot_transects(lon, depth, var_dict['v'], 'Longitude', **sargs)
            #     plot_transect(lon, depth, var_dict['v'], 'Longitude', **sargs)
            # elif transects[transect]['xaxis'] == 'latitude':
            #     # plot_transects(lat, depth, var_dict['v'], 'Latitude', **sargs)
            #     plot_transect(lon, depth, var_dict['v'], 'Latitude', **sargs)