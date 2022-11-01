from tropycal import realtime
from datetime import datetime as dt
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter as gfilt
from tropycal.utils.colors import get_colors_sshws
from tropycal.utils.generic_utils import wind_to_category, generate_nhc_cone
from tropycal import constants
import matplotlib.colors as mcolors
from cool_maps.plot import create
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patheffects as path_effects

# realtime_obj = realtime.Realtime()
# storms = [realtime_obj.get_storm(key) for key in realtime_obj.list_active_storms(basin='north_atlantic')]

# # realtime_obj.plot_summary(domain={'w':-100,'e':-10,'s':4,'n':60})

# #Get realtime forecasts
# forecasts = []
# for key in realtime_obj.storms:
#     if realtime_obj[key].invest == False:
#         try:
#             forecasts.append(realtime_obj.get_storm(key).get_forecast_realtime(True))
#         except:
#             forecasts.append({})
#     else:
#         forecasts.append({})
# forecasts = [entry if 'init' in entry.keys() and (dt.utcnow() - entry['init']).total_seconds() / 3600.0 <= 12 else {} for entry in forecasts]
# storms = [realtime_obj.get_storm(key) for key in realtime_obj.storms]

# fig, ax = create(extent=[-99, -79, 18, 31])

# two_prop={
#     'plot':True,
#     'fontsize':12,
#     'days':5,
#     'ms':15}
# invest_prop={
#     'plot':True,
#     'fontsize':12,
#     'linewidth':0.8,
#     'linecolor':'k',
#     'linestyle':'dotted',
#     'ms':14
#     }
# storm_prop={
#     'plot':True,
#     'fontsize':12,
#     'linewidth':0.8,
#     'linecolor':'k',
#     'linestyle':'dotted',
#     'fillcolor':'category',
#     'label_category':True,
#     'ms':14
#     }
# cone_prop={
#     'plot':True,
#     'linewidth':1.5,
#     'linecolor':'k',
#     'alpha':0.25,
#     'days':5,
#     'fillcolor':'category',
#     'label_category':True,
#     'ms':12}

# bbox_prop = {'facecolor':'white',
#              'alpha':0.5,
#              'edgecolor':'black',
#              'boxstyle':'round,pad=0.3'}

# zorder = 80

# def plot_storms(ax, storms, forecasts, zorder=0, proj=ccrs.PlateCarree()):
#     #Iterate over all storms
#     for storm_idx, storm in enumerate(storms):
#         # Plot invests
#         if storm.invest:
#             #Test
#             ax.plot(
#                 storm.lon[-1],
#                 storm.lat[-1],
#                 'X',
#                 ms=invest_prop['ms'],
#                 color='k',
#                 transform=proj,
#                 zorder=zorder)
            
#             #Transform coordinates for label
#             x1, y1 = ax.projection.transform_point(storm.lon[-1], storm.lat[-1], proj)
#             x2, y2 = ax.transData.transform((x1, y1))
#             x, y = ax.transAxes.inverted().transform((x2, y2))

#             # plot same point but using axes coordinates
#             a = ax.text(x,
#                         y-0.05,
#                         f"{storm.name.title()}",
#                         ha='center',
#                         va='top',
#                         transform=ax.transAxes,
#                         zorder=90,
#                         fontweight='bold',
#                         fontsize=invest_prop['fontsize'],
#                         clip_on=True,
#                         bbox=bbox_prop)
            
#             a.set_path_effects(
#                 [path_effects.Stroke(linewidth=0.5, foreground='w'), path_effects.Normal()]
#                 )
            
#             #Plot archive track
#             if invest_prop['linewidth'] > 0:
#                 ax.plot(
#                     storm.lon,
#                     storm.lat,
#                     color=invest_prop['linecolor'],
#                     linestyle=invest_prop['linestyle'],
#                     zorder=zorder+1,
#                     transform=proj
#                     )
        
#         #Plot tropical cyclones
#         else:
#             #Label dot
#             category = str(wind_to_category(storm.vmax[-1]))
#             color = get_colors_sshws(storm.vmax[-1])
#             if category == "0": 
#                 category = 'S'
#             if category == "-1":
#                 category = 'D'
#             if np.isnan(storm.vmax[-1]):
#                 category = 'U'
#                 color = 'w'

#             # 
#             if storm_prop['fillcolor']:
#                 if storm_prop['fillcolor'] != 'category': 
#                     color = storm_prop['fillcolor']

#                 # Plot red circle with black edge color where the storm is
#                 ax.plot(storm.lon[-1], storm.lat[-1], 'o', 
#                         ms=storm_prop['ms'],
#                         markeredgecolor='black',
#                         color=color,
#                         transform=proj,
#                         zorder=zorder+1)

#                 # Print hurricane category on red/black dot
#                 if storm_prop['label_category']:
#                     color = mcolors.to_rgb(color)
#                     red, green, blue = color
#                     textcolor = 'w'
#                     if (red*0.299 + green*0.587 + blue*0.114) > (160.0/255.0):
#                         textcolor = 'k'
#                     ax.text(storm.lon[-1], storm.lat[-1], category, 
#                             fontsize=storm_prop['ms']*0.83,
#                             ha='center',
#                             va='center',
#                             color=textcolor,
#                             zorder=90,
#                             transform=proj,
#                             clip_on=True)
#             else:
#                 ax.plot(
#                     storm.lon[-1],
#                     storm.lat[-1],
#                     'o',
#                     ms=storm_prop['ms'],
#                     color='none',
#                     mec='k',
#                     mew=3.0,
#                     transform=proj,
#                     zorder=zorder+1)

#                 ax.plot(
#                     storm.lon[-1],
#                     storm.lat[-1],
#                     'o',
#                     ms=storm_prop['ms'],
#                     color='none',
#                     mec='r',
#                     mew=2.0,
#                     transform=proj,
#                     zorder=zorder+1)                        
            
#             #Transform coordinates for label
#             x1, y1 = ax.projection.transform_point(storm.lon[-1], storm.lat[-1], proj)
#             x2, y2 = ax.transData.transform((x1, y1))
#             x, y = ax.transAxes.inverted().transform((x2, y2))

#             # plot same point but using axes coordinates
#             a = ax.text(x+0.04, y-0.03, f"{storm.name.title()}", 
#                         ha='center',
#                         va='top',
#                         transform=ax.transAxes,
#                         zorder=zorder+1,
#                         fontweight='bold',
#                         fontsize=storm_prop['fontsize'],
#                         clip_on=True,
#                         bbox=bbox_prop)
#             a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'), path_effects.Normal()])
            
#             #Plot previous track
#             ax.plot(storm.lon, storm.lat, color=storm_prop['linecolor'],
#                     linestyle=storm_prop['linestyle'],
#                     zorder=zorder+1,
#                     transform=proj)
                
#             #Plot cone
#             forecast_dict = forecasts[storm_idx]
            
#             # try:
#             #Fix longitudes for cone if crossing dateline
#             if np.nanmax(forecast_dict['lon']) > 165 or np.nanmin(forecast_dict['lon']) < -165:
#                 forecast_dict['lon'] = [i if i > 0 else i + 360.0 for i in forecast_dict['lon']]
#             cone = generate_nhc_cone(forecast_dict, storm.basin, cone_days=cone_prop['days'])

#             #Plot cone
#             if cone_prop['alpha'] > 0 and storm.basin in constants.NHC_BASINS:
#                 cone_2d = cone['cone']
#                 cone_2d = ndimage.gaussian_filter(cone_2d, sigma=0.5, order=0)

#                 # Cone shading
#                 ax.contourf(cone['lon2d'], cone['lat2d'], cone_2d, [0.9,1.1], 
#                             colors=['#ffffff', '#ffffff'],
#                             alpha=cone_prop['alpha'],
#                             zorder=zorder+1,
#                             transform=proj)

#                 # Cone outline
#                 ax.contour(cone['lon2d'], cone['lat2d'], cone_2d, [0.9], 
#                             linewidths=1.5,
#                             colors=['k'],
#                             zorder=zorder+1,
#                             transform=proj)

#             #Plot forecast center line & account for dateline crossing
#             ax.plot(cone['center_lon'], cone['center_lat'],
#                     color='k',
#                     linewidth=2.0,
#                     zorder=zorder+2,
#                     transform=proj
#                     ) 

#             #Plot forecast dots
#             for idx in range(len(forecast_dict['lat'])):
#                 if forecast_dict['fhr'][idx]/24.0 > cone_prop['days']:
#                     continue
                
#                 if cone_prop['ms'] == 0:
#                     continue
                
#                 color = get_colors_sshws(forecast_dict['vmax'][idx])
                
#                 if np.isnan(forecast_dict['vmax'][idx]): 
#                     color = 'w'
                    
#                 if cone_prop['fillcolor'] != 'category':
#                     color = cone_prop['fillcolor']
                
#                 marker = 'o'
#                 if forecast_dict['type'][idx] not in constants.TROPICAL_STORM_TYPES: 
#                     marker = '^'
                    
#                 if np.isnan(forecast_dict['vmax'][idx]): 
#                     marker = 'o'
                    
#                 ax.plot(forecast_dict['lon'][idx], forecast_dict['lat'][idx],
#                         marker,
#                         ms=cone_prop['ms'],
#                         mfc=color,
#                         mec='k',
#                         zorder=zorder+2,
#                         transform=proj,
#                         clip_on=True)

#                 if cone_prop['label_category'] and marker == 'o':
#                     category = str(wind_to_category(forecast_dict['vmax'][idx]))
#                     if category == "0": 
#                         category = 'S'
#                     if category == "-1":
#                         category = 'D'
#                     if np.isnan(forecast_dict['vmax'][idx]): 
#                         category = 'U'

#                     color = mcolors.to_rgb(color)
#                     red, green, blue = color
#                     textcolor = 'w'
#                     if (red*0.299 + green*0.587 + blue*0.114) > (160.0/255.0):
#                         textcolor = 'k'

#                     ax.text(forecast_dict['lon'][idx], forecast_dict['lat'][idx],
#                             category,
#                             fontsize=cone_prop['ms']*0.81,
#                             ha='center',
#                             va='center',
#                             color=textcolor,
#                             zorder=zorder+3,
#                             transform=proj,
#                             clip_on=True)
#             # except:
#                 # pass

# plot_storms(ax, storms, forecasts, zorder=zorder)
# plt.savefig('/Users/mikesmith/Documents/ian-test.png', dpi=150)

# %%
from tropycal import tracks
import pandas as pd
from tropycal import realtime

realtime_obj = realtime.Realtime()
realtime_obj.list_active_storms(basin='north_atlantic')
basin = tracks.TrackDataset(basin='north_atlantic', source='hurdat', include_btk=True)
storm = basin.get_storm(('ian', 2022))

#%%
# storm.plot_nhc_forecast(pd.Timestamp(2022, 9, 28, 2, 0, 0))
storm.plot()
# plt.show()

