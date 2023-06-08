import os
import pickle
import warnings
from itertools import cycle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxesSubplot
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
from oceans.ocfis import spdir2uv, uv2spdir
from shapely.geometry.polygon import LinearRing
from ioos_model_comparisons.calc import dd2dms
import ioos_model_comparisons.configs as conf
from scipy.io import loadmat
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from ioos_model_comparisons.plotting import (add_features, 
                                 add_bathymetry, add_ticks,
                                 map_add_eez, map_add_currents, 
                                 remove_quiver_handles,
                                 plot_regional_assets)
import cool_maps.plot as cplt
# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

proj = dict(
    map=ccrs.Mercator(), # the projection that you want the map to be in
    data=ccrs.PlateCarree() # the projection that the data is. 
    )

def surface_current_fronts(ds1, ds2, ds3, ds4, region,
                           bathy=None,
                           argo=None,
                           gliders=None,
                           eez=False,
                           cols=6,
                           transform=dict(
                            map=proj['map'], 
                            data=proj['data']
                            ),
                           path_save=os.getcwd(),
                           figsize=(14,8),
                           dpi=150,
                           overwrite=False
                           ):
    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']
    
    # Plot currents with magnitude and direction
    quiver_dir = path_save / f"surface_currents_fronts" / time.strftime('%Y/%m')
    os.makedirs(quiver_dir, exist_ok=True)

    # Generate filename
    sname = f'gom_{time.strftime("%Y-%m-%dT%H%M%SZ")}_surface_current_fronts-subplots'
    save_file_q = quiver_dir / f"{sname}.png"
    
    # sname2 = f'gom_{time.strftime("%Y-%m-%dT%H%M%SZ")}_surface_current_fronts-overlaid'
    # save_file_q_2 = quiver_dir / f"{sname2}.png"

    # Check if filename already exists
    if save_file_q.is_file():
        if not overwrite:
            print(f"{sname} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{sname} exists. Overwrite: True. Replotting.")

    # Convert u and v radial velocities to magnitude
    _, mag_r = uv2spdir(ds1['u'], ds1['v'])
    _, mag_g = uv2spdir(ds2['u'], ds2['v'])
    _, mag_c = uv2spdir(ds3['u'], ds3['v'])
    _, mag_a = uv2spdir(ds4['u'], ds4['v'])

    # Initialize first figure
    # Four Panel Plot
    fig, _ = plt.subplot_mosaic(
        """
        RGL
        CAL
        """,
        figsize=(16,9),
        layout="constrained",
        subplot_kw={
            'projection': proj['map']
            },
        # gridspec_kw={
        #     # set the height ratios between the rows
        #     "height_ratios": [1, 1, .5],
        #     # set the width ratios between the columns
        #     "width_ratios": [1, 1],
        #     },
        )
    
    axs = fig.axes
    ax1 = axs[0] # rtofs
    ax2 = axs[1] # gofs
    ax5 = axs[2] # legend
    ax3 = axs[3] # copernicus
    ax4 = axs[4] # amseas

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    
    # Loop through subplot axes and add stuff to maps
    for ax in [ax1, ax2, ax3, ax4]:
        # Make the map pretty
        map_add_features(ax, extent)
        if bathy:    
            map_add_bathymetry(ax,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5
                        )
            map_add_ticks(ax, extent)

        if eez:
            eez_h = map_add_eez(ax, zorder=1, color='red', linewidth=1)

        plot_regional_assets(ax, **rargs)
        gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree(), zorder=150)
        # gl.xlocator = mticker.FixedLocator(tick0x)
        # gl.ylocator = mticker.FixedLocator(tick0y)
        # ax.gridlines(crs=transform['data'])

    # Add speed contour
    # ax1.contour(ds1_depth['lon'], ds1_depth['lat'], mag_r, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)
    # ax2.contour(ds2_depth['lon'], ds2_depth['lat'], mag_g, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)    
    # ax3.contour(ds3_depth['lon'], ds3_depth['lat'], mag_c, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)    
    # ax4.contour(ds4_depth['lon'], ds4_depth['lat'], mag_a, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)
    levels = [0.771667, 1.54333, 3]
    colors = ['orange', 'firebrick']
    
    cs = ax1.contourf(ds1['lon'], ds1['lat'], mag_r, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)
    ax2.contourf(ds2['lon'], ds2['lat'], mag_g, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)    
    ax3.contourf(ds3['lon'], ds3['lat'], mag_c, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)    
    ax4.contourf(ds4['lon'], ds4['lat'], mag_a, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)    

    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
             for pc in cs.collections]
    
    ax1.legend(proxy, ["1.5 - 3.0 knots", "> 3.0 knots"], loc='upper left').set_zorder(200)
    ax2.legend(proxy, ["1.5 - 3.0 knots", "> 3.0 knots"], loc='upper left').set_zorder(200)
    ax3.legend(proxy, ["1.5 - 3.0 knots", "> 3.0 knots"], loc='upper left').set_zorder(200)
    ax4.legend(proxy, ["1.5 - 3.0 knots", "> 3.0 knots"], loc='upper left').set_zorder(200)

    # Add loop current contour from WHO Group
    from scipy.io import loadmat
    fname = '/Users/mikesmith/Downloads/GOM22 Fronts/2022-09-04_fronts.mat'
    data = loadmat(fname)
    for item in data['BZ_all'][0]:
        loop_y = item['y'].T
        loop_x = item['x'].T

        ax1.plot(loop_x, loop_y, 'k-', linewidth=3, transform=ccrs.PlateCarree(), zorder=120)
        ax2.plot(loop_x, loop_y, 'k-', linewidth=3, transform=ccrs.PlateCarree(), zorder=120)
        ax3.plot(loop_x, loop_y, 'k-', linewidth=3, transform=ccrs.PlateCarree(), zorder=120)
        ax4.plot(loop_x, loop_y, 'k-', linewidth=3, transform=ccrs.PlateCarree(), zorder=120)

        # Add arrows
        start_lon = item['bx'].T
        start_lat = item['by'].T
        end_lon = item['tx'].T
        end_lat = item['ty'].T

        for count, _ in enumerate(start_lon[0:-1:2]):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.arrow(
                    start_lon[count][0],
                    start_lat[count][0],
                    end_lon[count][0]-start_lon[count][0],
                    end_lat[count][0]-start_lat[count][0],
                    linewidth=3, 
                    width=.0000000001,
                    head_width=0.2, head_length=0.2, fc='black', ec='black',
                    transform=ccrs.PlateCarree(),
                    zorder=130,
                    # head_starts_at_zero=True
                    )

    # Label the subplots
    ax1.set_title(ds1.model.upper(), fontsize=16, fontweight="bold")
    ax2.set_title(ds2.model.upper(), fontsize=16, fontweight="bold")
    ax3.set_title(ds3.model.upper(), fontsize=16, fontweight="bold")
    ax4.set_title(ds4.model.upper(), fontsize=16, fontweight="bold")

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        t0 = []
        if isinstance(argo, pd.DataFrame):
            if not argo.empty:
                t0.append(argo.index.min()[1])

        if isinstance(gliders, pd.DataFrame):
            if not gliders.empty:
                t0.append(gliders.index.min()[1])

        if len(t0) > 0:
            t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
        else:
            t0 = None
            
        legstr = f'Glider/Argo Surfacings\nPrevious 5 days' # Legend title
        ax5.legend(h, l, loc='center', ncol=1, fontsize=9, title=legstr,
                   title_fontproperties={
                       "size": 10,
                       "weight": 'bold'
                   })
    ax5.set_axis_off()
        
    # Create a string for the title of the plot
    title_time = time.strftime("%Y-%m-%d %H:%M:%S")
    title = f"Surface Currents Contours - {title_time}\n"
    plt.suptitle(title, fontsize=22, fontweight="bold")

    # Save figure
    fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # # Second plot - All on same map
    # fig, ax = plt.subplots(
    #     figsize=figsize, #12,9
    #     subplot_kw=dict(projection=proj['map'])
    # )
    
    # # Make the map pretty
    # map_add_features(ax, extent)
    
    # if bathy:    
    #     map_add_bathymetry(ax,
    #                 bathy.longitude.values, 
    #                 bathy.latitude.values, 
    #                 bathy.elevation.values,
    #                 levels=(-1000, -100),
    #                 zorder=1.5
    #                 )
    #     map_add_ticks(ax, extent)

    # if eez:
    #     eez_h = map_add_eez(ax, zorder=1, color='red', linewidth=1)

    # plot_regional_assets(ax, **rargs)

    # # Add speed contour
    # ax.contour(ds1_depth['lon'], ds1_depth['lat'], mag_r, [0.771667], colors='red', linewidths=3, transform=ccrs.PlateCarree(), zorder=105, label='RTOFS')
    # ax.contour(ds2_depth['lon'], ds2_depth['lat'], mag_g, [0.771667], colors='green', linewidths=2, transform=ccrs.PlateCarree(), zorder=104, label="GOFS", alpha=.75)    
    # ax.contour(ds3_depth['lon'], ds3_depth['lat'], mag_c, [0.771667], colors='purple', linewidths=1, transform=ccrs.PlateCarree(), zorder=103, label="CMEMS", alpha=.5)    
    # # ax.contour(ds4_depth['lon'], ds4_depth['lat'], mag_a, [0.771667], colors='yellow', linewidths=1, transform=ccrs.PlateCarree(), zorder=102, label="AMSEAS")  

    # # plt.legend() 
    # plt.suptitle(title, fontsize=22, fontweight="bold")

    # # Save figure
    # fig.savefig(save_file_q_2, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    
def surface_current_fronts_single(ds1, region,
                                  bathy=None,
                                  argo=None,
                                  gliders=None,
                                  eez=False,
                                  cols=6,
                                  transform=dict(
                                      map=proj['map'], 
                                      data=proj['data']
                                      ),
                                  path_save=os.getcwd(),
                                  figsize=(14,8),
                                  dpi=150,
                                  overwrite=False
                                  ):
    time = pd.to_datetime(ds1.time.data)

    if isinstance(time, pd.core.indexes.datetimes.DatetimeIndex):
        time = time[0]
        
    model = ds1.model.upper()
    extent = region['extent']
    
    # Plot currents with magnitude and direction
    quiver_dir = path_save / f"surface_currents_fronts" / time.strftime('%Y/%m')
    quiver_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    sname = f'gom_{time.strftime("%Y-%m-%dT%H%M%SZ")}_surface_current_fronts-{model}'
    save_file_q = quiver_dir / f"{sname}.png"

    # Check if filename already exists
    if save_file_q.is_file():
        if not overwrite:
            print(f"{sname} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{sname} exists. Overwrite: True. Replotting.")

    # Convert u and v radial velocities to magnitude
    _, mag_r = uv2spdir(ds1['u'], ds1['v'])

    # Initialize single plot
    fig, ax = plt.subplots(
        figsize=(16,9),
        layout="constrained",
        subplot_kw={
            'projection': proj['map']
            },
        )
    ax.set_extent(extent)
    
    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  

    # Make the map pretty
    add_features(ax)
    if bathy:    
        # add_bathymetry(ax,
        #                bathy.longitude.values, 
        #                bathy.latitude.values, 
        #                bathy.elevation.values,
        #                levels=(-1000, -100),
        #                zorder=102)
        # bds = cplt.get_bathymetry(extent)
        bh = cplt.add_bathymetry(ax,
                                 bathy['longitude'],
                                 bathy['latitude'],
                                 bathy['elevation'], 
                                 levels=(-1000, -100)
                                 )
        levels = [-8000, -1000, -100, 0]
        colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']
        cs1 = ax.contourf(bathy['longitude'],
                         bathy['latitude'],
                         bathy['elevation'],
                         levels, colors=colors, transform=ccrs.PlateCarree(), ticks=False)

    add_ticks(ax, extent, gridlines=True)

    if eez:
        map_add_eez(ax, zorder=99, color='red', linewidth=1)
        eez_line = mlines.Line2D([], [], linestyle='-.', color='r', linewidth=1)

    plot_regional_assets(ax, **rargs)

    # Plot ARGO
    try:
        ha = ax.plot(argo['lon'], argo['lat'], 
                    marker='o', linestyle="None",
                    markersize=7, markeredgecolor='black', 
                    color='lime',
                    transform=transform['data'],
                    label='Argo (past 5 days)',
                    zorder=10000)
    except:
        pass

    try:
        # Plot Gliders
        for g, new_df in gliders.groupby(level=0):
            q = new_df.iloc[-1]
            ax.plot(new_df['lon'], new_df['lat'], 
                    color='purple', 
                    linewidth=1.5,
                    transform=transform['data'], 
                    zorder=10000)
            hg = ax.plot(q['lon'], q['lat'], 
                        marker='^',
                        markeredgecolor='black',
                        color="purple",
                        markersize=8.5, 
                        transform=transform['data'], 
                        label = "Glider (past 5 days)",
                        zorder=10000)
    except:
        pass
    # Plot gridlines
    # gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree(), zorder=150)

    # Plot velocity fronts
    # levels = [0.257, 0.771667, 1.28611, 3]
    levels = [0.3858, 0.77166, 1.1575, 10]
    colors = ['yellow', 'orange', 'firebrick']
    cs = ax.contourf(ds1['lon'], ds1['lat'], mag_r, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)

    # Add contour line at 1.5 knots
    test = ax.contour(ds1['lon'], ds1['lat'], mag_r, [.772], linestyles='-', colors=['red'], linewidths=1, alpha=.75, transform=ccrs.PlateCarree(), zorder=101)
        
    # Add loop current contour from WHO Group
    # fname = '/Users/mikesmith/Downloads/GOM front/2023-01-31_fronts.mat'
    # data = loadmat(fname)
 
    # fronts = []
    # for item in data['BZ_all'][0]:
    #     loop_y = item['y'].T
    #     loop_x = item['x'].T

    #     hf = ax.plot(loop_x, loop_y,
    #                  linestyle=item['LineStyle'][0],
    #                  color='black',
    #                  linewidth=4, 
    #                  transform=ccrs.PlateCarree(), 
    #                  zorder=120
    #                  )
    #     fronts.append(hf)

    #     # Add arrows
    #     start_lon = item['bx'].T
    #     start_lat = item['by'].T
    #     end_lon = item['tx'].T
    #     end_lat = item['ty'].T

    #     for count, _ in enumerate(start_lon):
    #         ah = ax.arrow(
    #             start_lon[count][0],
    #             start_lat[count][0],
    #             end_lon[count][0]-start_lon[count][0],
    #             end_lat[count][0]-start_lat[count][0],
    #             linewidth=0, 
    #             head_width=0.2,
    #             shape='full', 
    #             fc='black', 
    #             ec='black',
    #             transform=ccrs.PlateCarree(),
    #             zorder=130,
    #             )
    # fronts.reverse()

    # Legend 1
    legend_1_text = [
        "0.75 - 1.50 knots",
        "1.50 - 2.25 knots",
        "> 2.25 knots", 
        "1.5 knots = 0.77 m/s",
        # "EddyWatch - 1.5 knots",
        "Bathymetry"
        ] 
     
    legend_1 = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
    # legend_1.append(fronts[1][0])
    # legend_1.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    legend_1.append(mlines.Line2D([], [], linestyle='-', color='k', alpha=.75, linewidth=.75))
    # legend_1.append(fronts[0][0])
    # legend_1.append(ah)
    legend_1.append(mlines.Line2D([], [], linestyle='--', color='k', alpha=.5, linewidth=.75))

    try:
        legend_1.append(ha[0])
        legend_1_text.append("ARGO (Past 5 days)")
    except NameError:
        print("No Argo detected at this time.")

    try:
        legend_1.append(hg[0])
        legend_1_text.append("Glider (Past 5 days)")
    except NameError:
        print("No gliders detected at this time.")
    
    if eez:
        legend_1.append(eez_line)
        legend_1_text.append('EEZ')

    leg = ax.legend(legend_1, legend_1_text,
                    fontsize=9,
                    #   title="1 knot = 0.51444 m/s",
                    loc='upper right',
                    # bbox_transform=ccrs.PlateCarree()
                    )
    
    # change the line width for the legend
    # for line in leg.get_lines():
        # line.set_linewidth(4.0)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    leg.set_zorder(10001)

    # Create legend for contours 
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs1.collections]
    proxy.reverse()
    ax.legend(proxy, ["0-100m", "100-1000m", "1000+m"], loc='upper left').set_zorder(10001)
    plt.gca().add_artist(leg)

    # Create a string for the title of the plot
    title_time = time.strftime("%Y-%m-%d %H:%M")
    title = f"Surface Current Comparisons - {title_time} UTC - {model}\n"
    ax.set_title(title, fontsize=18, fontweight="bold", loc='left')

    # Save figure
    fig.savefig(save_file_q, dpi=500, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    # plt.close()

def surface_current_15knot_all(ds1, 
                               ds2,
                               ds3,
                               ds4,
                               ds5,
                               region,
                               bathy=None,
                               argo=None,
                               gliders=None,
                               eez=False,
                               cols=6,
                               transform=dict(
                                   map=proj['map'], 
                                   data=proj['data']
                                   ),
                               path_save=os.getcwd(),
                               figsize=(14,8),
                               dpi=150,
                               overwrite=False
                               ):
    time = pd.to_datetime(ds1.time.data)
    model = ds1.model.upper()
    
    extent = region['extent']
    
    # Plot currents with magnitude and direction
    quiver_dir = path_save / f"surface_currents_fronts" / time.strftime('%Y/%m')
    os.makedirs(quiver_dir, exist_ok=True)

    # Generate filename
    sname = f'gom_{time.strftime("%Y-%m-%dT%H%M%SZ")}_surface_current_fronts-{model}'
    save_file_q = quiver_dir / f"{sname}.png"

    # Check if filename already exists
    if save_file_q.is_file():
        if not overwrite:
            print(f"{sname} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{sname} exists. Overwrite: True. Replotting.")

    # Convert u and v radial velocities to magnitude
    _, mag_1 = uv2spdir(ds1['u'], ds1['v'])
    _, mag_2 = uv2spdir(ds2['u'], ds2['v'])
    _, mag_3 = uv2spdir(ds3['u'], ds3['v'])
    _, mag_4 = uv2spdir(ds4['u'], ds4['v'])
    _, mag_5 = uv2spdir(ds5['u'], ds5['v'])


    # Initialize single plot
    fig, ax = cplt.create(extent, bathymetry=False)
    bds = cplt.get_bathymetry(extent)
    bh = cplt.add_bathymetry(ax, bds['longitude'], bds['latitude'], bds['z'],
                             levels=(-1000, -100)
                             )

    levels = [-8000, -1000, -100, 0] # Contour levels (depths)
    colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue',] # contour colors

    # add filled contour to map
    cs = ax.contourf(bds['longitude'], bds['latitude'], bds['z'], levels, colors=colors, transform=ccrs.PlateCarree(), ticks=False)
    
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
    proxy.reverse()
    # fig, ax = plt.subplots(
    #     figsize=(16,9),
    #     layout="constrained",
    #     subplot_kw={
    #         'projection': proj['map']
    #         },
    #     )
    # ax.set_extent(extent)
    
    # Plot gliders and argo floats
    rargs = {}
    # rargs['argo'] = argo
    # rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  

    # Make the map pretty
    # add_features(ax)
    # if bathy:    
    #     add_bathymetry(ax,
    #                    bathy.longitude.values, 
    #                    bathy.latitude.values, 
    #                    bathy.elevation.values,
    #                    levels=(-1000, -100),
    #                    zorder=102)
    # add_ticks(ax, extent, gridlines=True)

    # if eez:
    #     map_add_eez(ax, zorder=99, color='red', linewidth=1)
    #     eez_line = mlines.Line2D([], [], linestyle='-.', color='r', linewidth=1)

    # plot_regional_assets(ax, **rargs)

    # # Plot ARGO
    # ha = ax.plot(argo['lon'], argo['lat'], 
    #              marker='o', linestyle="None",
    #              markersize=7, markeredgecolor='black', 
    #              color='lime',
    #              transform=transform['data'],
    #              label='Argo (past 5 days)',
    #              zorder=10000)

    # # Plot Gliders
    # for g, new_df in gliders.groupby(level=0):
    #     q = new_df.iloc[-1]
    #     ax.plot(new_df['lon'], new_df['lat'], 
    #             color='purple', 
    #             linewidth=1.5,
    #             transform=transform['data'], 
    #             zorder=10000)
    #     hg = ax.plot(q['lon'], q['lat'], 
    #                  marker='^',
    #                  markeredgecolor='black',
    #                  color="purple",
    #                  markersize=8.5, 
    #                  transform=transform['data'], 
    #                  label = "Glider (past 5 days)",
    #                  zorder=10000)

    # Plot gridlines
    # gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree(), zorder=150)

    # Plot velocity fronts
    # levels = [0.257, 0.771667, 1.28611, 3]
    # levels = [0.3858, 0.77166, 1.1575, 10]
    # colors = ['yellow', 'orange', 'firebrick']
    # cs = ax.contourf(ds1['lon'], ds1['lat'], mag_r, levels, colors=colors, transform=ccrs.PlateCarree(), zorder=100)

    # Add contour line at 1.5 knots
    test1 = ax.contour(ds1['lon'], ds1['lat'], mag_1, [.772], linestyles='-', colors=['red'], linewidths=2,  alpha=.75, transform=ccrs.PlateCarree(), zorder=101, label='RTOFS')
    test2 = ax.contour(ds2['lon'], ds2['lat'], mag_2, [.772], linestyles='-', colors=['green'], linewidths=2,  alpha=.75,transform=ccrs.PlateCarree(), zorder=101, label='GOFS')
    test3 = ax.contour(ds3['lon'], ds3['lat'], mag_3, [.772], linestyles='-', colors=['purple'], linewidths=2, alpha=.75,transform=ccrs.PlateCarree(), zorder=101, label='CMEMS')
    test4 = ax.contour(ds4['lon'], ds4['lat'], mag_4, [.772], linestyles='-', colors=['blue'], linewidths=2,  alpha=.75,transform=ccrs.PlateCarree(), zorder=101, label='AMSEAS')
    test5 = ax.contour(ds5['lon'], ds5['lat'], mag_5, [.772], linestyles='-', colors=['orange'], linewidths=2,  alpha=.75,transform=ccrs.PlateCarree(), zorder=101, label='AMSEAS')

    # # Add loop current contour from WHO Group
    fname = '/Users/mikesmith/Downloads/GOM front/2023-01-31_fronts.mat'
    data = loadmat(fname)
 
    fronts = []
    for item in data['BZ_all'][0]:
        loop_y = item['y'].T
        loop_x = item['x'].T

        hf = ax.plot(loop_x, loop_y,
                     linestyle=item['LineStyle'][0],
                     color='black',
                     linewidth=4,
                    #  alpha=.75,
                     transform=ccrs.PlateCarree(), 
                     zorder=120
                     )
        fronts.append(hf)

        # Add arrows
        start_lon = item['bx'].T
        start_lat = item['by'].T
        end_lon = item['tx'].T
        end_lat = item['ty'].T

        for count, _ in enumerate(start_lon):
            ah = ax.arrow(
                start_lon[count][0],
                start_lat[count][0],
                end_lon[count][0]-start_lon[count][0],
                end_lat[count][0]-start_lat[count][0],
                linewidth=0, 
                head_width=0.2,
                shape='full', 
                fc='black', 
                ec='black',
                transform=ccrs.PlateCarree(),
                zorder=130,
                )
    fronts.reverse()

    # Legend 1
    # legend_1_text = [
    #     "0.75 - 1.50 knots",
    #     "1.50 - 2.25 knots",
    #     "> 2.25 knots", 
    #     "1.5 knots = 0.77 m/s",
    #     "EddyWatch - 1.5 knots",
    #     "Bathymetry",
    #     "ARGO (Past 5 days)"
    #     ] 
    h1, _  = test1.legend_elements()
    h2, _  = test2.legend_elements()
    h3, _  = test3.legend_elements()
    h4, _  = test4.legend_elements()
    h5, _  = test5.legend_elements()
     
    # proxy.append(plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in test1.collections)
    # proxy.append(plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in test2.collections)
    # proxy.append(plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in test3.collections)
    # proxy.append(plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in test4.collections)
    # proxy.append(plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in test5.collections)
    proxy.append(h1[0])
    proxy.append(h2[0])
    proxy.append(h3[0])
    proxy.append(h4[0])
    proxy.append(h5[0])
    proxy.append(mlines.Line2D([], [], linestyle='-', color='k', linewidth=5))

    # # legend_1.append(fronts[1][0])
    
    # # legend_1.append(Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0))
    # legend_1.append(mlines.Line2D([], [], linestyle='-', color='k', alpha=.75, linewidth=.75))
    # legend_1.append(fronts[0][0])
    # # legend_1.append(ah)
    # legend_1.append(mlines.Line2D([], [], linestyle='--', color='k', alpha=.5, linewidth=.75))
    # legend_1.append(ha[0])

    # try:
    #     legend_1.append(hg[0])
    #     legend_1_text.append("Glider (Past 5 days)")
    # except NameError:
    #     print("No gliders detected at this time.")
    
    # if eez:
    #     legend_1.append(eez_line)
    #     legend_1_text.append('EEZ')

    # leg = ax.legend(legend_1, legend_1_text,
    #                 fontsize=9,
    #                 #   title="1 knot = 0.51444 m/s",
    #                 # loc='lower left',
    #                 # bbox_transform=ccrs.PlateCarree()
    #                 )
    # plt.legend(proxy, [], loc='lower left').set_zorder(10000)
    leg = ax.legend(proxy, 
                    ["0-100m", "100-1000m", "1000+m", 'RTOFS', 'GOFS', 'CMEMS', 'AMSEAS', 'CNAPS', 'Eddywatch 1.5 knot'],
                    loc='upper right', 
                    fontsize=10).set_zorder(10001)
    
    # change the line width for the legend
    # for line in leg.get_lines():
        # line.set_linewidth(4.0)

    # for legobj in leg.legendHandles:
        # legobj.set_linewidth(2.0)
    # leg.set_zorder(10001)
    # Create a string for the title of the plot
    title_time = time.strftime("%Y-%m-%d %H:%M")
    title = f"1.5 knot Surface Current Comparisons\n{title_time} UTC"
    ax.set_title(title, fontsize=18, fontweight="bold", loc='left')

    # Save figure
    fig.savefig(save_file_q, dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    # plt.close()

def plot_model_region_comparison_streamplot(ds1, ds2, region,
                                            bathy=None,
                                            argo=None,
                                            gliders=None,
                                            currents=None,
                                            eez=False,
                                            cols=6,
                                            transform=dict(map=proj['map'], 
                                                            data=proj['data']
                                                            ),
                                            path_save=os.getcwd(),
                                            figsize=(14,8),
                                            dpi=150,
                                            colorbar=True,
                                            overwrite=False
                                            ):

    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']
    cdict = region['currents']
    
    grid = """
    RG
    LL
    """

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for depth in cdict['depths']:
        print(f"Plotting currents @ {depth}m")
        ds1_depth = ds1.sel(depth=depth)
        ds2_depth = ds2.sel(depth=depth, method='nearest')
        
        # Plot currents with magnitude and direction
        quiver_dir = path_save / f"currents_{depth}m" / time.strftime('%Y/%m')
        os.makedirs(quiver_dir, exist_ok=True)

        # Generate descriptive filename
        sname = f'{region["folder"]}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_currents-{depth}m_{ds1.model.lower()}-vs-{ds2.model.lower()}'
        save_file_q = quiver_dir / f"{sname}.png"

        # Check if filename already exists
        if save_file_q.is_file():
            if not overwrite:
                print(f"{sname} exists. Overwrite: False. Skipping.")
                continue
            else:
                print(f"{sname} exists. Overwrite: True. Replotting.")

        # Convert u and v radial velocities to magnitude
        _, mag_r = uv2spdir(ds1_depth['u'], ds1_depth['v'])
        _, mag_g = uv2spdir(ds2_depth['u'], ds2_depth['v'])

        # Initialize qargs dictionary for input into contour plot of magnitude
        qargs = {}
        qargs['transform'] = transform['data']
        qargs['cmap'] = cmocean.cm.speed
        qargs['extend'] = "max"

        if 'limits' in cdict:
            lims = cdict['limits']
            qargs['levels'] = np.arange(lims[0], lims[1]+lims[2], lims[2])

        # Initialize figure
        fig, _ = plt.subplot_mosaic(
            grid,
            figsize=figsize,
            layout="constrained",
            subplot_kw={
                'projection': proj['map']
                },
            gridspec_kw={
                # set the height ratios between the rows
                "height_ratios": [4, 1],
                # set the width ratios between the columns
                # # "width_ratios": [1],
                },
            )
        axs = fig.axes
        ax1 = axs[0] # rtofs
        ax2 = axs[1] # gofs
        ax3 = axs[2] # legend for argo/gliders

        # Make the map pretty  
        map_add_features(ax1, extent)# zorder=0)
        map_add_features(ax2, extent)# zorder=0)
        if bathy:       
            map_add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.elevation.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )
            map_add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.elevation.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )
        map_add_ticks(ax1, extent)
        map_add_ticks(ax2, extent, label_left=False, label_right=True)

        # Plot gliders and argo floats
        rargs = {}
        rargs['argo'] = argo
        rargs['gliders'] = gliders
        rargs['transform'] = transform['data']  
        plot_regional_assets(ax1, **rargs)
        plot_regional_assets(ax2, **rargs)

        # Label the subplots
        ax1.set_title(ds1.model.upper(), fontsize=16, fontweight="bold")
        ax2.set_title(ds2.model.upper(), fontsize=16, fontweight="bold")
        
        # Deal with the third axes
        h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1
        if (len(h) > 0) & (len(l) > 0):
            ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

            # Add title to legend
            t0 = []
            if isinstance(argo, pd.DataFrame):
                if not argo.empty:
                    t0.append(argo.index.min()[1])

            if isinstance(gliders, pd.DataFrame):
                if not gliders.empty:
                    t0.append(gliders.index.min()[1])

            if len(t0) > 0:
                t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
            else:
                t0 = None
            legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
            ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style="italic")
            # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
        ax3.set_axis_off()

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88, wspace=.001)
        
        # Filled contour for each model variable
        m1 = ax1.contourf(ds1_depth["lon"], ds1_depth["lat"], mag_r, **qargs)
        m2 = ax2.contourf(ds2_depth["lon"], ds2_depth["lat"], mag_g, **qargs)

        # Set coarsening configs to a variable
        if 'coarsen' in cdict:
            coarsen = region['currents']['coarsen']
        else:
            coarsen['rtofs'] = 1
            coarsen['gofs'] = 1

        # Add streamlines
        s1 = map_add_currents(ax1, ds1_depth, coarsen=coarsen["rtofs"], **currents["kwargs"])
        s2 = map_add_currents(ax2, ds2_depth, coarsen=coarsen["gofs"], **currents["kwargs"])

        # Add EEZ
        if eez:
            eez1 = map_add_eez(ax1, zorder=1, color='red', linewidth=1)
            eez2 = map_add_eez(ax2, zorder=1, color='red', linewidth=1)

        if colorbar:
            cb = fig.colorbar(m1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
            # cb = add_colorbar(axs[:2], m1, location="bottom")
            cb.ax.tick_params(labelsize=12)
            cb.set_label(f'Magnitude (m/s)', fontsize=12, fontweight="bold")

        # Create a string for the title of the plot
        title_time = time.strftime("%Y-%m-%d %H:%M:%S")
        title = f"Currents ({depth} m) - {title_time}\n"
        plt.suptitle(title, fontsize=22, fontweight="bold")

        # subplot 1
        # if depth == 0:
        ax1.contour(ds1_depth['lon'], ds1_depth['lat'], mag_r, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)
        ax2.contour(ds2_depth['lon'], ds2_depth['lat'], mag_g, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)    

        # Save figure
        fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        # plt.close()
        if colorbar:
            cb.remove()
        # Delete contour handles and remove colorbar axes to use figure
        s1.lines.remove(), s2.lines.remove()
        remove_quiver_handles(ax1), remove_quiver_handles(ax2)
        [x.remove() for x in m1.collections]
        [x.remove() for x in m2.collections]