#! /usr/bin/env

"""
Author: Lori Garzio on 2/19/2021
Last modified: 4/27/2021
Tools for plotting specific storms
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ioos_model_comparisons.plotting as sp
from oceans.ocfis import uv2spdir, spdir2uv


def hurricane_intensity_cmap(categories):
    intensity_colors = [
        "#efefef",  # TS
        "#ffffb2",  # cat 1
        "#fed976",  # cat 2 "#feb24c"
        "#e69138",  # cat 3 "#fd8d3c"
        "#cc0000",  # cat 4 "#f03b20"
        "#990000",  # cat 5 "#bd0026"
    ]
    mincat = np.nanmin(categories)
    maxcat = np.nanmax(categories)
    custom_colors = intensity_colors[mincat: maxcat + 1]  # make the colors span the range of data
    hurricane_colormap = mpl.colors.ListedColormap(custom_colors)

    # make custom legend
    le = [Line2D([0], [0], marker='o', markerfacecolor='#efefef', mec='k', linestyle='none', label='TS'),
          Line2D([0], [0], marker='o', markerfacecolor='#ffffb2', mec='k', linestyle='none', label='Cat 1'),
          Line2D([0], [0], marker='o', markerfacecolor='#fed976', mec='k', linestyle='none', label='Cat 2'),
          Line2D([0], [0], marker='o', markerfacecolor='#e69138', mec='k', linestyle='none', label='Cat 3'),
          Line2D([0], [0], marker='o', markerfacecolor='#cc0000', mec='k', linestyle='none', label='Cat 4'),
          Line2D([0], [0], marker='o', markerfacecolor='#990000', mec='k', linestyle='none', label='Cat 5')]
    le_custom = le[mincat: maxcat + 1]  # make the legend handles span the range of data

    return hurricane_colormap, le_custom


def plot_transect(x, y, c, cmap, title=None, save_file=None, flip_y=None, levels=None, extend=None, xlab=None, clab=None):
    levels = levels or dict(deep=np.arange(0, 28), shallow=np.arange(14, 28))
    flip_y = flip_y or True
    extend = extend or 'both'
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'
    xlab = xlab or 'Longitude'

    fig, ax = plt.subplots(figsize=(12, 6))
    cs = plt.contourf(x, y, c, cmap=cmap, levels=levels['shallow'], extend=extend)
    if clab:
        plt.colorbar(cs, ax=ax, label=clab, pad=0.02)
    else:
        plt.colorbar(cs, ax=ax, pad=0.02)
    plt.contour(x, y, c, [26], colors='k')  # add contour at 26m

    if flip_y:
        #ax.set_ylim(-500, 0)
        ax.set_ylim(-200, 0)

    # Add titles and labels
    plt.title(title, size=12)
    plt.setp(ax, ylabel='Depth (m)', xlabel=xlab)
    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def surface_map_storm_forecast(ds, region,
                               bathy=None,
                               argo=None,
                               gliders=None,
                               transform=None,
                               model=None,
                               save_dir=None,
                               dpi=None,
                               forecast=None):
    """
    Written by Mike Smith
    Modified by Lori Garzio
    """
    bathy = bathy or None
    transform = transform or ccrs.PlateCarree()
    argo = argo or False
    gliders = gliders or False
    save_dir = save_dir or os.getcwd()
    model = model or 'rtofs'
    dpi = dpi or 150
    forecast = forecast or False  # add hurricane forecast track to map

    limits = region[1]
    extent = limits['lonlat']

    save_dir_maps = os.path.join(save_dir, 'surface_maps')
    os.makedirs(save_dir_maps, exist_ok=True)

    if model in ['gofs', 'cmems']:
        t1 = pd.to_datetime(ds.time.data)
    elif model == 'rtofs':
        t1 = pd.to_datetime(ds.time.data[0])
    else:
        return 'Incorrect model type. Please enter "gofs", "rtofs" or "cmems"'

    for k, v in limits.items():
        if k in ['lonlat', 'code', 'currents']:
            continue
        if k == 'salinity':
            var_str = 'Sea Surface Salinity'
        elif k == 'temperature':
            var_str = 'Sea Surface Temperature'

        for item in v:
            depth = item['depth']
            if depth > 0:  # only plot sea surface
                continue
            try:
                dsd = ds.sel(depth=depth)
            except KeyError:
                dsd = ds.sel(depth=slice(0, 1))
                if len(dsd.depth) > 1:
                    raise ValueError('More than one depth between 0-1m')

            title = f'{forecast[0]}: Storm Track Forecast on {forecast[1]["forecast_time"].strftime("%Y-%m-%d %H:%M UTC")}\n' \
                    f'{model.upper()} {var_str} at {str(t1)} UTC'
            sname = f'{forecast[0]}_{region[1]["code"]}_{model}_{k}_{t1.strftime("%Y-%m-%dT%H%M%SZ")}'
            save_file = os.path.join(save_dir_maps, sname)

            vargs = {}
            vargs['vmin'] = item['limits'][0]
            vargs['vmax'] = item['limits'][1]
            vargs['transform'] = transform
            vargs['cmap'] = sp.cmaps(ds[k].name)
            vargs['extend'] = 'both'

            if k == 'sea_surface_height':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
                limits['currents'] = True
            else:
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])

            try:
                vargs.pop('vmin'), vargs.pop('vmax')
            except KeyError:
                pass

            fig, ax = plt.subplots(
                figsize=(11, 8),
                subplot_kw=dict(projection=ccrs.Mercator())
            )

            h = plt.contourf(dsd['lon'], dsd['lat'], dsd[k].squeeze(), **vargs)

            if k == 'sea_surface_height':
                sub = 6
                qds = dsd.coarsen(lon=sub, boundary='pad').mean().coarsen(lat=sub, boundary='pad').mean()

                angle, speed = uv2spdir(qds['u'], qds['v'])  # convert u/v to angle and speed
                u, v = spdir2uv(  # convert angle and speed back to u/v, normalizing the arrow sizes
                    np.ones_like(speed),
                    angle,
                    deg=True
                )

                qargs = {}

                # qargs['norm'] = Normalize(vmin=velocity_min, vmax=velocity_max, clip=True)
                qargs['scale'] = 90
                # qargs['headwidth'] = 2.5
                # qargs['headlength'] = 4
                # qargs['headaxislength'] = 4
                qargs['headwidth'] = 2.75
                qargs['headlength'] = 2.75
                qargs['headaxislength'] = 2.5
                qargs['transform'] = ccrs.PlateCarree()
                # qargs['pivot'] = 'mid'
                # qargs['units'] = 'inches'
                # sub = 3

                lons, lats = np.meshgrid(qds['lon'], qds['lat'])
                q = plt.quiver(lons, lats, u, v, **qargs)

            if bathy:
                levels = np.arange(-100, 0, 50)
                bath_lat = bathy.variables['lat'][:]
                bath_lon = bathy.variables['lon'][:]
                bath_elev = bathy.variables['elevation'][:]

                CS = plt.contour(bath_lon, bath_lat, bath_elev,  levels, linewidths=.75, alpha=.5, colors='k', transform=ccrs.PlateCarree())
                ax.clabel(CS, [-100], inline=True, fontsize=6, fmt=sp.fmt)
                # plt.contourf(bath_lon, bath_lat, bath_elev, np.arange(-9000,9100,100), cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())

            sp.map_add_features(ax, extent)
            sp.map_add_ticks(ax, extent)

            if forecast:
                for fc_type, fc in forecast[1].items():
                    if 'forecast_time' not in fc_type:
                        ax.plot(fc['lon'], fc['lat'], ls=fc['plt']['ls'], color=fc['plt']['color'], linewidth=fc['plt']['lw'],
                                transform=ccrs.PlateCarree(), label=fc['plt']['name'])
                ax.legend(loc='best', fontsize=10)

            if argo:
                atimes = []
                for i in argo:
                    ax.plot(i.lon, i.lat, marker='o', markersize=7, color='w', markeredgecolor='black', label=i.name, transform=ccrs.PlateCarree())
                    atimes.append(pd.to_datetime(i.time))
                title = '{}\n Argo floats (circles): {} to {} '.format(title, pd.to_datetime(np.min(atimes)).strftime('%Y-%m-%dT%H:%M'),
                                                                       pd.to_datetime(np.max(atimes)).strftime('%Y-%m-%dT%H:%M'))
                    #ax.legend(loc='upper right', fontsize=6)

            if gliders:
                gltimes = []
                for glid, values in gliders.items():
                    ax.plot(values['longitude'], values['latitude'], color='white', linewidth=1.5, transform=ccrs.PlateCarree())
                    ax.plot(values['longitude'][-1], values['latitude'][-1], color='white', marker='^',
                            markeredgecolor='black', markersize=8.5, transform=ccrs.PlateCarree())
                    # change glider times to strings because some times for different gliders are in different formats
                    gltimes.append(pd.to_datetime(np.max(values['time'])).strftime('%Y-%m-%dT%H:%M:%S'))
                    gltimes.append(pd.to_datetime(np.min(values['time'])).strftime('%Y-%m-%dT%H:%M:%S'))
                # change glider times back to datetimes to find min and max
                gltimes = [pd.to_datetime(x) for x in gltimes]
                title = '{}\n Gliders (triangles): {} to {} '.format(title, pd.to_datetime(np.min(gltimes)).strftime('%Y-%m-%dT%H:%M'),
                                                                 pd.to_datetime(np.max(gltimes)).strftime('%Y-%m-%dT%H:%M'))

            # Plot title
            plt.title(title)

            # Set colorbar height equal to plot height
            divider = make_axes_locatable(ax)
            cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
            fig.add_axes(cax)

            # generate colorbar
            cbar = plt.colorbar(h, cax=cax)
            cbar.set_label(ds[k].units)

            plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()
