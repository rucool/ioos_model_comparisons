#! /usr/bin/env

"""
Author: Lori Garzio on 5/5/2021
Last modified: 5/11/2021
Tools for plotting specific gliders
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import ioos_model_comparisons.plotting as sp
from oceans.ocfis import uv2spdir, spdir2uv
import cmocean


def format_time_axis(axis):
    xfmt = mdates.DateFormatter('%d-%b\n%Y')
    axis.xaxis.set_major_formatter(xfmt)


def glider_track(ds, region, bathy=None, save_dir=None, dpi=None, custom_transect=None, landcolor=None,
                 current_glider_loc=None):
    """
    Written by Mike Smith
    Modified by Lori Garzio
    """
    bathy = bathy or None
    save_dir = save_dir or os.getcwd()
    dpi = dpi or 150
    landcolor = landcolor or 'tan'
    current_glider_loc = current_glider_loc or None

    limits = region[1]
    extent = limits['lonlat']

    try:
        glider_name = ds.deployment_name.split('-')[0]
    except AttributeError:
        glider_name = ds.title.split('-')[0]
    glidert0 = np.nanmin(ds.time.values)
    glidert1 = np.nanmax(ds.time.values)
    glidert0_str = pd.to_datetime(glidert0).strftime('%Y-%m-%dT%H:%M')
    glidert1_str = pd.to_datetime(glidert1).strftime('%Y-%m-%dT%H:%M')
    t0_save = pd.to_datetime(glidert0).strftime('%Y%m%dT%H%M')
    t1_save = pd.to_datetime(glidert1).strftime('%Y%m%dT%H%M')

    try:
        title = f'{ds.deployment_name}\nTrack: {glidert0_str} to {glidert1_str}'
    except AttributeError:
        title = f'{ds.title}\nTrack: {glidert0_str} to {glidert1_str}'
    sname = f'{glider_name}_track_{t0_save}-{t1_save}'
    save_file = os.path.join(save_dir, sname)

    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )

    if bathy:
        levels = np.arange(-9000, 9100, 100)
        bath_lat = bathy.variables['lat'][:]
        bath_lon = bathy.variables['lon'][:]
        bath_elev = bathy.variables['elevation'][:]

        plt.contourf(bath_lon, bath_lat, bath_elev, levels, cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())

        levs = np.arange(-100, 0, 50)
        CS = plt.contour(bath_lon, bath_lat, bath_elev, levs, linewidths=.75, alpha=.5, colors='k',
                         transform=ccrs.PlateCarree())
        ax.clabel(CS, [-100], inline=True, fontsize=7, fmt='%d')

    margs = dict()
    margs['landcolor'] = landcolor
    #margs['add_ticks'] = 'yes'
    sp.map_add_features(ax, extent, **margs)
    sp.map_add_ticks(ax, extent)

    # plot full glider track
    ax.scatter(ds.longitude.values, ds.latitude.values, color='k', marker='.', s=60, transform=ccrs.PlateCarree(),
               zorder=10)
    sct = ax.scatter(ds.longitude.values, ds.latitude.values, c=ds.time.values, marker='.', s=15, cmap='rainbow',
                     transform=ccrs.PlateCarree(), zorder=11)
    if current_glider_loc:
        ax.plot(ds.longitude.values[-1], ds.latitude.values[-1], color='white', marker='^', markeredgecolor='black',
                markersize=8.5, transform=ccrs.PlateCarree())

    if custom_transect:
        ax.plot(custom_transect['lon'], custom_transect['lat'], color='magenta', linewidth=1.5,
                transform=ccrs.PlateCarree())

    # Plot title
    plt.title(title)

    # Set colorbar height equal to plot height
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)

    # generate colorbar
    cbar = plt.colorbar(sct, cax=cax)
    cbar.ax.set_yticklabels(pd.to_datetime(cbar.ax.get_yticks()).strftime(date_format='%Y-%m-%d'))

    plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_transect(x, y, c, cmap=None, title=None, save_file=None, ylims=None, levels=None, extend=None, xlab=None, clab=None):
    """
    Plot one transect
    """
    cmap = cmap or 'jet'
    levels = levels or None
    ylims = ylims or None
    extend = extend or 'both'
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'
    xlab = xlab or 'Longitude'

    fig, ax = plt.subplots(figsize=(12, 6))
    if levels:
        cs = plt.contourf(x, y, c, cmap=cmap, levels=levels['shallow'], extend=extend)
    else:
        cs = plt.contourf(x, y, c, cmap=cmap, extend=extend)

    if clab:
        cb = plt.colorbar(cs, ax=ax, label=clab, pad=0.02)
    else:
        cb = plt.colorbar(cs, ax=ax, pad=0.02)

    cb.ax.tick_params(labelsize=14)
    cb.set_label(clab, fontsize=16)
    isotherm = plt.contour(x, y, c, [26], colors='k')  # add contour at 26C
    # test = isotherm.allsegs[0][0]
    # plt.axhline(test[:, 1].max())
    # xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
    # ax.xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)

    if ylims:
        ax.set_ylim(ylims)

    # if 'time' in xlab.lower():
    #     format_time_axis(ax)

    # ax.invert_yaxis()

    # # Plot Model transitional times
    # time_df = pd.DataFrame(x, columns=['datetime'])
    # time_df['hour'] = time_df['datetime'].dt.hour
    # transition_times = pd.concat([time_df[(time_df['hour'] == 0)], time_df[(time_df['hour'] == 6)]])
    # [plt.axvline(t, color='gray', linestyle='--') for t in transition_times['datetime']]

    # Add titles and labels
    plt.title(title, size=20, fontweight='bold')
    plt.ylabel('Depth (m)', fontsize=18, fontweight='bold')
    # plt.setp(ax, ylabel='Depth (m)', xlabel=xlab)
    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_transects(glx, gly, glc, modelx, modely, modelc, cmap, title0=None, title1=None, save_file=None,
                   ylims=None, levels=None, extend=None, xlab=None, clab=None):
    """
    Plot two transects that share x and y axes
    """
    levels = levels or None
    ylims = ylims or None
    extend = extend or 'both'
    title0 = title0 or 'Glider'
    title1 = title1 or 'Model'
    xlab = xlab or 'Longitude'
    save_file = save_file or 'transect.png'

    # Initiate transect plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 8))

    # plot glider data and model data
    if levels:
        ax0 = axs[0].contourf(glx, gly, glc, cmap=cmap, levels=levels['shallow'], extend=extend)
        ax1 = axs[1].contourf(modelx, modely, modelc, cmap=cmap, levels=levels['shallow'], extend=extend)
    else:
        ax0 = axs[0].contourf(glx, gly, glc, cmap=cmap, extend=extend)
        ax1 = axs[1].contourf(modelx, modely, modelc, cmap=cmap, extend=extend)

    # format glider plot
    axs[0].contour(glx, gly, glc, [26], colors='k')  # add contour at 26C

    if ylims:
        axs[0].set_ylim(ylims)

    if clab:
        cb = plt.colorbar(ax0, ax=axs[0], label=clab, pad=0.02)
    else:
        cb = plt.colorbar(ax0, ax=axs[0], pad=0.02)

    cb.ax.tick_params(labelsize=14)
    cb.set_label(clab, fontsize=16)
    # plt.contour(x, y, c, [26], colors='k')  # add contour at 26C
    xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
    axs[0].xaxis.set_major_formatter(xfmt)

    # axs[0].set_xticklabels(rotation=45, fontsize=13)
    # axs[0].set_yticklabels(fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=13)
    axs[0].tick_params(axis='both', which='minor', labelsize=8)
    axs[0].set_ylabel('Depth (m)', fontsize=15, fontweight='bold')
    axs[0].set_title(title0, fontsize=16, fontweight='bold')

    # # Plot Model transitional times
    # time_df = pd.DataFrame(modelx, columns=['datetime'])
    # time_df['hour'] = time_df['datetime'].dt.hour
    # transition_times = pd.concat([time_df[(time_df['hour'] == 9)], time_df[(time_df['hour'] == 12)]])
    # [axs[0].axvline(t, color='gray', linestyle='--') for t in transition_times['datetime']]

    # format model plot
    axs[1].contour(modelx, modely, modelc, [26], colors='k')  # add contour at 26C

    if ylims:
        axs[1].set_ylim(ylims)

    if clab:
        cb = plt.colorbar(ax1, ax=axs[1], label=clab, pad=0.02)
    else:
        cb = plt.colorbar(ax1, ax=axs[1], pad=0.02)

    # axs[0].invert_yaxis()
    cb.ax.tick_params(labelsize=14)
    cb.set_label(clab, fontsize=16)
    # plt.contour(x, y, c, [26], colors='k')  # add contour at 26C
    xfmt = mdates.DateFormatter('%d-%b-%Y\n%H:%M:%S')
    axs[1].xaxis.set_major_formatter(xfmt)

    plt.xticks(rotation=45)
    # axs[1].set_yticks(fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=13)
    axs[1].tick_params(axis='both', which='minor', labelsize=8)
    axs[1].set_xlabel('Time (GMT)', fontsize=15, fontweight='bold')
    axs[1].set_ylabel('Depth (m)', fontsize=15, fontweight='bold')
    axs[1].set_title(title1, fontsize=16, fontweight='bold')
    # [axs[1].axvline(t, color='gray', linestyle='--') for t in transition_times['datetime']]

    # Add titles and labels
    # plt.setp(axs[0], ylabel='Depth (m)')
    # plt.setp(axs[0], title=title0)
    #
    # plt.setp(axs[1], ylabel='Depth (m)', xlabel=xlab)
    # plt.setp(axs[1], title=title1)

    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def surface_map_glider_track(ds, region,
                             bathy=None,
                             argo=None,
                             gliders=None,
                             transform=None,
                             model=None,
                             save_dir=None,
                             dpi=None,
                             custom_transect=None,
                             current_glider_loc=None):
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
    custom_transect = custom_transect or None
    current_glider_loc = current_glider_loc or None

    limits = region[1]
    extent = limits['lonlat']

    save_dir_maps = os.path.join(save_dir, 'surface_maps', region[1]["code"])
    os.makedirs(save_dir_maps, exist_ok=True)
    glider_name = gliders.deployment_name.split('-')[0]
    glidert0 = np.nanmin(gliders.time.values)
    glidert1 = np.nanmax(gliders.time.values)
    glidert0_str = pd.to_datetime(glidert0).strftime('%Y-%m-%dT%H:%M')
    glidert1_str = pd.to_datetime(glidert1).strftime('%Y-%m-%dT%H:%M')

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

            title = f'{glider_name} track: {glidert0_str} to {glidert1_str}\n' \
                    f'{model.upper()} {var_str} at {str(t1)} UTC'
            sname = f'{glider_name}_{region[1]["code"]}_{model}_{k}_{t1.strftime("%Y-%m-%dT%H%M%SZ")}'
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
                ax.clabel(CS, [-100], inline=True, fontsize=7, fmt=sp.fmt)
                # plt.contourf(bath_lon, bath_lat, bath_elev, np.arange(-9000,9100,100), cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())

            sp.map_add_features(ax, extent)
            sp.map_add_ticks(ax, extent)

            if argo:
                atimes = []
                for i in argo:
                    ax.plot(i.lon, i.lat, marker='o', markersize=7, color='w', markeredgecolor='black', label=i.name, transform=ccrs.PlateCarree())
                    atimes.append(pd.to_datetime(i.time))
                title = '{}\n Argo floats (circles): {} to {} '.format(title, pd.to_datetime(np.min(atimes)).strftime('%Y-%m-%dT%H:%M'),
                                                                       pd.to_datetime(np.max(atimes)).strftime('%Y-%m-%dT%H:%M'))
                    #ax.legend(loc='upper right', fontsize=6)

            # plot full glider track
            ax.plot(gliders.longitude.values, gliders.latitude.values, color='white', linewidth=1.5,
                    label='Glider Track', transform=ccrs.PlateCarree())
            ax.legend(loc='upper left')

            if current_glider_loc:
                ax.plot(gliders.longitude.values[-1], gliders.latitude.values[-1], color='white', marker='^',
                        markeredgecolor='black', markersize=8.5, transform=ccrs.PlateCarree())

            if custom_transect:
                ax.plot(custom_transect['lon'], custom_transect['lat'], color='magenta', linewidth=1.5,
                        label='Model Comparison', transform=ccrs.PlateCarree())
                ax.legend(loc='upper left')

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
