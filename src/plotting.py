import os
from collections import namedtuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
from oceans.ocfis import uv2spdir, spdir2uv

LAND = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m',
    edgecolor='face',
    facecolor='tan'
)

state_lines = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none'
)

Argo = namedtuple('Argo', ['name', 'lon', 'lat'])
Glider = namedtuple('Glider', ['name', 'lon', 'lat'])


def add_map_features(axis, extent, edgecolor=None, landcolor=None):
    edgecolor = edgecolor or 'black'
    landcolor = landcolor or 'tan'

    # Axes properties and features
    axis.set_extent(extent)
    axis.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
    axis.add_feature(cfeature.RIVERS)
    axis.add_feature(cfeature.LAKES)
    axis.add_feature(cfeature.BORDERS)
    axis.add_feature(state_lines, zorder=11, edgecolor=edgecolor)

    # Gridlines and grid labels
    gl = axis.gridlines(
        draw_labels=True,
        linewidth=.5,
        color='black',
        alpha=0.25,
        linestyle='--'
    )

    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xlocator = mticker.MaxNLocator(integer=True)
    gl.ylocator = mticker.MaxNLocator(integer=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def cmaps(variable):
    if variable == 'salinity':
        cmap = cmocean.cm.haline
    elif variable == 'temperature':
        cmap = cmocean.cm.thermal
    elif variable == 'sea_surface_height':
        cmap = cmocean.cm.balance
    return cmap


def plot_model(variable, extent, title, save_file, vmin=None, vmax=None, bathy=None, ptype=None, cmap=None,
               markers=None, transform=None):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    bathy = bathy or None
    ptype = ptype or 'pcolor'
    markers = markers or None
    vmin = vmin or variable.min()
    vmax = vmax or variable.max()
    extent = extent or []
    transform = transform or ccrs.PlateCarree()
    cmap = cmap or cmocean.cm.thermal

    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=ccrs.Mercator())
    )

    # Plot title
    plt.title(title)

    vargs = {}
    vargs['vmin'] = vmin
    vargs['vmax'] = vmax
    vargs['transform'] = transform
    vargs['cmap'] = cmap

    if ptype == 'pcolor':
        h = plt.pcolormesh(variable['lon'], variable['lat'], variable, **vargs)
    elif ptype == 'contour':
        if variable.name == 'surf_el':
            vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], 0.1)
            vargs.pop('vmin')
            vargs.pop('vmax')
        elif variable.name == 'salinity':
            vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], 0.1)
            vargs['extend'] = 'both'
            try:
                vargs.pop('vmin')
                vargs.pop('vmax')
            except KeyError:
                pass
        elif variable.name in {'temperature', 'water_temp'}:
            vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], 0.5)
            vargs['extend'] = 'both'
            try:
                del vargs['vmin']
                del vargs['vmax']
            except KeyError:
                pass
        h = plt.contourf(variable['lon'], variable['lat'], variable.squeeze(), **vargs)
    else:
        print('Invalid plot type specified: specify "pcolor" or "contour"')
        return

    if bathy:
        bath_lat = bathy.variables['lat'][:]
        bath_lon = bathy.variables['lon'][:]
        bath_elev = bathy.variables['elevation'][:]

        plt.contour(bath_lon, bath_lat, bath_elev, [0], colors='k')
        plt.contourf(bath_lon, bath_lat, bath_elev, [0, 10000], colors='seashell')

    # Axes properties and features
    ax.set_extent(extent)
    ax.add_feature(LAND, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(state_lines, zorder=11, edgecolor='black')
    # ax.plot(-93.5, 27, transform=ccrs.Mercator())

    if markers:
        for i in markers:
            ax.plot(i.lon, i.lat, marker='o', label=i.name, transform=ccrs.PlateCarree())
            ax.legend(bbox_to_anchor=(0.5, -0.05), fancybox=True, loc='upper center', ncol=5)

            # try:
    #     t0 = pd.to_datetime(variable.time.data - np.timedelta64(6, 'h'))
    #     t1 = pd.to_datetime(variable.time.data + np.timedelta64(6, 'h'))
    # except:
    #     t0 = pd.to_datetime(variable.MT.data[0] - np.timedelta64(6, 'h'))
    #     t1 = pd.to_datetime(variable.MT.data[0] + np.timedelta64(6, 'h'))
    #
    # if markers:
    #     lons, lats = [], []
    #
    #     if 'argo' in markers:
    #         floats = active_argo_floats([extent[0], extent[1]],
    #                                     [extent[2], extent[3]], t0, t1)
    #
    #         if not floats.empty:
    #             most_recent = floats.loc[floats.groupby('platform_number')['time (UTC)'].idxmax()]
    #
    #             for float in most_recent.itertuples():
    #                 lons.append(float._4)
    #                 lats.append(float._5)
    #
    #     if 'gliders' in markers:
    #         gliders = active_gliders([extent[0], extent[1]],
    #                                     [extent[2], extent[3]], t0, t1)
    #         print()
    #
    #     fax = ax.scatter(lons, lats, 12, 'red', 'o', transform=ccrs.PlateCarree())

    # Gridlines and grid labels
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=.5,
        color='black',
        alpha=0.25,
        linestyle='--'
    )

    gl.xlabels_top = gl.ylabels_right = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xlocator = mticker.MaxNLocator(integer=True)
    gl.ylocator = mticker.MaxNLocator(integer=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Set colorbar height equal to plot height
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
    fig.add_axes(cax)

    # generate colorbar
    cbar = plt.colorbar(h, cax=cax)
    cbar.set_label(variable.units)

    plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_transects(x, y, c, cmap, title=None, save_file=None, flip_y=None, levels=None, extend=None):
    levels = levels or dict(deep=np.arange(0, 28), shallow=np.arange(14, 28))
    flip_y = flip_y or True
    extend = extend or 'both'
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    # Initiate transect plot
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

    # 1000m subplot
    ax1 = axs[0].contourf(x, y, c, cmap=cmap, levels=levels['deep'], extend=extend)
    axs[0].contour(x, y, c, [26], colors='k')  # add contour at 26m

    if flip_y:
        axs[0].set_ylim(-1000, 0)

    fig.colorbar(ax1, ax=axs[0], orientation='vertical')

    # 300m subplot
    ax2 = axs[1].contourf(x, y, c, cmap=cmap, levels=levels['shallow'], extend=extend)
    axs[1].contour(x, y, c, [26], colors='k')  # add contour at 26m

    if flip_y:
        axs[1].set_ylim(-300, 0)

    fig.colorbar(ax2, ax=axs[1], orientation='vertical')

    # Add titles and labels
    plt.suptitle(title, size=12)
    plt.setp(axs[0], ylabel='Depth (m)')
    plt.setp(axs[1], ylabel='Depth (m)', xlabel='Longitude')
    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_transect(x, y, c, cmap, title=None, save_file=None, flip_y=None, levels=None, extend=None):
    levels = levels or dict(deep=np.arange(0, 28), shallow=np.arange(14, 28))
    flip_y = flip_y or True
    extend = extend or 'both'
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    # Initiate transect plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot plot
    ax2 = ax.contourf(x, y, c, cmap=cmap, levels=levels['shallow'], extend=extend)
    ax.contour(x, y, c, [26], colors='k')  # add contour at 26m

    if flip_y:
        ax.set_ylim(-300, 0)

    fig.colorbar(ax2, ax=ax, orientation='vertical')

    # Add titles and labels
    plt.suptitle(title, size=12)
    plt.setp(ax, ylabel='Depth (m)')
    plt.setp(ax, ylabel='Depth (m)', xlabel='Longitude')
    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_model_region(ds, region, t1,
                      bathy=None,
                      argo=None,
                      gliders=None,
                      transform=None,
                      model=None,
                      save_dir=None,
                      dpi=None,
                      t0=None):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    bathy = bathy or None
    transform = transform or ccrs.PlateCarree()
    # argo = argo or pd.DataFrame()
    # gliders = gliders or pd.DataFrame()
    save_dir = save_dir or os.getcwd()
    model = model or 'rtofs'
    dpi = dpi or 150

    region_name = region[0]
    limits = region[1]
    extent = limits['lonlat']

    region_file_str = ('_').join(region_name.lower().split(' '))
    save_dir_region = os.path.join(save_dir, 'regions', region_file_str)

    for k, v in limits.items():
        if k == 'lonlat':
            continue
        elif k == 'currents':
            continue

        var_str = ' '.join(ds[k].standard_name.split('_')).title()
        save_dir_var = os.path.join(save_dir_region, k)

        for item in v:
            depth = item['depth']
            dsd = ds.sel(depth=depth)

            save_dir_depth = os.path.join(save_dir_var, f'{depth}m')

            title = f'Region: {region_name.title()}, Variable: {var_str} @ {depth}m\n'\
                    f'Time: {str(t1)} UTC, Model: {model.upper()}\n'\
                    f'Glider/Argo Window {str(t0)} to {str(t1)}'
            sname = f'{model}-{k}-{t1.strftime("%Y-%m-%dT%H%M%SZ")}'

            save_dir_final = os.path.join(save_dir_depth, t1.strftime('%Y/%m'))
            os.makedirs(save_dir_final, exist_ok=True)
            save_file = os.path.join(save_dir_final, sname)
            save_file = save_file + '.png'

            #             if os.path.isfile(save_file):
            #                 continue

            vargs = {}
            vargs['vmin'] = item['limits'][0]
            vargs['vmax'] = item['limits'][1]
            vargs['transform'] = transform
            vargs['cmap'] = cmaps(ds[k].name)
            vargs['extend'] = 'both'

            if k == 'sea_surface_height':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            elif k == 'salinity':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            elif k == 'temperature':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])

            try:
                vargs.pop('vmin'), vargs.pop('vmax')
            except KeyError:
                pass

            fig, ax = plt.subplots(
                figsize=(11, 8),
                subplot_kw=dict(projection=ccrs.Mercator())
            )

            # Plot title
            plt.title(title)

            h = plt.contourf(dsd['lon'], dsd['lat'], dsd[k].squeeze(), **vargs)

            if bathy:
                levels = np.arange(-100, 0, 50)
                bath_lat = bathy.variables['lat'][:]
                bath_lon = bathy.variables['lon'][:]
                bath_elev = bathy.variables['elevation'][:]

                CS = plt.contour(bath_lon, bath_lat, bath_elev, levels, linewidths=.75, alpha=.5, colors='k',
                                 transform=ccrs.PlateCarree())
                ax.clabel(CS, [-100], inline=True, fontsize=6, fmt=fmt)
                # plt.contourf(bath_lon, bath_lat, bath_elev, np.arange(-9000,9100,100), cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())

            # Axes properties and features
            ax.set_extent(extent)
            ax.add_feature(LAND, edgecolor='black')
            ax.add_feature(cfeature.RIVERS)
            ax.add_feature(cfeature.LAKES)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(state_lines, zorder=11, edgecolor='black')
            # ax.plot(-93.5, 27, transform=ccrs.Mercator())

            if limits['currents']['bool']:
                q = add_currents(limits['currents']['coarsen'], dsd)

            if not argo.empty:
                most_recent = argo.loc[argo.groupby('platform_number')['time (UTC)'].idxmax()]

                for float in most_recent.itertuples():
                    ax.plot(float._4, float._5, marker='o', markersize=7, markeredgecolor='black',
                            label=float.platform_number, transform=ccrs.PlateCarree())
                    ax.legend(loc='upper right', fontsize=6)

            if not gliders.empty:
                for g, new_df in gliders.groupby(level=0):
                    q = new_df.iloc[-1]
                    ax.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'], color='white',
                            linewidth=1.5, transform=ccrs.PlateCarree())
                    ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker='^',
                            markeredgecolor='black', markersize=8.5, label=g, transform=ccrs.PlateCarree())
                    ax.legend(loc='upper right', fontsize=6)

            # # Gridlines and grid labels
            # gl = ax.gridlines(
            #     draw_labels=True,
            #     linewidth=.5,
            #     color='black',
            #     alpha=0.25,
            #     linestyle='--'
            # )
            #
            # gl.top_labels = gl.right_labels = False
            # gl.xlabel_style = {'size': 10, 'color': 'black'}
            # gl.ylabel_style = {'size': 10, 'color': 'black'}
            # gl.xlocator = mticker.MaxNLocator(integer=True)
            # gl.ylocator = mticker.MaxNLocator(integer=True)
            # gl.xformatter = LONGITUDE_FORMATTER
            # gl.yformatter = LATITUDE_FORMATTER

            # [-100, -80, 18, 32]
            xl = [extent[0], extent[1]]
            yl = [extent[2], extent[3]]
            tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
            ax.set_xticks(tick0x, minor=True, crs=ccrs.PlateCarree())
            ax.set_xticks(tick1, crs=ccrs.PlateCarree())
            ax.set_xticklabels(ticklab)

            # get and add latitude ticks/labels
            tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
            ax.set_yticks(tick0y, minor=True, crs=ccrs.PlateCarree())
            ax.set_yticks(tick1, crs=ccrs.PlateCarree())
            ax.set_yticklabels(ticklab)
            gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.5, linestyle='--',
                              crs=ccrs.PlateCarree())  # ,crs=mercproj)
            gl.xlocator = mticker.FixedLocator(tick0x)
            gl.ylocator = mticker.FixedLocator(tick0y)

            # Set colorbar height equal to plot height
            divider = make_axes_locatable(ax)
            cax = divider.new_horizontal(size='5%', pad=0.05, axes_class=plt.Axes)
            fig.add_axes(cax)

            # generate colorbar
            cbar = plt.colorbar(h, cax=cax)
            cbar.set_label(ds[k].units)

            plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"


def add_currents(sub, dsd):
    """
    Add currents to map
    :param sub: amount to downsample by
    :param dsd: dataset
    :return:
    """
    try:
        qds = dsd.coarsen(lon=sub, boundary='pad').mean().coarsen(lat=sub, boundary='pad').mean()
        mesh = True
    except ValueError:
        qds = dsd.coarsen(X=sub, boundary='pad').mean().coarsen(Y=sub, boundary='pad').mean()
        mesh = False

    angle, speed = uv2spdir(qds['u'], qds['v'])  # convert u/v to angle and speed
    u, v = spdir2uv(  # convert angle and speed back to u/v, normalizing the arrow sizes
        np.ones_like(speed),
        angle,
        deg=True
    )

    qargs = {}

    qargs['scale'] = 90
    qargs['headwidth'] = 2.75
    qargs['headlength'] = 2.75
    qargs['headaxislength'] = 2.5
    qargs['transform'] = ccrs.PlateCarree()

    if mesh:
        lons, lats = np.meshgrid(qds['lon'], qds['lat'])
        q = plt.quiver(lons, lats, u, v, **qargs)
    else:
        q = plt.quiver(qds.lon.squeeze().data, qds.lat.squeeze().data, u.squeeze(), v.squeeze(), **qargs)
    return q


def region_subplot(axs, ds, var, extent, title, argo, gliders, bathy, vargs):
    h = axs.contourf(ds['lon'], ds['lat'], ds[var].squeeze(), **vargs)
    axs.set_title(title, fontsize=16)

    # if limits['currents']['bool']:
    #     q = add_currents(limits['currents']['coarsen'], dsd)

    # Axes properties and features
    axs.set_extent(extent)
    axs.add_feature(LAND, edgecolor='black')
    axs.add_feature(cfeature.RIVERS)
    axs.add_feature(cfeature.LAKES)
    axs.add_feature(cfeature.BORDERS)
    axs.add_feature(state_lines, zorder=11, edgecolor='black')

    if not argo.empty:
        most_recent = argo.loc[argo.groupby('platform_number')['time (UTC)'].idxmax()]

        for float in most_recent.itertuples():
            axs.plot(float._4, float._5, marker='o', markersize=7, markeredgecolor='black', label=float.platform_number,
                     transform=ccrs.PlateCarree())
            axs.legend(loc='upper right', fontsize=6)

    if not gliders.empty:
        for g, new_df in gliders.groupby(level=0):
            q = new_df.iloc[-1]
            axs.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'], color='white',
                     linewidth=1.5, transform=ccrs.PlateCarree())
            axs.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker='^', markeredgecolor='black',
                     markersize=8.5, label=g, transform=ccrs.PlateCarree())
            axs.legend(loc='upper right', fontsize=6)

    if bathy:
        # levels = np.arange(-1500, 0, 150)
        levels = np.array([-600, -100])
        bath_lat = bathy.variables['lat'][:]
        bath_lon = bathy.variables['lon'][:]
        bath_elev = bathy.variables['elevation'][:]

        CS = axs.contour(bath_lon, bath_lat, bath_elev, levels, linewidths=.75, alpha=.5, colors='k',
                         transform=ccrs.PlateCarree())
        axs.clabel(CS, levels, inline=True, fontsize=6, fmt=fmt)
        # plt.contourf(bath_lon, bath_lat, bath_elev, np.arange(-9000,9100,100), cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())

    return h


def plot_model_regions_comparison(ds, ds2,
                                  region,
                                  t1,
                                  bathy=None,
                                  argo=None,
                                  gliders=None,
                                  transform=None,
                                  model=None,
                                  save_dir=None,
                                  dpi=None,
                                  t0=None):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    bathy = bathy or None
    transform = transform or ccrs.PlateCarree()
    # argo = argo or pd.DataFrame()
    # gliders = gliders or pd.DataFrame()
    save_dir = save_dir or os.getcwd()
    model = model or 'rtofs'
    dpi = dpi or 150

    region_name = region[0]
    limits = region[1]
    extent = limits['lonlat']

    region_file_str = ('_').join(region_name.lower().split(' '))
    save_dir_region = os.path.join(save_dir, 'regions', region_file_str)

    for k, v in limits.items():
        if k == 'lonlat':
            continue
        elif k == 'currents':
            continue

        var_str = ' '.join(ds[k].standard_name.split('_')).title()
        save_dir_var = os.path.join(save_dir_region, k)

        for item in v:
            depth = item['depth']
            ds1d = ds.sel(depth=depth)
            ds2d = ds2.sel(depth=depth)

            save_dir_depth = os.path.join(save_dir_var, f'{depth}m')

            sname = f'{k}-model-comparison-{t1.strftime("%Y-%m-%dT%H%M%SZ")}'

            save_dir_final = os.path.join(save_dir_depth, t1.strftime('%Y/%m'))
            os.makedirs(save_dir_final, exist_ok=True)
            save_file = os.path.join(save_dir_final, sname)
            save_file = save_file + '.png'

            #             if os.path.isfile(save_file):
            #                 continue

            vargs = {}
            vargs['vmin'] = item['limits'][0]
            vargs['vmax'] = item['limits'][1]
            vargs['transform'] = transform
            vargs['cmap'] = cmaps(ds[k].name)
            vargs['extend'] = 'both'

            if k == 'sea_surface_height':
                continue
                # vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            elif k == 'salinity':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            elif k == 'temperature':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])

            try:
                vargs.pop('vmin'), vargs.pop('vmax')
            except KeyError:
                pass

            # fig = plt.figure(constrained_layout=True, figsize=(11, 8))
            # gs = fig.add_gridspec(1, 2, wspace=0.125)

            fig, axs = plt.subplots(
                1, 2,
                sharex=True,
                sharey=True,
                figsize=(14, 7),
                constrained_layout=True,
                subplot_kw=dict(projection=ccrs.Mercator()),
            )

            # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

            title = f'Region: {region_name.title()}, Variable: {var_str} @ {depth}m\n' \
                    f'Time: {str(t1)} UTC\n'\
                    f'Glider/Argo Search: {str(t0)} to {str(t1)}'


            h1 = region_subplot(axs[0], ds1d, k, extent, 'RTOFS', argo, gliders, bathy, vargs)
            # Gridlines and grid labels
            xl = [extent[0], extent[1]]
            yl = [extent[2], extent[3]]
            tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
            axs[0].set_xticks(tick0x, minor=True, crs=ccrs.PlateCarree())
            axs[0].set_xticks(tick1, crs=ccrs.PlateCarree())
            axs[0].set_xticklabels(ticklab)

            # get and add latitude ticks/labels
            tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
            axs[0].set_yticks(tick0y, minor=True, crs=ccrs.PlateCarree())
            axs[0].set_yticks(tick1, crs=ccrs.PlateCarree())
            axs[0].set_yticklabels(ticklab)

            gl = axs[0].gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.5, linestyle='--',
                              crs=ccrs.PlateCarree())  # ,crs=mercproj)
            gl.xlocator = mticker.FixedLocator(tick0x)
            gl.ylocator = mticker.FixedLocator(tick0y)
            # gl = axs[0].gridlines(
            #     draw_labels=True,
            #     dms=True,
            #     linewidth=.5,
            #     color='black',
            #     alpha=0.25,
            #     linestyle='--'
            # )
            #
            # gl.top_labels = gl.right_labels = False
            # gl.xlabel_style = {'size': 8, 'color': 'black'}
            # gl.ylabel_style = {'size': 8, 'color': 'black'}
            # gl.xlocator = mticker.MaxNLocator(integer=True)
            # gl.ylocator = mticker.MaxNLocator(integer=True)
            # gl.xformatter = LONGITUDE_FORMATTER
            # gl.yformatter = LATITUDE_FORMATTER

            plt.setp(axs[0], ylabel='Longitude', xlabel='Latitude')

            h2 = region_subplot(axs[1], ds2d, k, extent, 'GOFS', argo, gliders, bathy, vargs)
            # # Gridlines and grid labels
            # gl = axs[1].gridlines(
            #     draw_labels=True,
            #     linewidth=.5,
            #     color='black',
            #     alpha=0.25,
            #     linestyle='--'
            # )
            #
            # gl.top_labels = gl.right_labels = gl.ylabels_left = False
            # gl.xlabel_style = {'size': 8, 'color': 'black'}
            # gl.ylabel_style = {'size': 8, 'color': 'black'}
            # gl.xlocator = mticker.MaxNLocator(integer=True)
            # gl.ylocator = mticker.MaxNLocator(integer=True)
            # gl.xformatter = LONGITUDE_FORMATTER
            # gl.yformatter = LATITUDE_FORMATTER
            tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
            axs[1].set_xticks(tick0x, minor=True, crs=ccrs.PlateCarree())
            axs[1].set_xticks(tick1, crs=ccrs.PlateCarree())
            axs[1].set_xticklabels(ticklab)

            # get and add latitude ticks/labels
            tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
            axs[1].set_yticks(tick0y, minor=True, crs=ccrs.PlateCarree())
            axs[1].set_yticks(tick1, crs=ccrs.PlateCarree())
            axs[1].set_yticklabels(ticklab)

            gl = axs[1].gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.5, linestyle='--',
                                  crs=ccrs.PlateCarree())  # ,crs=mercproj)
            gl.xlocator = mticker.FixedLocator(tick0x)
            gl.ylocator = mticker.FixedLocator(tick0y)
            plt.setp(axs[1], ylabel='Longitude', xlabel='Latitude')

            # if limits['currents']['bool']:
            #     q = add_currents(limits['currents']['coarsen'], dsd)

            # divider = make_axes_locatable(axs[0])
            # cax = divider.append_axes("bottom", size="5%", pad=0.05)
            # plt.colorbar(h1, cax=cax)
            #
            # divider = make_axes_locatable(axs[1])
            # cax = divider.append_axes("bottom", size="5%", pad=0.05)
            # plt.colorbar(h2, cax=cax)

            cb = fig.colorbar(h1, ax=axs[0], location='bottom')
            cb.set_label(ds1d[k].units)

            cb = fig.colorbar(h2, ax=axs[1], location='bottom')
            cb.set_label(ds2d[k].units)

            # plt.tight_layout()

            plt.suptitle(title)

            plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()


# function to define major and minor tick locations and major tick labels
def get_ticks(bounds, dirs, otherbounds):
    dirs = dirs.lower()
    l0 = np.float(bounds[0])
    l1 = np.float(bounds[1])
    r = np.max([l1 - l0, np.float(otherbounds[1]) - np.float(otherbounds[0])])
    if r <= 1.5:
        # <1.5 degrees: 15' major ticks, 5' minor ticks
        minor_int = 1.0 / 12.0
        major_int = 1.0 / 4.0
    elif r <= 3.0:
        # <3 degrees: 30' major ticks, 10' minor ticks
        minor_int = 1.0 / 6.0
        major_int = 0.5
    elif r <= 7.0:
        # <7 degrees: 1d major ticks, 15' minor ticks
        minor_int = 0.25
        major_int = np.float(1)
    elif r <= 15:
        # <15 degrees: 2d major ticks, 30' minor ticks
        minor_int = 0.5
        major_int = np.float(2)
    elif r <= 30:
        # <30 degrees: 3d major ticks, 1d minor ticks
        minor_int = np.float(1)
        major_int = np.float(3)
    else:
        # >=30 degrees: 5d major ticks, 1d minor ticks
        minor_int = np.float(1)
        major_int = np.float(5)

    minor_ticks = np.arange(np.ceil(l0 / minor_int) * minor_int, np.ceil(l1 / minor_int) * minor_int + minor_int,
                            minor_int)
    minor_ticks = minor_ticks[minor_ticks <= l1]
    major_ticks = np.arange(np.ceil(l0 / major_int) * major_int, np.ceil(l1 / major_int) * major_int + major_int,
                            major_int)
    major_ticks = major_ticks[major_ticks <= l1]

    if major_int < 1:
        d, m, s = dd2dms(np.array(major_ticks))
        if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
            n = 'W' * sum(d < 0)
            p = 'E' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
                                 range(len(d))]
        elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
            n = 'S' * sum(d < 0)
            p = 'N' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" + dir[i] for i in
                                 range(len(d))]
        else:
            major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" + str(int(m[i])) + "'" for i in range(len(d))]
    else:
        d = major_ticks
        if dirs == 'we' or dirs == 'ew' or dirs == 'lon' or dirs == 'long' or dirs == 'longitude':
            n = 'W' * sum(d < 0)
            p = 'E' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
        elif dirs == 'sn' or dirs == 'ns' or dirs == 'lat' or dirs == 'latitude':
            n = 'S' * sum(d < 0)
            p = 'N' * sum(d >= 0)
            dir = n + p
            major_tick_labels = [str(np.abs(int(d[i]))) + u"\N{DEGREE SIGN}" + dir[i] for i in range(len(d))]
        else:
            major_tick_labels = [str(int(d[i])) + u"\N{DEGREE SIGN}" for i in range(len(d))]

    return minor_ticks, major_ticks, major_tick_labels


# decimal degrees to degree-minute-second converter
def dd2dms(vals):
    n = np.empty(np.shape(vals))
    n[:] = False
    n[vals < 0] = True
    vals[n == True] = -vals[n == True]
    d = np.floor(vals)
    rem = vals - d
    rem = rem * 60
    m = np.floor(rem)
    rem -= m
    s = np.round(rem * 60)
    d[n == True] = -d[n == True]
    return d, m, s

