import os
from collections import namedtuple
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from oceans.ocfis import uv2spdir, spdir2uv
from src.calc import dd2dms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
import matplotlib.colors
from itertools import cycle


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


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    From ImportanceOfBeingErnest
    https://stackoverflow.com/a/47232942/2643708
    :param nc: number of categories (colors)
    :param nsc: number of subcategories (shades for each color)
    :param cmap: matplotlib colormap
    :param continuous:
    :return:
    """
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


def cmaps(variable):
    if variable == 'salinity':
        cmap = cmocean.cm.haline
    elif variable == 'temperature':
        cmap = cmocean.cm.thermal
    elif variable == 'sea_surface_height':
        cmap = cmocean.cm.balance
    return cmap


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}"


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


def map_add_argo(ax, df, transform):
    most_recent = df.loc[df.groupby('platform_number')['time (UTC)'].idxmax()]

    custom_cmap = categorical_cmap(10, 4, cmap="tab10")
    marker = cycle(['o', 'h', 'p'])

    n = 0
    for float in most_recent.itertuples():
        ax.plot(float._4, float._5, marker=next(marker), markersize=7, markeredgecolor='black', color=custom_cmap.colors[n],
                label=float.platform_number,
                transform=transform)
        # map_add_legend(ax)
        n = n + 1
    return ax


def map_add_bathymetry(ax, ds, transform, levels=None):
    levels = levels or np.array([-600, -100])

    # levels = np.arange(-1500, 0, 150)
    bath_lat = ds.variables['lat'][:]
    bath_lon = ds.variables['lon'][:]
    bath_elev = ds.variables['elevation'][:]

    CS = ax.contour(bath_lon, bath_lat, bath_elev, levels, linewidths=.75, alpha=.5, colors='k',
                    transform=transform)
    ax.clabel(CS, levels, inline=True, fontsize=6, fmt=fmt)
    # plt.contourf(bath_lon, bath_lat, bath_elev, np.arange(-9000,9100,100), cmap=cmocean.cm.topo, transform=ccrs.PlateCarree())
    return ax


def map_add_currents(currents, sub=None):
    """
    Add currents to map
    :param dsd: dataset
    :param sub: amount to downsample by
    :return:
    """

    sub = sub or 2
    try:
        qds = currents.coarsen(lon=sub, boundary='pad').mean().coarsen(lat=sub, boundary='pad').mean()
        mesh = True
    except ValueError:
        qds = currents.coarsen(X=sub, boundary='pad').mean().coarsen(Y=sub, boundary='pad').mean()
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


def map_add_features(axis, extent, edgecolor=None, landcolor=None, add_ticks=None):
    edgecolor = edgecolor or 'black'
    landcolor = landcolor or 'tan'
    add_ticks = add_ticks or None

    # Axes properties and features
    axis.set_extent(extent)
    axis.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
    axis.add_feature(cfeature.RIVERS)
    axis.add_feature(cfeature.LAKES)
    axis.add_feature(cfeature.BORDERS)
    axis.add_feature(state_lines, zorder=11, edgecolor=edgecolor)

    if add_ticks:
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


def map_add_gliders(ax, df, transform):
    for g, new_df in df.groupby(level=0):
        q = new_df.iloc[-1]
        ax.plot(new_df['longitude (degrees_east)'], new_df['latitude (degrees_north)'], color='white',
                linewidth=1.5, transform=ccrs.PlateCarree())
        ax.plot(q['longitude (degrees_east)'], q['latitude (degrees_north)'], marker='^', markeredgecolor='black',
                markersize=8.5, label=g, transform=transform)
        # map_add_legend(ax)
    return ax


def map_add_legend(ax):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def map_add_ticks(ax, extent, fontsize=13):
    xl = [extent[0], extent[1]]
    yl = [extent[2], extent[3]]

    tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
    ax.set_xticks(tick0x, minor=True, crs=ccrs.PlateCarree())
    ax.set_xticks(tick1, crs=ccrs.PlateCarree())
    ax.set_xticklabels(ticklab, fontsize=fontsize)

    # get and add latitude ticks/labels
    tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
    ax.set_yticks(tick0y, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(tick1, crs=ccrs.PlateCarree())
    ax.set_yticklabels(ticklab, fontsize=fontsize)

    gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree())
    gl.xlocator = mticker.FixedLocator(tick0x)
    gl.ylocator = mticker.FixedLocator(tick0y)

    ax.tick_params(which='major',
                   direction='out',
                   bottom=True, top=True,
                   labelbottom=True, labeltop=False,
                   left=True, right=True,
                   labelleft=True, labelright=False,
                   length=5, width=2)

    ax.tick_params(which='minor',
                   direction='out',
                   bottom=True, top=True,
                   labelbottom=True, labeltop=False,
                   left=True, right=True,
                   labelleft=True, labelright=False,
                   width=1)
    return ax


def map_add_transects(ax, transects, projection):
    ax.plot(transects['lon'], transects['lat'], 'r-', transform=projection)
    return ax


def plot_map(lon, lat, extent, projection=None):
    projection = projection or ccrs.Mercator()
    fig, ax = plt.subplots(
        figsize=(11, 8),
        subplot_kw=dict(projection=projection)
    )

    plt.plot(lon, lat, 'r-')

    map_add_features(ax, extent)
    map_add_ticks(ax, extent)

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

            vargs = {}
            vargs['vmin'] = item['limits'][0]
            vargs['vmax'] = item['limits'][1]
            vargs['transform'] = transform
            vargs['cmap'] = cmaps(ds[k].name)

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
            vargs['argo'] = argo
            vargs['gliders'] = gliders
            vargs['bathy'] = bathy


            region_subplot(fig, ax, extent, dsd[k].squeeze(), title, **vargs)
            # ax.plot(track['lon'], track['lat'], 'k-', linewidth=6, transform=vargs['transform'])
            map_add_ticks(ax, extent)

            plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()


def plot_model_region_comparison(rtofs, gofs, region, time,
                                 bathy=None,
                                 argo=None,
                                 gliders=None,
                                 transform=None,
                                 save_dir=None,
                                 dpi=None,
                                 t0=None,
                                 ticks=True):
    """

    :param ds: model 1
    :param ds2:
    :param region:
    :param t1:
    :param bathy:
    :param argo:
    :param gliders:
    :param transform:
    :param save_dir:
    :param dpi:
    :param t0:
    :return:
    """
    bathy = bathy or None
    transform = transform or ccrs.PlateCarree()
    save_dir = save_dir or os.getcwd()
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

        var_str = ' '.join(rtofs[k].standard_name.split('_')).title()
        save_dir_var = os.path.join(save_dir_region, k)

        for item in v:
            depth = item['depth']
            rtofs_sub = rtofs.sel(depth=depth)
            gofs_sub = gofs.sel(depth=depth)

            save_dir_depth = os.path.join(save_dir_var, f'{depth}m')

            sname = f'{k}-model-comparison-{time.strftime("%Y-%m-%dT%H%M%SZ")}'

            save_dir_final = os.path.join(save_dir_depth, time.strftime('%Y/%m'))
            os.makedirs(save_dir_final, exist_ok=True)
            save_file = os.path.join(save_dir_final, sname)
            save_file = save_file + '.png'

            vargs = {}
            vargs['vmin'] = item['limits'][0]
            vargs['vmax'] = item['limits'][1]
            vargs['transform'] = transform
            vargs['cmap'] = cmaps(rtofs[k].name)
            vargs['ticks'] = ticks

            if k == 'sea_surface_height':
                continue
            elif k == 'salinity':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            elif k == 'temperature':
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])

            try:
                vargs.pop('vmin'), vargs.pop('vmax')
            except KeyError:
                pass

            # Initiate transect plot
            # fig, axs = plt.subplots(figsize=(20, 8))
            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            # plt.rcParams['figure.constrained_layout.use'] = True
            grid = plt.GridSpec(12, 20, hspace=0.2, wspace=0.2, figure=fig)
            ax1 = plt.subplot(grid[0:9, 0:9], projection=ccrs.Mercator())
            ax2 = plt.subplot(grid[0:9, 10:19], projection=ccrs.Mercator())
            ax3 = plt.subplot(grid[9:11, :])

            # grid = plt.GridSpec(8, 12,  hspace=1, figure=fig)
            # ax1 = plt.subplot(grid[0:4, 0:10], projection=ccrs.Mercator())
            # ax2 = plt.subplot(grid[4:, 0:10], projection=ccrs.Mercator())
            # ax3 = plt.subplot(grid[:, 10:11])

            first_line = f'Region:{region_name.title()}, Variable:{var_str}, Depth: {depth}m'
            second_line = f'\nTime: {str(time)} UTC\nGlider/Argo Search Window: {str(t0)} to {str(time)}'
            ax1 = region_subplot(fig, ax1, extent, rtofs_sub[k],  'RTOFS', argo, gliders, bathy, **vargs)
            ax2 = region_subplot(fig, ax2, extent, gofs_sub[k], 'GOFS', argo, gliders, bathy, **vargs)

            h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1


            if l:
                ax3.legend(h, l, ncol=6, loc='center', fontsize=10)
                ax3.set_axis_off()

            plt.suptitle(r"$\bf{" + first_line + "}$" + second_line, fontsize=13)
            # plt.tight_layout()

            plt.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            # plt.show()
            plt.close()


def transect(fig, ax, x, y, z, c, cmap=None, levels=None, isobath=None, flip_y=None):
    cmap = cmap or 'parula'
    levels = levels or dict(deep=np.arange(0, 28), shallow=np.arange(14, 28))
    flip_y = flip_y or True
    isobath = isobath or None
    levels = levels or [26]

    if not isinstance(isobath, list):
        isobath = [isobath]

    ax1 = ax.contourf(x, y, c, cmap=cmap, levels=levels['deep'], extend='both')

    if isobath:
        for line in isobath:
            ax.contour(x, y, c, [line], colors='k')  # add contour at 26m

    if flip_y:
        ax.set_ylim(z, 0)

    fig.colorbar(ax1, ax=ax, orientation='vertical')
    return ax


def plot_transect(x, y, c, cmap, title=None, save_file=None, flip_y=None, levels=None):
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    # Initiate transect plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax = transect(fig, ax, x, y, c, cmap, levels, flip_y)

    # Add titles and labels
    plt.suptitle(title, size=12)
    plt.setp(ax, ylabel='Depth (m)')
    plt.setp(ax, ylabel='Depth (m)', xlabel='Longitude')
    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_transects(x, y, c, xlabel, cmap=None, title=None, save_file=None, flip_y=None, levels=None, isobath=None):
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    fig, ax = plt.subplots(figsize=(11, 8))
    grid = plt.GridSpec(2, 1, wspace=0.2, hspace=0.2)
    ax1 = plt.subplot(grid[0, :])  # first half
    ax2 = plt.subplot(grid[1, :])  # second half

    # 1000m subplot
    ax1 = transect(fig, ax1, x, y, 1000, c, cmap, levels, isobath, flip_y)

    # 300m subplot
    ax2 = transect(fig, ax2, x, y, 300, c, cmap, levels, isobath, flip_y)

    # Add titles and labels
    plt.suptitle(title, size=12)
    plt.setp(ax1, ylabel='Depth (m)')
    plt.setp(ax2, ylabel='Depth (m)', xlabel=xlabel)
    # plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_region(da, extent, title, save_file, transects=None, argo=None, gliders=None, bathy=None, vmin=None,
                vmax=None, transform=None, cmap=None, levels=None):

    transform = transform or ccrs.PlateCarree()

    # fig, ax = plt.subplots(
    #     figsize=(11, 8),
    #     subplot_kw=dict(projection=transform),
    # )
    # plt.rcParams['figure.constrained_layout.use'] = True

    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['figure.constrained_layout.use'] = True
    grid = plt.GridSpec(20, 6, wspace=0.2, hspace=0.2, figure=fig)
    ax1 = plt.subplot(grid[0:16, :], projection=transform)
    ax2 = plt.subplot(grid[17:, :])

    vargs = {}
    vargs['vmin'] = vmin
    vargs['vmax'] = vmax
    vargs['transform'] = transform
    vargs['cmap'] = cmap
    vargs['levels'] = levels
    vargs['title'] = title
    vargs['argo'] = argo
    vargs['gliders'] = gliders
    vargs['bathy'] = bathy
    vargs['transects'] = transects
    vargs['da'] = da

    ax1 = region_subplot(fig, ax1, extent, **vargs)

    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1

    ax2.legend(h, l, ncol=10, loc='center', fontsize=8)
    ax2.set_axis_off()

    plt.savefig(save_file, dpi=150)


def region_subplot(fig, ax, extent, da=None, title=None, argo=None, gliders=None, bathy=None, transects=None, vmin=None,
                vmax=None, transform=None, cmap=None, levels=None, ticks=None, colorbar=True):
    if argo is None:
        argo = pd.DataFrame()

    if gliders is None:
        gliders = pd.DataFrame()

    if transects is None:
        transects = pd.DataFrame()
    # bathy = bathy or None
    # transects = transects or pd.DataFrame()

    if da is not None:
        cargs = {}
        cargs['vmin'] = vmin
        cargs['vmax'] = vmax
        cargs['transform'] = transform
        cargs['cmap'] = cmap
        cargs['levels'] = levels
        cargs['extend'] = 'both'
        h = ax.contourf(da['lon'], da['lat'], da.squeeze(), **cargs)

    # if limits['currents']['bool']:
    #     q = add_currents(limits['currents']['coarsen'], dsd)

    map_add_features(ax, extent)

    if not argo.empty:
        map_add_argo(ax, argo, transform)

    if not gliders.empty:
        map_add_gliders(ax, gliders, transform)

    if bathy:
        map_add_bathymetry(ax, bathy, transform)

    if not transects.empty:
        map_add_transects(ax, transects, transform)

    ax.set_title(title, fontsize=18, fontweight='bold')

    if ticks:
        map_add_ticks(ax, extent)

    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)

    axins = inset_axes(ax,  # here using axis of the lowest plot
                       width="2.5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 340% good for a (4x4) Grid
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    if colorbar:
        cb = fig.colorbar(h, cax=axins)
        cb.ax.tick_params(labelsize=12)

    return ax