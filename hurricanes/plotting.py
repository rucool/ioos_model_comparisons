import copy
import os
import pickle
import sys
import warnings
from collections import namedtuple
from glob import glob
from itertools import cycle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from dateutil import parser
from matplotlib import cm
from matplotlib import colors as c
# Normalize data with a set center.
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               TwoSlopeNorm)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from oceans.ocfis import spdir2uv, uv2spdir
from pytz import timezone

from hurricanes.calc import dd2dms

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

try:
    from urllib.request import urlopen, urlretrieve
except Exception:
    from urllib import urlopen, urlretrieve


proj = dict(
    map=ccrs.Mercator(), # the projection that you want the map to be in
    data=ccrs.PlateCarree() # the projection that the data is. 
    )

def export_fig(path, fname, script=None, dpi=150):
    """
    Helper function to save a figure with some nice formatting.
    Include script to print the script that created the plot for future ref.

    Args:
        path (str or Path): Full file name including path
        script (str, optional): Print name of script on plot. Defaults to None.
        dpi (int, optional): Dots per inch. Defaults to 150.
    """
    os.makedirs(path, exist_ok=True)
    
    if script:
        import datetime as dt
        now = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        plt.figtext(.98, 0.20, f"{script} {now}",  fontsize=10, rotation=90)
        
    plt.savefig(path / fname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    # plt.clf()


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
def  get_ticks(bounds, dirs, otherbounds):
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


def map_add_argo(ax, df, transform=proj['data']):
    tdf = df.reset_index()
    most_recent = tdf.loc[tdf.groupby('argo')['time'].idxmax()]

    if most_recent.shape[0] > 50:
        custom_cmap = matplotlib.colors.ListedColormap('red', N=most_recent.shape[0])
        marker = cycle(['o'])
    else:
        custom_cmap = categorical_cmap(10, 5, cmap="tab10")
        marker = cycle(['o', 'h', 'p'])

    n = 0
    for float in most_recent.itertuples():
        ax.plot(float.lon, float.lat, 
                marker=next(marker), linestyle="None",
                markersize=7, markeredgecolor='black', 
                color=custom_cmap.colors[n],
                label=float.argo,
                transform=transform,
                zorder=10000)
        # map_add_legend(ax)
        n = n + 1


def map_add_all_argo(ax, df, transform=proj['data']):
    grouped = df.groupby(['longitude (degrees_east)', 'latitude (degrees_north)'])
    for i, x in grouped:
        ax.plot(i[0], i[1], marker='o', markersize=7, markeredgecolor='black', color='green', transform=transform)


def map_add_bathymetry(ax, lon, lat, elevation, levels=(-1000), zorder=5,
                       transform=proj['data']):
    # lon = ds.variables['longitude'][:]
    # lat = ds.variables['latitude'][:]
    # elevation = ds.variables['elevation'][:]
    lons, lats = np.meshgrid(lon, lat)
    h = ax.contour(lons, lats, elevation, levels, 
                    linewidths=.75, alpha=.5, colors='k', 
                    transform=transform, 
                    # transform_first=True, # This might speed it up
                    zorder=zorder)
    ax.clabel(h, levels, inline=True, fontsize=6, fmt=fmt)
    return ax


def map_add_currents(ax, ds, coarsen=None, ptype="quiver",
                    scale=90, headwidth=2.75, headlength=2.75, headaxislength=2.5,
                    density=2, linewidth=.75, color='black',
                    transform=proj['data']):
    """
    Add currents to map

    Args:
        ax (ax): matplotlib.Axes
        ds (xarray.DataSet): xarray 
        coarsen (_type_, optional): Amount to downsample by. Defaults to None.
        ptype (str, optional): Plot type: "quiver" or "streamplot". Defaults to "quiver".
        scale (int, optional): _description_. Defaults to 90.
        headwidth (float, optional): _description_. Defaults to 2.75.
        headlength (float, optional): _description_. Defaults to 2.75.
        headaxislength (float, optional): _description_. Defaults to 2.5.
        transform (_type_, optional): _description_. Defaults to ccrs.PlateCarree().
        density (int, optional): _description_. Defaults to 3.
        linewidth (float, optional): Line width for streamplot. Defaults to .75.
        color (str, optional): Line color for streamplot. Defaults to 'black'.

    Returns:
        _type_: _description_
    """
    angle, speed = uv2spdir(ds['u'], ds['v'])  # convert u/v to angle and speed
    
    if ptype == "quiver":
        if coarsen:
            try:
                ds = ds.coarsen(lon=coarsen, boundary='pad').mean().coarsen(lat=coarsen, boundary='pad').mean()
                mesh = True
            except ValueError:
                ds = ds.coarsen(x=coarsen, boundary='pad').mean().coarsen(y=coarsen, boundary='pad').mean()
                mesh = False

        u, v = spdir2uv(  # convert angle and speed back to u/v, normalizing the arrow sizes
            np.ones_like(speed),
            angle,
            deg=True
        )

        qargs = {}
        qargs['scale'] = scale
        qargs['headwidth'] = headwidth
        qargs['headlength'] = headlength
        qargs['headaxislength'] = headaxislength
        qargs['transform'] = transform

        if mesh:
            lons, lats = np.meshgrid(ds['lon'], ds['lat'])
            q = ax.quiver(lons, lats, u, v, **qargs)
        else:
            q = ax.quiver(
                ds.lon.squeeze().data,
                ds.lat.squeeze().data, 
                u.squeeze(), 
                v.squeeze(), 
                **qargs)
    elif ptype == "streamplot":
        lons = ds.lon.squeeze().data
        lats = ds.lat.squeeze().data
        u = ds.u.squeeze().data
        v = ds.v.squeeze().data
        
        sargs = {}
        sargs["transform"] = transform
        sargs["density"] = density
        sargs["linewidth"] = linewidth
        if color:
            sargs["color"] = color
        else:
            sargs["color"] = speed
            sargs["cmap"] = cmocean.cm.speed
        q = ax.streamplot(lons, lats, u, v, **sargs)
    return q


def map_add_features(ax, extent, edgecolor="black", landcolor="tan", zorder=0):
    """_summary_

    Args:
        ax (_type_): _description_
        extent (_type_): _description_
        edgecolor (str, optional): _description_. Defaults to "black".
        landcolor (str, optional): _description_. Defaults to "tan".
        zorder (int, optional): _description_. Defaults to 0.
    """

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )

    LAND = cfeature.GSHHSFeature(scale='full')

    # Axes properties and features
    ax.set_extent(extent)
    # ax.add_feature(cfeature.OCEAN, zorder=zorder) #cfeature.OCEAN has a major performance hit
    ax.set_facecolor(cfeature.COLORS['water']) # way faster than adding the ocean feature above
    ax.add_feature(LAND, 
                   edgecolor=edgecolor, 
                   facecolor=landcolor,
                   zorder=zorder+10)
    ax.add_feature(cfeature.RIVERS, zorder=zorder+10.2)
    ax.add_feature(cfeature.LAKES, zorder=zorder+10.2, alpha=0.5)
    ax.add_feature(state_lines, edgecolor=edgecolor, zorder=zorder+10.3)
    ax.add_feature(cfeature.BORDERS, linestyle='--', zorder=zorder+10.3)


def map_add_gliders(ax, df, transform=proj['data'], color='white'):
    for g, new_df in df.groupby(level=0):
        q = new_df.iloc[-1]
        ax.plot(new_df['lon'], new_df['lat'], color=color,
                linewidth=1.5, transform=transform, zorder=10000)
        ax.plot(q['lon'], q['lat'], marker='^', markeredgecolor='black',
                markersize=8.5, label=g, transform=transform, zorder=10000)
        # map_add_legend(ax)


def map_add_legend(ax):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def map_add_ticks(ax, extent, fontsize=13, transform=proj['data']):
    xl = [extent[0], extent[1]]
    yl = [extent[2], extent[3]]

    tick0x, tick1, ticklab = get_ticks(xl, 'we', yl)
    ax.set_xticks(tick0x, minor=True, crs=transform)
    ax.set_xticks(tick1, crs=transform)
    ax.set_xticklabels(ticklab, fontsize=fontsize)

    # get and add latitude ticks/labels
    tick0y, tick1, ticklab = get_ticks(yl, 'sn', xl)
    ax.set_yticks(tick0y, minor=True, crs=transform)
    ax.set_yticks(tick1, crs=transform)
    ax.set_yticklabels(ticklab, fontsize=fontsize)

    # gl = ax.gridlines(draw_labels=False, linewidth=.5, color='gray', alpha=0.75, linestyle='--', crs=ccrs.PlateCarree())
    # gl.xlocator = mticker.FixedLocator(tick0x)
    # gl.ylocator = mticker.FixedLocator(tick0y)

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

    # if grid:
        # ax.grid(color='k', linestyle='--', zorder=zorder)
    return ax


def map_add_transects(ax, transects, transform=proj['data']):
    ax.plot(transects['lon'], transects['lat'], 'r-', transform=transform)


def map_load(figdir):
    with open(figdir, "rb") as file:
        fig = pickle.load(file)
    return fig


def map_save(fig, figdir):
    with open(figdir, 'wb') as file:
        pickle.dump(fig, file)

      
def map_create(extent, 
               proj=proj['map'],
               labelsize=14,
               ticks=True,
               labels=False,
               features=True, edgecolor="black", landcolor="tan",
               ax=None, figsize=(11,8), fig_init=False):
    """Create a cartopy map within a certain extent. 

    Args:
        extent (tuple or list): Extent (x0, x1, y0, y1) of the map in the given coordinate system.
        proj (cartopy.crs class, optional): Define a projected coordinate system with flat topology and Euclidean distance.
            Defaults to ccrs.Mercator().
        features (bool, optional): Add preferred map settings. 
            Defaults to True.
        ax (_type_, optional): Pass matplotlib axis to function. Not necessary if plotting to subplot. 
            Defaults to None.
        figsize (tuple, optional): (width, height) of the figure. Defaults to (11, 8).

    Returns:
        _type_: _description_
    """
    # If a matplotlib axis is not passed, create a new cartopy/mpl figure
    if not ax:
        fig_init = True
        fig, ax = plt.subplots(
            figsize=figsize, #12,9
            subplot_kw=dict(projection=proj)
        )

    # Make the map pretty
    if features:
        fargs = {
            "edgecolor": edgecolor,
            "landcolor": landcolor,
            }
        map_add_features(ax, extent, **fargs)

    # # Add bathymetry
    # if bathy:
    #     bargs = {
    #         "isobaths": isobaths,
    #         "zorder": 1.5      
    #     }
    #     map_add_bathymetry(ax, bathy, proj, **bargs)

    # Add ticks
    if ticks:
        map_add_ticks(ax, extent)

    if labels:
        # Set labels
        ax.set_xlabel('Longitude', fontsize=labelsize, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=labelsize, fontweight='bold')

    # If we generate a figure in this function, we have to return the figure
    # and axis to the calling function.
    if fig_init:
        return fig, ax


def plot_model_region(ds, region,
                      bathy=None,
                      argo=None,
                      gliders=None,
                      currents=dict(bool=False),
                      transform=dict(
                          map=proj['map'],
                          data=proj['data']
                          ),
                      legend=True,
                      model='rtofs',
                      path_save=os.getcwd(),
                      dpi=150,
                      t0=None):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    region_name = region["name"]
    extent = region["extent"]
    time = pd.to_datetime(ds.time.values)

    # Create subdirectory for region
    region_file_str = '_'.join(region_name.lower().split(' '))
    path_save_region = path_save / 'regions' / region_file_str

    if not isinstance(gliders, pd.DataFrame):
        gliders = pd.DataFrame()
    
    if not isinstance(argo, pd.DataFrame):
        argo = pd.DataFrame()

    # Iterate through the region dictionary. This dict contains information
    # on what variables and depths to plot. 
    for key, values in region["variables"].items():
        # Create subdirectory for variable under region directory
        var_str = ' '.join(key.split('_')).title()

        # Iterate through values of the key
        for item in values:
            depth = item['depth']

            # Select variable and depth to plot
            # print(ds[k].name)
            try:
                da = ds[key].sel(depth=depth)
            except KeyError:
                da = ds[key]

            # Create subdirectory for depth under variable subdirectory
            save_dir_final = path_save_region / f"{key}_{depth}m" / time.strftime('%Y/%m')
            os.makedirs(save_dir_final, exist_ok=True)

            # Create a string for the title of the plot
            title_time = time.strftime("%Y-%m-%d %H:%M:%S")
            title = f"{model.upper()} - {var_str} ({depth} m) - {title_time}\n"

            # if not gliders.empty or not argo.empty:
            #         title += f'Assets ({str(t0)} to {str(time)})'

            # Create a file name to save the plot as
            sname = f'{model}-{key}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            save_file = save_dir_final / f"{sname}.png"

            # Create a map figure and serialize it if one doesn't already exist
            region_name = "_".join(region["name"].split(' ')).lower()
            path_maps = path_save / "mapfigs"
            os.makedirs(path_maps, exist_ok=True)
            sfig = (path_maps / f"{region_name}_fig.pkl")

            if not sfig.exists():
                # Create an empty projection within set extent
                fig, ax = map_create(extent, proj=transform['map'])

                # Add bathymetry
                if bathy:
                    map_add_bathymetry(ax,
                                    bathy.longitude.values, 
                                    bathy.latitude.values, 
                                    bathy.elevation.values,
                                    levels=(-1000, -100),
                                    zorder=1.5)

                map_save(fig, sfig)        
            else:
                fig = map_load(sfig)
                ax = fig.axes[0]
                               
            rargs = {}
            rargs['argo'] = argo
            rargs['gliders'] = gliders
            rargs['transform'] = transform['data']  
            plot_regional_assets(ax, **rargs)

            cargs = {}
            cargs['vmin'] = item['limits'][0]
            cargs['vmax'] = item['limits'][1]
            cargs['transform'] = transform['data']
            cargs['cmap'] = cmaps(da.name)
            cargs['levels'] = np.arange(cargs['vmin'], cargs['vmax'], item['limits'][2])
            cargs['extend'] = 'both'

            try:
                cargs.pop('vmin'), cargs.pop('vmax')
            except KeyError:
                pass
            
            # If the xarray DataArray contains data, let's contour the data.
            if da is not None:
                h = ax.contourf(da['lon'], da['lat'], da.squeeze(), **cargs)

                # Create the colorbar
                axins = inset_axes(ax,  # here using axis of the lowest plot
                    width="2.5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 340% good for a (4x4) Grid
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0., 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0
                    )
                cb = plt.colorbar(h, cax=axins)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(f'{da.name.title()} ({da.units})', fontsize=13)

            ax.set_title(title, fontsize=16, fontweight='bold')

            if legend:
                h, l = ax.get_legend_handles_labels()  # get labels and handles from ax1

                if (len(h) > 0) & (len(l) > 0):
                    # Shrink current axis's height by 10% on the bottom
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                    box.width, box.height * 0.9])

                    # Put a legend below current axis
                    ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=True, ncol=5)
                    legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
                    plt.figtext(0.5, -0.07, legstr, ha="center", fontsize=10, fontweight='bold')

            fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Add currents
            if currents['bool']:
                quiver_dir = save_dir_final / "currents"
                os.makedirs(quiver_dir, exist_ok=True)
                
                save_file_q = quiver_dir / f"{sname}.png"
                coarsen = currents['coarsen']
                map_add_currents(ax, da, coarsen=coarsen[model], **currents['kwargs'])
                fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            
            plt.close()


def remove_quiver_handles(ax):
    for art in ax.get_children():
        if isinstance(art, matplotlib.patches.FancyArrowPatch):
            art.remove()      


def plot_model_region_comparison(ds1, ds2, region,
                                 bathy=None,
                                 argo=None,
                                 gliders=None,
                                 currents=None,
                                 eez=False,
                                 transform=dict(map=proj['map'], 
                                                data=proj['data']
                                                ),
                                 path_save=os.getcwd(),
                                 dpi=150,
                                 t0=None,
                                 colorbar=True,
                                 overwrite=False):
    
    time = pd.to_datetime(ds1.time.data)
    region_name = region["name"]
    extent = region['extent']

    # Create subdirectory for region
    region_file_str = ('_').join(region_name.lower().split(' '))
    path_save_region = path_save / 'regions' / region_file_str
    
    # # Create a map figure and serialize it if one doesn't already exist
    # region_name = "_".join(region["name"].split(' ')).lower()
    # mdir = path_save / "mapfigs"
    # os.makedirs(mdir, exist_ok=True)
    # sfig = mdir / f"{region_name}_fig.pkl"

    # if not sfig.exists():
    # Create figure

    # grid = [
    #     ["rtofs", "gofs"],
    #     ["legend", "legend"]
    #     ]
    grid = """
    RG
    LL
    """
    try:
        figsize = region['figure']['figsize']
    except KeyError:
        figsize = (16,9)

    fig, _ = plt.subplot_mosaic(
        grid,
        figsize=figsize,
        layout="constrained",
        subplot_kw={
            'projection': transform['map']
            },
        gridspec_kw={
            # set the height ratios between the rows
            "height_ratios": [4, 1],
            # set the width ratios between the columns
            # # "width_ratios": [1],
            },
        )
    axs = fig.axes
    ax1 = axs[0] # Model 1
    ax2 = axs[1] # Model 2
    ax3 = axs[2] # Legend for argo/gliders
            
    # Make the map pretty
    map_add_features(ax1, extent)# zorder=0)
    if bathy:
        map_add_bathymetry(ax1,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5)
    map_add_ticks(ax1, extent)

    # ax1.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    # ax1.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    # Make the map pretty
    map_add_features(ax2, extent)# zorder=0)
    if bathy:
        map_add_bathymetry(ax2,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5)
    map_add_ticks(ax2, extent)

    # ax2.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    #             with open(sfig, 'wb') as file:
    #                 pickle.dump(fig, file)
    # else:
    #     with open(sfig, "rb") as file:
    #         fig = pickle.load(file)
    #         axs = fig.axes
    #         ax1 = axs[0]
    #         ax2 = axs[1]
    #         ax3 = axs[2]

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    plot_regional_assets(ax1, **rargs)
    plot_regional_assets(ax2, **rargs)

    # Label the subplots
    ax1.set_title(ds1.model, fontsize=22, fontweight="bold")
    ax2.set_title(ds2.model, fontsize=22, fontweight="bold")
    
    # Deal with the third axes
    h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        ax3.legend(h, l, ncol=region['figure']['legend']['columns'], loc='center', fontsize=10)
        legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
        # ax3.set_title(legstr, loc="right", fontsize=10, fontweight="bold")
        # plt.figtext(0.5, 0.01, legstr, ha="center", fontsize=10, fontweight='bold')
        plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    
    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for k, v in region["variables"].items():
        # Create subdirectory for variable under region directory
        var_str = ' '.join(k.split('_')).title()

        for item in v:
            depth = item['depth']
            rsub = ds1[k].sel(depth=depth)
            gsub = ds2[k].sel(depth=depth, method='nearest')
            
            # Create subdirectory for depth under variable subdirectory
            save_dir_final = path_save_region / f"{k}_{depth}m" / time.strftime('%Y/%m')
            os.makedirs(save_dir_final, exist_ok=True)

            # Create a file name to save the plot as
            # sname = f'{ds1.model}_vs_{ds2.model}_{k}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            sname = f'{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}_{ds1.model.lower()}-vs-{ds2.model.lower()}'
            save_file = save_dir_final / f"{sname}.png"

            if save_file.is_file():
                if not overwrite:
                    print(f"{save_file} exists. Overwrite: False. Skipping.")
                    continue
                else:
                    print(f"{save_file} exists. Overwrite: True. Replotting.")
                    
            # else:

            # Add the super title (title for both subplots)
            title_time = time.strftime("%Y-%m-%d %H:%M:%S")
            title = f"{var_str} ({depth} m) - {title_time}\n"
            plt.suptitle(title, fontsize=24, fontweight="bold")

            # Create dictionary for variable argument inputs for contourf
            vargs = {}
            vargs['transform'] = transform['data']
            vargs['transform_first'] = True
            vargs['cmap'] = cmaps(ds1[k].name)
            vargs['extend'] = "both"

            if 'limits' in item:
                vargs['vmin'] = item['limits'][0]
                vargs['vmax'] = item['limits'][1]
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
        
            # Filled contour for each model variable
            if (rsub['lon'].ndim == 1) & rsub['lat'].ndim == 1:
                rlons, rlats = np.meshgrid(rsub['lon'], rsub['lat'])
            else:
                rlons = rsub['lon']
                rlats = rsub['lat']
            h1 = ax1.contourf(rlons, rlats, rsub.squeeze(), **vargs)

            # Check if ndims are 1, transform_first requires 2d array
            if (gsub['lon'].ndim == 1) & gsub['lat'].ndim == 1:
                glons, glats = np.meshgrid(gsub['lon'], gsub['lat'])
            else:
                glons = gsub['lon']
                glats = gsub['lat']
            h2 = ax2.contourf(glons, glats, gsub.squeeze(), **vargs)

            if colorbar:
                cb = fig.colorbar(h2, ax=axs[:2], location='bottom', aspect=80)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(f'{k.title()} ({rsub.units})', fontsize=12, fontweight="bold")

            # Add EEZ
            if eez:
                eez = '/Users/mikesmith/Documents/github/rucool/Daily_glider_models_comparisons/World_EEZ_v11_20191118/eez_boundaries_v11.shp'
                shape_feature = cfeature.ShapelyFeature(
                    Reader(eez).geometries(), 
                    proj['data'],
                    edgecolor='red', 
                    facecolor='none'
                    )
                ax1.add_feature(shape_feature, zorder=1)
                ax2.add_feature(shape_feature, zorder=1)
            
            # Plot size adjustments
            # fig.tight_layout()
            # fig.subplots_adjust(top=0.9) 

            # Save the figure. Using fig to savefig allows us to delete any
            # figure handles so that we can reuse the figure.
            fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

            # Add currents as overlays over variable plots
            if currents['bool']:
                quiver_dir = save_dir_final / "currents_overlay"
                os.makedirs(quiver_dir, exist_ok=True)
                
                save_file_q = quiver_dir / f"{sname}.png"
                coarsen = currents['coarsen']
                q1 = map_add_currents(ax1, rsub, coarsen=coarsen['rtofs'], **currents['kwargs'])
                q2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents['kwargs'])
                fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                q1.lines.remove(), q2.lines.remove()
                remove_quiver_handles(ax1), remove_quiver_handles(ax2)

            # Delete contour handles and remove colorbar axes to use figure
            [x.remove() for x in h1.collections]
            [x.remove() for x in h2.collections]
            cb.remove()
            # # plt.close()
            
            # # Plot currents with magnitude and direction
            # quiver_dir = path_save_region / f"currents_{depth}m" / time.strftime('%Y/%m') / "currents"
            # os.makedirs(quiver_dir, exist_ok=True)
            
            # sname = f'currents-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            # save_file_q = quiver_dir / f"{sname}.png"

            # if not save_file_q.is_file():
            #     _, mag_r = uv2spdir(rsub['u'], rsub['v'])
            #     _, mag_g = uv2spdir(gsub['u'], gsub['v'])

            #     qargs = {}
            #     qargs['vmin'] = 0
            #     qargs['vmax'] = 1.20
            #     qargs['levels'] = np.arange(0, 1.2 + .1, .10)
            #     qargs['transform'] = transform['data']
            #     qargs['cmap'] = cmocean.cm.speed
            #     qargs['extend'] = "both"

            #     # Set coarsening configs to a variable
            #     coarsen = currents['coarsen']
                
            #     # Filled contour for each model variable
            #     m1 = ax1.contourf(rsub["lon"], rsub["lat"], mag_r, **qargs)
            #     m2 = ax2.contourf(gsub["lon"], gsub["lat"], mag_g, **qargs)
            #     s1 = map_add_currents(ax1, rsub, coarsen=coarsen["rtofs"], **currents["kwargs"])
            #     s2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents["kwargs"])
                
            #     cb = fig.colorbar(s2.lines, ax=axs[:2], location='bottom', aspect=80)#, shrink=0.7, aspect=20*0.7)
            #     cb.ax.tick_params(labelsize=12)
            #     cb.set_label(f'Magnitude', fontsize=12, fontweight="bold")

            #     # Create a string for the title of the plot
            #     title = f"Currents ({depth} m) - {title_time}\n"
            #     plt.suptitle(title, fontsize=24, fontweight="bold")

            #     # Save figure
            #     fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            #     cb.remove()
            #     # Delete contour handles and remove colorbar axes to use figure
            #     s1.lines.remove(), s2.lines.remove()
            #     remove_quiver_handles(ax1), remove_quiver_handles(ax2)
            #     [x.remove() for x in m1.collections]
            #     [x.remove() for x in m2.collections]


def plot_regional_assets(ax, argo=None, gliders=None, 
                         transects=None,
                         transform=None):
    if argo is None:
        argo = pd.DataFrame()

    if gliders is None:
        gliders = pd.DataFrame()

    if transects is None:
        transects = pd.DataFrame()

    if not argo.empty:
        map_add_argo(ax, argo, transform)

    if not gliders.empty:
        map_add_gliders(ax, gliders, transform)

    if not transects.empty:
        map_add_transects(ax, transects, transform)



def transect(fig, ax, x, y, z, c, cmap=None, levels=None, isobath=None, flip_y=None):
    cmap = cmap or 'parula'
    levels = levels or dict(deep=np.arange(0, 28), shallow=np.arange(14, 28))
    flip_y = flip_y or True
    isobath = isobath or None
    levels = levels or [26]

    if not isinstance(isobath, list):
        isobath = [isobath]

    offset = TwoSlopeNorm(vcenter=0)

    ax1 = ax.contourf(x, y, c, cmap=cmap, levels=levels['deep'], extend='both', norm=offset)

    if isobath:
        for line in isobath:
            ax.contour(x, y, c, [line], colors='k')  # add contour at 26m

    if flip_y:
        ax.set_ylim(z, 0)

    cb = fig.colorbar(ax1, ax=ax, orientation='vertical')
    cb.set_label('m/s', fontweight='bold')
    return ax


def plot_transect(x, y, c, xlabel, cmap, title=None, save_file=None, flip_y=None, levels=None, isobath=None):
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    # Initiate transect plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax = transect(fig, ax, x, y, 1000, c, cmap, levels, isobath, flip_y)

    # Add titles and labels
    plt.suptitle(title, size=16, fontweight='bold')
    ax.set_ylabel('Depth (m)', size=12, fontweight='bold')
    ax.set_xlabel(xlabel, size=12, fontweight='bold')

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()


def plot_transects(x, y, c, xlabel, cmap=None, title=None, save_file=None, flip_y=None, levels=None, isobath=None):
    title = title or 'Transect Plot'
    save_file = save_file or 'transect.png'

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8)
        )

    # 1000m subplot
    ax1 = transect(fig, ax1, x, y, 1000, c, cmap, levels, isobath, flip_y)

    # 300m subplot
    ax2 = transect(fig, ax2, x, y, 100, c, cmap, levels, isobath, flip_y)

    # Add titles and labels
    plt.suptitle(title, size=16, fontweight='bold')
    ax1.set_ylabel('Depth (m)', size=12, fontweight='bold')
    ax2.set_ylabel('Depth (m)', size=12, fontweight='bold')
    ax2.set_xlabel(xlabel, size=12, fontweight='bold')

    plt.tight_layout()

    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def plot_model_region_comparison_streamplot(rtofs, gofs, region,
                                            bathy=None,
                                            argo=None,
                                            gliders=None,
                                            currents=None,
                                            transform=dict(map=proj['map'], 
                                                           data=proj['data']
                                                           ),
                                            path_save=os.getcwd(),
                                            dpi=150,
                                            t0=None,
                                            colorbar=True):

    time = pd.to_datetime(rtofs.time.data)
    region_name = region["name"]
    extent = region['extent']

    # Create subdirectory for region
    region_file_str = ('_').join(region_name.lower().split(' '))
    path_save_region = path_save / 'regions' / region_file_str
    
    grid = """
    RG
    LL
    """
    try:
        figsize = region['figure']['figsize']
    except KeyError:
        figsize = (16,9)

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
    if bathy:
        map_add_bathymetry(ax1,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5)
    map_add_ticks(ax1, extent)

    # ax1.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    # ax1.set_ylabel('Latitude', fontsize=14, fontweight='bold')

    # Make the map pretty
    map_add_features(ax2, extent)# zorder=0)
    if bathy:
        map_add_bathymetry(ax2,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5)
    map_add_ticks(ax2, extent)

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    plot_regional_assets(ax1, **rargs)
    plot_regional_assets(ax2, **rargs)

    # Label the subplots
    ax1.set_title("RTOFS", fontsize=22, fontweight="bold")
    ax2.set_title("GOFS", fontsize=22, fontweight="bold")
    
    # Deal with the third axes
    h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        ax3.legend(h, l, ncol=region['figure']['legend']['columns'], loc='center', fontsize=10)
        legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
        # ax3.set_title(legstr, loc="right", fontsize=10, fontweight="bold")
        # plt.figtext(0.5, 0.01, legstr, ha="center", fontsize=10, fontweight='bold')
        plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    
    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for depth in region['currents']['depths']:
        print(depth)
        rsub = rtofs.sel(depth=depth)
        gsub = gofs.sel(depth=depth)
        
        # Plot currents with magnitude and direction
        quiver_dir = path_save_region / f"currents_{depth}m" / time.strftime('%Y/%m') / "currents"
        os.makedirs(quiver_dir, exist_ok=True)
        
        sname = f'currents-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
        save_file_q = quiver_dir / f"{sname}.png"

        if not save_file_q.is_file():
            _, mag_r = uv2spdir(rsub['u'], rsub['v'])
            _, mag_g = uv2spdir(gsub['u'], gsub['v'])

            qargs = {}
            # qargs['vmin'] = 0
            # qargs['vmax'] = 1.20
            # qargs['levels'] = np.arange(0, 1.2 + .1, .10)
            qargs['transform'] = transform['data']
            qargs['cmap'] = cmocean.cm.speed
            qargs['extend'] = "both"

            # Set coarsening configs to a variable
            coarsen = region['currents']['coarsen']
            
            # Filled contour for each model variable
            m1 = ax1.contourf(rsub["lon"], rsub["lat"], mag_r, **qargs)
            m2 = ax2.contourf(gsub["lon"], gsub["lat"], mag_g, **qargs)
            s1 = map_add_currents(ax1, rsub, coarsen=coarsen["rtofs"], **currents["kwargs"])
            s2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents["kwargs"])
            
            cb = fig.colorbar(s2.lines, ax=axs[:2], location='bottom', aspect=80)#, shrink=0.7, aspect=20*0.7)
            cb.ax.tick_params(labelsize=12)
            cb.set_label(f'Magnitude', fontsize=12, fontweight="bold")

            # Create a string for the title of the plot
            title_time = time.strftime("%Y-%m-%d %H:%M:%S")
            title = f"Currents ({depth} m) - {title_time}\n"
            plt.suptitle(title, fontsize=24, fontweight="bold")

            # Save figure
            fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            cb.remove()
            # Delete contour handles and remove colorbar axes to use figure
            s1.lines.remove(), s2.lines.remove()
            remove_quiver_handles(ax1), remove_quiver_handles(ax2)
            [x.remove() for x in m1.collections]
            [x.remove() for x in m2.collections]