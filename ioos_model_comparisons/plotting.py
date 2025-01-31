import os
# import pickle
import warnings
from itertools import cycle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors
# import matplotlib.lines as mlines
import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxesSubplot
from ioos_model_comparisons.calc import categorical_cmap
from cool_maps.plot import (add_bathymetry, 
                            add_features, 
                            add_ticks,
                            create, 
                            save_fig, 
                            load_fig)
from matplotlib.colors import TwoSlopeNorm
# from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
from oceans.ocfis import spdir2uv, uv2spdir
from scipy.io import loadmat
from shapely.geometry.polygon import LinearRing

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import dd2dms
import scipy.ndimage as ndimage
import matplotlib.lines as mlines
import datetime as dt
import xarray as xr
import tcmarkers
# import xesmf as xe

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tropycal import tracks
from datetime import datetime
import numpy as np

# basin = tracks.TrackDataset(basin='north_atlantic', include_btk=True)

# Suppresing warnings for a "pretty output."
warnings.simplefilter("ignore")

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
    
    if isinstance(path, str):
        path = Path(path)
    
    os.makedirs(path, exist_ok=True)
    
    if script:
        now = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        plt.figtext(.98, 0.20, f"{script} {now}",  fontsize=10, rotation=90)
        
    plt.savefig(path / fname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def cmaps(variable):
    if variable == 'salinity':
        cmap = cmocean.cm.haline
    elif variable == 'temperature':
        cmap = cmocean.cm.thermal
    elif variable == 'sea_surface_height':
        cmap = cmocean.cm.balance
    return cmap


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


def map_add_all_argo(ax, df, transform=proj['data'], markersize=7):
    grouped = df.groupby(['lon', 'lat'])
    for i, x in grouped:
        ax.plot(i[0], i[1], marker='o', markersize=markersize, markeredgecolor='black', color='green', transform=transform)


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


def map_add_eez(ax, zorder=1, color='white', linewidth=0.75, linestyle='-.'):
    # reader = Reader('').geometries()

    # filtered = []
    # for record in reader.records():
    #     if record.attributes['LINE_TYPE'] == 'Straight Baseline':
    #         continue
    #     else:
    #         filtered.append(record.geometry)
            
    shape_feature = cfeature.ShapelyFeature(
        Reader(conf.eez_path).geometries(),
        proj['data'],
        linestyle=linestyle,
        linewidth=linewidth,
        edgecolor=color, 
        facecolor='none'
        )
    h = ax.add_feature(shape_feature, zorder=zorder)
    return h


def map_add_gliders(ax, df, transform=proj['data'], color='white'):
    for g, new_df in df.groupby(level=0):
        # if 'ru29' in g:
        q = new_df.iloc[-1]
        ax.plot(new_df['lon'], new_df['lat'], color=color,
                linewidth=8, transform=transform, zorder=10000)
        ax.plot(q['lon'], q['lat'], marker='^', markeredgecolor='black',
                markersize=14, label=g, transform=transform, zorder=10000)
        # map_add_legend(ax)


def map_add_inset(ax, x=.8, y=.3, size=.5, extent=None, zoom_extent=None):
    """_summary_

    Args:
        ax (_type_): _description_
        x (float, optional): inset x location relative to main plot (ax) in normalized units. Defaults to .8.
        y (float, optional): inset y location relative to main plot (ax) in normalized units. Defaults to .3.
        size (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    import cartopy
    # Inset Axis
    # axin = plt.axes([0, 0, 1, 1], projection=ccrs.Mercator())
    # position = [x - size / 2, y - size / 2, size, size]
    # ip = InsetPosition(ax, position)
    # axin.set_axes_locator(ip)
    axins = inset_axes(ax, width="40%", height="40%", loc="lower left", 
                       axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                       axes_kwargs=dict(map_projection=ccrs.Mercator()))
    axins.set_extent(extent)

    if zoom_extent:
        lonmin, lonmax, latmin, latmax = zoom_extent

        nvert = 100
        lons = np.r_[np.linspace(lonmin, lonmin, nvert),
                    np.linspace(lonmin, lonmax, nvert),
                    np.linspace(lonmax, lonmax, nvert)].tolist()
        lats = np.r_[np.linspace(latmin, latmax, nvert),
                    np.linspace(latmax, latmax, nvert),
                    np.linspace(latmax, latmin, nvert)].tolist()
        
        ring = LinearRing(list(zip(lons, lats)))
        axins.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=1, zorder=1000)
    return axins


def map_add_legend(ax):
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def map_add_transects(ax, transects, transform=proj['data']):
    ax.plot(transects['lon'], transects['lat'], 'r-', transform=transform)


def add_colorbar(ax, h, location="bottom", constrain=True):

    if isinstance(ax, list):
        # Multiple axes are being passed
        multi = True
    elif isinstance(ax, GeoAxesSubplot):
        multi = False

    # We want to minimize input parameters to functions. We need the fig
    # so we can use the .get_figure() method on the axes 
    if multi:
        fig = ax[0].get_figure()
    else:
        fig = ax.get_figure()

    if constrain:
        # Constrain the colorbar to the size of the plot
        if location == "bottom" or location == "top":
            if multi:
                # length = len(ax)
                widths = 250 + sum([a.bbox.width for a in ax])
                ax_ratio = 0.047*(widths/ax[0].bbox.height)
            else:
                ax_ratio = 0.047*(ax.bbox.width/ax.bbox.height)
        elif location == "left" or location == "right":
            if multi:
                ax_ratio = 0.047*(sum([a.bbox.height for a in ax])/ax[0].bbox.width)
            else:
                ax_ratio = 0.047*(ax.bbox.height/ax.bbox.width)
    else:
        ax_ratio = 0.15 # This is the default in matplotlib.
        
    # Add colorbar to axes
    cb = fig.colorbar(h, ax=ax, location=location, fraction=ax_ratio)
    return cb


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

            # if not sfig.exists():
                # Create an empty projection within set extent
            fig, ax = create(extent, proj=transform['map'])

            # Add bathymetry
            if bathy:
                add_bathymetry(ax,
                                bathy.longitude.values, 
                                bathy.latitude.values, 
                                bathy.z.values,
                                levels=(-1000, -100),
                                zorder=1.5)

            #     save_fig(fig, path_maps, f"{region_name}_fig.pkl")       
            # else:
            #     fig = load_fig(sfig)
            #     ax = fig.axes[0]
                               
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
            
            # # Add currents
            # if currents['bool']:
            #     quiver_dir = save_dir_final / "currents"
            #     os.makedirs(quiver_dir, exist_ok=True)
                
            #     save_file_q = quiver_dir / f"{sname}.png"
            #     coarsen = currents['coarsen']
            #     map_add_currents(ax, da, coarsen=coarsen[model], **currents['kwargs'])
            #     fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            
            plt.close()

def plot_model_region_currents(
    tds,
    region,
    transform=dict(
        map=proj['map'],
        data=proj['data']
        ),
    legend=True,
    path_save=Path(os.getcwd()),
    dpi=150,
    t0=None,
    currents=None
    ):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    # region_name = region["name"]
    # extent = region["extent"]
    region_name = 'yucatan'
    # extent = [-88, -84, 18, 22]
    extent = [-90, -78, 18, 28]

    # Create subdirectory for region
    region_file_str = '_'.join(region_name.lower().split(' '))
    path_save_region = path_save / 'regions' / region_file_str
    
    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save_region 
    os.makedirs(save_dir_final, exist_ok=True)

    # Create a string for the title of the plot
 
    # Create a file name to save the plot as
    save_file = save_dir_final / f"passengers.png"

    fig, ax = create(extent, proj=ccrs.Mercator(), bathymetry=False, figsize=(16,9))
                               
    url = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_overview/MapServer/WMSServer?request=GetCapabilities&service=WMS'
    url2 = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_general/MapServer/WMSServer?request=GetCapabilities&service=WMS'
    # url = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_coastal/MapServer/WMSServer?request=GetCapabilities&service=WMS'

    layers2 = [
        '1',
        '2',
        '3',
        '4',
        '5',
        #  '6',
        '7',
        '8',
        # '9',
        '10',
        '11',
        '12',
        '13',
        # '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        # '20',
        '21',
        # '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '31',
        # '32',
        '33',
        '34',
        # '35',
        '36',
        '37',
        # '38',
        '39',
        # '40',
        '41',
        '42',
        '43',
        # '44',
        '45',
        '46',
        '47',
        '48',
        # '49',
        '50',
        '51',
        '52',
        # '53',
        '54',
        # '55',
        '56',
        # '57',
        '58',
        '59',
        # '60',
        '61',
        # '62',
        # '63',
        '64',
        # '65',
        '66',
        '67',
        # '68',
        '69',
        # '70',
        '71',
        '72',
        '73',
        # '74',
        '75',
        # '76',
        '77',
        '78',
        # '79',
        '80',
        '81',
        # '82',
        '83',
        '84',
        '85',
        '86',
        # '87',
        '88',
        # '89',
        '90',
        '91',
        '92',
        '93',
        '94',
        # '95',
        '96',
        '97',
        '98',
        '99',
        '100',
        # '101',
        '102',
        '103',
        '104',
        '105',
        '106',
        '107',
        '108',
        # '109',
        '110',
        # '111',
        '112',
        '113',
        '114',
        '115',
        '116',
        '117',
        '118',
        '119',
        '120',
        '121',
        '122',
        '123',
        '124']

    layers = [
        '1',
        '2',
        '3',
        '4',
        '5',
        # '6',
        '7',
        '8',
        # '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        # '16',
        '17',
        # '18',
        '19',
        # '20',
        '21',
        # '22',
        '23',
        '24',
        # '25',
        '26',
        # '27',
        '28',
        '29',
        # '30',
        '31',
        # '32',
        '33',
        # '34',
        '35',
        '36',
        # '37',
        '38',
        '39',
        # '40',
        '41',
        '42',
        # '43',
        '44',
        # '45',
        '46',
        # '47',
        '48',
        '49',
        # '50',
        '51',
        '52',
        # '53',
        '54',
        # '55',
        '56',
        # '57',
        '58',
        '59',
        # '60',
        '61',
        '62',
        '63',
        # '64',
        '65',
        # '66',
        '67',
        '68',
        '69',
        '70',
        '71',
        # '72',
        '73',
        '74',
        '75',
        # '76',
        '77',
        # '78',
        '79',
        '80',
        '81',
        '82',
        '83',
        '84',
        # '85',
        '86',
        '87',
        '88',
        '89',
        '90',
        '91',
        '92',
        '93',
        '94',
        '95',
        '96',]


    
    # ax.add_wms(wms=url2, layers=layers2)
    
    # map_add_eez(ax, color='red')

        # # Create the colorbar
        # axins = inset_axes(ax,  # here using axis of the lowest plot
        #     width="2.5%",  # width = 5% of parent_bbox width
        #     height="100%",  # height : 340% good for a (4x4) Grid
        #     loc='lower left',
        #     bbox_to_anchor=(1.05, 0., 1, 1),
        #     bbox_transform=ax.transAxes,
        #     borderpad=0
        #     )
        # cb = plt.colorbar(h, cax=axins)
        # cb.ax.tick_params(labelsize=12)
        # cb.set_label(f'{da.name.title()} ({da.units})', fontsize=13)

    # ax.set_title(title, fontsize=16, fontweight='bold')

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

    # fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    # Add currents
    coarsen = currents['coarsen']
    map_add_currents(ax, tds, coarsen=coarsen['rtofs'], **currents['kwargs'])

    # Initialize qargs dictionary for input into contour plot of magnitude
    qargs = {}
    qargs['transform'] = transform['data']
    qargs['cmap'] = cmocean.cm.speed
    qargs['extend'] = "max"
    
    _, mag_r = uv2spdir(tds['u'], tds['v'])

    m1 = ax.contourf(tds["lon"], tds["lat"], mag_r, **qargs)
    cb = fig.colorbar(m1, ax=ax, orientation="vertical", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
    # cb = add_colorbar(axs[:2], m1, location="bottom")
    cb.ax.tick_params(labelsize=12)
    cb.set_label(f'Magnitude (m/s)', fontsize=12, fontweight="bold")

    field_lon = [-86.6021, -85.5144,-84.97, -84.9102, -85.1849, -85.5144, -86.6076]
    field_lat = [21.2004, 22.5968, 21.8544, 21.0056, 20.1880, 19.7544, 21.1901]
    ax.fill(field_lon, field_lat, color='coral', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)

    style = "Simple, tail_width=0.5, head_width=8, head_length=8"
    style1 = "Simple, tail_width=0.5, head_width=8, head_length=8"

    import matplotlib.patches as patches

    # Triangle
    kw = dict(arrowstyle=style1, color="red", alpha=.5, linewidth=2, transform=ccrs.PlateCarree(), zorder=3000, label='Fixed')
    a1 = patches.FancyArrowPatch((-85.700000, 20.300000), (-84.5, 21.5), **kw)
    # a2 = patches.FancyArrowPatch((-84.5, 21.5), (-85.25, 19.25), **kw)
    a3 = patches.FancyArrowPatch((-84.5, 21.5),(-85.700000, 18.850000), **kw)

    plt.gca().add_patch(a1)
    # plt.gca().add_patch(a2)
    plt.gca().add_patch(a3)

    # # Triangle 2
    kw = dict(arrowstyle=style, color="limegreen", linewidth=4, transform=ccrs.PlateCarree(), zorder=3000)
    a4 = patches.FancyArrowPatch((-87.051600, 18.819800), (-85.700000, 20.300000), **kw)
    a5 = patches.FancyArrowPatch((-85.700000, 20.300000), (-85.700000, 18.850000), **kw)
    a6 = patches.FancyArrowPatch((-85.700000, 18.850000), ( -87.051600, 18.819800), **kw)
    # 18.819800, -87.051600
    # 20.300000, -85.700000
    # 18.850000, -85.700000
    # 18.946810, -87.272580 - Deployment location

    d1 = ax.plot(-87.272580, 18.946810, markersize=8, marker='o', color='limegreen', linestyle='None', markeredgecolor='black', transform=ccrs.PlateCarree(), zorder=4000)
    
    plt.gca().add_patch(a4)
    plt.gca().add_patch(a5)
    plt.gca().add_patch(a6)

    # Curvy area
    kw = dict(arrowstyle=style, color="magenta", linewidth=4, transform=ccrs.PlateCarree(), zorder=3000, label='Expected')
    a4 = patches.FancyArrowPatch((-86.5, 20.5), (-86, 21.25), **kw)
    # a2 = patches.FancyArrowPatch((-85.75, 21.5), (-88, 25), connectionstyle="angle3, angleA=90, angleB=20", **kw)
    a5 = patches.FancyArrowPatch((-85.75, 21.5), (-88, 25), connectionstyle="angle3, angleA=70, angleB=15", **kw)
    a6 = patches.FancyArrowPatch((-85.75, 21.5), (-84, 23.5), connectionstyle="angle3, angleA=75, angleB=-20", **kw)
    a7 = patches.FancyArrowPatch((-85.75, 21.5), (-84, 24.5), connectionstyle="angle3, angleA=75, angleB=-40", **kw)
    a8 = patches.FancyArrowPatch((-84, 24), (-82.25, 24), **kw)
    # a6 = patches.FancyArrowPatch((-82, 24), (-81.25, 23.5), **kw)
    # a7 = patches.FancyArrowPatch((-80.75, 24), (-80, 25), **kw)
    # a3 = patches.FancyArrowPatch((-0.4, -0.6), (0.4, -0.6), connectionstyle="arc3,rad=.5", **kw)

    # for a in [a1, a2, a3]:
        # plt.gca().add_patch(a)
    plt.gca().add_patch(a4)
    plt.gca().add_patch(a5)
    plt.gca().add_patch(a6)
    plt.gca().add_patch(a7)
    plt.gca().add_patch(a8)
    # plt.gca().add_patch(a6)
    # plt.gca().add_patch(a7)

    # Area 1
    field_lon = [-86.5, -86.5, -87.3, -87.3, -86.5]
    field_lat = [19, 20, 20, 19, 19]
    ax.fill(field_lon, field_lat, color='coral', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    l1 = ax.plot(field_lon, field_lat, color='maroon', linewidth=4, zorder=3000, transform=ccrs.PlateCarree(), label='Fixed')
    
    # Area 2
    field_lon = [-86.75, -86, -85.5, -86.25, -86.75]
    field_lat = [20.75, 20.75, 21.5, 21.5, 20.75]
    ax.fill(field_lon, field_lat, color='coral', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    ax.plot(field_lon, field_lat, color='maroon', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # # Florida
    ax.text(-82, 27.5, 'Florida', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    # # Cuba
    ax.text(-81, 22.4, 'Cuba', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    # # Mexico
    ax.text(-89.5, 19, 'Mexico', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    # # Yucatan/Quintana Roo
    ax.text(-89.9, 21, 'Yucatan/Quintana Roo', fontsize=13, transform=ccrs.PlateCarree(), zorder=3000)


    # Add loop current contour from WHO Group
    fname = '/Users/mikesmith/Downloads/2023-11-11_fronts.mat'
    data = loadmat(fname)
 
    fronts = []
    for item in data['BZ_all'][0]:
        loop_y = item['y'].T
        loop_x = item['x'].T

        hf = ax.plot(loop_x, loop_y,
                     linestyle=item['LineStyle'][0],
                     color='black',
                     linewidth=3, 
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
            ax.arrow(
                start_lon[count][0],
                start_lat[count][0],
                end_lon[count][0]-start_lon[count][0],
                end_lat[count][0]-start_lat[count][0],
                linewidth=0, 
                head_width=0.1,
                shape='full', 
                fc='black', 
                ec='black',
                transform=ccrs.PlateCarree(),
                zorder=130,
                )
    fronts.reverse()
    url = 'https://gis.charttools.noaa.gov/arcgis/rest/services/MarineChart_Services/NOAACharts/MapServer/WMTS'
    w = ax.add_wmts(wmts=url, layer_name='MarineChart_Services_NOAACharts')
    # ax.legend([a1, l1[0], a6], ['Fixed', 'Fixed', 'Expected'], loc='upper left', title='Glider Tracks')
    legend_h = []
    import matplotlib.lines as mlines
    # legend_h.append(mlines.Line2D([], [], linestyle='-', color='red', linewidth=6))
    legend_h.append(mlines.Line2D([], [], linestyle='-', color='maroon', linewidth=6))
    legend_h.append(mlines.Line2D([], [], linestyle='-', color='limegreen', linewidth=6))
    legend_h.append(mlines.Line2D([], [], linestyle='-', color='magenta', linewidth=6))
    legend_h.append(d1[0])

    ax.legend(legend_h, ['HFR/ROCIS Area', 'Yucatan Inflow', 'Yucatan Outflow', "Deployment Location"], loc='lower right', title='Glider Tracks', title_fontproperties={'weight':'bold'}).set_zorder(10000)
    # plt.figtext(0.25, 0.01, 'Fixed tracks will be followed as shown.\nExpected tracks will approximately follow tracks as shown but may be affected by currents in unpredictable way.', ha="left", fontsize=9, fontstyle='italic')

    fig.savefig(save_file, dpi=150, bbox_inches='tight', pad_inches=0.1)

    plt.close()


def plot_grase(
    region,
    transform=dict(
        map=proj['map'],
        data=proj['data']
        ),
    legend=True,
    path_save=Path(os.getcwd()),
    dpi=150,
    t0=None,
    currents=None
    ):
    """

    :param lon: longitude
    :param lat: latitude
    :param variable: data variable you want to plot
    :param kwargs:
    :return:
    """
    # region_name = region["name"]
    # extent = region["extent"]
    region_name = 'yucatan'
    # extent = [-88, -84, 18, 22]
    extent = [-90, -78, 18, 28]

    # Create subdirectory for region
    region_file_str = '_'.join(region_name.lower().split(' '))
    path_save_region = path_save / 'regions' / region_file_str
    
    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save_region 
    os.makedirs(save_dir_final, exist_ok=True)

    # Create a string for the title of the plot
 
    # Create a file name to save the plot as
    save_file = save_dir_final / f"passengers.png"

    fig, ax = create(extent, proj=ccrs.PlateCarree(), bathymetry=False, figsize=(16,9))
                               
    url = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_overview/MapServer/WMSServer?request=GetCapabilities&service=WMS'
    url2 = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_general/MapServer/WMSServer?request=GetCapabilities&service=WMS'
    # url = 'https://encdirect.noaa.gov/arcgis/services/encdirect/enc_coastal/MapServer/WMSServer?request=GetCapabilities&service=WMS'

    layers2 = [
        '1',
        '2',
        '3',
        '4',
        '5',
        #  '6',
        '7',
        '8',
        # '9',
        '10',
        '11',
        '12',
        '13',
        # '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        # '20',
        '21',
        # '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '31',
        # '32',
        '33',
        '34',
        # '35',
        '36',
        '37',
        # '38',
        '39',
        # '40',
        '41',
        '42',
        '43',
        # '44',
        '45',
        '46',
        '47',
        '48',
        # '49',
        '50',
        '51',
        '52',
        # '53',
        '54',
        # '55',
        '56',
        # '57',
        '58',
        '59',
        # '60',
        '61',
        # '62',
        # '63',
        '64',
        # '65',
        '66',
        '67',
        # '68',
        '69',
        # '70',
        '71',
        '72',
        '73',
        # '74',
        '75',
        # '76',
        '77',
        '78',
        # '79',
        '80',
        '81',
        # '82',
        '83',
        '84',
        '85',
        '86',
        # '87',
        '88',
        # '89',
        '90',
        '91',
        '92',
        '93',
        '94',
        # '95',
        '96',
        '97',
        '98',
        '99',
        '100',
        # '101',
        '102',
        '103',
        '104',
        '105',
        '106',
        '107',
        '108',
        # '109',
        '110',
        # '111',
        '112',
        '113',
        '114',
        '115',
        '116',
        '117',
        '118',
        '119',
        '120',
        '121',
        '122',
        '123',
        '124']

    layers = [
        '1',
        '2',
        '3',
        '4',
        '5',
        # '6',
        '7',
        '8',
        # '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        # '16',
        '17',
        # '18',
        '19',
        # '20',
        '21',
        # '22',
        '23',
        '24',
        # '25',
        '26',
        # '27',
        '28',
        '29',
        # '30',
        '31',
        # '32',
        '33',
        # '34',
        '35',
        '36',
        # '37',
        '38',
        '39',
        # '40',
        '41',
        '42',
        # '43',
        '44',
        # '45',
        '46',
        # '47',
        '48',
        '49',
        # '50',
        '51',
        '52',
        # '53',
        '54',
        # '55',
        '56',
        # '57',
        '58',
        '59',
        # '60',
        '61',
        '62',
        '63',
        # '64',
        '65',
        # '66',
        '67',
        '68',
        '69',
        '70',
        '71',
        # '72',
        '73',
        '74',
        '75',
        # '76',
        '77',
        # '78',
        '79',
        '80',
        '81',
        '82',
        '83',
        '84',
        # '85',
        '86',
        '87',
        '88',
        '89',
        '90',
        '91',
        '92',
        '93',
        '94',
        '95',
        '96',]


    
    # ax.add_wms(wms=url2, layers=layers2)
    
    # map_add_eez(ax, color='red')

        # # Create the colorbar
        # axins = inset_axes(ax,  # here using axis of the lowest plot
        #     width="2.5%",  # width = 5% of parent_bbox width
        #     height="100%",  # height : 340% good for a (4x4) Grid
        #     loc='lower left',
        #     bbox_to_anchor=(1.05, 0., 1, 1),
        #     bbox_transform=ax.transAxes,
        #     borderpad=0
        #     )
        # cb = plt.colorbar(h, cax=axins)
        # cb.ax.tick_params(labelsize=12)
        # cb.set_label(f'{da.name.title()} ({da.units})', fontsize=13)

    # ax.set_title(title, fontsize=16, fontweight='bold')

    # if legend:
    #     h, l = ax.get_legend_handles_labels()  # get labels and handles from ax1

    #     if (len(h) > 0) & (len(l) > 0):
    #         # Shrink current axis's height by 10% on the bottom
    #         box = ax.get_position()
    #         ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                         box.width, box.height * 0.9])

    #         # Put a legend below current axis
    #         ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #                 fancybox=True, shadow=True, ncol=5)
    #         legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
    #         plt.figtext(0.5, -0.07, legstr, ha="center", fontsize=10, fontweight='bold')

    # fig.savefig(save_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    
    # # Add currents
    # coarsen = currents['coarsen']
    # map_add_currents(ax, tds, coarsen=coarsen['rtofs'], **currents['kwargs'])

    # # Initialize qargs dictionary for input into contour plot of magnitude
    # qargs = {}
    # qargs['transform'] = transform['data']
    # qargs['cmap'] = cmocean.cm.speed
    # qargs['extend'] = "max"
    
    # _, mag_r = uv2spdir(tds['u'], tds['v'])

    # m1 = ax.contourf(tds["lon"], tds["lat"], mag_r, **qargs)
    # cb = fig.colorbar(m1, ax=ax, orientation="vertical", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
    # # cb = add_colorbar(axs[:2], m1, location="bottom")
    # cb.ax.tick_params(labelsize=12)
    # cb.set_label(f'Magnitude (m/s)', fontsize=12, fontweight="bold")

    # field_lon = [-86.6021, -85.5144,-84.97, -84.9102, -85.1849, -85.5144, -86.6076]
    # field_lat = [21.2004, 22.5968, 21.8544, 21.0056, 20.1880, 19.7544, 21.1901]
    # ax.fill(field_lon, field_lat, color='coral', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)

    style = "Simple, tail_width=0.5, head_width=8, head_length=8"
    style1 = "Simple, tail_width=0.5, head_width=8, head_length=8"

    import matplotlib.patches as patches

    # # Curvy area
    # kw = dict(arrowstyle=style, color="magenta", linewidth=4, transform=ccrs.PlateCarree(), zorder=3000, label='Expected')
    # a4 = patches.FancyArrowPatch((-86.5, 20.5), (-86, 21.25), **kw)
    # a5 = patches.FancyArrowPatch((-85.75, 21.5), (-88, 25), connectionstyle="angle3, angleA=70, angleB=15", **kw)
    # a6 = patches.FancyArrowPatch((-85.75, 21.5), (-84, 23.5), connectionstyle="angle3, angleA=75, angleB=-20", **kw)
    # a7 = patches.FancyArrowPatch((-85.75, 21.5), (-84, 24.5), connectionstyle="angle3, angleA=75, angleB=-40", **kw)
    # a8 = patches.FancyArrowPatch((-84, 24), (-82.25, 24), **kw)


    # plt.gca().add_patch(a4)
    # plt.gca().add_patch(a5)
    # plt.gca().add_patch(a6)
    # plt.gca().add_patch(a7)
    # plt.gca().add_patch(a8)

    # Mexico coverage area.
    # Selected points: [(-87.03751803751804, 20.53968253968254), (-86.61904761904762, 21.181818181818183), (-88.17748917748918, 24.22655122655123), (-86.85714285714286, 25.366522366522368), (-84.72871572871573, 22.87734487734488)]
    # field_lon = [-87.03751803751804, -86.61904761904762, -88.17748917748918, -86.85714285714286, -84.72871572871573,-86.43867243867244, -87.37662337662337, -87.39105339105339, -87.05916305916305]
    # field_lat = [20.53968253968254, 21.181818181818183, 24.22655122655123, 25.366522366522368, 22.87734487734488, 18.952380952380953, 18.959595959595962, 19.998556998557, 20.503607503607505]

    # # Selected points: [(-87.40548340548341, 25.84992784992785), (-88.97835497835497, 25.546897546897547), (-86.64069264069263, 21.181818181818183), (-87.3982683982684, 19.911976911976915), (-87.39105339105339, 18.873015873015873), (-86.32323232323232, 18.865800865800868), (-85.03896103896103, 23.18759018759019)]
    # field_lon = [-87.41, -88.98, -86.64, -87.40, -87.39, -86.32, -85.50, -85.50, -85.04, -87.41, -87.41]
    # field_lat = [25.85, 25.55, 21.18, 20, 18.87, 18.87, 18.87, 21.5, 23.19, 25.85, 25.85]

    # Selected points: [(-87.39105339105339, 18.851370851370852), (-85.25541125541125, 18.865800865800868), (-85.35642135642135, 21.924963924963926), (-85.03896103896103, 23.151515151515152), (-87.40548340548341, 25.80663780663781), (-88.96392496392497, 25.51803751803752), (-86.66955266955267, 21.145743145743147), (-87.39105339105339, 19.962481962481963), (-87.40548340548341, 18.851370851370852)]
    field_lon = [-87.39, -85.36, -85.24, -84.74, -84, -85.35, -85.04, -87.40, -88.96, -86.66, -87.39, -87.40, -87.39]
    field_lat = [18.85, 18.87, 19.18, 19.57, 20, 21.92, 23.15, 25.80, 25.52, 21.14, 19.96, 18.85, 18.85]
    ax.plot(field_lon, field_lat, color='purple', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # # Area 1  (Mexico deployment 1)
    field_lon = [-86.44, -86.44, -87.3, -87.3, -86.44]
    field_lat = [19, 20, 20, 19, 19]
    ax.fill(field_lon, field_lat, color='purple', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    l1 = ax.plot(field_lon, field_lat, color='purple',  linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree(), label='Fixed')
    
    # # Area 2 (Mexico deployment 2)
    field_lon = [-86.75, -86, -85.55, -86.25, -86.75]
    field_lat = [20.75, 20.75, 21.5, 21.5, 20.75]
    ax.fill(field_lon, field_lat, color='purple', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    ax.plot(field_lon, field_lat, color='purple',  linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # Selected points: [(-86.95815295815295, 19.558441558441558), (-86.32323232323232, 19.8975468975469), (-86.66955266955267, 20.53968253968254), (-86.09235209235209, 20.734487734487736), (-86.4098124098124, 21.152958152958156), (-85.91919191919192, 21.34054834054834), (-86.87878787878788, 21.7950937950938), (-86.33766233766234, 22.581529581529583), (-87.34776334776335, 22.776334776334778), (-86.83549783549783, 23.5988455988456), (-87.93939393939394, 23.83694083694084), (-87.57142857142857, 24.883116883116884), (-89.78643578643579, 25.00577200577201)]
    track_lons = [-86.96, -86.32, -86.67, -86.09, -86.41, -85.92, -85.46, -85.92, -86.88, -86.34, -87.35, -86.84, -87.94, -87.57, -89.79]
    track_lats = [19.56, 19.90, 20.54, 20.73, 21.15, 21.34, 21.37, 21.34, 21.80, 22.58, 22.78, 23.60, 23.84, 24.88, 25.01]
    ax.plot(track_lons, track_lats, color='orange', linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    kw = dict(arrowstyle=style, color="orange", linewidth=4, transform=ccrs.PlateCarree(), zorder=3000)
    a4 = patches.FancyArrowPatch((-87.57, 24.88), (-89.79, 25.01), **kw)
    plt.gca().add_patch(a4)
    
    # Selected points: [(-85.29, 21.53), (-86.30, 19.75)] #arrow back to start
    # -85.457431, 21.376623
    a = (-85.46, 21.37)
    b = (-86.30, 19.75)
    a9 = patches.FancyArrowPatch(a, b, connectionstyle="angle3, angleA=70, angleB=15", **kw)
    # a9 = patches.FancyArrowPatch(a, b, connectionstyle="angle3, angleA=75, angleB=-20", **kw)
    # tail_width=0.5, head_width=8, head_length=8
    # a9 = patches.FancyArrowPatch(a, b, connectionstyle="angle3, angleA=75, angleB=-40", **kw)
    kw = dict(arrowstyle="->", color="orange", lw=4, transform=ccrs.PlateCarree(), zorder=3000, mutation_scale=20)
    a9 = patches.FancyArrowPatch(a, b, connectionstyle="arc3, rad=-0.5", **kw)
    
    # )
    plt.gca().add_patch(a9)

    # Area hfr
    # field_lon = [-86.75, -86, -85.5, -86.25, -86.75]
    # field_lat = [20.75, 20.75, 21.5, 21.5, 20.75]
    # ax.fill(field_lon, field_lat, color='red', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    # ax.plot(field_lon, field_lat, color='red', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # US Deployment area
    field_lon = [-84.5, -84.5, -83.50, -83.50, -84.5] # -83.08369408369408, 
    field_lat = [27, 25.96, 25.96, 27, 27] # 25.958152958152958
    ax.fill(field_lon, field_lat, color='blue', edgecolor='black', transform=ccrs.PlateCarree(), alpha=0.6, zorder=3000)
    ax.plot(field_lon, field_lat, color='blue', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # US Coverage area
    field_lon = [-82.87, -82.89, -87.49, -88.10, -84.92, -83.5, -82.87]
    field_lat = [24.96, 27, 27.93, 26.81, 23.24, 23.5, 24.96]
    ax.plot(field_lon, field_lat, color='blue', linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # # Selected points: [(-84.08658008658008, 26.42712842712843), (-85.65945165945166, 25.424242424242426), (-84.16594516594516, 25.034632034632036), (-85.22655122655122, 24.183261183261184), (-83.98556998556998, 24.183261183261184), (-84.05050505050505, 23.468975468975472), (-83.3001443001443, 24.204906204906205), (-84.07936507936508, 26.40548340548341)]
    # track_lons = [-84.09, -85.66, -84.17, -85.23, -83.99, -84.05, -83.30, -84.08, -84.09]
    # track_lats = [26.43, 25.42, 25.03, 24.18, 24.18, 23.47, 24.20, 26.41, 26.43]
    # ax.plot(track_lons, track_lats, color='orange', linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    # a5 = patches.FancyArrowPatch((-83.68, 25.27), (-84.08, 26.41),  **kw)
    # plt.gca().add_patch(a5)
    track_lons = [-84.08, -85.62, -84.08, -85.22, -83.30, -84.06, -84.07, -84.08]
    track_lats = [26.40, 25.41, 25.08, 24.16, 24.19, 23.49, 25.06, 26.40]
    ax.plot(track_lons, track_lats, color='orange', linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    a5 = patches.FancyArrowPatch((-84.07, 25.06), (-84.08, 26.40),  **kw)
    plt.gca().add_patch(a5)

    # Selected points: [(-84.20202020202021, 26.513708513708515), (-86.82828282828282, 27.494949494949495), (-86.65512265512265, 26.167388167388168), (-85.77489177489177, 26.58585858585859), (-85.7027417027417, 25.575757575757578), (-84.17316017316017, 26.470418470418473)]
    track_lons = [-84.20, -86.83, -86.66, -85.77, -85.70, -84.17, -84.20]
    track_lats = [26.51, 27.49, 26.17, 26.59, 25.58, 26.47, 26.51]
    ax.plot(track_lons, track_lats, color='orange', linestyle='-', linewidth=4, zorder=3000, transform=ccrs.PlateCarree())

    a8 = patches.FancyArrowPatch((-84.20, 26.51), (-86.83, 27.49),  **kw)
    plt.gca().add_patch(a8)
    
    # # Florida
    ax.text(-82, 27.5, 'Florida', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    # # Cuba
    ax.text(-81, 22.4, 'Cuba', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    # # Mexico
    ax.text(-89.5, 19, 'Mexico', fontsize=14, transform=ccrs.PlateCarree(), zorder=3000)
    
    ax.text(-89.9, 21, 'Yucatan/Quintana Roo', fontsize=13, transform=ccrs.PlateCarree(), zorder=3000)


    # Add loop current contour from WHO Group
    fname = '/Users/mikesmith/Downloads/fronts/2023-06-12_fronts.mat'
    data = loadmat(fname)
 
    fronts = []
    for item in data['BZ_all'][0]:
        loop_y = item['y'].T
        loop_x = item['x'].T

        hf = ax.plot(loop_x, loop_y,
                    #  linestyle=item['LineStyle'][0],
                    linestyle = '--',
                     color='black',
                     linewidth=3, 
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
            ax.arrow(
                start_lon[count][0],
                start_lat[count][0],
                end_lon[count][0]-start_lon[count][0],
                end_lat[count][0]-start_lat[count][0],
                linewidth=0, 
                head_width=0.1,
                shape='full', 
                fc='black', 
                ec='black',
                transform=ccrs.PlateCarree(),
                zorder=130,
                )
    fronts.reverse()
    url = 'https://gis.charttools.noaa.gov/arcgis/rest/services/MarineChart_Services/NOAACharts/MapServer/WMTS'
    w = ax.add_wmts(wmts=url, layer_name='MarineChart_Services_NOAACharts')
    # ax.legend([a1, l1[0], a6], ['Fixed', 'Fixed', 'Expected'], loc='upper left', title='Glider Tracks')
    legend_h = []
    import matplotlib.lines as mlines
    # legend_h.append(mlines.Line2D([], [], linestyle='-', color='red', linewidth=6))
    # legend_h.append(mlines.Line2D([], [], linestyle='-', color='maroon', linewidth=6))
    # legend_h.append(mlines.Line2D([], [], linestyle='-', color='limegreen', linewidth=6))
    # legend_h.append(mlines.Line2D([], [], linestyle='-', color='magenta', linewidth=6))

    # ax.legend(legend_h, ['HFR/ROCIS Area', 'Yucatan Inflow', 'Yucatan Outflow', "Deployment Location"], loc='lower right', title='Glider Tracks', title_fontproperties={'weight':'bold'}).set_zorder(10000)
    # plt.figtext(0.25, 0.01, 'Fixed tracks will be followed as shown.\nExpected tracks will approximately follow tracks as shown but may be affected by currents in unpredictable way.', ha="left", fontsize=9, fontstyle='italic')

    # Use ginput to select points (2 in this case)
    # points = plt.ginput(n=20, timeout=0)  # You can change n to the number of points you want to select

    # Print the selected points
    # print("Selected points:", points)
    
    fig.savefig(save_file, dpi=150, bbox_inches='tight', pad_inches=0.1)

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
                                       cols=6,
                                       transform=dict(map=proj['map'], 
                                                      data=proj['data']
                                                      ),
                                       path_save=os.getcwd(),
                                       figsize=(14,8),
                                       dpi=150,
                                       colorbar=True,
                                       overwrite=False,
                                       legend=True,
                                       hurricanes=False,
                                       ):

    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.data)
    time1 = time
    time2 = pd.to_datetime(ds2.time.data)
    extent = region['extent']

    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Create subdirectory for region
    # region_file_str = ('_').join(region_name.lower().split(' '))
    # path_save_region = path_save / region['folder']
    
    # # Create a map figure and serialize it if one doesn't already exist
    # region_name = "_".join(region["name"].split(' ')).lower()
    # mdir = path_save / "mapfigs"
    # os.makedirs(mdir, exist_ok=True)
    # sfig = mdir / f"{region_name}_fig.pkl"

    # if not sfig.exists():
    # Create figure

    grid = """
    RG
    LL
    """

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

    # Set map extent
    ax1.set_extent(extent)
    ax2.set_extent(extent)
          
    # Make the map pretty
    add_features(ax1)# zorder=0)
    add_features(ax2)# zorder=0)

    # Add bathymetry lines
    if bathy:
        try:
            add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
            add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
        except ValueError:
            print("Bathymetry deeper than specified levels.")

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)


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
    ax1.set_title(f"{ds1.model} - {time1}", fontsize=16, fontweight="bold")
    ax2.set_title(f"{ds2.model} - {time2}", fontsize=16, fontweight="bold")
    txt = plt.suptitle("", fontsize=22, fontweight="bold")

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
        legend = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        legend._legend_box.sep = 1
        # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    # if hurricanes:
    #     if type(hurricanes) == bool:
    #         # Realtime hurricane plotting
    #         plot_hurricane_track(ax1, time, basin, storm_id="AL142024")
    #         plot_hurricane_track(ax2, time, basin, storm_id="AL142024")
    #     else:
    #         # Single hurricane plotting
    #         plot_hurricane_track(ax1, time, basin, storm_id=hurricanes)
    #         plot_hurricane_track(ax2, time, basin, storm_id=hurricanes)
    # plot_hurricane_track(ax1, time, basin, storm_id="AL092024")
    # plot_hurricane_track(ax2, time, basin, storm_id="AL092024")
    # plot_hurricane_track(ax1, time, basin, storm_id="AL142024", linecolor='lime')
    # plot_hurricane_track(ax2, time, basin, storm_id="AL142024", linecolor='lime')

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for k, v in region["variables"].items():
        # Create subdirectory for variable under region directory
        var_str = ' '.join(k.split('_')).title()

        for item in v:
            print(f"Plotting {k} @ {item['depth']}")
            depth = item['depth']
            rsub = ds1[k].sel(depth=depth)
            gsub = ds2[k].sel(depth=depth, method='nearest')
            
            # Create subdirectory for depth under variable subdirectory
            save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
            os.makedirs(save_dir_final, exist_ok=True)

            # Create a file name to save the plot as
            # sname = f'{ds1.model}_vs_{ds2.model}_{k}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            sname = f'{"-".join(region["folder"].split("_"))}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}-{depth}m_{ds1.model.lower()}-vs-{ds2.model.lower()}'
            save_file = save_dir_final / f"{sname}.png"

            if save_file.is_file():
                if not overwrite:
                    print(f"{save_file} exists. Overwrite: False. Skipping.")
                    continue
                else:
                    print(f"{save_file} exists. Overwrite: True. Replotting.")
                    
            # Add the super title (title for both subplots)
            txt.set_text(f"{var_str} ({depth} m)\n")

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

            # Add contour lines for 15°C isotherm in MAB. This identifies the north wall of the Gulf Stream
            if region['name'] == 'Mid Atlantic Bight' and k == 'temperature' and depth == 200:
                ax1.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)
                ax2.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)

            if colorbar:
                cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(f'{k.title()} ({rsub.units})', fontsize=12, fontweight="bold")

            # Add EEZ
            if eez:
                eez1 = map_add_eez(ax1, zorder=1)
                eez2 = map_add_eez(ax2, zorder=1)

            # points = [(-75.22321918204588, 35.06592247775561), (-74.45437209443314, 35.612148199326924), (-74.35516601861214, 35.81056035096893), (-74.13195234801489, 35.95936946470042), (-73.83433412055189, 36.083377059476675), (-73.56151741204414, 36.157781616342426), (-72.7926703244314, 36.75301807126843), (-72.71826576756564, 37.025834779776176), (-72.84227336234188, 37.29865148828392), (-72.5942581727894, 37.472262120970676), (-72.32144146428163, 37.64587275365743), (-72.17263235055013, 37.819483386344174), (-71.8998156420424, 38.04269705694143), (-71.47818981980313, 38.16670465171768), (-71.18057159234013, 38.19150617067293), (-70.9077548838324, 38.19150617067293), (-70.53573209950363, 38.09230009485193), (-70.16370931517488, 37.96829250007568), (-69.84128956875665, 37.844284905299425), (-69.41966374651739, 37.79468186738893), (-69.02283944323338, 37.79468186738893), (-68.72522121577039, 37.869086424254675), (-68.27879387457588, 37.918689462165176), (-67.78276349547087, 37.893887943209926), (-67.26193159741064, 37.918689462165176), (-66.81550425621612, 37.96829250007568), (-66.29467235815588, 38.09230009485193), (-65.94745109278239, 38.19150617067293), (-65.69943590322987, 38.38991832231493), (-65.57542830845362, 38.53872743604643), (-65.27781008099063, 38.61313199291218), (-65.10419944830386, 38.73713958768843), (-64.98019185352763, 38.811544144554176), (-64.73217666397514, 38.93555173933043), (-63.83932198158612, 40.27483376291393), (-63.41769615934688, 40.250032243958685), (-62.97126881815237, 40.00201705440618), (-62.50003995800262, 39.62999427007743), (-62.10321565471863, 39.233169966793426), (-61.68158983247937, 38.93555173933043), (-61.35917008606111, 38.761941106643675), (-61.03675033964288, 38.712338068733175), (-60.78873515009036, 38.73713958768843), (-60.639926036358865, 38.78674262559893), (-60.59032299844838, 39.00995629619618), (-60.639926036358865, 39.08436085306193), (-61.061551858598115, 39.33237604261443), (-61.45837616188213, 39.48118515634593), (-61.904803503076614, 39.50598667530118), (-62.351230844271115, 39.70439882694318), (-62.54964299591313, 39.87800945962993), (-62.67365059068936, 40.02681857336143), (-62.822459704420865, 40.250032243958685), (-62.82245970442088, 40.32443680082443), (-62.92166578024187, 40.49804743351118), (-62.822459704420865, 40.74606262306368), (-62.62404755277888, 41.01887933157143), (-62.326429325315864, 41.24209300216869), (-62.028811097852866, 41.31649755903443), (-61.63198679456888, 41.465306672765934), (-61.33436856710586, 41.465306672765934), (-61.086353377553365, 41.26689452112393), (-60.813536669045625, 41.01887933157143), (-60.44151388471686, 40.57245199037693), (-60.06949110038811, 40.10122313022718), (-59.72226983501463, 39.75400186485368), (-59.15183489904386, 39.62999427007743), (-58.755010595759856, 39.53078819425643), (-58.234178697699626, 39.60519275112218), (-58.23417869769961, 39.60519275112218), (-57.71334679963936, 39.60519275112218), (-57.34132401531062, 39.60519275112218), (-57.018904268892356, 39.729200345898434), (-56.994102749937106, 40.10122313022718), (-57.06850730680287, 40.34923831977968), (-57.49013312904211, 40.52284895246643), (-57.93656047023661, 40.721261104108436), (-58.085369583968124, 40.89487173679518), (-58.283781735610106, 41.21729148321343), (-58.35818629247586, 41.490108191721184), (-58.35818629247587, 41.63891730545268), (-58.333384773520606, 41.86213097604993), (-58.13497262187861, 41.961337051870935), (-57.83735439441562, 41.961337051870935), (-57.46533161008686, 42.010940089781435), (-57.06850730680286, 41.83732945709468), (-56.795690598295124, 41.614115786497436), (-56.448469332921604, 41.39090211590018), (-56.101248067548106, 41.192489964258186), (-55.828431359040366, 40.99407781261618)]
            # lons = [point[0] for point in points]
            # lats = [point[1] for point in points]
            
            # ax1.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)
            # ax2.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)
    
            fig.savefig(save_dir_final / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

            # Add currents as overlays over variable plots
            # if currents['bool']:
            #     quiver_dir = save_dir_final / "currents_overlay"
            #     os.makedirs(quiver_dir, exist_ok=True)
                
            #     coarsen = currents['coarsen']
            #     q1 = map_add_currents(ax1, rsub, coarsen=coarsen['rtofs'], **currents['kwargs'])
            #     q2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents['kwargs'])

            #     if eez:
            #         eez1._kwargs['edgecolor']= 'white'                
            #         eez2._kwargs['edgecolor']= 'white'
                
            #     fig.savefig(quiver_dir / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            #     # export_fig(quiver_dir, f"{sname}.png", dpi=dpi)

            #     # Remove quiver handles from each axes
            #     q1.lines.remove(), q2.lines.remove()
            #     remove_quiver_handles(ax1), remove_quiver_handles(ax2)

            # Remove handles so we can reuse figure
            # Delete contour handles 
            [x.remove() for x in h1.collections] # axes 1
            [x.remove() for x in h2.collections] # axes 2

            if colorbar: 
                cb.remove()
                
            if eez:
                eez1.remove()
                eez2.remove()            

    plt.close()

def plot_model_region_comparison_idalia(ds1, ds2, region,
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
                                       overwrite=False,
                                       legend=True,
                                       ):

    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']

    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Create subdirectory for region
    # region_file_str = ('_').join(region_name.lower().split(' '))
    # path_save_region = path_save / region['folder']
    
    # # Create a map figure and serialize it if one doesn't already exist
    # region_name = "_".join(region["name"].split(' ')).lower()
    # mdir = path_save / "mapfigs"
    # os.makedirs(mdir, exist_ok=True)
    # sfig = mdir / f"{region_name}_fig.pkl"

    # if not sfig.exists():
    # Create figure

    grid = """
    RG
    LL
    """

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

    # Set map extent
    ax1.set_extent(extent)
    ax2.set_extent(extent)
          
    # Make the map pretty
    add_features(ax1)# zorder=0)
    add_features(ax2)# zorder=0)

    # Add bathymetry lines
    if bathy:
        try:
            add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
            add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
        except ValueError:
            print("Bathymetry deeper than specified levels.")

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)


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
    ax1.set_title(ds1.model, fontsize=16, fontweight="bold")
    ax2.set_title(ds2.model, fontsize=16, fontweight="bold")
    txt = plt.suptitle("", fontsize=22, fontweight="bold")

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
        legend = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        legend._legend_box.sep = 1
        # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    plot_hurricane_track(ax1, time, basin, storm_id="AL102023")
    plot_hurricane_track(ax2, time, basin, storm_id="AL102023")
    # AL102023 Idalia
    # AL092024 Helene
    # AL142024 Milton
    

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for k, v in region["variables"].items():
        # Create subdirectory for variable under region directory
        var_str = ' '.join(k.split('_')).title()

        for item in v:
            print(f"Plotting {k} @ {item['depth']}")
            depth = item['depth']
            rsub = ds1[k].sel(depth=depth)
            gsub = ds2[k].sel(depth=depth, method='nearest')
            
            # Create subdirectory for depth under variable subdirectory
            save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
            os.makedirs(save_dir_final, exist_ok=True)

            # Create a file name to save the plot as
            # sname = f'{ds1.model}_vs_{ds2.model}_{k}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            sname = f'{"-".join(region["folder"].split("_"))}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}-{depth}m_{ds1.model.lower()}-vs-{ds2.model.lower()}'
            save_file = save_dir_final / f"{sname}.png"

            if save_file.is_file():
                if not overwrite:
                    print(f"{save_file} exists. Overwrite: False. Skipping.")
                    continue
                else:
                    print(f"{save_file} exists. Overwrite: True. Replotting.")
                    
            # Add the super title (title for both subplots)
            txt.set_text(f"{var_str} ({depth} m) - {tstr_title}\n")

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

            # Add contour lines for 15°C isotherm in MAB. This identifies the north wall of the Gulf Stream
            if region['name'] == 'Mid Atlantic Bight' and k == 'temperature' and depth == 200:
                ax1.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)
                ax2.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)

            if colorbar:
                cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(f'{k.title()} ({rsub.units})', fontsize=12, fontweight="bold")

            # Add EEZ
            if eez:
                eez1 = map_add_eez(ax1, zorder=1)
                eez2 = map_add_eez(ax2, zorder=1)

            # Save the figure. Using fig to savefig allows us to delete any
            # figure handles so that we can reuse the figure.
            # export_fig(save_dir_final, sname, dpi=dpi)
            fig.savefig(save_dir_final / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

            # Add currents as overlays over variable plots
            # if currents['bool']:
            #     quiver_dir = save_dir_final / "currents_overlay"
            #     os.makedirs(quiver_dir, exist_ok=True)
                
            #     coarsen = currents['coarsen']
            #     q1 = map_add_currents(ax1, rsub, coarsen=coarsen['rtofs'], **currents['kwargs'])
            #     q2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents['kwargs'])

            #     if eez:
            #         eez1._kwargs['edgecolor']= 'white'                
            #         eez2._kwargs['edgecolor']= 'white'
                
            #     fig.savefig(quiver_dir / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            #     # export_fig(quiver_dir, f"{sname}.png", dpi=dpi)

            #     # Remove quiver handles from each axes
            #     q1.lines.remove(), q2.lines.remove()
            #     remove_quiver_handles(ax1), remove_quiver_handles(ax2)

            # Remove handles so we can reuse figure
            # Delete contour handles 
            [x.remove() for x in h1.collections] # axes 1
            [x.remove() for x in h2.collections] # axes 2

            if colorbar: 
                cb.remove()
                
            if eez:
                eez1.remove()
                eez2.remove()            

    plt.close()
    
def plot_hurricane_track(ax, center_time, basin, hurricane_name=None, storm_id=None, year=None, linecolor='red', markersize=None, plot_datetime=False):


    # Fetch storm data by ID or name
    storm = None
    if storm_id:
        storm = basin.get_storm(storm_id)
    elif hurricane_name and year:
        storm = basin.get_storm((hurricane_name, year))
    
    if storm is None:
        print("Storm data not found in the dataset.")
        return
    
    # Extract track data (lat, lon, wind speed, date)
    lats = storm.dict['lat']
    lons = storm.dict['lon']
    vmax = storm.dict['vmax']
    times = storm.dict['time']  # List of datetime objects
    storm_types = storm.dict['type']

    # Find the closest time point to the given center_time
    time_diffs = [abs((time - center_time).total_seconds()) for time in times]
    closest_idx = np.argmin(time_diffs)

    # Map wind speed to category size and color
    def get_category_and_size(wind_speed):
        if wind_speed < 34:
            return "TD", 50  # Tropical Depression
        elif 34 <= wind_speed < 64:
            return "TS", 75  # Tropical Storm
        elif 64 <= wind_speed < 83:
            return "C1", 100  # Category 1 Hurricane
        elif 83 <= wind_speed < 96:
            return "C2", 120  # Category 2 Hurricane
        elif 96 <= wind_speed < 113:
            return "C3", 140  # Category 3 Hurricane
        elif 113 <= wind_speed < 137:
            return "C4", 160  # Category 4 Hurricane
        else:
            return "C5", 180  # Category 5 Hurricane

    # NOAA color mapping for each storm type/category
    # type_colors = {
    #     "TD": '#A0A0A0',  # Gray
    #     "TS": '#FFFF00',  # Yellow
    #     "C1": '#FFA500',  # Light Orange
    #     "C2": '#FF4500',  # Dark Orange
    #     "C3": '#FF0000',  # Red
    #     "C4": '#8B0000',  # Dark Red
    #     "C5": '#000000'   # Purple
    # }
    # NOAA color mapping for each storm type/category
    type_colors = {
        "TD": '#808080',  # Dark Gray for Tropical Depression
        "TS": '#FFCC00',  # Yellow for Tropical Storm
        "C1": '#FF9900',  # Light Orange for Category 1
        "C2": '#FF6600',  # Dark Orange for Category 2
        "C3": '#FF0000',  # Red for Category 3
        "C4": '#990000',  # Dark Red for Category 4
        "C5": '#660066'   # Purple for Category 5
    }
        #  if tmp.type == 'DB':
        #     color = 'gold'
        #     markersize = 11
        # elif tmp.type == 'TD':
        #     color = 'yellow'
        #     markersize = 12
        # elif tmp.type == 'TS':
        #     color = 'orangered'
        #     markersize = 13
        # elif tmp.type == 'HU':
        #     color = 'red'
        #     markersize = 14
        # else:
        #     color='gold'

    # Set up Cartopy map
    # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # ax.set_extent(extent, crs=ccrs.PlateCarree())
    # ax.coastlines()
    # ax.gridlines(draw_labels=True)
    extent = ax.get_extent(crs=ccrs.PlateCarree())

    # Plot each point along the hurricane track with color for type and size for category
    for i in range(len(lats)):
        storm_type, size = get_category_and_size(vmax[i])

        if markersize:
            size = markersize
        
        color = type_colors.get(storm_type, 'gray')
        edgecolor = 'black'

        # If this is the closest point to center_time, highlight it
        if i == closest_idx:
            ax.scatter(
                lons[i], lats[i],
                color=color,
                s=size * 6,  # Make the highlighted point larger
                edgecolor='lime',
                linewidth=2,
                # label=f"{storm_type} (Closest to Center Time)",
                transform=ccrs.PlateCarree(),
                zorder=30000,
                marker=tcmarkers.tc_marker(vmax[i])
            )
            
        else:
            ax.scatter(
                lons[i], lats[i],
                color=color,
                s=size*4,
                edgecolor='black',
                # label=storm_type if i == 0 else "",  # Label each type only once for legend
                transform=ccrs.PlateCarree(), 
                zorder=30000,
                marker=tcmarkers.tc_marker(vmax[i])
            )
                    # Add time as text next to the plotted point
        # Format the time to a readable string
        # time_str = times[i].strftime("%Y-%m-%d %H:%M")
        # ax.text(
        #     lons[i], lats[i], 
        #     time_str, 
        #     fontsize=8, 
        #     color='black', 
        #     transform=ccrs.PlateCarree(),
        #     zorder=30001,
        #     ha='left',  # Align text to the left of the point
        #     va='bottom'  # Align text to the bottom of the point
        # )

    ln = ax.plot(lons, lats, color=linecolor, transform=ccrs.PlateCarree(), zorder=29999, linewidth=4
                 )  # Connect the points with a line

            
    handles = []
    # Add legend for storm categories
    for stype, color in type_colors.items():
        # Determine a representative wind speed for each storm category
        if stype == "TD":
            wind_speed = 30  # Example for Tropical Depression
        elif stype == "TS":
            wind_speed = 50  # Example for Tropical Storm
        elif stype == "C1":
            wind_speed = 75  # Example for Category 1 Hurricane
        elif stype == "C2":
            wind_speed = 90  # Example for Category 2 Hurricane
        elif stype == "C3":
            wind_speed = 105  # Example for Category 3 Hurricane
        elif stype == "C4":
            wind_speed = 125  # Example for Category 4 Hurricane
        elif stype == "C5":
            wind_speed = 145  # Example for Category 5 Hurricane

        # Use tcmarkers.tc_marker to get the appropriate marker
        marker = tcmarkers.tc_marker(wind_speed)

        # Create the legend handle
        handles.append(plt.Line2D(
            [0], [0], 
            marker=marker, 
            color='w', 
            label=stype,
            markerfacecolor=color, 
            markeredgecolor='black', 
            markersize=10, 
            linestyle='none',
        ))
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=stype, markerfacecolor=color, linestyle='none', markeredgecolor='black', markersize=10) for stype, color in type_colors.items()]
    leg = ax.legend(handles=handles, title="Storm Category", loc='lower left', fontsize=12,
                                handleheight=2,  # Increase vertical spacing between legend items
                                handletextpad=1,  # Add space between the marker and the text
    )
    leg.set_zorder(50001)

    # Title and show plot
    # plt.title(f"Hurricane {storm.name} ({storm.year}) - Category & Type\nClosest Point to {center_time}")
    # plt.show()

def plot_model_region_difference(ds1, ds2, region,
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
    
    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']

    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Create subdirectory for region
    # region_file_str = ('_').join(region_name.lower().split(' '))
    # path_save_region = path_save / region['folder']
    
    # # Create a map figure and serialize it if one doesn't already exist
    # region_name = "_".join(region["name"].split(' ')).lower()
    # mdir = path_save / "mapfigs"
    # os.makedirs(mdir, exist_ok=True)
    # sfig = mdir / f"{region_name}_fig.pkl"

    # if not sfig.exists():
    # Create figure

    grid = """
    RR
    LL
    """

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
    ax2 = axs[1] # Legend for argo/gliders

    # Set map extent
    ax1.set_extent(extent)
          
    # Make the map pretty
    add_features(ax1)# zorder=0)

    # Add bathymetry lines
    if bathy:
        try:
            add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.elevation.values,
                            levels=(-1000, -100),
                            zorder=1.5)
        except ValueError:
            print("Bathymetry deeper than specified levels.")

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)


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

    # Label the subplots
    ax1.set_title(ds1.model, fontsize=16, fontweight="bold")
    txt = plt.suptitle("", fontsize=22, fontweight="bold")
    
    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
        legend = ax2.legend(h, l, ncol=cols, loc='center', fontsize=8)

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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax2.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        legend._legend_box.sep = 1
        # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax2.set_axis_off()

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for k, v in region["variables"].items():
        # Create subdirectory for variable under region directory
        var_str = ' '.join(k.split('_')).title()

        for item in v:
            print(f"Plotting {k} @ {item['depth']}")
            depth = item['depth']
            rsub = ds1[k].sel(depth=depth)
            gsub = ds2[k].sel(depth=depth, method='nearest')

            if ds2.model == 'ROTFS':
                diff = rsub - gsub
            else:
                # Setup regridder
                grid_out = xr.Dataset({
                    'lat': rsub['lat'],
                    'lon': rsub['lon']
                    })

                regridder = xe.Regridder(gsub, grid_out,
                                         'bilinear',
                                         periodic=False,
                                         extrap_method="nearest_s2d")
                diff = rsub - regridder(gsub)
            
            # Create subdirectory for depth under variable subdirectory
            save_dir_final = path_save / f"{k}_{depth}m" / 'diff' / time.strftime('%Y/%m')
            os.makedirs(save_dir_final, exist_ok=True)

            # Create a file name to save the plot as
            # sname = f'{ds1.model}_vs_{ds2.model}_{k}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
            sname = f'{"-".join(region["folder"].split("_"))}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}-{depth}m_{ds1.model.lower()}-{ds2.model.lower()}'
            save_file = save_dir_final / f"{sname}.png"

            if save_file.is_file():
                if not overwrite:
                    print(f"{save_file} exists. Overwrite: False. Skipping.")
                    continue
                else:
                    print(f"{save_file} exists. Overwrite: True. Replotting.")
                    
            # Add the super title (title for both subplots)
            txt.set_text(f"{var_str} ({depth} m) - {tstr_title}\n")

            # Create dictionary for variable argument inputs for contourf
            vargs = {}
            vargs['transform'] = transform['data']
            vargs['transform_first'] = True
            vargs['cmap'] = cmaps(ds1[k].name)
            vargs['extend'] = "both"

            # if 'limits' in item:
            #     vargs['vmin'] = item['limits'][0]
            #     vargs['vmax'] = item['limits'][1]
            #     vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
            if k == 'temperature':
                vargs['vmin'] = -5
                vargs['vmax'] = 5
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax']+1, 1)
                vargs['cmap'] = cmocean.cm.balance
            elif k == 'salinity':
                vargs['vmin'] = -1
                vargs['vmax'] = 1
                vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax']+0.1, 0.1)
                vargs['cmap'] = cmocean.cm.balance

        
            # Filled contour for each model variable
            if (rsub['lon'].ndim == 1) & rsub['lat'].ndim == 1:
                rlons, rlats = np.meshgrid(rsub['lon'], rsub['lat'])
            else:
                rlons = rsub['lon']
                rlats = rsub['lat']
                
            h1 = ax1.contourf(rlons, rlats, diff.squeeze(), **vargs)

            # Add contour lines for 15°C isotherm in MAB. This identifies the north wall of the Gulf Stream
            if region['name'] == 'Mid Atlantic Bight' and k == 'temperature' and depth == 200:
                ax1.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)

            if colorbar:
                cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
                cb.ax.tick_params(labelsize=12)
                cb.set_label(f'{k.title()} ({rsub.units})', fontsize=12, fontweight="bold")

            # Add EEZ
            if eez:
                eez1 = map_add_eez(ax1, zorder=1)

            # Save the figure. Using fig to savefig allows us to delete any
            # figure handles so that we can reuse the figure.
            # export_fig(save_dir_final, sname, dpi=dpi)
            fig.savefig(save_dir_final / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

            # # Add currents as overlays over variable plots
            # if currents['bool']:
            #     quiver_dir = save_dir_final / "currents_overlay"
            #     os.makedirs(quiver_dir, exist_ok=True)
                
            #     coarsen = currents['coarsen']
            #     q1 = map_add_currents(ax1, rsub, coarsen=coarsen['rtofs'], **currents['kwargs'])
            #     q2 = map_add_currents(ax2, gsub, coarsen=coarsen["gofs"], **currents['kwargs'])

            #     if eez:
            #         eez1._kwargs['edgecolor']= 'white'                
            #         eez2._kwargs['edgecolor']= 'white'
                
            #     fig.savefig(quiver_dir / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            #     # export_fig(quiver_dir, f"{sname}.png", dpi=dpi)

            #     # Remove quiver handles from each axes
            #     q1.lines.remove(), q2.lines.remove()
            #     remove_quiver_handles(ax1), remove_quiver_handles(ax2)

            # Remove handles so we can reuse figure
            # Delete contour handles 
            [x.remove() for x in h1.collections] # axes 1

            if colorbar: 
                cb.remove()
                
            if eez:
                eez1.remove()

    plt.close()


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
                                                overwrite=False, 
                                                legend=True,
                                                hurricanes=False
                                                ):

    time = pd.to_datetime(ds1.time.data)
    time1 = time
    time2 = pd.to_datetime(ds2.time.data)
    extent = region['extent']
    cdict = region['currents']
    
    grid = """
    RG
    LL
    """
    # Add loop current contour from WHO Group
    # fname = '/Users/mikesmith/Downloads/GOM22 Fronts/2022-09-04_fronts.mat'
    # data = loadmat(fname)

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for depth in cdict['depths']:
        print(f"Plotting currents @ {depth}m")
        ds1_depth = ds1.sel(depth=depth)

        try:
            ds2_depth = ds2.sel(depth=depth, method='nearest')
        except:
            ds2_depth = ds2
        
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

        # Set map extents
        ax1.set_extent(extent)
        ax2.set_extent(extent)

        # Make the map pretty  
        add_features(ax1)# zorder=0)
        add_features(ax2)# zorder=0)
        if bathy:       
            try:
                add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )

                add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )
            except ValueError:
                print("Bathymetry deeper than specified levels.")
        add_ticks(ax1, extent)
        add_ticks(ax2, extent, label_left=False, label_right=True)

        # Plot gliders and argo floats
        rargs = {}
        rargs['argo'] = argo
        rargs['gliders'] = gliders
        rargs['transform'] = transform['data']  
        plot_regional_assets(ax1, **rargs)
        plot_regional_assets(ax2, **rargs)

        # Label the subplots
        ax1.set_title(f"{ds1.model.upper()} - {time1}", fontsize=16, fontweight="bold")
        ax2.set_title(f"{ds2.model.upper()} - {time2}", fontsize=16, fontweight="bold")

        # plot_hurricane_track(ax1, time, basin, storm_id="AL092024", markersize=30)
        # plot_hurricane_track(ax1, time, basin, storm_id="AL142024", linecolor='green', markersize=30)

        # plot_hurricane_track(ax2, time, basin, storm_id="AL092024", markersize=30)
        # plot_hurricane_track(ax2, time, basin, storm_id="AL142024", linecolor='green', markersize=30)
        # if hurricanes:
        #     if type(hurricanes) == bool:
        #         # Realtime hurricane plotting
        #         plot_hurricane_track(ax1, time, basin, storm_id="AL142024")
        #         plot_hurricane_track(ax2, time, basin, storm_id="AL142024")
        #     else:
        #         # Single hurricane plotting
        #         plot_hurricane_track(ax1, time, basin, storm_id=hurricanes)
        #         plot_hurricane_track(ax2, time, basin, storm_id=hurricanes)

        

        # plot_hurricane_track(ax1, time, basin, storm_id="AL092024")
        # plot_hurricane_track(ax2, time, basin, storm_id="AL092024")
        # plot_hurricane_track(ax1, time, basin, storm_id="AL142024", linecolor='lime')
        # plot_hurricane_track(ax2, time, basin, storm_id="AL142024", linecolor='lime')

        # # Add EddyWatch Fronts
        # for ax in [ax1, ax2]:
        #     fronts = []
        #     for item in data['BZ_all'][0]:
        #         loop_y = item['y'].T
        #         loop_x = item['x'].T

        #         hf = ax.plot(loop_x, loop_y,
        #                     linestyle=item['LineStyle'][0],
        #                     color='black',
        #                     linewidth=3, 
        #                     transform=ccrs.PlateCarree(), 
        #                     zorder=120
        #                     )
        #         fronts.append(hf)

        #         # Add arrows
        #         start_lon = item['bx'].T
        #         start_lat = item['by'].T
        #         end_lon = item['tx'].T
        #         end_lat = item['ty'].T

        #         for count, _ in enumerate(start_lon):
        #             ax.arrow(
        #                 start_lon[count][0],
        #                 start_lat[count][0],
        #                 end_lon[count][0]-start_lon[count][0],
        #                 end_lat[count][0]-start_lat[count][0],
        #                 linewidth=0, 
        #                 head_width=0.2,
        #                 shape='full', 
        #                 fc='black', 
        #                 ec='black',
        #                 transform=ccrs.PlateCarree(),
        #                 zorder=130,
        #                 )
        # fronts.reverse()

        # Deal with the third axes
        h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

        h_n = h #+ fronts
        l_n = l #+ ["EddyWatch - 1.5 knots (Active)", "EddyWatch - Eddy Remnants"]
        
        if (len(h) > 0) & (len(l) > 0):
            ax3.legend(h_n, l_n, ncol=cols, loc='center', fontsize=8)

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

        fig.tight_layout()
        fig.subplots_adjust(top=0.88, wspace=.001)
        
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
            eez1 = map_add_eez(ax1, zorder=1, color='red', linewidth=3, linestyle='-')
            eez2 = map_add_eez(ax2, zorder=1, color='red', linewidth=3, linestyle='-')

        if colorbar:
            cb = fig.colorbar(m1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
            # cb = add_colorbar(axs[:2], m1, location="bottom")
            cb.ax.tick_params(labelsize=12)
            cb.set_label(f'Magnitude (m/s)', fontsize=12, fontweight="bold")

        # Create a string for the title of the plot
        # title_time = time.strftime("%Y-%m-%d %H:%M:%S")
        title = f"Currents ({depth} m)\n"
        plt.suptitle(title, fontsize=22, fontweight="bold")

        # points = [(-75.22321918204588, 35.06592247775561), (-74.45437209443314, 35.612148199326924), (-74.35516601861214, 35.81056035096893), (-74.13195234801489, 35.95936946470042), (-73.83433412055189, 36.083377059476675), (-73.56151741204414, 36.157781616342426), (-72.7926703244314, 36.75301807126843), (-72.71826576756564, 37.025834779776176), (-72.84227336234188, 37.29865148828392), (-72.5942581727894, 37.472262120970676), (-72.32144146428163, 37.64587275365743), (-72.17263235055013, 37.819483386344174), (-71.8998156420424, 38.04269705694143), (-71.47818981980313, 38.16670465171768), (-71.18057159234013, 38.19150617067293), (-70.9077548838324, 38.19150617067293), (-70.53573209950363, 38.09230009485193), (-70.16370931517488, 37.96829250007568), (-69.84128956875665, 37.844284905299425), (-69.41966374651739, 37.79468186738893), (-69.02283944323338, 37.79468186738893), (-68.72522121577039, 37.869086424254675), (-68.27879387457588, 37.918689462165176), (-67.78276349547087, 37.893887943209926), (-67.26193159741064, 37.918689462165176), (-66.81550425621612, 37.96829250007568), (-66.29467235815588, 38.09230009485193), (-65.94745109278239, 38.19150617067293), (-65.69943590322987, 38.38991832231493), (-65.57542830845362, 38.53872743604643), (-65.27781008099063, 38.61313199291218), (-65.10419944830386, 38.73713958768843), (-64.98019185352763, 38.811544144554176), (-64.73217666397514, 38.93555173933043), (-63.83932198158612, 40.27483376291393), (-63.41769615934688, 40.250032243958685), (-62.97126881815237, 40.00201705440618), (-62.50003995800262, 39.62999427007743), (-62.10321565471863, 39.233169966793426), (-61.68158983247937, 38.93555173933043), (-61.35917008606111, 38.761941106643675), (-61.03675033964288, 38.712338068733175), (-60.78873515009036, 38.73713958768843), (-60.639926036358865, 38.78674262559893), (-60.59032299844838, 39.00995629619618), (-60.639926036358865, 39.08436085306193), (-61.061551858598115, 39.33237604261443), (-61.45837616188213, 39.48118515634593), (-61.904803503076614, 39.50598667530118), (-62.351230844271115, 39.70439882694318), (-62.54964299591313, 39.87800945962993), (-62.67365059068936, 40.02681857336143), (-62.822459704420865, 40.250032243958685), (-62.82245970442088, 40.32443680082443), (-62.92166578024187, 40.49804743351118), (-62.822459704420865, 40.74606262306368), (-62.62404755277888, 41.01887933157143), (-62.326429325315864, 41.24209300216869), (-62.028811097852866, 41.31649755903443), (-61.63198679456888, 41.465306672765934), (-61.33436856710586, 41.465306672765934), (-61.086353377553365, 41.26689452112393), (-60.813536669045625, 41.01887933157143), (-60.44151388471686, 40.57245199037693), (-60.06949110038811, 40.10122313022718), (-59.72226983501463, 39.75400186485368), (-59.15183489904386, 39.62999427007743), (-58.755010595759856, 39.53078819425643), (-58.234178697699626, 39.60519275112218), (-58.23417869769961, 39.60519275112218), (-57.71334679963936, 39.60519275112218), (-57.34132401531062, 39.60519275112218), (-57.018904268892356, 39.729200345898434), (-56.994102749937106, 40.10122313022718), (-57.06850730680287, 40.34923831977968), (-57.49013312904211, 40.52284895246643), (-57.93656047023661, 40.721261104108436), (-58.085369583968124, 40.89487173679518), (-58.283781735610106, 41.21729148321343), (-58.35818629247586, 41.490108191721184), (-58.35818629247587, 41.63891730545268), (-58.333384773520606, 41.86213097604993), (-58.13497262187861, 41.961337051870935), (-57.83735439441562, 41.961337051870935), (-57.46533161008686, 42.010940089781435), (-57.06850730680286, 41.83732945709468), (-56.795690598295124, 41.614115786497436), (-56.448469332921604, 41.39090211590018), (-56.101248067548106, 41.192489964258186), (-55.828431359040366, 40.99407781261618)]
        # lons = [point[0] for point in points]
        # lats = [point[1] for point in points]
        
        # ax1.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)
        # ax2.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)

        # subplot 1
        # if depth == 0:
        # ax1.contour(ds1_depth['lon'], ds1_depth['lat'], mag_r, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)
        # ax2.contour(ds2_depth['lon'], ds2_depth['lat'], mag_g, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)    

        # Save figure
        fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        if colorbar:
            cb.remove()
            
        # Delete contour handles and remove colorbar axes to use figure
        s1.lines.remove(), s2.lines.remove()
        remove_quiver_handles(ax1), remove_quiver_handles(ax2)
        [x.remove() for x in m1.collections]
        [x.remove() for x in m2.collections]

def plot_model_region_comparison_streamplot_idalia(ds1, ds2, region,
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
                                                overwrite=False, 
                                                legend=True,
                                                ):

    # basin = tracks.TrackDataset(basin='north_atlantic', include_btk=True)

    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']
    cdict = region['currents']
    
    grid = """
    RG
    """
    # Add loop current contour from WHO Group
    # fname = '/Users/mikesmith/Downloads/GOM22 Fronts/2022-09-04_fronts.mat'
    # data = loadmat(fname)

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for depth in cdict['depths']:
        print(f"Plotting currents @ {depth}m")
        ds1_depth = ds1.sel(depth=depth)

        try:
            ds2_depth = ds2.sel(depth=depth, method='nearest')
        except:
            ds2_depth = ds2
        
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
                # "height_ratios": [4, 1],
                # set the width ratios between the columns
                # # "width_ratios": [1],
                },
            )
        axs = fig.axes
        ax1 = axs[0] # rtofs
        ax2 = axs[1] # gofs
        # ax3 = axs[2] # legend for argo/gliders

        # Set map extents
        ax1.set_extent(extent)
        ax2.set_extent(extent)

        # Make the map pretty  
        add_features(ax1)# zorder=0)
        add_features(ax2)# zorder=0)
        if bathy:       
            try:
                add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )

                add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )
            except ValueError:
                print("Bathymetry deeper than specified levels.")
        add_ticks(ax1, extent)
        add_ticks(ax2, extent, label_left=False, label_right=True)

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

        plot_hurricane_track(ax1, time, basin, storm_id="AL102023")
        plot_hurricane_track(ax2, time, basin, storm_id="AL102023")
        # AL102023 Idalia
        # AL092024 Helene
        # AL142024 Milton

        # # Add EddyWatch Fronts
        # for ax in [ax1, ax2]:
        #     fronts = []
        #     for item in data['BZ_all'][0]:
        #         loop_y = item['y'].T
        #         loop_x = item['x'].T

        #         hf = ax.plot(loop_x, loop_y,
        #                     linestyle=item['LineStyle'][0],
        #                     color='black',
        #                     linewidth=3, 
        #                     transform=ccrs.PlateCarree(), 
        #                     zorder=120
        #                     )
        #         fronts.append(hf)

        #         # Add arrows
        #         start_lon = item['bx'].T
        #         start_lat = item['by'].T
        #         end_lon = item['tx'].T
        #         end_lat = item['ty'].T

        #         for count, _ in enumerate(start_lon):
        #             ax.arrow(
        #                 start_lon[count][0],
        #                 start_lat[count][0],
        #                 end_lon[count][0]-start_lon[count][0],
        #                 end_lat[count][0]-start_lat[count][0],
        #                 linewidth=0, 
        #                 head_width=0.2,
        #                 shape='full', 
        #                 fc='black', 
        #                 ec='black',
        #                 transform=ccrs.PlateCarree(),
        #                 zorder=130,
        #                 )
        # fronts.reverse()

        # Deal with the third axes
        h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

        h_n = h #+ fronts
        l_n = l #+ ["EddyWatch - 1.5 knots (Active)", "EddyWatch - Eddy Remnants"]
        
        # if (len(h) > 0) & (len(l) > 0):
        #     ax3.legend(h_n, l_n, ncol=cols, loc='center', fontsize=8)

        #     # Add title to legend
        #     t0 = []
        #     if isinstance(argo, pd.DataFrame):
        #         if not argo.empty:
        #             t0.append(argo.index.min()[1])

        #     if isinstance(gliders, pd.DataFrame):
        #         if not gliders.empty:
        #             t0.append(gliders.index.min()[1])

        #     if len(t0) > 0:
        #         t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
        #     else:
        #         t0 = None
        #     legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
        #     ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style="italic")
        #     # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
        # ax3.set_axis_off()

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
            eez1 = map_add_eez(ax1, zorder=1, color='red', linewidth=3, linestyle='-')
            eez2 = map_add_eez(ax2, zorder=1, color='red', linewidth=3, linestyle='-')

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
        # ax1.contour(ds1_depth['lon'], ds1_depth['lat'], mag_r, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)
        # ax2.contour(ds2_depth['lon'], ds2_depth['lat'], mag_g, [0.771667], colors='k', linewidths=3, transform=ccrs.PlateCarree(), zorder=100)    

        # Save figure
        fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        if colorbar:
            cb.remove()
            
        # Delete contour handles and remove colorbar axes to use figure
        s1.lines.remove(), s2.lines.remove()
        remove_quiver_handles(ax1), remove_quiver_handles(ax2)
        [x.remove() for x in m1.collections]
        [x.remove() for x in m2.collections]


def salinity_max(ds, extent, region_name,
                 limits=None,
                 bathy=None,
                 argo=None,
                 gliders=None,
                 eez=False,
                 cols=6,
                 path_save=os.getcwd(), 
                 transform=dict(map=ccrs.Mercator(), 
                                data=ccrs.PlateCarree()
                                ), 
                 figsize=(14,8),
                 dpi=150,
                 overwrite=False):
    
    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds.time.values)
    
    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Generate filename
    fname = f"{path_save.name}_{tstr_folder}_salinity_max_{ds.model.lower()}.png"

    # Append salinity_max, year, and month to path_save
    path_save = path_save / 'salinity_max' / year / month

    save_file = path_save / fname
    
    if save_file.is_file():
        if not overwrite:
            print(f"{save_file} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{save_file} exists. Overwrite: True. Replotting.")
        
    # Make sure path_save exists
    os.makedirs(path_save, exist_ok=True)
    
    print(f"Plotting Salinity Max of {region_name} at {tstr_title}")
    
    # Get the maximum salinity over the dimension 'depth'
    smax = ds['salinity'].max("depth")

    # Get the depth that the maximum salinity occurs at.
    # We find the indexes of the salinity maximum using the .argmax() method.
    # We use the .isel() method to select the depths that the salinity max occured.
    smax_depth = ds['salinity'].idxmax("depth")

    # Initialize figure    
    fig, _ = plt.subplot_mosaic(
        """
        RG
        LL
        """,
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
    ax1 = axs[0] # rtofs
    ax2 = axs[1] # gofs
    ax3 = axs[2] # legend for argo/gliders

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    
    for ax in [ax1, ax2]:
        ax.set_extent(extent)

        # Add features to both map axes. Land, water, coastlines, etc.
        add_features(ax)
        
        # Add bathymetry lines
        if bathy:
            add_bathymetry(ax,
                           bathy.longitude.values, 
                           bathy.latitude.values, 
                           bathy.elevation.values,
                           levels=(-1000, -100),
                           zorder=1.5
                           )
        # Add eez lines
        if eez:
            map_add_eez(ax, color='white', zorder=10)

        plot_regional_assets(ax, **rargs)

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
    ax3.set_axis_off()

    # Calculate contours
    if limits:
        salt_min = limits[0]
        salt_max = limits[1]
        salt_stride = limits[2]
    else:
        # Calculate the colorbar limits automatically
        percentiles = [.25, .75]
        salinity_quantile = smax.quantile(percentiles)
        salt_min = np.floor(salinity_quantile[0])
        salt_max = np.ceil(salinity_quantile[1])
        salt_stride = .1

    # Calculate salinity contours
    levels = np.arange(
        salt_min,
        salt_max+salt_stride,
        salt_stride
        )

    # Calculate depth contours
    depths = np.arange(40, 200, 20)

    # Salinity Max Plot
    h1 = ax1.contourf(smax['lon'], smax['lat'], smax,
                        levels=levels, 
                        extend="both",
                        cmap=cmocean.cm.haline,
                        transform=transform['data'])

    # Salinity Max Depth Plot
    h2 = ax2.contourf(smax_depth['lon'], smax_depth['lat'], smax_depth, 
                        levels=depths,
                        extend="both",
                        cmap=cmocean.cm.deep,
                        transform=transform['data'])
    
    # Add colorbar to first axes
    cb = add_colorbar(ax1, h1)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(f'Salinity', fontsize=11, fontweight="bold")
    
    # Add colorbar to second axes
    cb = add_colorbar(ax2, h2)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(f'Depth (m)', fontsize=11, fontweight="bold")

    # Set title for each axes
    ax1.set_title("Salinity Maximum", fontsize=16, fontweight='bold')
    ax2.set_title("Depth of Salinity Maximum", fontsize=16, fontweight='bold')
    fig.suptitle(f"{ds.model.upper()} - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for axs in ax.flat:
    #     axs.label_inner()

    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    
    export_fig(path_save, fname, dpi=dpi)
    plt.close()

def plot_storm_track(ax, storm, time, zorder=90, proj=proj['data']):
    cone = storm['cone'][time]
    storm_lon = storm['track'].lon
    storm_lat = storm['track'].lat
    
    #Plot cone
    # cone_2d = cone
    cone_2d = ndimage.gaussian_filter(cone, sigma=0.5, order=0)

    # # # Cone shading
    # ax.contourf(cone_2d['lon2d'], cone_2d['lat2d'], cone_2d, [0.9,1.1], 
    #             colors=['#ffffff', '#ffffff'],
    #             alpha=.25,
    #             zorder=zorder+1,
    #             transform=proj)

    # Cone outline
    ax.contour(cone_2d['lon2d'], cone_2d['lat2d'], cone_2d, [0.9], 
                linewidths=1.5,
                colors=['k'],
                zorder=zorder+1,
                transform=proj)

    #Plot forecast center line & account for dateline crossing
    ax.plot(cone['center_lon'], cone['center_lat'],
            # color='g-o',
            'g-o',
            linewidth=8.0,
            zorder=1600,
            transform=proj
            )

    #Plot previous track
    ax.plot(storm_lon, storm_lat, color='red',
            # linestyle='dotted',
            linewidth=8,
            # zorder=zorder+1,
            zorder=1600,
            transform=proj)

def salinity_max_comparison(ds1, ds2, extent, region_name,
                            limits=None,
                            bathy=None,
                            argo=None,
                            gliders=None,
                            eez=False,
                            cols=6,
                            path_save=os.getcwd(), 
                            transform=dict(map=ccrs.Mercator(), 
                                           data=ccrs.PlateCarree()), 
                            figsize=(14,8),
                            dpi=150,
                            overwrite=False,
                            storms=None):
    
    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.values)
    
    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Generate filename
    fname_max = f"{path_save.name}_{tstr_folder}_salinity_max_{ds1.model.lower()}-{ds2.model.lower()}.png"
    fname_depth = f"{path_save.name}_{tstr_folder}_salinity_max_{ds1.model.lower()}-{ds2.model.lower()}-depth.png"

    # Append salinity_max, year, and month to path_save
    path_save = path_save / 'salinity_max' / year / month

    save_file = path_save / fname_max
    
    if save_file.is_file():
        if not overwrite:
            print(f"{save_file} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{save_file} exists. Overwrite: True. Replotting.")
        
    # Make sure path_save exists
    os.makedirs(path_save, exist_ok=True)
    
    print(f"Plotting Salinity Max of {region_name} at {tstr_title}")
    
    # Get the maximum salinity over the dimension 'depth'
    smax1 = ds1['salinity'].max("depth")
    smax2 = ds2['salinity'].max("depth")

    # Initialize figure    
    fig, _ = plt.subplot_mosaic(
        """
        RG
        LL
        """,
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
    ax1 = axs[0] # rtofs
    ax2 = axs[1] # gofs
    ax3 = axs[2] # legend for argo/gliders

    ax1.set_extent(extent)
    ax2.set_extent(extent)

    # Argo/Glider Data Dicts
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']
    

    for ax in [ax1, ax2]:
        # Make the map pretty  
        add_features(ax)# zorder=0)
        
        # Add features to both map axes. Land, water, coastlines, etc.
        add_features(ax)

        # Add bathymetry lines
        # if bathy:
        #     add_bathymetry(ax,
        #                    bathy.longitude.values, 
        #                    bathy.latitude.values, 
        #                    bathy.elevation.values,
        #                    levels=(-1000, -100),
        #                    zorder=1.5
        #                    )
            
        # Add eez lines
        # if eez:
            # map_add_eez(ax, color='white', zorder=10, linewidth=1)

        # Plot gliders and argo floats 
        plot_regional_assets(ax, **rargs)

        # if storms:
        #     for s in storms.keys():
        #         storms['track']
        #         plot_storm_track(ax, lon, lat, cone)
                
    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
    ax3.set_axis_off()

    # Calculate contours
    if limits:
        salt_min = limits[0]
        salt_max = limits[1]
        salt_stride = limits[2]
    else:
        # Calculate the colorbar limits automatically
        percentiles = [.25, .75]
        salinity_quantile = smax1.quantile(percentiles)
        salt_min = np.floor(salinity_quantile[0])
        salt_max = np.ceil(salinity_quantile[1])
        salt_stride = .1

    # Calculate salinity contours
    levels = np.arange(
        salt_min,
        salt_max+salt_stride,
        salt_stride
        )

    # Salinity Max Plot
    h1 = ax1.contourf(smax1['lon'], smax1['lat'], smax1,
                      levels=levels, 
                      extend="both",
                      cmap=cmocean.cm.haline,
                      transform=transform['data'])

    h2 = ax2.contourf(smax2['lon'], smax2['lat'], smax2,
                      levels=levels, 
                      extend="both",
                      cmap=cmocean.cm.haline,
                      transform=transform['data'])

    # Add colorbar to first axes
    cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
    cb.ax.tick_params(labelsize=10)
    cb.set_label('Salinity', fontsize=11, fontweight="bold")
    
    # Set title for each axes
    ax1.set_title(f"{ds1.model.upper()}", fontsize=16, fontweight='bold')
    ax2.set_title(f"{ds2.model.upper()}", fontsize=16, fontweight='bold')
    fig.suptitle(f"Salinity Maximum - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)
    
    export_fig(path_save, fname_max, dpi=dpi)

    # Remove handles so we can reuse figure
    # Delete contour handles 
    [x.remove() for x in h1.collections] # axes 1
    [x.remove() for x in h2.collections] # axes 2

    cb.remove()

    # Get the depth that the maximum salinity occurs at.
    # We find the indexes of the salinity maximum using the .argmax() method.
    # We use the .isel() method to select the depths that the salinity max occured.
    smax1_depth = ds1['salinity'].idxmax("depth")
    smax2_depth = ds2['salinity'].idxmax("depth")

    # Calculate depth contours
    depths = np.arange(40, 200, 20)
    
    # Salinity Max Depth Plot
    h1 = ax1.contourf(smax1_depth['lon'], smax1_depth['lat'], smax1_depth, 
                      levels=depths,
                      extend="both",
                      cmap=cmocean.cm.deep,
                      transform=transform['data'])
    h2 = ax2.contourf(smax2_depth['lon'], smax2_depth['lat'], smax2_depth, 
                      levels=depths,
                      extend="both",
                      cmap=cmocean.cm.deep,
                      transform=transform['data'])

    # Add colorbar to first axes
    cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
    cb.ax.tick_params(labelsize=10)
    cb.set_label('Depth (m)', fontsize=11, fontweight="bold")
    
    # Set title for each axes
    ax1.set_title(f"{ds1.model.upper()}", fontsize=16, fontweight='bold')
    ax2.set_title(f"{ds2.model.upper()}", fontsize=16, fontweight='bold')
    fig.suptitle(f"Depth of Salinity Maximum - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)
    
    export_fig(path_save, fname_depth, dpi=dpi)
    
    plt.close()


def plot_ohc(ds1, ds2, extent, region_name,
             limits=None,
             bathy=None,
             argo=None,
             gliders=None,
             eez=False,
             cols=6,
             path_save=os.getcwd(), 
             transform=dict(map=ccrs.Mercator(), 
                            data=ccrs.PlateCarree()
                            ), 
             figsize=(14,8),
             dpi=150,
             overwrite=False
             ):

    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.values)
    
    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Generate filename
    fname = f"{path_save.name}_{tstr_folder}_heat_content_{ds1.model.lower()}-{ds2.model.lower()}.png"

    # Append salinity_max, year, and month to path_save
    path_save = path_save / 'ocean_heat_content' / year / month

    save_file = path_save / fname
    
    if save_file.is_file():
        if not overwrite:
            print(f"{save_file} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{save_file} exists. Overwrite: True. Replotting.")    
    
    # Make sure path_save exists
    os.makedirs(path_save, exist_ok=True)
    
    print(f"Plotting Ocean Heat Content of {region_name} at {tstr_title}")
 
    # Initialize figure    
    fig, _ = plt.subplot_mosaic(
        """
        RG
        LL
        """,
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
    ax1 = axs[0] # rtofs
    ax2 = axs[1] # gofs
    ax3 = axs[2] # legend for argo/gliders

    # Setup keyword arguments dictionary for plot_regional_assets
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    
    for ax in [ax1, ax2]:
        create(extent, ax=ax1, ticks=False)
        create(extent, ax=ax2, ticks=False)
        
        # Make the map pretty  
        # add_features(ax)# zorder=0)

        # Add features to both map axes. Land, water, coastlines, etc.
        # add_features(ax)

        # Add bathymetry lines
        if bathy:       
            add_bathymetry(ax,
                           bathy.longitude.values, 
                           bathy.latitude.values, 
                           bathy.z.values,
                           levels=(-1000, -100),
                           zorder=1.5
                           )
        # Add eez lines
        if eez:
            map_add_eez(ax, color='black', zorder=10)

        # Plot gliders and argo floats
        plot_regional_assets(ax, **rargs)

        # plot_hurricane_track(ax, time, basin, storm_id="AL092024")
        # AL102023 Idalia
        # AL092024 Helene
        # AL142024 Milton
        plot_hurricane_track(ax, time, basin, storm_id="AL092024", markersize=20)
        plot_hurricane_track(ax, time, basin, storm_id="AL142024", linecolor='green', markersize=20)

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    # Calculate contours
    if limits:
        ohc_min = limits[0]
        ohc_max = limits[1]
        ohc_stride = limits[2]
    else:
        # Calculate the colorbar limits automatically
        percentiles = [.25, .75]
        quantile = ds1['ohc'].quantile(percentiles)
        ohc_min = np.floor(quantile[0])
        ohc_max = np.ceil(quantile[1])
        ohc_stride = 10

    # Calculate salinity contours
    levels = np.arange(
        ohc_min,
        ohc_max+ohc_stride,
        ohc_stride
        )

    cmap = cmocean.cm.thermal

    # Ocean Heat Content Plot
    h1 = ax1.contourf(ds1['lon'], ds1['lat'], ds1['ohc'],
                      levels=levels, 
                      extend="max",
                      cmap=cmap,
                      transform=transform['data'])
    h3 = ax1.contour(ds1['lon'], 
                     ds1['lat'], 
                     ds1['ohc'], 
                     [60],
                     linestyles='-',
                     colors=['silver'],
                     linewidths=1,
                     alpha=1,
                     transform=ccrs.PlateCarree(),
                     zorder=101)


    # Ocean Heat Content Plot
    h2 = ax2.contourf(ds2['lon'], ds2['lat'], ds2['ohc'], 
                      levels=levels,
                      extend="max",
                      cmap=cmap,
                      transform=transform['data']
                      )
    ax2.contour(ds2['lon'], 
                ds2['lat'], 
                ds2['ohc'], 
                [60],
                linestyles='-',
                colors=['silver'],
                linewidths=1,
                alpha=1,
                transform=ccrs.PlateCarree(),
                zorder=101)
    h0 = []
    l0 = []
    
    h0.append(mlines.Line2D([], [], linestyle='-', color='silver', alpha=1, linewidth=1))
    l0.append('60 kJ cm-2')
    # h0.append(mlines.Line2D([], [], linestyle='-', color='white', alpha=1, linewidth=1))
    # l0.append('Past 5 days')
    leg1 = ax1.legend(h0, l0, loc='upper left', fontsize=9)
    leg2 = ax2.legend(h0, l0, loc='upper left', fontsize=9)

    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        # h.append(mlines.Line2D([], [], linestyle='-', color='silver', alpha=1, linewidth=1))
        # l.append('60 kJ cm-2')
        # h.append(mlines.Line2D([], [], linestyle='-', color='white', alpha=1, linewidth=1))
        # l.append('Tracks - Past 5 days')
        # h.append()
        
        # Add handles to legend
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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
    ax3.set_axis_off()
    # plt.gca().add_artist(leg1)
    # plt.gca().add_artist(leg2)

    leg1.set_zorder(10001)
    leg2.set_zorder(10001)    
    
    # Add colorbar to first axes
    cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
    cb.ax.tick_params(labelsize=12)
    cb.set_label('kJ/cm^2', fontsize=12, fontweight="bold")

    # Set title for each axes
    ax1.set_title(f"{ds1.model.upper()}", fontsize=16, fontweight='bold')
    ax2.set_title(f"{ds2.model.upper()}", fontsize=16, fontweight='bold')
    fig.suptitle(f"Ocean Heat Content - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)

    # from ioos_model_comparisons.plotting_hurricanes import plot_storms
    # from tropycal import realtime
    # import datetime as dt
    
    # realtime_obj = realtime.Realtime()

    # realtime_obj.list_active_storms(basin='north_atlantic')

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

    # plot_storms(ax, storms, forecasts, zorder=80)


    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for axs in ax.flat:
    #     axs.label_inner()

    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    
    export_fig(path_save, fname, dpi=dpi)
    plt.close()

# import pandas as pd

def plot_ohc_single(ds1, extent, region_name,
             limits=None,
             bathy=None,
             argo=None,
             gliders=None,
             eez=False,
             cols=6,
             path_save=os.getcwd(), 
             transform=dict(map=ccrs.Mercator(), 
                            data=ccrs.PlateCarree()
                            ), 
             figsize=(14,8),
             dpi=150,
             overwrite=False,
             storms=None
             ):
    from pandas import to_datetime, DataFrame
    
    # Convert ds.time value to a normal datetime
    time = to_datetime(ds1.time.values)
    
    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Generate filename
    fname = f"{path_save.name}_{tstr_folder}_heat_content_{ds1.model.lower()}.png"

    # Append salinity_max, year, and month to path_save
    path_save = path_save / 'ocean_heat_content' / year / month

    save_file = path_save / fname
    
    if save_file.is_file():
        if not overwrite:
            print(f"{save_file} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{save_file} exists. Overwrite: True. Replotting.")    
    
    # Make sure path_save exists
    os.makedirs(path_save, exist_ok=True)
    
    print(f"Plotting Ocean Heat Content of {region_name} at {tstr_title}")
 

    # Adjust the figure size to create more space at the bottom
    fig = plt.figure(figsize=figsize)  # Adjusted figure size

    # Create main plot axes
    ax1 = fig.add_subplot(1, 1, 1, projection=proj['map'])


    # Create an additional axes for the legend at the bottom
    # ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # Adjust these values as needed

    create(extent, ax=ax1, ticks=False)
        
    if bathy:       
        add_bathymetry(ax1,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.z.values,
                        levels=(-1000, -100),
                        zorder=1.5
                        )
    # Add eez lines
    # if eez:
    map_add_eez(ax1, color='black', zorder=10, linestyle='-', linewidth=2)

    levels = [-8000, -1000, -100, 0]
    colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']
    cs = ax1.contourf(bathy['longitude'], bathy['latitude'], bathy['z'], levels, colors=colors, transform=ccrs.PlateCarree(), ticks=False)

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
    proxy.reverse()
    first_legend = ax1.legend(proxy, ["0-100m", "100-1000m", "1000+m"], loc='upper right', title='Depth')
    first_legend.set_zorder(10000)
    ax1.add_artist(first_legend)

    # Setup keyword arguments dictionary for plot_regional_assets
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    
    # Plot gliders and argo floats
    plot_regional_assets(ax1, **rargs)
    import cartopy.mpl.ticker as cticker


    # Add gridlines at every 1 degree without labels
    gl = ax1.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,           # Disable automatic labels
        linestyle="--",
        color="black",
        alpha=0.5,
    )
    gl.xlocator = plt.MultipleLocator(1)  # Set 1-degree interval for gridlines
    gl.ylocator = plt.MultipleLocator(1)
    gl.set_zorder(9.99)

    # Manually add labeled ticks at every 5 degrees on both axes
    # Longitude (x-axis)
    # ax1.set_xticks(range(int(extent[0]), int(extent[1]) + 1, 5), crs=ccrs.PlateCarree())
    ax1.set_xticks(np.arange(-85, -59, 5), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax1.xaxis.set_minor_formatter(cticker.LongitudeFormatter())
    
    # Latitude (y-axis)
    # ax1.set_yticks(range(int(extent[2]), int(extent[3]) + 1, 5), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(10, 25, 5), crs=ccrs.PlateCarree())
    ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax1.yaxis.set_minor_formatter(cticker.LatitudeFormatter())

    # Customize tick label style if needed
    ax1.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1)
    for tick in ax1.xaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.yaxis.get_majorticklabels():
        tick.set_fontweight('bold')

    # Calculate contours
    if limits:
        ohc_min = limits[0]
        ohc_max = limits[1]
        ohc_stride = limits[2]
    else:
        # Calculate the colorbar limits automatically
        percentiles = [.25, .75]
        quantile = ds1['ohc'].quantile(percentiles)
        ohc_min = np.floor(quantile[0])
        ohc_max = np.ceil(quantile[1])
        ohc_stride = 10
    ohc_stride = 20

    # Calculate salinity contours
    levels = np.arange(
        ohc_min,
        ohc_max+ohc_stride,
        ohc_stride
        )

    cmap = cmocean.cm.thermal

    # # Ocean Heat Content Plot
    h1 = ax1.contourf(ds1['lon'], ds1['lat'], ds1['ohc'],
                      levels=levels, 
                      extend="max",
                      cmap=cmap,
                      transform=transform['data'])
    
    h3 = ax1.contour(ds1['lon'], 
                     ds1['lat'], 
                     ds1['ohc'], 
                     [60],
                     linestyles='-',
                     colors=['silver'],
                     linewidths=1,
                     alpha=1,
                     transform=ccrs.PlateCarree(),
                     zorder=101)

    # h0 = []
    # l0 = []
    
    # h0.append(mlines.Line2D([], [], linestyle='-', color='silver', alpha=1, linewidth=1))
    # l0.append('60 kJ cm-2')
    # h0.append(mlines.Line2D([], [], linestyle='-', color='white', alpha=1, linewidth=1))
    # l0.append('Past 5 days')
    # leg1 = ax1.legend(h0, l0, loc='upper right', fontsize=9)
    # ax1.add_artist(leg1)

    # # Deal with the third axes
    # h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    # if (len(h) > 0) & (len(l) > 0):
 
    #     # Add handles to legend
    #     leg = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

    #     # Add title to legend
    #     t0 = []
    #     if isinstance(argo, DataFrame):
    #         if not argo.empty:
    #             t0.append(argo.index.min()[1])

    #     if isinstance(gliders, DataFrame):
    #         if not gliders.empty:
    #             t0.append(gliders.index.min()[1])

    #     if len(t0) > 0:
    #         t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
    #     else:
    #         t0 = None
    #     legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
    #     leg.set_title(legstr, prop={'size': 9, 'weight': 'bold', 'style': 'italic'})
    #     # ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
    # ax3.set_axis_off()

    # leg1.set_zorder(10001)

    # plot_hurricane_track(ax1, time, basin, storm_id="AL092024", linecolor='white', markersize=60)
    # plot_hurricane_track(ax1, time, basin, storm_id="AL142024", linecolor='white', markersize=60)
    # plot_hurricane_track(ax1, time, basin, storm_id="AL022024", linecolor='white', markersize=150)
    
    # Add colorbar to first axes
    cb = fig.colorbar(h1, ax=ax1, orientation="vertical", shrink=.95, aspect=15)#, shrink=0.7, aspect=20*0.7)
    # cb = fig.colorbar(h1, ax=ax1, orientation="vertical", shrink=.8, aspect=20)#, shrink=0.7, aspect=20*0.7)
    
    cb.ax.tick_params(labelsize=12)
    cb.set_label('kJ/cm^2', fontsize=12, fontweight="bold")

    # Set title for each axes
    title_str = f"Ocean Heat Content - {ds1.model.upper()} - {time.strftime(tstr_title)}"
    # title_str = f" Hurricane Beryl - RU29"
    ax1.set_title(title_str, fontsize=18, fontweight='bold')
    # plt.suptitle(title_str, fontweight="bold", fontsize=16)

    # from ioos_model_comparisons.plotting_hurricanes import plot_storms
    # # import tropycal.tracks as tracks

    # # hurricane_data = tracks.TrackDataset(basin='north_atlantic', include_btk=True)
    # # idalia = hurricane_data.get_storm(("helene", 2024))

    # fcast = idalia.get_nhc_forecast_dict(time=time)  # This will return the NHC forecast dictionary
    # forecasts = [fcast]
    # # storms = [idalia]
    # plot_storms(ax1, storms, time, zorder=80)

    # if storms:
    #     for s in storms.keys():
    #         # storms['track']
    #         # lon = storms[s]['track']['lon']
    #         # lat = storms[s]['track']['lat']
    #         # cone = storms[s]['cone']
            # plot_storm_track(ax1, storms[s], time)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for axs in ax.flat:
        # axs.label_inner()

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.15)
    # fig.subplots_adjust(top=0.80)
    
    export_fig(path_save, fname, dpi=dpi)
    plt.close()

def plot_salt_single(ds1, extent, region_name,
             limits=None,
             bathy=None,
             argo=None,
             gliders=None,
             eez=False,
             currents=None,
             cols=6,
             path_save=os.getcwd(), 
             transform=dict(map=ccrs.Mercator(), 
                            data=ccrs.PlateCarree()
                            ), 
             figsize=(14,8),
             dpi=150,
             overwrite=False
             ):

    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.values)
    
    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')
    tstr_folder = time.strftime('%Y-%m-%dT%H%M%SZ')
    year = time.strftime("%Y")
    month = time.strftime("%m")

    # Generate filename
    fname = f"{path_save.name}_{tstr_folder}_heat_content_{ds1.model.lower()}.png"

    # Append salinity_max, year, and month to path_save
    path_save = path_save / 'salinity' / year / month

    save_file = path_save / fname
    
    if save_file.is_file():
        if not overwrite:
            print(f"{save_file} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{save_file} exists. Overwrite: True. Replotting.")    
    
    # Make sure path_save exists
    os.makedirs(path_save, exist_ok=True)
    
    print(f"Plotting Salinity of {region_name} at {tstr_title}")
 

    # Adjust the figure size to create more space at the bottom
    fig = plt.figure(figsize=figsize)  # Adjusted figure size

    # Create main plot axes
    ax1 = fig.add_subplot(1, 1, 1, projection=proj['map'])


    # Create an additional axes for the legend at the bottom
    # ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # Adjust these values as needed

    create(extent, ax=ax1, ticks=False)
        
    if bathy:       
        add_bathymetry(ax1,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.z.values,
                        levels=(-1000, -100),
                        zorder=1.5
                        )
    # Add eez lines
    if eez:
        map_add_eez(ax1, color='white', zorder=10, linestyle='-', linewidth=2)

    # Setup keyword arguments dictionary for plot_regional_assets
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    
    # Plot gliders and argo floats
    plot_regional_assets(ax1, **rargs)

    import cartopy.mpl.ticker as cticker


    # Add gridlines at every 1 degree without labels
    gl = ax1.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,           # Disable automatic labels
        linestyle="--",
        color="black",
        alpha=0.5,
    )
    gl.xlocator = plt.MultipleLocator(1)  # Set 1-degree interval for gridlines
    gl.ylocator = plt.MultipleLocator(1)
    gl.set_zorder(9.99)

    # Manually add labeled ticks at every 5 degrees on both axes
    # Longitude (x-axis)
    # ax1.set_xticks(range(int(extent[0]), int(extent[1]) + 1, 5), crs=ccrs.PlateCarree())
    ax1.set_xticks(np.arange(-85, -59, 5), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax1.xaxis.set_minor_formatter(cticker.LongitudeFormatter())
    
    # Latitude (y-axis)
    # ax1.set_yticks(range(int(extent[2]), int(extent[3]) + 1, 5), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(10, 25, 5), crs=ccrs.PlateCarree())
    ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax1.yaxis.set_minor_formatter(cticker.LatitudeFormatter())

    ax1.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1)
    for tick in ax1.xaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.yaxis.get_majorticklabels():
        tick.set_fontweight('bold')
    # plot_hurricane_track(ax1, time, basin, storm_id="AL022024", linecolor='white', markersize=60)


    # Add ticks
    # add_ticks(ax1, extent)

    # Calculate contours
    # if limits:
    ohc_min = 34.6
    ohc_max = 36.3
    ohc_stride = .1
    # else:
    #     # Calculate the colorbar limits automatically
    #     percentiles = [.25, .75]
    #     quantile = ds1['ohc'].quantile(percentiles)
    #     ohc_min = np.floor(quantile[0])
    #     ohc_max = np.ceil(quantile[1])
    #     ohc_stride = 10

    # Calculate salinity contours
    levels = np.arange(
        ohc_min,
        ohc_max+ohc_stride,
        ohc_stride
        )

    cmap = cmocean.cm.haline

    # Ocean Heat Content Plot
    h1 = ax1.contourf(ds1['lon'], ds1['lat'], ds1['salinity'],
                      levels=levels, 
                      extend="both",
                      cmap=cmap,
                      transform=transform['data'])
    # h3 = ax1.contour(ds1['lon'], 
    #                 ds1['lat'], 
    #                 ds1['salinity'], 
    #                 [60],
    #                 linestyles='-',
    #                 colors=['silver'],
    #                 linewidths=1,
    #                 alpha=1,
    #                 transform=ccrs.PlateCarree(),
    #                 zorder=101)

    h0 = []
    l0 = []
    
    # h0.append(mlines.Line2D([], [], linestyle='-', color='silver', alpha=1, linewidth=1))
    # l0.append('60 kJ cm-2')
    # h0.append(mlines.Line2D([], [], linestyle='-', color='white', alpha=1, linewidth=1))
    # l0.append('Past 5 days')
    leg1 = ax1.legend(h0, l0, loc='upper left', fontsize=9)

    # # # Deal with the third axes
    # h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    # if (len(h) > 0) & (len(l) > 0):
 
    #     # Add handles to legend
    #     leg = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

    #     # Add title to legend
    #     t0 = []
    #     if isinstance(argo, pd.DataFrame):
    #         if not argo.empty:
    #             t0.append(argo.index.min()[1])

    #     if isinstance(gliders, pd.DataFrame):
    #         if not gliders.empty:
    #             t0.append(gliders.index.min()[1])

    #     if len(t0) > 0:
    #         t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
    #     else:
    #         t0 = None
    #     legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
    #     leg.set_title(legstr, prop={'size': 9, 'weight': 'bold', 'style': 'italic'})
    #     # ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
    # ax3.set_axis_off()


    leg1.set_zorder(10001)
    
    # Add colorbar to first axes
    cb = fig.colorbar(h1, ax=ax1, orientation="vertical", shrink=.95, aspect=15)#, shrink=0.7, aspect=20*0.7)
    cb.ax.tick_params(labelsize=14)
    # cb.set_label('kJ/cm^2', fontsize=12, fontweight="bold")

    # Set title for each axes
    # ax1.set_title(f"{ds1.model.upper()}", fontsize=16, fontweight='bold')
    # fig.suptitle(f"Salinity (0m) - {ds1.model.upper()} - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)
    ax1.set_title(f"Salinity (0m) - {ds1.model.upper()} - {time.strftime(tstr_title)}", fontweight="bold", fontsize=20)
    # import tropycal.tracks as tracks
    # from ioos_model_comparisons.plotting_hurricanes import plot_storms

    # hurricane_data = tracks.TrackDataset(basin='north_atlantic', include_btk=True)
    # idalia = hurricane_data.get_storm(("idalia", 2023))
    # fcast = idalia.get_nhc_forecast_dict(time=pd.Timestamp(2023, 8, 28))  # This will return the NHC forecast dictionary
    # forecasts = [fcast]
    # storms = [idalia]
    # plot_storms(ax1, storms, forecasts, zorder=80)


    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for axs in ax.flat:
        # axs.label_inner()

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.15)
    
    export_fig(path_save, fname, dpi=dpi)
    plt.close()


def plot_model_region_ts_vs_speed(ds,
                                  region,
                                  depths,
                                  bathy=None,
                                  argo=None,
                                  gliders=None,
                                  cols=6,
                                  transform=dict(map=proj['map'], 
                                                 data=proj['data']
                                                 ),
                                  path_save=os.getcwd(),
                                  figsize=(14,8),
                                  dpi=150,
                                  overwrite=False
                                  ):
    
    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds.time.data)
    extent = region['extent']

    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')

    # Create figure
    grid = """
    RG
    LL
    """

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
    add_features(ax1)# zorder=0)
    add_features(ax2)# zorder=0)  

    # Add bathymetry lines
    if bathy:
        add_bathymetry(ax1,
                           bathy.longitude.values, 
                           bathy.latitude.values, 
                           bathy.elevation.values,
                           levels=(-1000, -100),
                           zorder=1.5)
        add_bathymetry(ax2,
                           bathy.longitude.values, 
                           bathy.latitude.values, 
                           bathy.elevation.values,
                           levels=(-1000, -100),
                           zorder=1.5)

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    plot_regional_assets(ax1, **rargs)
    plot_regional_assets(ax2, **rargs)

    # Label the subplots
    # ax1.set_title(ds1.model, fontsize=16, fontweight="bold")
    # ax2.set_title(ds2.model, fontsize=16, fontweight="bold")
    txt = plt.suptitle("", fontsize=22, fontweight="bold")
    
    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
        legend = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        legend._legend_box.sep = 1
        # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for k, v in region["variables"].items():  
        var_str = ' '.join(k.split('_')).title()      
        for item in v:
            if item['depth'] in depths:
                print(f"Plotting {k} @ {item['depth']}")
                depth = item['depth']
                tds = ds[k].sel(depth=depth)
                _, speed = uv2spdir(tds['u'], tds['v'])
                # ds_mag = ds[k].sel(depth=depth, method='nearest')
                
                # Create subdirectory for depth under variable subdirectory
                save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
                os.makedirs(save_dir_final, exist_ok=True)

                # Create a file name to save the plot as
                sname = f'{"-".join(region["folder"].split("_"))}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}-{depth}m_{ds.model.lower()}'
                save_file = save_dir_final / f"{sname}.png"

                if save_file.is_file():
                    if not overwrite:
                        print(f"{save_file} exists. Overwrite: False. Skipping.")
                        continue
                    else:
                        print(f"{save_file} exists. Overwrite: True. Replotting.")
                        
                # Add the super title (title for both subplots)
                txt.set_text(f"{var_str} ({depth} m) - {tstr_title}\n")
            
                # Filled contour for each model variable
                # Check if ndims are 1, transform_first requires 2d array
                if (tds['lon'].ndim == 1) & tds['lat'].ndim == 1:
                    rlons, rlats = np.meshgrid(tds['lon'], tds['lat'])
                else:
                    rlons = tds['lon']
                    rlats = tds['lat']

                # Plot first subplot
                # Create dictionary for variable argument inputs for contourf
                vargs = {}
                vargs['transform'] = transform['data']
                vargs['transform_first'] = True
                vargs['cmap'] = cmaps(ds[k].name)
                vargs['extend'] = "both"

                if 'limits' in item:
                    vargs['vmin'] = item['limits'][0]
                    vargs['vmax'] = item['limits'][1]
                    vargs['levels'] = np.arange(vargs['vmin'], vargs['vmax'], item['limits'][2])
                h1 = ax1.contourf(rlons, rlats, tds.squeeze(), **vargs)
                cb1 = fig.colorbar(h1, ax=ax1, orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
                cb1.ax.tick_params(labelsize=12)
                cb1.set_label(f'{k.title()} ({tds.units})', fontsize=12, fontweight="bold")

                # Plot second subplot
                margs = {}
                margs['transform'] = transform['data']
                margs['transform_first'] = True
                margs['cmap'] = cmocean.cm.speed
                margs['extend'] = "both"
                margs['levels'] = np.arange(0, .6, .05)
                h2 = ax2.contourf(rlons, rlats, speed, **margs)
                cb2 = fig.colorbar(h2, ax=ax2, orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
                cb2.ax.tick_params(labelsize=12)
                cb2.set_label(f'Speed (m/s)', fontsize=12, fontweight="bold")

                # Plot streamlines
                sargs = {}
                sargs["transform"] = transform['data']
                sargs["density"] = 3
                sargs["linewidth"] = .75
                sargs["color"] = 'red'
                sargs['zorder'] = 100
                    
                q = ax2.streamplot(rlons.values, rlats.values, 
                                   tds['u'].values, tds['v'].values,
                                   **sargs)
                fig.savefig(save_dir_final / sname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

                # Remove quiver handles from each axes
                q.lines.remove()
                remove_quiver_handles(ax2)

                # Remove handles so we can reuse figure
                # Delete contour handles 
                [x.remove() for x in h1.collections] # axes 1
                [x.remove() for x in h2.collections] # axes 2

                cb1.remove()
                cb2.remove()    

                plt.close()


def plot_model_region_single_streamplot(ds1, region,
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
    
    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    for depth in cdict['depths']:
        print(f"Plotting currents @ {depth}m")
        ds1_depth = ds1.sel(depth=depth, method='nearest')

        # Plot currents with magnitude and direction
        quiver_dir = path_save / f"currents_{depth}m" / time.strftime('%Y/%m')
        os.makedirs(quiver_dir, exist_ok=True)

        # Generate descriptive filename
        sname = f'{region["folder"]}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_currents-{depth}m_{ds1.model.lower()}'
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

        # Initialize qargs dictionary for input into contour plot of magnitude
        qargs = {}
        qargs['transform'] = transform['data']
        qargs['cmap'] = cmocean.cm.speed
        qargs['extend'] = "max"

        if 'limits' in cdict:
            lims = cdict['limits']
            qargs['levels'] = np.arange(lims[0], lims[1]+lims[2], lims[2])

        # Initialize figure
        # Adjust the figure size to create more space at the bottom
        fig = plt.figure(figsize=figsize)  # Adjusted figure size

        # Create main plot axes
        ax1 = fig.add_subplot(1, 1, 1, projection=proj['map'])

        # Create an additional axes for the legend at the bottom
        # ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.05])  # Adjust these values as needed
        
        create(extent, ax=ax1, ticks=False)

        if bathy:       
            try:
                add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5
                            )
            except ValueError:
                print("Bathymetry deeper than specified levels.")

        # Add EEZ
        if eez:
            eez1 = map_add_eez(ax1, zorder=1, color='red', linewidth=2, linestyle='-')
                
        # Setup keyword arguments dictionary for plot_regional_assets
        rargs = {}
        rargs['argo'] = argo
        rargs['gliders'] = gliders
        rargs['transform'] = transform['data']  

        # Plot gliders and argo floats
        plot_regional_assets(ax1, **rargs)
        import cartopy.mpl.ticker as cticker

        # Add gridlines at every 1 degree without labels
        gl = ax1.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,           # Disable automatic labels
            linestyle="--",
            color="black",
            alpha=0.5,
        )
        gl.xlocator = plt.MultipleLocator(1)  # Set 1-degree interval for gridlines
        gl.ylocator = plt.MultipleLocator(1)
        gl.set_zorder(9000)

        # Manually add labeled ticks at every 5 degrees on both axes
        # Longitude (x-axis)
        # ax1.set_xticks(range(int(extent[0]), int(extent[1]) + 1, 5), crs=ccrs.PlateCarree())
        ax1.set_xticks(np.arange(-90, -75, 5), crs=ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax1.xaxis.set_minor_formatter(cticker.LongitudeFormatter())
        
        # Latitude (y-axis)
        # ax1.set_yticks(range(int(extent[2]), int(extent[3]) + 1, 5), crs=ccrs.PlateCarree())
        ax1.set_yticks(np.arange(20, 35, 5), crs=ccrs.PlateCarree())
        ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax1.yaxis.set_minor_formatter(cticker.LatitudeFormatter())

        # Customize tick label style if needed
        ax1.tick_params(axis='both', which='major', labelsize=12, direction='out', length=6, width=1)

        # # Add ticks
        # add_ticks(ax1, extent)

        # Set coarsening configs to a variable
        if 'coarsen' in cdict:
            coarsen = region['currents']['coarsen']
        else:
            coarsen['rtofs'] = 1
            coarsen['gofs'] = 1

        # Filled contour for each model variable
        m1 = ax1.contourf(ds1_depth["lon"], ds1_depth["lat"], mag_r, **qargs)

        # Add streamlines
        s1 = map_add_currents(ax1, ds1_depth, coarsen=coarsen[ds1.model.lower()], **currents["kwargs"])
        
        # # # Deal with the third axes
        # h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
        # if (len(h) > 0) & (len(l) > 0):
    
        #     # Add handles to legend
        #     leg = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

        #     # Add title to legend
        #     t0 = []
        #     if isinstance(argo, pd.DataFrame):
        #         if not argo.empty:
        #             t0.append(argo.index.min()[1])

        #     if isinstance(gliders, pd.DataFrame):
        #         if not gliders.empty:
        #             t0.append(gliders.index.min()[1])

        #     if len(t0) > 0:
        #         t0 = min(t0).strftime('%Y-%m-%d %H:00:00')
        #     else:
        #         t0 = None
        #     legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        #     leg.set_title(legstr, prop={'size': 9, 'weight': 'bold', 'style': 'italic'})
        #     # ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        # ax3.set_axis_off()

        # leg1.set_zorder(10001)  
          
        # plot_hurricane_track(ax1, time, basin, storm_id="AL092024", linecolor='white', markersize=60)
        plot_hurricane_track(ax1, time, basin, storm_id="AL142024", linecolor='white', markersize=60)
    
        # Add colorbar to first axes
        cb = fig.colorbar(m1, ax=ax1, orientation="vertical", shrink=.8, aspect=20)#, shrink=0.7, aspect=20*0.7)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(f'Magnitude (m/s)', fontsize=12, fontweight="bold")

        # Create a string for the title of the plot
        title_time = time.strftime("%Y-%m-%d %H:%M:%S")
        title_str = f"Currents ({depth} m) - {ds1.model} - {title_time}\n"
        ax1.set_title(title_str, fontsize=18, fontweight='bold')
        # plt.suptitle(title_str, fontsize=20, fontweight="bold")
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.15)
 

        # Save figure
        fig.savefig(save_file_q, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        if colorbar:
            cb.remove()
            
        # Delete contour handles and remove colorbar axes to use figure
        s1.lines.remove()
        remove_quiver_handles(ax1)
        [x.remove() for x in m1.collections]

import datetime 
def plot_sst(ds1, ds2, region,
             bathy=None,
             argo=None,
             gliders=None,
             cols=6,
             transform=dict(
                 map=proj['map'], 
                 data=proj['data']
                 ),
             path_save=os.getcwd(),
             figsize=(14,8),
             dpi=150,
             colorbar=True,
             overwrite=False
             ):
    k = 'temperature'
    depth = 0
    time = pd.to_datetime(ds1.time.data)
    # Convert to a Python datetime object
    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
    os.makedirs(save_dir_final, exist_ok=True)
    
    sname = f'{"-".join(region["folder"].split("_"))}_{time.strftime("%Y-%m-%dT%H%M%SZ")}_{k}-{depth}m_{ds1.model.lower()}-vs-GOES'

    sfname = save_dir_final / sname

    if sfname.is_file():
        if not overwrite:
            print(f"{sfname} exists. Overwrite: False. Skipping.")
            return
        else:
            print(f"{sfname} exists. Overwrite: True. Replotting.")
    
    # Convert ds.time value to a normal datetime
    time = pd.to_datetime(ds1.time.data)
    extent = region['extent']

    # Formatter for time
    tstr_title = time.strftime('%Y-%m-%d %H:%M:%S')

    # Create subdirectory for region
    # region_file_str = ('_').join(region_name.lower().split(' '))
    # path_save_region = path_save / region['folder']
    
    # # Create a map figure and serialize it if one doesn't already exist
    # region_name = "_".join(region["name"].split(' ')).lower()
    # mdir = path_save / "mapfigs"
    # os.makedirs(mdir, exist_ok=True)
    # sfig = mdir / f"{region_name}_fig.pkl"

    # if not sfig.exists():
    # Create figure

    grid = """
    RG
    LL
    """

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

    # Set map extent
    ax1.set_extent(extent)
    ax2.set_extent(extent)
          
    # Make the map pretty
    add_features(ax1)# zorder=0)
    add_features(ax2)# zorder=0)

    # Add bathymetry lines
    if bathy:
        try:
            add_bathymetry(ax1,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
            add_bathymetry(ax2,
                            bathy.longitude.values, 
                            bathy.latitude.values, 
                            bathy.z.values,
                            levels=(-1000, -100),
                            zorder=1.5)
        except ValueError:
            print("Bathymetry deeper than specified levels.")

    # Add ticks
    add_ticks(ax1, extent, label_left=True)
    add_ticks(ax2, extent, label_left=False, label_right=True)

    # Plot gliders and argo floats
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    plot_regional_assets(ax1, **rargs)
    plot_regional_assets(ax2, **rargs)

    # Label the subplots
    ax1.set_title(ds1.model, fontsize=16, fontweight="bold")
    ax2.set_title('GOES', fontsize=16, fontweight="bold")
    txt = plt.suptitle("", fontsize=22, fontweight="bold")
    
    # Deal with the third axes
    h, l = ax1.get_legend_handles_labels()  # get labels and handles from ax1
    if (len(h) > 0) & (len(l) > 0):
        
        # Add handles to legend
        legend = ax3.legend(h, l, ncol=cols, loc='center', fontsize=8)

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
        legstr = f'Glider/Argo Search Window: {t0} to {str(time)}'
        ax3.set_title(legstr, loc="center", fontsize=9, fontweight="bold", style='italic')
        legend._legend_box.sep = 1
        # plt.figtext(0.5, 0.001, legstr, ha="center", fontsize=10, fontweight='bold')
    ax3.set_axis_off()

    # Iterate through the variables to be plotted for each region. 
    # This dict contains information on what variables and depths to plot. 
    # for k, v in region["variables"].items():
    k = 'temperature'
    depth = 0
    # Create subdirectory for variable under region directory
    var_str = ' '.join(k.split('_')).title()

    print(f"Plotting {k} @ 0m")
    rsub = ds1[k].sel(depth=0)
    gsub = ds2['SST_C']
            
            
    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
    os.makedirs(save_dir_final, exist_ok=True)


    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save / f"{k}_{depth}m" / time.strftime('%Y/%m')
    os.makedirs(save_dir_final, exist_ok=True)

    # Create a file name to save the plot as
    # sname = f'{ds1.model}_vs_{ds2.model}_{k}-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
    save_file = save_dir_final / f"{sname}.png"
                    
    # Add the super title (title for both subplots)
    txt.set_text(f"{var_str} ({depth} m) - {tstr_title}\n")

    # Create dictionary for variable argument inputs for contourf
    vargs = {}
    vargs['transform'] = transform['data']
    vargs['transform_first'] = True
    vargs['cmap'] = cmaps(ds1[k].name)
    vargs['extend'] = "both"

    vargs['vmin'] = region['variables']['temperature'][0]['limits'][0]
    vargs['vmax'] = region['variables']['temperature'][0]['limits'][1]
    vargs['levels'] = np.arange(
        region['variables']['temperature'][0]['limits'][0],
        region['variables']['temperature'][0]['limits'][1],
        region['variables']['temperature'][0]['limits'][2]
        )

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

    # Add contour lines for 15°C isotherm in MAB. This identifies the north wall of the Gulf Stream
    # if region['name'] == 'Mid Atlantic Bight' and k == 'temperature' and depth == 200:
    #     ax1.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)
    #     ax2.contour(rlons, rlats, rsub.squeeze(), levels=[15], colors='red', transform=transform['data'], zorder=10000)

    if colorbar:
        cb = fig.colorbar(h1, ax=axs[:2], orientation="horizontal", shrink=.95, aspect=80)#, shrink=0.7, aspect=20*0.7)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(f'{k.title()} ({rsub.units})', fontsize=12, fontweight="bold")

    # Add EEZ
    # eez1 = map_add_eez(ax1, zorder=1)
    # eez2 = map_add_eez(ax2, zorder=1)
    # points = [(-75.22321918204588, 35.06592247775561), (-74.45437209443314, 35.612148199326924), (-74.35516601861214, 35.81056035096893), (-74.13195234801489, 35.95936946470042), (-73.83433412055189, 36.083377059476675), (-73.56151741204414, 36.157781616342426), (-72.7926703244314, 36.75301807126843), (-72.71826576756564, 37.025834779776176), (-72.84227336234188, 37.29865148828392), (-72.5942581727894, 37.472262120970676), (-72.32144146428163, 37.64587275365743), (-72.17263235055013, 37.819483386344174), (-71.8998156420424, 38.04269705694143), (-71.47818981980313, 38.16670465171768), (-71.18057159234013, 38.19150617067293), (-70.9077548838324, 38.19150617067293), (-70.53573209950363, 38.09230009485193), (-70.16370931517488, 37.96829250007568), (-69.84128956875665, 37.844284905299425), (-69.41966374651739, 37.79468186738893), (-69.02283944323338, 37.79468186738893), (-68.72522121577039, 37.869086424254675), (-68.27879387457588, 37.918689462165176), (-67.78276349547087, 37.893887943209926), (-67.26193159741064, 37.918689462165176), (-66.81550425621612, 37.96829250007568), (-66.29467235815588, 38.09230009485193), (-65.94745109278239, 38.19150617067293), (-65.69943590322987, 38.38991832231493), (-65.57542830845362, 38.53872743604643), (-65.27781008099063, 38.61313199291218), (-65.10419944830386, 38.73713958768843), (-64.98019185352763, 38.811544144554176), (-64.73217666397514, 38.93555173933043), (-63.83932198158612, 40.27483376291393), (-63.41769615934688, 40.250032243958685), (-62.97126881815237, 40.00201705440618), (-62.50003995800262, 39.62999427007743), (-62.10321565471863, 39.233169966793426), (-61.68158983247937, 38.93555173933043), (-61.35917008606111, 38.761941106643675), (-61.03675033964288, 38.712338068733175), (-60.78873515009036, 38.73713958768843), (-60.639926036358865, 38.78674262559893), (-60.59032299844838, 39.00995629619618), (-60.639926036358865, 39.08436085306193), (-61.061551858598115, 39.33237604261443), (-61.45837616188213, 39.48118515634593), (-61.904803503076614, 39.50598667530118), (-62.351230844271115, 39.70439882694318), (-62.54964299591313, 39.87800945962993), (-62.67365059068936, 40.02681857336143), (-62.822459704420865, 40.250032243958685), (-62.82245970442088, 40.32443680082443), (-62.92166578024187, 40.49804743351118), (-62.822459704420865, 40.74606262306368), (-62.62404755277888, 41.01887933157143), (-62.326429325315864, 41.24209300216869), (-62.028811097852866, 41.31649755903443), (-61.63198679456888, 41.465306672765934), (-61.33436856710586, 41.465306672765934), (-61.086353377553365, 41.26689452112393), (-60.813536669045625, 41.01887933157143), (-60.44151388471686, 40.57245199037693), (-60.06949110038811, 40.10122313022718), (-59.72226983501463, 39.75400186485368), (-59.15183489904386, 39.62999427007743), (-58.755010595759856, 39.53078819425643), (-58.234178697699626, 39.60519275112218), (-58.23417869769961, 39.60519275112218), (-57.71334679963936, 39.60519275112218), (-57.34132401531062, 39.60519275112218), (-57.018904268892356, 39.729200345898434), (-56.994102749937106, 40.10122313022718), (-57.06850730680287, 40.34923831977968), (-57.49013312904211, 40.52284895246643), (-57.93656047023661, 40.721261104108436), (-58.085369583968124, 40.89487173679518), (-58.283781735610106, 41.21729148321343), (-58.35818629247586, 41.490108191721184), (-58.35818629247587, 41.63891730545268), (-58.333384773520606, 41.86213097604993), (-58.13497262187861, 41.961337051870935), (-57.83735439441562, 41.961337051870935), (-57.46533161008686, 42.010940089781435), (-57.06850730680286, 41.83732945709468), (-56.795690598295124, 41.614115786497436), (-56.448469332921604, 41.39090211590018), (-56.101248067548106, 41.192489964258186), (-55.828431359040366, 40.99407781261618)]
    # lons = [point[0] for point in points]
    # lats = [point[1] for point in points]

    # ax1.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)
    # ax2.plot(lons, lats, color='red', transform=transform['data'], zorder=1000)

    # points = plt.ginput(n=150, timeout=0)  # You can change n to the number of points you want to select

    # # Print the selected points
    # print("Selected points:", points)
    # plt.show()

    # Save the figure. Using fig to savefig allows us to delete any
    # figure handles so that we can reuse the figure.
    # export_fig(save_dir_final, sname, dpi=dpi)
    fig.savefig(sfname, dpi=dpi, bbox_inches='tight', pad_inches=0.1)

    plt.close()

from pyproj import Geod
geod = Geod(ellps='WGS84')

# from matplotlib.patches import Polygon
from shapely.geometry import Polygon

def plot_model_region_scott(ds, region,
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
    # for key, values in region["variables"].items():
    # Create subdirectory for variable under region directory
    var_str = 'Currents'
    depth = 0

    da = ds.sel(depth=depth, method='nearest')
    # Select variable and depth to plot
    # print(ds[k].name)
    u = da['u']
    v = da['v']

    # Create subdirectory for depth under variable subdirectory
    save_dir_final = path_save_region / f"currents_0m" / time.strftime('%Y/%m')
    os.makedirs(save_dir_final, exist_ok=True)

    # Create a string for the title of the plot
    title_time = time.strftime("%Y-%m-%d %H:%M:%S")
    title = f"{model.upper()} - {var_str} ({depth} m) - {title_time}\n"

    # if not gliders.empty or not argo.empty:
    #         title += f'Assets ({str(t0)} to {str(time)})'

    # Create a file name to save the plot as
    sname = f'{model}-currents_0m-{time.strftime("%Y-%m-%dT%H%M%SZ")}'
    save_file = save_dir_final / f"{sname}.png"

    # Create a map figure and serialize it if one doesn't already exist
    region_name = "_".join(region["name"].split(' ')).lower()
    path_maps = path_save / "mapfigs"
    os.makedirs(path_maps, exist_ok=True)
    sfig = (path_maps / f"{region_name}_fig.pkl")

    # if not sfig.exists():
        # Create an empty projection within set extent
    fig, ax = create(extent, proj=transform['map'])

    # Add bathymetry
    if bathy:
        add_bathymetry(ax,
                        bathy.longitude.values, 
                        bathy.latitude.values, 
                        bathy.elevation.values,
                        levels=(-1000, -100),
                        zorder=1.5)

    lon1 = -86.7878500
    lat1 = 21.4719000
    ax.plot(lon1, lat1, 'o', color='lime', markeredgecolor='black', transform=transform['data'], zorder=2000)
    radius_km = 233  # Radius in kilometers
    angles = [0, 155]  # Angles in degrees (clockwise from north)
    # Calculate the polygon points
    polygon_points = []
    for angle in np.linspace(angles[0], angles[1], num=100):
        lon_dest, lat_dest, _ = geod.fwd(lon1, lat1, angle, radius_km * 1000)
        polygon_points.append((lon_dest, lat_dest))

    # Add the origin point to close the polygon
    polygon_points.append((lon1, lat1))
    polygon = Polygon(polygon_points)

    ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='red', edgecolor='black', alpha=0.5)
    
    lon2 = -86.8676667
    lat2 = 20.8679333
    angles = [36, 152]  # Angles in degrees (clockwise from north)
    ax.plot(lon2, lat2, 'o', color='lime', markeredgecolor='black', transform=transform['data'], zorder=2000)
    # Calculate the polygon points
    polygon_points = []
    for angle in np.linspace(angles[0], angles[1], num=100):
        lon_dest, lat_dest, _ = geod.fwd(lon2, lat2, angle, radius_km * 1000)
        polygon_points.append((lon_dest, lat_dest))

    # Add the origin point to close the polygon
    polygon_points.append((lon2, lat2))
    polygon = Polygon(polygon_points)
    ax.add_geometries([polygon], crs=ccrs.PlateCarree(), facecolor='blue', edgecolor='black', alpha=0.5)
    

    #     save_fig(fig, path_maps, f"{region_name}_fig.pkl")       
    # else:
    #     fig = load_fig(sfig)
    #     ax = fig.axes[0]
                        
    rargs = {}
    rargs['argo'] = argo
    rargs['gliders'] = gliders
    rargs['transform'] = transform['data']  
    plot_regional_assets(ax, **rargs)

    cargs = {}
    cargs['vmin'] = 0
    cargs['vmax'] = 2.0
    cargs['transform'] = transform['data']
    cargs['cmap'] = cmocean.cm.speed
    cargs['levels'] = np.arange(0, 2.1, .1)
    cargs['extend'] = 'both'

    try:
        cargs.pop('vmin'), cargs.pop('vmax')
    except KeyError:
        pass

    ang, spd = uv2spdir(u, v)
    
    # If the xarray DataArray contains data, let's contour the data.
    h = ax.contourf(da['lon'], da['lat'], spd, **cargs)

    currents = dict(
        bool=True,
        depths = [0, 150],
        limits = [0, 1.5, .1],
        coarsen=dict(rtofs=5, gofs=6),
        kwargs=dict(
            ptype="streamplot",
            color="black",
            density=3
            )
    )
    
    map_add_currents(ax, da, **currents['kwargs'])

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
    cb.set_label(f'Speed (cm/s)', fontsize=13)

    ax.set_title(title, fontsize=16, fontweight='bold')

    # if legend:
    #     h, l = ax.get_legend_handles_labels()  # get labels and handles from ax1

    #     if (len(h) > 0) & (len(l) > 0):
    #         # Shrink current axis's height by 10% on the bottom
    #         box = ax.get_position()
    #         ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                         box.width, box.height * 0.9])

    #         # Put a legend below current axis
    #         ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #                 fancybox=True, shadow=True, ncol=5)
    #         legstr = f'Glider/Argo Search Window: {str(t0)} to {str(time)}'
    #         plt.figtext(0.5, -0.07, legstr, ha="center", fontsize=10, fontweight='bold')

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