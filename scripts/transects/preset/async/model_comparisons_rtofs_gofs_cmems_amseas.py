#!/usr/bin/env python

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmocean import cm
import pandas as pd
import seawater
from oceans.ocfis import spdir2uv, uv2spdir
from ioos_model_comparisons.plotting import map_add_ticks, export_fig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ioos_model_comparisons.models import rtofs, gofs, cmems, amseas
# from hurricanes.calc import calculate_transect
import ioos_model_comparisons.configs as conf
import os

# Set path to save plots
path_save = conf.path_plots / "transects" / "preset"

# Select time we want to look at
t0 = pd.Timestamp(2022, 8, 2, 0, 0, 0)
t1 = pd.Timestamp(2022, 8, 5, 0, 0, 0)
times = pd.date_range(t0, t1, freq='6H')

# Isla Contoy to Cuba
lons = [-88.6, -83]
lats = [17.8, 25]
title = "Straits of Yucatán (Isla Contoy to Western Cuba)"
rstr = "yucatan_straits"
xaxis = 'lon'
start = -86.79, 21.46
end = -84.95, 21.86
points = 500
depth_spacing = 1
depth=400
# # Contour levels
# levels_salinity = np.arange(34.8, 36.9, .1)
# levels_temperature = np.arange(8, 28, 1)
# levels_u = np.arange(-1.0, 1.0, .1)
# levels_v = np.arange(-1.0, 1.0, .1)
# # levels_density = np.arange(1023, 1030, 1)
# levels_density = np.arange(1023, 1030, .5)

# # Straits of Florida
# lons = [-83.77, -79.64]
# lats = [22.8, 24.9]
# title = "Straits of Florida (Cuba to Key West)"
# rstr = "florida_straits"
# xaxis = 'lat'
# start = -81.80, 23.2
# end = -81.80, 24.5
# points = 500
# depth_spacing = 1
# depth = 400

# # Eastern Caribbean (-60w)
# lons = [-89, -55]
# lats = [7, 24]
# title = "Eastern Caribbean (-60W, 20N to 8.25N)"
# rstr = "eastern_caribbean_60w"
# xaxis = 'lat'
# start = -60, 20
# end = -60, 8.25
# points = 500
# depth_spacing = 1

# # Eastern Caribbean (-60w)
# lons = [-89, -55]
# lats = [7, 24]
# title = "Eastern Caribbean (-63W, 20N to 10N)"
# rstr = "eastern_caribbean_63w"
# xaxis = 'lat'
# start = -63, 20
# end = -63, 10
# points = 500
# depth_spacing = 1

# # South Atlantic Bight
# lons = [-82, -64]
# lats = [25, 36]
# title = "South Atlantic Bight (-81.2W to -77.8W, 30N)"
# rstr = "south_atlantic_bight"
# xaxis = 'lon'
# start = -81.2, 30
# end = -77.8, 30
# points = 500
# depth_spacing = 

# Contour levels
levels_salinity = np.arange(34.8, 36.9, .1)
levels_temperature = np.arange(8, 28, 1)
levels_u = np.arange(-1.0, 1.0, .1)
levels_v = np.arange(-1.0, 1.0, .1)
# levels_density = np.arange(1023, 1030, 1)
levels_density = np.arange(1023, 1030, .5)

# Sort the lons and lats so they are in monotonically increasing order 
lons.sort()
lats.sort()

# Create the final path based off the region string
path_final = path_save / rstr
os.makedirs(path_final, exist_ok=True)

# Convert lons (-180, 180) into lons (0, 360)
glons = np.mod(lons, 360)

def transect2rtofs(pts, grid_lons, grid_lats, grid_x, grid_y):
    # if not grid_x:
    #     grid_x = np.arange(0, len(grid_lons))
    # if not grid_y:
    #     grid_y = np.arange(0, len(grid_lats))
    
    # Convert points to x and y index for rtofs
    # Use piecewise linear interpolation (np.interp) to find the partial index (float instead of an int)
    # of the points (lon,lat) lying on the RTOFS x,y grid 
    
    # We pass the following:
    # np.interp(x, grid_lon, grid_x)
    # np.interp(y, grid_lat, grid_y)
    lonidx = np.interp(pts[:,0], grid_lons, grid_x)
    latidx = np.interp(pts[:,1], grid_lats, grid_y)
    return lonidx, latidx

def calculate_transect(start, end, npts):
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    pts = g.inv_intermediate(start[0], start[1], end[0], end[1], npts) 
    return np.column_stack([pts.lons, pts.lats])

def plot_map(da, cmap, vrange=None, title=None, figsize=(20,12), sname=None, dpi=300):   
    ctime = pd.to_datetime(da.time.values)
    ctimestr = ctime.strftime("%Y%m%dT%H%M%SZ")
    
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.Mercator())
    )
    
    edgecolor = 'black'
    landcolor = 'tan'
    
    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )
    
    LAND = cfeature.GSHHSFeature(scale='full')
    
    # Axes properties and features
    # ax.set_extent(extent)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(LAND, edgecolor=edgecolor, facecolor=landcolor)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(state_lines, edgecolor=edgecolor)
    ax.add_feature(cfeature.BORDERS)
    
    # Fill dictionary for keyword arguments
    cargs = {}
    cargs['transform'] = ccrs.PlateCarree()
    cargs['extend'] = 'both'
    cargs['cmap'] = cmap
    
    if vrange:
        vmin = vrange[0]
        vmax = vrange[1]
        vstep = vrange[2]
    else:
        vmin = np.floor(np.nanpercentile(da, 2))
        vmax = np.ceil(np.nanpercentile(da, 98))
        vstep = (vmax-vmin)/12

    cargs['vmin'] = vmin
    cargs['vmax'] = vmax
    cargs['levels'] = np.arange(vmin, vmax+vstep, vstep)

    # Plot filled contour of data variable
    h = ax.contourf(da['lon'], da['lat'], da.squeeze(), **cargs)
    axins = inset_axes(ax,  # here using axis of the lowest plot
    width="2.5%",  # width = 5% of parent_bbox width
    height="100%",  # height : 340% good for a (4x4) Grid
    loc='lower left',
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0
    )
    
    # ax.plot(tlons_1, tlats_1, 'r-.', transform=ccrs.PlateCarree(), linewidth=4)
    # ax.plot(tlons_2, tlats_2, 'r-.', transform=ccrs.PlateCarree(), linewidth=4)
    # ax.grid(zorder=40)
    # ax.grid(draw_labels=True)
    
    # Colorbar
    cb = fig.colorbar(h, cax=axins)
    cb.ax.tick_params(labelsize=15)
    cb.set_label(f'{da.name.title()} ({da.units})', fontsize=16, weight='bold')
    
    # Title and axes
    if title:
        title = f"{title}: {da.name.title()} ({int(da.depth.values)} m) - {pd.to_datetime(da.time.values)}"
    else:
        title = f"{da.name.title()} ({int(da.depth.values)} m) - {pd.to_datetime(da.time.values)}"   
        
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=18, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=18, fontweight="bold")
    
    # Ticks
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    map_add_ticks(ax, [da["lon"].min(), da["lon"].max(), da["lat"].min(), da["lat"].max()])
    
    if sname:
        sname = f"{sname}_{da.name}_map_{ctimestr}.png"
        plt.savefig(sname, dpi=dpi, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
        
    return ax

def plot_transect(ds, var, cmap, levels=None, xaxis="lon", title=None, ax=None, cbar=False, fontfrac=1, contour=None, xlabel=None, ylabel=None, sname=None, dpi=300):
    var = ds[var]
    # var_name = ' '.join(var.standard_name.split("_")).title()
    try:
        var_units = var.units
    except AttributeError:
            var_units = ''
    ctime = pd.to_datetime(var.time.values)
    ctimestr = ctime.strftime("%Y%m%dT%H%M%SZ")

    if not ax:
        fig, ax = plt.subplots(
            figsize=(30, 12),
        )

    h = var.plot(ax=ax, x=xaxis, y="depth", levels=levels, cmap=cmap, extend="both",  add_colorbar=False, add_labels=False)
    
    if contour:
        ax.contour(ds[xaxis], ds['depth'], var, contour, colors='k')
    
    if cbar:
        # cb = plt.colorbar(h, ax=ax, orientation="vertical")
        cb = plt.colorbar(h, **cbar)
        cb.set_label(label=f"{var.name.title()} ({var_units})", size=14*fontfrac, weight='bold')
        cb.ax.tick_params(labelsize=12*fontfrac)

    ax.set_ylim([depth, 0])
    plt.setp(ax.get_xticklabels(), fontsize=16*fontfrac)
    plt.setp(ax.get_yticklabels(), fontsize=16*fontfrac)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18*fontfrac, fontweight="bold")
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18*fontfrac, fontweight="bold")
    
    if title:
        ax.set_title(title, fontsize=24*fontfrac, fontweight="bold")
        
    if sname:
        sname = f"{sname}_{var.name}_{ctimestr}.png"
        plt.savefig(sname, dpi=dpi, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
    return ax

# Load and munge RTOFS
# # Use the xarray open_dataset function to access the dataset via opendap. Set lon and lat to coordinates for easy plotting and select the center time above and grab only the top 1000m
rtofs = rtofs()
rtofs = rtofs.set_coords(['lon', 'lat']).sel(depth=slice(0, 1000))

# Save rtofs lon and lat as variables to speed up indexing calculation
grid_lons = rtofs.lon.values[0,:]
grid_lats = rtofs.lat.values[:,0]
grid_x = rtofs.x.values
grid_y = rtofs.y.values

# Find x, y index of area we want to subset
lons_ind = np.interp(lons, grid_lons, grid_x)
lats_ind = np.interp(lats, grid_lats, grid_y)

# We use np.floor on the first index of each slice and np.ceiling on the second index of each slice 
# in order to widen the area of the extent slightly around each index. 
# This returns a float so we have to broadcast the datatype to an integer in order to use .isel
extent = np.floor(lons_ind[0]).astype(int), np.ceil(lons_ind[1]).astype(int), np.floor(lats_ind[0]).astype(int), np.ceil(lats_ind[1]).astype(int)

# Use the xarray .isel selector on x/y since we know the exact indexes we want to slice
rtofs = rtofs.isel(
    x=slice(extent[0], extent[1]),
    y=slice(extent[2], extent[3])
    )

# Load and munge GOFS
gofs = gofs(rename=True)
gofs = gofs.sel(depth=slice(0,1000), lon=slice(glons[0], glons[1]), lat=slice(lats[0], lats[1]))

gofs["lon"] = np.mod(gofs["lon"]+180, 360)-180

# Load and munge AMSEAS
am = amseas(rename=True).sel(depth=slice(0,1000), lon=slice(glons[0], glons[1]), lat=slice(lats[0], lats[1]))
am["lon"] = np.mod(am["lon"]+180, 360)-180

# Load Copernicus data
cmems = cmems(rename=True).sel(depth=slice(0,1000), lon=slice(lons[0], lons[1]), lat=slice(lats[0], lats[1]))

# Loop through times
for ctime in times:
    ctime_str = ctime.strftime("%Y-%m-%d %H:%M:%S")
    ctime_save_str = ctime.strftime("%Y-%m-%dT%H%M%SZ")

    # Select ctime with RTOFS
    try:
        trtofs = rtofs.sel(time=ctime)
    except:
        continue

    # Use xr.apply_ufunc to apply a function over all dimensions
    trtofs["pressure"] = xr.apply_ufunc(seawater.eos80.pres, trtofs.depth, trtofs.lat)
    trtofs["density"] = xr.apply_ufunc(seawater.eos80.dens, trtofs.salinity, trtofs.temperature, trtofs.pressure)

    # GOFS
    try:
        tgofs = gofs.sel(time=ctime)
    except:
        continue
    tgofs["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tgofs.depth, tgofs.lat)
    tgofs["density"] = xr.apply_ufunc(seawater.eos80.dens, tgofs.salinity, tgofs.temperature, tgofs.pressure)

    try:
        # AMSEAS
        tam = am.sel(time=ctime)
    except:
        continue
    tam["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tam.depth, tam.lat)
    tam["density"] = xr.apply_ufunc(seawater.eos80.dens, tam.salinity, tam.temperature, tam.pressure)

    # Copernicus
    try:
        tcmems = cmems.sel(time=ctime, method='nearest')
    except:
        continue
    tcmems["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tcmems.depth, tcmems.lat)
    tcmems["density"] = xr.apply_ufunc(seawater.eos80.dens, tcmems.salinity, tcmems.temperature, tcmems.pressure)

    # Calculate transect line between the start point and end point
    # returns all points as a numpy array with lon in the first column and lat in the second column
    pts = calculate_transect(start, end, points)
        
    # Convert to the x, y indexes so we can work with the RTOFS model
    grid_lons = trtofs.lon.values[0,:]
    grid_lats = trtofs.lat.values[:,0]
    grid_x = trtofs.x.values
    grid_y = trtofs.y.values
    lonidx, latidx = transect2rtofs(pts, grid_lons, grid_lats, grid_x=grid_x, grid_y=grid_y)

    # Create map plot
    ax = plot_map(trtofs["salinity"].sel(depth=0), cm.haline, (35, 37, .1), title="RTOFS", figsize=(30,20))
    ax.plot(pts[:,0], pts[:,1], 'r-.', linewidth=4, transform=ccrs.PlateCarree())
    export_fig(path_final, f'{rstr}_transect_map_salinity_{ctime_save_str}.png', dpi=conf.dpi)

    # Interpolate model data to transect
    # Interpolate RTOFS Transect
    rds = trtofs.interp(
        x=xr.DataArray(lonidx, dims="point"),
        y=xr.DataArray(latidx, dims="point"),
        depth=xr.DataArray(np.arange(0, trtofs.depth.max()+depth_spacing, depth_spacing), dims="depth")
    )

    # Interpolate GOFS Transect
    gds = tgofs.interp(
        lon=xr.DataArray(pts[:,0], dims="point"),
        lat=xr.DataArray(pts[:,1], dims="point"),
        depth=xr.DataArray(np.arange(0, tgofs.depth.max()+depth_spacing, depth_spacing), dims="depth")
    )

    # Interpolate AMSEAS Transect
    ads = tam.interp(
        lon=xr.DataArray(pts[:,0], dims="point"),
        lat=xr.DataArray(pts[:,1], dims="point"),
        depth=xr.DataArray(np.arange(0, tam.depth.max()+depth_spacing, depth_spacing), dims="depth")
    )

    # Interpolate Copernicus Transect
    cds = tcmems.interp(
        lon=xr.DataArray(pts[:,0], dims="point"),
        lat=xr.DataArray(pts[:,1], dims="point"),
        depth=xr.DataArray(np.arange(0, tcmems.depth.max()+depth_spacing, depth_spacing), dims="depth")
    )

    # Create salinity, temperature, density plots
    fig, ax = plt.subplots(4, 3, figsize=(40,20))
    if xaxis =='lon':
        xlabel = "Longitude"
    elif xaxis == 'lat':
        xlabel = "Latitude"
    plot_transect(rds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[0,0], cbar=dict(ax=ax[0,0], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[36.6], ylabel="RTOFS\nDepth (m)")
    plot_transect(gds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[1,0], cbar=dict(ax=ax[1,0], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[36.6], ylabel="GOFS\nDepth (m)")
    plot_transect(ads, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[2,0], cbar=dict(ax=ax[2,0], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[36.6], ylabel="AMSEAS\nDepth (m)")
    plot_transect(cds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[3,0], cbar=dict(ax=ax[3,0], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[36.6], ylabel="Copernicus\nDepth (m)", xlabel=xlabel)

    plot_transect(rds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[0,1], cbar=dict(ax=ax[0,1], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(gds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[1,1], cbar=dict(ax=ax[1,1], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(ads, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[2,1], cbar=dict(ax=ax[2,1], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(cds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[3,1], cbar=dict(ax=ax[3,1], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26], xlabel=xlabel)

    plot_transect(rds, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[0,2], cbar=dict(ax=ax[0,2], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(gds, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[1,2], cbar=dict(ax=ax[1,2], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(ads, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[2,2], cbar=dict(ax=ax[2,2], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26])
    plot_transect(cds, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[3,2], cbar=dict(ax=ax[3,2], orientation='vertical', pad=0.01), fontfrac=1.25, contour=[26], xlabel=xlabel)

    plt.suptitle(f"{title}\n{ctime_str}", fontsize=32, fontweight="bold")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        top=0.92, 
                        wspace=0.0075, 
                        hspace=0.2)
    export_fig(path_final, f'{rstr}_tsd_{ctime_save_str}.png', dpi=conf.dpi)

    # Create velocity plots
    fig, ax = plt.subplots(4, 2, figsize=(40,30))
    plot_transect(rds, "u", cm.balance, levels=levels_u, xaxis=xaxis, title="Eastward Velocity", ax=ax[0,0], cbar=dict(ax=ax[0,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="RTOFS\nDepth (m)")
    plot_transect(gds, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[1,0], cbar=dict(ax=ax[1,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="GOFS\nDepth (m)")
    plot_transect(ads, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[2,0], cbar=dict(ax=ax[2,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="AMSEAS\nDepth (m)")
    plot_transect(cds, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[3,0], cbar=dict(ax=ax[3,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="Copernicus\nDepth (m)", xlabel=xlabel)

    plot_transect(rds, "v", cm.balance, levels=levels_v, xaxis=xaxis, title="Northward Velocity", ax=ax[0,1], cbar=dict(ax=ax[0,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    plot_transect(gds, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[1,1], cbar=dict(ax=ax[1,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    plot_transect(ads, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[2,1], cbar=dict(ax=ax[2,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    plot_transect(cds, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[3,1], cbar=dict(ax=ax[3,1], orientation='vertical', pad=0.01), fontfrac=1.5, xlabel=xlabel)


    plt.suptitle(f"{title}\n{ctime_str}", fontsize=32, fontweight="bold")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        top=0.94, 
                        wspace=0.005, 
                        hspace=0.1)
    export_fig(path_final, f'{rstr}_velocities_{ctime_save_str}.png', dpi=conf.dpi)



    # # 0m
    # d = 0
    # # Salinity
    # plot_map(rtofs["salinity"].sel(depth=0), cm.haline, (36, 36.4, .05), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map(gofs["salinity"].sel(depth=0), cm.haline, (36, 36.4, .05), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map(am["salinity"].sel(depth=0), cm.haline, (36, 36.4, .05), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map(cmems["salinity"].sel(depth=0, method='nearest'), cm.haline, (36, 36.4, .05), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # Temperature
    # plot_map(rtofs["temperature"].sel(depth=0), cm.thermal, (24, 29, .5), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map(gofs["temperature"].sel(depth=0), cm.thermal, (24, 29, .5), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map(am["temperature"].sel(depth=0), cm.thermal, (24, 29, .5), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map(cmems["temperature"].sel(depth=0, method='nearest'), cm.thermal, (24, 29, .5), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # Streamplots (Northward Velocity)
    # plot_map_with_streamplot(rtofs["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="RTOFS", density=2, figsize=(20, 16), sname=f"{rstr}_rtofs_{d}m")
    # plot_map_with_streamplot(gofs["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="GOFS", density=2, figsize=(20, 16), sname=f"{rstr}_gofs_{d}m")
    # plot_map_with_streamplot(am["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", density=2, figsize=(20, 16), sname=f"{rstr}_amseas_{d}m")
    # plot_map_with_streamplot(cmems["v"].sel(depth=d, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="Copernicus", density=2, figsize=(20, 16), sname=f"{rstr}_cmems_{d}m")

    # # 100m
    # # Salinity
    # plot_map(rtofs["salinity"].sel(depth=100), cm.haline, (36, 36.7, .05), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map(gofs["salinity"].sel(depth=100), cm.haline, (36, 36.7, .05), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map(am["salinity"].sel(depth=100), cm.haline, (36, 36.7, .05), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map(cmems["salinity"].sel(depth=100, method='nearest'), cm.haline, (36, 36.7, .05), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # Temperature
    # plot_map(rtofs["temperature"].sel(depth=100), cm.thermal, (18, 27, 1), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map(gofs["temperature"].sel(depth=100), cm.thermal, (18, 27, 1), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map(am["temperature"].sel(depth=100), cm.thermal, (18, 27, 1), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_am_{d}m")
    # plot_map(cmems["temperature"].sel(depth=100, method='nearest'), cm.thermal, (18, 27, 1), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # d = 200
    # # plot_map_with_streamplot(trtofs["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="RTOFS", density=2, figsize=(20, 16), sname=f"{tname}_rtofs_{d}m")
    # # plot_map_with_streamplot(tgofs["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="GOFS", density=2, figsize=(20, 16), sname=f"{tname}_gofs_{d}m")
    # # plot_map_with_streamplot(tam["v"].sel(depth=d), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", density=2, figsize=(20, 16), sname=f"{tname}_amseas_{d}m")
    # # plot_map_with_streamplot(tcmems["v"].sel(depth=d, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="Copernicus", density=2, figsize=(20, 16), sname=f"{tname}_cmems_{d}m")

    # # Coarsen the models for quiver plots
    # trtofs = rtofs.coarsen(x=2, boundary='pad').mean().coarsen(y=2, boundary='pad').mean().set_coords(['u', 'v'])
    # tgofs = gofs.coarsen(lon=3, boundary="pad").mean().coarsen(lat=3, boundary="pad").mean().set_coords(['u', 'v'])
    # tam = am.coarsen(lon=5, boundary="pad").mean().coarsen(lat=5, boundary="pad").mean().set_coords(['u', 'v'])
    # tcmems = cmems.coarsen(lon=2, boundary="pad").mean().coarsen(lat=2, boundary="pad").mean().set_coords(['u', 'v'])

    # # 0m
    # d = 0

    # # U
    # plot_map_with_quiver(trtofs["u"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map_with_quiver(tgofs["u"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map_with_quiver(tam["u"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map_with_quiver(tcmems["u"].sel(depth=0, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # V (Northward Velocity)
    # plot_map_with_quiver(trtofs["v"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="RTOFS", figsize=(20,16), sname=f"{rstr}_rtofs_{d}m")
    # plot_map_with_quiver(tgofs["v"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="GOFS", figsize=(20,16), sname=f"{rstr}_gofs_{d}m")
    # plot_map_with_quiver(tam["v"].sel(depth=0), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", figsize=(20,16), sname=f"{rstr}_amseas_{d}m")
    # plot_map_with_quiver(tcmems["v"].sel(depth=0, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="CMEMS", figsize=(20,16), sname=f"{rstr}_cmems_{d}m")

    # # 100m
    # d = 100

    # # U
    # plot_map_with_quiver(trtofs["u"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map_with_quiver(tgofs["u"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map_with_quiver(tam["u"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map_with_quiver(tcmems["u"].sel(depth=100, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")

    # # V
    # plot_map_with_quiver(trtofs["v"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="RTOFS", figsize=(30,20), sname=f"{rstr}_rtofs_{d}m")
    # plot_map_with_quiver(tgofs["v"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="GOFS", figsize=(30,20), sname=f"{rstr}_gofs_{d}m")
    # plot_map_with_quiver(tam["v"].sel(depth=100), cm.balance, (-1.5, 1.5, .1), title="AMSEAS", figsize=(30,20), sname=f"{rstr}_amseas_{d}m")
    # plot_map_with_quiver(tcmems["v"].sel(depth=100, method='nearest'), cm.balance, (-1.5, 1.5, .1), title="CMEMS", figsize=(30,20), sname=f"{rstr}_cmems_{d}m")