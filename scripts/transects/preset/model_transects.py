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
from cool_maps.plot import add_ticks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cool_maps.plot import create, add_ticks
from ioos_model_comparisons.models import (rtofs as r, 
                                           cmems as c, 
                                           amseas as a, 
                                           gofs as g)

# %%
# Select time we want to look at
t0 = pd.Timestamp(2022, 11, 1, 0, 0, 0)
t1 = pd.Timestamp(2022, 11, 2, 0, 0, 0)
times = pd.date_range(t0, t1, freq='10H')
# times = [pd.Timestamp(2022, 9, 25, 12)]

# Select models
include_rtofs = True
include_gofs = True
include_cmems = True
include_amseas = True

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
depth_limits = [0, 1000] #min, max

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

# # Mid Atlantic Bight
# lons = [-77, -67]
# lats = [35, 43]
# title = "Mid Atlantic Bight (Endurance Line)"
# rstr = "mid_atlantic_bight"
# xaxis = 'lon'
# start = -81.2, 30
# end = -77.8, 30
# start = -74.3, 39.52
# end = -71.85, 38.96
# points = 500
# depth_spacing = 1

# Contour levels
levels_salinity = np.arange(35, 37, .1)
levels_temperature = np.arange(10, 28, 1)
levels_u = np.arange(-1.0, 1.0, .1)
levels_v = np.arange(-1.0, 1.0, .1)
# levels_density = np.arange(1023, 1030, 1)
levels_density = np.arange(1023, 1027, .5)


# Sort the lons and lats so they are in monotonically increasing order 
lons.sort()
lats.sort()

# Convert lons (-180, 180) into lons (0, 36
glons = np.mod(lons, 360)

# convert depth limits to a slice
depth_slice = slice(depth_limits[0], depth_limits[1])


def transect2rtofs(pts, grid_lons, grid_lats, grid_x, grid_y):
    # if not grid_x:
    #     grid_x = np.arange(0, len(grid_lons))
    # if not grid_y:
    #     grid_y = np.arange(0, len(grid_lats))
    
    # Convert points to x and y index for rtofs
    # Use piecewise linear interpolation (np.interp) to find the partial index (float instead of an int) of the points (lon,lat) we are calculating would lie
    # on the x,y grid for RTOFS
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

def plot_transect(ds, var, cmap, levels=None, xaxis="lon", title=None, ax=None, cbar=False, fontfrac=1, contour=None, xlabel=None, ylabel=None, sname=None, dpi=300, cbar_label=None):
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

        if cbar_label:
            cb.set_label(label=cbar_label, size=14*fontfrac, weight='bold')
            cb.ax.tick_params(labelsize=13*fontfrac)
        else:
            cb.set_label(label=f"{var.name.title()} ({var_units})", size=14*fontfrac, weight='bold')
            cb.ax.tick_params(labelsize=13*fontfrac)

    ax.axvline(-85.87044879876521, color='red', linestyle='-.', linewidth=5)

    ax.set_ylim([1000, 0])
    plt.setp(ax.get_xticklabels(), fontsize=16*fontfrac)
    plt.setp(ax.get_yticklabels(), fontsize=16*fontfrac)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20*fontfrac, fontweight="bold")
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20*fontfrac, fontweight="bold")
    
    if title:
        ax.set_title(title, fontsize=24*fontfrac, fontweight="bold")
        
    if sname:
        sname = f"{sname}_{var.name}_{ctimestr}.png"
        plt.savefig(sname, dpi=dpi, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
    return ax

if include_rtofs:
    # Load and munge RTOFS
    # # Use the xarray open_dataset function to access the dataset via opendap. Set lon and lat to coordinates for easy plotting and select the center time above and grab only the top 1000m
    rtofs = r().sel(depth=depth_slice)

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

if include_gofs:
    # Load and munge GOFS
    gofs = g(rename=True).sel(depth=depth_slice)
    gofs["lon"] = np.mod(gofs["lon"]+180, 360)-180

if include_amseas:
    # Load and munge AMSEAS
    am = a(rename=True).sel(depth=depth_slice)
    am["lon"] = np.mod(am["lon"]+180, 360)-180

if include_cmems:
    # Load Copernicus data
    cmems = c(rename=True).sel(depth=depth_slice, lon=slice(lons[0], lons[1]), lat=slice(lats[0], lats[1]))

# Calculate transect line between the start point and end point
# returns all points as a numpy array with lon in the first column and lat in the second column
pts = calculate_transect(start, end, points)

# Iterate through times
times = [pd.Timestamp(2022, 11, 1, 12, 0, 0)]

for ctime in times:
    ctime_str = ctime.strftime("%Y-%m-%d %H:%M:%S")
    ctime_save_str = ctime.strftime("%Y-%m-%dT%H%M%SZ")

    if include_rtofs:
        # Select ctime with RTOFS
        trtofs = rtofs.sel(time=ctime)

        # Use xr.apply_ufunc to apply a function over all dimensions
        # trtofs["pressure"] = xr.apply_ufunc(seawater.eos80.pres, trtofs.depth, trtofs.lat)
        # trtofs["density"] = xr.apply_ufunc(seawater.eos80.dens, trtofs.salinity, trtofs.temperature, trtofs.pressure)

        # Convert to the x, y indexes so we can work with the RTOFS model
        grid_lons = trtofs.lon.values[0,:]
        grid_lats = trtofs.lat.values[:,0]
        grid_x = trtofs.x.values
        grid_y = trtofs.y.values
        lonidx, latidx = transect2rtofs(pts, grid_lons, grid_lats, grid_x=grid_x, grid_y=grid_y)

        # Interpolate RTOFS Transect
        rds = trtofs.interp(
            x=xr.DataArray(lonidx, dims="point"),
            y=xr.DataArray(latidx, dims="point"),
            depth=xr.DataArray(np.arange(0, trtofs.depth.max()+depth_spacing, depth_spacing), dims="depth")
        )

    if include_gofs:
        # GOFS
        tgofs = gofs.sel(time=ctime)
        # tgofs["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tgofs.depth, tgofs.lat)
        # tgofs["density"] = xr.apply_ufunc(seawater.eos80.dens, tgofs.salinity, tgofs.temperature, tgofs.pressure)

        # Interpolate GOFS Transect
        gds = tgofs.interp(
            lon=xr.DataArray(pts[:,0], dims="point"),
            lat=xr.DataArray(pts[:,1], dims="point"),
            depth=xr.DataArray(np.arange(0, tgofs.depth.max()+depth_spacing, depth_spacing), dims="depth")
        )

    if include_amseas:
        # AMSEAS
        tam = am.sel(time=ctime)
        # tam["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tam.depth, tam.lat)
        # tam["density"] = xr.apply_ufunc(seawater.eos80.dens, tam.salinity, tam.temperature, tam.pressure)

        # Interpolate AMSEAS Transect
        ads = tam.interp(
            lon=xr.DataArray(pts[:,0], dims="point"),
            lat=xr.DataArray(pts[:,1], dims="point"),
            depth=xr.DataArray(np.arange(0, tam.depth.max()+depth_spacing, depth_spacing), dims="depth")
        )

    if include_cmems:
        # Copernicus
        tcmems = cmems.sel(time=ctime, method='nearest')
        # tcmems["pressure"] = xr.apply_ufunc(seawater.eos80.pres, tcmems.depth, tcmems.lat)
        # tcmems["density"] = xr.apply_ufunc(seawater.eos80.dens, tcmems.salinity, tcmems.temperature, tcmems.pressure)

        # Interpolate Copernicus Transect
        cds = tcmems.interp(
            lon=xr.DataArray(pts[:,0], dims="point"),
            lat=xr.DataArray(pts[:,1], dims="point"),
            depth=xr.DataArray(np.arange(0, tcmems.depth.max()+depth_spacing, depth_spacing), dims="depth")
        )

    import cool_maps.plot as cplt
    import cartopy.feature as cfeature
    from cartopy.io.shapereader import Reader


    def map_add_eez(ax, zorder=1, color='white', linewidth=0.75):
        shape_feature = cfeature.ShapelyFeature(
            Reader('/Users/mikesmith/Documents/data/eez/World_EEZ_v11_20191118/eez_boundaries_v11.shp').geometries(), 
            ccrs.PlateCarree(),
            linestyle='-.',
            linewidth=linewidth,
            edgecolor=color, 
            facecolor='none'
            )
        h = ax.add_feature(shape_feature, zorder=zorder)
        return h
    
    # Create map plot
    map = ccrs.Mercator()
    proj = ccrs.PlateCarree()
    extent_map = [-87.5, -84.5, 19.5, 23]

    # fig, ax = plt.subplots(1, 2, figsize=(16,8), subplot_kw=dict(projection=map))
    # cplt.create(extent=extent_map, ax=ax[0], gridlines=True)
    # trtofs['temperature'].sel(depth=0).plot.contourf(x='lon', y='lat', ax=ax[0], cmap=cm.thermal, levels=np.arange(27, 29.5, .25), extend='both', transform=proj)
    # ax[0].plot(pts[:,0], pts[:,1], 'k-.', linewidth=4, transform=proj)
    # map_add_eez(ax[0], color='red', linewidth=1.5)

    # cplt.create(extent=extent_map, ax=ax[1], gridlines=True)
    # trtofs['salinity'].sel(depth=0).plot.contourf(x='lon', y='lat', ax=ax[1], cmap=cm.haline, levels=np.arange(35.8, 36.2, .05), extend='both', transform=proj)
    # ax[1].plot(pts[:,0], pts[:,1], 'k-.', linewidth=4, transform=proj)
    # map_add_eez(ax[1], color='red', linewidth=1.5)
    # plt.suptitle('RTOFS', fontweight='bold', fontsize=20)
    from oceans.ocfis import uv2spdir

    tds = trtofs.sel(depth=0)
    _, speed = uv2spdir(tds.u, tds.v)
    tds['speed'] = (('y', 'x'), speed)
    tds['speed'] = tds['speed']*100
    
    
    fig, ax = plt.subplots(figsize=(16,8), subplot_kw=dict(projection=map))
    cplt.create(extent=extent_map, ax=ax, gridlines=True)
    tds['speed'].plot.contourf(x='lon', y='lat', ax=ax, cmap=cm.speed, levels=np.arange(0, 170, 10), extend='both', transform=proj)
    ax.plot(pts[:,0], pts[:,1], 'k-.', linewidth=4, transform=proj)
    map_add_eez(ax, color='red', linewidth=1.5)

    plt.savefig(f'/Users/mikesmith/Documents/{rstr}_transect_maps_1000m_{ctime_save_str}.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    if xaxis =='lon':
        xlabel = "Longitude"
    elif xaxis == 'lat':
        xlabel = "Latitude"

    # fig, ax = plt.subplots(4, 3, figsize=(40,30))

    # if include_rtofs:
    #     plot_transect(rds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, title="Temperature", ax=ax[0,0], cbar=dict(ax=ax[0,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="RTOFS\nDepth (m)")
    #     plot_transect(rds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, title="Salinity", ax=ax[0,1], cbar=dict(ax=ax[0,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    #     plot_transect(rds, "density", cm.dense, levels=levels_density, xaxis=xaxis, title="Density", ax=ax[0,1], cbar=dict(ax=ax[0,1], orientation='vertical', pad=0.01), fontfrac=1.5)

    # if include_gofs:
    #     plot_transect(gds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[1,0], cbar=dict(ax=ax[1,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="GOFS\nDepth (m)")
    #     plot_transect(gds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[1,1], cbar=dict(ax=ax[1,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    #     plot_transect(gds, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[1,1], cbar=dict(ax=ax[1,1], orientation='vertical', pad=0.01), fontfrac=1.5)

    # if include_amseas:
    #     plot_transect(ads, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[2,0], cbar=dict(ax=ax[2,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="AMSEAS\nDepth (m)")
    #     plot_transect(ads, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[2,1], cbar=dict(ax=ax[2,1], orientation='vertical', pad=0.01), fontfrac=1.5)
    #     plot_transect(ads, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[2,1], cbar=dict(ax=ax[2,1], orientation='vertical', pad=0.01), fontfrac=1.5)

    # if include_cmems:
    #     plot_transect(cds, "temperature", cm.thermal, levels=levels_temperature, xaxis=xaxis, ax=ax[3,0], cbar=dict(ax=ax[3,0], orientation='vertical', pad=0.01), fontfrac=1.5, ylabel="Copernicus\nDepth (m)", xlabel=xlabel)
    #     plot_transect(cds, "salinity", cm.haline, levels=levels_salinity, xaxis=xaxis, ax=ax[3,1], cbar=dict(ax=ax[3,1], orientation='vertical', pad=0.01), fontfrac=1.5, xlabel=xlabel)
    #     plot_transect(cds, "density", cm.dense, levels=levels_density, xaxis=xaxis, ax=ax[3,1], cbar=dict(ax=ax[3,1], orientation='vertical', pad=0.01), fontfrac=1.5, xlabel=xlabel)

    # plt.suptitle(f"{title}\n{ctime_str}", fontsize=32, fontweight="bold")
    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.1, 
    #                     top=0.94, 
    #                     wspace=0.005, 
    #                     hspace=0.1)
    # plt.savefig(f'{rstr}_tsd_{ctime_save_str}.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    
    fig, ax = plt.subplots(4, 2, figsize=(40,30))
    if include_rtofs:
        plot_transect(rds, "u", cm.balance, levels=levels_u, xaxis=xaxis, title="Eastward Velocity", ax=ax[0,0], cbar=dict(ax=ax[0,0], orientation='vertical', pad=0.01), cbar_label='U (m/s)', fontfrac=1.5, ylabel="RTOFS\nDepth (m)")
        plot_transect(rds, "v", cm.balance, levels=levels_v, xaxis=xaxis, title="Northward Velocity", ax=ax[0,1], cbar=dict(ax=ax[0,1], orientation='vertical', pad=0.01), cbar_label='V (m/s)', fontfrac=1.5)

    if include_gofs:
        plot_transect(gds, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[1,0], cbar=dict(ax=ax[1,0], orientation='vertical', pad=0.01), cbar_label='U (m/s)', fontfrac=1.5, ylabel="GOFS\nDepth (m)")
        plot_transect(gds, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[1,1], cbar=dict(ax=ax[1,1], orientation='vertical', pad=0.01), cbar_label='V (m/s)', fontfrac=1.5)

    if include_amseas:
        plot_transect(ads, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[2,0], cbar=dict(ax=ax[2,0], orientation='vertical', pad=0.01), cbar_label='U (m/s)', fontfrac=1.5, ylabel="AMSEAS\nDepth (m)")
        plot_transect(ads, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[2,1], cbar=dict(ax=ax[2,1], orientation='vertical', pad=0.01), cbar_label='V (m/s)', fontfrac=1.5)

    if include_cmems:
        plot_transect(cds, "u", cm.balance, levels=levels_u, xaxis=xaxis, ax=ax[3,0], cbar=dict(ax=ax[3,0], orientation='vertical', pad=0.01), cbar_label='U (m/s)', fontfrac=1.5, ylabel="Copernicus\nDepth (m)", xlabel=xlabel)
        plot_transect(cds, "v", cm.balance, levels=levels_v, xaxis=xaxis, ax=ax[3,1], cbar=dict(ax=ax[3,1], orientation='vertical', pad=0.01), cbar_label='V (m/s)', fontfrac=1.5, xlabel=xlabel)

    plt.suptitle(f"{title} - {ctime_str}", fontsize=40, fontweight="bold")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        top=0.94, 
                        wspace=0.005, 
                        hspace=0.1)
    plt.savefig(f'{rstr}_velocities_{ctime_save_str}.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close()