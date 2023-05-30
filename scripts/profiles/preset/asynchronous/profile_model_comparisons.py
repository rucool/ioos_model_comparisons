import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seawater
import xarray as xr
from ioos_model_comparisons.calc import depth_interpolate, lon180to360, lon360to180
from ioos_model_comparisons.models import gofs, rtofs, cmems, amseas
# from ioos_model_comparisons.plotting import map_create, map_add_ticks
from cool_maps.plot import create, add_ticks
from ioos_model_comparisons.regions import region_config
import ioos_model_comparisons.configs as conf
from pathlib import Path

save_dir = conf.path_plots / 'profiles' / 'preset'

dpi = conf.dpi
depths = slice(0, 1000)
ctime = pd.Timestamp(2023, 5, 17, 6, 0, 0)
temp_save_dir = Path('/Users/mikesmith/Documents/')

# # St Lucia Passage
# lat = 14 + 15.669/60
# lon = -60 - 58.580/60

# # St Vincent Passage
# lat = 13 + 34.002/60
# lon = -61 - 11.456/60

# # Furthest South WPT
# lat = 12 + 35.593/60
# lon = -61 - 36.940/60

# Western Carib -north 
lat = 13
lon = -78

# # Western Carib - South 
lat = 20
lon = -83

extent = region_config('caribbean')['extent']

include_rtofs = True
include_gofs = True
include_copernicus = False
include_amseas = True

tstr = ctime.strftime("%Y-%m-%d %H:%M:%S") # create time string
save_str = f'profile-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'
full_file = temp_save_dir / save_str
diff_file = temp_save_dir / f'profile-difference-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'

# Check if profile exists already
if full_file.is_file():
    print(f"{full_file} already exists. Checking if difference plot has been generated.")
    profile_exist = True
else:
    print(f"Processing profile that occured at {ctime}")
    # Profile does not exist yet
    profile_exist = False   

# Calculate depths to interpolate 
depths_interp = np.arange(
    conf.min_depth, 
    conf.max_depth+conf.stride, 
    conf.stride)

# Load models
if include_rtofs:
    rds = rtofs().sel(depth=depths)

    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlons = rds.lon.data[0, :]
    rlats = rds.lat.data[:, 0] 
    rx = rds.x.data
    ry = rds.y.data
    
    # interpolating lon and lat to x and y index of the rtofs grid
    rlonI = np.interp(lon, rlons, rx)
    rlatI = np.interp(lat, rlats, ry)

    rdsp = rds.sel(time=ctime, method='nearest')
    rdsi = rdsp.interp(
        x=rlonI,
        y=rlatI,
        # depth=xr.DataArray(depths_interp, dims='depth')
    )
    
    # Calculate density for rtofs profile
    rdsi['pressure'] = xr.apply_ufunc(seawater.eos80.pres, rdsi.depth, rdsi.lat)
    rdsi['density'] = xr.apply_ufunc(seawater.eos80.dens, rdsi.salinity, rdsi.temperature, rdsi.pressure)


if include_gofs:
    gds = gofs(rename=True).sel(depth=depths)
    gdsp = gds.sel(time=ctime, method='nearest')
    gdsi = gdsp.interp(
        lon=lon180to360(lon), # Convert longitude to 360 convention
        lat=lat,
        # depth=xr.DataArray(depths_interp, dims='depth')
        )
    # Convert the lon back to a 180 degree lon
    gdsi['lon'] = lon360to180(gdsi['lon'])

    # Calculate density for gofs profile
    gdsi["pressure"] = xr.apply_ufunc(seawater.eos80.pres, gdsi.depth, gdsi.lat)
    gdsi["density"] = xr.apply_ufunc(seawater.eos80.dens, gdsi.salinity, gdsi.temperature, gdsi.pressure)

if include_copernicus:
    cds = cmems(rename=True).sel(depth=depths)
    cdsp = cds.sel(time=ctime, method='nearest')
    # cdsi = cdsp.interp(
    #     lon=lon, 
    #     lat=lat,
    #     depth=xr.DataArray(depths_interp, dims='depth')
    #     )
    cdsi = cdsp.sel(
        lon=lon, 
        lat=lat,
        method='nearest'
        # depth=xr.DataArray(depths_interp, dims='depth')
        )
    
    
    # Calculate density for rtofs profile
    cdsi['pressure'] = xr.apply_ufunc(seawater.eos80.pres, cdsi.depth, cdsi.lat)
    cdsi['density'] = xr.apply_ufunc(seawater.eos80.dens, cdsi.salinity, cdsi.temperature, cdsi.pressure)

if include_amseas:
    ads = amseas(rename=True).sel(depth=depths)
    adsp = ads.sel(time=ctime, method='nearest')
    adsi = adsp.interp(
                lon=lon180to360(lon), # Convert longitude to 360 convention
                lat=lat,
                # depth=xr.DataArray(depths_interp, dims='depth')
                )
        
    # Convert the lon back to a 180 degree lon
    adsi['lon'] = lon360to180(adsi['lon'])

    # Calculate density for gofs profile
    adsi["pressure"] = xr.apply_ufunc(seawater.eos80.pres, adsi.depth, adsi.lat)
    adsi["density"] = xr.apply_ufunc(seawater.eos80.dens, adsi.salinity, adsi.temperature, adsi.pressure)

# Plot the argo profile
if not profile_exist:
    fig = plt.figure(constrained_layout=True, figsize=(16, 6))
    widths = [1, 1, 1, 1.5]
    heights = [1, 2, 1]

    gs = fig.add_gridspec(3, 4, width_ratios=widths,
                            height_ratios=heights)

    ax1 = fig.add_subplot(gs[:, 0]) # Temperature
    ax2 = fig.add_subplot(gs[:, 1], sharey=ax1)  # Salinity
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax3 = fig.add_subplot(gs[:, 2], sharey=ax1) # Density
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax4 = fig.add_subplot(gs[0, -1]) # Title
    ax5 = fig.add_subplot(gs[1, -1], projection=conf.projection['map']) # Map
    ax6 = fig.add_subplot(gs[2, -1]) # Legend

    alon = round(lon, 2)
    alat = round(lat, 2)
    glon = gdsi.lon.data.round(2)
    glat = gdsi.lat.data.round(2)
    rlon = rdsi.lon.data.round(2)
    rlat = rdsi.lat.data.round(2)

    if include_copernicus:
        clon = cdsi.lon.data.round(2)
        clat = cdsi.lat.data.round(2)
        clabel = f"Copernicus [{ clon }, { clat }]"

    if include_amseas:
        alon = adsi.lon.data.round(2)
        alat = adsi.lat.data.round(2)
        alabel = f"AMSEAS [{ alon }, { alat }]"

    # Legend labels
    # alabel = f'{wmo} [{alon}, {alat}]'
    glabel = f'GOFS [{ glon }, { glat }]'
    rlabel = f'RTOFS [{ rlon }, { rlat }]'

    # Temperature 
    # ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
    ax1.plot(gdsi['temperature'], gdsi['depth'], 'g-o', label=glabel)
    ax1.plot(rdsi['temperature'], rdsi['depth'], 'r-o', label=rlabel)
    # ax1.plot(cdsi['temperature'], cdsi['depth'], 'm-o', label=clabel)
    ax1.plot(adsi['temperature'], adsi['depth'], 'k-o', label=alabel)

    ax1.set_ylim([400, 0])
    ax1.grid(True, linestyle='--', linewidth=.5)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

    # Salinity
    # ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
    ax2.plot(gdsi['salinity'], gdsi['depth'],'g-o', label=glabel)
    ax2.plot(rdsi['salinity'], rdsi['depth'], 'r-o', label=rlabel)
    # ax2.plot(cdsi['salinity'], cdsi['depth'],  'm-o', label=clabel)
    ax2.plot(adsi['salinity'], adsi['depth'],  'k-o', label=alabel)

    ax2.set_ylim([400, 0])
    ax2.grid(True, linestyle='--', linewidth=.5)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Depth (m)', fontsize=14)

    # Density
    # ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
    ax3.plot(gdsi['density'], gdsi['depth'],'g-o', label=glabel)
    ax3.plot(rdsi['density'], rdsi['depth'], 'r-o', label=rlabel)
    # ax3.plot(cdsi['density'], cdsi['depth'], 'm-o', label=clabel)
    ax3.plot(adsi['density'], adsi['depth'], 'k-o', label=alabel)

    ax3.set_ylim([400, 0])
    ax3.grid(True, linestyle='--', linewidth=.5)
    ax3.tick_params(axis='both', labelsize=13)
    ax3.set_xlabel('Density', fontsize=14, fontweight='bold')
    # ax3.set_ylabel('Depth (m)', fontsize=14)

    text = ax4.text(0.125, 1.0, 
                    f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
                    f'GOFS : {pd.to_datetime(gdsi.time.data)}\n'
                    # f'CMEMS: {pd.to_datetime(cdsi.time.data)}\n'
                    f'AMSEAS: {pd.to_datetime(adsi.time.data)}\n',
                    ha='left', va='top', size=15, fontweight='bold')

    text.set_path_effects([path_effects.Normal()])
    ax4.set_axis_off()

    # map_create(extent, ax=ax5, ticks=False)
    create(extent, ax=ax5, ticks=False)
    add_ticks(ax5, extent, fontsize=10)
    ax5.plot(lon, lat, 'ro', transform=conf.projection['data'])

    h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

    ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
    ax6.set_axis_off()
    
    # plt.figtext(0.15, 0.001, f'Depths interpolated to every {conf.stride}m', ha="center", fontsize=10, fontstyle='italic')


    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 

    plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()