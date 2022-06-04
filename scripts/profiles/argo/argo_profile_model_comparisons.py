import datetime as dt
import os

import hurricanes.configs as configs
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seawater
import xarray as xr
from hurricanes.calc import depth_interpolate, lon180to360, lon360to180
from hurricanes.models import gofs, rtofs
from hurricanes.platforms import get_argo_floats_by_time
from hurricanes.plotting import map_create
from hurricanes.regions import region_config

save_dir = configs.path_plots / 'profiles' / 'argo'

# Configs
# argos = ["4902350", "4903250", "6902854", "4903224", "4903227"]
days = configs.days
dpi = configs.dpi
vars = 'platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal'

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

# Load models
rds = rtofs().sel(depth=slice(0, 400))
gds = gofs(rename=True).sel(depth=slice(0, 400))
# cmems = cmems()

# Get today and yesterday dates
date_list = [dt.date.today() - dt.timedelta(days=x+1) for x in range(days)]
date_list.insert(0, dt.date.today())

now = pd.Timestamp.utcnow()
t0 = pd.to_datetime(now - pd.Timedelta(days, 'd')).tz_localize(None)
t1 = pd.to_datetime(now).tz_localize(None)

# cmems_salinity = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-so', user, pw)
# cmems_temp = grab_cmems('global-analysis-forecast-phy-001-024-3dinst-thetao', user, pw)
# cmems = xr.merge([cmems_salinity, cmems_temp])
# cmems = cmems.sel(depth=slice(0, 400), time=slice(t0, t1),)

# Loop through regions
for item in configs.regions:
    region = region_config(item)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')

    temp_save_dir = save_dir / '_'.join(region['name'].split(' ')).lower()
    os.makedirs(temp_save_dir, exist_ok=True)

    # Download argo floats from ifremer erddap server
    floats = get_argo_floats_by_time(extent, t0, t1, variables=vars)

    if floats.empty:
        print(f"No Argo floats found in {region['name']}")
        continue

    # Iterate through argo float profiles
    for gname, df in floats.reset_index().groupby(['argo', 'time']):
        wmo = gname[0] # wmo id from gname
        ctime = gname[1] # time from gname
        tstr = ctime.strftime("%Y-%m-%d %H:%M:%S") # create time string
        save_str = f'{wmo}-profile-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'
        full_file = temp_save_dir / save_str

        if full_file.is_file():
            print(f"{full_file} already exists. Skipping.")
            continue

        # Calculate depth from pressure and lat
        df = df.assign(
            depth=seawater.eos80.dpth(
                df['pres (decibar)'],
                df['lat']
                )
        )

        # Filter out the dataframe
        salinity_mask = np.abs(stats.zscore(df['psal (PSU)'])) < 3
        depth_mask = df['depth'] <= 400
        df = df[salinity_mask & depth_mask]

        # Calculate density. Use .assign to avoid SettingWithCopyWarning
        df = df.assign(
            density=seawater.eos80.dens(
                df['psal (PSU)'], 
                df['temp (degree_Celsius)'], 
                df['pres (decibar)']
                )
            )

        # Interpolate argo profile to configuration depths
        df = depth_interpolate(df, 
                               depth_var='depth', 
                               depth_min=configs.min_depth,
                               depth_max=configs.max_depth,
                               stride=configs.stride)

        # Grab lon and lat of argo profile
        lon, lat = df['lon'].unique()[-1], df['lat'].unique()[-1]

        # %%
        # For each model, select nearest time to argo time
        # Interpolate to the argo float location and depths
        
        # Calculate depths to interpolate 
        depths_interp = np.arange(
            configs.min_depth, 
            configs.max_depth+configs.stride, 
            configs.stride)

        # GOFS
        gdsp = gds.sel(time=ctime, method='nearest')
        gdsi = gdsp.interp(
            lon=lon180to360(lon), # Convert longitude to 360 convention
            lat=lat,
            depth=xr.DataArray(depths_interp, dims='depth')
            )
        # Convert the lon back to a 180 degree lon
        gdsi['lon'] = lon360to180(gdsi['lon'])

        # Calculate density for gofs profile
        gdsi["pressure"] = xr.apply_ufunc(seawater.eos80.pres, gdsi.depth, gdsi.lat)
        gdsi["density"] = xr.apply_ufunc(seawater.eos80.dens, gdsi.salinity, gdsi.temperature, gdsi.pressure)

        # RTOFS
        # Setting RTOFS lon and lat to their own variables speeds up the script
        rlon = rds.lon.data[0, :]
        rlat = rds.lat.data[:, 0]
        rx = rds.x.data
        ry = rds.y.data

        # interpolating lon and lat to x and y index of the rtofs grid
        rlonI = np.interp(lon, rlon, rx)
        rlatI = np.interp(lat, rlat, ry)

        rdsp = rds.sel(time=ctime, method='nearest')
        rdsi = rdsp.interp(
            x=rlonI,
            y=rlatI,
            depth=xr.DataArray(depths_interp, dims='depth')
        )
        # Calculate density for rtofs profile
        rdsi['pressure'] = xr.apply_ufunc(seawater.eos80.pres, rdsi.depth, rdsi.lat)
        rdsi['density'] = xr.apply_ufunc(seawater.eos80.dens, rdsi.salinity, rdsi.temperature, rdsi.pressure)

        # # cmems_sub = cmems.sel(longitude=x, latitude=y, time=t_float, method='nearest')
        # grid = """
        # TTSSDDGG
        # TTSSDDMM
        # TTSSDDLL
        # """
        # # try:
        # #     figsize = region['figure']['figsize']
        # # except KeyError:
        # figsize = (7,4)

        # fig, _ = plt.subplot_mosaic(
        #     grid,
        #     figsize=figsize,
        #     layout="constrained",
        #     gridspec_kw={
        #         # set the height ratios between the rows
        #         # "height_ratios": [4, 1],
        #         # set the width ratios between the columns
        #         # # "width_ratios": [1],
        #         },
        #     )

        # fig = plt.figure(figsize=(14, 8))
        # plt.rcParams['figure.constrained_layout.use'] = True
        # grid = plt.GridSpec(6, 10, hspace=0.2, wspace=0.2, figure=fig)
        # ax1 = plt.subplot(grid[:, :3])  # Temperature
        # ax2 = plt.subplot(grid[:, 4:7], sharey=ax1)  # Salinity
        # ax3 = plt.subplots(grid[:, :], sharey=)
        # ax3 = plt.subplot(grid[0, 7:])  # Plot Title
        # ax4 = plt.subplot(grid[2:4, 7:], projection=projection['map'])  # Map
        # ax5 = plt.subplot(grid[5, 7:])  # Legend for profile plots

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
        ax5 = fig.add_subplot(gs[1, -1], projection=configs.projection['map']) # Map
        ax6 = fig.add_subplot(gs[2, -1]) # Legend

        alon = round(lon, 2)
        alat = round(lat, 2)
        glon = gdsi.lon.data.round(2)
        glat = gdsi.lat.data.round(2)
        rlon = rdsi.lon.data.round(2)
        rlat = rdsi.lat.data.round(2)
        alabel = f'{wmo} [{alon}, {alat}]'
        glabel = f'GOFS [{ glon }, { glat }]'
        rlabel = f'RTOFS [{ rlon }, { rlat }]'

        # Temperature 
        ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
        ax1.plot(gdsi['temperature'], gdsi['depth'], 'g-o', label=glabel)
        ax1.plot(rdsi['temperature'], rdsi['depth'], 'r-o', label=rlabel)
        # ax1.plot(cmems_sub['thetao'].squeeze(), cmems_sub['depth'].squeeze(),
        #            'm-o',
        #            label=f'CMEMS [{ cmems_sub.longitude.data.round(2) }, { cmems_sub.latitude.data.round(2) }]')

        ax1.set_ylim([400, 0])
        ax1.grid(True, linestyle='--', linewidth=.5)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

        # Salinity
        ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
        ax2.plot(gdsi['salinity'], gdsi['depth'],'g-o', label=glabel)
        ax2.plot(rdsi['salinity'], rdsi['depth'], 'r-o', label=rlabel)
        # ax2.plot(cmems_sub['so'].squeeze(), cmems_sub['depth'].squeeze(),
        #            'm-o',
        #            label=f'CMEMS [{ cmems_sub.longitude.data.round(2) }, { cmems_sub.latitude.data.round(2) }]')

        ax2.set_ylim([400, 0])
        ax2.grid(True, linestyle='--', linewidth=.5)
        ax2.tick_params(axis='both', labelsize=13)
        ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
        # ax2.set_ylabel('Depth (m)', fontsize=14)

        # Density
        ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
        ax3.plot(gdsi['density'], gdsi['depth'],'g-o', label=glabel)
        ax3.plot(rdsi['density'], rdsi['depth'], 'r-o', label=rlabel)
        # ax2.plot(cmems_sub['so'].squeeze(), cmems_sub['depth'].squeeze(),
        #            'm-o',
        #            label=f'CMEMS [{ cmems_sub.longitude.data.round(2) }, { cmems_sub.latitude.data.round(2) }]')

        ax3.set_ylim([400, 0])
        ax3.grid(True, linestyle='--', linewidth=.5)
        ax3.tick_params(axis='both', labelsize=13)
        ax3.set_xlabel('Density', fontsize=14, fontweight='bold')
        # ax3.set_ylabel('Depth (m)', fontsize=14)

        text = ax4.text(0.125, 1.0, 
                        f'Argo #{wmo}\n'
                        f'ARGO:  { tstr }\n'
                        f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
                        f'GOFS:  {pd.to_datetime(gdsi.time.data)}',
                        ha='left', va='top', size=15, fontweight='bold')

        text.set_path_effects([path_effects.Normal()])
        ax4.set_axis_off()

        map_create(extent, ax=ax5, ticks=False)
        ax5.plot(lon, lat, 'ro', transform=configs.projection['data'])

        h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

        ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
        ax6.set_axis_off()
        
        plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')


        fig.tight_layout()
        fig.subplots_adjust(top=0.9) 

        plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()