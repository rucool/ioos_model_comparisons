import datetime as dt
import os

import ioos_model_comparisons.configs as configs
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seawater
import xarray as xr
from ioos_model_comparisons.calc import depth_interpolate, lon180to360, lon360to180, difference
from ioos_model_comparisons.models import rtofs, gofs, cmems
from ioos_model_comparisons.platforms import get_argo_floats_by_time
from ioos_model_comparisons.plotting import map_create
from ioos_model_comparisons.regions import region_config

save_dir = configs.path_plots / 'profiles' / 'argo' / 'rtofs_comparisons'

# Configs
# argos = ["4902350", "4903250", "6902854", "4903224", "4903227"]
# argos = [4903250, 3901861, 6903137]
argos = [4903227]
days = 4
dpi = configs.dpi
depths = slice(0, 400)
vars = 'platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal'

# initialize keyword arguments for map plot
kwargs = dict()
kwargs['model'] = 'rtofs'
kwargs['save_dir'] = save_dir
kwargs['dpi'] = dpi

# Load models
r20 = rtofs().sel(depth=depths)
# r20 = xr.open_mfdataset('/Users/mikesmith/Documents/data/rtofs/2.0/*/*.nc').rename({"MT": 'time', "Depth": "depth", "Y": "y", "X": "x", "Latitude": "lat", "Longitude": "lon"})
r22 = xr.open_mfdataset('/Users/mikesmith/Documents/data/rtofs/2.2/rtofs.20220621/*.nc').rename({"MT": 'time', "Depth": "depth", "Y": "y", "X": "x", "Latitude": "lat", "Longitude": "lon"})
g31 = gofs(rename=True)
cp = cmems(rename=True)

t0 = pd.to_datetime(r22.time.min().values) - dt.timedelta(days=1)
t1 = r22.time.max().values

# Loop through regions
# for item in configs.regions:
for item in ["gom", "caribbean"]:
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

        if wmo in argos:
            ctime = gname[1] # time from gname
            tstr = ctime.strftime("%Y-%m-%d %H:%M:%S") # create time string
            save_str = f'{wmo}-profile-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'
            full_file = temp_save_dir / save_str
            diff_file = temp_save_dir / f'{wmo}-profile-difference-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png'

            profile_exist = False
            profile_diff_exist = False 

            # Check if profile exists already
            if full_file.is_file():
                print(f"{full_file} already exists. Checking if difference plot has been generated.")
                
                profile_exist = True
                
                # Check if the difference profile has been plotted.
                if diff_file.is_file():
                    # Profile difference exists. Continue to the next float
                    profile_diff_exist = True
                else:
                    # Profile difference does not exist yet
                    profile_diff_exist = False
            else:
                # Profile does not exist yet
                profile_exist = False   


            if not (profile_exist) or not (profile_diff_exist):
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

                # Check if the dataframe has any data after filtering.
                if df.empty:
                    continue
                
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
                    configs.stride
                    )

                # RTOFS 2.0
                # Setting RTOFS lon and lat to their own variables speeds up the script
                rlon = r20.lon.data[0, :]
                rlat = r20.lat.data[:, 0]
                rx = r20.x.data
                ry = r20.y.data

                # interpolating lon and lat to x and y index of the rtofs grid
                rlonI = np.interp(lon, rlon, rx)
                rlatI = np.interp(lat, rlat, ry)

                r20i = r20.sel(time=ctime, method='nearest').interp(
                    x=rlonI,
                    y=rlatI,
                    depth=xr.DataArray(depths_interp, dims='depth')
                )
                # Calculate density for rtofs profile
                r20i['pressure'] = xr.apply_ufunc(seawater.eos80.pres, r20i.depth, r20i.lat)
                r20i['density'] = xr.apply_ufunc(seawater.eos80.dens, r20i.salinity, r20i.temperature, r20i.pressure)

                # RTOFS 2.2
                r22i = r22.sel(time=ctime, method='nearest').interp(
                    x=rlonI,
                    y=rlatI,
                    depth=xr.DataArray(depths_interp, dims='depth')
                ).load()
                
                # Calculate density for rtofs profile
                r22i['pressure'] = xr.apply_ufunc(seawater.eos80.pres, r22i.depth, r22i.lat)
                r22i['density'] = xr.apply_ufunc(seawater.eos80.dens, r22i.salinity, r22i.temperature, r22i.pressure)

                # GOFS 3.1
                g31i = g31.sel(time=ctime, method='nearest').interp(
                    lon=lon180to360(lon), # Convert longitude to 360 convention
                    lat=lat,
                    depth=xr.DataArray(depths_interp, dims='depth')

                    )
                # Convert the lon back to a 180 degree lon
                g31i['lon'] = lon360to180(g31i['lon'])

                # Calculate density for gofs profile
                g31i["pressure"] = xr.apply_ufunc(seawater.eos80.pres, g31i.depth, g31i.lat)
                g31i["density"] = xr.apply_ufunc(seawater.eos80.dens, g31i.salinity, g31i.temperature, g31i.pressure)

                # Copernicus
                cpi = cp.sel(time=ctime, method="nearest").interp(
                    lon=lon,
                    lat=lat,
                    depth=xr.DataArray(depths_interp, dims='depth')
                )
                cpi["pressure"] = xr.apply_ufunc(seawater.eos80.pres, g31i.depth, g31i.lat)
                cpi["density"] = xr.apply_ufunc(seawater.eos80.dens, g31i.salinity, g31i.temperature, g31i.pressure)
            
                alon = round(lon, 2)
                alat = round(lat, 2)

                # Legend labels
                alabel = f'{wmo}'
                r20label = f'RTOFS 2.0'
                r22label = f'RTOFS 2.2'
                g31label = f"GOFS 3.1"
                cplabel = f"CMEMS"
                
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
                ax5 = fig.add_subplot(gs[1, -1], projection=configs.projection['map']) # Map
                ax6 = fig.add_subplot(gs[2, -1]) # Legend

                # Temperature 
                ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
                ax1.plot(g31i['temperature'], g31i['depth'], 'g-o', label=g31label)
                ax1.plot(r20i['temperature'], r20i['depth'], 'r-o', label=r20label)
                ax1.plot(r22i['temperature'], r22i['depth'], 'r-o', alpha=0.5, label=r22label)
                ax1.plot(cpi["temperature"], cpi["depth"], 'm-o', label=cplabel)

                ax1.set_ylim([400, 0])
                ax1.grid(True, linestyle='--', linewidth=.5)
                ax1.tick_params(axis='both', labelsize=13)
                ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

                # Salinity
                ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
                ax2.plot(g31i['salinity'], g31i['depth'], 'g-o', label=g31label)
                ax2.plot(r20i['salinity'], r20i['depth'], 'r-o', label=r20label)
                ax2.plot(r22i['salinity'], r22i['depth'], 'r-o', alpha=0.5, label=r22label)
                ax2.plot(cpi["salinity"], cpi["depth"], 'm-o', label=cplabel)

                ax2.set_ylim([400, 0])
                ax2.grid(True, linestyle='--', linewidth=.5)
                ax2.tick_params(axis='both', labelsize=13)
                ax2.set_xlabel('Salinity', fontsize=14, fontweight='bold')
                # ax2.set_ylabel('Depth (m)', fontsize=14)

                # Density
                ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
                ax3.plot(g31i['density'], g31i['depth'], 'g-o', label=g31label)
                ax3.plot(r20i['density'], r20i['depth'], 'r-o', label=r20label)
                ax3.plot(r22i['density'], r22i['depth'], 'r-o', alpha=0.5, label=r22label)
                ax3.plot(cpi["density"], cpi["depth"], 'm-o', label=cplabel)

                ax3.set_ylim([400, 0])
                ax3.grid(True, linestyle='--', linewidth=.5)
                ax3.tick_params(axis='both', labelsize=13)
                ax3.set_xlabel('Density', fontsize=14, fontweight='bold')

                text = ax4.text(0.125, 1.0, 
                                f'Argo #{wmo} [{alon}, {alat}]\n'
                                f'ARGO:  { tstr }\n'
                                f'CMEMS: {pd.to_datetime(cpi.time.data)}\n'
                                f'GOFS 3.1: {pd.to_datetime(g31i.time.data)}\n'
                                f'RTOFS 2.0: {pd.to_datetime(r20i.time.data)}\n'
                                f'RTOFS 2.2: {pd.to_datetime(r22i.time.data)}\n',
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

            # Plot the profile differences 
            if not profile_diff_exist:
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

                # Temperature 
                diff_g = difference(g31i['temperature'], df['temp (degree_Celsius)'])
                diff_r = difference(r20i['temperature'], df['temp (degree_Celsius)'])
                diff_r2 = difference(r22i['temperature'], df['temp (degree_Celsius)'])
                diff_c = difference(cpi["temperature"], df['temp (degree_Celsius)'])
                
                ax1.plot(diff_g[0], g31i['depth'], 'g-o', label=f'{diff_g[1]}, {diff_g[2]}')
                ax1.plot(diff_r[0], r20i['depth'], 'r-o', label=f'{diff_r[1]}, {diff_r[2]}')
                ax1.plot(diff_r2[0], r22i['depth'], 'r-o', alpha=0.5, label=f"{diff_r2[1]}, {diff_r2[2]}")
                ax1.plot(diff_c[0], cpi['depth'], 'm-o', label=f"{diff_c[1]}, {diff_c[2]}")
                ax1.axvline(0, color="blue")

                ax1.set_ylim([400, 0])
                ax1.grid(True, linestyle='--', linewidth=.5)
                ax1.tick_params(axis='both', labelsize=13)
                ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
                ax1.legend(title="bias, rms", loc=3, fontsize='small',)

                # Salinity
                diff_g = difference(g31i['salinity'], df['psal (PSU)'])
                diff_r = difference(r20i['salinity'], df['psal (PSU)'])
                diff_r2 = difference(r22i['salinity'], df['psal (PSU)'])
                diff_c = difference(cpi['salinity'], df['psal (PSU)'])
                
                ax2.plot(diff_g[0], g31i['depth'], 'g-o', label=f'{diff_g[1]}, {diff_g[2]}')
                ax2.plot(diff_r[0], r20i['depth'], 'r-o', label=f'{diff_r[1]}, {diff_r[2]}')
                ax2.plot(diff_r2[0], r22i['depth'], 'r-o', alpha=0.5, label=f"{diff_r2[1]}, {diff_r2[2]}")
                ax2.plot(diff_c[0], cpi['depth'], 'm-o', label=f"{diff_c[1]}, {diff_c[2]}")
                ax2.axvline(0, color="blue")

                ax2.set_ylim([400, 0])
                ax2.grid(True, linestyle='--', linewidth=.5)
                ax2.tick_params(axis='both', labelsize=13)
                ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
                ax2.legend(title="bias, rms", loc=3, fontsize='small',)
                # ax2.set_ylabel('Depth (m)', fontsize=14)

                # Density
                diff_g = difference(g31i['density'], df['density'])
                diff_r = difference(r20i['density'], df['density'])
                diff_r2 = difference(r22i['density'], df['density'])
                diff_c = difference(cpi['density'], df['density'])
                
                ax3.plot(diff_g[0], g31i['depth'], 'g-o', label=f'{diff_g[1]}, {diff_g[2]}')
                ax3.plot(diff_r[0], r20i['depth'], 'r-o', label=f'{diff_r[1]}, {diff_r[2]}')
                ax3.plot(diff_r2[0], r22i['depth'], 'r-o', alpha=0.5, label=f"{diff_r2[1]}, {diff_r2[2]}")
                ax3.plot(diff_c[0], cpi['depth'], 'm-o', label=f"{diff_c[1]}, {diff_c[2]}")

                ax3.set_ylim([400, 0])
                ax3.grid(True, linestyle='--', linewidth=.5)
                ax3.tick_params(axis='both', labelsize=13)
                ax3.set_xlabel('Density', fontsize=14, fontweight='bold')
                ax3.legend(title="bias, rms", loc=3, fontsize='small',)
                v = ax3.axvline(0, color="blue")
                # ax3.set_ylabel('Depth (m)', fontsize=14)

                text = ax4.text(0.125, 1.0, 
                                f'Argo #{wmo} [{alon}, {alat}]\n'
                                f'ARGO:  { tstr }\n'
                                f'CMEMS: {pd.to_datetime(cpi.time.data)}\n'
                                f'GOFS 3.1: {pd.to_datetime(g31i.time.data)}\n'
                                f'RTOFS 2.0: {pd.to_datetime(r20i.time.data)}\n'
                                f'RTOFS 2.2: {pd.to_datetime(r22i.time.data)}\n',
                                ha='left', va='top', size=15, fontweight='bold')
                
                text.set_path_effects([path_effects.Normal()])
                ax4.set_axis_off()

                map_create(extent, ax=ax5, ticks=False)
                ax5.plot(lon, lat, 'ro', transform=configs.projection['data'])

                h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

                ax6.legend([v] + h, [f"{wmo}", 'GOFS 3.1', 'RTOFS 2.0', 'RTOFS 2.2', "CMEMS",], ncol=1, loc='center', fontsize=12)
                # ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
                ax6.set_axis_off()
                
                plt.figtext(0.15, 0.001, f'Depths interpolated to every {configs.stride}m', ha="center", fontsize=10, fontstyle='italic')


                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                
                diff_file = temp_save_dir / f'{wmo}-profile-difference-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png' 

                plt.savefig(diff_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                plt.close()