import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seawater
import xarray as xr
from ioos_model_comparisons.calc import depth_interpolate, lon180to360, lon360to180, difference
from ioos_model_comparisons.models import gofs, rtofs, cmems
from ioos_model_comparisons.platforms import get_argo_floats_by_time
from ioos_model_comparisons.plotting import map_create, map_add_ticks
from ioos_model_comparisons.regions import region_config
import ioos_model_comparisons.configs as conf

save_dir = conf.path_plots / 'profiles' / 'argo'

# Configs
parallel = True

# argos = ["4902350", "4903250", "6902854", "4903224", "4903227"]
# argos = [4903227]
# conf.regions = ['caribbean-leeward'] # For debug purposes
days = 10
dpi = conf.dpi
depths = slice(0, 1000)
vars = ['platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal']

# Load models
rds = rtofs().sel(depth=depths)
gds = gofs(rename=True).sel(depth=depths)
cds = cmems(rename=True).sel(depth=depths)

# Create a date list ending today and starting x days in the past
date_end = pd.Timestamp.utcnow().tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor('1d')

# Get extent for all configured regions to download argo/glider data one time
extent_list = []
for region in conf.regions:
    extent_list.append(region_config(region)["extent"])

extent_df = pd.DataFrame(
    np.array(extent_list),
    columns=['lonmin', 'lonmax', 'latmin', 'latmax']
    )

global_extent = [
    extent_df.lonmin.min(),
    extent_df.lonmax.max(),
    extent_df.latmin.min(),
    extent_df.latmax.max()
    ]

# import time
# startTime = time.time() # Start time to see how long the script took
# Download argo floats from ifremer erddap server
floats = get_argo_floats_by_time(global_extent,
                                 date_start,
                                 date_end, 
                                 variables=vars)
# print('Execution time in seconds: ' + str(time.time() - startTime))
# search_window_t0 = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
# search_window_t1 = ctime.strftime(tstr) 
        
def process_argo(region):    
    # Loop through regions
    region = region_config(region)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')
    # extended = np.add(extent, [-1, 1, -1, 1]).tolist()
    temp_save_dir = save_dir / region['folder']
    os.makedirs(temp_save_dir, exist_ok=True)

    # Filter to region
    if floats.empty:
        print(f"No Argo floats found in {region['name']}")
        return
    else:
        argo_region = floats[
            (extent[0] <= floats['lon']) & (floats['lon'] <= extent[1]) 
            &
            (extent[2] <= floats['lat']) & (floats['lat'] <= extent[3])
            ]

    # Setting RTOFS lon and lat to their own variables speeds up the script
    rlons = rds.lon.data[0, :]
    rlats = rds.lat.data[:, 0] 
    rx = rds.x.data
    ry = rds.y.data

    # Iterate through argo float profiles
    for gname, df in argo_region.reset_index().groupby(['argo', 'time']):
        wmo = gname[0] # wmo id from gname
        ctime = gname[1] # time from gname
        print(f"Checking ARGO {wmo} for new profiles")
        
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
            print(f"Processing ARGO {wmo} profile that occured at {ctime}")
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
                                   depth_min=conf.min_depth,
                                   depth_max=conf.max_depth,
                                   stride=conf.stride)

            # Grab lon and lat of argo profile
            lon, lat = df['lon'].unique()[-1], df['lat'].unique()[-1]

            # %%
            # For each model, select nearest time to argo time
            # Interpolate to the argo float location and depths
            
            # Calculate depths to interpolate 
            depths_interp = np.arange(
                conf.min_depth, 
                conf.max_depth+conf.stride, 
                conf.stride)

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
            # interpolating lon and lat to x and y index of the rtofs grid
            rlonI = np.interp(lon, rlons, rx)
            rlatI = np.interp(lat, rlats, ry)

            rdsp = rds.sel(time=ctime, method='nearest')
            rdsi = rdsp.interp(
                x=rlonI,
                y=rlatI,
                depth=xr.DataArray(depths_interp, dims='depth')
            )
            
            # Calculate density for rtofs profile
            rdsi['pressure'] = xr.apply_ufunc(seawater.eos80.pres, rdsi.depth, rdsi.lat)
            rdsi['density'] = xr.apply_ufunc(seawater.eos80.dens, rdsi.salinity, rdsi.temperature, rdsi.pressure)

            # Copernicus
            cdsp = cds.sel(time=ctime, method='nearest')
            cdsi = cdsp.interp(
                lon=lon, 
                lat=lat,
                depth=xr.DataArray(depths_interp, dims='depth')
                )
            
            # Calculate density for rtofs profile
            cdsi['pressure'] = xr.apply_ufunc(seawater.eos80.pres, cdsi.depth, cdsi.lat)
            cdsi['density'] = xr.apply_ufunc(seawater.eos80.dens, cdsi.salinity, cdsi.temperature, cdsi.pressure)

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
            clon = cdsi.lon.data.round(2)
            clat = cdsi.lat.data.round(2)

            # Legend labels
            alabel = f'{wmo} [{alon}, {alat}]'
            glabel = f'GOFS [{ glon }, { glat }]'
            rlabel = f'RTOFS [{ rlon }, { rlat }]'
            clabel = f"Copernicus [{ clon }, { clat }]"

            # Temperature 
            ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
            ax1.plot(gdsi['temperature'], gdsi['depth'], 'g-o', label=glabel)
            ax1.plot(rdsi['temperature'], rdsi['depth'], 'r-o', label=rlabel)
            ax1.plot(cdsi['temperature'], cdsi['depth'], 'm-o', label=clabel)

            ax1.set_ylim([400, 0])
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

            # Salinity
            ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
            ax2.plot(gdsi['salinity'], gdsi['depth'],'g-o', label=glabel)
            ax2.plot(rdsi['salinity'], rdsi['depth'], 'r-o', label=rlabel)
            ax2.plot(cdsi['salinity'], cdsi['depth'],  'm-o', label=clabel)

            ax2.set_ylim([400, 0])
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
            # ax2.set_ylabel('Depth (m)', fontsize=14)

            # Density
            ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
            ax3.plot(gdsi['density'], gdsi['depth'],'g-o', label=glabel)
            ax3.plot(rdsi['density'], rdsi['depth'], 'r-o', label=rlabel)
            ax3.plot(cdsi['density'], cdsi['depth'], 'm-o', label=clabel)

            ax3.set_ylim([400, 0])
            ax3.grid(True, linestyle='--', linewidth=.5)
            ax3.tick_params(axis='both', labelsize=13)
            ax3.set_xlabel('Density', fontsize=14, fontweight='bold')
            # ax3.set_ylabel('Depth (m)', fontsize=14)

            text = ax4.text(0.125, 1.0, 
                            f'Argo #{wmo}\n'
                            f'ARGO:  { tstr }\n'
                            f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
                            f'GOFS : {pd.to_datetime(gdsi.time.data)}\n'
                            f'CMEMS: {pd.to_datetime(cdsi.time.data)}',
                            ha='left', va='top', size=15, fontweight='bold')

            text.set_path_effects([path_effects.Normal()])
            ax4.set_axis_off()

            map_create(extent, ax=ax5, ticks=False)
            map_add_ticks(ax5, extent, fontsize=10)
            ax5.plot(lon, lat, 'ro', transform=conf.projection['data'])

            h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

            ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
            ax6.set_axis_off()
            
            plt.figtext(0.15, 0.001, f'Depths interpolated to every {conf.stride}m', ha="center", fontsize=10, fontstyle='italic')


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
            ax5 = fig.add_subplot(gs[1, -1], projection=conf.projection['map']) # Map
            ax6 = fig.add_subplot(gs[2, -1]) # Legend

            # Temperature 
            diff_g = difference(gdsi['temperature'], df['temp (degree_Celsius)'])
            diff_r = difference(rdsi['temperature'], df['temp (degree_Celsius)'])
            diff_c = difference(cdsi['temperature'], df['temp (degree_Celsius)'])
            
            glabel = f'{diff_g[1]}, {diff_g[2]}'
            rlabel = f'{diff_r[1]}, {diff_r[2]}'
            clabel = f"{diff_c[1]}, {diff_c[2]}"
            
            ax1.plot(diff_g[0], gdsi['depth'], 'g-o', label=glabel)
            ax1.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
            ax1.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)
            ax1.axvline(0)

            ax1.set_ylim([400, 0])
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
            ax1.legend(title="bias, rms", loc=3, fontsize='small',)

            # Salinity
            diff_g = difference(gdsi['salinity'], df['psal (PSU)'])
            diff_r = difference(rdsi['salinity'], df['psal (PSU)'])
            diff_c = difference(cdsi['salinity'], df['psal (PSU)'])
            
            glabel = f'{diff_g[1]}, {diff_g[2]}'
            rlabel = f'{diff_r[1]}, {diff_r[2]}'
            clabel = f"{diff_c[1]}, {diff_c[2]}"
            
            # ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
            ax2.plot(diff_g[0], gdsi['depth'], 'g-o', label=glabel)
            ax2.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
            ax2.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)
            ax2.axvline(0)

            ax2.set_ylim([400, 0])
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')
            ax2.legend(title="bias, rms", loc=3, fontsize='small',)
            # ax2.set_ylabel('Depth (m)', fontsize=14)

            # Density
            diff_g = difference(gdsi['density'], df['density'])
            diff_r = difference(rdsi['density'], df['density'])
            diff_c = difference(cdsi['density'], df['density'])
            
            glabel = f'{diff_g[1]}, {diff_g[2]}'
            rlabel = f'{diff_r[1]}, {diff_r[2]}'
            clabel = f"{diff_c[1]}, {diff_c[2]}"
            
            # ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
            ax3.plot(diff_g[0], gdsi['depth'],'g-o', label=glabel)
            ax3.plot(diff_r[0], rdsi['depth'], 'r-o', label=rlabel)
            ax3.plot(diff_c[0], cdsi['depth'], 'm-o', label=clabel)

            ax3.set_ylim([400, 0])
            ax3.grid(True, linestyle='--', linewidth=.5)
            ax3.tick_params(axis='both', labelsize=13)
            ax3.set_xlabel('Density', fontsize=14, fontweight='bold')
            ax3.legend(title="bias, rms", loc=3, fontsize='small',)
            ax3.axvline(0)
            # ax3.set_ylabel('Depth (m)', fontsize=14)

            text = ax4.text(0.125, 1.0, 
                            f'Argo #{wmo}\n'
                            f'ARGO:  { tstr }\n'
                            f'RTOFS: {pd.to_datetime(rdsi.time.data)}\n'
                            f'GOFS : {pd.to_datetime(gdsi.time.data)}\n'
                            f'CMEMS: {pd.to_datetime(cdsi.time.data)}',
                            ha='left', va='top', size=15, fontweight='bold')

            text.set_path_effects([path_effects.Normal()])
            ax4.set_axis_off()

            map_create(extent, ax=ax5, ticks=False)
            map_add_ticks(ax5, extent, fontsize=10)
            
            ax5.plot(lon, lat, 'ro', transform=conf.projection['data'])

            h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1

            ax6.legend(h, [f'GOFS [{ glon }, { glat }]', f'RTOFS [{ rlon }, { rlat }]', f"Copernicus [{ clon }, { clat }]"], ncol=1, loc='center', fontsize=12)
            ax6.set_axis_off()
            
            plt.figtext(0.15, 0.001, f'Depths interpolated to every {conf.stride}m', ha="center", fontsize=10, fontstyle='italic')


            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            diff_file = temp_save_dir / f'{wmo}-profile-difference-{ctime.strftime("%Y-%m-%dT%H%M%SZ")}.png' 

            plt.savefig(diff_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()

def main():    
    if parallel: 
        import concurrent.futures
        if isinstance(parallel, bool):
            workers = 6
        elif isinstance(parallel, int):
            workers = parallel

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(process_argo, conf.regions)
    else:
        for region in conf.regions:
            process_argo(region)

if __name__ == "__main__":
    main()