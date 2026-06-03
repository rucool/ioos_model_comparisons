#/usr/bin/env python
import json
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from ioos_model_comparisons.calc import (
    # depth_interpolate,
    lon180to360,
    lon360to180,
    difference,
    density,
    ocean_heat_content
    )
from ioos_model_comparisons.platforms import (
    ARGO_DATA_QC_VARIABLES,
    ARGO_GOOD_QC_FLAGS,
    ARGO_LOCATION_QC_VARIABLES,
    ARGO_PROFILE_QC_VARIABLES,
    get_argo_floats_by_time,
    get_ohc,
)
from ioos_model_comparisons.regions import region_config
import ioos_model_comparisons.configs as conf
import cool_maps.plot as cplt
from gsw import z_from_p
import glob
import cartopy.feature as cfeature
import re
from datetime import datetime
from cool_maps.plot import get_bathymetry

save_dir = conf.path_plots / 'profiles' / 'argo'

# Configs
parallel = True
depth = 400

# Which models should we plot?
plot_rtofs = True
plot_espc = True
plot_cmems = True
plot_para = False

# argos = ["4902350", "4903250", "6902854", "4903224", "4903227"]
# argos = [4903227]
# conf.regions = ['caribbean-leeward'] # For debug purposes
days = 7
dpi = conf.dpi
vars = ['platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal']

depths = slice(0, depth)
ARGO_TEMP_POINT_QC_COLUMNS = ('temp_qc',)
ARGO_SALINITY_POINT_QC_COLUMNS = ('psal_qc',)
ARGO_DENSITY_POINT_QC_COLUMNS = ('pres_qc', 'temp_qc', 'psal_qc')
QC_FLAGGED_HANDLE = Line2D(
    [0], [0],
    linestyle='None',
    marker='o',
    markerfacecolor='none',
    markeredgecolor='red',
    markeredgewidth=1.5,
    markersize=7,
    label='Argo QC flagged',
)

# Load models
if plot_rtofs:
    from ioos_model_comparisons.models import rtofs
    rds = rtofs(source='west').sel(depth=depths)

if plot_para:
    from ioos_model_comparisons.models import rtofs
    pds = rtofs(source='parallel').sel(depth=depths)

if plot_espc:
    from ioos_model_comparisons.models import espc_ts
    espc_loaded = espc_ts(rename=True).sel(depth=depths)
    # gds = espc(rename=True).sel(depth=depths)
    # espc_obj = ESPC(uv=False)
    # print()


if plot_cmems:
    from ioos_model_comparisons.models import CMEMS
    cobj = CMEMS()
    # cds = cobj.data.sel(depth=depths)

# Create a date list ending today and starting x days in the past
date_end = pd.Timestamp.utcnow().tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor('1d')

# For symlink folder
then = pd.Timestamp.today() - pd.Timedelta(days=14) # get the date 14 days ago
then = pd.Timestamp(then.strftime('%Y-%m-%d')) # convert back to timestamp

# Get extent for all configured regions to download argo/glider data one time
extent_list = []
conf.regions = ['hawaii', 'mexico_pacific']
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

# Formatter for date
date_fmt = "%Y-%m-%dT%H:%MZ"

# import time
# startTime = time.time() # Start time to see how long the script took
# Download argo floats from ifremer erddap server
floats = get_argo_floats_by_time(global_extent,
                                 date_start,
                                 date_end,
                                 variables=vars,
                                 include_qc=True)
print(f"Loaded Argo QC flags for {len(floats)} samples")
# print('Execution time in seconds: ' + str(time.time() - startTime))
# search_window_t = (ctime - dt.timedelta(hours=conf.search_hours)).strftime(tstr)
# search_window_t1 = ctime.strftime(tstr)

# Convert pressure to depth
# floats['depth'] = seawater.dpth(floats['pres (decibar)'], floats['lat'])
floats['depth'] = -z_from_p(floats['pres (decibar)'], floats['lat'])

# Mask argo float based off of the maximum depth
depth_mask = floats['depth'] <= depth
floats = floats[depth_mask]

levels = [-8000, -1000, -100, 0]
colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']

# %% Define functions
def line_limits(fax, delta=1):
    """Function to get the minimum and maximum of a series of lines from a
    Matplotlib axis.

    Args:
        fax (_type_): Matplotlib Axes
        delta (float, optional): Delta for . Defaults to 1.

    Returns:
        _type_: _description_
    """
    mins = [np.nanmin(line.get_xdata()) for line in fax.lines]
    maxs = [np.nanmax(line.get_xdata()) for line in fax.lines]
    return min(mins)-delta, max(maxs)+delta


def normalize_qc_flag(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def qc_flag_mask(df, columns):
    columns = [column for column in columns if column in df.columns]
    if not columns:
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    for column in columns:
        flags = df[column].map(normalize_qc_flag)
        mask |= flags.notna() & ~flags.isin(ARGO_GOOD_QC_FLAGS)
    return mask.fillna(False)


def profile_qc_value(df, column):
    if column not in df.columns:
        return 'NA'

    values = [value for value in df[column].map(normalize_qc_flag).dropna().unique().tolist() if value]
    if not values:
        return 'NA'
    if len(values) == 1:
        return values[0]
    return ','.join(values[:3]) + ('+' if len(values) > 3 else '')


def add_flagged_points(ax, x, y, mask):
    if bool(mask.any()):
        ax.scatter(
            x[mask],
            y[mask],
            s=42,
            facecolors='none',
            edgecolors='red',
            linewidths=1.5,
            zorder=20,
        )

def process_argo(region):
    # Loop through regions
    region = region_config(region)
    extent = region['extent']
    print(f'Region: {region["name"]}, Extent: {extent}')
    # extended = np.add(extent, [-1, 1, -1, 1]).tolist()
    temp_save_dir = save_dir / region['folder']
    os.makedirs(temp_save_dir, exist_ok=True)

    # Create symlink directory
    symlink_dir = temp_save_dir / 'last_14_days'
    os.makedirs(symlink_dir, exist_ok=True)

    # Cleanup symlink directory
    for f in sorted(glob.glob(os.path.join(symlink_dir, '*.png'))):
        # Extract date and time from string
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{4}Z', f)
        if match:
            date_time_str = match.group()

            # Convert string to datetime object
            date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%dT%H%MZ')

            # Check if the date in the string is older than 14 days
            if (datetime.now() - date_time_obj).days > 14:
                print(f"The file {f} is older than 14 days.")
                # Uncomment the following line to delete the file
                os.remove(f)

    locations_file = symlink_dir / 'locations.json'
    if locations_file.exists():
        try:
            with open(locations_file, 'r') as f:
                locations = json.load(f)
            locations = {k: v for k, v in locations.items() if (symlink_dir / k).exists()}
            with open(locations_file, 'w') as f:
                json.dump(locations, f)
        except Exception as e:
            print(f"Error cleaning up locations.json: {e}")

    try:
        bathy = get_bathymetry(extent)
        bathy_flag = True
    except:
        bathy_flag = False
        pass

    # bathy_flag = False

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

    if plot_rtofs:
        # Setting RTOFS lon and lat to their own variables speeds up the script
        rlons = rds.lon.data[0, :]
        rlats = rds.lat.data[:, 0]
        rx = rds.x.data
        ry = rds.y.data

        # Find x, y indexes of the area we want to subset
        lons_ind = np.interp(extent[:2], rlons, rx)
        lats_ind = np.interp(extent[2:], rlats, ry)

    # Iterate through argo float profiles
    for gname, df in argo_region.reset_index().groupby(['argo', 'time']):
        wmo = gname[0] # wmo id from gname
        ctime = gname[1] # time from gname
        print(f"Checking ARGO {wmo} for new profiles")

        tstr = ctime.strftime(date_fmt) # create time string
        save_str = f'{wmo}-profile-{ctime.strftime("%Y-%m-%dT%H%MZ")}.png'
        tdir = temp_save_dir / ctime.strftime("%Y") / ctime.strftime("%m") / ctime.strftime("%d")
        os.makedirs(tdir, exist_ok=True)
        full_file = tdir /  save_str
        diff_file = tdir / ctime.strftime("%Y") / ctime.strftime("%m") / ctime.strftime("%d") / f'{wmo}-profile-difference-{ctime.strftime("%Y-%m-%dT%H%MZ")}.png'

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
            # # Calculate depth from pressure and lat
            df = df.assign(depth=-z_from_p(df['pres (decibar)'].values, df['lat'].values))

            depth_mask = df['depth'] <= depth
            df = df[depth_mask].copy()

            # Check if the dataframe has any data after filtering.
            if df.empty:
                continue

            df = df.assign(
                density=density(
                    df['temp (degree_Celsius)'].values,
                    -df['depth'].values,
                    df['psal (PSU)'].values,
                    df['lat'].values,
                    df['lon'].values
                )
            )

            temp_flagged = qc_flag_mask(df, ARGO_TEMP_POINT_QC_COLUMNS)
            salinity_flagged = qc_flag_mask(df, ARGO_SALINITY_POINT_QC_COLUMNS)
            density_flagged = qc_flag_mask(df, ARGO_DENSITY_POINT_QC_COLUMNS)
            any_flagged = bool(temp_flagged.any() or salinity_flagged.any() or density_flagged.any())

            ohc_float = ocean_heat_content(
                df['depth'],
                df['temp (degree_Celsius)'],
                df['density']
                )

            # Grab lon and lat of argo profile
            lon, lat = df['lon'].unique()[-1], df['lat'].unique()[-1]

            mlon360 = lon180to360(lon)

            alon = round(lon, 2) # argo lon
            alat = round(lat, 2) # argo lat
            alabel = f'{wmo} [{alon}, {alat}]'
            leg_str = f'Argo #{wmo}\n'
            leg_str += f'ARGO: { tstr }\n'

            if plot_espc:
                try:
                    # ESPC
                    gdsi = espc_loaded.sel(lon=mlon360, lat=lat, method='nearest')
                    gdsi = gdsi.sel(time=ctime, method="nearest")

                    # Convert the lon back to a 180 degree lon
                    gdsi['lon'] = lon360to180(gdsi['lon'])

                    # Calculate density for gofs profile
                    gdsi['density'] = density(gdsi.temperature, -gdsi.depth, gdsi.salinity, gdsi.lat, gdsi.lon)

                    # Calculate ocean heat content for profile
                    ohc_espc = ocean_heat_content(
                        gdsi['depth'].values,
                        gdsi['temperature'].values,
                        gdsi['density'].values
                        )

                    glon = gdsi.lon.data.round(2)
                    glat = gdsi.lat.data.round(2)
                    glabel = f'ESPC [{ glon }, { glat }]'
                    leg_str += f'ESPC : {pd.to_datetime(gdsi.time.data).strftime(date_fmt)}\n'
                    espc_flag = True
                except KeyError as error:
                    print(f"ESPC: False - {error}")
                    espc_flag = False
            else:
                espc_flag = False

            if plot_rtofs:
                try:
                    # RTOFS
                    # interpolating lon and lat to x and y index of the rtofs grid
                    rlonI = np.interp(lon, rlons, rx)
                    rlatI = np.interp(lat, rlats, ry)

                    rdsp = rds.sel(time=ctime, method='nearest')
                    rdsi = rdsp.sel(
                        x=rlonI,
                        y=rlatI,
                        method='nearest'
                    )

                    rdsi.load()

                    # Calculate density for rtofs profile
                    rdsi['density'] = density(rdsi.temperature, -rdsi.depth, rdsi.salinity, rdsi.lat, rdsi.lon)

                    # Calculate ocean heat content for profile
                    ohc_rtofs = ocean_heat_content(
                        rdsi['depth'].values,
                        rdsi['temperature'].values,
                        rdsi['density'].values
                        )
                    rlon = rdsi.lon.data.round(2)
                    rlat = rdsi.lat.data.round(2)
                    rlabel = f'RTOFS [{rlon:.2f}, {rlat:.2f}]'
                    leg_str += f'RTOFS: {pd.to_datetime(rdsi.time.data).strftime(date_fmt)}\n'

                    # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice
                    # in order to widen the area of the extent slightly.
                    extent_ind = [
                        np.floor(lons_ind[0]).astype(int),
                        np.ceil(lons_ind[1]).astype(int),
                        np.floor(lats_ind[0]).astype(int),
                        np.ceil(lats_ind[1]).astype(int)
                        ]

                    rdsp = rdsp.isel(depth=0,
                                    x=slice(extent_ind[0], extent_ind[1]),
                                    y=slice(extent_ind[2], extent_ind[3]))
                    rtofs_flag = True
                except KeyError as error:
                    print(f"RTOFS: False - {error}")
                    rtofs_flag = False
            else:
                rtofs_flag = False

            if plot_para:
                try:
                    # RTOFS
                    # interpolating lon and lat to x and y index of the rtofs grid
                    rlonI = np.interp(lon, rlons, rx)
                    rlatI = np.interp(lat, rlats, ry)

                    pdsp = pds.sel(time=ctime, method='nearest')
                    pdsi = pdsp.sel(
                        x=rlonI,
                        y=rlatI,
                        method='nearest'
                    )
                    pdsi.load()

                    # Calculate density for rtofs profile
                    pdsi['density'] = density(pdsi.temperature, -pdsi.depth, pdsi.salinity, pdsi.lat, pdsi.lon)

                    # Calculate ocean heat content for profile
                    ohc_rtofsp = ocean_heat_content(
                        pdsi['depth'].values,
                        pdsi['temperature'].values,
                        pdsi['density'].values
                        )
                    rlon = pdsi.lon.data.round(2)
                    rlat = pdsi.lat.data.round(2)
                    plabel = f'RTOFS-P [{rlon:.2f}, {rlat:.2f}]'
                    leg_str += f'RTOFS-P: {pd.to_datetime(pdsi.time.data).strftime(date_fmt)}\n'

                    # Use np.floor on the 1st index and np.ceil on the 2nd index of each slice
                    # in order to widen the area of the extent slightly.
                    extent_ind = [
                        np.floor(lons_ind[0]).astype(int),
                        np.ceil(lons_ind[1]).astype(int),
                        np.floor(lats_ind[0]).astype(int),
                        np.ceil(lats_ind[1]).astype(int)
                        ]

                    rdsp = pdsp.isel(depth=0,
                                    x=slice(extent_ind[0], extent_ind[1]),
                                    y=slice(extent_ind[2], extent_ind[3]))
                    rtofsp_flag = True
                except KeyError as error:
                    print(f"RTOFS: False - {error}")
                    rtofsp_flag = False
            else:
                rtofsp_flag = False

            if plot_cmems:
                try:
                    # Copernicus
                    cdsi = cobj.get_point(lon, lat, ctime)
                    cdsi = cdsi.sel(depth=depths)

                    # Calculate density for rtofs profile
                    cdsi['density'] = density(cdsi.temperature, -cdsi.depth, cdsi.salinity, cdsi.lat, cdsi.lon)

                    clon = cdsi.lon.data.round(2)
                    clat = cdsi.lat.data.round(2)

                    # Calculate ocean heat content for profile
                    ohc_cmems = ocean_heat_content(
                        cdsi['depth'].values,
                        cdsi['temperature'].values,
                        cdsi['density'].values
                        )

                    clabel = f"CMEMS [{clon:.2f}, {clat:.2f}]"
                    leg_str += f'CMEMS: {pd.to_datetime(cdsi.time.data).strftime(date_fmt)}\n'
                    cmems_flag = True
                except KeyError as error:
                    print(f"CMEMS: False - {error}")
                    cmems_flag = False
            else:
                cmems_flag = False

            leg_str += 'QC kept; red open circles = level QC flagged\n'
            leg_str += f'QC time/pos: {profile_qc_value(df, "time_qc")}/{profile_qc_value(df, "position_qc")}\n'
            leg_str += (
                f'QC prof P/T/S: '
                f'{profile_qc_value(df, "profile_pres_qc")}/'
                f'{profile_qc_value(df, "profile_temp_qc")}/'
                f'{profile_qc_value(df, "profile_psal_qc")}\n'
            )
            leg_str += (
                f'Flagged pts T/S/D: '
                f'{int(temp_flagged.sum())}/{int(salinity_flagged.sum())}/{int(density_flagged.sum())}\n'
            )

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

            # ARGO
            ax1.plot(df['temp (degree_Celsius)'], df['depth'], 'b-o', label=alabel)
            ax2.plot(df['psal (PSU)'], df['depth'], 'b-o', label=alabel)
            ax3.plot(df['density'], df['depth'], 'b-o', label=alabel)
            add_flagged_points(ax1, df['temp (degree_Celsius)'], df['depth'], temp_flagged)
            add_flagged_points(ax2, df['psal (PSU)'], df['depth'], salinity_flagged)
            add_flagged_points(ax3, df['density'], df['depth'], density_flagged)

            # ESPC
            if espc_flag:
                ax1.plot(gdsi['temperature'], gdsi['depth'], linestyle='-',  marker='o',color='green', label=glabel)
                ax2.plot(gdsi['salinity'], gdsi['depth'],linestyle= '-',   marker='o', color='green', label=glabel)
                ax3.plot(gdsi['density'], gdsi['depth'], linestyle='-',   marker='o', color='green', label=glabel)

            # RTOFS
            if rtofs_flag:
                ax1.plot(rdsi['temperature'], rdsi['depth'], linestyle='-',  marker='o', color='red', label=rlabel)
                ax2.plot(rdsi['salinity'], rdsi['depth'], linestyle='-', marker='o', color='red', label=rlabel)
                ax3.plot(rdsi['density'], rdsi['depth'], linestyle='-',  marker='o',color='red', label=rlabel)

            # CMEMS
            if cmems_flag:
                ax1.plot(cdsi['temperature'], cdsi['depth'], linestyle='-', marker='o', color='magenta', label=clabel)
                ax2.plot(cdsi['salinity'], cdsi['depth'], linestyle='-',  marker='o', color='magenta', label=clabel)
                ax3.plot(cdsi['density'], cdsi['depth'], linestyle='-',  marker='o',color='magenta', label=clabel)

            if rtofsp_flag:
                ax1.plot(pdsi['temperature'], pdsi['depth'], linestyle='-',  marker='o', color='orange', label=plabel)
                ax2.plot(pdsi['salinity'], pdsi['depth'], linestyle='-', marker='o', color='orange', label=plabel)
                ax3.plot(pdsi['density'], pdsi['depth'], linestyle='-',  marker='o',color='orange', label=plabel)

            try:
                # Get min and max of each plot. Add a delta to each for x limits
                tmin, tmax = line_limits(ax1, delta=.5)
                smin, smax = line_limits(ax2, delta=.25)
                dmin, dmax = line_limits(ax3, delta=.5)
            except ValueError:
                print('Some kind of error')
                pass


            ax1.set_ylim([depth, 0])
            ax1.set_xlim([tmin, tmax])
            ax1.grid(True, linestyle='--', linewidth=.5)
            ax1.tick_params(axis='both', labelsize=13)
            ax1.set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')

            ax2.set_ylim([depth, 0])
            ax2.set_xlim([smin, smax])
            ax2.grid(True, linestyle='--', linewidth=.5)
            ax2.tick_params(axis='both', labelsize=13)
            ax2.set_xlabel('Salinity (psu)', fontsize=14, fontweight='bold')

            ax3.set_ylim([depth, 0])
            ax3.set_xlim([dmin, dmax])
            ax3.grid(True, linestyle='--', linewidth=.5)
            ax3.tick_params(axis='both', labelsize=13)
            ax3.set_xlabel('Density', fontsize=14, fontweight='bold')

            text = ax4.text(0.125, 1.0,
                            leg_str,
                            ha='left', va='top', size=13, fontweight='bold')

            text.set_path_effects([path_effects.Normal()])
            ax4.set_axis_off()

            cplt.create(extent, ax=ax5, bathymetry=False)
            cplt.add_ticks(ax5, extent, fontsize=8)
            ax5.plot(lon, lat, 'bo', transform=conf.projection['data'], zorder=101)

            import shapely
            bathy_flag=False
            if bathy_flag:
                try:
                    ax5.contourf(
                        bathy['longitude'],
                        bathy['latitude'],
                        bathy['z'],
                        levels,
                        colors=colors,
                        transform=conf.projection['data'],
                        ticks=False,
                        zorder=98
                        )
                except shapely.errors.GEOSException:
                    pass

            h, l = ax2.get_legend_handles_labels()  # get labels and handles from ax1
            if any_flagged:
                h.append(QC_FLAGGED_HANDLE)
                l.append(QC_FLAGGED_HANDLE.get_label())

            ax6.legend(h, l, ncol=1, loc='center', fontsize=12)
            ax6.set_axis_off()

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

            ohc_string = 'Ocean Heat Content (kJ/cm^2) - '
            try:
                if np.isnan(np.nanmean(ohc_float)):
                    ohc_string += 'Argo: N/A,  '
                else:
                    ohc_string += f"Argo: {np.nanmean(ohc_float):.4f},  "
            except:
                pass

            try:
                if np.isnan(ohc_rtofs):
                    ohc_string += 'RTOFS: N/A,  '
                else:
                    ohc_string += f"RTOFS: {ohc_rtofs:.4f},  "
            except:
                pass

            try:
                if np.isnan(ohc_rtofsp):
                    ohc_string += 'RTOFS (Parallel): N/A,  '
                else:
                    ohc_string += f"RTOFS (Parallel): {ohc_rtofsp:.4f},  "
            except:
                pass

            try:
                if np.isnan(ohc_espc):
                    ohc_string += 'ESPC: N/A,  '
                else:
                    ohc_string += f"ESPC: {ohc_espc:.4f},  "
            except:
                pass

            try:
                if np.isnan(ohc_cmems):
                    ohc_string += 'CMEMS: N/A,  '
                else:
                    ohc_string += f"CMEMS: {ohc_cmems:.4f},  "
            except:
                pass

            plt.figtext(0.4, 0.001, ohc_string, ha="center", fontsize=10, fontstyle='italic')

            plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            # Create a symlink directory
            if ctime > then:
                os.symlink(full_file, symlink_dir / save_str)

                locations_file = symlink_dir / 'locations.json'
                locations = {}
                if locations_file.exists():
                    try:
                        with open(locations_file, 'r') as f:
                            locations = json.load(f)
                    except Exception:
                        pass
                locations[save_str] = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'wmo': str(wmo),
                    'time': tstr
                }
                try:
                    with open(locations_file, 'w') as f:
                        json.dump(locations, f)
                except Exception as e:
                    print(f"Error saving locations.json: {e}")

def main():
    if parallel:
        import concurrent.futures
        if isinstance(parallel, bool):
            workers = 6
        elif isinstance(parallel, int):
            workers = parallel

        print(f"Running in parallel mode with {workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(process_argo, conf.regions)
    else:
        for region in conf.regions:
            process_argo(region)

if __name__ == "__main__":
    main()
