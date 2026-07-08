#!/usr/bin/env python
"""
Argo assimilation check using RTOFS binary regional NetCDFs.

Same 4-day / 4-time-step layout as check_assimilation_of_argo_profiles_by_day.py
but replaces the live RTOFS model connection with pre-processed regional binary
files (rtofs_glo_YYYYMMDDTHH_{region}.nc).
"""
import datetime as dt
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from matplotlib.lines import Line2D

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.calc import lon180to360
from ioos_model_comparisons.models import espc_ts, CMEMS
from ioos_model_comparisons.platforms import get_argo_floats_by_time

# ── Save dir ──────────────────────────────────────────────────────────────────

save_dir = conf.path_plots / 'profiles' / 'argo' / 'assimilation'
os.makedirs(save_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

argo_dict = {
    '3902345': dt.datetime(2026, 7, 5),  # adjust to the profile surfacing date
}

RTOFS_DATA_DIR = Path("/home/hurricaneadm/data/rtofs_archv")
# RTOFS_DATA_DIR = Path("/Users/mikesmith/Downloads/rtofs_global/")
RTOFS_REGION = 'guam'
RTOFS_MAX_HOURS = 24

days_to_check_for_argo_surfacing = 2
days_pre_surfacing = 1
days_post_surfacing = 2
depth = 400
interp = False

dpi = 150
figsize = (16, 10)
line = ('-', '--', '-.', ':')
labels = ['+0600', '+1200', '+1800', '+2400']
alpha = (1, .75, .5, .25)

rtofs_include = True
espc_include = True
cmems_include = True

# ── Model connections (non-binary) ────────────────────────────────────────────

legend_elements = [Line2D([0], [0], color='b', lw=2, label='ARGO')]

if espc_include:
    espc_ds = espc_ts(rename=True).sel(depth=slice(0, depth))[['temperature', 'salinity']]
    legend_elements.append(Line2D([0], [0], color='g', lw=2, label='ESPC'))

if cmems_include:
    cobj = CMEMS()
    legend_elements.append(Line2D([0], [0], color='m', lw=2, label='CMEMS'))

if rtofs_include:
    legend_elements.append(Line2D([0], [0], color='r', lw=2, label='RTOFS (binary)'))

line_handles = [Line2D([0], [0], color='k', linewidth=3, linestyle=l) for l in line]

# ── RTOFS binary helpers ──────────────────────────────────────────────────────

def _parse_rtofs_time(nc_path):
    stem = Path(nc_path).stem
    time_part = stem.split("_")[2]
    return pd.Timestamp(dt.datetime.strptime(time_part, "%Y%m%dT%H"))


def find_rtofs_file(region_name, target_time, data_dir=RTOFS_DATA_DIR, max_hours=RTOFS_MAX_HOURS):
    candidates = sorted(data_dir.glob(f"*/*/*/rtofs_glo_*_{region_name}.nc"))
    if not candidates:
        return None
    target = pd.Timestamp(target_time)
    best = min(candidates, key=lambda p: abs((_parse_rtofs_time(p) - target).total_seconds()))
    diff_h = abs((_parse_rtofs_time(best) - target).total_seconds()) / 3600
    return best if diff_h <= max_hours else None


def load_rtofs_point(nc_path, lon, lat, max_depth=400):
    try:
        ds = xr.open_dataset(nc_path)
        file_lon_min = float(ds.lon.min())
        lookup_lon = lon180to360(lon) if file_lon_min > 90 and lon < 0 else lon
        point = ds.sel(lat=lat, lon=lookup_lon, method='nearest')
        point = point.rename({"temp": "temperature", "salin": "salinity", "z": "depth"})
        point = point.sel(depth=slice(0, max_depth))
        drop_vars = [v for v in ("u-vel.", "v-vel.") if v in point]
        if drop_vars:
            point = point.drop_vars(drop_vars)
        point.load()
        return point
    except Exception as e:
        print(f"RTOFS binary: load failed ({e})")
        return None

# ── Main loop ─────────────────────────────────────────────────────────────────

for key, value in argo_dict.items():
    time_end = value + dt.timedelta(days=1)
    time_start = time_end - dt.timedelta(days=days_to_check_for_argo_surfacing)

    argo_data = get_argo_floats_by_time(
        bbox=(129.75, 160.25, 4.75, 25.25),  # Guam region
        time_start=time_start,
        time_end=time_end,
        wmo_id=key,
        variables=['pres', 'temp', 'psal'],
    )

    if argo_data.empty:
        print(f"No Argo data found for {key}")
        continue

    for t_float, df in argo_data.groupby(level=1):
        mask_salinity = np.abs(stats.zscore(df['psal (PSU)'])) < 3
        mask_depth = df['pres (decibar)'] <= depth
        df = df[mask_salinity & mask_depth]

        lon = df['lon'].unique()[0]
        lat = df['lat'].unique()[0]
        ctime = t_float.date()

        pre = ctime - dt.timedelta(days=days_pre_surfacing)
        post = ctime + dt.timedelta(days=days_post_surfacing + 1)
        date_ranges = pd.date_range(pre, post, freq='24H', inclusive='left')
        time_ranges = pd.date_range(pre, post, freq='6H', inclusive='right')

        # ESPC
        if espc_include:
            esub = espc_ds.sel(time=time_ranges)
            if interp:
                esub = esub.interp(lon=lon180to360(lon), lat=lat)
            else:
                esub = esub.sel(lon=lon180to360(lon), lat=lat, method='nearest')
            esub.load()

        # CMEMS — one point profile per day
        cmems_profiles = {}
        if cmems_include:
            for d in date_ranges:
                try:
                    cp = cobj.get_point(lon, lat, d, vars=['temperature', 'salinity'])
                    cp = cp.sel(depth=slice(0, depth))
                    cp.load()
                    cmems_profiles[d] = cp
                except Exception as e:
                    print(f"CMEMS: failed for {d} ({e})")
                    cmems_profiles[d] = None

        # RTOFS binary — load one profile per 6-hour time step
        rtofs_profiles = {}
        if rtofs_include:
            for t in time_ranges:
                nc = find_rtofs_file(RTOFS_REGION, t)
                if nc is not None:
                    rtofs_profiles[t] = load_rtofs_point(nc, lon, lat, max_depth=depth)
                else:
                    rtofs_profiles[t] = None
                    print(f"RTOFS binary: no file within {RTOFS_MAX_HOURS}h of {t}")

        # ── Plot ─────────────────────────────────────────────────────────────

        fig, ax = plt.subplots(2, 4, figsize=figsize, constrained_layout=True, sharey=True)

        temp_x = []
        salt_x = []

        for i, t_array in enumerate(np.split(time_ranges, 4)):
            n = 0

            for t in t_array:
                ax[0, i].plot(df['temp (degree_Celsius)'], df['pres (decibar)'], 'b-')
                ax[1, i].plot(df['psal (PSU)'], df['pres (decibar)'], 'b-')

                if espc_include:
                    ep = esub.sel(time=t, method='nearest')
                    ax[0, i].plot(ep['temperature'].squeeze(), ep['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])
                    ax[1, i].plot(ep['salinity'].squeeze(), ep['depth'].squeeze(), f'g{line[n]}', alpha=alpha[n])

                if cmems_include and cmems_profiles.get(date_ranges[i]) is not None:
                    cp = cmems_profiles[date_ranges[i]]
                    ax[0, i].plot(cp['temperature'].squeeze(), cp['depth'].squeeze(), f'm{line[1]}', alpha=alpha[n])
                    ax[1, i].plot(cp['salinity'].squeeze(), cp['depth'].squeeze(), f'm{line[1]}', alpha=alpha[n])

                if rtofs_include and rtofs_profiles.get(t) is not None:
                    rp = rtofs_profiles[t]
                    ax[0, i].plot(rp['temperature'].squeeze(), rp['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])
                    ax[1, i].plot(rp['salinity'].squeeze(), rp['depth'].squeeze(), f'r{line[n]}', alpha=alpha[n])

                temp_x.append(ax[0, i].xaxis.get_data_interval())
                salt_x.append(ax[1, i].xaxis.get_data_interval())
                ax[0, i].set_xlabel('Temperature (˚C)', fontsize=14, fontweight='bold')
                ax[1, i].set_xlabel('Salinity', fontsize=14, fontweight='bold')

                n += 1

        temp_x = pd.DataFrame(np.array(temp_x), columns=['min', 'max'])
        salt_x = pd.DataFrame(np.array(salt_x), columns=['min', 'max'])
        temp_x_min = math.floor(temp_x['min'].min() * 4) / 4
        temp_x_max = math.ceil(temp_x['max'].max() * 4) / 4
        salt_x_min = math.floor(salt_x['min'].min() * 4) / 4
        salt_x_max = math.ceil(salt_x['max'].max() * 4) / 4

        for j in range(4):
            ax[0, j].set_xlim([temp_x_min, temp_x_max])
            ax[1, j].set_xlim([salt_x_min, salt_x_max])

        for axs in ax.flat:
            axs.set_ylim([depth, 0])
            axs.grid(True, linestyle='--', linewidth=0.5)
            axs.tick_params(axis='both', labelsize=10)
            axs.set_ylabel('Depth (m)', fontsize=14, fontweight='bold')
            lh = axs.legend(line_handles, labels, loc='lower right', title='Hours', fontsize=8)
            axs.add_artist(lh)
            axs.legend(handles=legend_elements, loc='upper left', fontsize=10)
            for legobj in lh.legend_handles:
                legobj.set_linewidth(1.0)

        ax[0, 0].set_title(f'{date_ranges[0]}\n-1 Day', fontsize=15, fontweight='bold')
        ax[0, 1].set_title(f'{date_ranges[1]}\n+0 Day', fontsize=15, fontweight='bold')
        ax[0, 2].set_title(f'{date_ranges[2]}\n+1 Day', fontsize=15, fontweight='bold')
        ax[0, 3].set_title(f'{date_ranges[3]}\n+2 Day', fontsize=15, fontweight='bold')

        plt.suptitle(
            f'Profile Comparisons\nArgo {key} Surfacing: {ctime}\n',
            fontsize=18,
            fontweight='bold',
            y=0.98,
        )

        loc = 'interp' if interp else 'nearest'
        save_str = f'argo_{key}-{ctime}-4day-profile-comparisons-rtofs-binary-{loc}.png'
        full_file = save_dir / save_str

        plt.savefig(full_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved: {full_file}")
