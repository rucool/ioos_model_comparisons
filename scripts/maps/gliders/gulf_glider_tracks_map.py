#!/usr/bin/env python3
"""
Map of all Gulf of Mexico glider deployments over the past 20 years.
Each deployment is drawn as a line colored by year (plasma colormap).
Produces two output files: with and without EEZ boundaries.

Requires gulf_glider_tracks_download.py to have been run first.
"""
import matplotlib
matplotlib.use('agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cool_maps.plot as cplt
from pathlib import Path

from ioos_model_comparisons.plotting import map_add_eez

SAVE_DIR = Path.home() / "Documents" / "data" / "gliders"
SAVE_FILE = SAVE_DIR / "gulf_glider_tracks_20yr.parquet"

PLOT_DIR = Path.home() / "Documents" / "plots" / "gliders" / "gulf_of_mexico"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXTENT = [-99, -79.5, 17.5, 31.5]
data_proj = ccrs.PlateCarree()


def main():
    df = pd.read_parquet(SAVE_FILE).reset_index()
    df = df[df['longitude'] <= -81.5]
    # df = df[~((df['latitude'] < 22) & (df['longitude'] > -88))]
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year

    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    n_obs = len(df)
    n_deployments = df['glider'].nunique()

    print(f"Loaded {n_obs:,} observations from {n_deployments} deployments ({year_min}-{year_max})")

    bathy_levels = [-8000, -1000, -100, 0]
    bathy_colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']
    bs = xr.open_dataset('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc')
    bs = bs.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))

    cmap = cm.get_cmap('plasma', year_max - year_min + 1)
    norm = mcolors.Normalize(vmin=year_min, vmax=year_max)

    for eez in [False, True]:
        fig, ax = cplt.create(
            EXTENT,
            bathymetry=False,
            gridlines=False,
            labelsize=12,
            figsize=(14, 9),
        )

        ax.contourf(bs['lon'], bs['lat'], bs['z'], bathy_levels,
                    colors=bathy_colors, transform=data_proj, zorder=1)

        bathy_legend = [
            mpatches.Patch(color=bathy_colors[0], label='> 1000 m'),
            mpatches.Patch(color=tuple(bathy_colors[1]), label='100 – 1000 m'),
            mpatches.Patch(color=bathy_colors[2], label='0 – 100 m'),
        ]
        leg = ax.legend(handles=bathy_legend, loc='lower right', fontsize=9,
                        title='Depth', title_fontsize=9, framealpha=0.8)
        leg.set_zorder(1000)

        for name, grp in df.groupby('glider'):
            grp_sorted = grp.sort_values('time')
            year_mean = grp_sorted['year'].mean()
            color = cmap(norm(year_mean))
            ax.plot(
                grp_sorted['longitude'], grp_sorted['latitude'],
                linewidth=0.9, alpha=0.75, color=color,
                transform=data_proj, zorder=10,
            )

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.65, pad=0.02)
        cbar.set_label('Year', fontsize=12)
        tick_step = max(1, (year_max - year_min) // 10)
        cbar.set_ticks(np.arange(year_min, year_max + 1, tick_step))

        if eez:
            map_add_eez(ax, color='white', linewidth=0.75, zorder=20)

        ax.set_title(
            f'Gulf of Mexico Glider Deployments\n'
            f'{n_deployments} Deployments  |  {n_obs:,} Observations  ({year_min}–{year_max})',
            fontsize=13, fontweight='bold', pad=10,
        )

        suffix = '_eez' if eez else ''
        out_file = PLOT_DIR / f'gulf_glider_tracks{suffix}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
