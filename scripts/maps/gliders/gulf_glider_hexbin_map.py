#!/usr/bin/env python3
"""
Hexbin density heatmap of all Gulf of Mexico glider observations over 20 years.
Each hex cell is colored by the log10 count of observations within it.
Produces two output files: with and without EEZ boundaries.

Requires gulf_glider_tracks_download.py to have been run first.
"""
import matplotlib
matplotlib.use('agg')

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
map_proj = ccrs.Mercator()
data_proj = ccrs.PlateCarree()


def main():
    df = pd.read_parquet(SAVE_FILE).reset_index()
    df = df[df['longitude'] <= -81.5]
    # df = df[~((df['latitude'] < 22) & (df['longitude'] > -88))]
    df['time'] = pd.to_datetime(df['time'])

    year_min = int(df['time'].dt.year.min())
    year_max = int(df['time'].dt.year.max())
    n_obs = len(df)
    n_deployments = df['glider'].nunique()

    print(f"Loaded {n_obs:,} observations from {n_deployments} deployments ({year_min}-{year_max})")

    lons = df['longitude'].values.astype(float)
    lats = df['latitude'].values.astype(float)

    # Project lon/lat to Mercator coordinates so hexbin draws correctly on the map axes.
    xy = map_proj.transform_points(data_proj, lons, lats)
    x, y = xy[:, 0], xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    bathy_levels = [-8000, -1000, -100, 0]
    bathy_colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']
    bs = xr.open_dataset('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc')
    bs = bs.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))

    for eez in [False, True]:
        _, ax = cplt.create(
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

        h = ax.hexbin(
            x, y,
            gridsize=80,
            cmap='YlOrRd',
            bins='log',
            mincnt=1,
            zorder=10,
        )

        cbar = plt.colorbar(h, ax=ax, orientation='vertical', shrink=0.65, pad=0.02)
        cbar.set_label('Observation Count (log₁₀ scale)', fontsize=12)

        if eez:
            map_add_eez(ax, color='white', linewidth=0.75, zorder=20)

        ax.set_title(
            f'Gulf of Mexico Glider Observation Density\n'
            f'{n_deployments} Deployments  |  {n_obs:,} Observations  ({year_min}–{year_max})',
            fontsize=13, fontweight='bold', pad=10,
        )

        suffix = '_eez' if eez else ''
        out_file = PLOT_DIR / f'gulf_glider_hexbin{suffix}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
