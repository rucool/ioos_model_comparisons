#!/usr/bin/env python3
"""
Bathymetry map with EEZ boundaries, planned glider transect lines,
and RU29 glider tracks (2025 and 2026).

Domain: 68W–50W, 6N–20N
Transect line: 12N from 57W to 53W, plus approach from SW Barbados
"""
import matplotlib
matplotlib.use('agg')

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cool_maps.plot as cplt
from pathlib import Path

from ioos_model_comparisons.platforms import get_glider_by_id
from ioos_model_comparisons.plotting import map_add_eez

PLOT_DIR = Path.home() / "Documents" / "plots" / "maps"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXTENT = [-68.25, -49.75, 5.75, 20.25]

# Transect along 12N from 57W to 53W
TRANSECT_LON = [-57, -53]
TRANSECT_LAT = [12, 12]

# Approach line from SW Barbados (~13.05N, 59.65W) to start of transect (12N, 57W)
APPROACH_LON = [-59.65, -57]
APPROACH_LAT = [13.05, 12]

GLIDER_DATASET = 'ru29-20250715T1838'

data_proj = ccrs.PlateCarree()


def main():
    # Fetch RU29 track data
    print(f"Fetching glider data for {GLIDER_DATASET}...")
    glider_df = get_glider_by_id(dataset_id=GLIDER_DATASET)
    glider_df.index = pd.to_datetime(glider_df.index)

    track_2025 = glider_df

    # Bathymetry
    bathy_levels = [-8000, -1000, -100, 0]
    bathy_colors = ['cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']

    bs = xr.open_dataset('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc')
    bs = bs.sel(lon=slice(EXTENT[0], EXTENT[1]), lat=slice(EXTENT[2], EXTENT[3]))

    fig, ax = cplt.create(
        EXTENT,
        bathymetry=False,
        ticks=False,
        gridlines=False,
        labelsize=12,
        figsize=(14, 9),
    )

    ax.set_xticks([-65, -60, -55], crs=data_proj)
    ax.set_xticks(np.arange(-68, -49, 1), minor=True, crs=data_proj)
    ax.set_xticklabels(['65°W', '60°W', '55°W'], fontsize=12)
    ax.set_yticks([10, 15, 20], crs=data_proj)
    ax.set_yticks(np.arange(6, 21, 1), minor=True, crs=data_proj)
    ax.set_yticklabels(['10°N', '15°N', '20°N'], fontsize=12)
    ax.gridlines(draw_labels=False, linewidth=1, color='gray',
                 alpha=0.7, linestyle='--',
                 xlocs=np.arange(-68, -49, 1),
                 ylocs=np.arange(6, 21, 1))

    ax.contourf(bs['lon'], bs['lat'], bs['z'], bathy_levels,
                colors=bathy_colors, transform=data_proj, zorder=1)

    map_add_eez(ax, color='white', linewidth=0.75, zorder=20)

    # Glider tracks
    # 2025 glider track
    if not track_2025.empty:
        ax.plot(track_2025['longitude'], track_2025['latitude'],
                color='cyan', linewidth=1.5, alpha=0.8,
                transform=data_proj, zorder=25)

    # 2026 planned transect lines
    ax.plot(APPROACH_LON, APPROACH_LAT, color='red', linewidth=2.5,
            marker='o', markersize=6, transform=data_proj, zorder=30)
    ax.plot(TRANSECT_LON, TRANSECT_LAT, color='red', linewidth=2.5,
            marker='o', markersize=6, transform=data_proj, zorder=30)

    # Legend
    legend_handles = [
        mpatches.Patch(color=bathy_colors[2], label='0 – 100 m'),
        mpatches.Patch(color=tuple(bathy_colors[1]), label='100 – 1000 m'),
        mpatches.Patch(color=bathy_colors[0], label='> 1000 m'),
        mlines.Line2D([], [], color='cyan', linewidth=1.5,
                      label='2025 Glider track'),
        mlines.Line2D([], [], color='red', linewidth=2.5, marker='o',
                      markersize=6, label='2026 Glider track'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              title='Legend', title_fontsize=9, framealpha=0.8).set_zorder(1000)

    out_file = PLOT_DIR / 'bathymetry_glider_line.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
