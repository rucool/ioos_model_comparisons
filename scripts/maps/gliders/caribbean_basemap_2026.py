#!/usr/bin/env python3
"""
2026 Caribbean base map - no Argo, no gliders, no FVON, no EEZ.

Companion to caribbean_glider_tracks_2026.py, meant to be shown as the "before" slide
that flips to caribbean_glider_argo_tracks_2026.png (gliders + Argo + FVON, no EEZ) in
a presentation. To make that flip look like the map itself doesn't move, this uses the
exact same EXTENT/figsize/padding/bathymetry/gridlines settings, and reserves the same
title (2 lines, same fontsize) and bottom legend row (same handles/fontsize/position,
just fully transparent) that the data version draws - since both scripts save with
bbox_inches='tight', pad_inches=0.15, dpi=300, matching that reserved space is what
keeps the two PNGs pixel-identical in size (verified: both are 3907x2493).
"""
import cartopy.crs as ccrs
import cool_maps.plot as cplt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXTENT = [-89, -50, 7, 27]
SAVE_PATH = '/Users/mikesmith/Documents/caribbean_basemap_2026.png'

DATA_PROJECTION = ccrs.PlateCarree()

fig, ax = cplt.create(
    EXTENT,
    bathymetry=True,
    bathymetry_method='banded',
    gridlines=True,
    labelsize=12,
    figsize=(16, 8.5),
    padding=(0, 0.25),
)

# Invisible legend row, same shape as the data version's (track, position, Argo, FVON)
# so it reserves the same vertical space below the axes without showing anything.
track_proxy = Line2D([], [], color='red', linewidth=3, solid_capstyle='round', alpha=0)
marker_proxy = Line2D([], [], marker='^', markersize=13, linestyle='None',
                       markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, alpha=0)
argo_proxy = Line2D([], [], marker='o', markersize=7, linestyle='None',
                     markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.4, alpha=0)
fvon_proxy = Line2D([], [], marker='D', markersize=7, linestyle='None',
                     markerfacecolor='blue', markeredgecolor='black', markeredgewidth=0.4, alpha=0)
handles = [track_proxy, marker_proxy, argo_proxy, fvon_proxy]
labels = ['', '', '', '']
legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    ncol=len(handles), fontsize=10, frameon=True, facecolor='none')
legend.get_frame().set_edgecolor('none')

# Same title shape (2 lines, same fontsize/weight) as the data version, different text.
ax.set_title("2026 Caribbean\nBase Map", fontsize=16, fontweight='bold')

plt.savefig(SAVE_PATH, bbox_inches='tight', pad_inches=0.15, dpi=300)
print(f"Saved {SAVE_PATH}")
plt.close(fig)
