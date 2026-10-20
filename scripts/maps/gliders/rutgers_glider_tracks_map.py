#!/usr/bin/env python3
"""
Global map of every RUCOOL/Rutgers glider deployment, all history (2003-present).
Each deployment is drawn as a simple red track on a plain Robinson projection.

Requires rutgers_glider_tracks_download.py to have been run first.
"""
import matplotlib
matplotlib.use('agg')

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

SAVE_DIR = Path.home() / "Documents" / "data" / "gliders"
SAVE_FILE = SAVE_DIR / "rutgers_glider_tracks_all.parquet"

PLOT_DIR = Path.home() / "Documents" / "plots" / "gliders" / "rutgers_fleet"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LAND_COLOR = '#f2a672'
OCEAN_COLOR = '#a8cce4'
TRACK_COLOR = 'red'

DEPLOYMENT_YEAR_RE = re.compile(r'-(\d{4})\d{4}T\d{4}$')


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def drop_bad_fixes(grp, window=61, thresh_km=75):
    """Drop track points that are wildly displaced from their local track
    (bad fixes / glitches in the source data) by comparing each point to a
    rolling median position. The median is robust to a run of several
    consecutive bad fixes as long as they're a minority of the window.
    Assumes grp is already in the track's natural (temporal) point order."""
    grp = grp.reset_index(drop=True)
    n = len(grp)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return grp
    med_lat = grp['latitude'].rolling(w, center=True, min_periods=5).median()
    med_lon = grp['longitude'].rolling(w, center=True, min_periods=5).median()
    dist = haversine_km(grp['longitude'], grp['latitude'], med_lon, med_lat)
    return grp[dist < thresh_km]


def split_at_antimeridian(lon, lat):
    """Insert NaNs where consecutive fixes jump across the dateline so plot()
    doesn't draw a spurious line straight across the map."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]
    if jumps.size == 0:
        return lon, lat
    lon = np.insert(lon, jumps + 1, np.nan)
    lat = np.insert(lat, jumps + 1, np.nan)
    return lon, lat


def deployment_years(names):
    years = []
    for name in names:
        m = DEPLOYMENT_YEAR_RE.search(name)
        if m:
            years.append(int(m.group(1)))
    return years


def main():
    df = pd.read_parquet(SAVE_FILE).reset_index()

    n_obs = len(df)
    deployment_names = df['glider'].unique()
    n_deployments = len(deployment_names)
    years = deployment_years(deployment_names)
    year_range = f"{min(years)}-{max(years)}" if years else "unknown"
    print(f"Loaded {n_obs:,} track points from {n_deployments} deployments ({year_range})")

    proj = ccrs.Robinson()
    data_proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 10.15))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_global()

    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND_COLOR,
                    edgecolor='black', linewidth=0.5, zorder=1)

    for name, grp in df.groupby('glider', sort=False):
        grp_clean = drop_bad_fixes(grp)
        lon, lat = split_at_antimeridian(
            grp_clean['longitude'].values, grp_clean['latitude'].values
        )
        ax.plot(
            lon, lat,
            linewidth=1.3, color=TRACK_COLOR,
            transform=data_proj, zorder=10,
        )

    ax.spines['geo'].set_visible(False)
    plt.tight_layout(pad=0)

    out_file = PLOT_DIR / 'rutgers_glider_tracks_global.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
