#!/usr/bin/env python3
"""
2026 Caribbean glider (and Argo, and FVON) tracks maps.

Pulls every glider dataset active from 2026-06-01 (hurricane season start) through
today from the IOOS Glider DAC, plus every Argo float profile and every FVON profile
from the same window, within an extended Caribbean / tropical western Atlantic box.
Renders four maps over tri-color bathymetry:

  1. caribbean_glider_tracks_2026.png              gliders only
  2. caribbean_glider_tracks_2026_eez.png           gliders only, + EEZ boundaries
  3. caribbean_glider_argo_tracks_2026.png          gliders + Argo (green) + FVON (blue)
  4. caribbean_glider_argo_tracks_2026_eez.png      gliders + Argo + FVON, + EEZ boundaries

Glider tracks/markers are uniformly red - individual platforms aren't distinguished.

The box is the production "Caribbean" region's extent (see
ioos_model_comparisons.regions, key "caribbean") widened east from -58 to -50 so RU29's
full 2026 track (which reaches ~-50.8 this year) fits on the map.

At the time this was written there was no available source for the SPICE cruise ship
track (no file in the repo, no obvious API), so it is not included here. Add it as a
separate ax.plot(..., transform=DATA_PROJECTION) call once track data is available.
"""
import datetime as dt

import cartopy.crs as ccrs
import cool_maps.plot as cplt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ioos_model_comparisons.platforms import get_active_gliders, get_argo_floats_by_time
from ioos_model_comparisons.plotting import map_add_eez

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
T0 = dt.datetime(2026, 6, 1)  # 2026 Atlantic hurricane season start
T1 = dt.datetime.utcnow()

# Caribbean extent, widened east from the production region's -58 to stop at 50W
# (fully captures RU29's 2026 track) and north from 23.5 to 27 (fully captures the
# FVON Bahamas cluster, which runs up to ~26.4N - well north of the production
# region's original boundary).
EXTENT = [-89, -50, 7, 27]

SAVE_DIR = '/Users/mikesmith/Documents'

GLIDER_COLOR = 'red'
ARGO_COLOR = 'green'
FVON_COLOR = 'blue'

DATA_PROJECTION = ccrs.PlateCarree()


def drop_pacific(df):
    """Drop rows on the Pacific side of Central America.

    EXTENT's rectangular southwest corner reaches into the Pacific (Costa Rica's
    Nicoya/Osa coast and Panama's Gulf of Chiriqui) even though the box is meant to
    cover the Caribbean. That area only exists south of about lat 11 - west of -84.5
    it's Pacific up to lat 11 (Costa Rica), and between -84.5 and -77.5 it's Pacific
    only south of lat 9 (Panama's Gulf of Chiriqui/Panama; north of that, at that
    longitude, is open Caribbean Sea off Panama/Colombia). Both clauses are latitude-
    bounded so this never touches anything north of ~11 - e.g. the Yucatan/CICESE
    glider up near 19N, which is west of -84.5 but nowhere near the Pacific.
    """
    lon, lat = df['lon'], df['lat']
    pacific = ((lon < -84.5) & (lat < 11.0)) | ((lon >= -84.5) & (lon < -77.5) & (lat < 9.0))
    return df[~pacific]


def drop_northwest(df):
    """Drop rows north of 22N and west of 80W (Gulf of Mexico / stray northwest data)."""
    lon, lat = df['lon'], df['lat']
    return df[~((lat > 22.0) & (lon < -80.0))]


# ---------------------------------------------------------------------------
# Fetch every glider active in the box since June 1, 2026
# ---------------------------------------------------------------------------
print(f"Fetching active gliders {T0:%Y-%m-%d} to {T1:%Y-%m-%d} in extent {EXTENT}...")
gliders = get_active_gliders(
    bbox=EXTENT,
    t0=T0,
    t1=T1,
    variables=['time', 'longitude', 'latitude', 'profile_id'],
    parallel=False,
)

if gliders.empty:
    raise SystemExit("No glider data returned - nothing to plot.")

gliders = gliders.reset_index()

# get_active_gliders uses the bbox only to pick which datasets to fetch, not to trim
# the rows it returns, so some deployments (e.g. the WHOI Spray gliders) carry a lot of
# track outside the Caribbean. Clip to the plotted extent.
gliders = gliders[
    gliders['lon'].between(EXTENT[0], EXTENT[1])
    & gliders['lat'].between(EXTENT[2], EXTENT[3])
]
gliders = drop_pacific(gliders)
gliders = drop_northwest(gliders)

# Group deployments by platform (strip the "-YYYYMMDDTHHMM" deployment suffix) so a
# platform that redeployed mid-year (e.g. uvi_02, ng598) is one continuous track
# instead of two.
gliders['platform'] = gliders['glider'].str.replace(r'-\d{8}T\d{4}$', '', regex=True)

platforms = sorted(gliders['platform'].unique())
print(f"Platforms ({len(platforms)}): {', '.join(platforms)}")

# Total profiles across every deployment - profile_id is unique within a deployment
# but not necessarily across deployments, so count nunique() per deployment and sum.
total_glider_profiles = int(gliders.groupby('glider')['profile_id'].nunique().sum())
print(f"Total glider profiles: {total_glider_profiles}")

# ---------------------------------------------------------------------------
# Fetch every Argo float profile in the box during hurricane season
# ---------------------------------------------------------------------------
print(f"Fetching Argo floats {T0:%Y-%m-%d} to {T1:%Y-%m-%d} in extent {EXTENT}...")
argo = get_argo_floats_by_time(bbox=EXTENT, time_start=T0, time_end=T1).reset_index()
argo = drop_pacific(argo)
argo = drop_northwest(argo)
n_argo_floats = argo['argo'].nunique()
n_argo_profiles = len(argo)
print(f"Argo floats: {n_argo_floats}, profiles: {n_argo_profiles}")

# ---------------------------------------------------------------------------
# Fetch every FVON profile in the box during hurricane season
# ---------------------------------------------------------------------------
print(f"Fetching FVON profiles {T0:%Y-%m-%d} to {T1:%Y-%m-%d} in extent {EXTENT}...")
fvon_url = (
    "https://data.pmel.noaa.gov/generic/erddap/tabledap/fvon_nrt_data.csv"
    "?wmo_platform_code,time,latitude,longitude,segment_type"
    f"&time>={T0:%Y-%m-%dT%H:%M:%SZ}"
    f"&time<={T1:%Y-%m-%dT%H:%M:%SZ}"
    '&orderBy("wmo_platform_code,time")'
)
fvon_raw = pd.read_csv(fvon_url, skiprows=[1], parse_dates=['time'])
fvon_raw = fvon_raw.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
fvon_raw['time'] = pd.to_datetime(fvon_raw['time']).dt.tz_convert(None)
# Keep downcast only, matching fvon_profile_model_comparisons.py - each (platform, time)
# is one profile cast spread across many depth rows, so downcast-only + collapsing to
# one lon/lat per cast (mean, like that script does) avoids double-counting/duplicating
# the same physical profile via its paired up/down segments.
fvon_raw = fvon_raw[fvon_raw['segment_type'] == 'Profiling Down']
fvon = fvon_raw.groupby(['wmo_platform_code', 'time'], as_index=False)[['lon', 'lat']].mean()
fvon = fvon[fvon['lon'].between(EXTENT[0], EXTENT[1]) & fvon['lat'].between(EXTENT[2], EXTENT[3])]
# drop_pacific only (not drop_northwest) - FVON's own Bahamas deployments legitimately
# sit north of 22N/west of 80W, the exact area drop_northwest was built to exclude for
# the Gulf-of-Mexico bleed seen in the glider/Argo pulls.
fvon = drop_pacific(fvon)
n_fvon_platforms = fvon['wmo_platform_code'].nunique()
n_fvon_profiles = len(fvon)
print(f"FVON platforms: {n_fvon_platforms}, profiles: {n_fvon_profiles}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_map(save_path, include_argo, eez):
    fig, ax = cplt.create(
        EXTENT,
        bathymetry=True,
        bathymetry_method='banded',
        gridlines=True,
        labelsize=12,
        figsize=(16, 8.5),
        padding=(0, 0.25),  # no lon padding so the map stops exactly at 50W
    )

    if eez:
        map_add_eez(ax, zorder=15, color='white', linewidth=2)

    # Stacking order (bottom to top): Argo, glider tracks, glider triangles, FVON.
    # Argo is dense enough (especially around the Caymans) that plotting it above the
    # glider tracks blots out stretches of them - put it on the bottom instead, so the
    # full glider track stays visible with Argo floats showing through underneath it.
    if include_argo:
        ax.plot(argo['lon'], argo['lat'], marker='o', markersize=7, linestyle='None',
                 markerfacecolor=ARGO_COLOR, markeredgecolor='black',
                 markeredgewidth=0.4, transform=DATA_PROJECTION, zorder=20)

    for platform in platforms:
        pdf = gliders[gliders['platform'] == platform]

        for _deployment, ddf in pdf.groupby('glider'):
            ddf = ddf.sort_values('time')
            ax.plot(ddf['lon'], ddf['lat'], linewidth=3, color=GLIDER_COLOR,
                     solid_capstyle='round', transform=DATA_PROJECTION, zorder=50)

        # Mark the most recent fix for this platform
        last = pdf.sort_values('time').iloc[-1]
        ax.plot(last['lon'], last['lat'], marker='^', markersize=13, linestyle='None',
                 markerfacecolor=GLIDER_COLOR, markeredgecolor='black',
                 markeredgewidth=1, transform=DATA_PROJECTION, zorder=100)

    if include_argo:
        ax.plot(fvon['lon'], fvon['lat'], marker='D', markersize=7, linestyle='None',
                 markerfacecolor=FVON_COLOR, markeredgecolor='black',
                 markeredgewidth=0.4, transform=DATA_PROJECTION, zorder=150)

    # Bottom legend for the track/marker symbols. Built from standalone proxy artists
    # (rather than reusing a plotted handle) so each entry's swatch exactly matches
    # what's on the map - a handle straight off ax.plot(..., marker=...) without
    # linestyle='None' draws a spurious connecting line through the marker.
    track_proxy = Line2D([], [], color=GLIDER_COLOR, linewidth=3, solid_capstyle='round')
    marker_proxy = Line2D([], [], marker='^', markersize=13, linestyle='None',
                           markerfacecolor=GLIDER_COLOR, markeredgecolor='black', markeredgewidth=1)
    handles = [track_proxy, marker_proxy]
    labels = ['Glider track', 'Most recent position']
    if include_argo:
        argo_proxy = Line2D([], [], marker='o', markersize=7, linestyle='None',
                             markerfacecolor=ARGO_COLOR, markeredgecolor='black', markeredgewidth=0.4)
        handles.append(argo_proxy)
        labels.append('Argo float')
        fvon_proxy = Line2D([], [], marker='D', markersize=7, linestyle='None',
                             markerfacecolor=FVON_COLOR, markeredgecolor='black', markeredgewidth=0.4)
        handles.append(fvon_proxy)
        labels.append('FVON profile')
    if eez:
        eez_proxy = Line2D([], [], color='white', linewidth=2)
        handles.append(eez_proxy)
        labels.append('EEZ boundary')

    # Note: don't also ax.add_artist() this legend - on top of ax.legend() already
    # attaching it, that second attachment makes savefig(bbox_inches='tight') drop the
    # legend from the saved image entirely (reproduced in isolation; root cause unclear,
    # but ax.legend()'s own return value is already attached and is all that's needed).
    # facecolor is light gray, not the default white, so the white EEZ line swatch
    # (which matches its color on the map) doesn't disappear against the legend background.
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              ncol=len(handles), fontsize=16, markerscale=1.3, handlelength=2.5,
              borderpad=0.7, columnspacing=1.8, frameon=True, facecolor='0.9')

    # Summary stats as a legend-style box in the upper right of the map.
    stats_lines = [f"{len(platforms)} gliders", f"{total_glider_profiles} glider profiles"]
    if include_argo:
        stats_lines += [f"{n_argo_floats} Argo floats", f"{n_argo_profiles} Argo profiles"]
        stats_lines += [f"{n_fvon_platforms} FVON platforms", f"{n_fvon_profiles} FVON profiles"]
    ax.text(0.98, 0.98, "\n".join(stats_lines), transform=ax.transAxes, ha='right', va='top',
            fontsize=13, fontweight='bold', linespacing=1.6, zorder=200,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

    title = "2026 Caribbean Glider" + (" & Argo & FVON" if include_argo else "") + " Tracks"
    ax.set_title(f"{title}\n{T0:%Y-%m-%d} – {T1:%Y-%m-%d}", fontsize=16, fontweight='bold')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.15, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)


make_map(f'{SAVE_DIR}/caribbean_glider_tracks_2026.png', include_argo=False, eez=False)
make_map(f'{SAVE_DIR}/caribbean_glider_tracks_2026_eez.png', include_argo=False, eez=True)
make_map(f'{SAVE_DIR}/caribbean_glider_argo_tracks_2026.png', include_argo=True, eez=False)
make_map(f'{SAVE_DIR}/caribbean_glider_argo_tracks_2026_eez.png', include_argo=True, eez=True)
