"""
FVON NRT — Map + Profile Plots (2026)
--------------------------------------
Produces:
  bahamas_fvon_profiles_2026_SE_coast.png  — regional map
  profiles_<WMOID>.png                     — T/depth profiles per WMO platform
  profile_summary.png                      — summary table

Requires: matplotlib, cartopy, pandas, numpy

Run:
    python plot_bahamas_fvon_2026.py

Input CSV: fvon_nrt_data_5eae_2eb3_eb69.csv  (NOAA PMEL ERDDAP download)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE = '/Users/mikesmith/Downloads/fvon_nrt_data_5eae_2eb3_eb69.csv'
DPI      = 180

# Map extent
LON0, LON1 = -81.0, -75.0
LAT0, LAT1 =  22.0,  28.0

# ── Colors / markers per WMO platform ────────────────────────────────────────
PALETTE = {
    '3BXL95N': '#7F77DD',
    '7RGXNRQ': '#D4537E',
    'AUH5G2X': '#639922',
    'CJMKBPC': '#1D9E75',
    'GHWJKFV': '#D85A30',
    'NECRNWH': '#BA7517',
    'UCEC87V': '#378ADD',
    'YXVNJJK': '#5DCAA5',
}

GEAR_MARKERS = {
    'TRAPS (NEI)':   '^',
    'DIVING FIXED':  'o',
    'PURSE SEINES':  's',
    'HANDLINES AND HAND-OPERATED POLE-AND-LINES': 'D',
}

GEAR_LABELS = {
    'TRAPS (NEI)':   'Traps',
    'DIVING FIXED':  'Diving fixed',
    'PURSE SEINES':  'Purse seines',
    'HANDLINES AND HAND-OPERATED POLE-AND-LINES': 'Handlines',
}

# ── Load data — no filtering ──────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE, skiprows=[1])
df['time'] = pd.to_datetime(df['time'], utc=True)

# One row per unique profile location (for map)
prof = df.drop_duplicates(subset=['time', 'wmo_platform_code']).copy()

# Per-profile stats (each time x wmo x segment_type = one cast)
prof_stats = df.groupby(['wmo_platform_code', 'time', 'segment_type']).agg(
    n_points  = ('depth', 'count'),
    max_depth = ('depth', 'max'),
    min_temp  = ('temperature', 'min'),
    max_temp  = ('temperature', 'max'),
).reset_index().sort_values(['wmo_platform_code', 'time'])

all_wmo = sorted(df['wmo_platform_code'].unique())
print(f"All WMO IDs ({len(all_wmo)}): {all_wmo}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAP
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating map...")

fig = plt.figure(figsize=(16, 12), dpi=DPI)
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
ax.set_extent([LON0, LON1, LAT0, LAT1], crs=ccrs.PlateCarree())
ax.set_facecolor('#aaccdd')

SCALE = '10m'   # '50m' for faster render; '10m' for publication quality
ax.add_feature(cfeature.LAND      .with_scale(SCALE), facecolor='#ddd8bb', zorder=2)
ax.add_feature(cfeature.COASTLINE .with_scale(SCALE), edgecolor='#555555', linewidth=0.7, zorder=3)
ax.add_feature(cfeature.BORDERS   .with_scale(SCALE), edgecolor='#777777', linewidth=0.5,
               linestyle='--', zorder=3)
ax.add_feature(cfeature.STATES    .with_scale(SCALE), edgecolor='#999999', linewidth=0.35,
               linestyle=':', zorder=3)
ax.add_feature(cfeature.LAKES     .with_scale(SCALE), facecolor='#aaccdd', edgecolor='#555555',
               linewidth=0.4, zorder=3)
ax.add_feature(cfeature.RIVERS    .with_scale(SCALE), edgecolor='#6699aa', linewidth=0.4,
               zorder=3)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                  alpha=0.45, linestyle='--', crs=ccrs.PlateCarree())
gl.top_labels   = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(-90, -64, 5))
gl.ylocator = mticker.FixedLocator(range(20, 37, 2))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'color': '#444'}
gl.ylabel_style = {'size': 9, 'color': '#444'}

legend_handles = []
for wmo, grp in prof.groupby('wmo_platform_code'):
    gear   = grp['vessel_gear_type'].iloc[0]
    marker = GEAR_MARKERS.get(gear, 'o')
    color  = PALETTE.get(wmo, '#888888')
    n      = len(grp)
    ms     = 30 if n > 50 else 80

    ax.scatter(grp['longitude'], grp['latitude'],
               c=color, marker=marker, s=ms,
               linewidths=0.6, edgecolors='white',
               alpha=0.85, zorder=7,
               transform=ccrs.PlateCarree())

    label = f'{wmo}  ·  {GEAR_LABELS.get(gear, gear)}  (n={n:,})'
    legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='white',
                       linewidth=0.5, label=label))

geo_labels = [
    (-80.2, 25.8,  'Florida'),
    (-77.3, 24.2,  'Bahamas'),
    (-82.5, 23.2,  'Cuba'),
    (-72.5, 21.0,  'Hispaniola'),
    (-75.5, 34.5,  'North Carolina'),
    (-81.5, 30.3,  'Georgia / S.C.'),
    (-84.5, 30.4,  'Florida\nPanhandle'),
    (-85.0, 22.5,  'Gulf of\nMexico'),
    (-68.5, 22.5,  'Atlantic\nOcean'),
]
for lon, lat, name in geo_labels:
    ax.text(lon, lat, name, transform=ccrs.PlateCarree(),
            fontsize=8, color='#333333', ha='center',
            style='italic', zorder=6,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1.5))

zoom_box = plt.Polygon(
    [[-80.8, 24.1], [-76.2, 24.1], [-76.2, 27.1], [-80.8, 27.1]],
    closed=True, fill=False,
    edgecolor='#1D9E75', linewidth=1.5, linestyle='-',
    zorder=8, transform=ccrs.PlateCarree())
ax.add_patch(zoom_box)
ax.text(-78.5, 27.35, 'Profile cluster', transform=ccrs.PlateCarree(),
        fontsize=7.5, color='#1D9E75', ha='center', fontweight='500', zorder=8)

leg = ax.legend(handles=legend_handles,
                title='FVON platforms  ·  gear type  ·  # profiles',
                title_fontsize=9, fontsize=8.5,
                loc='lower left', framealpha=0.93,
                edgecolor='#cccccc', borderpad=0.9,
                handlelength=1.3, handleheight=1.1)
leg.get_title().set_fontweight('500')

ax.set_title(
    'FVON NRT — Fishing Vessel Temperature Profiles, SE U.S. Coast & Bahamas (2026)',
    fontsize=12, fontweight='500', pad=10, loc='left')

d0   = prof['time'].min().strftime('%d %b')
d1   = prof['time'].max().strftime('%d %b %Y')
foot = (f"{len(prof):,} profiles  ·  {prof['wmo_platform_code'].nunique()} WMO platforms  ·  "
        f"{len(df):,} obs. rows  ·  {d0} – {d1}")
ax.text(0.5, -0.020, foot, transform=ax.transAxes,
        fontsize=7.5, ha='center', va='top', color='#555')
ax.text(0.5, -0.045, 'Source: NOAA PMEL ERDDAP / Fishing Vessel Observing Network (FVON)',
        transform=ax.transAxes, fontsize=6.5, ha='center', va='top', color='#888')

plt.tight_layout()
map_out = 'bahamas_fvon_profiles_2026_SE_coast.png'
plt.savefig(map_out, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved → {map_out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROFILE PLOTS — one figure per WMO, all casts overlaid on a single axes
#    Colored by date (plasma), solid=Down / dashed=Up, depth increasing downward
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating profile plots...")

for wmo in all_wmo:
    wmo_df    = df[df['wmo_platform_code'] == wmo].copy()
    gear      = wmo_df['vessel_gear_type'].iloc[0]
    wmo_stats = prof_stats[prof_stats['wmo_platform_code'] == wmo]
    n_prof    = len(wmo_df.groupby(['time', 'segment_type']))

    # Color each cast by its timestamp
    times_sorted = sorted(wmo_df['time'].unique())
    t_index      = {t: i for i, t in enumerate(times_sorted)}
    cmap         = matplotlib.colormaps.get_cmap('plasma')
    norm         = mcolors.Normalize(vmin=0, vmax=max(len(times_sorted) - 1, 1))

    seg_ls = {'Profiling Down': '-', 'Profiling Up': '--'}

    fig, ax = plt.subplots(figsize=(7, 9), layout='constrained')

    for (t, seg), grp in wmo_df.groupby(['time', 'segment_type']):
        grp_s = grp.sort_values('depth')          # depth ascending (surface→bottom)
        ax.plot(grp_s['temperature'], grp_s['depth'],
                color=cmap(norm(t_index[t])),
                linewidth=0.8, linestyle=seg_ls.get(seg, '-'), alpha=0.65)

    ax.invert_yaxis()                              # 0 at top, max depth at bottom
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=10)

    # Colorbar — date
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02)
    n_ticks  = min(8, len(times_sorted))
    tick_idx = np.linspace(0, len(times_sorted) - 1, n_ticks, dtype=int)
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([
        pd.Timestamp(times_sorted[i]).strftime('%b %d') for i in tick_idx
    ])
    cbar.set_label('Profile date (UTC)', fontsize=9)

    # Segment legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='gray', lw=1.2, ls='-',  label='Profiling Down'),
        Line2D([0], [0], color='gray', lw=1.2, ls='--', label='Profiling Up'),
    ], fontsize=9, loc='lower right')

    stat_txt = (
        f"n profiles: {n_prof}  |  "
        f"max depth: {wmo_stats['max_depth'].max():.1f} m  |  "
        f"pts/profile: {wmo_stats['n_points'].min()}–{wmo_stats['n_points'].max()}"
    )
    ax.set_title(stat_txt, fontsize=9, color='#555', pad=4)
    fig.suptitle(
        f"WMO {wmo}  ·  {GEAR_LABELS.get(gear, gear)}  ·  {n_prof} profiles",
        fontsize=12, fontweight='500')

    out = f'profiles_{wmo}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating summary table...")

# Longitude filter
LON_FILTER_MIN, LON_FILTER_MAX = -81.0, -75.0
df_filt = df[(df['longitude'] >= LON_FILTER_MIN) & (df['longitude'] <= LON_FILTER_MAX)].copy()


summary = df_filt.groupby('wmo_platform_code').apply(lambda g: pd.Series({
    'gear_type':     g['vessel_gear_type'].iloc[0],
    'platform_type': g['gear_type'].iloc[0],
    'n_profiles':    g.groupby(['time', 'segment_type']).ngroups,
    'n_obs_rows':    len(g),
    'max_depth_m':   g['depth'].max(),
    'min_depth_m':   g['depth'].min(),
    'date_start':    g['time'].min().strftime('%Y-%m-%d'),
    'date_end':      g['time'].max().strftime('%Y-%m-%d'),
    'temp_min_C':    g['temperature'].min(),
    'temp_max_C':    g['temperature'].max(),
})).reset_index()

pts_filt = df_filt.groupby(['wmo_platform_code', 'time', 'segment_type']).agg(
    n_points=('depth', 'count')).reset_index()
pts = pts_filt.groupby('wmo_platform_code')['n_points'].agg(['min', 'max', 'mean'])
pts.columns = ['pts_min', 'pts_max', 'pts_mean']
summary = summary.merge(pts, on='wmo_platform_code')
summary['pts_mean'] = summary['pts_mean'].round(1)

# Render as figure
col_labels = [
    'WMO ID', 'Platform type', 'Gear type',
    'Profiles', 'Obs rows', 'Max depth (m)',
    'Pts/profile\n(min – max)', 'Date start', 'Date end',
    'Temp range (°C)'
]
table_data = []
for _, r in summary.iterrows():
    table_data.append([
        r['wmo_platform_code'],
        r['platform_type'],
        GEAR_LABELS.get(r['gear_type'], r['gear_type']),
        int(r['n_profiles']),
        int(r['n_obs_rows']),
        f"{r['max_depth_m']:.1f}",
        f"{int(r['pts_min'])} – {int(r['pts_max'])}",
        r['date_start'],
        r['date_end'],
        f"{r['temp_min_C']:.2f} – {r['temp_max_C']:.2f}",
    ])

fig, ax = plt.subplots(figsize=(15, 0.65 * (len(summary) + 2.5)))
ax.set_axis_off()

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i, row in enumerate(summary.itertuples(), start=1):
    wmo   = row.wmo_platform_code
    color = PALETTE.get(wmo, '#888888')
    tbl[i, 0].set_facecolor(color)
    tbl[i, 0].set_text_props(color='white', fontweight='500')
    shade = '#f4f6f8' if i % 2 == 0 else 'white'
    for j in range(1, len(col_labels)):
        tbl[i, j].set_facecolor(shade)

ax.set_title(
    f'FVON NRT 2026 — Profile Summary by WMO Platform  '
    f'(lon {LON_FILTER_MIN}° to {LON_FILTER_MAX}°)',
    fontsize=12, fontweight='500', pad=16)
fig.text(0.5, 0.01,
         'Platform type: FIXED = anchored gear, MOBILE = active fishing  ·  '
         'Source: NOAA PMEL ERDDAP / Fishing Vessel Observing Network',
         ha='center', fontsize=7.5, color='#888')

plt.tight_layout()
out = 'profile_summary.png'
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-PROFILE CSV — one row per cast (lon-filtered)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating per-profile CSV...")

prof_tbl = df_filt.groupby(['wmo_platform_code', 'time', 'segment_type']).agg(
    n_points      = ('depth',            'count'),
    lat           = ('latitude',         'first'),
    lon           = ('longitude',        'first'),
    depth_min     = ('depth',            'min'),
    depth_max     = ('depth',            'max'),
    temp_min      = ('temperature',      'min'),
    temp_max      = ('temperature',      'max'),
    gear          = ('vessel_gear_type', 'first'),
    platform_type = ('gear_type',        'first'),
).reset_index()

surf_t = df_filt.groupby(['wmo_platform_code', 'time', 'segment_type']).apply(
    lambda g: g.sort_values('depth').iloc[0]['temperature']).rename('temp_surface')
deep_t = df_filt.groupby(['wmo_platform_code', 'time', 'segment_type']).apply(
    lambda g: g.sort_values('depth').iloc[-1]['temperature']).rename('temp_bottom')

prof_tbl = prof_tbl.merge(surf_t, on=['wmo_platform_code', 'time', 'segment_type'])
prof_tbl = prof_tbl.merge(deep_t, on=['wmo_platform_code', 'time', 'segment_type'])
prof_tbl['temp_range'] = (prof_tbl['temp_max'] - prof_tbl['temp_min']).round(4)
prof_tbl['gear']       = prof_tbl['gear'].map(lambda g: GEAR_LABELS.get(g, g))
prof_tbl = prof_tbl.sort_values(
    ['wmo_platform_code', 'time', 'segment_type']).reset_index(drop=True)

for col in ['lat', 'lon', 'depth_min', 'depth_max',
            'temp_min', 'temp_max', 'temp_surface', 'temp_bottom']:
    prof_tbl[col] = prof_tbl[col].round(4)

prof_tbl['time'] = prof_tbl['time'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')

prof_tbl = prof_tbl[['wmo_platform_code', 'platform_type', 'gear',
                      'time', 'segment_type', 'lat', 'lon',
                      'n_points', 'depth_min', 'depth_max',
                      'temp_surface', 'temp_bottom', 'temp_min', 'temp_max', 'temp_range']]

out = 'profile_table_per_profile.csv'
prof_tbl.to_csv(out, index=False)
print(f"  Saved → {out}  ({len(prof_tbl)} rows)")

print("\nDone. Output files:")
print(f"  bahamas_fvon_profiles_2026_SE_coast.png")
for wmo in all_wmo:
    print(f"  profiles_{wmo}.png")
print(f"  profile_summary.png")
print(f"  profile_table_per_profile.csv")