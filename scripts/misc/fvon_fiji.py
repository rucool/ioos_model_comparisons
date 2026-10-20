"""
FishSOOP South Pacific — Map + Profile Plots (2026)
----------------------------------------------------
Produces:
  fishsoop_fiji_nc_2026_map.png          — regional map
  profiles_<serial>.png                  — T/depth profiles per instrument
  profile_summary.png                    — summary table
  profile_table_per_deployment.csv       — one row per deployment

Input: /Users/mikesmith/Downloads/vessel_fishsoop_realtime_qc/  (parquet)
"""

import datetime as dt
import platform
import pandas as pd
import numpy as np
import xarray as xr
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from erddapy import ERDDAP
from requests.exceptions import HTTPError as rHTTPError

DPI = 180

# ── Bathymetry ─────────────────────────────────────────────────────────────────
BATHY_FILE = ('/Users/mikesmith/Documents/data/SRTM15_V2.4.nc'
              if platform.system() == 'Darwin'
              else '/home/hurricaneadm/data/srtm15/SRTM15_V2.4.nc')
BATHY_LEVELS = [-30000, -6000, -1000, -100, 0]
BATHY_COLORS = ['royalblue', 'cornflowerblue', cfeature.COLORS['water'], 'lightsteelblue']


def add_tricolor_bathymetry(ax, lon_min, lon_max, lat_min, lat_max):
    """Add tricolor bathymetry contourf, handling antimeridian crossing.

    lon_min/lon_max should be in 0–360 space (e.g. 159.5, 190.5).
    Data is loaded by rolling SRTM15 to 0–360 then slicing, so the slice
    spans the antimeridian without splitting.  Cartopy's PlateCarree
    transform accepts longitudes > 180 and reprojects them correctly.
    """
    bathy = xr.open_dataset(BATHY_FILE)
    # Roll to 0–360 so the slice is contiguous across the antimeridian
    bathy = bathy.assign_coords(lon=(bathy['lon'] % 360)).sortby('lon')
    bathy = bathy.sel(lon=slice(lon_min, lon_max),
                      lat=slice(lat_min, lat_max))
    bathy = bathy.rename({'lon': 'longitude', 'lat': 'latitude'})
    ax.contourf(
        bathy['longitude'], bathy['latitude'], bathy['z'],
        BATHY_LEVELS, colors=BATHY_COLORS,
        transform=ccrs.PlateCarree(), zorder=0.05,
    )
    bathy.close()

# ── Load & filter ─────────────────────────────────────────────────────────────
df_all = pd.read_parquet('/Users/mikesmith/Downloads/vessel_fishsoop_realtime_qc/')
df_all['TIME'] = pd.to_datetime(df_all['TIME'], utc=True)

# 2026, Fiji + New Caledonia bounding box
fiji_mask = (df_all['LATITUDE'].between(-26, -5)) & (df_all['LONGITUDE'].between(160, 180))
nc_mask   = (df_all['LATITUDE'].between(-26, -5)) & (df_all['LONGITUDE'].between(0, 10))
df = df_all[
    (df_all['TIME'].dt.year == 2026) & (nc_mask | fiji_mask)
].copy()

print(f"Filtered: {len(df):,} rows, {df['instrument_serial_number'].nunique()} instruments")
print(f"Date range: {df['TIME'].min()} → {df['TIME'].max()}")

# ── Deployment grouping key ────────────────────────────────────────────────────
# 'TIME' is per-observation (ms precision); 'timestamp' identifies a deployment
GROUP_KEYS = ['instrument_serial_number', 'timestamp']

# ── Per-deployment stats ───────────────────────────────────────────────────────
dep_stats = df.groupby(GROUP_KEYS).agg(
    n_points  = ('DEPTH',       'count'),
    max_depth = ('DEPTH',       'max'),
    min_temp  = ('TEMPERATURE', 'min'),
    max_temp  = ('TEMPERATURE', 'max'),
    time_utc  = ('TIME',        'first'),  # representative time for colorbar labels
    lat       = ('LATITUDE',    'first'),
    lon       = ('LONGITUDE',   'first'),
).reset_index().sort_values(GROUP_KEYS)

# ── Cast-splitting helper ─────────────────────────────────────────────────────
def split_casts(grp):
    """Split a time-sorted deployment into individual up/down cast segments.
    Returns a list of DataFrames, one per cast."""
    grp_s     = grp.sort_values('TIME').reset_index(drop=True)
    depths    = grp_s['DEPTH'].values
    direction = np.sign(np.diff(depths)).astype(float)
    for i in range(len(direction)):
        if direction[i] == 0 and i > 0:
            direction[i] = direction[i - 1]
    reversals = np.where(np.diff(direction) != 0)[0] + 1
    split_idx = np.concatenate([[0], reversals, [len(grp_s)]])
    return [grp_s.iloc[split_idx[j]:split_idx[j + 1]] for j in range(len(split_idx) - 1)
            if split_idx[j + 1] - split_idx[j] >= 2]

# ── Build cast-level table ────────────────────────────────────────────────────
cast_rows = []
for (serial, ts), grp in df.groupby(GROUP_KEYS):
    for cast_num, seg in enumerate(split_casts(grp), start=1):
        cast_rows.append({
            'instrument_serial_number': serial,
            'timestamp':  ts,
            'cast_num':   cast_num,
            'instrument': grp['instrument'].iloc[0],
            'time_utc':   seg['TIME'].iloc[0],
            'lat':        seg['LATITUDE'].iloc[0],
            'lon':        seg['LONGITUDE'].iloc[0],
            'n_points':   len(seg),
            'depth_min':  seg['DEPTH'].min(),
            'depth_max':  seg['DEPTH'].max(),
            'temp_surface': seg.sort_values('DEPTH').iloc[0]['TEMPERATURE'],
            'temp_bottom':  seg.sort_values('DEPTH').iloc[-1]['TEMPERATURE'],
            'temp_min':   seg['TEMPERATURE'].min(),
            'temp_max':   seg['TEMPERATURE'].max(),
        })

cast_stats = pd.DataFrame(cast_rows)
cast_stats['temp_range'] = (cast_stats['temp_max'] - cast_stats['temp_min']).round(4)

# Cast counts per instrument (used in map legend, summary, profile titles)
casts_per_instrument = cast_stats.groupby('instrument_serial_number')['cast_num'].count().rename('n_casts')

# ── Colors per instrument serial ──────────────────────────────────────────────
all_serials = sorted(df['instrument_serial_number'].unique())
cmap_inst   = matplotlib.colormaps.get_cmap('tab10')
PALETTE     = {s: mcolors.to_hex(cmap_inst(i % 10))
               for i, s in enumerate(all_serials)}

print(f"\nAll instrument serials ({len(all_serials)}): {all_serials}")
print("\nVerifying deployment grouping:")
print(df.groupby('timestamp')['TIME'].agg(['min', 'max', 'count']).head(10))


# ─────────────────────────────────────────────────────────────────────────────
# Fetch Argo profiles from Ifremer ERDDAP (same spatial domain, 2026)
# ─────────────────────────────────────────────────────────────────────────────

# Column rename map from erddapy output → clean names
rename_argo = {
    'platform_number (1)':      'argo',
    'longitude (degrees_east)': 'longitude',
    'latitude (degrees_north)': 'latitude',
}

def get_argo_floats_by_time(bbox=(-110, -45, 0, 46),
                            time_start=None, time_end=dt.date.today(),
                            wmo_id=None, variables=None):
    if isinstance(variables, tuple):
        variables = list(variables)

    time_start = time_start or (time_end - dt.timedelta(days=1))

    default_variables = ['platform_number', 'time', 'longitude', 'latitude']

    constraints = {
        'time>=': str(time_start),
        'time<=': str(time_end),
    }

    if bbox:
        constraints['longitude>='] = bbox[0]
        constraints['longitude<='] = bbox[1]
        constraints['latitude>='] = bbox[2]
        constraints['latitude<='] = bbox[3]

    if wmo_id:
        if isinstance(wmo_id, (int, float)):
            wmo_id = str(wmo_id)
        constraints['platform_number='] = wmo_id

    if variables:
        default_variables = list(set(default_variables + variables))

    e = ERDDAP(server='IFREMER', protocol='tabledap', response='csv')
    e.dataset_id   = 'ArgoFloats'
    e.constraints  = constraints
    e.variables    = default_variables

    try:
        df = e.to_pandas(
            index_col="time (UTC)",
            parse_dates=True,
        ).dropna().tz_localize(None)
        df = df.reset_index()
        print(f"  Argo columns: {list(df.columns)}")  # debug — remove once working
        # Rename whatever erddapy returns to clean names
        df.columns = [c.split(' (')[0].strip() for c in df.columns]
        df = df.rename(columns={'platform_number': 'argo'})
        df = df.set_index(["argo", "time"]).sort_index()
    except rHTTPError as e:
        print(f"  WARNING: HTTP error — {e}")
        df = pd.DataFrame()
    return df

print("\nFetching Argo data from Ifremer ERDDAP...")

argo1 = get_argo_floats_by_time(
    bbox=(160, 180, -26, -5),
    time_start=dt.date(2026, 1, 1),
    time_end=dt.date(2026, 12, 31),
)

# Get argo east of 180° by adjusting longitude values
argo2 = get_argo_floats_by_time(
    bbox=(0, 10, -26, -5),
    time_start=dt.date(2026, 1, 1),
    time_end=dt.date(2026, 12, 31),
)

# Combine argo1 and argo2, adjusting longitudes in argo2 to be >180°
if not argo1.empty and not argo2.empty:
    argo2['longitude'] = argo2['longitude'] + 180
    argo = pd.concat([argo1, argo2], ignore_index=False).sort_index()
elif not argo1.empty:
    argo = argo1
elif not argo2.empty:
    argo = argo2
else:
    argo = pd.DataFrame()

if not argo.empty:
    argo_plot = argo.reset_index()
    print(f"  → {len(argo_plot):,} Argo profiles, "
          f"{argo_plot['argo'].nunique()} floats")
else:
    argo_plot = pd.DataFrame()
    print("  WARNING: No Argo data returned")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAP  (centered on 180° to avoid antimeridian split)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating map...")

proj = ccrs.PlateCarree(central_longitude=180)
fig  = plt.figure(figsize=(14, 10), dpi=DPI)
ax   = fig.add_subplot(1, 1, 1, projection=proj)

ax.set_facecolor('white')

# Tricolor bathymetry — lon in 0-360 space (map spans 159.75°E → 189.75°E)
add_tricolor_bathymetry(ax, lon_min=159.5, lon_max=190.5, lat_min=-27.0, lat_max=-4.5)

# Re-enforce extent after contourf, which can auto-expand the view
ax.set_extent([-20.25, 9.75, -26.25, -4.75], crs=proj)

SCALE = '10m'
ax.add_feature(cfeature.LAND     .with_scale(SCALE), facecolor='#ddd8bb', zorder=2)
ax.add_feature(cfeature.COASTLINE.with_scale(SCALE), edgecolor='#555555', linewidth=0.7, zorder=3)
ax.add_feature(cfeature.BORDERS  .with_scale(SCALE), edgecolor='#777777', linewidth=0.5,
               linestyle='--', zorder=3)
ax.add_feature(cfeature.LAKES    .with_scale(SCALE), facecolor=cfeature.COLORS['water'],
               edgecolor='#555555', linewidth=0.4, zorder=3)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                  alpha=0.45, linestyle='--', crs=ccrs.PlateCarree())
gl.top_labels   = False
gl.right_labels = False
gl.xlocator     = mticker.FixedLocator(range(160, 186, 4))
gl.ylocator     = mticker.FixedLocator(range(-26, -4, 2))
gl.xformatter   = LONGITUDE_FORMATTER
gl.yformatter   = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 9, 'color': '#444'}
gl.ylabel_style = {'size': 9, 'color': '#444'}

legend_handles = []
for serial, grp in dep_stats.groupby('instrument_serial_number'):
    color = PALETTE[serial]
    n     = len(grp)
    ms    = 30 if n > 50 else 80
    ax.scatter(grp['lon'], grp['lat'],
               c=color, marker='D', s=ms,
               linewidths=0.6, edgecolors='white',
               alpha=0.85, zorder=7,
               transform=ccrs.PlateCarree())
    n_casts = casts_per_instrument.get(serial, 0)
    legend_handles.append(
        Line2D([0], [0], marker='D', color='none',
               markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=0.5, markersize=7, linewidth=0,
               label=f'Inst. {serial}  (n={n_casts:,} casts)'))

# Argo float profiles
if not argo_plot.empty:
    ax.scatter(argo_plot['longitude'], argo_plot['latitude'],
               c='green', marker='o', s=30,
               linewidths=0.9, edgecolors='#222222',
               alpha=0.7, zorder=6000,
               transform=ccrs.PlateCarree())
    legend_handles.append(
        Line2D([0], [0], marker='o', color='#222222',
               markerfacecolor='green', markersize=7,
               linewidth=0, markeredgewidth=1.0,
               label=f'Argo floats  (n={len(argo_plot):,} profiles, '
                     f'{argo_plot["argo"].nunique()} floats)'))

geo_labels = [
    (165.5, -21.3, 'New\nCaledonia'),
    (178, -17.8, 'Fiji'),
]
for lon, lat, name in geo_labels:
    ax.text(lon, lat, name, transform=ccrs.PlateCarree(),
            fontsize=8.5, color='#333', ha='center', style='italic', zorder=6,
            bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=1.5))

leg = ax.legend(handles=legend_handles,
                title='FishSOOP instruments  ·  # casts',
                title_fontsize=9, fontsize=8.5,
                loc='lower right', framealpha=0.93,
                edgecolor='#cccccc', borderpad=0.9,
                handlelength=1.3, handleheight=1.1)
leg.get_title().set_fontweight('500')
leg.set_zorder(7000)

bathy_handles = [
    Patch(facecolor='lightsteelblue',                edgecolor='none', label='0 – 100 m'),
    Patch(facecolor=tuple(cfeature.COLORS['water']), edgecolor='none', label='100 – 1000 m'),
    Patch(facecolor='cornflowerblue',                edgecolor='none', label='1000 – 6000 m'),
    Patch(facecolor='royalblue',                     edgecolor='none', label='6000+ m'),
]
bleg = ax.legend(handles=bathy_handles, loc='lower left', fontsize=8.5,
                 title='Bathymetry', title_fontsize=9,
                 framealpha=0.93, edgecolor='#cccccc')
bleg.get_title().set_fontweight('500')
bleg.set_zorder(7000)
ax.add_artist(leg)  # re-add instrument legend displaced by the second legend call

ax.set_title(
    'FishSOOP — Fishing Vessel Temperature Profiles, Fiji & New Caledonia (2026)',
    fontsize=12, fontweight='500', pad=10, loc='left')

d0       = df['TIME'].min().strftime('%d %b')
d1       = df['TIME'].max().strftime('%d %b %Y')
n_casts_total = len(cast_stats)
foot  = (f"{n_casts_total:,} casts  ·  {df['instrument_serial_number'].nunique()} instruments  ·  "
         f"{len(df):,} obs. rows  ·  {d0} – {d1}")
ax.text(0.5, -0.020, foot, transform=ax.transAxes,
        fontsize=7.5, ha='center', va='top', color='#555')
ax.text(0.5, -0.045, 'Sources: FishSOOP realtime QC  ·  Argo (Ifremer ERDDAP)',
        transform=ax.transAxes, fontsize=6.5, ha='center', va='top', color='#888')

plt.tight_layout()
map_out = 'fishsoop_fiji_nc_2026_map.png'
plt.savefig(map_out, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved → {map_out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROFILE PLOTS — one figure per instrument
#    Each deployment binned into 5 m depth layers: mean line + std envelope
#    Colored by deployment date (plasma)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating profile plots...")

for serial in all_serials:
    sub       = df[df['instrument_serial_number'] == serial].copy()
    sub_stats = dep_stats[dep_stats['instrument_serial_number'] == serial].copy()
    n_dep     = sub['timestamp'].nunique()

    if n_dep == 0:
        print(f"  Skipping {serial} — no deployments found")
        continue

    deps_sorted = sorted(sub['timestamp'].unique())
    t_index     = {t: i for i, t in enumerate(deps_sorted)}
    cmap_t      = matplotlib.colormaps.get_cmap('plasma')
    norm        = mcolors.Normalize(vmin=0, vmax=max(len(deps_sorted) - 1, 1))

    fig, ax = plt.subplots(figsize=(7, 9), layout='constrained')

    plotted = 0
    for ts, grp in sub.groupby('timestamp'):
        grp_s = grp.sort_values('TIME').reset_index(drop=True)
        if len(grp_s) < 2:
            continue

        color = cmap_t(norm(t_index[ts]))

        for seg in split_casts(grp_s):
            ax.plot(seg['TEMPERATURE'], seg['DEPTH'],
                    color=color, linewidth=0.7, alpha=0.6)
            plotted += 1

    n_casts_serial = casts_per_instrument.get(serial, 0)
    print(f"  {serial}: {n_dep} deployments, {n_casts_serial} casts, {plotted} plotted")

    ax.invert_yaxis()
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=10)

    # Colorbar — labeled by deployment date
    sm = cm.ScalarMappable(cmap=cmap_t, norm=norm)
    sm.set_array([])
    cbar     = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    n_ticks  = min(8, len(deps_sorted))
    tick_idx = np.linspace(0, len(deps_sorted) - 1, n_ticks, dtype=int)
    cbar.set_ticks(tick_idx)
    tick_labels = []
    for i in tick_idx:
        ts_val = deps_sorted[i]
        row    = sub_stats[sub_stats['timestamp'] == ts_val]
        label  = pd.Timestamp(row.iloc[0]['time_utc']).strftime('%b %d') if len(row) else str(ts_val)
        tick_labels.append(label)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Deployment date (UTC)', fontsize=9)

    stat_txt = (
        f"n casts: {n_casts_serial}  |  "
        f"max depth: {sub_stats['max_depth'].max():.1f} m  |  "
        f"pts/deployment: {sub_stats['n_points'].min()}–{sub_stats['n_points'].max()}"
    )
    ax.set_title(stat_txt, fontsize=9, color='#555', pad=4)
    fig.suptitle(
        f"Instrument {serial}  ·  {n_casts_serial} casts  ·  {n_dep} deployments",
        fontsize=12, fontweight='500')

    out = f'profiles_{serial}.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating summary table...")

summary = df.groupby('instrument_serial_number').apply(lambda g: pd.Series({
    'instrument':  g['instrument'].iloc[0],
    'n_obs_rows':  len(g),
    'max_depth_m': g['DEPTH'].max(),
    'min_depth_m': g['DEPTH'].min(),
    'date_start':  g['TIME'].min().strftime('%Y-%m-%d'),
    'date_end':    g['TIME'].max().strftime('%Y-%m-%d'),
    'temp_min_C':  g['TEMPERATURE'].min(),
    'temp_max_C':  g['TEMPERATURE'].max(),
})).reset_index()

# Cast counts and pts-per-cast from cast_stats
cast_summary = cast_stats.groupby('instrument_serial_number').agg(
    n_casts  = ('cast_num',  'count'),
    pts_min  = ('n_points',  'min'),
    pts_max  = ('n_points',  'max'),
    pts_mean = ('n_points',  'mean'),
).reset_index()
cast_summary['pts_mean'] = cast_summary['pts_mean'].round(1)
summary = summary.merge(cast_summary, on='instrument_serial_number')

col_labels = [
    'Serial #', 'Instrument', 'Casts', 'Obs rows',
    'Max depth (m)', 'Pts/cast\n(min – max)',
    'Date start', 'Date end', 'Temp range (°C)'
]
table_data = []
for _, r in summary.iterrows():
    table_data.append([
        r['instrument_serial_number'],
        r['instrument'],
        int(r['n_casts']),
        int(r['n_obs_rows']),
        f"{r['max_depth_m']:.1f}",
        f"{int(r['pts_min'])} – {int(r['pts_max'])}",
        r['date_start'],
        r['date_end'],
        f"{r['temp_min_C']:.2f} – {r['temp_max_C']:.2f}",
    ])

fig, ax = plt.subplots(figsize=(14, 0.65 * (len(summary) + 2.5)))
ax.set_axis_off()

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

# Custom column widths: narrow Serial #, wide Instrument
col_widths = [0.07, 0.22, 0.06, 0.08, 0.10, 0.11, 0.10, 0.10, 0.13]
for (row, col), cell in tbl.get_celld().items():
    cell.set_width(col_widths[col])

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i, row in enumerate(summary.itertuples(), start=1):
    color = PALETTE.get(row.instrument_serial_number, '#888888')
    tbl[i, 0].set_facecolor(color)
    tbl[i, 0].set_text_props(color='white', fontweight='500')
    shade = '#f4f6f8' if i % 2 == 0 else 'white'
    for j in range(1, len(col_labels)):
        tbl[i, j].set_facecolor(shade)

ax.set_title('FishSOOP 2026 — Cast Summary, Fiji & New Caledonia',
             fontsize=12, fontweight='500', pad=16)
fig.text(0.5, 0.01, 'Source: FishSOOP realtime QC parquet',
         ha='center', fontsize=7.5, color='#888')

plt.tight_layout()
plt.savefig('profile_summary.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved → profile_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-CAST CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating per-cast CSV...")

out_cast = cast_stats.copy()
for col in ['lat', 'lon', 'depth_min', 'depth_max',
            'temp_min', 'temp_max', 'temp_surface', 'temp_bottom', 'temp_range']:
    out_cast[col] = out_cast[col].round(4)

out_cast['time_utc'] = out_cast['time_utc'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
out_cast = out_cast.sort_values(['instrument_serial_number', 'timestamp', 'cast_num']).reset_index(drop=True)
out_cast = out_cast[['instrument_serial_number', 'timestamp', 'cast_num', 'instrument',
                      'time_utc', 'lat', 'lon',
                      'n_points', 'depth_min', 'depth_max',
                      'temp_surface', 'temp_bottom',
                      'temp_min', 'temp_max', 'temp_range']]

out = 'profile_table_per_cast.csv'
out_cast.to_csv(out, index=False)
print(f"  Saved → {out}  ({len(out_cast)} rows)")

print("\nDone. Output files:")
print("  fishsoop_fiji_nc_2026_map.png")
for serial in all_serials:
    print(f"  profiles_{serial}.png")
print("  profile_summary.png")
print("  profile_table_per_cast.csv")