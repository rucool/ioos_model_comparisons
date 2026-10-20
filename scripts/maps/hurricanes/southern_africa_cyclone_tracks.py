"""
Map every tropical cyclone track off southern Africa that reached at least
`min_category` *inside* the map box between `y0` and `y1`. Two presets in
``REGIONS``:

    python southern_africa_cyclone_tracks.py --region indian    # default
    python southern_africa_cyclone_tracks.py --region atlantic
    python southern_africa_cyclone_tracks.py --extent 20 80 -40 -5

``indian`` is the Southwest Indian Ocean (southeast African coast, Mozambique
Channel, Madagascar). ``atlantic`` is the southeast Atlantic off Angola,
Namibia and South Africa's west coast. The South Atlantic basin in IBTrACS
holds only three storms (2004, 2010, 2011), all off Brazil, so that map is
expected to be empty; it is drawn anyway to document the absence.

Adapted from ``hurricane_history_by_years.py``. tropycal's HURDAT source only
covers the Atlantic/East Pacific, so this reads IBTrACS instead. tropycal 1.3
defaults to IBTrACS v04r00, which stops in mid-2024; v04r01 (``IBTRACS_URL``)
is updated weekly and carries the provisional current seasons.

Winds are the JTWC 1-minute values (``ibtracs_mode="jtwc"``) so the
Saffir-Simpson categories match the Atlantic maps. Météo-France La Réunion
(the WMO center here) uses 10-minute winds, which run ~10% lower.

Track lines are colored by the calendar year the storm formed in, and markers
(every ``MARKER_EVERY_HOURS``) show intensity with Southern Hemisphere
(clockwise) symbols.
"""
import argparse

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tcmarkers
from cool_maps.plot import create
from tropycal import tracks

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.plotting import export_fig

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.(basin).list.v04r01.csv"
)

# Minimum sustained wind speed (knots) for each Saffir-Simpson category;
# -1 = any system, 0 = tropical storm.
CATEGORY_MIN_KT = {-1: 0, 0: 34, 1: 64, 2: 83, 3: 96, 4: 113, 5: 137}

REGIONS = {
    "indian": dict(
        title="Southwest Indian Ocean",
        basin="south_indian",
        extent=[20, 70, -40, -5],  # [lon_min, lon_max, lat_min, lat_max]
        min_category=1,
        folder="southwest_indian_cyclones",
        fname="southwest_indian",
    ),
    "atlantic": dict(
        title="Southeast Atlantic (SW African Coast)",
        basin="south_atlantic",
        extent=[-15, 20, -40, -5],
        min_category=-1,
        folder="southeast_atlantic_cyclones",
        fname="southeast_atlantic",
    ),
}

# NOAA colors, matching plotting.plot_hurricane_track.
CATEGORIES = [
    # label, min kt, color, marker size
    ("TD", 0, "#808080", 18),
    ("TS", 34, "#FFCC00", 30),
    ("C1", 64, "#FF9900", 42),
    ("C2", 83, "#FF6600", 50),
    ("C3", 96, "#FF0000", 60),
    ("C4", 113, "#990000", 72),
    ("C5", 137, "#660066", 85),
]

# IBTrACS is 3-hourly; markers every point bury the map at this scale.
MARKER_EVERY_HOURS = 12


def _in_box(lons, lats, extent):
    return (
        (lons >= extent[0]) & (lons <= extent[1])
        & (lats >= extent[2]) & (lats <= extent[3])
    )


def _collect_storms(basin, extent, start, end, wind_threshold):
    """Return storms active in [start, end) that reached >= wind_threshold kt
    while inside `extent`."""
    storms = []
    for storm_id, d in basin.data.items():
        times = pd.to_datetime(d["time"])
        in_time = (times >= start) & (times < end)
        if not in_time.any():
            continue

        lons = np.asarray(d["lon"], dtype=float)
        lats = np.asarray(d["lat"], dtype=float)
        vmax = np.asarray(d["vmax"], dtype=float)
        in_region = _in_box(lons, lats, extent) & in_time
        vmax_in_region = np.where(in_region, vmax, np.nan)
        if np.isnan(vmax_in_region).all() or np.nanmax(vmax_in_region) < wind_threshold:
            continue

        peak_idx = np.nanargmax(vmax_in_region)
        storms.append(
            {
                "id": storm_id,
                "name": d["name"],
                "year": times[0].year,
                "time": times,
                "lon": lons,
                "lat": lats,
                "vmax": vmax,
                "peak_kt": float(vmax_in_region[peak_idx]),
                "peak_lon": float(lons[peak_idx]),
                "peak_lat": float(lats[peak_idx]),
            }
        )
    return sorted(storms, key=lambda s: s["time"][0])


def _plot_storm(ax, storm, linecolor, extent, label=True):
    """Draw one track line plus intensity markers grouped by category."""
    lon, lat, vmax = storm["lon"], storm["lat"], storm["vmax"]
    ax.plot(lon, lat, color=linecolor, linewidth=2,
            transform=ccrs.PlateCarree(), zorder=29999)

    hours = storm["time"].hour
    on_cadence = (hours % MARKER_EVERY_HOURS == 0) & (storm["time"].minute == 0)
    valid = ~np.isnan(vmax) & np.asarray(on_cadence)
    bounds = [c[1] for c in CATEGORIES[1:]] + [np.inf]
    for (_, lo, color, size), hi in zip(CATEGORIES, bounds):
        sel = valid & (vmax >= lo) & (vmax < hi)
        if not sel.any():
            continue
        ax.scatter(
            lon[sel], lat[sel],
            color=color, s=size, edgecolor="black", linewidths=0.4,
            marker=tcmarkers.tc_marker(max(lo, 1), lat=-1),
            transform=ccrs.PlateCarree(), zorder=30000,
        )

    if label:
        name = storm["name"].title() if storm["name"] != "UNNAMED" else storm["id"]
        # Flip the label to the left of the peak near the east edge so it
        # stays inside the map.
        near_east = storm["peak_lon"] > extent[1] - 4
        ax.text(
            storm["peak_lon"] + (-0.5 if near_east else 0.5),
            storm["peak_lat"] + 0.4, name,
            ha="right" if near_east else "left",
            fontsize=8, fontweight="bold", color=linecolor,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            transform=ccrs.PlateCarree(), zorder=30001, clip_on=True,
        )


def _configure_axes(ax, extent):
    create(extent, ax=ax, ticks=False)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon_ticks = np.arange(np.ceil(extent[0] / 10) * 10, extent[1] + 0.1, 10)
    lat_ticks = np.arange(np.ceil(extent[2] / 5) * 5, extent[3] + 0.1, 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linestyle="--",
                      color="black", alpha=0.5, xlocs=lon_ticks, ylocs=lat_ticks)
    gl.set_zorder(9.99)

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=12, direction="out",
                   length=7, width=2, top=True, right=True)
    for tick in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels():
        tick.set_fontweight("bold")


def _add_category_legend(ax):
    handles = [
        mlines.Line2D([0], [0], marker=tcmarkers.tc_marker(max(lo, 1), lat=-1),
                      color="w", label=cat, markerfacecolor=color,
                      markeredgecolor="black", markersize=10, linestyle="none")
        for cat, lo, color, _ in CATEGORIES
    ]
    legend = ax.legend(handles=handles, title="Storm\nCategory", loc="upper right",
                       fontsize=8, handleheight=2, framealpha=0.95,
                       borderpad=0.3, labelspacing=1.0)
    legend.get_frame().set_facecolor("0.85")
    legend.get_frame().set_edgecolor("0.3")
    legend.set_zorder(100001)
    return legend


def _add_year_legend(ax, years, year_colors):
    handles = [
        mlines.Line2D([], [], color=year_colors[y], linewidth=3, label=str(y))
        for y in years
    ]
    legend = ax.legend(handles=handles, title="Year", loc="lower right",
                       fontsize=8, framealpha=0.95, borderpad=0.3, ncol=2)
    legend.get_frame().set_facecolor("0.85")
    legend.get_frame().set_edgecolor("0.3")
    legend.set_zorder(100000)
    return legend


def _fmt_lon(lon):
    return f"{abs(lon):.1f}{'E' if lon >= 0 else 'W'}"


def main(region="indian", y0=2017, y1=2026, min_category=None, extent=None):
    cfg = REGIONS[region]
    # A custom extent gets its bounds in the filename so it doesn't overwrite
    # the preset map.
    extent_tag = ""
    if extent is None:
        extent = cfg["extent"]
    else:
        extent_tag = f"_{_fmt_lon(extent[0])}-{_fmt_lon(extent[1])}".replace(".0", "")
    if min_category is None:
        min_category = cfg["min_category"]
    label_storms = True  # Name each storm at its peak intensity inside the box

    wind_threshold = CATEGORY_MIN_KT[min_category]
    basin = tracks.TrackDataset(
        basin=cfg["basin"], source="ibtracs", ibtracs_mode="jtwc",
        ibtracs_url=IBTRACS_URL,
    )

    storms = _collect_storms(
        basin, extent,
        pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1 + 1}-01-01"),
        wind_threshold,
    )
    print(f"{len(storms)} storms reached >= {wind_threshold} kt inside {extent} "
          f"between {y0} and {y1}:")
    for s in storms:
        print(f"  - {s['name']:<10} ({s['id']}) {s['time'][0]:%Y-%m-%d}  "
              f"peak {s['peak_kt']:.0f} kt at {abs(s['peak_lat']):.1f}S, "
              f"{_fmt_lon(s['peak_lon'])}")

    years = list(range(y0, y1 + 1))
    cmap = plt.get_cmap("tab10" if len(years) <= 10 else "tab20")
    year_colors = {y: cmap(i % cmap.N) for i, y in enumerate(years)}

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=conf.projection["map"])
    _configure_axes(ax, extent)

    for storm in storms:
        _plot_storm(ax, storm, year_colors[storm["year"]], extent, label=label_storms)

    cat_legend = _add_category_legend(ax)
    if storms:
        _add_year_legend(ax, years, year_colors)
        ax.add_artist(cat_legend)
    else:
        record = sorted(
            f"{pd.to_datetime(d['time'][0]):%Y}" for d in basin.data.values()
        )
        ax.text(
            0.5, 0.5,
            f"No tropical cyclones recorded in this box, {y0}-{y1}\n"
            f"(IBTrACS {cfg['basin'].replace('_', ' ')} basin: {len(record)} storms "
            f"on record{', ' + ', '.join(record) if record else ''})",
            transform=ax.transAxes, ha="center", va="center", fontsize=16,
            fontweight="bold", zorder=100002,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.9),
        )

    if min_category == -1:
        cat_label = "Tropical Cyclones (any intensity)"
    elif min_category == 0:
        cat_label = "Tropical Storms"
    else:
        cat_label = f"Category {min_category}+ Cyclones"
    ax.set_title(
        f"{cfg['title']} {cat_label} — {y0}-{y1}\n"
        f"IBTrACS v04r01, JTWC 1-min winds ({len(storms)} storms)",
        fontsize=18, fontweight="bold", pad=15,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")
        spine.set_zorder(100)

    fig.tight_layout()

    cat_tag = "all" if min_category == -1 else f"category{min_category}plus"
    save_root = conf.path_plots / "maps" / cfg["folder"]
    fname = f"{cfg['fname']}_{cat_tag}_cyclones_{y0}_to_{y1}{extent_tag}.png"
    export_fig(save_root, fname, dpi=conf.dpi)
    plt.close(fig)
    print(f"Saved {fname} to {save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=REGIONS, default="indian")
    parser.add_argument("--start", type=int, default=2017, help="first year")
    parser.add_argument("--end", type=int, default=2026, help="last year")
    parser.add_argument("--min-category", type=int, choices=CATEGORY_MIN_KT,
                        help="-1 any, 0 TS, 1-5 Saffir-Simpson (default: per region)")
    parser.add_argument("--extent", type=float, nargs=4,
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        help="override the region's map box")
    args = parser.parse_args()
    main(args.region, args.start, args.end, args.min_category, args.extent)
