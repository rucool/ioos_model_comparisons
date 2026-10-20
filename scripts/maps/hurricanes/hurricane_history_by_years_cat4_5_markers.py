"""
Generate a single Gulf of Mexico map that contains the track of every storm
that passed through the Gulf and reached at least `min_category` (set in
``main``) between `y0` and `y1`.  Styling mirrors ``plot_ohc_single`` while
``plot_active_hurricanes`` draws each storm track. Unlike
``hurricane_history_by_years.py``, only Category 4 and 5 intensity markers
are drawn along each track (the full track line is still shown).
"""
import colorsys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import tcmarkers
from cool_maps.plot import add_bathymetry, create, get_bathymetry
from tropycal import tracks

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.gulf_of_mexico import bounding_box_mask, gulf_mask
from ioos_model_comparisons.plotting import export_fig, plot_active_hurricanes


def _compute_extent(lons, lats, buffer=5.0):
    """Return padded extent for a given storm track."""
    lon_min = np.nanmin(lons) - buffer
    lon_max = np.nanmax(lons) + buffer
    lat_min = np.nanmin(lats) - buffer
    lat_max = np.nanmax(lats) + buffer

    return [
        max(-180.0, lon_min),
        min(180.0, lon_max),
        max(-90.0, lat_min),
        min(90.0, lat_max),
    ]


def _build_ticks(min_val, max_val, step):
    """Create nicely spaced tick values with at least three entries."""
    if max_val <= min_val:
        return np.array([min_val, max_val])

    start_tick = step * np.ceil(min_val / step)
    end_tick = step * np.floor(max_val / step)

    if start_tick > end_tick:
        start_tick = min_val
        end_tick = max_val

    ticks = np.arange(start_tick, end_tick + step * 0.5, step)
    ticks = ticks[(ticks >= min_val) & (ticks <= max_val)]

    if ticks.size < 2:
        ticks = np.linspace(min_val, max_val, 2)

    return ticks


def _configure_axes(ax, extent, bathy=None):
    """Apply the styling used by ``plot_ohc_single`` to an axes."""
    create(extent, ax=ax, ticks=False)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if bathy is not None:
        add_bathymetry(
            ax,
            bathy.longitude.values,
            bathy.latitude.values,
            bathy.z.values,
            levels=(-1000, -100),
            zorder=1.5,
        )
        levels_bt = [-8000, -1000, -100, 0]
        colors_bt = ["cornflowerblue", cfeature.COLORS["water"], "lightsteelblue"]
        ax.contourf(
            bathy["longitude"],
            bathy["latitude"],
            bathy["z"],
            levels_bt,
            colors=colors_bt,
            transform=ccrs.PlateCarree(),
            ticks=False,
            zorder=1.4,
        )

    lon_ticks = _build_ticks(extent[0], extent[1], 5)
    lat_ticks = _build_ticks(extent[2], extent[3], 5)

    lon_minor = np.arange(np.ceil(extent[0]), np.floor(extent[1]) + 1, 1)
    lat_minor = np.arange(np.ceil(extent[2]), np.floor(extent[3]) + 1, 1)

    if lon_minor.size == 0:
        lon_minor = np.array(extent[:2])
    if lat_minor.size == 0:
        lat_minor = np.array(extent[2:])

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linestyle="--",
        color="black",
        alpha=0.5,
        xlocs=lon_ticks,
        ylocs=lat_ticks,
    )
    gl.set_zorder(9.99)

    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_xticks(lon_minor, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_minor, minor=True, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        direction="out",
        length=7,
        width=2,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,
        width=1,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
        labelleft=False,
        labelbottom=False,
    )

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_fontweight("bold")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontweight("bold")


def _add_category_legend(ax):
    """Add the NOAA-style hurricane intensity legend.

    Only Category 4 and 5 are included, since those are the only markers
    plotted along each track in this script.
    """
    type_colors = {
        "C4": "#990000",
        "C5": "#660066",
    }

    legend_handles = []
    for category, color in type_colors.items():
        if category == "C4":
            wind_speed = 125
        else:
            wind_speed = 145

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=tcmarkers.tc_marker(wind_speed),
                color="w",
                label=category,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=10,
                linestyle="none",
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        title="Storm\nCategory",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.001),
        fontsize=8,
        handleheight=2,
        handletextpad=1,
        framealpha=0.95,
        borderpad=0.3,
        labelspacing=1.45,
    )
    legend.get_frame().set_facecolor("0.85")
    legend.get_frame().set_edgecolor("0.3")
    legend.get_title().set_multialignment("center")
    legend.set_zorder(100001)
    return legend


def _add_storm_legend(ax, hurricanes, colors):
    """Add a legend that maps line colors to individual storms."""
    handles = []
    for storm, color in zip(hurricanes, colors):
        label = (
            storm["name"].title()
            if storm["name"] and storm["name"].upper() != "UNNAMED"
            else storm["id"]
        )
        handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))

    if not handles:
        return None

    legend = ax.legend(
        handles=handles,
        title="Storm Tracks",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        fontsize=8,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.8,
        columnspacing=1.0,
    )
    legend.get_frame().set_facecolor("0.85")
    legend.get_frame().set_edgecolor("0.3")
    legend.set_zorder(100000)
    return legend


# Minimum sustained wind speed (knots) for each Saffir-Simpson category.
CATEGORY_MIN_KT = {1: 64, 2: 83, 3: 96, 4: 113, 5: 137}


def _distinct_colors(n):
    """Return n evenly-spaced, fully-saturated hex colors (no pastels)."""
    return [
        mcolors.to_hex(colorsys.hsv_to_rgb(i / n, 0.85, 0.85))
        for i in range(n)
    ]


def _collect_hurricanes(basin, year, wind_threshold, use_gulf_polygon=True, force_include_names=frozenset()):
    """Return metadata for every storm in `year` that reached >=wind_threshold kt
    *while inside* the Gulf of Mexico.

    Membership in the Gulf uses the ``gulf_of_mexico`` polygon (Yucatan
    Channel/Straits of Florida excluded), not the map's rectangular plotting
    extent, so a storm that only touches the region at low intensity — e.g.
    it peaked at Cat 4/5 in the Caribbean and merely brushed the Gulf on its
    way in or out — is excluded. Storms reaching the threshold outside the
    Gulf are printed so they can be double-checked by hand.

    `force_include_names` falls back to the coarser rectangular bounding box
    for the named storms, for cases like Irma where the polygon's Straits of
    Florida cut excludes a storm we want kept in anyway.
    """
    season = basin.get_season(year)
    storm_ids = season.summary()["id"]
    hurricanes = []

    for storm_id in storm_ids:
        storm = basin.get_storm(storm_id)
        vmax = np.asarray(storm.dict["vmax"], dtype=float)
        if np.isnan(vmax).all():
            continue

        name_upper = str(getattr(storm, "name", storm_id)).upper()
        lons = np.asarray(storm.dict["lon"], dtype=float)
        lats = np.asarray(storm.dict["lat"], dtype=float)

        if use_gulf_polygon:
            in_region = (
                bounding_box_mask(lons, lats)
                if name_upper in force_include_names
                else gulf_mask(lons, lats)
            )
            if not np.any(in_region):
                continue
            vmax_in_region = np.where(in_region, vmax, np.nan)
        else:
            vmax_in_region = vmax

        if np.isnan(vmax_in_region).all() or np.nanmax(vmax_in_region) < wind_threshold:
            if use_gulf_polygon and np.nanmax(vmax) >= wind_threshold:
                print(
                    f"  (excluded) {name_upper} ({storm_id}): reached the category "
                    f"threshold outside the region only — verify by hand"
                )
            continue

        times = pd.to_datetime(storm.dict["time"])
        center_time = pd.Timestamp(times[-1])
        duration = times[-1] - times[0]
        lookback_days = max(5, int(np.ceil(duration.total_seconds() / 86400.0)) + 2)

        extent = _compute_extent(lons, lats)

        peak_idx = np.nanargmax(vmax_in_region)

        hurricanes.append(
            {
                "id": storm_id,
                "name": getattr(storm, "name", storm_id),
                "center_time": center_time,
                "lookback_days": lookback_days,
                "extent": extent,
                "peak_lon": float(lons[peak_idx]),
                "peak_lat": float(lats[peak_idx]),
            }
        )

    return hurricanes


# Storm names to omit by hand, across all four slides below. Now largely
# redundant with the Gulf polygon (see gulf_of_mexico.py), which already
# excludes storms whose peak was in the Caribbean/Straits of Florida --
# kept here as a documented belt-and-suspenders case. DELTA's only Cat 4/5
# point (120kt at 18.5N, -83.3E on 2020-10-06) was in the NW Caribbean; it
# had weakened to Cat 3 (105kt) by the time it reached the Gulf.
EXCLUDE_NAMES = {"MATTHEW", "DELTA"}

# Storms kept in despite peaking outside the Gulf polygon (see
# gulf_of_mexico.bounding_box_mask). Irma's only Cat 4 points sit in the
# Straits of Florida near Key West -- by hand call to keep it in.
FORCE_INCLUDE_NAMES = {"IRMA"}

# The four year ranges used for the slide deck.
YEAR_RANGES = [(1980, 1997), (1998, 2008), (2009, 2016), (2017, 2025)]
YEAR_RANGES = [(1980, 1995), (1996, 2005), (2006, 2015),(2016, 2025)]


def _generate_slide(basin, y0, y1, min_category=4):
    extent = [-98.25, -79.75, 17.75, 31.25]  # Gulf of Mexico -- map plotting extent
    exclude_names = EXCLUDE_NAMES
    force_include_names = FORCE_INCLUDE_NAMES
    wind_threshold = CATEGORY_MIN_KT[min_category]

    hurricanes = []
    for year in range(y0, y1 + 1):
        try:
            hurricanes.extend(
                _collect_hurricanes(
                    basin, year, wind_threshold, force_include_names=force_include_names
                )
            )
        except Exception as exc:
            print(f"Warning: Could not load season {year}: {exc}")

    hurricanes = [
        storm for storm in hurricanes
        if str(storm["name"]).upper() not in exclude_names
    ]

    if not hurricanes:
        print(f"No Gulf of Mexico category {min_category}+ hurricanes detected between {y0} and {y1}.")

    try:
        bathy = get_bathymetry(extent)
    except Exception as exc:
        bathy = None
        print(f"Warning: Could not load bathymetry: {exc}")

    print(f"Creating Gulf of Mexico category {min_category}+ hurricane map for {y0}-{y1}.")
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1, projection=conf.projection["map"])
    _configure_axes(ax, extent, bathy=bathy)

    storm_colors = _distinct_colors(len(hurricanes))

    for idx, storm in enumerate(hurricanes):
        display_name = (
            storm["name"].upper()
            if storm["name"] and storm["name"].upper() != "UNNAMED"
            else storm["id"]
        )
        flag = (
            "  <-- peak near the Yucatan Channel/W. Caribbean; verify by hand"
            if storm["peak_lat"] < 21.5
            else ""
        )
        peak_ns = "N" if storm["peak_lat"] >= 0 else "S"
        peak_ew = "E" if storm["peak_lon"] >= 0 else "W"
        print(
            f"  - plotting {display_name} ({storm['id']}); peaked at "
            f"{abs(storm['peak_lat']):.2f}{peak_ns}, "
            f"{abs(storm['peak_lon']):.2f}{peak_ew}{flag}"
        )
        plot_active_hurricanes(
            ax=ax,
            time=storm["center_time"].to_pydatetime(),
            extent=extent,
            basin=basin,
            markersize=60,
            lookback_days=storm["lookback_days"],
            lookahead_days=1,
            linecolor=storm_colors[idx],
            storm_ids=[storm["id"]],
            boost_categories=["C4", "C5"],
            boost_factor=2.5,
            only_categories=["C4", "C5"],
        )

    cat_legend = _add_category_legend(ax)
    storm_legend = _add_storm_legend(ax, hurricanes, storm_colors)
    if cat_legend:
        ax.add_artist(cat_legend)

    title = f"Gulf of Mexico Category {min_category}+ Hurricanes"
    ax.set_title(f"{title} — {y0}-{y1}", fontsize=20, fontweight="bold", pad=20)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")
        spine.set_zorder(100)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    save_root = conf.path_plots / "maps" / "atlantic_hurricanes"
    fname = f"gulf_of_mexico_category{min_category}plus_hurricanes_{y0}_to_{y1}_cat4_5_markers.png"
    export_fig(save_root, fname, dpi=conf.dpi)
    plt.close(fig)
    print(f"Saved {fname} to {save_root}")


def main():
    basin = tracks.TrackDataset(basin="north_atlantic", include_btk=True)
    for y0, y1 in YEAR_RANGES:
        _generate_slide(basin, y0, y1)


if __name__ == "__main__":
    main()
