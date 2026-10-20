"""
Three-panel summary figure of Gulf Category 4+ hurricanes:

- top: cumulative histogram of Cat 4+ storms per year (1980 - current year)
- bottom left / right: track maps for two decades (default 1996-2005, 2016-2025)

Reuses the helpers from ``gulf_of_mexico_category_histograms.py`` and
``hurricane_history_by_years_cat4_5_markers.py`` so the methodology (Gulf
polygon, excluded/forced storms, C4/C5-only markers) is identical.
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
from cool_maps.plot import get_bathymetry
from tropycal import tracks

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.plotting import export_fig, plot_active_hurricanes

import gulf_of_mexico_category_histograms as hist
import hurricane_history_by_years_cat4_5_markers as maps

DPI = 500
MAP_RANGES = [(1996, 2005), (2016, 2025)]
MIN_CATEGORY = 4
# Two styles: "full" for full-page-width use, "large_text" for when the figure
# is shrunk to ~half size in a document (bigger fonts/legends, every other year).
STYLES = {
    "full": dict(title=16, label=12, tick=9, map_tick=12, bar=9, legend=8,
                 legend_marker=8, year_step=1, tick_len=7, tick_w=2,
                 hist_ratio=0.7, hspace=0.48, wspace=0.14, fname_suffix=""),
    "large_text": dict(title=17, label=20, tick=16, map_tick=18, bar=15, legend=14,
                       legend_marker=9, year_step=2, tick_len=9, tick_w=2.5,
                       hist_ratio=0.9, hspace=0.55, wspace=0.2,
                       fname_suffix="_large_text"),
}
ST = STYLES["full"]
EXTENT = [-98.25, -79.75, 17.75, 31.25]


def _draw_histogram(ax, basin, y0, y1):
    years = list(range(y0, y1 + 1))
    counts = []
    for year in years:
        try:
            _, cat4_plus, _ = hist._year_counts(
                basin, year, maps.EXCLUDE_NAMES, maps.FORCE_INCLUDE_NAMES
            )
        except Exception as exc:
            print(f"Warning: Could not load season {year}: {exc}")
            cat4_plus = 0
        counts.append(cat4_plus)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="black", alpha=0.3)
    # Shade the decades that are mapped below
    for r0, r1 in MAP_RANGES:
        ax.axvspan(r0 - 0.5, r1 + 0.5, color="#F2C94C", alpha=0.35, zorder=0,
                   linewidth=0)
        ax.axvline(r0 - 0.5, color="#C9A227", linewidth=1.5, zorder=1)
        ax.axvline(r1 + 0.5, color="#C9A227", linewidth=1.5, zorder=1)
    bars = ax.bar(years, counts, color="#990000", edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(count), ha="center", va="bottom", fontsize=ST['bar'], fontweight="bold")

    ax.set_xlabel("Year", fontsize=ST['label'], fontweight="bold")
    ax.set_ylabel("Number of Storms", fontsize=ST['label'], fontweight="bold")
    ax.set_title(
        f"Gulf Category 4 or 5 Hurricanes \u2014 {y0}-{y1}",
        fontsize=ST['title'], fontweight="bold", pad=10,
    )
    # Label every other year so the larger labels don't run into each other
    label_years = years[::ST['year_step']]
    other_years = [y for y in years if y not in label_years]
    # Bold, longer major ticks sit under the labeled years; the in-between
    # years get thin minor ticks.
    ax.set_xticks(label_years)
    ax.set_xticklabels([str(y) for y in label_years], rotation=45, ha="right",
                       fontsize=ST['tick'])
    ax.set_xticks(other_years, minor=True)
    ax.tick_params(axis="x", which="major", length=ST['tick_len'] + 2,
                   width=ST['tick_w'] + 0.5)
    ax.tick_params(axis="x", which="minor", length=ST['tick_len'] - 3, width=1)
    ax.tick_params(axis="y", labelsize=ST['tick'])
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0, max(counts) + 0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _enlarge_legend(legend):
    """Bump legend text and marker sizes (they are built small in the shared
    map module) so they stay readable when the figure is shrunk to fit a page."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(ST['legend'])
    legend.get_title().set_fontsize(ST['legend'] + 1)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(ST['legend_marker'])
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(3)


def _draw_map(ax, basin, bathy, y0, y1, show_category_key=True):
    wind_threshold = maps.CATEGORY_MIN_KT[MIN_CATEGORY]
    hurricanes = []
    for year in range(y0, y1 + 1):
        try:
            hurricanes.extend(
                maps._collect_hurricanes(
                    basin, year, wind_threshold,
                    force_include_names=maps.FORCE_INCLUDE_NAMES,
                )
            )
        except Exception as exc:
            print(f"Warning: Could not load season {year}: {exc}")
    hurricanes = [h for h in hurricanes if str(h["name"]).upper() not in maps.EXCLUDE_NAMES]

    maps._configure_axes(ax, EXTENT, bathy=bathy)
    colors = maps._distinct_colors(len(hurricanes))
    for storm, color in zip(hurricanes, colors):
        plot_active_hurricanes(
            ax=ax,
            time=storm["center_time"].to_pydatetime(),
            extent=EXTENT,
            basin=basin,
            markersize=60,
            lookback_days=storm["lookback_days"],
            lookahead_days=1,
            linecolor=color,
            storm_ids=[storm["id"]],
            boost_categories=["C4", "C5"],
            boost_factor=2.5,
            only_categories=["C4", "C5"],
        )

    # The C4/C5 key is only drawn on the first map: on the 2016-2025 map it sits
    # on top of Harvey's only Cat 4+ points, and the key is the same for both.
    cat_legend = maps._add_category_legend(ax) if show_category_key else None
    storm_legend = maps._add_storm_legend(ax, hurricanes, colors)
    if cat_legend:
        ax.add_artist(cat_legend)
    for leg in (cat_legend, storm_legend):
        _enlarge_legend(leg)
    ax.tick_params(axis="both", which="major", labelsize=ST['map_tick'], length=ST['tick_len'], width=ST['tick_w'])

    ax.set_title(f"Gulf Category {MIN_CATEGORY} or 5 Hurricanes — {y0}-{y1}",
                 fontsize=ST['title'], fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")
        spine.set_zorder(100)


def main():
    h0, h1 = 1980, datetime.date.today().year
    basin = tracks.TrackDataset(basin="north_atlantic", include_btk=True)
    try:
        bathy = get_bathymetry(EXTENT)
    except Exception as exc:
        bathy = None
        print(f"Warning: Could not load bathymetry: {exc}")

    global ST
    for name, ST in STYLES.items():
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[ST["hist_ratio"], 1.5],
                              hspace=ST["hspace"], wspace=ST["wspace"])

        _draw_histogram(fig.add_subplot(gs[0, :]), basin, h0, h1)
        for col, (y0, y1) in enumerate(MAP_RANGES):
            ax = fig.add_subplot(gs[1, col], projection=conf.projection["map"])
            _draw_map(ax, basin, bathy, y0, y1, show_category_key=(col == 0))

        save_root = conf.path_plots / "maps" / "atlantic_hurricanes"
        fname = f"gulf_category4plus_summary_figure{ST['fname_suffix']}.png"
        export_fig(save_root, fname, dpi=DPI)
        plt.close(fig)
        print(f"Saved {fname} to {save_root}")


if __name__ == "__main__":
    main()
