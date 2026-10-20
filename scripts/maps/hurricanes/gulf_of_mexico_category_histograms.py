"""
Generate yearly bar-chart histograms of hurricanes that passed through the
Gulf of Mexico (region defined in ``hurricane_history_by_years.py``).

A storm counts only if it reached the threshold *while inside* the Gulf of
Mexico polygon (see ``ioos_model_comparisons.gulf_of_mexico``) -- a storm
that peaked at category 4/5 in the Caribbean or the Straits of Florida and
merely brushed the Gulf at lower intensity does not count, matching the
methodology used in ``hurricane_history_by_years.py`` /
``hurricane_history_by_years_cat4_5_markers.py``.

Three histograms are produced, since "at least category N" is inherently
cumulative -- a storm that peaks at category 5 also satisfies "reached at
least category 4":

- Category 5: storms that reached category 5.
- Category 4+ (cumulative): storms that reached category 4 OR 5. A
  category 5 storm counts here too, so these bars are always >= the
  category 5 bars for the same year.
- Category 4 (peak only): storms whose peak was category 4 and did NOT
  reach category 5. Mutually exclusive from the category 5 histogram --
  these bars plus the category 5 bars sum to the category 4+ bars.
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
from tropycal import tracks

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.gulf_of_mexico import bounding_box_mask, gulf_mask
from ioos_model_comparisons.plotting import export_fig

# Minimum sustained wind speed (knots) for each Saffir-Simpson category.
CATEGORY_MIN_KT = {1: 64, 2: 83, 3: 96, 4: 113, 5: 137}


def _year_counts(basin, year, exclude_names=frozenset(), force_include_names=frozenset()):
    """Return (cat5, cat4_plus, cat4_only) counts of storms that reached at
    least category 4 *while inside the Gulf of Mexico polygon* during `year`.

    Classification uses each storm's peak intensity while inside the Gulf,
    not its lifetime peak -- this matches the methodology used in
    ``hurricane_history_by_years.py`` / ``hurricane_history_by_years_cat4_5_
    markers.py``: a storm that only reached category 4/5 in the Caribbean or
    Straits of Florida does not count here even if it also brushed the Gulf
    at a lower intensity. Storms that reach the threshold outside the Gulf
    are printed so they can be double-checked by hand. Any name in
    `exclude_names` is dropped regardless of its in-region peak; any name in
    `force_include_names` falls back to the coarser rectangular bounding box
    instead of the polygon (e.g. Irma, kept in despite peaking in the
    Straits of Florida).
    """
    season = basin.get_season(year)
    storm_ids = season.summary()["id"]

    cat5 = cat4_plus = cat4_only = 0

    for storm_id in storm_ids:
        storm = basin.get_storm(storm_id)
        vmax = np.asarray(storm.dict["vmax"], dtype=float)
        if np.isnan(vmax).all():
            continue

        name = str(getattr(storm, "name", storm_id)).upper()
        if name in exclude_names:
            continue

        lons = np.asarray(storm.dict["lon"], dtype=float)
        lats = np.asarray(storm.dict["lat"], dtype=float)
        in_region = bounding_box_mask(lons, lats) if name in force_include_names else gulf_mask(lons, lats)
        if not np.any(in_region):
            continue

        vmax_in_region = np.where(in_region, vmax, np.nan)
        if np.isnan(vmax_in_region).all():
            continue
        peak = np.nanmax(vmax_in_region)

        if peak < CATEGORY_MIN_KT[4]:
            if np.nanmax(vmax) >= CATEGORY_MIN_KT[4]:
                print(
                    f"  (excluded) {year} {name} ({storm_id}): reached the "
                    f"category threshold outside the region only — verify by hand"
                )
            continue

        cat4_plus += 1
        if peak >= CATEGORY_MIN_KT[5]:
            cat5 += 1
        else:
            cat4_only += 1

    return cat5, cat4_plus, cat4_only


def _plot_histogram(years, counts, title, fname, color):
    fig, ax = plt.subplots(figsize=(max(12, len(years) * 0.35), 6))

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="black", alpha=0.3)

    bars = ax.bar(years, counts, color=color, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        if count:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Storms", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    save_root = conf.path_plots / "maps" / "atlantic_hurricanes"
    export_fig(save_root, fname, dpi=conf.dpi)
    plt.close(fig)
    print(f"Saved {fname} to {save_root}")


def main():
    y0 = 1980
    y1 = datetime.date.today().year
    # Storm names to omit by hand -- matches the map scripts. Now largely
    # redundant with the Gulf polygon, kept as documented belt-and-suspenders.
    # DELTA's only Cat 4/5 point (120kt at 18.5N, -83.3E on 2020-10-06) was
    # in the NW Caribbean; it had weakened to Cat 3 (105kt) by the time it
    # reached the Gulf.
    exclude_names = {"DELTA"}
    # Storms kept in despite peaking outside the Gulf polygon -- matches the
    # map scripts. Irma's only Cat 4 points sit in the Straits of Florida
    # near Key West.
    force_include_names = {"IRMA"}
    basin = tracks.TrackDataset(basin="north_atlantic", include_btk=True)

    years = list(range(y0, y1 + 1))
    counts_cat5 = []
    counts_cat4_plus = []
    counts_cat4_only = []

    for year in years:
        try:
            cat5, cat4_plus, cat4_only = _year_counts(basin, year, exclude_names, force_include_names)
        except Exception as exc:
            print(f"Warning: Could not load season {year}: {exc}")
            cat5, cat4_plus, cat4_only = 0, 0, 0

        counts_cat5.append(cat5)
        counts_cat4_plus.append(cat4_plus)
        counts_cat4_only.append(cat4_only)

    _plot_histogram(
        years, counts_cat5,
        f"Gulf of Mexico Category 5 Hurricanes by Year ({y0}-{y1})",
        f"gulf_of_mexico_category5_histogram_{y0}_to_{y1}.png",
        color="#660066",
    )
    _plot_histogram(
        years, counts_cat4_plus,
        f"Gulf of Mexico Category 4+ Hurricanes by Year ({y0}-{y1})\n(cumulative -- includes category 5 storms)",
        f"gulf_of_mexico_category4plus_cumulative_histogram_{y0}_to_{y1}.png",
        color="#990000",
    )
    _plot_histogram(
        years, counts_cat4_only,
        f"Gulf of Mexico Category 4 Hurricanes by Year ({y0}-{y1})\n(peaked at category 4, excludes category 5)",
        f"gulf_of_mexico_category4only_histogram_{y0}_to_{y1}.png",
        color="#cc6600",
    )


if __name__ == "__main__":
    main()
