"""
Map every Argo float surfacing (one dot per profile) in the Southwest Indian
Ocean over a date range, on the same map frame as
``southern_africa_cyclone_tracks.py --extent 20 80 -40 -5``.

Positions come from Ifremer's ERDDAP ``ArgoFloats`` dataset via
``platforms.get_argo_floats_by_time``. By default dots are colored by month
and each float's surfacings are joined by a faint line to show its drift.
``--green`` instead draws plain green dots (the Argo style of
plotting.map_add_all_argo) with no lines or colorbar, and a float/profile
count legend in the upper left.

    python southern_africa_argo_surfacings.py
    python southern_africa_argo_surfacings.py --green
    python southern_africa_argo_surfacings.py --start 2025-10-01 --end 2026-09-30
    python southern_africa_argo_surfacings.py --green --days 5   # last 5 days
"""
import argparse
import datetime as dt

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.platforms import get_argo_floats_by_time
from ioos_model_comparisons.plotting import export_fig

# Run as a script, so the sibling module is importable from this directory.
from southern_africa_cyclone_tracks import _configure_axes


def main(start, end, extent, green=False):
    # The helper's time filter is inclusive, so stop at the last second of `end`.
    t1 = dt.datetime.combine(end, dt.time(23, 59, 59))
    df = get_argo_floats_by_time(
        bbox=tuple(extent), time_start=dt.datetime.combine(start, dt.time()),
        time_end=t1,
    )
    if df.empty:
        raise SystemExit("No Argo surfacings returned (ERDDAP empty or unreachable).")

    df = df.reset_index().drop_duplicates(["argo", "time"])
    n_floats = df["argo"].nunique()
    print(f"{len(df)} surfacings from {n_floats} floats "
          f"({df['time'].min():%Y-%m-%d} to {df['time'].max():%Y-%m-%d})")

    # One color per calendar month in the window.
    months = pd.period_range(start, end, freq="M")
    month_idx = (df["time"].dt.to_period("M") - months[0]).apply(lambda d: d.n)
    cmap = plt.get_cmap("turbo", len(months))
    norm = mcolors.BoundaryNorm(np.arange(len(months) + 1) - 0.5, len(months))

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=conf.projection["map"])
    _configure_axes(ax, extent)

    if green:
        # Same marker as plotting.map_add_all_argo, scaled down for
        # thousands of points.
        ax.scatter(
            df["lon"], df["lat"], color="green", s=30, edgecolor="black",
            linewidths=0.4, transform=ccrs.PlateCarree(), zorder=30000,
        )
        handles = [
            mlines.Line2D([], [], marker="o", linestyle="none", color="green",
                          markeredgecolor="black", markersize=9,
                          label=f"{n_floats} unique floats"),
            mlines.Line2D([], [], linestyle="none",
                          label=f"{len(df):,} total profiles"),
        ]
        legend = ax.legend(handles=handles, title="Argo", loc="upper left",
                           fontsize=13, title_fontsize=14, framealpha=0.95,
                           borderpad=0.6)
        legend.get_title().set_fontweight("bold")
        legend.get_frame().set_edgecolor("0.3")
        legend.set_zorder(100001)
    else:
        for _, track in df.sort_values("time").groupby("argo"):
            ax.plot(track["lon"], track["lat"], color="0.25", linewidth=0.6,
                    alpha=0.5, transform=ccrs.PlateCarree(), zorder=29999)

        sc = ax.scatter(
            df["lon"], df["lat"], c=month_idx, cmap=cmap, norm=norm,
            s=16, edgecolor="black", linewidths=0.3,
            transform=ccrs.PlateCarree(), zorder=30000,
        )

        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05,
                          fraction=0.035, aspect=50, ticks=np.arange(len(months)))
        cb.ax.set_xticklabels([m.strftime("%b\n%Y") for m in months],
                              fontsize=10, fontweight="bold")
        cb.set_label("Surfacing month", fontsize=12, fontweight="bold")

    ax.set_title(
        f"Southwest Indian Ocean Argo Float Surfacings — "
        f"{start:%b %-d, %Y} to {end:%b %-d, %Y}\n"
        f"Ifremer ERDDAP ArgoFloats ({len(df):,} surfacings from {n_floats} floats)",
        fontsize=18, fontweight="bold", pad=15,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")
        spine.set_zorder(100)

    save_root = conf.path_plots / "maps" / "southwest_indian_argo"
    fname = (f"southwest_indian_argo_surfacings_{start:%Y%m%d}_to_{end:%Y%m%d}"
             f"_{extent[0]:g}E-{extent[1]:g}E{'_green' if green else ''}.png")
    export_fig(save_root, fname, dpi=conf.dpi)
    plt.close(fig)
    print(f"Saved {fname} to {save_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=dt.date.fromisoformat,
                        default=dt.date(2025, 10, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat,
                        default=dt.date(2026, 9, 30), help="inclusive")
    parser.add_argument("--extent", type=float, nargs=4,
                        default=[20, 80, -40, -5],
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--days", type=int,
                        help="last N days ending today (overrides --start/--end)")
    parser.add_argument("--green", action="store_true",
                        help="plain green dots with a count legend, no drift lines or colorbar")
    args = parser.parse_args()
    if args.days:
        args.end = dt.date.today()
        args.start = args.end - dt.timedelta(days=args.days - 1)
    main(args.start, args.end, args.extent, args.green)
