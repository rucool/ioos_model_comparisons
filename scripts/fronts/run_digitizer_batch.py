"""
run_digitizer_batch.py — run trace_front() over the past N days and build a
QC summary archive, so thresholds (--min-support, a future fill-frac gate,
day-over-day displacement) can be set from real distributions instead of
guessed from two scenes.

Loads the full time window ONCE (one remote read), then reuses it for every
day's persistence_fill() rather than reopening the dataset per day.

Note: this backfills the WALL and its QC only — it does not detect rings.
Use run_digitizer_goes19.py, day by day in order, if you want the altimetry
ring archive (ring persistence chains from the previous day's file).

Usage:
    python3 scripts/fronts/run_digitizer_batch.py --days 14
"""
import argparse

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cool_maps.plot import add_features, add_ticks

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.fronts import DEFAULT_OUTPUT_DIR
from ioos_model_comparisons.fronts.digitizer import (
    WallConfig, persistence_fill, trace_front, front_to_geojson, plot_front,
    wall_displacement_km)
from ioos_model_comparisons.platforms import get_goes
from ioos_model_comparisons.regions import region_config

OUT_DIR = DEFAULT_OUTPUT_DIR
FILL_DAYS = 3
DILATE_PX = 2
MIN_SUPPORT = 0.9
# Set from the first two-week archive: median displacement never exceeded
# 15.2 km/day even on the two visibly worst days (fragmented wall, a ring
# wrapped instead of passed by) — both still read as ~9 km median because a
# long stable line dilutes a localized derailment. 40 km leaves ~2.6x margin
# above the observed max while still catching a wholesale wrong-feature jump.
# Revisit once more days accumulate; this is not yet a validated threshold.
MAX_DISP_KM = 40.0
# Localized gate, west of 68.5W only (the stable, hand-validated sector):
# the same archive never exceeded 36 km there, so 75 km (~2x) is meaningful.
# East of 68.5W the localized metric is REPORTED but not gated — real
# meander/ring evolution produced 90-270 km localized displacement on 9 of
# 13 archive days, indistinguishable from a derailment (the confirmed
# 2026-08-16 ring-wrap included) without ground truth.
MAX_LOCAL_WEST_KM = 75.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=14, help="How many past days to run (default 14)")
    p.add_argument("--end", type=pd.Timestamp, default=None,
                    help="Last day to include (default: latest available scene)")
    p.add_argument("--no-plots", action="store_true", help="Skip per-day PNGs, write GeoJSON only")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    region = region_config("gulf_stream")
    extent = region["extent"]

    print("Loading GOES-19 SST (one remote read for the whole window)...")
    sst_full = get_goes(satellite="goes19")
    if sst_full is None:
        print("Failed to load GOES-19 data.")
        return
    sub = sst_full["cleaned_sst"].sel(lon=slice(extent[0], extent[1]),
                                      lat=slice(extent[2], extent[3]))
    end = args.end or pd.Timestamp(sub.time.values[-1])
    start = end - pd.Timedelta(days=args.days + FILL_DAYS)
    window = sub.sel(time=slice(start, end)).load()
    times = pd.to_datetime(window.time.values)
    targets = times[times >= (end - pd.Timedelta(days=args.days - 1))]
    print(f"Window loaded: {len(times)} scenes {times[0]} .. {times[-1]}; "
          f"{len(targets)} target day(s) to digitize")

    cfg = WallConfig(lat_bounds=(33.0, 44.0), lon_bounds=(extent[0], extent[1]))

    rows = []
    prev_wall = None
    prev_date = None
    for t in targets:
        stack = window.sel(time=slice(t - pd.Timedelta(days=FILL_DAYS), t))
        sst, age = persistence_fill(stack, dilate_px=DILATE_PX)
        filled_frac = float((age.values > 0).mean())

        trace = trace_front(sst, cfg, age=age)
        n_anchor = int(np.isfinite(trace.anchors["lat"].values).sum())
        support_frac = trace.support_frac()
        fill_stats = trace.wall_fill_stats()

        disp = None
        if prev_wall is not None and trace.wall:
            disp = wall_displacement_km(prev_wall, trace.wall)

        qc_pass = bool(support_frac >= MIN_SUPPORT)
        if disp is not None:
            qc_pass = (qc_pass and disp["median_km"] <= MAX_DISP_KM
                       and disp["local_west_km"] <= MAX_LOCAL_WEST_KM)

        row = dict(
            date=pd.Timestamp(t).strftime("%Y-%m-%d"), time=str(pd.Timestamp(t)),
            n_anchor=n_anchor, n_cols=trace.lons.size,
            lon_coverage=round(trace.lon_coverage(), 3), n_pieces=len(trace.wall),
            n_warm_rings=len(trace.warm_rings), n_cold_rings=len(trace.cold_rings),
            support_frac=round(support_frac, 4) if np.isfinite(support_frac) else None,
            wall_fill_frac=round(fill_stats["frac"], 4) if fill_stats else None,
            wall_mean_fill_age_d=round(fill_stats["mean_age_days"], 2) if fill_stats else None,
            wall_max_fill_age_d=round(fill_stats["max_age_days"], 2) if fill_stats else None,
            scene_filled_frac=round(filled_frac, 4),
            qc_pass=qc_pass,
            disp_median_km=round(disp["median_km"], 1) if disp else None,
            disp_max_km=round(disp["max_km"], 1) if disp else None,
            disp_local_km=round(disp["local_km"], 1) if disp else None,
            disp_local_west_km=round(disp["local_west_km"], 1) if disp else None,
            worst_lon=round(disp["worst_lon"], 2) if disp and disp["worst_lon"] else None,
            worst_lat=round(disp["worst_lat"], 2) if disp and disp["worst_lat"] else None,
            disp_n=disp["n"] if disp else None,
        )
        rows.append(row)
        print(f"{row['date']}: anchors {n_anchor}/{row['n_cols']} | "
              f"coverage {row['lon_coverage']:.0%} in {row['n_pieces']} piece(s) | "
              f"rings {row['n_warm_rings']}w/{row['n_cold_rings']}c | "
              f"support {support_frac:.1%} | fill {row['wall_fill_frac']:.1%} | "
              f"qc_pass {qc_pass}" +
              (f" | vs {prev_date}: median {disp['median_km']:.1f} km, "
               f"local {disp['local_km']:.0f} km @ {disp['worst_lon']:.1f}W "
               f"(west {disp['local_west_km']:.0f} km)" if disp else ""))

        geojson_path = OUT_DIR / f"gulf_stream_north_wall_{pd.Timestamp(t):%Y%m%dT%H%M}.geojson"
        front_to_geojson(trace, geojson_path, time=pd.Timestamp(t), source="GOES-19",
                         extra={"fill_days": FILL_DAYS, "dilate_px": DILATE_PX,
                                "scene_filled_frac": round(filled_frac, 4),
                                "min_support": MIN_SUPPORT, "qc_pass": qc_pass,
                                "disp_median_km": row["disp_median_km"],
                                "disp_max_km": row["disp_max_km"],
                                "disp_local_km": row["disp_local_km"],
                                "disp_local_west_km": row["disp_local_west_km"],
                                "disp_worst_lon": row["worst_lon"],
                                "disp_worst_lat": row["worst_lat"]})

        if not args.no_plots:
            fig, ax = plt.subplots(figsize=(14, 8),
                                   subplot_kw={"projection": conf.projection["map"]},
                                   layout="constrained")
            ax.set_extent(extent, crs=conf.projection["data"])
            add_features(ax); add_ticks(ax, extent, label_left=True)
            h = ax.pcolormesh(sst["lon"], sst["lat"], sst, cmap="turbo",
                              vmin=20, vmax=29, transform=conf.projection["data"])
            if filled_frac:
                ax.contourf(sst["lon"], sst["lat"], np.where(age.values > 0, 1.0, np.nan),
                            levels=[0.5, 1.5], colors="none", hatches=["...."],
                            transform=conf.projection["data"])
            fig.colorbar(h, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05,
                         label="Sea Water Temperature (°C)")
            plot_front(ax, trace, transform=conf.projection["data"],
                      wall_kw=dict(label="Digitized north wall"))
            first_bad = True
            for l, s in zip(trace.wall, trace.support):
                if s.all():
                    continue
                ax.plot(l[:, 0], np.where(~s, l[:, 1], np.nan), "-", color="red",
                        lw=3.5, zorder=22, transform=conf.projection["data"],
                        label="unsupported (no gradient)" if first_bad else None)
                first_bad = False
            ax.legend(loc="lower left", fontsize=9)
            qc_tag = "PASS" if qc_pass else "FAIL"
            ax.set_title(f"GOES-19  {pd.Timestamp(t):%Y-%m-%d %H:%M UTC}   "
                         f"[QC {qc_tag}: support {support_frac:.0%}]",
                         fontsize=14, fontweight="bold")
            png_path = OUT_DIR / f"gulf_stream_north_wall_{pd.Timestamp(t):%Y%m%dT%H%M}.png"
            fig.savefig(png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

        prev_wall = trace.wall
        prev_date = row["date"]

    df = pd.DataFrame(rows)
    # name by span, not a hardcoded "2weeks" — --days is variable
    csv_path = OUT_DIR / f"qc_summary_{df.date.min()}_{df.date.max()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print("\n--- summary ---")
    print(f"support_frac:     min={df.support_frac.min():.1%} "
          f"median={df.support_frac.median():.1%} max={df.support_frac.max():.1%}")
    print(f"wall_fill_frac:   min={df.wall_fill_frac.min():.1%} "
          f"median={df.wall_fill_frac.median():.1%} max={df.wall_fill_frac.max():.1%}")
    d = df.disp_median_km.dropna()
    if len(d):
        print(f"day-over-day median displacement: min={d.min():.1f} km "
              f"median={d.median():.1f} km max={d.max():.1f} km")
    lw = df.disp_local_west_km.dropna()
    if len(lw):
        print(f"localized, west of 68.5W (GATED):     min={lw.min():.1f} km "
              f"median={lw.median():.1f} km max={lw.max():.1f} km")
    la = df.disp_local_km.dropna()
    if len(la):
        print(f"localized, whole domain (reported):   min={la.min():.1f} km "
              f"median={la.median():.1f} km max={la.max():.1f} km")
        print("  ^ not gated: real meander/ring evolution east of 68.5W is "
              "indistinguishable from derailment by this metric")
    print(f"qc_pass: {df.qc_pass.sum()}/{len(df)} days")
    for _, r in df[~df.qc_pass].iterrows():
        print(f"  FAIL {r.date}: support {r.support_frac:.1%}, "
              f"median {r.disp_median_km} km, local-west {r.disp_local_west_km} km")


if __name__ == "__main__":
    main()
