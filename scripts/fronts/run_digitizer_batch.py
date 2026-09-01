"""
run_digitizer_batch.py — run the full digitizer over the past N days.

Does exactly what run_digitizer_goes19.py does for one scene — wall, rings
from altimetry, both map figures, the Leaflet overlays and the MongoDB
writes — repeated over a date range, plus a QC summary CSV across the run.
The per-scene work is shared via ioos_model_comparisons/fronts/pipeline.py so
a backfill and a nightly run cannot produce different products.

The one thing it does differently is fetching: the whole SST time window is
read ONCE and re-sliced per day, instead of a fresh remote read each time.
The CMEMS SLA dataset handle is likewise opened once (see eddies._open_sla),
which takes a 14-day run from ~2 minutes of redundant opening to ~15 seconds.

Ring persistence (days_tracked) chains day to day, so run a range in
chronological order — which this does — for the counts to build up.

Usage:
    python3 scripts/fronts/run_digitizer_batch.py --days 14
    python3 scripts/fronts/run_digitizer_batch.py --days 30 --no-plots
    python3 scripts/fronts/run_digitizer_batch.py --days 5 --no-eddies --no-mongo
"""
import argparse

import matplotlib
matplotlib.use("agg")
import numpy as np
import pandas as pd

from ioos_model_comparisons.env import load_env
from ioos_model_comparisons.fronts import DEFAULT_OUTPUT_DIR
from ioos_model_comparisons.fronts.digitizer import persistence_fill
from ioos_model_comparisons.fronts.pipeline import SceneOptions, process_scene
from ioos_model_comparisons.platforms import get_goes
from ioos_model_comparisons.regions import region_config

OUT_DIR = DEFAULT_OUTPUT_DIR
FILL_DAYS = 3


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=14, help="how many past days (default 14)")
    p.add_argument("--end", type=pd.Timestamp, default=None,
                   help="last day to include (default: latest available scene)")
    p.add_argument("--dilate-px", type=int, default=2, metavar="N",
                   help="grow each scene's cloud mask N px (default 2)")
    p.add_argument("--min-support", type=float, default=0.9, metavar="F")
    p.add_argument("--max-displacement-km", type=float, default=40.0, metavar="KM")
    p.add_argument("--max-local-west-km", type=float, default=75.0, metavar="KM")
    p.add_argument("--min-eddy-days", type=int, default=2, metavar="N")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the two per-day figures (overlays are still written, "
                        "since the editor needs them)")
    p.add_argument("--no-eddies", action="store_true", help="skip altimetry rings")
    p.add_argument("--no-eddy-coupling", action="store_true")
    p.add_argument("--no-mongo", action="store_true", help="files only")
    p.add_argument("--force-auto", action="store_true",
                   help="write automatic versions even for days that already have "
                        "a hand-edited one in MongoDB")
    p.add_argument("--verbose", action="store_true",
                   help="full per-scene output instead of one line per day")
    return p.parse_args()


def main():
    args = parse_args()
    load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extent = region_config("gulf_stream")["extent"]

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
          f"{len(targets)} target day(s)\n")

    opts = SceneOptions(
        fill_days=FILL_DAYS, dilate_px=args.dilate_px,
        min_support=args.min_support,
        max_displacement_km=args.max_displacement_km,
        max_local_west_km=args.max_local_west_km,
        min_eddy_days=args.min_eddy_days, no_eddies=args.no_eddies,
        no_eddy_coupling=args.no_eddy_coupling, no_mongo=args.no_mongo,
        force_auto=args.force_auto, no_plots=args.no_plots,
        verbose=args.verbose)

    rows = []
    for t in targets:
        t = pd.Timestamp(t)
        # Re-slice the window already in memory rather than re-reading it.
        stack = window.sel(time=slice(t - pd.Timedelta(days=FILL_DAYS), t))
        sst, age = persistence_fill(stack, dilate_px=args.dilate_px)
        try:
            r = process_scene(sst, age, t, extent=extent, out_dir=OUT_DIR,
                              opts=opts, n_composited=stack.sizes["time"])
        except Exception as exc:
            # One bad day must not abandon the rest of the range.
            print(f"{t:%Y-%m-%d}: FAILED ({exc})")
            rows.append(dict(date=f"{t:%Y-%m-%d}", failed=str(exc)))
            continue

        d = r["disp"]
        rows.append(dict(
            date=f"{t:%Y-%m-%d}", time=str(r["time"]), stamp=r["stamp"],
            n_anchor=r["n_anchor"], n_cols=r["n_cols"],
            lon_coverage=round(r["lon_coverage"], 3), n_pieces=r["n_pieces"],
            support_frac=round(r["support_frac"], 4) if np.isfinite(r["support_frac"]) else None,
            wall_fill_frac=round(r["fill_stats"]["frac"], 4) if r["fill_stats"] else None,
            scene_filled_frac=round(r["scene_filled_frac"], 4),
            n_eddies=r["n_eddies"], n_warm=r["n_warm"], n_cold=r["n_cold"],
            n_confirmed=r["n_confirmed"],
            disp_median_km=round(d["median_km"], 1) if d else None,
            disp_local_km=round(d["local_km"], 1) if d else None,
            disp_local_west_km=round(d["local_west_km"], 1) if d else None,
            worst_lon=round(d["worst_lon"], 2) if d and d["worst_lon"] else None,
            explained_by_ring=(r["explained"] or {}).get("explained"),
            prior_source=r["prior_meta"].get("prior_source"),
            wall_version=r["wall_version"], n_overlays=r["n_overlays"],
            qc_pass=r["qc_pass"]))

        if not args.verbose:
            print(f"{t:%Y-%m-%d}: anchors {r['n_anchor']}/{r['n_cols']} | "
                  f"{r['lon_coverage']:.0%} in {r['n_pieces']} piece(s) | "
                  f"rings {r['n_warm']}w/{r['n_cold']}c ({r['n_confirmed']} conf) | "
                  f"support {r['support_frac']:.1%} | "
                  + (f"median {d['median_km']:.1f} km | " if d else "")
                  + f"qc {'pass' if r['qc_pass'] else 'FAIL'}"
                  + (f" | mongo v{r['wall_version']}" if r["wall_version"] else "")
                  + f" | overlays->db {r['n_overlays']}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"qc_summary_{df.date.min()}_{df.date.max()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    ok = df[df.get("failed").isna()] if "failed" in df else df
    if len(ok):
        print("\n--- summary ---")
        for col, label in (("support_frac", "support"),
                           ("wall_fill_frac", "wall on filled px"),
                           ("disp_median_km", "day-over-day median (km)"),
                           ("disp_local_west_km", "localized west of 68.5W (km)")):
            v = ok[col].dropna() if col in ok else []
            if len(v):
                print(f"  {label:32s} min={v.min():.3g} median={v.median():.3g} max={v.max():.3g}")
        if "n_eddies" in ok:
            print(f"  {'rings per day':32s} min={ok.n_eddies.min()} "
                  f"median={ok.n_eddies.median():.0f} max={ok.n_eddies.max()}")
        if "n_overlays" in ok:
            sent, want = int(ok.n_overlays.sum()), 2 * len(ok)
            print(f"  {'overlays written to mongo':32s} {sent}/{want}"
                  + ("" if sent == want else
                     "   <- some did not reach the database; check MONGODB_URI"))
        print(f"  qc_pass: {int(ok.qc_pass.sum())}/{len(ok)} days")
        for _, r in ok[~ok.qc_pass].iterrows():
            print(f"    FAIL {r.date}: support {r.support_frac}, "
                  f"median {r.disp_median_km} km, local-west {r.disp_local_west_km} km")
    if "failed" in df and df.failed.notna().any():
        print(f"  {int(df.failed.notna().sum())} day(s) errored — see rows in the CSV")


if __name__ == "__main__":
    main()
