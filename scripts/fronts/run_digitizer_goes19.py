"""
run_digitizer_goes19.py — digitize the Gulf Stream north wall and rings for
one scene, with QC.

Pipeline, for the GOES-19 "cleaned_sst" scene nearest `--time`:
  1. composite the last --fill-days scenes to fill cloud (digitizer.persistence_fill)
  2. trace the wall (digitizer.trace_front: gradient anchors -> calibrated isotherm)
  3. detect rings from CMEMS altimetry (eddies.detect_eddies) -- NOT from SST;
     see eddies.py for the measurement behind that choice
  4. QC against the most recent prior saved day: gradient support, wall
     displacement, and ring persistence (days_tracked)

Writes to outputs/gulf_stream_fronts/:
    gulf_stream_north_wall_<stamp>.geojson   wall + QC properties
    gulf_stream_eddies_<stamp>.geojson       rings + days_tracked
    gulf_stream_north_wall_<stamp>.png       SST map
    gulf_stream_sla_<stamp>.png              altimetry map, same extent/size

The two PNGs are pixel-registered — same figsize, extent, projection and
colorbar geometry — so you can flip back and forth between them to compare
the thermal front against the sea-level field. Both carry the wall and ring
overlays; SST-only QC layers (fill stipple, gradient anchors, unsupported
spans) appear on the SST map only, since they describe the retrieval rather
than the ocean.

Outputs are always written; a QC failure sets qc_pass=false and prints a
warning rather than suppressing them.

Usage:
    python3 scripts/fronts/run_digitizer_goes19.py
    python3 scripts/fronts/run_digitizer_goes19.py --time 2026-08-16T10:55
    python3 scripts/fronts/run_digitizer_goes19.py --no-eddies      # skip the CMEMS call

Ring persistence (days_tracked) chains from the previous day's eddies file,
so run days in order if you want a populated archive; a first run on an empty
directory reports every ring as unconfirmed.

Per-scene work lives in ioos_model_comparisons/fronts/pipeline.py, shared with
scripts/fronts/run_digitizer_batch.py so the two cannot drift apart.
"""
import argparse

import matplotlib
matplotlib.use("agg")
import pandas as pd


from ioos_model_comparisons.env import load_env
from ioos_model_comparisons.fronts import DEFAULT_OUTPUT_DIR
from ioos_model_comparisons.fronts.digitizer import persistence_fill
from ioos_model_comparisons.fronts.pipeline import SceneOptions, process_scene
from ioos_model_comparisons.platforms import get_goes
from ioos_model_comparisons.regions import region_config

OUT_DIR = DEFAULT_OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time", type=pd.Timestamp, default=None,
                    help="Scene to digitize (default: latest available GOES-19 scene)")
    p.add_argument("--fill-days", type=int, default=3, metavar="N",
                    help="Fill cloud holes from up to N prior days (0 disables; default 3)")
    p.add_argument("--dilate-px", type=int, default=2, metavar="N",
                    help="Grow each scene's cloud mask N px to kill cold edge residue (default 2)")
    p.add_argument("--min-support", type=float, default=0.9, metavar="F",
                    help="QC gate: flag the scene if fewer than this fraction of wall "
                         "vertices are confirmed by the independent gradient check (default 0.9)")
    p.add_argument("--max-displacement-km", type=float, default=40.0, metavar="KM",
                    help="QC gate: flag the scene if the wall's median displacement from "
                         "the most recent prior saved wall exceeds this (default 40 km; "
                         "set from a 2-week archive where daily median never exceeded 15.2 "
                         "km even on visibly bad days — not yet validated beyond that)")
    p.add_argument("--max-local-west-km", type=float, default=75.0, metavar="KM",
                    help="QC gate: flag if any 50 km stretch of wall WEST of 68.5W sits "
                         "this far from the prior wall along its whole length (default 75 km; "
                         "archive max there was 36 km). Not applied east of 68.5W, where real "
                         "meander/ring evolution is indistinguishable from a derailment.")
    p.add_argument("--no-mongo", action="store_true",
                    help="Skip writing to MongoDB (files only). The prior-day QC "
                         "lookup then falls back to the files in --out-dir.")
    p.add_argument("--force-auto", action="store_true",
                    help="Write an automatic version even for a day that already has "
                         "a hand-edited one in Mongo. Without this, the digitizer "
                         "refuses, so a routine re-run cannot silently supersede an edit.")
    p.add_argument("--no-eddies", action="store_true",
                    help="Skip altimetry ring detection (CMEMS SLA)")
    p.add_argument("--no-eddy-coupling", action="store_true",
                    help="Detect rings but do not use them to interpret the wall. "
                         "Disables the QC explanation (whether a displacement hotspot "
                         "sits on a tracked ring) and the WallConfig.eddy_core_penalty "
                         "path, which ships off because it A/B-tested as a no-op.")
    p.add_argument("--min-eddy-days", type=int, default=2, metavar="N",
                    help="Rings tracked fewer than N consecutive days are drawn dashed and "
                         "flagged unconfirmed (default 2). 39 of 87 tracks in the August 2026 "
                         "archive were single-day; persistence is the best quality signal "
                         "available without ground truth.")
    return p.parse_args()


def main():
    args = parse_args()
    load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extent = region_config("gulf_stream")["extent"]

    print("Loading GOES-19 SST (this fetches the full remote dataset lazily)...")
    sst_full = get_goes(satellite="goes19")
    if sst_full is None:
        print("Failed to load GOES-19 data.")
        return

    ctime = args.time or pd.Timestamp(sst_full.time.values[-1])
    sub = sst_full["cleaned_sst"].sel(
        lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))
    actual_time = pd.Timestamp(sub.sel(time=ctime, method="nearest").time.values)
    print(f"Requested {ctime}, using nearest scene {actual_time}")

    stack = sub.sel(time=slice(
        actual_time - pd.Timedelta(days=args.fill_days), actual_time)).load()
    sst, age = persistence_fill(stack, dilate_px=args.dilate_px)

    # Everything per-scene lives in fronts/pipeline.py so this script and
    # run_digitizer_batch.py cannot drift apart again.
    process_scene(sst, age, actual_time, extent=extent, out_dir=OUT_DIR,
                  n_composited=stack.sizes["time"],
                  opts=SceneOptions(
                      fill_days=args.fill_days, dilate_px=args.dilate_px,
                      min_support=args.min_support,
                      max_displacement_km=args.max_displacement_km,
                      max_local_west_km=args.max_local_west_km,
                      min_eddy_days=args.min_eddy_days,
                      no_eddies=args.no_eddies,
                      no_eddy_coupling=args.no_eddy_coupling,
                      no_mongo=args.no_mongo, force_auto=args.force_auto))


if __name__ == "__main__":
    main()
