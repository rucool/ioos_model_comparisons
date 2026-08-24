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

Backfill many days at once with scripts/fronts/run_digitizer_batch.py (wall + QC only).
"""
import argparse
import json

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
    read_front, wall_displacement_km, explain_displacement)
from ioos_model_comparisons.fronts.webmap import write_overlay_png
from ioos_model_comparisons.fronts import store
from ioos_model_comparisons.env import load_env
from ioos_model_comparisons.fronts.eddies import (
    EddyConfig, get_sla, detect_eddies, match_eddies, eddies_to_geojson,
    read_eddies, plot_eddies)
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


def _find_prior_wall(before, out_dir, stem="gulf_stream_north_wall"):
    """Most recent saved `stem` file from a calendar day strictly earlier
    than `before`'s, or None if there isn't one (first-ever run, or a gap
    wider than anything on disk).

    Compares by DAY, not exact timestamp: this is a once-daily product, and
    the saved filename truncates to the minute while `before` (straight off
    the source dataset) carries fractional seconds — comparing at full
    precision let a file from the SAME scene, written by an earlier run,
    read as its own "prior" at a ~0 km displacement.
    """
    candidates = sorted(out_dir.glob(f"{stem}_*.geojson"))
    today = before.normalize()
    prior = None
    for p in candidates:
        try:
            t = pd.Timestamp(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if t.normalize() < today and (prior is None or t > prior[0]):
            prior = (t, p)
    return prior


def _prior_wall(before, out_dir, use_mongo=True):
    """(lines, meta) for the most recent day before `before`.

    Mongo first, because it is the store of record and the only place a hand
    edit exists; then the files in out_dir, which is what runs when Mongo is
    unreachable (no tunnel from a laptop). `meta` records which source
    answered so the QC number written into the output can be interpreted
    later — a comparison against a hand-edited, simplified prior is not the
    same measurement as one against an automatic full-resolution prior.
    """
    if use_mongo:
        day = before.normalize().strftime("%Y-%m-%d")
        doc = store.fetch_prior_wall(day)
        if doc is not None:
            return (store.geometry_to_lines(doc.get("geometry")),
                    {"prior_source": "mongo", "prior_stamp": doc.get("stamp"),
                     "prior_version": doc.get("version"),
                     "prior_origin": doc.get("origin"),
                     "prior_resolution": doc.get("resolution"),
                     "prior_edited_by": doc.get("edited_by")})
    prior = _find_prior_wall(before, out_dir)
    if prior is None:
        return None, {"prior_source": None}
    t, path = prior
    parts, _ = read_front(path)
    return parts["wall"], {"prior_source": "file", "prior_stamp": f"{t:%Y%m%dT%H%M}",
                           "prior_origin": "unknown", "prior_resolution": "unknown"}


def _prior_rings(before, out_dir, use_mongo=True):
    """Yesterday's ring set, for days_tracked chaining. Mongo first, then files."""
    if use_mongo:
        day = before.normalize().strftime("%Y-%m-%d")
        doc = store.fetch_prior_rings(day)
        if doc is not None:
            return store.normalize_rings(doc.get("rings")), "mongo"
    pe = _find_prior_wall(before, out_dir, stem="gulf_stream_eddies")
    if pe is None:
        return [], None
    return read_eddies(pe[1]), "file"


def main():
    args = parse_args()
    load_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    region = region_config("gulf_stream")
    extent = region["extent"]

    print("Loading GOES-19 SST (this fetches the full remote dataset lazily)...")
    sst_full = get_goes(satellite="goes19")
    if sst_full is None:
        print("Failed to load GOES-19 data.")
        return

    ctime = args.time or pd.Timestamp(sst_full.time.values[-1])
    sub = sst_full["cleaned_sst"].sel(
        lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3])
    )
    actual_time = pd.Timestamp(sub.sel(time=ctime, method="nearest").time.values)
    print(f"Requested {ctime}, using nearest scene {actual_time}")
    stamp = f"{actual_time:%Y%m%dT%H%M}"      # canonical key: filenames AND Mongo

    stack = sub.sel(
        time=slice(actual_time - pd.Timedelta(days=args.fill_days), actual_time)
    ).load()
    sst, age = persistence_fill(stack, dilate_px=args.dilate_px)
    filled_frac = float((age > 0).mean())
    print(f"Composited {stack.sizes['time']} scene(s): {filled_frac:.1%} of pixels "
          f"persistence-filled, {float(np.isnan(age).mean()):.1%} still missing")

    # ---- rings first: they inform the wall trace ---------------------------
    # Detected before trace_front so the tracked eddies can bias the wall away
    # from ring cores. Altimetry is an independent sensor, so this is real
    # added information rather than another constraint derived from the same
    # SST field the wall is being traced from.
    eddies, eddy_path, sla_ok = [], None, False
    sla = sla_time = lag = None
    if not args.no_eddies:
        try:
            # get_sla pads the request so contours near the edge close
            # properly; drop anything whose centre lands in the pad, or the
            # archive fills with eddies that are not in the region at all.
            sla, sla_time, lag = get_sla(extent, time=actual_time)
            sla_ok = True                 # the SLA panel only needs this much
            eddies = [e for e in detect_eddies(sla, EddyConfig())
                      if extent[0] <= e["lon"] <= extent[1]
                      and extent[2] <= e["lat"] <= extent[3]]
            # carry a persistence count forward from yesterday's file
            ages = [1] * len(eddies)
            prior_eddies, rings_src = _prior_rings(actual_time, OUT_DIR,
                                                   use_mongo=not args.no_mongo)
            if prior_eddies:
                for k, m in enumerate(match_eddies(prior_eddies, eddies)):
                    if m is not None:
                        ages[k] = int(prior_eddies[m].get("days_tracked") or 1) + 1
            for e, a in zip(eddies, ages):
                e["days_tracked"] = a
            n_conf = sum(1 for a in ages if a >= args.min_eddy_days)
            print(f"Rings (CMEMS SLA {sla_time:%Y-%m-%d}, lag {lag:.1f} d): "
                  f"{len(eddies)} detected "
                  f"({sum(1 for e in eddies if e['kind']=='warm')} warm, "
                  f"{sum(1 for e in eddies if e['kind']=='cold')} cold), "
                  f"{n_conf} confirmed >={args.min_eddy_days} d")
            eddy_path = OUT_DIR / f"gulf_stream_eddies_{actual_time:%Y%m%dT%H%M}.geojson"
            eddies_to_geojson(eddies, eddy_path, time=actual_time, ages=ages,
                              extra={"sla_time": str(sla_time),
                                     "sla_lag_days": round(lag, 2)})
            print(f"Wrote {eddy_path}")
            # Mongo second: the file is the durable artifact the figures and
            # plot_front depend on, so a database outage must not cost us it.
            if not args.no_mongo:
                rings_doc = [dict(e, geometry={
                    "type": "Polygon",
                    "coordinates": [[[float(x), float(y)] for x, y in e["poly"]]]},
                    feature=f"{e['kind']}_core_ring")
                    for e in eddies]
                for r in rings_doc:
                    r.pop("poly", None)
                v = store.save_rings_version(
                    stamp, rings_doc, origin="auto", source="CMEMS-SLA",
                    sla_time=str(sla_time), sla_lag_days=round(lag, 2),
                    skip_if_manual=not args.force_auto)
                if v:
                    print(f"  rings -> mongo v{v}")
        except Exception as exc:
            print(f"WARNING: altimetry ring detection failed ({exc}); "
                  f"continuing with the wall only.")
            eddies = []

    # No hand-tuned latitude cap: the per-longitude corridor in WallConfig
    # keeps the path off the Slope Sea rings and the shelf-slope front, so the
    # search window can stay wide. Verified against a hand-checked wall west
    # of 65W; see GS_CORRIDOR_LAT for the caveat about the eastern anchors.
    cfg = WallConfig(
        lat_bounds=(33.0, 44.0),
        lon_bounds=(extent[0], extent[1]),
        eddy_core_penalty=0.0 if args.no_eddy_coupling else WallConfig.eddy_core_penalty,
    )
    # Only claim "eddy-aware" if the penalty is actually non-zero; it ships
    # off by default because it measured as a no-op (see WallConfig).
    coupling_on = bool(eddies) and cfg.eddy_core_penalty > 0
    print("Running trace_front() (gradient anchors + calibrated isotherm"
          + (", eddy-aware)" if coupling_on else ")") + "...")
    trace = trace_front(sst, cfg, age=age,
                        eddies=None if args.no_eddy_coupling else eddies)
    n_anchor = int(np.isfinite(trace.anchors["lat"].values).sum())
    support_frac = trace.support_frac()
    fill_stats = trace.wall_fill_stats()

    # Day-over-day displacement against the most recent prior saved wall.
    # This is the check the support/coverage checks structurally can't do:
    # they verify a front exists and is drawn continuously, not that it's
    # the SAME front as yesterday. A ring wrapped instead of passed by can
    # score >99% support and 100% coverage and still be the wrong feature.
    disp = None
    prior_lines, prior_meta = _prior_wall(actual_time, OUT_DIR,
                                          use_mongo=not args.no_mongo)
    if prior_lines:
        if trace.wall:
            disp = wall_displacement_km(prior_lines, trace.wall)
        prior_time = pd.Timestamp(prior_meta["prior_stamp"])
    else:
        print("No prior saved wall found — skipping displacement check (first run?)")

    qc_pass = support_frac >= args.min_support
    if disp is not None:
        qc_pass = (qc_pass and disp["median_km"] <= args.max_displacement_km
                   and disp["local_west_km"] <= args.max_local_west_km)
    qc_pass = bool(qc_pass)

    print(f"Anchors: {n_anchor} / {trace.lons.size} columns | "
          f"wall: {len(trace.wall)} piece(s), {trace.lon_coverage():.0%} lon coverage | "
          f"rings: {len(trace.warm_rings)} warm, {len(trace.cold_rings)} cold | "
          f"support: {support_frac:.1%} | "
          f"wall on filled px: {fill_stats['frac']:.1%} "
          f"(max age {fill_stats['max_age_days']:.0f} d)" +
          (f" | vs {prior_time:%Y-%m-%d}: median {disp['median_km']:.1f} km, "
           f"local {disp['local_km']:.0f} km @ {disp['worst_lon']:.1f}W "
           f"(west-of-{-disp['west_of']:.1f}W: {disp['local_west_km']:.0f} km)"
           if disp else ""))
    # Altimetry closes the gap the distance metric could not: a hotspot on a
    # long-lived tracked ring is the ocean moving, not the trace slipping.
    expl = (explain_displacement(disp, eddies, min_days=args.min_eddy_days)
            if (disp and eddies and not args.no_eddy_coupling) else None)
    if disp is not None and disp["local_km"] > args.max_local_west_km \
            and disp["worst_lon"] >= disp["west_of"]:
        where = f"{-disp['worst_lon']:.1f}W/{disp['worst_lat']:.1f}N"
        if expl and expl["explained"]:
            print(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                  f"— EXPLAINED: {expl['kind']} ring r={expl['radius_km']:.0f} km "
                  f"tracked {expl['days_tracked']} d, {expl['edge_km']:.0f} km away. "
                  f"Consistent with real ring/meander evolution.")
        elif expl:
            print(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                  f"— NOT explained by a tracked ring (nearest is "
                  f"{expl['edge_km']:.0f} km away). East of "
                  f"{-disp['west_of']:.1f}W this is still not gated, but with no "
                  f"eddy to account for it this one is worth an eyeball.")
        else:
            # no ring check ran (coupling off, altimetry down, or no tracked
            # rings) — say so rather than implying a negative result
            why = ("eddy coupling disabled" if args.no_eddy_coupling
                   else "no tracked rings available")
            print(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                  f"— not evaluated against rings ({why}). East of "
                  f"{-disp['west_of']:.1f}W this is not gated.")
    if not qc_pass:
        reasons = []
        if support_frac < args.min_support:
            reasons.append(f"support {support_frac:.1%} < {args.min_support:.0%}")
        if disp is not None and disp["median_km"] > args.max_displacement_km:
            reasons.append(f"median displacement {disp['median_km']:.1f} km > {args.max_displacement_km:.0f} km")
        if disp is not None and disp["local_west_km"] > args.max_local_west_km:
            reasons.append(f"localized west-of-{-disp['west_of']:.1f}W "
                           f"{disp['local_west_km']:.0f} km > {args.max_local_west_km:.0f} km")
        print(f"WARNING: QC FAIL — {'; '.join(reasons)}. "
              f"Outputs are written but marked qc_pass=false; treat this wall as suspect.")

    geojson_path = OUT_DIR / f"gulf_stream_north_wall_{actual_time:%Y%m%dT%H%M}.geojson"
    front_to_geojson(trace, geojson_path, time=actual_time, source="GOES-19",
                     extra={"fill_days": args.fill_days,
                            "dilate_px": args.dilate_px,
                            "filled_frac": round(filled_frac, 4),
                            "mean_fill_age_days": round(float(np.nanmean(age.values[age.values > 0])), 2) if filled_frac else 0.0,
                            "min_support": args.min_support,
                            "max_displacement_km": args.max_displacement_km,
                            "max_local_west_km": args.max_local_west_km,
                            "disp_median_km": round(disp["median_km"], 1) if disp else None,
                            "disp_max_km": round(disp["max_km"], 1) if disp else None,
                            "disp_local_km": round(disp["local_km"], 1) if disp else None,
                            "disp_local_west_km": round(disp["local_west_km"], 1) if disp else None,
                            "disp_worst_lon": round(disp["worst_lon"], 2) if disp and disp["worst_lon"] else None,
                            "disp_worst_lat": round(disp["worst_lat"], 2) if disp and disp["worst_lat"] else None,
                            "disp_explained_by_ring": expl["explained"] if expl else None,
                            "disp_ring_edge_km": round(expl["edge_km"], 1) if expl else None,
                            "disp_ring_days_tracked": expl["days_tracked"] if expl else None,
                            "eddy_coupling": not args.no_eddy_coupling,
                            # which wall the displacement was measured against:
                            # comparing to a hand-edited, simplified prior is not
                            # the same measurement as comparing to an automatic
                            # full-resolution one, and the number is meaningless
                            # later without knowing which it was
                            **prior_meta,
                            "qc_pass": qc_pass})
    print(f"Wrote {geojson_path}")
    if not args.no_mongo:
        # skip_if_manual: a routine re-run must not silently supersede a hand
        # edit, since "current" is simply the highest version.
        wall_props = read_front(geojson_path)[1]
        v = store.save_wall_version(
            stamp, store.lines_to_geometry(trace.wall), wall_props,
            origin="auto", resolution="full", source="GOES-19",
            qc_pass=qc_pass, skip_if_manual=not args.force_auto)
        if v:
            print(f"  wall -> mongo v{v}")

    # ---- figures ----------------------------------------------------------
    # Two maps of the same scene — SST and altimetry — built by one function
    # with identical figsize/extent/projection/colorbar geometry, so the map
    # axes land on the same pixels in both PNGs and you can flip between them
    # to compare. Anything that changes the decoration size (title length,
    # colorbar label, legend) is kept structurally identical for that reason;
    # `verify_registration` below asserts it actually held.
    qc_tag = "PASS" if qc_pass else "FAIL"
    title_tail = (f"[QC {qc_tag}: support {support_frac:.0%}"
                  + (f", displacement {disp['median_km']:.0f} km" if disp else "") + "]")

    def build_map(field_kind):
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": conf.projection["map"]},
            layout="constrained",
        )
        ax.set_extent(extent, crs=conf.projection["data"])
        add_features(ax)
        add_ticks(ax, extent, label_left=True)

        if field_kind == "sst":
            # Deliberately tighter than the region config's 14-30C: in the
            # 35-42N band where the wall lives, p1-p99 is ~21-29.6C, so a 14C
            # floor spends roughly 40% of the colormap on temperatures that
            # barely occur here and flattens the cross-frontal contrast.
            h = ax.pcolormesh(sst["lon"], sst["lat"], sst, cmap="turbo",
                              vmin=20, vmax=29, transform=conf.projection["data"])
            cb_label = "Sea Water Temperature (°C)"
            title = f"GOES-19 SST  {actual_time:%Y-%m-%d %H:%M UTC}   {title_tail}"
            # stipple persistence-filled pixels so staleness stays visible —
            # the front moves ~10-20 km/day, so filled areas are least
            # reliable right where a meander is actively changing
            if filled_frac:
                ax.contourf(sst["lon"], sst["lat"],
                            np.where(age.values > 0, 1.0, np.nan),
                            levels=[0.5, 1.5], colors="none", hatches=["...."],
                            transform=conf.projection["data"])
        else:
            # diverging, symmetric about zero: sign is the physical signal
            # (positive = anticyclonic = warm core)
            h = ax.pcolormesh(sla["longitude"], sla["latitude"], sla,
                              cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                              transform=conf.projection["data"])
            cb_label = "Sea Level Anomaly (m)"
            title = (f"CMEMS altimetry SLA  {sla_time:%Y-%m-%d}"
                     f" (lag {lag:.1f} d)   {title_tail}")

        fig.colorbar(h, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05,
                     label=cb_label)

        # features common to both panels, so they register when flipping
        plot_front(ax, trace, transform=conf.projection["data"],
                   wall_kw=dict(label="Digitized north wall"))
        if eddies:
            # short-lived rings dashed, so an unconfirmed detection never
            # looks as solid as one tracked across several days
            plot_eddies(ax, eddies, min_days=args.min_eddy_days,
                        transform=conf.projection["data"])

        if field_kind == "sst":
            # SST-only QC layers: these describe the SST retrieval, not the
            # ocean, so they would be misleading drawn over altimetry
            first_bad = True
            for l, s in zip(trace.wall, trace.support):
                if s.all():
                    continue
                ax.plot(l[:, 0], np.where(~s, l[:, 1], np.nan), "-", color="red",
                        lw=3.5, zorder=22, transform=conf.projection["data"],
                        label="unsupported (no gradient)" if first_bad else None)
                first_bad = False
            ax.plot(trace.anchors["lon"], trace.anchors["lat"], color="0.35",
                    lw=1, ls="--", transform=conf.projection["data"],
                    label="gradient anchors")
            if disp is not None and disp["worst_lon"] is not None and disp["local_km"] > 50:
                ax.plot(disp["worst_lon"], disp["worst_lat"], "o", mfc="none",
                        mec="yellow", mew=2.5, ms=18, zorder=25,
                        transform=conf.projection["data"],
                        label=f"largest local change ({disp['local_km']:.0f} km)")

        ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85,
                  borderpad=0.4, columnspacing=1.0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return fig

    written = []
    # NOTE: no bbox_inches="tight" here. It crops to the drawn extent, which
    # differs between the two panels (colorbar label width, title length), and
    # that shifts the map by a few pixels — enough to ruin flipping between
    # them. A fixed figsize with constrained layout gives both PNGs identical
    # dimensions and identical axes placement.
    for kind, name in (("sst", "gulf_stream_north_wall"),
                       ("sla", "gulf_stream_sla")):
        if kind == "sla" and not sla_ok:
            continue                      # altimetry unavailable this run
        fig = build_map(kind)
        p = OUT_DIR / f"{name}_{stamp}.png"
        fig.savefig(p, dpi=conf.dpi)
        plt.close(fig)
        written.append(p)
        print(f"Wrote {p}")

    if len(written) == 2:
        try:
            from PIL import Image
            sizes = [Image.open(p).size for p in written]
            if sizes[0] != sizes[1]:
                print(f"WARNING: the two PNGs differ in size {sizes[0]} vs "
                      f"{sizes[1]} — they will not register when flipped.")
        except ImportError:
            pass

    # ---- bare Web Mercator overlays for the Leaflet editor ----------------
    # The figures above carry titles, colorbars and axes, so they cannot be
    # used as map overlays. These are decoration-free and reprojected to Web
    # Mercator so L.imageOverlay places them correctly (see webmap.py — an
    # equirectangular image stretched to lat/lon bounds is off by ~35 km at
    # mid-domain). Written here because the fields are already in memory;
    # the editor then never has to refetch GOES or CMEMS.
    overlays = {}
    overlays["sst"] = write_overlay_png(
        sst.values, sst["lat"].values, sst["lon"].values, extent,
        OUT_DIR / f"gulf_stream_overlay_sst_{stamp}.png",
        cmap="turbo", vmin=20, vmax=29)
    if sla_ok:
        overlays["sla"] = write_overlay_png(
            sla.values, sla["latitude"].values, sla["longitude"].values, extent,
            OUT_DIR / f"gulf_stream_overlay_sla_{stamp}.png",
            cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    # sidecar so the overlays are self-describing — the editor should not have
    # to assume the region extent still matches what a given PNG was drawn to
    (OUT_DIR / f"gulf_stream_overlay_{stamp}.json").write_text(json.dumps({
        "time": str(actual_time), "extent": list(extent), "fields": overlays}))
    print(f"Wrote {len(overlays)} map overlay(s) + sidecar for the editor")
    if not args.no_mongo:
        # The editor runs on a different host from the digitizer, and Mongo is
        # the only channel that already spans both — hence imagery here rather
        # than a second rsync path that can silently go stale.
        n = 0
        for field, meta in overlays.items():
            png = OUT_DIR / f"gulf_stream_overlay_{field}_{stamp}.png"
            if png.is_file() and store.save_overlay(
                    stamp, field, png.read_bytes(),
                    dict(meta, region="gulf_stream", stamp=stamp, field=field)):
                n += 1
        if n:
            print(f"  {n} overlay(s) -> mongo")


if __name__ == "__main__":
    main()
