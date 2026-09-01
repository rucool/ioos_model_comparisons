"""
pipeline.py — the per-scene digitizer run, shared by both runners.

This exists because there were two copies of it. `run_digitizer_goes19.py`
(one scene) and `run_digitizer_batch.py` (N days) started as the same
sequence, then only the single-scene one gained ring detection, the map
overlays and the MongoDB writes — so a backfill silently produced a thinner
product than a nightly run. Everything either script does per scene now lives
here; the scripts only differ in how they obtain the SST.

Order matters in one place: rings are detected BEFORE the wall trace, so the
tracked eddies can inform it (see WallConfig.eddy_core_penalty). Files are
written before MongoDB, so a database outage cannot cost the durable artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ioos_model_comparisons.fronts import store
from ioos_model_comparisons.fronts.digitizer import (
    WallConfig, explain_displacement, front_to_geojson, plot_front, read_front,
    trace_front, wall_displacement_km)
from ioos_model_comparisons.fronts.eddies import (
    EddyConfig, detect_eddies, eddies_to_geojson, get_sla, match_eddies,
    plot_eddies, read_eddies)
from ioos_model_comparisons.fronts.webmap import write_overlay_png

WALL_STEM = "gulf_stream_north_wall"
EDDY_STEM = "gulf_stream_eddies"


@dataclass
class SceneOptions:
    """Everything both runners expose as CLI flags."""
    fill_days: int = 3
    dilate_px: int = 2
    min_support: float = 0.9
    max_displacement_km: float = 40.0
    max_local_west_km: float = 75.0
    min_eddy_days: int = 2
    no_eddies: bool = False
    no_eddy_coupling: bool = False
    no_mongo: bool = False
    force_auto: bool = False
    no_plots: bool = False
    verbose: bool = True          # batch prints its own one-line-per-day summary


def find_prior_file(before, out_dir, stem=WALL_STEM):
    """Most recent saved `stem` file from a calendar day before `before`.

    Compares by DAY, not exact timestamp: this is a once-daily product and the
    filename truncates to the minute, so full-precision comparison once let a
    file from the SAME scene read as its own prior at ~0 km displacement.
    """
    prior = None
    for p in sorted(Path(out_dir).glob(f"{stem}_*.geojson")):
        try:
            t = pd.Timestamp(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if t.normalize() < before.normalize() and (prior is None or t > prior[0]):
            prior = (t, p)
    return prior


def prior_wall(before, out_dir, use_mongo=True):
    """(lines, meta) for the most recent day before `before`.

    MongoDB first, because it is the store of record and the only place a hand
    edit exists; then the files, which is what runs when MongoDB is
    unreachable. `meta` records which source answered — a displacement
    measured against a hand-edited, simplified prior is not the same
    measurement as one against an automatic full-resolution prior.
    """
    if use_mongo:
        doc = store.fetch_prior_wall(before.normalize().strftime("%Y-%m-%d"))
        if doc is not None:
            return (store.geometry_to_lines(doc.get("geometry")),
                    {"prior_source": "mongo", "prior_stamp": doc.get("stamp"),
                     "prior_version": doc.get("version"),
                     "prior_origin": doc.get("origin"),
                     "prior_resolution": doc.get("resolution"),
                     "prior_edited_by": doc.get("edited_by")})
    got = find_prior_file(before, out_dir)
    if got is None:
        return None, {"prior_source": None}
    t, path = got
    parts, _ = read_front(path)
    return parts["wall"], {"prior_source": "file",
                           "prior_stamp": f"{t:%Y%m%dT%H%M}",
                           "prior_origin": "unknown",
                           "prior_resolution": "unknown"}


def prior_rings(before, out_dir, use_mongo=True):
    """Yesterday's ring set for days_tracked chaining. Mongo first, then files."""
    if use_mongo:
        doc = store.fetch_prior_rings(before.normalize().strftime("%Y-%m-%d"))
        if doc is not None:
            return store.normalize_rings(doc.get("rings")), "mongo"
    got = find_prior_file(before, out_dir, stem=EDDY_STEM)
    if got is None:
        return [], None
    return read_eddies(got[1]), "file"


def process_scene(sst, age, actual_time, *, extent, out_dir, opts=None,
                  n_composited=None):
    """Run one scene end to end. Returns a dict of everything measured.

    `sst`/`age` come from persistence_fill so the caller controls how the SST
    was obtained — a single fetch for one day, or one windowed read reused
    across a batch.
    """
    import ioos_model_comparisons.configs as conf
    import matplotlib.pyplot as plt
    from cool_maps.plot import add_features, add_ticks

    opts = opts or SceneOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{actual_time:%Y%m%dT%H%M}"      # canonical key: filenames AND Mongo
    say = (lambda *a: print(*a)) if opts.verbose else (lambda *a: None)

    filled_frac = float((age.values > 0).mean())
    if n_composited:
        say(f"Composited {n_composited} scene(s): {filled_frac:.1%} of pixels "
            f"persistence-filled, {float(np.isnan(age.values).mean()):.1%} still missing")

    # ---- rings first: they inform the wall trace --------------------------
    eddies, sla, sla_time, lag, sla_ok = [], None, None, None, False
    if not opts.no_eddies:
        try:
            # get_sla pads the request so edge contours close; drop anything
            # centred in the pad or the archive fills with out-of-region eddies
            sla, sla_time, lag = get_sla(extent, time=actual_time)
            sla_ok = True
            eddies = [e for e in detect_eddies(sla, EddyConfig())
                      if extent[0] <= e["lon"] <= extent[1]
                      and extent[2] <= e["lat"] <= extent[3]]
            ages = [1] * len(eddies)
            prev, _src = prior_rings(actual_time, out_dir,
                                     use_mongo=not opts.no_mongo)
            if prev:
                for k, m in enumerate(match_eddies(prev, eddies)):
                    if m is not None:
                        ages[k] = int(prev[m].get("days_tracked") or 1) + 1
            for e, a in zip(eddies, ages):
                e["days_tracked"] = a
            n_conf = sum(1 for a in ages if a >= opts.min_eddy_days)
            say(f"Rings (CMEMS SLA {sla_time:%Y-%m-%d}, lag {lag:.1f} d): "
                f"{len(eddies)} detected "
                f"({sum(1 for e in eddies if e['kind']=='warm')} warm, "
                f"{sum(1 for e in eddies if e['kind']=='cold')} cold), "
                f"{n_conf} confirmed >={opts.min_eddy_days} d")
            eddy_path = out_dir / f"{EDDY_STEM}_{stamp}.geojson"
            eddies_to_geojson(eddies, eddy_path, time=actual_time, ages=ages,
                              extra={"sla_time": str(sla_time),
                                     "sla_lag_days": round(lag, 2)})
            say(f"Wrote {eddy_path}")
            if not opts.no_mongo:
                rings_doc = []
                for e in eddies:
                    d = dict(e, feature=f"{e['kind']}_core_ring",
                             geometry={"type": "Polygon",
                                       "coordinates": [[[float(x), float(y)]
                                                        for x, y in e["poly"]]]})
                    d.pop("poly", None)
                    rings_doc.append(d)
                v = store.save_rings_version(
                    stamp, rings_doc, origin="auto", source="CMEMS-SLA",
                    sla_time=str(sla_time), sla_lag_days=round(lag, 2),
                    skip_if_manual=not opts.force_auto)
                if v:
                    say(f"  rings -> mongo v{v}")
        except Exception as exc:
            say(f"WARNING: altimetry ring detection failed ({exc}); "
                f"continuing with the wall only.")
            eddies = []

    # ---- wall -------------------------------------------------------------
    cfg = WallConfig(
        lat_bounds=(33.0, 44.0), lon_bounds=(extent[0], extent[1]),
        eddy_core_penalty=0.0 if opts.no_eddy_coupling else WallConfig.eddy_core_penalty)
    coupling_on = bool(eddies) and cfg.eddy_core_penalty > 0
    say("Running trace_front() (gradient anchors + calibrated isotherm"
        + (", eddy-aware)" if coupling_on else ")") + "...")
    trace = trace_front(sst, cfg, age=age,
                        eddies=None if opts.no_eddy_coupling else eddies)
    n_anchor = int(np.isfinite(trace.anchors["lat"].values).sum())
    support_frac = trace.support_frac()
    fill_stats = trace.wall_fill_stats()

    disp, prior_time = None, None
    lines, prior_meta = prior_wall(actual_time, out_dir,
                                   use_mongo=not opts.no_mongo)
    if lines:
        if trace.wall:
            disp = wall_displacement_km(lines, trace.wall)
        prior_time = pd.Timestamp(prior_meta["prior_stamp"])
    else:
        say("No prior saved wall found — skipping displacement check (first run?)")

    qc_pass = support_frac >= opts.min_support
    if disp is not None:
        qc_pass = (qc_pass and disp["median_km"] <= opts.max_displacement_km
                   and disp["local_west_km"] <= opts.max_local_west_km)
    qc_pass = bool(qc_pass)

    say(f"Anchors: {n_anchor} / {trace.lons.size} columns | "
        f"wall: {len(trace.wall)} piece(s), {trace.lon_coverage():.0%} lon coverage | "
        f"rings: {len(trace.warm_rings)} warm, {len(trace.cold_rings)} cold | "
        f"support: {support_frac:.1%} | "
        f"wall on filled px: {fill_stats['frac']:.1%} "
        f"(max age {fill_stats['max_age_days']:.0f} d)" +
        (f" | vs {prior_time:%Y-%m-%d}: median {disp['median_km']:.1f} km, "
         f"local {disp['local_km']:.0f} km @ {disp['worst_lon']:.1f}W "
         f"(west-of-{-disp['west_of']:.1f}W: {disp['local_west_km']:.0f} km)"
         if disp else ""))

    expl = (explain_displacement(disp, eddies, min_days=opts.min_eddy_days)
            if (disp and eddies and not opts.no_eddy_coupling) else None)
    if disp is not None and disp["local_km"] > opts.max_local_west_km \
            and disp["worst_lon"] >= disp["west_of"]:
        where = f"{-disp['worst_lon']:.1f}W/{disp['worst_lat']:.1f}N"
        if expl and expl["explained"]:
            say(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                f"— EXPLAINED: {expl['kind']} ring r={expl['radius_km']:.0f} km "
                f"tracked {expl['days_tracked']} d, {expl['edge_km']:.0f} km away. "
                f"Consistent with real ring/meander evolution.")
        elif expl:
            say(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                f"— NOT explained by a tracked ring (nearest is "
                f"{expl['edge_km']:.0f} km away). East of "
                f"{-disp['west_of']:.1f}W this is still not gated, but with no "
                f"eddy to account for it this one is worth an eyeball.")
        else:
            why = ("eddy coupling disabled" if opts.no_eddy_coupling
                   else "no tracked rings available")
            say(f"NOTE: {disp['local_km']:.0f} km localized change near {where} "
                f"— not evaluated against rings ({why}). East of "
                f"{-disp['west_of']:.1f}W this is not gated.")
    if not qc_pass:
        reasons = []
        if support_frac < opts.min_support:
            reasons.append(f"support {support_frac:.1%} < {opts.min_support:.0%}")
        if disp is not None and disp["median_km"] > opts.max_displacement_km:
            reasons.append(f"median displacement {disp['median_km']:.1f} km "
                           f"> {opts.max_displacement_km:.0f} km")
        if disp is not None and disp["local_west_km"] > opts.max_local_west_km:
            reasons.append(f"localized west-of-{-disp['west_of']:.1f}W "
                           f"{disp['local_west_km']:.0f} km > {opts.max_local_west_km:.0f} km")
        say(f"WARNING: QC FAIL — {'; '.join(reasons)}. "
            f"Outputs are written but marked qc_pass=false; treat this wall as suspect.")

    # ---- wall geojson + mongo ---------------------------------------------
    geojson_path = out_dir / f"{WALL_STEM}_{stamp}.geojson"
    front_to_geojson(trace, geojson_path, time=actual_time, source="GOES-19",
                     extra={"fill_days": opts.fill_days,
                            "dilate_px": opts.dilate_px,
                            "filled_frac": round(filled_frac, 4),
                            "mean_fill_age_days": round(float(np.nanmean(
                                age.values[age.values > 0])), 2) if filled_frac else 0.0,
                            "min_support": opts.min_support,
                            "max_displacement_km": opts.max_displacement_km,
                            "max_local_west_km": opts.max_local_west_km,
                            "disp_median_km": round(disp["median_km"], 1) if disp else None,
                            "disp_max_km": round(disp["max_km"], 1) if disp else None,
                            "disp_local_km": round(disp["local_km"], 1) if disp else None,
                            "disp_local_west_km": round(disp["local_west_km"], 1) if disp else None,
                            "disp_worst_lon": round(disp["worst_lon"], 2) if disp and disp["worst_lon"] else None,
                            "disp_worst_lat": round(disp["worst_lat"], 2) if disp and disp["worst_lat"] else None,
                            "disp_explained_by_ring": expl["explained"] if expl else None,
                            "disp_ring_edge_km": round(expl["edge_km"], 1) if expl else None,
                            "disp_ring_days_tracked": expl["days_tracked"] if expl else None,
                            "eddy_coupling": not opts.no_eddy_coupling,
                            **prior_meta,
                            "qc_pass": qc_pass})
    say(f"Wrote {geojson_path}")
    wall_version = None
    if not opts.no_mongo:
        # skip_if_manual: a routine re-run must not silently supersede a hand
        # edit, since "current" is simply the highest version.
        wall_version = store.save_wall_version(
            stamp, store.lines_to_geometry(trace.wall),
            read_front(geojson_path)[1], origin="auto", resolution="full",
            source="GOES-19", qc_pass=qc_pass,
            skip_if_manual=not opts.force_auto)
        if wall_version:
            say(f"  wall -> mongo v{wall_version}")

    # ---- figures ----------------------------------------------------------
    # Two maps of the same scene built by one function with identical
    # figsize/extent/projection/colorbar geometry, so the axes land on the
    # same pixels and you can flip between them.
    qc_tag = "PASS" if qc_pass else "FAIL"
    title_tail = (f"[QC {qc_tag}: support {support_frac:.0%}"
                  + (f", displacement {disp['median_km']:.0f} km" if disp else "") + "]")

    # Each model's own 200m/15C isotherm, as stored by plotting.py's
    # _save_isotherm_lines() against the MAB region -- shown on the SST panel
    # only (below), one color per model, and only if MongoDB actually has
    # something for this day; there is no on-disk fallback for these.
    model_isotherms = {}
    if not opts.no_mongo:
        try:
            from ioos_model_comparisons.regions import region_config
            mab_folder = region_config('mab')['folder']
            model_isotherms = store.fetch_isotherm_lines_for_day(
                mab_folder, actual_time.normalize().strftime('%Y-%m-%d'))
            if model_isotherms:
                say(f"Model 200m isotherms from mongo: {', '.join(sorted(model_isotherms))}")
        except Exception as exc:
            say(f"WARNING: could not fetch model isotherms from MongoDB: {exc}")
    ISOTHERM_COLORS = ['orange', 'dodgerblue', 'lime', 'purple', 'gold', 'deeppink', 'saddlebrown']

    def build_map(field_kind):
        fig, ax = plt.subplots(figsize=(14, 8),
                               subplot_kw={"projection": conf.projection["map"]},
                               layout="constrained")
        ax.set_extent(extent, crs=conf.projection["data"])
        add_features(ax)
        add_ticks(ax, extent, label_left=True)
        if field_kind == "sst":
            # tighter than the region config's 14-30C: p1-p99 in the 35-42N
            # band is ~21-29.6C, so a 14C floor wastes ~40% of the colormap
            h = ax.pcolormesh(sst["lon"], sst["lat"], sst, cmap="turbo",
                              vmin=20, vmax=29, transform=conf.projection["data"])
            cb_label = "Sea Water Temperature (°C)"
            title = f"GOES-19 SST  {actual_time:%Y-%m-%d %H:%M UTC}   {title_tail}"
            if filled_frac:
                # stipple filled pixels: the front moves 10-20 km/day, so
                # stale fill is least reliable exactly where it is changing
                ax.contourf(sst["lon"], sst["lat"],
                            np.where(age.values > 0, 1.0, np.nan),
                            levels=[0.5, 1.5], colors="none", hatches=["...."],
                            transform=conf.projection["data"])
        else:
            h = ax.pcolormesh(sla["longitude"], sla["latitude"], sla,
                              cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                              transform=conf.projection["data"])
            cb_label = "Sea Level Anomaly (m)"
            title = (f"CMEMS altimetry SLA  {sla_time:%Y-%m-%d}"
                     f" (lag {lag:.1f} d)   {title_tail}")
        fig.colorbar(h, ax=ax, orientation="horizontal", shrink=0.6, pad=0.05,
                     label=cb_label)
        plot_front(ax, trace, transform=conf.projection["data"],
                   wall_kw=dict(label="Digitized north wall"))
        # Rings are drawn on the SLA panel only. They are derived from the
        # sea-level field, so that is where they are interpretable; on the SST
        # figure they clutter the thermal front without adding information,
        # since SST is not what detected them. Detection, the GeoJSON, MongoDB
        # and the web editor are unaffected — this is presentation only.
        if eddies and field_kind == "sla":
            plot_eddies(ax, eddies, min_days=opts.min_eddy_days,
                        transform=conf.projection["data"])
        if field_kind == "sst":
            for m_i, (model_name, lines) in enumerate(sorted(model_isotherms.items())):
                color = ISOTHERM_COLORS[m_i % len(ISOTHERM_COLORS)]
                for i, l in enumerate(lines):
                    ax.plot(l[:, 0], l[:, 1], "-", color=color, lw=1.75, zorder=15,
                            transform=conf.projection["data"],
                            label=f"{model_name} 15°C (200m)" if i == 0 else None)
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
    if not opts.no_plots:
        # No bbox_inches="tight": it crops to the drawn extent, which differs
        # between panels (colorbar label width, title length), shifting the map
        # by a few pixels and ruining flip-comparison.
        for kind, name in (("sst", WALL_STEM), ("sla", "gulf_stream_sla")):
            if kind == "sla" and not sla_ok:
                continue
            fig = build_map(kind)
            p = out_dir / f"{name}_{stamp}.png"
            fig.savefig(p, dpi=conf.dpi)
            plt.close(fig)
            written.append(p)
            say(f"Wrote {p}")
        if len(written) == 2:
            try:
                from PIL import Image
                sizes = [Image.open(p).size for p in written]
                if sizes[0] != sizes[1]:
                    say(f"WARNING: the two PNGs differ in size {sizes[0]} vs "
                        f"{sizes[1]} — they will not register when flipped.")
            except ImportError:
                pass

    # ---- bare Web Mercator overlays for the Leaflet editor ----------------
    overlays = {}
    overlays["sst"] = write_overlay_png(
        sst.values, sst["lat"].values, sst["lon"].values, extent,
        out_dir / f"gulf_stream_overlay_sst_{stamp}.png",
        cmap="turbo", vmin=20, vmax=29)
    if sla_ok:
        overlays["sla"] = write_overlay_png(
            sla.values, sla["latitude"].values, sla["longitude"].values, extent,
            out_dir / f"gulf_stream_overlay_sla_{stamp}.png",
            cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    (out_dir / f"gulf_stream_overlay_{stamp}.json").write_text(json.dumps({
        "time": str(actual_time), "extent": list(extent), "fields": overlays}))
    say(f"Wrote {len(overlays)} map overlay(s) + sidecar for the editor")
    n_overlay = 0
    if opts.no_mongo:
        say("  overlays NOT sent to mongo (--no-mongo)")
    else:
        # The editor runs on a different host from the digitizer and MongoDB is
        # the only channel that already spans both, so imagery goes here rather
        # than down a second rsync path that can silently go stale.
        for f, meta in overlays.items():
            png = out_dir / f"gulf_stream_overlay_{f}_{stamp}.png"
            if png.is_file() and store.save_overlay(
                    stamp, f, png.read_bytes(),
                    dict(meta, region="gulf_stream", stamp=stamp, field=f)):
                n_overlay += 1
        if n_overlay:
            say(f"  {n_overlay} overlay(s) -> mongo")
        else:
            # Silence here used to be ambiguous between "disabled", "database
            # unreachable" and "the PNG was missing" — name which it was.
            say(f"  WARNING: 0 of {len(overlays)} overlay(s) reached mongo "
                f"(is MONGODB_URI set and the database reachable?)")

    return {
        "stamp": stamp, "time": actual_time, "trace": trace,
        "n_anchor": n_anchor, "n_cols": int(trace.lons.size),
        "lon_coverage": trace.lon_coverage(), "n_pieces": len(trace.wall),
        "support_frac": support_frac, "fill_stats": fill_stats,
        "scene_filled_frac": filled_frac, "disp": disp, "qc_pass": qc_pass,
        "explained": expl, "prior_meta": prior_meta,
        "n_eddies": len(eddies),
        "n_warm": sum(1 for e in eddies if e.get("kind") == "warm"),
        "n_cold": sum(1 for e in eddies if e.get("kind") == "cold"),
        "n_confirmed": sum(1 for e in eddies
                           if (e.get("days_tracked") or 1) >= opts.min_eddy_days),
        "wall_version": wall_version, "n_overlays": n_overlay,
        "sla_ok": sla_ok,
    }
