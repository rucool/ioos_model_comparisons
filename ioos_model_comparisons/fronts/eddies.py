"""
eddies.py — Gulf Stream ring detection from satellite altimetry (CMEMS SLA).

Why altimetry and not SST: rings are geostrophic features with a sea-level
signature that does not care about seasonal surface heating. SST-based ring
detection fails in summer for both ring types independently — cold-core rings
get capped by a warm seasonal mixed layer so their core is invisible from
above, and warm-core rings sit in a Slope Sea whose surface is also warm.
Measured on a 14-day August 2026 archive, scoring each method by whether a
detection reappears within 60 km the next day (real rings live for weeks and
drift ~5-10 km/day):

    SST closed-isotherm rings   11% next-day match, 79% single-day
    SSH (this module)           83% next-day match, 11 tracks spanning 14 d

Method is the standard Chelton-style one, on SLA (the mean dynamic topography
is already removed, so the *mean* Gulf Stream is not in the field and only
transient features remain):

    seeds = local SLA extrema
    eddy  = OUTERMOST closed SLA contour still enclosing exactly one seed
    keep  = amplitude, radius and roundness inside physical ring range

Anticyclonic (SLA maximum) = warm core; cyclonic (SLA minimum) = cold core.

Requires: numpy, scipy, xarray, contourpy, matplotlib, copernicusmarine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DEG2KM = 111.195

# Global near-real-time L4 gridded altimetry. The 0.25deg NRT product is
# stale (ended 2024-11); this 0.125deg one tracks to the current day.
SLA_DATASET = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"


@dataclass
class EddyConfig:
    min_amp_m: float = 0.05        # core-to-boundary sea level difference
    min_radius_km: float = 25.0    # smaller than this is below the grid's skill
    max_radius_km: float = 200.0   # larger is a basin feature, not a ring
    min_compactness: float = 0.45  # 4*pi*A/P^2; rings are round, meanders are not
    level_step_m: float = 0.01     # contour interval when growing outward
    peak_footprint: int = 5        # ~70 km at 0.125deg; seed separation
    dedup_km: float = 40.0         # merge cores that yield the same contour
    match_km: float = 60.0         # day-over-day linking distance


_sla_ds = None


def _open_sla(username=None, password=None):
    """Open (once) the CMEMS SLA dataset.

    Cached at module level because opening it costs ~9 s: a 14-day batch that
    re-opened per day would spend minutes doing nothing. The handle is lazy,
    so caching it holds coordinates, not data.
    """
    global _sla_ds
    if _sla_ds is not None:
        return _sla_ds
    import copernicusmarine as cm
    _sla_ds = cm.open_dataset(
        dataset_id=SLA_DATASET,
        username=username or "maristizabalvar",
        password=password or "MariaCMEMS2018",
        chunk_size_limit=0,
    )
    return _sla_ds


def get_sla(extent, time=None, username=None, password=None, pad=2.0):
    """Load CMEMS SLA for `extent` ([lon0, lon1, lat0, lat1]).

    Returns (sla_2d, actual_time, lag_days) where `sla_2d` is an xr.DataArray
    on (latitude, longitude). `lag_days` is how far the returned field is
    from the requested time — NRT altimetry can trail by a day or more, and
    a ring map built from stale SSH should say so.
    """
    ds = _open_sla(username, password)
    da = ds["sla"].sel(
        longitude=slice(extent[0] - pad, extent[1] + pad),
        latitude=slice(extent[2] - pad, extent[3] + pad),
    )
    want = pd.Timestamp(time) if time is not None else pd.Timestamp(da.time.values[-1])
    got = da.sel(time=want, method="nearest")
    actual = pd.Timestamp(got.time.values)
    return got.load(), actual, abs((actual - want).total_seconds()) / 86400.0


def _polygon_metrics(l, ref_lat):
    """(area_km2, radius_km, compactness) for a closed lon/lat ring."""
    x, y = l[:, 0], l[:, 1]
    area_deg = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    area_km2 = area_deg * DEG2KM ** 2 * np.cos(np.radians(ref_lat))
    seg = np.hypot(np.diff(x) * DEG2KM * np.cos(np.radians(ref_lat)),
                   np.diff(y) * DEG2KM)
    perim = seg.sum()
    if perim <= 0 or area_km2 <= 0:
        return 0.0, 0.0, 0.0
    return area_km2, np.sqrt(area_km2 / np.pi), 4 * np.pi * area_km2 / perim ** 2


def detect_eddies(sla, cfg=None, lat_name="latitude", lon_name="longitude"):
    """Detect eddies in a 2-D SLA field.

    Returns a list of dicts: kind ('warm'/'cold'), lon, lat, radius_km,
    amplitude_cm, compactness, and `poly` as an (N, 2) lon/lat boundary.
    """
    from contourpy import contour_generator
    from matplotlib.path import Path as MplPath
    from scipy.ndimage import maximum_filter, minimum_filter

    cfg = cfg or EddyConfig()
    da = sla.sortby(lat_name).sortby(lon_name)
    lats = da[lat_name].values.astype(float)
    lons = da[lon_name].values.astype(float)
    Z = da.transpose(lat_name, lon_name).values.astype(float)

    valid = np.isfinite(Z)
    if valid.sum() < 100:
        return []
    S = np.where(valid, Z, 0.0)
    fp = cfg.peak_footprint
    seeds = []
    for sign, mask in ((+1, (S == maximum_filter(S, fp)) & valid & (S > 0)),
                       (-1, (S == minimum_filter(S, fp)) & valid & (S < 0))):
        for i, j in zip(*np.where(mask)):
            seeds.append(dict(lat=float(lats[i]), lon=float(lons[j]),
                              sign=sign, val=float(S[i, j])))

    cg = contour_generator(x=lons, y=lats, z=np.ma.array(Z, mask=~valid))
    found = []
    for sd in seeds:
        sign, v0 = sd["sign"], sd["val"]
        # Walk outward from the core: levels approach zero, contours grow.
        # Keep the last one that still wraps exactly this seed — once a
        # contour swallows a second core the two eddies have merged and the
        # boundary is no longer this eddy's.
        levels = (np.arange(v0 - cfg.level_step_m, 0, -cfg.level_step_m) if sign > 0
                  else np.arange(v0 + cfg.level_step_m, 0, cfg.level_step_m))
        best = None
        for lev in levels:
            here = None
            for l in cg.lines(float(lev)):
                l = np.asarray(l)
                if len(l) < 8 or not np.allclose(l[0], l[-1]):
                    continue
                if MplPath(l).contains_point((sd["lon"], sd["lat"])):
                    here = l
                    break
            if here is None:
                continue
            path = MplPath(here)
            if sum(1 for s2 in seeds if s2["sign"] == sign
                   and path.contains_point((s2["lon"], s2["lat"]))) != 1:
                break
            best = (here, float(lev))
        if best is None:
            continue
        poly, lev = best
        amp = abs(v0 - lev)
        if amp < cfg.min_amp_m:
            continue
        clat = float(poly[:, 1].mean())
        _, r_km, comp = _polygon_metrics(poly, clat)
        if not (cfg.min_radius_km <= r_km <= cfg.max_radius_km):
            continue
        if comp < cfg.min_compactness:
            continue
        found.append(dict(kind="warm" if sign > 0 else "cold",
                          lon=float(poly[:, 0].mean()), lat=clat,
                          radius_km=float(r_km), amplitude_cm=float(amp * 100),
                          compactness=float(comp), poly=poly))

    # neighbouring cores can grow into the same boundary; keep the strongest
    out = []
    for e in sorted(found, key=lambda z: -z["amplitude_cm"]):
        if any(e["kind"] == u["kind"] and _km(e["lon"], e["lat"], u["lon"], u["lat"])
               < cfg.dedup_km for u in out):
            continue
        out.append(e)
    return out


def _km(lon1, lat1, lon2, lat2):
    return np.hypot((lon1 - lon2) * np.cos(np.radians(lat1)) * DEG2KM,
                    (lat1 - lat2) * DEG2KM)


def match_eddies(prev, curr, cfg=None):
    """Link yesterday's eddies to today's by nearest centre, same sign.

    Returns a list, parallel to `curr`, of the matched index in `prev` (or
    None). Persistence is the single best quality signal available without
    ground truth: a real ring recurs and drifts slowly, an artefact does not.
    """
    cfg = cfg or EddyConfig()
    used, out = set(), []
    for e in curr:
        cands = [(_km(e["lon"], e["lat"], p["lon"], p["lat"]), i)
                 for i, p in enumerate(prev)
                 if i not in used and p["kind"] == e["kind"]]
        hit = None
        if cands:
            d, i = min(cands)
            if d <= cfg.match_km:
                hit = i
                used.add(i)
        out.append(hit)
    return out


def eddies_to_geojson(eddies, path, time=None, source="CMEMS-SLA", extra=None,
                      ages=None):
    """Write eddies as GeoJSON polygons, one Feature each.

    `ages` (optional, parallel to `eddies`) is how many consecutive days each
    eddy has been tracked; it is written as `days_tracked` so downstream can
    filter to persistent rings.
    """
    feats = []
    for k, e in enumerate(eddies):
        props = {
            "time": str(time) if time is not None else None,
            "source": source,
            "feature": f"{e['kind']}_core_ring",
            "kind": e["kind"],
            "radius_km": round(e["radius_km"], 1),
            "amplitude_cm": round(e["amplitude_cm"], 1),
            "compactness": round(e["compactness"], 3),
            "centroid_lon": round(e["lon"], 4),
            "centroid_lat": round(e["lat"], 4),
        }
        if ages is not None:
            props["days_tracked"] = int(ages[k])
        props.update(extra or {})
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[[float(x), float(y)]
                                                    for x, y in e["poly"]]]},
                      "properties": props})
    Path(path).write_text(json.dumps({"type": "FeatureCollection",
                                      "features": feats}))
    return path


def read_eddies(path):
    """Inverse of eddies_to_geojson(); returns the same dicts detect_eddies does."""
    fc = json.loads(Path(path).read_text())
    out = []
    for f in fc["features"]:
        p = f["properties"]
        if not p.get("feature", "").endswith("_core_ring"):
            continue
        out.append(dict(kind=p["kind"], lon=p["centroid_lon"], lat=p["centroid_lat"],
                        radius_km=p["radius_km"], amplitude_cm=p["amplitude_cm"],
                        compactness=p.get("compactness", np.nan),
                        days_tracked=p.get("days_tracked"),
                        poly=np.asarray(f["geometry"]["coordinates"][0])))
    return out


def plot_eddies(ax, eddies, warm_kw=None, cold_kw=None, min_days=None, **kw):
    """Overlay eddy boundaries. `min_days` (with days_tracked present) draws
    short-lived candidates dashed and thin instead of hiding them."""
    ws = {**dict(color="magenta", lw=2, zorder=22), **(warm_kw or {})}
    cs = {**dict(color="cyan", lw=2, zorder=22), **(cold_kw or {})}
    seen = set()
    for e in eddies:
        s = dict(ws if e["kind"] == "warm" else cs)
        d = e.get("days_tracked")
        if min_days is not None and d is not None and d < min_days:
            s.update(ls="--", lw=1.2, alpha=0.75)
            tag = f"{e['kind']} (unconfirmed)"
        else:
            tag = f"{e['kind']} core ring"
        s.update(kw)
        ax.plot(e["poly"][:, 0], e["poly"][:, 1],
                label=None if tag in seen else tag, **s)
        seen.add(tag)
    return ax
