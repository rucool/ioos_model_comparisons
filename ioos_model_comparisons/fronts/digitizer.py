"""
gs_north_wall.py — semi-automatic digitizing of the Gulf Stream north wall
from a gridded SST field (GOES-19, ACSPO, MUR, or a model surface layer).

Two stages:

1. detect_north_wall() — column-wise (per-longitude) search for the strongest
   poleward SST decrease inside a climatological corridor, stitched by a
   Viterbi-style dynamic program with hard jump limits. Robust, but only ever
   one latitude per longitude, so it cannot wrap a necking meander, and it
   drops columns where the gradient dips below its floor.

2. trace_front() — uses stage 1's picks only as CALIBRATION: samples SST at
   them, builds a per-longitude wall temperature T_wall(lon) (the wall cools
   downstream, so no single isotherm works basin-wide), and traces the zero
   contour of SST - T_wall(lon) as a true 2-D curve. That restores continuity
   through weak-gradient stretches, wraps meanders that are about to pinch
   off, and yields closed contours = warm/cold-core ring candidates near the
   front.

Output is GeoJSON (wall line plus ring polygons, one file per timestamp) and
an optional long-format table for stacking many dates.

Requires: numpy, scipy, xarray; trace_front additionally needs contourpy
(already present wherever matplotlib is installed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

DEG2KM = 111.195


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
# Approximate Gulf Stream north wall mean path, used as the default corridor
# center. NOTE: these anchors are hand-set, not loaded from a published
# climatology. West of ~65W they reproduce a hand-validated wall to <1 km;
# east of ~61W they are unverified and the wall itself becomes ambiguous as
# the Stream broadens into the North Atlantic Current. Fit them to your own
# hand-digitized scenes before trusting the eastern end.
GS_CORRIDOR_LON = (-76.0, -75.0, -72.0, -70.0, -67.0, -65.0,
                   -62.0, -60.0, -57.0, -55.0, -52.0, -50.0)
GS_CORRIDOR_LAT = (35.0, 35.3, 36.5, 37.5, 38.3, 38.7,
                   38.9, 39.0, 38.9, 38.8, 38.6, 38.5)


@dataclass
class WallConfig:
    lat_bounds: tuple = (33.0, 44.0)   # meridional search window
    lon_bounds: tuple = (-75.0, -50.0)
    smooth_km: float = 25.0            # along-column Gaussian smoothing
    min_grad: float = 0.015            # degC/km; absolute floor on a candidate
    min_grad_rel: float = 0.25         # drop picks weaker than this x the scene's
                                        # typical front strength (see detect_north_wall)
    max_gap_km: float = 60.0           # cloud gap you're willing to interpolate
    min_valid_frac: float = 0.4        # per-column data coverage floor
    n_candidates: int = 4              # candidate fronts kept per column
    jump_penalty: float = 0.02         # cost per degree of lat displacement
    max_slope: float = 3.0             # cap on |dlat/dlon| between kept picks
    max_jump: float = 0.35             # ABSOLUTE cap (deg lat) on one transition,
                                        # regardless of gap width — see _viterbi
    restart_gap: float = 0.25           # deg lon; only start a new segment after
                                        # a gap this wide (see _viterbi)
    corridor_lon: tuple = GS_CORRIDOR_LON   # corridor center anchors; set both to
    corridor_lat: tuple = GS_CORRIDOR_LAT   # () to disable the corridor entirely
    corridor_half_width: float = 1.5   # deg lat either side of the center
    despike_window: int = 5            # median filter on the final latitudes
    lon_stride: int = 1                # subsample columns for speed

    # --- trace_front() (calibrated-isotherm stage) ---
    trace_smooth_px: float = 2.0       # gaussian smoothing of the SST field
    calib_halfwidth: float = 1.0       # deg lon; window of the running median
                                        # that builds T_wall(lon) from anchors
    calib_min_pts: int = 5             # min anchors per calibration window
    calib_smooth_cols: int = 60        # smooth T_wall(lon) so an anchor
                                        # handover can't print a vertical jump
    anchor_match_deg: float = 0.25     # contour piece counts as the wall if it
    anchor_match_frac: float = 0.05    # passes within match_deg of >= this
                                        # fraction of the anchor points
    detect_rings: bool = False         # OFF by default — see trace_front docstring.
                                        # SST closed-isotherm rings measured 11%
                                        # next-day persistence vs 83% for the
                                        # altimetry detector in eddies.py.
    ring_area: tuple = (0.1, 8.0)      # deg^2; closed-contour size window
    ring_compactness: float = 0.25     # 4*pi*A/P^2 floor — rings are round,
                                        # filament slivers are not
    ring_lat_range: tuple = (-0.3, 3.5)  # deg rel. to wall: warm rings live
                                        # here; cold rings in the mirror image
    support_radius_px: int = 3         # third-check search radius (~5 km): a
                                        # wall vertex is "supported" if 2-D
                                        # gradient magnitude >= min_grad occurs
                                        # within this many pixels of it

    # --- altimetry-eddy coupling (see _eddy_core_penalty) ---
    # DEFAULT OFF, and deliberately so: A/B tested over 5 days each carrying
    # ~20 tracked rings, this changed the wall by exactly 0.00 km every time.
    # Only 10 of 1413 columns had any penalised candidate and the largest
    # penalty reached 0.0018 degC/km against a typical strength of 0.02-0.05.
    # That is consistent with the measurement it was built from rather than a
    # bug: the wall runs TANGENTIALLY along rings (median r/R = 0.79), which
    # is physically right, and almost never through their cores — so a
    # core penalty has nearly nothing to act on. Kept because it is correct
    # and may matter in a season or region where rings interact more
    # strongly; set > 0 to enable. The useful eddy coupling is on the QC
    # side instead — see explain_displacement().
    eddy_core_penalty: float = 0.0     # degC/km of effective strength removed
                                        # at an eddy centre; 0 disables
    eddy_core_frac: float = 0.67       # penalty starts inside this fraction of
                                        # the eddy radius and ramps to the core
    eddy_min_days: int = 2             # only eddies tracked this many days
                                        # inform the wall


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------
def _column_candidates(temp, lats, cfg, dy_km):
    """Return (lat, strength) candidates for one longitude column.

    Strength is the magnitude of the poleward temperature decrease in degC/km.
    """
    valid = np.isfinite(temp)
    if valid.mean() < cfg.min_valid_frac or valid.sum() < 5:
        return np.empty(0), np.empty(0)

    # interpolate across small cloud gaps only
    max_gap = max(1, int(round(cfg.max_gap_km / dy_km)))
    t = _interp_short_gaps(temp, max_gap)

    sigma = max(1.0, cfg.smooth_km / dy_km)
    dlat = float(lats[1] - lats[0])

    lat_all, str_all = [], []
    for s, e in _valid_runs(t, min_len=max(5, int(round(2 * sigma)))):
        seg, seg_lat = t[s:e], lats[s:e]
        ts = gaussian_filter1d(seg, sigma, mode="nearest")

        # dT/dy in degC/km; negative = cooling northward
        strength = -np.gradient(ts, seg_lat) / DEG2KM

        idx, _ = find_peaks(strength, height=cfg.min_grad)
        for i in idx:
            # sub-gridpoint refinement by parabolic fit on the strength curve
            a, b, c = strength[i - 1], strength[i], strength[i + 1]
            denom = a - 2 * b + c
            shift = np.clip(0.5 * (a - c) / denom, -1, 1) if denom != 0 else 0.0
            lat_all.append(seg_lat[i] + shift * dlat)
            str_all.append(strength[i])

    if not lat_all:
        return np.empty(0), np.empty(0)

    lat_all = np.asarray(lat_all)
    str_all = np.asarray(str_all)
    order = np.argsort(str_all)[::-1][: cfg.n_candidates]
    return lat_all[order], str_all[order]


def _interp_short_gaps(y, max_gap):
    y = np.asarray(y, float).copy()
    bad = ~np.isfinite(y)
    if not bad.any() or bad.all():
        return y
    i = np.arange(y.size)
    filled = np.interp(i, i[~bad], y[~bad])
    # re-blank any gap longer than max_gap
    runs = np.flatnonzero(np.diff(np.r_[0, bad.view(np.int8), 0]) != 0).reshape(-1, 2)
    for s, e in runs:
        if e - s > max_gap or s == 0 or e == y.size:
            filled[s:e] = np.nan
    return filled


def _valid_runs(y, min_len=5):
    """Index spans of contiguous finite values, longest first."""
    good = np.isfinite(y)
    if not good.any():
        return []
    runs = np.flatnonzero(np.diff(np.r_[0, good.view(np.int8), 0]) != 0).reshape(-1, 2)
    runs = [(int(s), int(e)) for s, e in runs if e - s >= min_len]
    return sorted(runs, key=lambda r: r[1] - r[0], reverse=True)


def _eddy_core_penalty(lon, lats, eddies, cfg):
    """Cost added to candidates lying inside a tracked eddy's core.

    Measured on a 14-day August 2026 archive: 23% of wall vertices fall
    inside a tracked altimetry eddy, but 65% of those sit in the OUTER third
    (median r/R = 0.79) and only 17% deep inside (r < 0.5R). The wall running
    tangentially along a ring is physically right — rings are shed from
    meanders, so the Stream wraps their outer edge — which is why this is a
    soft penalty and not an exclusion; excluding eddy interiors outright
    would delete a large amount of legitimate wall.

    What is not defensible is the wall cutting through a ring's rotating
    core. Penalty is zero at and outside `eddy_core_frac` of the radius and
    ramps linearly to `eddy_core_penalty` at the centre.
    """
    pen = np.zeros(lats.size)
    if not eddies or cfg.eddy_core_penalty <= 0:
        return pen
    for e in eddies:
        if e.get("days_tracked", 1) < cfg.eddy_min_days:
            continue
        R = e.get("radius_km", 0.0)
        if R <= 0:
            continue
        d = np.hypot((lon - e["lon"]) * np.cos(np.radians(e["lat"])) * DEG2KM,
                     (lats - e["lat"]) * DEG2KM) / R
        inside = d < cfg.eddy_core_frac
        if inside.any():
            depth = 1.0 - d[inside] / cfg.eddy_core_frac      # 0 at rim, 1 at core
            pen[inside] = np.maximum(pen[inside],
                                     cfg.eddy_core_penalty * depth)
    return pen


def _viterbi(cands, lons, jump_penalty, max_slope=None, max_jump=None,
             restart_gap=0.0, penalty=None):
    """Pick one candidate per column minimizing -strength + penalty*|dlat|.

    `jump_penalty` is a *soft* cost, and that is not enough on its own: over a
    long enough run, a stronger-but-wrong feature (a shelf-slope front, or a
    warm-core ring sitting in the Slope Sea just north of the wall) can
    out-earn the one-time cost of jumping onto it.

    Two hard constraints bound that. `max_slope` caps |dlat/dlon|, scaled by
    the longitude gap so a real cloud gap gets proportionally more slack.
    `max_jump` caps the absolute displacement of a single transition and is
    the one that actually matters: a slope-only cap RATCHETS, because its
    allowance (max_slope * dlon) grows without bound across gaps, so a run of
    cloud gaps walks the path north one maximal-but-legal step at a time. On
    a cloudy August scene that reproduced the exact failure the slope cap was
    added to prevent, just in more steps.

    A column whose candidates are all unreachable is skipped, and starts a
    fresh segment only once `restart_gap` degrees of longitude have passed
    without a kept pick. Both halves matter. Restarting immediately lets the
    line teleport: at an adjacent column the "new segment" is unconstrained,
    so it re-enters exactly the jump `max_jump` just forbade (observed as a
    2.15 deg step across 0.018 deg of longitude). Never restarting is also
    wrong — the path would be stuck hunting a link across a gap the physics
    won't bridge. Skipping is safe here only because `max_jump` is absolute:
    the widening lookback can no longer buy a bigger jump.

    Traceback recovers every segment. A single traceback from the last valid
    column silently returns only the final fragment.
    """
    n = len(cands)
    best_cost = [None] * n
    back = [None] * n

    for j in range(n):
        lat_j, str_j = cands[j]
        if lat_j.size == 0:
            continue
        # `penalty` biases the choice only; the strength carried out in the
        # path stays raw, so the reported gradient and the support QC still
        # see the true measurement rather than a cost-adjusted one.
        local = -str_j.astype(float)
        if penalty is not None and penalty[j] is not None:
            local = local + penalty[j]
        prev = next((k for k in range(j - 1, -1, -1) if best_cost[k] is not None), None)
        if prev is not None:
            lat_p, cost_p = cands[prev][0], best_cost[prev]
            dlon = abs(lons[j] - lons[prev])
            # allow proportionally more wander across a wider longitude gap
            pen = jump_penalty / max(dlon, 1e-6) ** 0.5
            dlat = np.abs(lat_j[:, None] - lat_p[None, :])
            total = cost_p[None, :] + pen * dlat
            allow = np.inf
            if max_slope is not None:
                allow = min(allow, max_slope * dlon)
            if max_jump is not None:
                allow = min(allow, max_jump)
            if np.isfinite(allow):
                total = np.where(dlat > allow, np.inf, total)
            pick = np.argmin(total, axis=1)
            cost = local + total[np.arange(lat_j.size), pick]
            reachable = np.isfinite(cost)
            if reachable.any():
                best_cost[j] = np.where(reachable, cost, np.inf)
                back[j] = [(prev, p) if ok else (-1, -1)
                           for p, ok in zip(pick, reachable)]
                continue
            if abs(lons[j] - lons[prev]) < restart_gap:
                continue        # too close to restart — that would teleport
        best_cost[j], back[j] = local, [(-1, -1)] * lat_j.size   # new segment

    # Trace back every segment, not just the chain ending at the last column.
    path = {}
    for j in range(n - 1, -1, -1):
        if best_cost[j] is None or j in path:
            continue
        i = int(np.argmin(best_cost[j]))
        k = j
        while k >= 0 and k not in path:
            path[k] = (float(cands[k][0][i]), float(cands[k][1][i]))
            k, i = back[k][i]
    return path


def detect_north_wall(sst, cfg=None, lat_name="lat", lon_name="lon", eddies=None):
    """Digitize the north wall from a 2-D SST DataArray.

    Parameters
    ----------
    sst : xr.DataArray with dims (lat, lon), degrees C, NaN where masked
    cfg : WallConfig

    Returns
    -------
    xr.Dataset with lon, lat(lon), grad(lon), and a `corridor` coord giving
    the corridor center latitude used at each longitude (NaN if disabled).
    """
    cfg = cfg or WallConfig()

    da = sst.sortby(lat_name).sortby(lon_name)
    da = da.sel({lat_name: slice(*cfg.lat_bounds), lon_name: slice(*cfg.lon_bounds)})
    if cfg.lon_stride > 1:
        da = da.isel({lon_name: slice(None, None, cfg.lon_stride)})

    lats = da[lat_name].values.astype(float)
    lons = da[lon_name].values.astype(float)
    dy_km = float(np.abs(np.median(np.diff(lats)))) * DEG2KM
    field = da.transpose(lat_name, lon_name).values.astype(float)

    cands = [_column_candidates(field[:, j], lats, cfg, dy_km) for j in range(lons.size)]

    # Scene-wide reference strength, computed from the per-column BEST
    # candidate before any path is chosen. The QC below used the median of
    # the selected path, which is circular: when the path locked onto strong
    # wrong features (a ring edge, the shelf-slope front) the median rose and
    # the QC then blanked the correct, weaker picks — the opposite of intent.
    per_col_best = [st.max() for _, st in cands if st.size]
    ref_grad = float(np.median(per_col_best)) if per_col_best else np.nan

    # Corridor: restrict candidates to +/- half_width of a per-longitude
    # center line. This is what keeps the path off warm-core rings sitting in
    # the Slope Sea just north of the wall, which are close enough and strong
    # enough that no purely local constraint excludes them.
    center = np.full(lons.size, np.nan)
    if cfg.corridor_lon and cfg.corridor_lat and cfg.corridor_half_width > 0:
        center = np.interp(lons, np.asarray(cfg.corridor_lon, float),
                            np.asarray(cfg.corridor_lat, float))
        cands = [
            (la[k], st[k]) if la.size else (la, st)
            for la, st, k in (
                (la, st, np.abs(la - c) <= cfg.corridor_half_width if la.size else None)
                for (la, st), c in zip(cands, center)
            )
        ]

    # discourage picks inside a tracked eddy's rotating core (altimetry —
    # an independent sensor, so this is real information rather than another
    # constraint invented from the same SST field)
    pen = None
    if eddies and cfg.eddy_core_penalty > 0:
        pen = [(_eddy_core_penalty(lons[j], cands[j][0], eddies, cfg)
                if cands[j][0].size else None) for j in range(lons.size)]

    path = _viterbi(cands, lons, cfg.jump_penalty, cfg.max_slope, cfg.max_jump,
                    cfg.restart_gap, pen)

    lat_out = np.full(lons.size, np.nan)
    grad_out = np.full(lons.size, np.nan)
    for j, (la, st) in path.items():
        lat_out[j], grad_out[j] = la, st

    # relative-strength QC against the scene reference (not the chosen line)
    if np.isfinite(ref_grad) and cfg.min_grad_rel > 0:
        weak = grad_out < cfg.min_grad_rel * ref_grad
        lat_out[weak] = np.nan
        grad_out[weak] = np.nan

    if cfg.despike_window > 1 and np.isfinite(lat_out).sum() > cfg.despike_window:
        filled = _interp_short_gaps(lat_out, lat_out.size)
        smoothed = median_filter(filled, size=cfg.despike_window, mode="nearest")
        lat_out = np.where(np.isfinite(lat_out), smoothed, np.nan)

    return xr.Dataset(
        {
            "lat": (lon_name, lat_out),
            "grad": (lon_name, grad_out),
        },
        coords={lon_name: lons, "corridor": (lon_name, center)},
        attrs={"method": "column_max_gradient+viterbi+corridor",
               "ref_grad_c_per_km": ref_grad, **asdict(cfg)},
    )


# ----------------------------------------------------------------------------
# cloud handling
# ----------------------------------------------------------------------------
def persistence_fill(stack, time_name="time", dilate_px=2):
    """Composite a (time, lat, lon) stack into one cloud-filled scene.

    The last timestep is the target; its holes are filled from progressively
    older scenes (newest first). Before use, every scene's NaN mask is grown
    by `dilate_px` pixels: cloud masks err permissive and the residual pixels
    at mask edges are cold-biased, which reads as a false front to both the
    gradient detector and the isotherm trace — on a raw scene those residues
    produce spurious "cold core rings" in the speckle south of the Stream.
    Dilation alone shrinks coverage (that's the trade), so it should always
    be paired with the fill that wins the coverage back.

    Returns (filled, age): 2-D DataArrays on the target's coords, `age` in
    days since the data at each pixel was actually observed (0 = target
    scene, NaN = never observed in the window). Callers should keep `age`
    visible downstream — the front moves ~10-20 km/day, so old fill is least
    trustworthy exactly at an actively pinching meander.
    """
    from scipy.ndimage import binary_dilation

    times = stack[time_name].values
    target = stack.isel({time_name: -1})

    def _dilated(arr):
        if dilate_px <= 0:
            return arr
        return np.where(binary_dilation(~np.isfinite(arr), iterations=dilate_px),
                        np.nan, arr)

    filled = _dilated(target.values.astype(float))
    age = np.where(np.isfinite(filled), 0.0, np.nan)
    for k in range(stack.sizes[time_name] - 2, -1, -1):
        older = _dilated(stack.isel({time_name: k}).values.astype(float))
        hole = ~np.isfinite(filled) & np.isfinite(older)
        filled[hole] = older[hole]
        age[hole] = float((times[-1] - times[k]) / np.timedelta64(1, "D"))

    dims = [d for d in target.dims if d != time_name]
    coords = {d: target[d] for d in dims}
    return (xr.DataArray(filled, dims=dims, coords=coords),
            xr.DataArray(age, dims=dims, coords=coords))


# ----------------------------------------------------------------------------
# calibrated-isotherm front tracing
# ----------------------------------------------------------------------------
def _nan_smooth(T, sigma):
    """Gaussian smooth ignoring NaN; blank cells without enough valid support."""
    from scipy.ndimage import gaussian_filter
    V = np.where(np.isfinite(T), T, 0.0)
    W = np.isfinite(T).astype(float)
    Vs = gaussian_filter(V, sigma)
    Ws = gaussian_filter(W, sigma)
    out = Vs / np.maximum(Ws, 1e-9)
    out[Ws < 0.3] = np.nan
    return out


def _shoelace_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@dataclass
class FrontTrace:
    """Result of trace_front(): the wall as open 2-D polylines plus ring
    candidates as closed polygons, all as (N, 2) lon/lat arrays."""
    wall: list                 # open contour pieces matched to the anchors
    warm_rings: list           # closed, warm interior, north of the wall
    cold_rings: list           # closed, cold interior, south of the wall
    anchors: xr.Dataset        # the detect_north_wall() output used to calibrate
    t_wall_lon: np.ndarray     # calibrated wall temperature per longitude
    lons: np.ndarray
    support: list = None       # per wall piece: bool per vertex — independent
                               # 2-D gradient magnitude confirms a front there
    fill_age: list = None      # per wall piece: days per vertex since the pixel
                               # under it was actually observed (0 = fresh)

    def lon_coverage(self):
        """Fraction of grid longitudes crossed by at least one wall piece."""
        if not self.wall:
            return 0.0
        edges = np.linspace(self.lons[0], self.lons[-1], self.lons.size + 1)
        covered = set()
        for l in self.wall:
            covered.update(np.unique(np.digitize(l[:, 0], edges)))
        return len(covered) / self.lons.size

    def support_frac(self):
        """Fraction of wall vertices confirmed by the independent gradient
        check. This is the third check: the corridor decides WHICH front,
        the isotherm gives continuity, and this verifies a front physically
        exists at each vertex — with no knowledge of the other two."""
        if not self.support:
            return float("nan")
        n = sum(s.size for s in self.support)
        return sum(int(s.sum()) for s in self.support) / n if n else float("nan")

    def wall_fill_stats(self):
        """How much of the wall rests on persistence-filled (stale) pixels.

        Returns dict(frac, mean_age_days, max_age_days) or None if the trace
        was run without an age field. Fill is least trustworthy exactly where
        a meander is actively moving, so a high frac deserves suspicion even
        when the gradient support check passes — yesterday's front has real
        gradients too, just possibly in yesterday's position.
        """
        if not self.fill_age:
            return None
        a = np.concatenate(self.fill_age)
        on = a > 0
        return {
            "frac": float(on.mean()) if a.size else float("nan"),
            "mean_age_days": float(a[on].mean()) if on.any() else 0.0,
            "max_age_days": float(a[on].max()) if on.any() else 0.0,
        }


def trace_front(sst, cfg=None, lat_name="lat", lon_name="lon", age=None,
                eddies=None):
    """Trace the north wall as a 2-D curve and collect ring candidates.

    Runs detect_north_wall() for anchors, calibrates T_wall(lon) from SST
    sampled at those anchors (running median over `calib_halfwidth`, then
    smoothed — the wall cools downstream, which is why a single global
    isotherm dives into the Sargasso east of ~65W), and traces the zero
    contour of SST - T_wall(lon). Open contour pieces passing near enough of
    the anchors are the wall; closed pieces near the front, big enough and
    round enough (`ring_compactness` culls filament slivers), are ring
    candidates, split warm/cold by the anomaly at their centroid.

    Ring detection here is DISABLED by default (`cfg.detect_rings`). It was
    measured against next-day persistence over a 14-day August 2026 archive
    and found to be mostly noise: 79% of detections never recurred, median
    diameter 56 km against a real ring's 100-300 km, and more cold rings
    than warm — backwards, since summer heating caps a cold core under a
    warm mixed layer and hides it from SST entirely. Use `eddies.py`
    (altimetry, 83% next-day persistence) for rings; set detect_rings=True
    only to reproduce the old behaviour.

    Caveats: the calibration inherits whatever the anchor detector did that
    day, so day-over-day QC on the anchors covers this stage too.
    """
    from contourpy import contour_generator
    from scipy.spatial import cKDTree

    cfg = cfg or WallConfig()
    anchors = detect_north_wall(sst, cfg, lat_name=lat_name, lon_name=lon_name,
                                eddies=eddies)
    alon = anchors[lon_name].values
    alat = anchors["lat"].values
    fin = np.isfinite(alat)

    da = sst.sortby(lat_name).sortby(lon_name)
    lats = da[lat_name].values.astype(float)
    lons = da[lon_name].values.astype(float)
    T = da.transpose(lat_name, lon_name).values.astype(float)
    Ts = _nan_smooth(T, cfg.trace_smooth_px)

    li = np.clip(np.searchsorted(lats, alat[fin]), 0, lats.size - 1)
    ji = np.clip(np.searchsorted(lons, alon[fin]), 0, lons.size - 1)
    t_at = Ts[li, ji]
    t_wall_lon = np.full(lons.size, np.nan)
    for j in range(lons.size):
        w = np.abs(alon[fin] - lons[j]) <= cfg.calib_halfwidth
        if w.sum() >= cfg.calib_min_pts:
            t_wall_lon[j] = np.nanmedian(t_at[w])
    ok = np.isfinite(t_wall_lon)
    if not ok.any():
        return FrontTrace([], [], [], anchors, t_wall_lon, lons, [])
    t_wall_lon = np.interp(lons, lons[ok], t_wall_lon[ok])
    t_wall_lon = gaussian_filter1d(t_wall_lon, cfg.calib_smooth_cols)

    A = Ts - t_wall_lon[None, :]
    cg = contour_generator(x=lons, y=lats, z=np.ma.array(A, mask=~np.isfinite(A)))
    lines = [np.asarray(l) for l in cg.lines(0.0) if len(l) >= 10]

    apts = np.c_[alon[fin], alat[fin]]
    wall, warm, cold = [], [], []
    lo, hi = cfg.ring_lat_range
    for l in lines:
        if not np.allclose(l[0], l[-1]):
            d, _ = cKDTree(l).query(apts)
            if (d < cfg.anchor_match_deg).mean() < cfg.anchor_match_frac:
                continue
            # A contour piece that spends most of its length inside a tracked
            # eddy core is wrapping the ring rather than tracing the wall —
            # the 2026-08-16 excursion at 62.5W was exactly this. Tangential
            # contact is normal and stays (see _eddy_core_penalty), so this
            # only rejects a piece dominated by core interior.
            if eddies and cfg.eddy_core_penalty > 0 and len(l):
                core = np.zeros(len(l), bool)
                for e in eddies:
                    if e.get("days_tracked", 1) < cfg.eddy_min_days:
                        continue
                    R = e.get("radius_km", 0.0)
                    if R <= 0:
                        continue
                    dd = np.hypot((l[:, 0] - e["lon"]) * np.cos(np.radians(e["lat"])) * DEG2KM,
                                  (l[:, 1] - e["lat"]) * DEG2KM) / R
                    core |= dd < 0.5 * cfg.eddy_core_frac
                if core.mean() > 0.5:
                    continue
            wall.append(l)
            continue
        if not cfg.detect_rings:
            continue
        area = _shoelace_area(l)
        if not (cfg.ring_area[0] <= area <= cfg.ring_area[1]):
            continue
        perim = np.sum(np.hypot(*np.diff(l, axis=0).T))
        if 4 * np.pi * area / perim ** 2 < cfg.ring_compactness:
            continue
        cx, cy = l[:, 0].mean(), l[:, 1].mean()
        w_at = np.interp(cx, alon[fin], alat[fin])
        ci = np.clip(np.searchsorted(lats, cy), 0, lats.size - 1)
        cj = np.clip(np.searchsorted(lons, cx), 0, lons.size - 1)
        a_c = A[ci, cj]
        if not np.isfinite(a_c):
            continue
        if lo <= cy - w_at <= hi and a_c > 0:
            warm.append(l)
        elif -hi <= cy - w_at <= -lo and a_c < 0:
            cold.append(l)

    # Third check, independent of corridor and calibration: 2-D gradient
    # magnitude of the field, sampled near each wall vertex. Flags spans
    # drawn through featureless water (stale fill, weak front). It verifies
    # a front EXISTS there, not that it is the Gulf Stream.
    from scipy.ndimage import maximum_filter
    dlat = float(np.median(np.diff(lats)))
    dlon = float(np.median(np.diff(lons)))
    gy, gx = np.gradient(Ts)
    dx_km = dlon * DEG2KM * np.cos(np.deg2rad(lats))[:, None]
    G = np.hypot(gy / (dlat * DEG2KM), gx / dx_km)
    Gmax = maximum_filter(np.where(np.isfinite(G), G, 0.0),
                          size=2 * cfg.support_radius_px + 1)
    support = []
    for l in wall:
        li = np.clip(np.searchsorted(lats, l[:, 1]), 0, lats.size - 1)
        ji = np.clip(np.searchsorted(lons, l[:, 0]), 0, lons.size - 1)
        support.append(Gmax[li, ji] >= cfg.min_grad)

    # per-vertex data age, if the caller composited with persistence_fill()
    fill_age = None
    if age is not None:
        ga = (age.sortby(lat_name).sortby(lon_name)
                 .transpose(lat_name, lon_name).values.astype(float)
              if isinstance(age, xr.DataArray) else np.asarray(age, float))
        fill_age = []
        for l in wall:
            li = np.clip(np.searchsorted(lats, l[:, 1]), 0, lats.size - 1)
            ji = np.clip(np.searchsorted(lons, l[:, 0]), 0, lons.size - 1)
            fill_age.append(ga[li, ji])

    return FrontTrace(wall, warm, cold, anchors, t_wall_lon, lons, support,
                      fill_age)


def wall_displacement_km(prev_wall, curr_wall, ref_lat=38.0, step_deg=0.02,
                         window_km=50.0, west_of=-68.5):
    """Symmetric nearest-point (Chamfer) distance between two wall traces, km.

    This is a fourth, independent check: the corridor picks WHICH front, the
    isotherm gives continuity, gradient support confirms a front physically
    exists — none of those catch the wall settling on a DIFFERENT front than
    yesterday while still passing all three, e.g. by wrapping a ring instead
    of passing it by. Day-over-day displacement catches that, because the
    Gulf Stream itself only moves ~10-20 km/day.

    Deliberately NOT longitude interpolation: trace_front() can wrap a
    meander, so a wall piece legitimately visits the same longitude more
    than once (72 direction changes measured on one real piece), and
    np.interp on non-monotonic x silently returns garbage instead of
    raising — an earlier version of this function did exactly that and
    reported a bogus ~620 km "jump" from it. Resampling each piece at fixed
    arc-length spacing and taking nearest-neighbor distance is shape-correct
    regardless of wrapping.

    `ref_lat` sets the longitude->km scaling (cos(lat)); one value is enough
    for a metric this coarse across a ~33-44N domain.  Returns None if
    either side has no wall pieces.

    Besides median_km/max_km, returns `local_km`: the localized-derailment
    metric. The median dilutes a localized derailment across a long stable
    line — a visually confirmed ring-wrap on 2026-08-16 read 8.9 km median
    (137.8 max) and passed a median gate — while the raw max is too noisy to
    gate on (edge pieces come and go between days). `local_km` is the max,
    over every `window_km`-long along-front window of each wall, of the
    windowed MINIMUM distance to the other day's wall: a window only scores
    x if the entire window sits >= x km away, so isolated spikes can't fire
    it and a wrapped ring can't hide in the median. Windows are only scored
    where BOTH days cover the longitude (otherwise honest new coverage after
    a cloudy day reads as a jump), and both directions are scored (today far
    from yesterday = wrap/excursion; yesterday far from today = a feature
    the new trace dropped).

    `local_west_km` restricts that to window centers west of `west_of`, and
    is the only localized value worth GATING on. A 14-day archive
    (2026-08-03..16) showed local_km is a continuum of 45-270 km/day with no
    good/bad separation: in the 68.5-58W meander/ring sector, REAL evolution
    (a meander pinching off a ring during that fortnight) produces the same
    90-270 km localized signatures as a trace derailment, so day-pair
    displacement fundamentally cannot separate them there — even the
    confirmed 08-16 ring-wrap is indistinguishable by this metric alone.
    West of 68.5W the same fortnight never exceeded 36 km, so a modest
    threshold there is meaningful. For the meander sector, `worst_lon`/
    `worst_lat`/`local_km` say where a human should look.
    """
    from scipy.ndimage import minimum_filter1d
    from scipy.spatial import cKDTree

    cosr = np.cos(np.radians(ref_lat))

    def to_km(p):
        return np.c_[p[:, 0] * cosr, p[:, 1]] * DEG2KM

    def resample(pieces):
        out = []
        for l in pieces:
            xy = to_km(l)
            seg = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            if seg[-1] == 0:
                continue
            n = max(2, int(seg[-1] / (step_deg * DEG2KM)))
            s = np.linspace(0, seg[-1], n)
            out.append(np.c_[np.interp(s, seg, l[:, 0]), np.interp(s, seg, l[:, 1])])
        return out

    def lon_intervals(pieces, margin=0.25):
        """Merged [min, max] longitude spans of a wall, minus a margin so
        coverage-boundary windows don't score."""
        iv = sorted((l[:, 0].min() + margin, l[:, 0].max() - margin) for l in pieces)
        merged = []
        for a0, b0 in iv:
            if a0 >= b0:
                continue
            if merged and a0 <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b0))
            else:
                merged.append((a0, b0))
        return merged

    pa, pb = resample(prev_wall), resample(curr_wall)
    if not pa or not pb:
        return None
    ta = cKDTree(to_km(np.concatenate(pa)))
    tb = cKDTree(to_km(np.concatenate(pb)))
    d_ab, _ = ta.query(to_km(np.concatenate(pb)))
    d_ba, _ = tb.query(to_km(np.concatenate(pa)))
    d = np.concatenate([d_ab, d_ba])

    n_w = max(2, int(round(window_km / (step_deg * DEG2KM))))

    def local_one_way(pieces_rs, tree_other, cov):
        """(lon, lat, windowed-min km) for every valid window center."""
        rows = []
        for P in pieces_rs:
            if P.shape[0] < n_w:
                continue
            dd, _ = tree_other.query(to_km(P))
            inside = np.zeros(P.shape[0], bool)
            for a0, b0 in cov:
                inside |= (P[:, 0] >= a0) & (P[:, 0] <= b0)
            dd = np.where(inside, dd, -np.inf)
            # windowed min; -inf at borders/out-of-coverage voids any window
            # that isn't fully inside mutual coverage
            wm = minimum_filter1d(dd, size=n_w, mode="constant", cval=-np.inf)
            ok = np.isfinite(wm)
            if ok.any():
                rows.append(np.c_[P[ok, 0], P[ok, 1], wm[ok]])
        return rows

    rows = (local_one_way(pb, ta, lon_intervals(prev_wall))
            + local_one_way(pa, tb, lon_intervals(curr_wall)))
    local = local_west = 0.0
    worst_lon = worst_lat = None
    if rows:
        W = np.concatenate(rows)
        i = int(np.argmax(W[:, 2]))
        local = float(W[i, 2])
        worst_lon, worst_lat = float(W[i, 0]), float(W[i, 1])
        west = W[W[:, 0] < west_of]
        if west.size:
            local_west = float(west[:, 2].max())

    return {"n": int(d.size), "median_km": float(np.median(d)),
            "max_km": float(d.max()), "local_km": local,
            "local_west_km": local_west, "west_of": float(west_of),
            "worst_lon": worst_lon, "worst_lat": worst_lat,
            "window_km": float(window_km)}


def explain_displacement(disp, eddies, min_days=2, near_km=50.0):
    """Is a localized wall displacement explained by a tracked eddy?

    This closes a gap that displacement alone could not. Localized
    displacement east of ~68.5W was measured at 45-270 km/day on every
    archive day — a continuum with no good/bad separation — because a
    meander genuinely pinching off a ring moves the front as far as a
    derailment does. Distance to yesterday's line cannot tell them apart.

    Altimetry can, because it is an independent sensor: on the same 14-day
    archive, 9 of 13 displacement hotspots sat within 25 km of a tracked
    eddy edge (median 0 km), most on eddies tracked 7-13 days. A hotspot
    coinciding with a long-lived ring is the ocean moving; one with no eddy
    anywhere near it is the trace losing the front.

    Returns None if there is nothing to explain, else a dict with the
    nearest eddy, its edge distance, and `explained`.
    """
    if not disp or disp.get("worst_lon") is None or not eddies:
        return None
    best = None
    for e in eddies:
        if e.get("days_tracked", 1) < min_days:
            continue
        R = e.get("radius_km", 0.0)
        c = np.hypot((disp["worst_lon"] - e["lon"]) * np.cos(np.radians(e["lat"])) * DEG2KM,
                     (disp["worst_lat"] - e["lat"]) * DEG2KM)
        edge = max(0.0, c - R)
        if best is None or edge < best[0]:
            best = (edge, e)
    if best is None:
        return None
    edge, e = best
    return {"explained": bool(edge <= near_km), "edge_km": float(edge),
            "kind": e["kind"], "radius_km": e.get("radius_km"),
            "days_tracked": e.get("days_tracked"),
            "lon": e["lon"], "lat": e["lat"]}


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def to_geojson(wall, path, time=None, source=None, lon_name="lon", extra=None):
    """Write the wall as a GeoJSON LineString (split into segments at gaps)."""
    lons = wall[lon_name].values
    lats = wall["lat"].values
    good = np.isfinite(lats)
    if good.sum() < 2:
        raise ValueError("nothing to write — no valid wall points")

    runs = np.flatnonzero(np.diff(np.r_[0, good.view(np.int8), 0]) != 0).reshape(-1, 2)
    coords = [
        [[float(x), float(y)] for x, y in zip(lons[s:e], lats[s:e])]
        for s, e in runs
        if e - s >= 2
    ]
    geom = (
        {"type": "LineString", "coordinates": coords[0]}
        if len(coords) == 1
        else {"type": "MultiLineString", "coordinates": coords}
    )

    props = {
        "time": str(time) if time is not None else None,
        "source": source,
        "feature": "gulf_stream_north_wall",
        "mean_grad_c_per_km": float(np.nanmean(wall["grad"].values)),
        "n_points": int(good.sum()),
    }
    props.update(extra or {})

    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": geom, "properties": props}],
    }
    Path(path).write_text(json.dumps(fc))
    return path


def append_table(wall, path, time, source=None, lon_name="lon"):
    """Append to a long-format CSV archive: time, lon, lat, grad, source.

    Easier than a pile of GeoJSONs once you have hundreds of timestamps.
    """
    import pandas as pd

    df = wall.to_dataframe().reset_index().rename(columns={lon_name: "lon"})
    df = df.dropna(subset=["lat"])
    df.insert(0, "time", np.datetime64(time))
    df["source"] = source
    p = Path(path)
    df.to_csv(p, mode="a", header=not p.exists(), index=False)
    return path


def read_geojson(path):
    """Return a list of (lon, lat) arrays plus the properties dict."""
    fc = json.loads(Path(path).read_text())
    feat = fc["features"][0]
    g = feat["geometry"]
    parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
    return [np.asarray(p).T for p in parts], feat["properties"]


def plot_wall(ax, path_or_wall, lon_name="lon", **kw):
    """Overlay a saved wall on any matplotlib/cartopy axis.

    Pass transform=ccrs.PlateCarree() in kw for a cartopy GeoAxes.
    """
    style = dict(color="k", lw=2, zorder=20)
    style.update(kw)
    if isinstance(path_or_wall, (str, Path)):
        parts, props = read_geojson(path_or_wall)
        label = style.pop("label", props.get("time"))
        for i, (x, y) in enumerate(parts):
            ax.plot(x, y, label=label if i == 0 else None, **style)
    else:
        w = path_or_wall
        ax.plot(w[lon_name].values, w["lat"].values, **style)
    return ax


def front_to_geojson(trace, path, time=None, source=None, extra=None):
    """Write a FrontTrace: wall as (Multi)LineString + one Polygon per ring."""
    if not trace.wall:
        raise ValueError("nothing to write — no wall pieces in trace")

    def _coords(l):
        return [[float(x), float(y)] for x, y in l]

    wall_geom = (
        {"type": "LineString", "coordinates": _coords(trace.wall[0])}
        if len(trace.wall) == 1
        else {"type": "MultiLineString", "coordinates": [_coords(l) for l in trace.wall]}
    )
    props = {
        "time": str(time) if time is not None else None,
        "source": source,
        "feature": "gulf_stream_north_wall",
        "method": "calibrated_isotherm",
        "t_wall_min_c": float(np.nanmin(trace.t_wall_lon)),
        "t_wall_max_c": float(np.nanmax(trace.t_wall_lon)),
        "lon_coverage": round(trace.lon_coverage(), 3),
        "n_pieces": len(trace.wall),
        "support_frac": (round(trace.support_frac(), 3)
                          if trace.support else None),
    }
    fill_stats = trace.wall_fill_stats()
    if fill_stats is not None:
        props.update(
            wall_fill_frac=round(fill_stats["frac"], 3),
            wall_mean_fill_age_days=round(fill_stats["mean_age_days"], 2),
            wall_max_fill_age_days=round(fill_stats["max_age_days"], 2),
        )
    props.update(extra or {})
    features = [{"type": "Feature", "geometry": wall_geom, "properties": props}]

    for kind, rings in [("warm_core_ring", trace.warm_rings),
                        ("cold_core_ring", trace.cold_rings)]:
        for l in rings:
            area = _shoelace_area(l)
            perim = np.sum(np.hypot(*np.diff(l, axis=0).T))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [_coords(l)]},
                "properties": {
                    "time": str(time) if time is not None else None,
                    "source": source,
                    "feature": kind,
                    "area_deg2": round(float(area), 4),
                    "compactness": round(float(4 * np.pi * area / perim ** 2), 3),
                    "centroid_lon": round(float(l[:, 0].mean()), 4),
                    "centroid_lat": round(float(l[:, 1].mean()), 4),
                },
            })

    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def read_front(path):
    """Read a front_to_geojson() file.

    Returns (parts, props) where parts maps 'wall' / 'warm_core_ring' /
    'cold_core_ring' to lists of (N, 2) lon/lat arrays, and props is the wall
    feature's properties dict.
    """
    fc = json.loads(Path(path).read_text())
    parts = {"wall": [], "warm_core_ring": [], "cold_core_ring": []}
    props = {}
    for feat in fc["features"]:
        kind = feat["properties"].get("feature")
        g = feat["geometry"]
        if kind == "gulf_stream_north_wall":
            props = feat["properties"]
            coords = ([g["coordinates"]] if g["type"] == "LineString"
                      else g["coordinates"])
            parts["wall"] += [np.asarray(c) for c in coords]
        elif kind in parts:
            parts[kind].append(np.asarray(g["coordinates"][0]))
    return parts, props


def plot_front(ax, path_or_trace, wall_kw=None, warm_kw=None, cold_kw=None, **kw):
    """Overlay a FrontTrace (or saved front GeoJSON) on a matplotlib/cartopy
    axis. Pass transform=ccrs.PlateCarree() in kw for a cartopy GeoAxes; kw
    is applied to every element, the per-element dicts override."""
    if isinstance(path_or_trace, (str, Path)):
        parts, _ = read_front(path_or_trace)
        groups = [(parts["wall"], "wall"), (parts["warm_core_ring"], "warm ring"),
                  (parts["cold_core_ring"], "cold ring")]
    else:
        t = path_or_trace
        groups = [(t.wall, "wall"), (t.warm_rings, "warm ring"),
                  (t.cold_rings, "cold ring")]
    styles = [{**dict(color="k", lw=2.5, zorder=20), **(wall_kw or {})},
              {**dict(color="magenta", lw=2, zorder=21), **(warm_kw or {})},
              {**dict(color="cyan", lw=2, zorder=21), **(cold_kw or {})}]
    for (lines, label), style in zip(groups, styles):
        s = {**kw, **style}
        label = s.pop("label", label)
        for i, l in enumerate(lines):
            ax.plot(l[:, 0], l[:, 1], label=label if i == 0 else None, **s)
    return ax