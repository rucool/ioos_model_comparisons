#!/usr/bin/env python3
"""
Step 2 of the glider-coverage analysis: decorrelation scales.

Reads <workdir>/<region>/<model>/fields*.nc and mask.nc (written by download.py) and, for
each field (sst, t100, t200, heat, and ssh when present), separately for shelf
(<200 m) and deep (>200 m) water inside the region (gc_regions.py):

- anomalies = field minus each point's mean + annual + semiannual harmonics
- spatial scale: median correlation of anomaly time series between ~2500
  random point pairs, binned by distance (1/e, rho^2=0.5 and zero crossings)
- temporal scale: lagged autocorrelation at each point (1/e crossing)
- a mapped local 1/e radius for the heat proxy (reference points every 0.5 deg)
- areas: total, shelf, >200 m, >1000 m

Writes scales.json and masks.npz to <workdir>/<region>/<model>/ and prints a summary.

Usage:
    python scripts/tools/glider_coverage/scales.py --model cmems [--region caribbean_pr] [--workdir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from gc_regions import REGIONS
from ioos_model_comparisons import configs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", choices=["cmems", "rtofs", "espc"], required=True)
ap.add_argument("--region", choices=list(REGIONS), default="gulf")
ap.add_argument("--workdir", type=Path, default=configs.path_data / "glider_coverage")
args = ap.parse_args()
W = args.workdir / args.region / args.model

rng = np.random.default_rng(0)
R_E = 6371.0

# fields.nc (temperature) plus fields_salt.nc / fields_uv.nc when they have been downloaded
F = xr.merge([xr.open_dataset(f) for f in sorted(W.glob("fields*.nc"))], combine_attrs="override")
M = xr.open_dataset(W / "mask.nc")
lon, lat = np.meshgrid(F.longitude.values, F.latitude.values)

# --- masks and areas ------------------------------------------------------
inside = REGIONS[args.region]["inside"](lon.ravel(), lat.ravel()).reshape(lon.shape)
wet = inside & M.wet.values
deep200 = inside & M.deep200.values
deep1000 = inside & M.deep1000.values
shelf = wet & ~deep200
dx = 1 / 12 * np.pi / 180 * R_E
cell = dx * dx * np.cos(np.deg2rad(lat))
areas = {k: float(cell[m].sum()) for k, m in
         [("total", wet), ("shelf_lt200", shelf), ("deep_gt200", deep200), ("deep_gt1000", deep1000)]}
print(F.attrs.get("model"), "areas km2:", {k: f"{v:,.0f}" for k, v in areas.items()})

LABELS = {
    "sst": "Sea surface temperature",
    "t100": f"Temperature at {F.attrs.get('t100_depth', '~100 m')}",
    "t200": f"Temperature at {F.attrs.get('t200_depth', '~200 m')}",
    "heat": f"Mean temperature {F.attrs.get('heat_depth', '0-200 m')} (heat proxy)",
    "ssh": "Sea surface height",
    "sss": "Sea surface salinity",
    "s100": f"Salinity at {F.attrs.get('s100_depth', '~100 m')}",
    "s200": f"Salinity at {F.attrs.get('s200_depth', '~200 m')}",
    "salt": f"Mean salinity {F.attrs.get('salt_depth', '0-200 m')}",
    **{f"{c}{H}": f"{'Eastward' if c == 'u' else 'Northward'} current, mean 0-{H} m" for H in (200, 500, 1000) for c in "uv"},
}
# Fields that need water deeper than ~200 m are not computed for the shelf.
DEEP_ONLY = {"t200", "heat", "s200", "salt", "u200", "v200", "u500", "v500", "u1000", "v1000"}
fields = {k: F[k] for k in LABELS if k in F}

tt = (F.time.values - F.time.values[0]) / np.timedelta64(1, "D")
w = 2 * np.pi * tt / 365.25
H = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])


def anomalies(da, mask):
    X = da.values.reshape(da.shape[0], -1)[:, mask.ravel()]
    good = np.all(np.isfinite(X), axis=0)
    X = X[:, good]
    beta, *_ = np.linalg.lstsq(H, X, rcond=None)
    A = X - H @ beta
    idx = np.flatnonzero(mask.ravel())[good]
    return A, idx


def hav(lo1, la1, lo2, la2):
    lo1, la1, lo2, la2 = map(np.deg2rad, (lo1, la1, lo2, la2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_E * np.arcsin(np.sqrt(a))


bins = np.arange(0, 805, 10)
centers = 0.5 * (bins[1:] + bins[:-1])


def spatial_curve(A, idx, n=2500):
    sel = rng.choice(A.shape[1], size=min(n, A.shape[1]), replace=False)
    S = A[:, sel]
    S = (S - S.mean(0)) / S.std(0)
    C = S.T @ S / S.shape[0]
    lo, la = lon.ravel()[idx[sel]], lat.ravel()[idx[sel]]
    D = hav(lo[:, None], la[:, None], lo[None], la[None])
    iu = np.triu_indices(len(sel), 1)
    d, c = D[iu], C[iu]
    k = np.digitize(d, bins) - 1
    ok = (k >= 0) & (k < len(centers))
    curve = np.array([np.median(c[ok & (k == i)]) if np.any(ok & (k == i)) else np.nan
                      for i in range(len(centers))])
    return np.concatenate([[1.0], curve]), np.concatenate([[0.0], centers])


def crossing(x, y, thr):
    """First x where y drops below thr (linear interp)."""
    for i in range(1, len(y)):
        if y[i] < thr:
            return float(x[i - 1] + (thr - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return np.nan


def temporal_curve(A, maxlag=90):
    S = (A - A.mean(0)) / A.std(0)
    n = S.shape[0]
    return np.array([np.mean(S[: n - L] * S[L:], axis=0) if L else np.ones(S.shape[1]) for L in range(maxlag + 1)])


results, curves = {}, {}
for name, da in fields.items():
    for region, mask in [("deep", deep200), ("shelf", shelf)]:
        if name in DEEP_ONLY and region == "shelf":
            continue
        A, idx = anomalies(da, mask)
        cs, xs = spatial_curve(A, idx)
        rt = temporal_curve(A)
        tau_pts = np.array([crossing(np.arange(rt.shape[0]), rt[:, j], 1 / np.e) for j in range(rt.shape[1])])
        res = dict(
            label=LABELS[name],
            n_points=int(A.shape[1]),
            L_efold_km=crossing(xs, cs, 1 / np.e),
            L_half_var_km=crossing(xs, cs, np.sqrt(0.5)),  # rho^2 = 0.5
            L_zero_km=crossing(xs, cs, 0.0),
            tau_efold_days_median=float(np.nanmedian(tau_pts)),
            tau_efold_days_p25_p75=[float(np.nanpercentile(tau_pts, 25)), float(np.nanpercentile(tau_pts, 75))],
            anomaly_std=float(np.median(A.std(0))),
        )
        key = f"{name}|{region}"
        results[key] = res
        curves[key] = dict(x=xs.tolist(), rho=cs.tolist(), lag=list(range(rt.shape[0])), rt=np.median(rt, axis=1).tolist())
        print(f"{name:5s} {region:5s} L(1/e)={res['L_efold_km']:6.0f} km  L(rho2=.5)={res['L_half_var_km']:6.0f} km  "
              f"L0={res['L_zero_km']:6.0f} km  tau={res['tau_efold_days_median']:5.1f} d "
              f"[{res['tau_efold_days_p25_p75'][0]:.0f}-{res['tau_efold_days_p25_p75'][1]:.0f}]  n={res['n_points']}")

# --- local (mapped) e-folding scale for the heat proxy, deep water -------
A, idx = anomalies(fields["heat"], deep200)
S = (A - A.mean(0)) / A.std(0)
plo, pla = lon.ravel()[idx], lat.ravel()[idx]
refs = [j for j in range(len(idx)) if (round(plo[j] * 12) % 6 == 0 and round(pla[j] * 12) % 6 == 0)]
local = []
for j in refs:
    d = hav(plo[j], pla[j], plo, pla)
    near = d < 400
    c = S[:, near].T @ S[:, j] / S.shape[0]
    k = np.digitize(d[near], bins) - 1
    prof = np.array([np.median(c[k == i]) if np.any(k == i) else np.nan for i in range(40)])
    L = crossing(np.concatenate([[0], centers[:40]]), np.concatenate([[1], prof]), 1 / np.e)
    local.append((float(plo[j]), float(pla[j]), L))
print("local L(1/e) heat proxy: median %.0f km, p10 %.0f, p90 %.0f" % tuple(
    np.nanpercentile([l[2] for l in local], [50, 10, 90])))

json.dump(dict(region=args.region, region_name=REGIONS[args.region]["name"], model=F.attrs.get("model", args.model), sampling=F.attrs.get("sampling"),
               period=[str(F.time.values[0])[:10], str(F.time.values[-1])[:10]], n_days=int(F.sizes["time"]),
               areas=areas, results=results, curves=curves, local=local), open(W / "scales.json", "w"))
np.savez(W / "masks.npz", lon=lon, lat=lat, deep200=deep200, shelf=shelf, wet=wet)
