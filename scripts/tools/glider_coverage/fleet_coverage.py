#!/usr/bin/env python3
"""
Step 3 of the glider-coverage analysis: fleet size vs coverage.

Simulates N gliders flying at 20 km/day in the region's deep (>200 m) water (straight
lines, new random heading on reaching the 200 m isobath, a profile every 6 h)
and scores each 1/6 deg grid point by the best single profile from the last
60 days: rho_space(distance) * rho_time(age), using the measured curves (heat proxy by default, --var)
curves from scales.json. Reports the fraction of the deep region above
rho = 1/e ("some influence") and rho = 0.71 (>= 50% of variance explained),
plus a station-keeping (static) comparison.

Using the single best profile ignores the extra skill of combining profiles
(conservative); ignoring currents that push gliders around is optimistic.

Writes coverage.json (or coverage_<var>.json for --var other than heat) to <workdir>/<region>/<model>/.
Takes ~5 min.

Usage:
    python scripts/tools/glider_coverage/fleet_coverage.py --model cmems [--region caribbean_pr] [--var salt] [--workdir DIR]
"""
import json
import numpy as np

import argparse
from pathlib import Path
from gc_regions import REGIONS, center
from ioos_model_comparisons import configs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", choices=["cmems", "rtofs", "espc"], required=True)
ap.add_argument("--var", default="heat", help="scales.json variable whose curves to use (heat, salt, u200, v200, ...)")
ap.add_argument("--region", choices=list(REGIONS), default="gulf")
ap.add_argument("--workdir", type=Path, default=configs.path_data / "glider_coverage")
args = ap.parse_args()
W = args.workdir / args.region / args.model


rng = np.random.default_rng(1)
S = json.load(open(W / "scales.json"))
cv = S["curves"][f"{args.var}|deep"]
xs, rs = np.array(cv["x"]), np.clip(np.array(cv["rho"]), 0, 1)
lags, rt = np.array(cv["lag"]), np.clip(np.array(cv["rt"]), 0, 1)
m = np.load(W / "masks.npz")
lon, lat, deep = m["lon"], m["lat"], m["deep200"]

# km coordinates (equirectangular about the region centre), coarsen grid to 1/6 deg
lon0, lat0 = center(args.region)
X = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
Y = (lat - lat0) * 111.32
gx, gy = X[::2, ::2][deep[::2, ::2]], Y[::2, ::2][deep[::2, ::2]]
DX, DY = X[deep], Y[deep]
cell = 111.32 / 12

SPEED, DT, WINDOW, SPIN, EVAL = 20.0, 0.25, 60, 90, range(90, 181, 15)  # km/d, d, d, d


def in_deep(x, y):
    i = np.clip(np.round((lat0 + y / 111.32 - lat[0, 0]) * 12).astype(int), 0, deep.shape[0] - 1)
    j = np.clip(np.round((lon0 + x / (111.32 * np.cos(np.deg2rad(lat0))) - lon[0, 0]) * 12).astype(int), 0, deep.shape[1] - 1)
    return deep[i, j]


def spread(n):
    """Farthest-point sampling of n start positions in deep water."""
    k = rng.integers(len(DX))
    pts = [k]
    d = np.hypot(DX - DX[k], DY - DY[k])
    for _ in range(n - 1):
        k = int(np.argmax(d))
        pts.append(k)
        d = np.minimum(d, np.hypot(DX - DX[k], DY - DY[k]))
    return DX[pts].copy(), DY[pts].copy()


def simulate(n):
    x, y = spread(n)
    th = rng.uniform(0, 2 * np.pi, n)
    obs = []  # (t, x, y)
    out = {}
    for step in range(int(max(EVAL) / DT) + 1):
        t = step * DT
        obs.append(np.column_stack([np.full(n, t), x, y]))
        if any(abs(t - e) < 1e-9 for e in EVAL):
            O = np.concatenate(obs)
            O = O[O[:, 0] > t - WINDOW]
            best = np.zeros(len(gx))
            for c in range(0, len(O), 4000):
                o = O[c:c + 4000]
                d = np.hypot(gx[:, None] - o[None, :, 1], gy[:, None] - o[None, :, 2])
                r = np.interp(d, xs, rs, right=0) * np.interp(t - o[None, :, 0], lags, rt, right=0)
                best = np.maximum(best, r.max(1))
            out[t] = best
        # move: straight line, new random heading when leaving deep water
        nx, ny = x + SPEED * DT * np.cos(th), y + SPEED * DT * np.sin(th)
        bad = ~in_deep(nx, ny)
        th[bad] = rng.uniform(0, 2 * np.pi, bad.sum())
        x, y = np.where(bad, x, nx), np.where(bad, y, ny)
    return out


def static(n):
    """Station-keeping gliders/moorings at spread positions (infinite time memory)."""
    x, y = spread(n)
    d = np.hypot(gx[:, None] - x[None], gy[:, None] - y[None]).min(1)
    return np.interp(d, xs, rs, right=0)


Ns = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150]
thr = {"rho>=0.71 (>=50% var explained)": np.sqrt(0.5), "rho>=1/e (some influence)": 1 / np.e}
table = []
for n in Ns:
    mob = {k: [] for k in thr}
    for rep in range(3):
        for best in simulate(n).values():
            for k, v in thr.items():
                mob[k].append(np.mean(best >= v))
    st = static(n)
    row = dict(N=n, **{f"mobile {k}": float(np.mean(v)) for k, v in mob.items()},
               **{f"static {k}": float(np.mean(st >= v)) for k, v in thr.items()})
    table.append(row)
    print(n, {k: round(v, 3) for k, v in row.items() if k != "N"}, flush=True)
json.dump(table, open(W / ("coverage.json" if args.var == "heat" else f"coverage_{args.var}.json"), "w"))
