#!/usr/bin/env python3
"""
Step 4 of the glider-coverage analysis: summary figure.

Four panels from <workdir>/<region>/<model>/scales.json + coverage.json: spatial
correlation vs distance, temporal autocorrelation, deep-water coverage vs number
of gliders, and a map of the local e-folding radius. Writes
glider_scales.png to <workdir>/<region>/<model>/.

Usage:
    python scripts/tools/glider_coverage/figure.py --model cmems [--region caribbean_pr] [--workdir DIR]
"""
import json
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import argparse
from pathlib import Path
from gc_regions import REGIONS
from ioos_model_comparisons import configs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", choices=["cmems", "rtofs", "espc"], required=True)
ap.add_argument("--region", choices=list(REGIONS), default="gulf")
ap.add_argument("--workdir", type=Path, default=configs.path_data / "glider_coverage")
args = ap.parse_args()
W = args.workdir / args.region / args.model
RN = REGIONS[args.region]["name"]


S = json.load(open(W / "scales.json")); C = json.load(open(W / "coverage.json"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 9.5), facecolor="#fcfcfb")
series = [(k, S["results"][k]["label"]) for k in ("heat|deep", "t100|deep", "ssh|deep" if "ssh|deep" in S["results"] else "t200|deep")]

ax = fig.add_subplot(2, 2, 1)
for (k, lab), c in zip(series, COL):
    cv = S["curves"][k]; ax.plot(cv["x"], cv["rho"], color=c, lw=2, label=lab)
for y, t in [(np.sqrt(.5), "ρ = 0.71 (50% of variance)"), (1/np.e, "ρ = 1/e"), (0, "")]:
    ax.axhline(y, color=INK2 if y else INK2, lw=0.8, ls=":" if y else "-")
    if t: ax.text(790, y + .015, t, ha="right", va="bottom", color=INK2, fontsize=8.5)
r = S["results"]["heat|deep"]
ax.set(xlim=(0, 800), ylim=(-0.3, 1.02), xlabel="Separation (km)", ylabel="Correlation of anomalies",
       title="a) How far apart before the ocean decorrelates?")
ax.annotate(f"~{r['L_half_var_km']:.0f} km", (r["L_half_var_km"], np.sqrt(.5)), (150, 0.85), color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate(f"~{r['L_efold_km']:.0f} km", (r["L_efold_km"], 1/np.e), (230, 0.55), color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.42), fontsize=9)

ax = fig.add_subplot(2, 2, 2)
for (k, lab), c in zip(series, COL):
    cv = S["curves"][k]; ax.plot(cv["lag"], cv["rt"], color=c, lw=2, label=lab)
ax.axhline(1/np.e, color=INK2, lw=0.8, ls=":"); ax.text(88, 1/np.e + .015, "ρ = 1/e", ha="right", color=INK2, fontsize=8.5)
ax.axhline(0, color=INK2, lw=0.8)
ax.annotate(f"~{r['tau_efold_days_median']:.0f} days", (r["tau_efold_days_median"], 1/np.e), (30, 0.6), color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set(xlim=(0, 90), ylim=(-0.3, 1.02), xlabel="Lag (days)", ylabel="Autocorrelation (median over deep water)",
       title="b) How long does a correction stay relevant?")
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper right", fontsize=9)

ax = fig.add_subplot(2, 2, 3)
N = [c["N"] for c in C]
k1, k2 = "rho>=0.71 (>=50% var explained)", "rho>=1/e (some influence)"
ax.plot(N, [100 * c[f"mobile {k2}"] for c in C], color=COL[0], lw=2, marker="o", ms=4, label="Some influence (ρ ≥ 1/e)")
ax.plot(N, [100 * c[f"mobile {k1}"] for c in C], color=COL[1], lw=2, marker="o", ms=4, label="Well constrained (ρ ≥ 0.71)")
ax.plot(N, [100 * c[f"static {k1}"] for c in C], color=COL[1], lw=1.2, ls="--", label="Well constrained, station-keeping")
for y in (50, 90): ax.axhline(y, color=INK2, lw=0.6, ls=":")
ax.set(xlim=(0, 150), ylim=(0, 102), xlabel="Number of gliders (simultaneously in water)",
       ylabel="% of deep water (>200 m) covered", title="c) Coverage of deep water vs fleet size")
ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, loc="lower right", fontsize=9)

ax = fig.add_subplot(2, 2, 4, projection=ccrs.Mercator())
ax.set_extent(REGIONS[args.region]["bbox"], crs=ccrs.PlateCarree())
lo, la, L = np.array(S["local"]).T
sc = ax.scatter(lo, la, c=np.clip(L, 60, 300), s=38, cmap="Blues", vmin=60, vmax=300, transform=ccrs.PlateCarree(),
                edgecolor="#fcfcfb", linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="#e4e3df", zorder=0); ax.coastlines(lw=0.5, color=INK2)
cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.set_label("e-folding radius (km)", color=INK)
ax.set_title("d) Local e-folding radius, heat proxy", loc="center")

fig.suptitle(f"{RN}: decorrelation scales and glider coverage — {S['model']}, {S['sampling']}, {S['period'][0]} – {S['period'][1]}",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.text(0.02, 0.005, "Anomalies after removing each point's mean + annual/semiannual cycle. Coverage: simulated gliders at 20 km/day in water >200 m; "
         "a point counts as covered\nif some profile from the last 60 days has ρ_space·ρ_time above the threshold. Uses the single best profile, not combined profiles (conservative). Ignores currents pushing gliders and the assimilation system's own error model.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(W / "glider_scales.png", dpi=150, facecolor=fig.get_facecolor())
