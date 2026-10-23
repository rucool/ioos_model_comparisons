"""Step 3: glider vs model hexbin plots (OHC, salinity max, mixed layer depth, 26 C isotherm depth).

usage: python step3_plot_hexbin.py <title> <model> <t0> <t1> <region folder>
   e.g. python step3_plot_hexbin.py "Caribbean" cmems 2026-06-01 2026-10-22 caribbean

- Fit line is an orthogonal (total least squares) fit: both axes carry error (a glider profile has
  eddies/internal waves/sampling noise a model can't reproduce), and ordinary least squares of model
  on glider is biased flat by that noise (slope -> r * sy/sx).
- Colours are real profile counts per hexagon; the hexagon size is fixed per variable.
- Axes are sized to the data (99.9th percentile); stats use every valid pair.
- OHC pairs need qc == 1 on both sides (see ohc_with_qc in step2); MLD pairs need qcU == 1 on both.
"""
import os
import sys
from pathlib import Path

import cmocean as cmo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from season_config import PLOTS, ROOT

title_region, model, t0, t1, folder = sys.argv[1:6]
base_dir = Path(ROOT) / folder
save_dir = PLOTS
MODEL_NAMES = {'rtofs': 'RTOFS', 'cmems': 'CMEMS', 'espc': 'ESPC', 'amseas': 'AMSEAS', 'parallel': 'RTOFS-P'}
model_str = MODEL_NAMES.get(model, model)
reg_name = title_region.replace(' ', '_').lower()
MODELS = {'cmems', 'rtofs', 'espc', 'parallel'}

GLIDER_COLS = ['glider_ohc', 'glider_ohc_qc', 'glider_salinity_max', 'glider_mldU', 'glider_qcU', 'glider_iso26_depth']
MODEL_COLS = ['model_ohc', 'model_ohc_qc', 'model_salinity_max', 'model_mldU', 'model_qcU', 'model_iso26_depth']


def load_pairs():
    rows = []
    for calc_dir in base_dir.glob('*/calculated'):
        for g in calc_dir.glob('*.csv'):
            if g.stem.rsplit('_', 1)[-1] in MODELS:
                continue
            m = calc_dir / f'{g.stem}_{model}.csv'
            if not m.exists():
                continue
            gd = pd.read_csv(g, nrows=1)
            md = pd.read_csv(m, nrows=1)
            if not set(GLIDER_COLS) <= set(gd.columns) or not set(MODEL_COLS) <= set(md.columns):
                continue
            rows.append({'glider': gd['glider'].iloc[0], 'time': gd['time'].iloc[0],
                         **gd[GLIDER_COLS].iloc[0].to_dict(), **md[MODEL_COLS].iloc[0].to_dict()})
    return pd.DataFrame(rows)


def stats(x, y):
    r = np.corrcoef(x, y)[0, 1]
    w, v = np.linalg.eigh(np.cov(x, y))
    major = v[:, np.argmax(w)]               # principal axis of the scatter
    slope = major[1] / major[0]
    return dict(n=len(x), bias=np.mean(y - x), rmse=np.sqrt(np.mean((y - x) ** 2)), r=r,
                slope=slope, intercept=y.mean() - slope * x.mean())


def hexbin_plot(x, y, prefix, label, units, binw, lo=None):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        print(f'{prefix}: only {len(x)} valid pairs, skipped')
        return None
    s = stats(x, y)

    both = np.concatenate([x, y])
    if lo is None:
        lo = np.floor(np.percentile(both, 0.1) / (5 * binw)) * 5 * binw
    hi = np.ceil(np.percentile(both, 99.9) / (5 * binw)) * 5 * binw
    lims = (lo, hi)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = cmo.tools.crop_by_percent(cmo.cm.algae, 15, which='min')
    hb = ax.hexbin(x, y, gridsize=int(round((hi - lo) / binw)), extent=(lo, hi, lo, hi),
                   mincnt=1, cmap=cmap, linewidths=0.1)
    counts = hb.get_array()
    levels = MaxNLocator(nbins=10, integer=True, min_n_ticks=4).tick_values(1, counts.max())
    levels = levels[levels >= 1]
    if levels[0] > 1:
        levels = np.insert(levels, 0, 1)
    if levels[-1] <= counts.max():
        levels = np.append(levels, counts.max() + 1)
    hb.set_norm(BoundaryNorm(levels, cmap.N))
    cb = fig.colorbar(hb, ax=ax, boundaries=levels, ticks=levels[:-1])
    cb.set_label('Profiles per hexagon')

    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    xf = np.array([max(lo, x.min()), min(hi, x.max())])
    ax.plot(xf, s['intercept'] + s['slope'] * xf, 'r-', lw=2, label='Orthogonal fit')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel(f'Glider {label} ({units})', fontsize='x-large')
    ax.set_ylabel(f'{model_str} {label} ({units})', fontsize='x-large')
    ax.set_title(f'{title_region} {label} - Glider vs {model_str}\n{t0} to {t1}', fontsize='large')
    ax.text(0.03, 0.97,
            f"n = {s['n']}\nbias = {s['bias']:.2f}\nRMSE = {s['rmse']:.2f}\nr = {s['r']:.2f}\n"
            f"slope = {s['slope']:.2f}\nintercept = {s['intercept']:.2f}",
            transform=ax.transAxes, va='top', fontsize='medium',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.legend(loc='lower right')

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'{prefix}_glider_vs_{model_str}_hexbin_{t0}_{t1}_{reg_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot to {fname}  (max {int(counts.max())} per hexagon)')
    return {'region': title_region, 'model': model_str, 'variable': label, 'units': units, **s}


df = load_pairs()
print(f'{len(df)} glider/{model_str} pairs in {base_dir}')

ohc = df[(df.glider_ohc_qc == 1) & (df.model_ohc_qc == 1)]
# Pairs where neither profile has water >= 26 C agree trivially; left in, they pile up at (0, 0),
# pin the fit to the origin and inflate r. Keep pairs where at least one side has heat content.
both_zero = (ohc.glider_ohc == 0) & (ohc.model_ohc == 0)
print(f'OHC: {len(ohc)} of {len(df)} pairs pass qc '
      f'(glider qc counts {df.glider_ohc_qc.value_counts().to_dict()}, model {df.model_ohc_qc.value_counts().to_dict()}); '
      f'{both_zero.sum()} with OHC = 0 in both dropped')
ohc = ohc[~both_zero]
mld = df[(df.glider_qcU == 1) & (df.model_qcU == 1)]

results = [
    hexbin_plot(ohc.glider_ohc.values, ohc.model_ohc.values, 'ohc', 'OHC', 'kJ/cm²', binw=4, lo=0),
    hexbin_plot(df.glider_iso26_depth.values, df.model_iso26_depth.values, '26c', '26°C Isotherm Depth', 'm', binw=4, lo=0),
    hexbin_plot(mld.glider_mldU.values, mld.model_mldU.values, 'mld', 'Mixed Layer Depth', 'm', binw=2, lo=0),
    hexbin_plot(df.glider_salinity_max.values, df.model_salinity_max.values, 'salmax', 'Salinity Max', 'PSU', binw=0.02),
]

summary = os.path.join(save_dir, 'summary.csv')
new = pd.DataFrame([r for r in results if r]).assign(t0=t0, t1=t1)
if os.path.exists(summary):
    old = pd.read_csv(summary)
    old = old[~((old.region == title_region) & (old.model == model_str))]
    new = pd.concat([old, new], ignore_index=True)
new.round(3).to_csv(summary, index=False)
