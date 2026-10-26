"""Step 1 for ESPC: pair glider AND Argo profiles with the HYCOM ESPC-D-V02 archive, in one pass.

usage: python step1_espc_pairs.py [region ...] [--workers N]      (default: all four regions)

Why separate from step1_pair_model_profiles.py: the HYCOM archive (tds.hycom.org ESPC-D-V02/{t3z,s3z}/<year>)
costs ~20 s per model time step however small the box, so it is read once per (region, model time) for
every glider and Argo profile that falls on that time, and each profile's nearest column is sampled
locally. Reading per profile would be ~10x slower. Run SEASON_PLATFORM-independent: it writes
<root>/<region>/... for gliders and <root>/argo/<region>/... for Argo, in the layout steps 2-3 read.

Pair rule is the same as the other models: every 6 h, each platform's profile closest in time within
+-3 h is matched with the nearest model column at that model time. Already-written pairs are skipped,
so it resumes. ESPC occasionally serves a placeholder time step that is flat with depth; those profiles
are skipped and counted (not replaced by an earlier step).
"""
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
from season_config import BASE, REGIONS, YEAR

TOL = pd.Timedelta('3h')
PAD = 0.3            # degrees around the profiles' bounding box
MAX_DEPTH = 1100
URL = 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02'
PLATFORMS = {'glider': BASE, 'argo': os.path.join(BASE, 'argo')}

_t = _s = None


def _init():
    global _t, _s
    from ioos_model_comparisons.models import _open_cf_dataset_sanitize_time
    _t = _open_cf_dataset_sanitize_time(f'{URL}/t3z/{YEAR}', drop_variables='tau')['water_temp']
    _s = _open_cf_dataset_sanitize_time(f'{URL}/s3z/{YEAR}', drop_variables='tau')['salinity']


def _flat(v):
    v = v[np.isfinite(v)]
    return v.size >= 2 and np.allclose(v, v[0], atol=1e-3)


def work(job):
    """One (region, model time): read the bounding box once, write every pair's model column."""
    region, date, rows = job
    lon = np.array([r['lon'] for r in rows]) % 360
    lat = np.array([r['lat'] for r in rows])
    box = dict(lon=slice(lon.min() - PAD, lon.max() + PAD), lat=slice(lat.min() - PAD, lat.max() + PAD),
               depth=slice(0, MAX_DEPTH))
    try:
        t = _t.sel(**box).sel(time=date).load()
        s = _s.sel(**box).sel(time=date).load()
    except KeyError:
        return 0, 0, len(rows)          # no model field at this time
    wrote = flat = 0
    for r, lo in zip(rows, lon):
        tp = t.sel(lon=lo, lat=r['lat'], method='nearest')
        sp = s.sel(lon=lo, lat=r['lat'], method='nearest')
        if _flat(tp.values):
            flat += 1
            continue
        mdf = pd.DataFrame({'depth': tp.depth.values, 'lat': float(tp.lat), 'lon': (float(tp.lon) + 180) % 360 - 180,
                            'time': date, 'temperature': tp.values, 'salinity': sp.values})
        mdf.to_csv(r['fname'], index=False)
        wrote += 1
    return wrote, flat, 0


def collect(region):
    """All pairs still to do for a region, across platforms, grouped by model time."""
    jobs, done = {}, 0
    for plat, root in PLATFORMS.items():
        f = os.path.join(root, f'{YEAR}_{region}_{plat}s.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, parse_dates=['time'], dtype={'glider': str})
        prof = df.drop_duplicates(['glider', 'time'])[['glider', 'time', 'lat', 'lon']].copy()
        prof['date'] = prof.time.dt.round('6h')
        prof['dt'] = (prof.time - prof.date).abs()
        prof = prof[prof.dt < TOL].sort_values('dt').drop_duplicates(['glider', 'date'])
        existing = set(glob.glob(os.path.join(root, region, '*', '*_espc.csv')))
        for p in prof.itertuples():
            stem = f"{p.glider}_{p.time:%Y%m%dT%H%M%SZ}"
            fname = os.path.join(root, region, p.glider, f'{stem}_espc.csv')
            if fname in existing:
                done += 1
                continue
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            gfile = os.path.join(root, region, p.glider, f'{stem}.csv')
            if not os.path.exists(gfile):                    # same glider profile the other models used
                df[(df.glider == p.glider) & (df.time == p.time)].to_csv(gfile, index=False)
            jobs.setdefault(p.date, []).append(dict(lon=p.lon, lat=p.lat, fname=fname))
    return [(region, d, rows) for d, rows in sorted(jobs.items())], done


def main():
    args = sys.argv[1:]
    workers = 8
    if '--workers' in args:
        i = args.index('--workers')
        workers = int(args[i + 1])
        del args[i:i + 2]
    regions = args or list(REGIONS)
    jobs, done = [], 0
    for r in regions:
        j, d = collect(r)
        print(f'{r}: {sum(len(x[2]) for x in j)} pairs to do in {len(j)} model times ({d} already saved)', flush=True)
        jobs += j
    if not jobs:
        return
    jobs.sort(key=lambda j: j[1])
    wrote = flat = missing = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc='espc', mininterval=60):
            try:
                w, fl, m = f.result()
                wrote += w; flat += fl; missing += m
            except Exception as e:
                print(f'Job failed: {type(e).__name__} {str(e)[:200]}', flush=True)
    print(f'DONE espc: wrote {wrote} pairs; skipped {flat} flat-with-depth profiles and {missing} with no model time',
          flush=True)


if __name__ == '__main__':
    main()
