"""Guam RTOFS, local part 1: pick glider/model-time pairs (same rule as step1) and write the glider profiles.

Guam RTOFS only exists as the binary-converted files on baffin
(/home/hurricaneadm/data/rtofs_archv/YYYY/MM/YYYYMMDD/rtofs_glo_YYYYMMDDTHH_guam.nc), so the model columns
are extracted there by extract_guam_rtofs.py from the pairs.csv written here. See run_guam_rtofs.sh.
"""
import glob
import os

import pandas as pd

from season_config import ROOT, glider_csv

save_dir = f'{ROOT}/guam'
tol = pd.Timedelta('3h')
model = 'rtofs'

gliders = pd.read_csv(glider_csv('guam'), parse_dates=['time'], dtype={'glider': str})
existing = set(glob.glob(f'{save_dir}/*/*_{model}.csv'))

pairs = []
t0, t1 = gliders.time.min().floor('D'), gliders.time.max().ceil('D')
for date in pd.date_range(t0, t1, freq='6h'):
    pdf = gliders[(gliders['time'] > date - tol) & (gliders['time'] < date + tol)]
    for name, group in pdf.groupby('glider'):
        best_time = group.loc[(group['time'] - date).abs().idxmin(), 'time']
        stem = f"{name}_{best_time.strftime('%Y%m%dT%H%M%SZ')}"
        if f'{save_dir}/{name}/{stem}_{model}.csv' in existing:
            continue
        prof = group.loc[group['time'] == best_time]
        os.makedirs(f'{save_dir}/{name}', exist_ok=True)
        gfile = f'{save_dir}/{name}/{stem}.csv'
        if not os.path.exists(gfile):
            prof.to_csv(gfile, index=False)
        pairs.append(dict(glider=name, stem=stem, date=date, lon=prof.lon.iloc[0], lat=prof.lat.iloc[0]))

pairs = pd.DataFrame(pairs).drop_duplicates('stem')
pairs.to_csv(f'{ROOT}/guam_rtofs_pairs.csv', index=False)
print(f'{len(pairs)} Guam glider/RTOFS pairs to extract ({len(existing)} already done)')
