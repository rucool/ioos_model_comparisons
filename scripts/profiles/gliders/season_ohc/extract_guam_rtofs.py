"""Guam RTOFS, runs on baffin: nearest RTOFS column for each glider/model-time pair in pairs.csv.

usage (baffin, model_comps env):
    python extract_guam_rtofs.py pairs.csv out_dir [archive_dir]

Writes out_dir/<glider>/<stem>_rtofs.csv with the same columns as the other regions
(depth, lat, lon, time, temperature, salinity). Pairs whose model time has no file are skipped,
matching step1's behaviour for missing RTOFS times.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

pairs_csv, out_dir = sys.argv[1], sys.argv[2]
archive = sys.argv[3] if len(sys.argv) > 3 else '/home/hurricaneadm/data/rtofs_archv'
max_depth = 1100   # gliders stop at <= 1000 m

pairs = pd.read_csv(pairs_csv, parse_dates=['date'])
written = missing = 0
for date, group in pairs.groupby('date'):
    f = f"{archive}/{date:%Y/%m/%Y%m%d}/rtofs_glo_{date:%Y%m%dT%H}_guam.nc"
    if not os.path.exists(f):
        missing += len(group)
        continue
    with xr.open_dataset(f) as ds:
        sub = ds[['temp', 'salin']].sel(z=slice(0, max_depth)).load()
    for p in group.itertuples():
        col = sub.sel(lon=p.lon, lat=p.lat, method='nearest')
        mdf = pd.DataFrame({'depth': col.z.values, 'lat': float(col.lat), 'lon': float(col.lon), 'time': date,
                            'temperature': col.temp.values, 'salinity': col.salin.values})
        os.makedirs(f'{out_dir}/{p.glider}', exist_ok=True)
        mdf.to_csv(f'{out_dir}/{p.glider}/{p.stem}_rtofs.csv', index=False)
        written += 1
print(f'wrote {written} model profiles; {missing} pairs had no RTOFS file for their model time')
