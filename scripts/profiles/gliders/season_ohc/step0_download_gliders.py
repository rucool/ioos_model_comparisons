"""Step 0: download every glider profile in a region for the season, clipped and cleaned.

usage: python step0_download_gliders.py <region> [end date, default today]

Cleaning (the raw download is kept as <file>_raw.csv):
- clip to the region box: the DAC search matches datasets by box, but returns their whole track
- mexico_pacific: Pacific side only. The box (-126..-92 E) also covers the western Gulf of Mexico,
  so drop north of 18N and east of 100W; and drop north of 32.5N (US-Mexico border; CalCOFI Spray lines)
- duplicate datasets: some platforms are published twice for the same period (e.g. sg148-...T0000 and
  sg148-...T1939, same track, different profile segmentation). Where two datasets of one platform
  overlap in time, keep the one that ends last.
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
from ioos_model_comparisons.platforms import get_active_gliders
from ioos_model_comparisons.regions import region_config

from season_config import ROOT, T0, glider_csv

region = sys.argv[1]
t1 = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.now().floor('D')
extent = region_config(region)['extent']
os.makedirs(ROOT, exist_ok=True)

variables = ['time', 'latitude', 'longitude', 'depth', 'temperature', 'salinity', 'profile_id']
# parallel=True fails with current erddapy (joblib can't pickle the ERDDAP object)
df = get_active_gliders(extent, T0, t1, variables=variables, timeout=180, parallel=False).reset_index()
out = glider_csv(region)
df.to_csv(out.replace('.csv', '_raw.csv'), index=False)
print(f'{region}: downloaded {len(df)} rows from {df.glider.nunique() if len(df) else 0} datasets')

if len(df):
    df = df[df.lon.between(extent[0], extent[1]) & df.lat.between(extent[2], extent[3])]
    if region == 'mexico_pacific':
        df = df[~((df.lat > 18) & (df.lon > -100)) & (df.lat < 32.5)]

    span = df.groupby('glider').time.agg(['min', 'max'])
    span['platform'] = span.index.str.split('-').str[0].str.lower()
    drop = set()
    for _, g in span.groupby('platform'):
        g = g.sort_values('max', ascending=False)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]           # a ends later than b
                if b['max'] > a['min'] and g.index[j] not in drop:
                    drop.add(g.index[j])
    if drop:
        print(f'{region}: dropping duplicate datasets {sorted(drop)}')
    df = df[~df.glider.isin(drop)]

    prof = df.drop_duplicates(['glider', 'time'])
    print(f'{region}: {prof.glider.nunique()} gliders, {len(prof)} profiles kept')
    print(prof.groupby('glider').agg(t0=('time', 'min'), t1=('time', 'max'), nprof=('time', 'size'),
                                     lat=('lat', 'mean'), lon=('lon', 'mean')).round(2).to_string())
df.to_csv(out, index=False)
