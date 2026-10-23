"""Step 0 (Argo): download every Argo profile in a region for the season, QC'd and clipped.

usage: SEASON_PLATFORM=argo python step0_download_argos.py <region> [end date, default today]

- source: IFREMER ERDDAP ArgoFloats (platforms.get_argo_floats_by_time)
- QC: keep measurements whose time, position, pressure, temperature and salinity flags are good (1, 2, 5, 8).
  The per-profile grades (A-F) are not used.
- depth from pressure with gsw.z_from_p
- mexico_pacific: Pacific side only, south of 32.5N (same as the gliders)
Writes <ROOT>/<year>_<region>_argos.csv with the columns steps 1-3 expect; the float WMO id goes in 'glider'.
"""
import os
import sys
import warnings

import pandas as pd
from gsw import z_from_p

warnings.filterwarnings('ignore')
from ioos_model_comparisons.platforms import (ARGO_DATA_QC_VARIABLES, ARGO_LOCATION_QC_VARIABLES,
                                              filter_argo_by_qc, get_argo_floats_by_time)
from ioos_model_comparisons.regions import region_config

from season_config import PLATFORM, ROOT, T0, mexico_pacific_mask, profile_csv

if PLATFORM != 'argo':
    sys.exit('run with SEASON_PLATFORM=argo')

region = sys.argv[1]
t1 = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.now().floor('D')
extent = region_config(region)['extent']
os.makedirs(ROOT, exist_ok=True)

qc_cols = list(ARGO_LOCATION_QC_VARIABLES + ARGO_DATA_QC_VARIABLES)
raw = get_argo_floats_by_time(extent, T0.strftime('%Y-%m-%dT%H:%M:%SZ'), t1.strftime('%Y-%m-%dT%H:%M:%SZ'),
                              variables=['pres', 'temp', 'psal'], include_qc=qc_cols).reset_index()
out = profile_csv(region)
raw.to_csv(out.replace('.csv', '_raw.csv'), index=False)
print(f'{region}: downloaded {len(raw)} rows from {raw.argo.nunique() if len(raw) else 0} floats')

df = filter_argo_by_qc(raw, qc_columns=qc_cols)
df = pd.DataFrame({
    'glider': df['argo'].astype(str), 'time': df['time'], 'lat': df['lat'], 'lon': df['lon'],
    'depth': -z_from_p(df['pres (decibar)'].values, df['lat'].values),
    'temperature': df['temp (degree_Celsius)'], 'salinity': df['psal (PSU)'],
})
df = df[df.lon.between(extent[0], extent[1]) & df.lat.between(extent[2], extent[3])]
df = df[df.depth >= 0]   # a few near-surface pressures are slightly negative
if region == 'mexico_pacific':
    df = df[mexico_pacific_mask(df)]
df = df.sort_values(['glider', 'time', 'depth'])

prof = df.drop_duplicates(['glider', 'time'])
print(f'{region}: {len(df)} QC-good rows kept ({len(raw) - len(df)} dropped), '
      f'{prof.glider.nunique()} floats, {len(prof)} profiles')
df.to_csv(out, index=False)
