"""Step 0 (Argo): download every Argo profile in a region for the season, QC'd and clipped.

usage: SEASON_PLATFORM=argo python step0_download_argos.py <region> [end date, default today]

Source: IFREMER ERDDAP ArgoFloats. Depth from pressure with gsw.z_from_p.

QC, so profiles that fail never reach the statistics:
1. Profile level: drop the whole profile unless its pressure, temperature and salinity profile grades
   (profile_*_qc) are all A or B (>= 75% good levels) and its position and time flags are good
   (1, 2, 5 or 8). Grades C-F or no grade fail.
2. Level level: in kept profiles, drop measurements whose pres/temp/psal flag is not good.
Every excluded profile is listed with its reason in <region>_argo_qc_excluded.csv, and the counts go to
argo_qc_summary.csv. Pair files left from an earlier run for profiles that now fail are deleted, so
steps 2-3 can't pick them up.

mexico_pacific: Pacific side only, south of 32.5N (same as the gliders).
Writes <ROOT>/<year>_<region>_argos.csv with the columns steps 1-3 expect; the float WMO id goes in 'glider'.
"""
import glob
import os
import sys
import warnings

import pandas as pd
from erddapy import ERDDAP
from gsw import z_from_p

warnings.filterwarnings('ignore')
from ioos_model_comparisons.platforms import ARGO_GOOD_QC_FLAGS
from ioos_model_comparisons.regions import region_config

from season_config import PLATFORM, ROOT, T0, mexico_pacific_mask, profile_csv

if PLATFORM != 'argo':
    sys.exit('run with SEASON_PLATFORM=argo')

GOOD_GRADES = {'A', 'B'}
GOOD = set(ARGO_GOOD_QC_FLAGS)

region = sys.argv[1]
t1 = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.now().floor('D')
extent = region_config(region)['extent']
os.makedirs(ROOT, exist_ok=True)

e = ERDDAP(server='IFREMER', protocol='tabledap', response='csv')
e.dataset_id = 'ArgoFloats'
e.constraints = {'time>=': T0.strftime('%Y-%m-%dT%H:%M:%SZ'), 'time<=': t1.strftime('%Y-%m-%dT%H:%M:%SZ'),
                 'longitude>=': extent[0], 'longitude<=': extent[1],
                 'latitude>=': extent[2], 'latitude<=': extent[3]}
e.variables = ['platform_number', 'time', 'longitude', 'latitude', 'pres', 'temp', 'psal', 'data_mode',
               'time_qc', 'position_qc', 'pres_qc', 'temp_qc', 'psal_qc',
               'profile_pres_qc', 'profile_temp_qc', 'profile_psal_qc']
raw = e.to_pandas(dtype=str)        # strings: keep flags/grades exactly as served, no dropna
raw.columns = [c.split(' (')[0] for c in raw.columns]
out = profile_csv(region)
raw.to_csv(out.replace('.csv', '_raw.csv'), index=False)

raw['glider'] = raw.platform_number.str.strip()
raw['time'] = pd.to_datetime(raw.time).dt.tz_localize(None)
for c in ['longitude', 'latitude', 'pres', 'temp', 'psal']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
flags = {c: raw[c].fillna('').str.strip() for c in raw.columns if c.endswith('_qc')}

# --- 1. profile-level QC ---------------------------------------------------------------------
prof = raw.assign(**{c: v for c, v in flags.items()}).groupby(['glider', 'time']).first()
reasons = pd.Series('', index=prof.index)
for c in ['profile_pres_qc', 'profile_temp_qc', 'profile_psal_qc']:
    bad = ~prof[c].isin(GOOD_GRADES)
    reasons[bad] += prof.loc[bad, c].replace('', 'none').radd(f'{c.split("_")[1]} grade ') + '; '
for c in ['position_qc', 'time_qc']:
    bad = ~prof[c].isin(GOOD)
    reasons[bad] += prof.loc[bad, c].radd(f'{c} ') + '; '
in_box = prof.longitude.between(extent[0], extent[1]) & prof.latitude.between(extent[2], extent[3])
if region == 'mexico_pacific':
    in_box &= mexico_pacific_mask(prof.rename(columns={'latitude': 'lat', 'longitude': 'lon'}))
failed = reasons[(reasons != '') & in_box].str.rstrip('; ')
keep = prof.index[(reasons == '') & in_box]

excl = failed.rename('reason').reset_index()
excl = excl.merge(prof[['longitude', 'latitude', 'data_mode', 'profile_pres_qc', 'profile_temp_qc',
                        'profile_psal_qc']].reset_index(), on=['glider', 'time'])
excl.to_csv(os.path.join(ROOT, f'{region}_argo_qc_excluded.csv'), index=False)

# --- 2. level-level QC -----------------------------------------------------------------------
df = raw.set_index(['glider', 'time']).loc[keep].reset_index()
lev_ok = pd.Series(True, index=raw.index)
for c in ['pres_qc', 'temp_qc', 'psal_qc']:
    lev_ok &= flags[c].isin(GOOD)
df = raw[lev_ok.values & raw.set_index(['glider', 'time']).index.isin(keep)]
df = pd.DataFrame({
    'glider': df.glider, 'time': df.time, 'lat': df.latitude, 'lon': df.longitude,
    'depth': -z_from_p(df.pres.values, df.latitude.values), 'temperature': df.temp, 'salinity': df.psal,
}).dropna(subset=['depth', 'temperature'])
df = df[df.depth >= 0].sort_values(['glider', 'time', 'depth'])   # a few near-surface pressures are negative
df.to_csv(out, index=False)
kept = df.drop_duplicates(['glider', 'time'])

# --- accounting ------------------------------------------------------------------------------
n_in = int(in_box.sum())
summ = pd.DataFrame([{'region': region, 'profiles_in_region': n_in, 'failed_profile_qc': len(failed),
                      'no_good_levels_left': n_in - len(failed) - len(kept), 'kept': len(kept),
                      'floats_kept': kept.glider.nunique()}])
sfile = os.path.join(ROOT, 'argo_qc_summary.csv')
if os.path.exists(sfile):
    old = pd.read_csv(sfile)
    summ = pd.concat([old[old.region != region], summ], ignore_index=True)
summ.to_csv(sfile, index=False)
print(f'{region}: {n_in} profiles in region, {len(failed)} failed profile QC, '
      f'{len(kept)} profiles kept from {kept.glider.nunique()} floats')
if len(failed):
    print(failed.str.split(';').explode().str.strip().str.replace(r' ([A-Z0-9]*|none)$', '', regex=True).str.strip()
          .value_counts().to_string())

# --- remove pair files for profiles that no longer pass ------------------------------------------
keep_stems = {f"{g}_{t:%Y%m%dT%H%M%SZ}" for g, t in zip(kept.glider, kept.time)}
removed = 0
for f in glob.glob(os.path.join(ROOT, region, '*', '*.csv')) + glob.glob(os.path.join(ROOT, region, '*', 'calculated', '*.csv')):
    stem = os.path.basename(f)[:-4]
    base = stem.rsplit('_', 1)[0] if stem.rsplit('_', 1)[-1] in ('cmems', 'rtofs', 'espc', 'parallel') else stem
    if base not in keep_stems:
        os.remove(f)
        removed += 1
print(f'{region}: removed {removed} pair/metric files for profiles that are no longer kept')
