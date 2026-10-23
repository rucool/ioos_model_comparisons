"""Step 4: glider (or Argo) OHC vs the NESDIS satellite-derived OHC product.

usage: python step4_nesdis_ohc.py <title> <t0> <t1> <region folder>
   e.g. python step4_nesdis_ohc.py "Caribbean" 2026-06-01 2026-10-22 caribbean

Uses the profile OHC from step 2 (ohc_with_qc; only qc == 1 profiles), so run steps 1-2 for CMEMS first.
NESDIS OHC (CoastWatch, OHC{NA,NP}14DayAgg: a 14-day composite updated daily, NA = northwest Atlantic,
NP = north Pacific 100E-80W) is read once for the region box and season from THREDDS, then sampled at
each profile's day (12Z, nearest step within a day) and nearest grid point. NESDIS stores OHC = 0 in
cells it does not compute (shallow/coastal water, flagged good and not land); those are treated as
missing: OHC = 0 with NESDIS SST >= 26 C is impossible. Pairs with OHC = 0 on both sides are left out,
as in step 3. Plots (data-sized and fixed 0-250 axes) and summary.csv rows as step 3.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ioos_model_comparisons.regions import region_config
from season_config import ROOT
from step3_plot_hexbin import MODELS, plot_both, save_summary

title_region, t0, t1, folder = sys.argv[1:5]
extent = region_config(folder)['extent']
basin = 'NA' if (-100 <= extent[0] and extent[1] <= 0) else 'NP'


def profiles():
    rows = []
    for g in Path(ROOT, folder).glob('*/calculated/*.csv'):
        if g.stem.rsplit('_', 1)[-1] in MODELS:
            continue
        d = pd.read_csv(g, nrows=1, usecols=['glider', 'time', 'lat', 'lon', 'glider_ohc', 'glider_ohc_qc'],
                        dtype={'glider': str})
        rows.append(d.iloc[0].to_dict())
    return pd.DataFrame(rows)


def nesdis_box():
    """Season x region subset of the NESDIS 14-day OHC aggregation, longitudes in -180..180."""
    url = f'https://coastwatch.noaa.gov/thredds/dodsC/OHC{basin}14DayAgg'
    ds = xr.open_dataset(url)[['ohc', 'sst']]
    lon_lo, lon_hi = (extent[0] % 360, extent[1] % 360) if basin == 'NP' else (extent[0], extent[1])
    lat = ds.latitude.values
    lon = ds.longitude.values
    iy = np.where((lat >= extent[2] - 0.5) & (lat <= extent[3] + 0.5))[0]
    ix = np.where((lon >= lon_lo - 0.5) & (lon <= lon_hi + 0.5))[0]
    ds = ds.sel(time=slice(pd.Timestamp(t0) - pd.Timedelta('2D'), pd.Timestamp(t1) + pd.Timedelta('2D')))
    ds = ds.isel(latitude=slice(iy.min(), iy.max() + 1), longitude=slice(ix.min(), ix.max() + 1)).load()
    if basin == 'NP':
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby('longitude')
    return ds


prof = profiles()
prof['time'] = pd.to_datetime(prof.time)
prof = prof[prof.glider_ohc_qc == 1]
ds = nesdis_box()
units = ds.ohc.attrs.get('units', '?')
print(f'{title_region}: {len(prof)} QC-passing profiles; NESDIS {basin} OHC {units}, '
      f'{ds.sizes["time"]} steps {pd.Timestamp(ds.time.values[0]):%Y-%m-%d} to {pd.Timestamp(ds.time.values[-1]):%Y-%m-%d}')

target = prof.time.dt.floor('D') + pd.Timedelta('12h')
ti = pd.DatetimeIndex(ds.time.values).get_indexer(pd.DatetimeIndex(target), method='nearest',
                                                  tolerance=pd.Timedelta('1D'))
ok = ti >= 0
yi = np.abs(ds.latitude.values[None, :] - prof.lat.values[:, None]).argmin(1)
xi = np.abs(ds.longitude.values[None, :] - prof.lon.values[:, None]).argmin(1)
vals = np.full(len(prof), np.nan)
sst = np.full(len(prof), np.nan)
vals[ok] = ds.ohc.values[ti[ok], yi[ok], xi[ok]]
sst[ok] = ds.sst.values[ti[ok], yi[ok], xi[ok]]
not_computed = (vals == 0) & (sst >= 26)
vals[not_computed] = np.nan
prof['nesdis_ohc'] = vals
prof['nesdis_sst'] = sst
prof['nesdis_not_computed'] = not_computed
prof['nesdis_time'] = pd.NaT
prof.loc[ok, 'nesdis_time'] = ds.time.values[ti[ok]]

prof.to_csv(os.path.join(ROOT, f'{folder}_nesdis_ohc_pairs.csv'), index=False)
pairs = prof.dropna(subset=['nesdis_ohc'])
both_zero = (pairs.glider_ohc == 0) & (pairs.nesdis_ohc == 0)
print(f'{len(pairs)} pairs with a NESDIS value ({(~ok).sum()} profiles had no NESDIS day, '
      f'{not_computed.sum()} on cells NESDIS does not compute, '
      f'{len(prof) - (~ok).sum() - not_computed.sum() - len(pairs)} on other missing cells); {both_zero.sum()} with OHC = 0 in both dropped')
pairs = pairs[~both_zero]

kw = dict(title_region=title_region, model_str='NESDIS', t0=t0, t1=t1)
res = plot_both(pairs.glider_ohc.values, pairs.nesdis_ohc.values, 'ohc', 'OHC', 'kJ/cm²', binw=4, lo=0, **kw)
save_summary([res], title_region, 'NESDIS', t0, t1)
