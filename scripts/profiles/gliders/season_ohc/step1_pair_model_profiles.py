"""Step 1: pair glider profiles with model profiles (same matching as oneoffs/glider_profile_save_parallel.py).

At every 6 h model time, each glider's profile closest in time (within +-3 h) is paired with the
nearest model grid column at that model time, and saved as
    <root>/<region>/<glider>/<glider>_<time>.csv          (glider profile)
    <root>/<region>/<glider>/<glider>_<time>_<model>.csv  (model column)
Already-written model files are skipped, so re-running resumes / extends the season.

All models are read per glider per week as one small box (RTOFS from the Rutgers THREDDS, CMEMS from
its time-series-chunked ARCO store, ESPC from HYCOM), then sampled locally: far fewer remote reads than
one per profile. Guam RTOFS is not on THREDDS; see run_guam_rtofs.sh.

usage: python step1_pair_model_profiles.py <region> <model: rtofs|cmems|espc> [workers]
"""
import glob
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from season_config import ROOT, RTOFS_SOURCE, glider_csv as glider_csv_path

warnings.filterwarnings('ignore')

region_str, model = sys.argv[1], sys.argv[2]
max_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
glider_csv = glider_csv_path(region_str)
tol = pd.Timedelta('3h')
window = '7D'        # CMEMS/ESPC: one box read per glider per window
pad = 0.15           # degrees around the glider track, so the nearest grid point is inside the box
max_depth = 1100     # gliders stop at <= 1000 m

save_dir = f'{ROOT}/{region_str}'
existing_files = set(glob.glob(f'{save_dir}/*/*_{model}.csv'))

gliders = pd.read_csv(glider_csv, parse_dates=['time'], dtype={'glider': str})   # Argo WMO ids are numeric

# ---- model handles ---------------------------------------------------------------------------
LOCAL = False
if model == 'rtofs':
    if region_str not in RTOFS_SOURCE:
        sys.exit(f'No RTOFS grid covers {region_str}')
    src = RTOFS_SOURCE[region_str]
    # Local archive on the fileserver (mount /Volumes/hurricaneadm): ~100x faster than THREDDS, same files.
    # Set RTOFS_ARCHIVE=none to force THREDDS.
    local_dir = os.path.join(os.environ.get('RTOFS_ARCHIVE', '/Volumes/hurricaneadm/data'),
                             'rtofs' if src == 'east' else 'rtofs_west')
    LOCAL = os.path.isdir(local_dir)
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')   # file locking fails on the share
    if LOCAL:
        import xarray as xr

        def rtofs_file(date):
            # 00Z field of day D is f024 of the cycle on D-1; 06/12/18Z are f006/f012/f018 of day D
            if date.hour == 0:
                day, fh = date - pd.Timedelta('1D'), 'f024'
            else:
                day, fh = date, {6: 'f006', 12: 'f012', 18: 'f018'}[date.hour]
            return f'{local_dir}/rtofs.{day:%Y%m%d}/rtofs_glo_3dz_{fh}_6hrly_hvr_US_{src}.nc'

        ref = next(f for f in (rtofs_file(d) for d in pd.date_range(gliders.time.min().floor('D') + pd.Timedelta('6h'), periods=40, freq='6h'))
                   if os.path.exists(f))
        with xr.open_dataset(ref) as d0:
            rlon, rlat = d0.Longitude.values[0, :], d0.Latitude.values[:, 0]
        print(f'RTOFS {src}: reading the local archive {local_dir}', flush=True)
    else:
        from ioos_model_comparisons.models import rtofs
        ds = rtofs(rename=True, chunks={'time': 1}, source=src)
        rlon = ds.lon.values[0, :]
        rlat = ds.lat.values[:, 0]
    # Profiles outside the regional grid would snap to its edge; drop them.
    gliders = gliders[gliders.lon.between(rlon.min(), rlon.max()) & gliders.lat.between(rlat.min(), rlat.max())]
elif model == 'cmems':
    import copernicusmarine as cm
    from ioos_model_comparisons import env
    user, pw = env.cmems_credentials()
    # Same products as models.CMEMS, but the time-series-chunked store.
    temp_da = cm.open_dataset(dataset_id='cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i', username=user, password=pw,
                              service='arco-time-series', chunk_size_limit=0)['thetao']
    salt_da = cm.open_dataset(dataset_id='cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i', username=user, password=pw,
                              service='arco-time-series', chunk_size_limit=0)['so']
    LON, LAT, to_lon = 'longitude', 'latitude', (lambda x: x)
elif model == 'espc':
    sys.exit('ESPC pairs gliders and Argo together: use step1_espc_pairs.py')
    from ioos_model_comparisons.models import _open_cf_dataset_sanitize_time
    from ioos_model_comparisons.calc import lon180to360
    year = gliders.time.min().year
    base = 'https://tds.hycom.org/thredds/dodsC/ESPC-D-V02'
    temp_da = _open_cf_dataset_sanitize_time(f'{base}/t3z/{year}', drop_variables='tau')['water_temp']
    salt_da = _open_cf_dataset_sanitize_time(f'{base}/s3z/{year}', drop_variables='tau')['salinity']
    LON, LAT, to_lon = 'lon', 'lat', lon180to360
else:
    raise ValueError(f'Model {model} is not supported.')

# ---- glider/model-time pairs (identical rule to glider_profile_save_parallel.py) -------------
t0, t1 = gliders.time.min().floor('D'), gliders.time.max().ceil('D')
pairs = []
for date in pd.date_range(t0, t1, freq='6h'):
    pdf = gliders[(gliders['time'] > date - tol) & (gliders['time'] < date + tol)]
    for name, group in pdf.groupby('glider'):
        best_time = group.loc[(group['time'] - date).abs().idxmin(), 'time']
        fname = os.path.join(save_dir, name, f"{name}_{best_time.strftime('%Y%m%dT%H%M%SZ')}_{model}.csv")
        if fname in existing_files:
            continue
        row = group.loc[group['time'] == best_time].iloc[0]
        pairs.append(dict(glider=name, date=date, best_time=best_time, lon=row.lon, lat=row.lat, fname=fname))
pairs = pd.DataFrame(pairs)
print(f'{region_str}/{model}: {gliders.glider.nunique()} gliders, {t0:%Y-%m-%d} to {t1:%Y-%m-%d}, '
      f'{len(existing_files)} already saved, {len(pairs)} to do', flush=True)
if pairs.empty:
    sys.exit(0)
pairs = pairs.drop_duplicates('fname')   # one glider profile can be closest to two model times
for name in pairs.glider.unique():
    os.makedirs(f'{save_dir}/{name}', exist_ok=True)


def save(p, mdf):
    best_glider = gliders[(gliders.glider == p.glider) & (gliders.time == p.best_time)]
    best_glider.to_csv(os.path.join(save_dir, p.glider, f"{p.glider}_{p.best_time.strftime('%Y%m%dT%H%M%SZ')}.csv"),
                       index=False)
    mdf.to_csv(p.fname, index=False)


def do_rtofs(group):
    """RTOFS: load one index box for this glider-week, then take the nearest column per pair at the exact model time."""
    ix = np.array([np.abs(rlon - lon).argmin() for lon in group.lon])
    iy = np.array([np.abs(rlat - lat).argmin() for lat in group.lat])
    x0, y0 = ix.min(), iy.min()
    tsel = slice(group.date.min(), group.date.max())
    sub = ds[['temperature', 'salinity']].isel(x=slice(x0, ix.max() + 1), y=slice(y0, iy.max() + 1)).sel(time=tsel).load()
    times = set(pd.DatetimeIndex(sub.time.values))
    n = 0
    for p, i, j in zip(group.itertuples(), ix, iy):
        if p.date not in times:
            continue   # same as the original ds.sel(time=date) KeyError skip
        rds = sub.sel(time=p.date).isel(x=i - x0, y=j - y0)
        save(p, rds.to_dataframe().reset_index())
        n += 1
    return n


def do_rtofs_local(group):
    """RTOFS from the local archive: one file per model time, one box read for all its profiles."""
    date = group.date.iloc[0]
    f = rtofs_file(date)
    if not os.path.exists(f):
        return 0                       # same as the THREDDS KeyError skip
    ix = np.array([np.abs(rlon - lon).argmin() for lon in group.lon])
    iy = np.array([np.abs(rlat - lat).argmin() for lat in group.lat])
    x0, y0 = ix.min(), iy.min()
    with xr.open_dataset(f) as d:
        sub = d[['temperature', 'salinity']].isel(X=slice(x0, ix.max() + 1), Y=slice(y0, iy.max() + 1)).load()
    n = 0
    for p, i, j in zip(group.itertuples(), ix, iy):
        col = sub.isel(MT=0, X=i - x0, Y=j - y0)
        mdf = pd.DataFrame({'depth': col.Depth.values, 'temperature': col.temperature.values,
                            'salinity': col.salinity.values, 'y': int(col.Y), 'x': int(col.X), 'time': date,
                            'Date': float(f'{date:%Y%m%d}') + date.hour / 24,
                            'lat': float(col.Latitude), 'lon': float(col.Longitude)})
        save(p, mdf)
        n += 1
    return n


def do_box(group):
    """CMEMS/ESPC: load one small box for this glider-week, then take the nearest column per pair."""
    lons = to_lon(group.lon.values)
    lo0, lo1 = np.min(lons) - pad, np.max(lons) + pad
    la0, la1 = group.lat.min() - pad, group.lat.max() + pad
    tsel = slice(group.date.min() - pd.Timedelta('6h'), group.date.max() + pd.Timedelta('6h'))
    box = {LON: slice(lo0, lo1), LAT: slice(la0, la1), 'time': tsel, 'depth': slice(0, max_depth)}
    t = temp_da.sel(**box).load()
    s = salt_da.sel(**box).load()
    n = 0
    for p, plon in zip(group.itertuples(), lons):
        pt = {LON: plon, LAT: p.lat, 'time': p.date}
        tp = t.sel(**pt, method='nearest')
        if abs(pd.Timestamp(tp.time.values) - p.date) > tol:
            continue   # model has no field near this time
        sp = s.sel(**pt, method='nearest')
        mdf = pd.DataFrame({'depth': tp.depth.values, 'lat': float(tp[LAT]), 'lon': float(tp[LON]),
                            'time': pd.Timestamp(tp.time.values), 'temperature': tp.values, 'salinity': sp.values})
        if model == 'espc':
            mdf['lon'] = (mdf['lon'] + 180) % 360 - 180
        save(p, mdf)
        n += 1
    return n


if model == 'cmems':
    # Group by the store's own chunks (32 x 32 cells x 526 six-hourly steps, one depth level each):
    # one read fetches every chunk a group needs, however many profiles fall in it. Grouping per
    # glider-week re-read the same chunks ~18 times per glider, and for Argo (one profile per float
    # every ~10 days) it meant ~150 MB per profile.
    enc = temp_da.encoding.get('preferred_chunks') or temp_da.encoding.get('chunks') or {}
    cx, cy, ct = enc.get('longitude', 32), enc.get('latitude', 32), enc.get('time', 526)
    lonv, latv = temp_da[LON].values, temp_da[LAT].values
    tv = pd.DatetimeIndex(temp_da['time'].values)
    ix = np.abs(lonv[None, :] - pairs.lon.values[:, None]).argmin(1) // cx
    iy = np.abs(latv[None, :] - pairs.lat.values[:, None]).argmin(1) // cy
    it = tv.get_indexer(pd.DatetimeIndex(pairs.date), method='nearest') // ct
    pairs['win'] = list(zip(ix, iy, it))
    jobs = [g for _, g in pairs.groupby('win')]
elif LOCAL:
    jobs = [g for _, g in pairs.groupby('date')]
else:
    pairs['win'] = pairs.date.dt.floor(window)
    jobs = [g for _, g in pairs.groupby(['glider', 'win'])]
fn = (do_rtofs_local if LOCAL else do_rtofs) if model == 'rtofs' else do_box

if LOCAL:
    max_workers = 1       # the netCDF library is not thread-safe; local reads are fast anyway
total = 0
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(fn, j): j for j in jobs}
    for f in tqdm(as_completed(futures), total=len(futures), desc=f'{region_str}/{model}', mininterval=60):
        try:
            total += f.result()
        except Exception as e:
            print(f'Job failed: {type(e).__name__} {str(e)[:300]}', flush=True)
print(f'DONE {region_str}/{model}: wrote {total} new profile pairs', flush=True)
