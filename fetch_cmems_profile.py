#!/usr/bin/env python
"""
fetch_cmems_profile.py

Scans a glider data directory for *_glider.nc files and downloads the
corresponding CMEMS profile for each one that is missing a *_cmems.nc file.

Saves output as {glider_id}_{date}_cmems.nc alongside the glider file,
matching the format of the existing *_espc.nc files.

Usage:
    python fetch_cmems_profile.py /path/to/data/2026
    python fetch_cmems_profile.py /path/to/data/2026 --depth_max 400 --overwrite

Credentials:
    Run once:  copernicusmarine login
    Or set:    export CMEMS_USERNAME=... CMEMS_PASSWORD=...
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine as cm

CMEMS_T_ID = 'cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i'
CMEMS_S_ID = 'cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i'


def open_cmems(username=None, password=None):
    """Open CMEMS temperature and salinity datasets once (lazy / Zarr)."""
    print('Opening CMEMS temperature dataset...')
    ds_t = cm.open_dataset(
        dataset_id=CMEMS_T_ID,
        username=username,
        password=password,
        chunk_size_limit=0,
        service='arco-geo-series',
    )
    print('Opening CMEMS salinity dataset...')
    ds_s = cm.open_dataset(
        dataset_id=CMEMS_S_ID,
        username=username,
        password=password,
        chunk_size_limit=0,
        service='arco-geo-series',
    )
    return ds_t, ds_s


def extract_profile(ds_t, ds_s, lon, lat, time_dt, depth_max):
    """Select the nearest-neighbour profile from already-open CMEMS datasets."""
    sel = dict(time=time_dt, longitude=lon, latitude=lat, method='nearest')

    temp_da = ds_t['thetao'].sel(**sel).sel(depth=slice(0, depth_max)).squeeze().load()
    sal_da  = ds_s['so'].sel(**sel).sel(depth=slice(0, depth_max)).squeeze().load()

    depth = temp_da.depth.values.astype(float)
    temp  = temp_da.values.astype(float)
    sal   = sal_da.values.astype(float)

    # Mask fill values
    temp[np.abs(temp) > 1e10] = np.nan
    sal[np.abs(sal)   > 1e10] = np.nan

    valid = ~np.isnan(temp) & ~np.isnan(sal)
    depth, temp, sal = depth[valid], temp[valid], sal[valid]

    if len(depth) == 0:
        raise ValueError('No valid CMEMS data at this location/time')

    # Density via TEOS-10
    pressure = gsw.p_from_z(-depth, lat)
    SA       = gsw.SA_from_SP(sal, pressure, lon, lat)
    CT       = gsw.CT_from_t(SA, temp, pressure)
    density  = gsw.rho(SA, CT, pressure)

    return depth, temp, sal, density


def save_profile(out_path, glider_id, depth, temp, sal, density, lon, lat, time_dt):
    """Write a CMEMS profile NetCDF matching the convention of the espc files."""
    ds = xr.Dataset(
        {
            'temperature': xr.DataArray(temp,    dims=['depth'], attrs={'units': 'degrees_C'}),
            'salinity':    xr.DataArray(sal,     dims=['depth'], attrs={'units': 'PSU'}),
            'density':     xr.DataArray(density, dims=['depth'], attrs={'units': 'kg m-3'}),
        },
        coords={
            'depth': xr.DataArray(depth, dims=['depth'], attrs={'units': 'm', 'positive': 'down'}),
            'lon':   float(lon),
            'lat':   float(lat),
            'time':  np.datetime64(pd.Timestamp(time_dt)),
        },
        attrs={
            'model':       'CMEMS',
            'glider_id':   glider_id,
            'description': 'CMEMS profile corresponding to glider location',
            'created':     datetime.utcnow().isoformat(),
        },
    )
    ds.to_netcdf(out_path)


def main():
    parser = argparse.ArgumentParser(
        description='Download CMEMS profiles for every glider NC file in a data directory.'
    )
    parser.add_argument('data_path',
                        help='Root data directory, e.g. .../gulf_gliders/data/2026')
    parser.add_argument('--depth_max', type=float, default=400,
                        help='Maximum depth in metres (default: 400)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-fetch even if a _cmems.nc file already exists')
    args = parser.parse_args()

    username = os.environ.get('CMEMS_USERNAME') or None
    password = os.environ.get('CMEMS_PASSWORD') or None

    glider_files = sorted(Path(args.data_path).rglob('*_glider.nc'))
    if not glider_files:
        print(f'No *_glider.nc files found under {args.data_path}')
        sys.exit(1)

    print(f'Found {len(glider_files)} glider file(s)')

    # Separate files that still need a CMEMS fetch
    todo = []
    skipped = []
    for gf in glider_files:
        out = gf.with_name(gf.name.replace('_glider.nc', '_cmems.nc'))
        if out.exists() and not args.overwrite:
            skipped.append(gf)
        else:
            todo.append((gf, out))

    print(f'  {len(skipped)} already fetched — skipping')
    print(f'  {len(todo)} to fetch')

    if not todo:
        print('\nNothing to do.')
        return

    print(f'\nOpening CMEMS datasets...\n')
    ds_t, ds_s = open_cmems(username, password)

    errors = []
    for gf, out in todo:
        print(f'→ {gf.name}')
        try:
            with xr.open_dataset(gf) as ds_g:
                lon       = float(ds_g['lon'].values)
                lat       = float(ds_g['lat'].values)
                time_dt   = pd.Timestamp(ds_g['time'].values)
                glider_id = ds_g.attrs.get('glider_id', gf.stem.rsplit('_', 2)[0])

            depth, temp, sal, density = extract_profile(
                ds_t, ds_s, lon, lat, time_dt, args.depth_max
            )
            save_profile(out, glider_id, depth, temp, sal, density, lon, lat, time_dt)
            print(f'  saved: {out}')

        except Exception as exc:
            print(f'  ERROR: {exc}')
            errors.append((gf.name, str(exc)))

    n_ok = len(todo) - len(errors)
    print(f'\nDone — {n_ok} saved, {len(errors)} error(s).')
    if errors:
        for name, msg in errors:
            print(f'  {name}: {msg}')


if __name__ == '__main__':
    main()
