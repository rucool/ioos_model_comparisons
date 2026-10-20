#!/usr/bin/env python3
"""
Download Gulf of Mexico glider tracks from the IOOS National Glider DAC.
Covers all deployments within the past 20 years.

Run this script first to cache data locally, then use:
  gulf_glider_tracks_map.py   - individual glider track map
  gulf_glider_hexbin_map.py   - observation density heatmap
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from erddapy import ERDDAP

# Gulf of Mexico bounding box [min_lon, max_lon, min_lat, max_lat]
BBOX = [-99, -81, 17.5, 31.5]

# 20-year window
T1 = dt.datetime.utcnow()
T0 = T1 - dt.timedelta(days=365 * 20)

SAVE_DIR = Path.home() / "Documents" / "data" / "gliders"
SAVE_FILE = SAVE_DIR / "gulf_glider_tracks_20yr.parquet"


def search_datasets(e, t0_str, t1_str):
    kw = {
        'min_lon': BBOX[0], 'max_lon': BBOX[1],
        'min_lat': BBOX[2], 'max_lat': BBOX[3],
        'min_time': t0_str, 'max_time': t1_str,
    }
    search_url = e.get_search_url(response='csv', **kw)
    search = pd.read_csv(search_url)
    return search['Dataset ID'].values


def normalize_columns(df):
    """Strip units from column names, e.g. 'latitude (degrees_north)' -> 'latitude'."""
    return df.rename(columns={c: c.split(' ')[0] for c in df.columns})


def download_track(e, dataset_id, t0_str, t1_str):
    e.dataset_id = dataset_id
    e.variables = ['time', 'latitude', 'longitude']
    e.constraints = {
        'time>=': t0_str, 'time<=': t1_str,
        'longitude>=': BBOX[0], 'longitude<=': BBOX[1],
    }
    try:
        df = e.to_pandas(
            response='csv',
            index_col='time',
            parse_dates=True,
            skiprows=(1,),
        ).dropna().tz_localize(None)
        df = normalize_columns(df)
        df.index.name = 'time'
        return df
    except Exception as ex:
        print(f"  Skipping {dataset_id}: {ex}")
        return pd.DataFrame()


if __name__ == '__main__':
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    t0_str = T0.strftime('%Y-%m-%dT%H:%M:%SZ')
    t1_str = T1.strftime('%Y-%m-%dT%H:%M:%SZ')

    e = ERDDAP(server='NGDAC', protocol='tabledap', response='csv')
    e.requests_kwargs['timeout'] = 60

    print(f"Searching NGDAC for Gulf of Mexico gliders\n  {t0_str} to {t1_str}")
    ids = search_datasets(e, t0_str, t1_str)
    print(f"Found {len(ids)} datasets")

    # When both real-time and delayed-mode exist for the same deployment,
    # keep only the delayed-mode (quality-controlled) version.
    delayed_bases = {i.replace('-delayed', '') for i in ids if 'delayed' in i}
    ids_filtered = [i for i in ids if i not in delayed_bases]
    print(f"After preferring delayed-mode: {len(ids_filtered)} datasets\n")

    all_dfs = {}
    for i, gid in enumerate(ids_filtered):
        print(f"[{i + 1}/{len(ids_filtered)}] {gid}")
        df = download_track(e, gid, t0_str, t1_str)
        if not df.empty:
            all_dfs[gid] = df

    if not all_dfs:
        print("No data downloaded.")
    else:
        combined = pd.concat(all_dfs, names=['glider', 'time'])
        combined.to_parquet(SAVE_FILE)
        print(
            f"\nSaved {len(combined):,} observations from "
            f"{len(all_dfs)} deployments to:\n{SAVE_FILE}"
        )
