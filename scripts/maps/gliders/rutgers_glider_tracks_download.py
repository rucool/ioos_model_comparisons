#!/usr/bin/env python3
"""
Download the full track history of the RUCOOL/Rutgers glider fleet from
Rutgers' own deployment database API (the same one behind the RU-COOL
Deployments Map at https://marine.rutgers.edu/cool/data/gliders/map/historical/).

This is the authoritative, complete fleet record -- 793 deployments as of
writing, spanning every glider type Rutgers has flown (Slocum *and*
Seaglider), not just what happens to be mirrored on the Slocum ERDDAP.

Run this script first to cache data locally, then use:
  rutgers_glider_tracks_map.py   - global track map (all deployments, all time)
"""
import time
from pathlib import Path

import pandas as pd
import requests

DEPLOYMENTS_URL = 'https://marine.rutgers.edu/cool/data/gliders/api/deployments/?'
TRACK_URL = 'https://marine.rutgers.edu/cool/data/gliders/api/tracks/?deployment='

SAVE_DIR = Path.home() / "Documents" / "data" / "gliders"
SAVE_FILE = SAVE_DIR / "rutgers_glider_tracks_all.parquet"
LOG_FILE = SAVE_DIR / "rutgers_glider_tracks_download.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def list_deployments():
    r = requests.get(DEPLOYMENTS_URL, timeout=60)
    r.raise_for_status()
    return r.json()['data']


def download_track(session, deployment_name):
    """Return a DataFrame of [longitude, latitude] for one deployment, taken
    from the track API's LineString geometry (falls back to any Point
    features, in order, if no LineString is present -- e.g. single-fix
    deployments)."""
    r = session.get(TRACK_URL + deployment_name, timeout=60)
    r.raise_for_status()
    features = r.json().get('features', [])

    coords = []
    for f in features:
        geom = f.get('geometry') or {}
        if geom.get('type') == 'LineString':
            coords.extend(geom['coordinates'])
    if not coords:
        for f in features:
            geom = f.get('geometry') or {}
            if geom.get('type') == 'Point':
                coords.append(geom['coordinates'])

    if not coords:
        return pd.DataFrame(columns=['longitude', 'latitude'])
    return pd.DataFrame(coords, columns=['longitude', 'latitude'])


if __name__ == '__main__':
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text('')

    log("Fetching full deployment list from the RU-COOL deployment database")
    deployments = list_deployments()
    log(f"Found {len(deployments)} deployments\n")

    session = requests.Session()

    all_dfs = {}
    for i, dep in enumerate(deployments):
        name = dep['deployment_name']
        log(f"[{i + 1}/{len(deployments)}] {name}")
        try:
            df = download_track(session, name)
        except Exception as ex:
            log(f"  Skipping {name}: {ex}")
            continue
        if not df.empty:
            all_dfs[name] = df
        time.sleep(0.05)  # be polite to a shared institutional server

    if not all_dfs:
        log("No data downloaded.")
    else:
        combined = pd.concat(all_dfs, names=['glider', 'point'])
        combined.to_parquet(SAVE_FILE)
        log(
            f"\nSaved {len(combined):,} track points from "
            f"{len(all_dfs)} deployments to:\n{SAVE_FILE}"
        )
