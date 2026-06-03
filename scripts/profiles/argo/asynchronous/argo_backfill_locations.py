#!/usr/bin/env python
"""
Backfill locations.json for Argo profiles that already exist as PNGs
in the past-week dated directories but are missing from last_14_days/locations.json.

For each missing entry the script queries the Argo ERDDAP (IFREMER) to recover
the float's lat/lon at that profile time, then creates the symlink and JSON entry.
"""
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import ioos_model_comparisons.configs as conf
from ioos_model_comparisons.platforms import get_argo_floats_by_time
from ioos_model_comparisons.regions import region_config

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
REGIONS = [
    'caribbean',
    'gom',
    'sab',
    'mab',
    'tropical_western_atlantic',
    'hawaii',
    'mexico_pacific',
    'guam',
    'fiji',
]

LOOKBACK_DAYS = 7
DATE_FMT = "%Y-%m-%dT%H:%MZ"

# Matches: {wmo}-profile-{YYYY}-{MM}-{DD}T{HHMM}Z.png
FNAME_RE = re.compile(
    r'^(?P<wmo>\d+)-profile-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2})(?P<minute>\d{2})Z\.png$'
)

save_dir = conf.path_plots / 'profiles' / 'argo'
then = pd.Timestamp.today() - pd.Timedelta(days=14)
then = pd.Timestamp(then.strftime('%Y-%m-%d'))

# -------------------------------------------------------------------


def load_locations(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_locations(path: Path, locations: dict) -> None:
    with open(path, 'w') as f:
        json.dump(locations, f, indent=2)


def process_region(region_key: str) -> None:
    region = region_config(region_key)
    extent = region['extent']   # (lonmin, lonmax, latmin, latmax)
    region_dir = save_dir / region['folder']
    symlink_dir = region_dir / 'last_14_days'
    os.makedirs(symlink_dir, exist_ok=True)

    locations_file = symlink_dir / 'locations.json'
    locations = load_locations(locations_file)

    today = datetime.utcnow().date()
    added = 0
    skipped = 0

    for delta in range(LOOKBACK_DAYS):
        day = today - timedelta(days=delta)
        dated_dir = region_dir / str(day.year) / f'{day.month:02d}' / f'{day.day:02d}'
        if not dated_dir.exists():
            continue

        for png in sorted(dated_dir.glob('*.png')):
            fname = png.name
            # skip difference plots
            if 'difference' in fname:
                continue

            if fname in locations and (symlink_dir / fname).exists():
                skipped += 1
                continue

            m = FNAME_RE.match(fname)
            if not m:
                print(f"  [skip] unrecognised filename: {fname}")
                continue

            wmo = m.group('wmo')
            profile_time = datetime(
                int(m.group('year')),
                int(m.group('month')),
                int(m.group('day')),
                int(m.group('hour')),
                int(m.group('minute')),
            )
            tstr = profile_time.strftime(DATE_FMT)

            # Only keep within the 14-day symlink window
            if pd.Timestamp(profile_time) < then:
                continue

            # Recover lat/lon from ERDDAP if not already in locations.json
            if fname not in locations:
                t_start = profile_time - timedelta(hours=2)
                t_end   = profile_time + timedelta(hours=2)

                try:
                    df = get_argo_floats_by_time(
                        bbox=extent,
                        time_start=t_start,
                        time_end=t_end,
                        wmo_id=wmo,
                    )
                except Exception as exc:
                    print(f"  [error] ERDDAP query failed for {wmo} @ {tstr}: {exc}")
                    continue

                if df.empty:
                    print(f"  [miss]  no ERDDAP data for {wmo} @ {tstr}")
                    continue

                row = df.iloc[0]
                lat = float(row['lat'])
                lon = float(row['lon'])

                locations[fname] = {
                    'lat': lat,
                    'lon': lon,
                    'wmo': str(wmo),
                    'time': tstr,
                }
                print(f"  [add]   {fname}  lat={lat:.4f}  lon={lon:.4f}")

            # Create symlink if missing
            link = symlink_dir / fname
            if not link.exists():
                try:
                    os.symlink(png.resolve(), link)
                except FileExistsError:
                    pass

            added += 1

    write_locations(locations_file, locations)
    print(f"  Region {region_key}: added {added}, skipped {skipped} already-present entries.")


def main():
    for region_key in REGIONS:
        print(f"\nProcessing region: {region_key}")
        try:
            process_region(region_key)
        except Exception as exc:
            print(f"  [error] {region_key}: {exc}")


if __name__ == '__main__':
    main()
