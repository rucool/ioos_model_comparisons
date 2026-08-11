import requests
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import os
from datetime import datetime, timedelta
import requests

# Base URL adjustments
prod_url = 'https://noaa-nws-rtofs-pds.s3.amazonaws.com'
para_url = 'https://noaa-nws-rtofs-pds.s3.amazonaws.com/rtofs.parallel.v2.3'

# AWS is the preferred source, but it occasionally lags or is missing files.
# NOMADS mirrors the same rtofs.YYYYMMDD/<file> layout for the operational
# (prod) run and is used as a fallback. There is no NOMADS equivalent of the
# parallel run, so the fallback only applies when prod=True.
nomads_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod'

# Local directory adjustments
prod_ddir = Path('/home/hurricaneadm/data/rtofs/')
para_ddir = Path('/home/hurricaneadm/data/rtofs.parallel.v2.3/')
# prod_ddir = Path('/Users/mikesmith/data/rtofs')
# para_ddir = Path('/Users/mikesmith/data/rtofs.parallel.v2.3/')

# File names to download
fnames2grab = [
    'rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f006_6hrly_hvr_US_west.nc',
    'rtofs_glo_3dz_f012_6hrly_hvr_US_west.nc',
    'rtofs_glo_3dz_f018_6hrly_hvr_US_west.nc',
    'rtofs_glo_3dz_f024_6hrly_hvr_US_west.nc',
    'rtofs_glo_2ds_f006_diag.nc',
    'rtofs_glo_2ds_f012_diag.nc',
    'rtofs_glo_2ds_f018_diag.nc',
    'rtofs_glo_2ds_f024_diag.nc',

]

def generate_date_strs(days=2):
    """
    Generates a list of date strings for the past 'days' days, including today.
    """
    return [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days)]


def download_file(url, destination, retries=3):
    """Attempt to download a single file from `url`.

    Returns True on success, False if every retry failed (e.g. the file
    doesn't exist at this URL) so the caller can try a fallback source.
    """
    # Check if the file has already been downloaded and is not empty
    if destination.exists() and destination.stat().st_size > 0:
        print(f"{destination.name} already exists and is not empty. Skipping download.")
        return True

    attempt = 0
    while attempt < retries:
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))

                with open(destination, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as progress_bar:
                    for chunk in r.iter_content(1024):  # 1 Kibibyte chunks
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            # Verify download size
            downloaded_size = os.path.getsize(destination)
            if downloaded_size == total_size:
                print(f"Successfully downloaded {destination.name}")
                return True
            else:
                print(f"Download failed or file is incomplete, trying again... {attempt + 1}/{retries}")
        except requests.exceptions.HTTPError as e:
            print(f"{url} returned an error: {e}")
            break  # file isn't there / server error - retrying won't help

        attempt += 1

    if destination.exists():
        os.remove(destination)
    return False

def download_rtofs_data(date_str, prod=True):
    date = pd.to_datetime(date_str)
    fstr = date.strftime('%Y%m%d')

    base_url = prod_url if prod else para_url.rstrip('/')
    ddir = prod_ddir if prod else para_ddir

    for fname in fnames2grab:
        sdir = ddir / f"rtofs.{fstr}"
        os.makedirs(sdir, exist_ok=True)
        file_path = sdir / fname
        file_url = f"{base_url}/rtofs.{fstr}/{fname}"

        print(f"Downloading: {file_url} to {file_path}")
        if download_file(file_url, file_path):
            continue

        # AWS didn't have it (or failed) - fall back to NOMADS for prod data.
        if prod:
            fallback_url = f"{nomads_url}/rtofs.{fstr}/{fname}"
            print(f"AWS download failed, falling back to NOMADS: {fallback_url}")
            if not download_file(fallback_url, file_path):
                print(f"Failed to download {fname} for {date_str} from both AWS and NOMADS")
        else:
            print(f"Failed to download {fname} for {date_str} (no NOMADS fallback for the parallel run)")
    print(f"Completed downloads for {date_str} - {'Prod' if prod else 'Parallel'}")

if __name__ == "__main__":
    days = 5  # Number of days in the past to download data for, including today
    date_strs = generate_date_strs(days)
    for date_str in date_strs:
        download_rtofs_data(date_str, prod=True)
        download_rtofs_data(date_str, prod=False)