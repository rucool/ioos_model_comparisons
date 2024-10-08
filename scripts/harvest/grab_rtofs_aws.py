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
    # Check if the file has already been downloaded and is not empty
    if destination.exists() and destination.stat().st_size > 0:
        print(f"{destination.name} already exists and is not empty. Skipping download.")
        return
    
    attempt = 0
    while attempt < retries:
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
                return
            else:
                print(f"Download failed or file is incomplete, trying again... {attempt + 1}/{retries}")
                attempt += 1

    raise Exception(f"Failed to download {url} after {retries} attempts.")

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
        try:
            download_file(file_url, file_path)
        except requests.exceptions.HTTPError as e:
            print(f"Failed to download {file_url} with error: {e}")
            continue
    print(f"Completed downloads for {date_str} - {'Prod' if prod else 'Parallel'}")

if __name__ == "__main__":
    days = 5  # Number of days in the past to download data for, including today
    date_strs = generate_date_strs(days)
    for date_str in date_strs:
        download_rtofs_data(date_str, prod=True)
        download_rtofs_data(date_str, prod=False)