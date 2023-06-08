import requests
from pathlib import Path
import datetime as dt
from urllib.request import urlretrieve
import os

def download_file(url, file_name, ddir):
    ddir.mkdir(parents=True, exist_ok=True)
    
    # Join the directory with the filename
    file_path = ddir / file_name
    
    # Send a HTTP request to the url of the file we want to access
    response = requests.get(os.path.join(url, file_name), stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # If the request was successful, download the file
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded the file. HTTP Response Code: {response.status_code}")
        # urlretrieve(url, file_path)
    else:
        print(f"Failed to download the file. HTTP Response Code: {response.status_code}")

# The URL of the file we want to download
url = 'http://tendral.com/TOPSv2/latest_composite/'

# Get the current date
now = dt.datetime.now()

# Subtract one day from the current date to get yesterday's date
yesterday = now - dt.timedelta(days=1)
tomorrow = now + dt.timedelta(days=1)


for d in [yesterday, now, tomorrow]:
    # Format the date
    formatted_date_1 = yesterday.strftime('%Y%m%d')
    formatted_date_2 = d.strftime('%Y%m%d')

    # The name we want to give to the downloaded file
    file_name = f'tops_compositem_{formatted_date_1}_{formatted_date_2}.nc'

    # Directory where the file should be downloaded
    ddir = Path('/home/hurricaneadm/data/tops/')
    # ddir = Path('/Users/mikesmith/data/tops/')

    download_file(url, file_name, ddir / d.strftime('%Y/%m'))
