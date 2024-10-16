import os
from ftplib import FTP, error_perm
import datetime as dt
from tqdm import tqdm
from pathlib import Path

ddir = Path('/Users/mikesmith/Documents/data/rtofs/')
# ddir = Path('/home/hurricaneadm/data/rtofs')

url = 'ftp.ncep.noaa.gov'
rtofs_dir = 'pub/data/nccf/com/rtofs/'

fnames2grab = [
    'rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',
    'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc'
    ]

def download(remote_fname, local_fname, remote_size):
    print(f'Downloading {remote_fname}')
    with open(local_fname, 'wb') as local_file:
        with tqdm(total=remote_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def cb(data):
                pbar.update(len(data))
                local_file.write(data)

            ftp.retrbinary('RETR {}'.format(f), cb)
    print(f'Filename: {local_fname} Size: {os.stat(local_fname).st_size}')

# Login to ftp file
ftp = FTP(url)
ftp.login()

# %% load RTOFS nc files
print('Attempting to download RTOFS NetCDF files from FTP server')
  
ftp.cwd(rtofs_dir) #prod/rtofs.{}'.format(tstr))

# The rtofs directory contains multiple versions. Lets grab them all 
# for version in ftp.nlst(): #When different versions are available
for version in ['prod']:
    # print(f"Saving RTOFS {version}")
    print(f"CD into {version}")
    ftp.cwd(version)

    # The rtofs data only exists for 2 days at a time. Let's grab them both
    for date in ftp.nlst():
        # sdir = ddir / version / date
        sdir = ddir / date
        os.makedirs(sdir, exist_ok=True)

        if ftp.nlst(date):
            print(f"CD into {date}")
            ftp.cwd(date)
            print(f"CWD: {ftp.pwd()}")

            # Download only the files specified by fnames2grab
            for f in fnames2grab:
                remote_size = ftp.size(f)
                fname = sdir / f

                # If the file exists, let's run through some checks to make sure
                # that we don't have corrupt files downloaded on the file server.
                if os.path.isfile(fname):
                    local_size = os.stat(fname).st_size

                    # If the local size is equal to the remote size.
                    # Don't download. 
                    if local_size == remote_size:
                        print(f'{f}: Local ({fname})/Remote (ftp) - {local_size}/{ftp.size(f)}. File sizes match')
                        continue
                    # If the local size is not equal to the remote size. The local 
                    # file may be corrupt. Re-download.
                    else:
                        print(f'{f}: Local({fname})/Remote (ftp)  - {local_size}/{ftp.size(f)}. File may be corrupt. Re-downloading.')
                        os.remove(fname)
                        download(f, fname, remote_size)
                else:
                    print(f'{fname} not on local server. Downloading now')
                    download(f, fname, remote_size)
            ftp.cwd('..')
            print(f"CWD: {ftp.pwd()}")
    ftp.cwd('..')
ftp.quit()
