import os
from ftplib import FTP, error_perm
import datetime as dt
from tqdm import tqdm

# ldir = '/Users/mikesmith/Documents/github/rucool/hurricanes/data/rtofs/'

ldir = '/home/hurricaneadm/data/rtofs'
url = 'ftp.ncep.noaa.gov'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

today = dt.date.today()
yesterday = today - dt.timedelta(days=1)

today = today.strftime('%Y%m%d')
yesterday = yesterday.strftime('%Y%m%d')

time_list = [yesterday, today]


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
for tstr in time_list:
    new_dir = os.path.join(ldir, f'rtofs.{tstr}')
    os.makedirs(new_dir, exist_ok=True)
    ftp.cwd('/')
    try:
        ftp.cwd('pub/data/nccf/com/rtofs/prod/rtofs.{}'.format(tstr))
    except error_perm:
        print(f'Error: {tstr} not on ftp server. Adding newest time to time_list')
        continue

    for f in nc_files_RTOFS:
        # Download nc files
        fname = os.path.join(new_dir, f)
        remote_size = ftp.size(f)
        if os.path.isfile(fname):
            local_size = os.stat(fname).st_size
            if local_size == remote_size:
                print(f'{f}: Local ({fname})/Remote (ftp) - {local_size}/{ftp.size(f)}. File sizes match')
                continue
            else:
                print(f'{f}: Local({fname})/Remote (ftp)  - {local_size}/{ftp.size(f)}. File may be corrupt. Re-downloading.')
                os.remove(fname)
                download(f, fname, remote_size)
        else:
            print(f'{fname} not on local server. Downloading now')
            download(f, fname, remote_size)
ftp.quit()
