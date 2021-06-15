#! /usr/bin/env python3

"""
Author: Lori Garzio on 2/22/2021
Last modified: Lori Garzio on 2/22/2021
"""
import datetime as dt
import os
import numpy as np
from scripts.harvest import grab_cmems
from src.common import limits

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
# ncCOP_global = '/home/lgarzio/cmems/global-analysis-forecast-phy-001-024_1565877333169.nc'  # on server
# ncCOP_global = '/Users/garzio/Documents/rucool/hurricane_glider_project/CMEMS/global-analysis-forecast-phy-001-024_1565877333169.nc'  # on local machine

out = '/Users/garzio/Documents/rucool/hurricane_glider_project/CMEMS'
maxdepth = 500
username = 'user'
password = 'pwd'
today = dt.datetime.combine(dt.date.today(), dt.datetime.min.time())
tomorrow = today + dt.timedelta(days=1)
regions = limits('gofs')

# add Atlantic Ocean limits
regions['Atlantic'] = dict()
regions['Atlantic'].update(lonlat=[-60, -10, 8, 45])
regions['Atlantic'].update(code='atl')
regions['Gulf of Mexico'].update(code='gom')
regions['South Atlantic Bight'].update(code='sab')
regions['Mid Atlantic Bight'].update(code='mab')
regions['Caribbean'].update(code='carib')

outdir = os.path.join(out, today.strftime('%Y%m%d'))
os.makedirs(outdir, exist_ok=True)

for region in regions.items():
    if region[1]['name'] == 'carib':
        print('Downloading CMEMS file for: {}'.format(region[0]))
        fname = 'cmems_{}_{}_{}m.nc'.format(region[1]['code'], today.strftime('%Y%m%d'), maxdepth)
        extent = np.add(region[1]['lonlat'], [-1, 1, -1, 1]).tolist()
        grab_cmems.download_ds(outdir, fname, today, tomorrow, extent, maxdepth, username, password)
