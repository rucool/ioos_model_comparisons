#! /usr/bin/env python3

"""
Author: Lori Garzio on 5/5/2021
Last modified: Lori Garzio on 5/5/2021
List gliders available in the IOOS glider DAC for a specified time range and region
"""

import os
import datetime as dt
import pandas as pd
import hurricanes.gliders as gld
from hurricanes import limits

pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


save_dir = '/Users/garzio/Documents/rucool/hurricane_glider_project/gliders'
t0 = dt.datetime(2021, 5, 1)
t1 = dt.datetime(2021, 5, 5)
regions = limits(regions=['mab', 'gom', 'carib', 'sab'])

# add shortened codes for the regions
regions['Gulf of Mexico'].update(code='gom')
regions['South Atlantic Bight'].update(code='sab')
regions['Mid Atlantic Bight'].update(code='mab')
regions['Caribbean'].update(code='carib')

for region in regions.items():
    current_gliders = gld.glider_data(region[1]['lonlat'], t0, t1)
    if len(current_gliders) > 0:
        if save_dir:
            gl_savename = 'gliders_{}_{}-{}.csv'.format(region[1]['code'], t0.strftime('%Y%m%d'), t1.strftime('%Y%m%d'))
            gld.glider_summary(current_gliders, os.path.join(save_dir, gl_savename))
