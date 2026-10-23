"""Shared settings for the hurricane-season glider vs model comparison (see README.md).

Override with environment variables:
    SEASON_YEAR      default: current year
    SEASON_START     default: <year>-06-01
    GLIDER_OHC_ROOT  default: ~/Documents/glider_ohc_<year>   (data, logs and plots; not in git)
"""
import os

import pandas as pd

YEAR = int(os.environ.get('SEASON_YEAR', pd.Timestamp.now().year))
T0 = pd.Timestamp(os.environ.get('SEASON_START', f'{YEAR}-06-01'))
ROOT = os.path.expanduser(os.environ.get('GLIDER_OHC_ROOT', f'~/Documents/glider_ohc_{YEAR}'))
PLOTS = os.path.join(ROOT, 'plots')

# region key (config/regions/<key>.yml) -> plot title
REGIONS = {'caribbean': 'Caribbean', 'hawaii': 'Hawaii', 'guam': 'Guam', 'mexico_pacific': 'Mexico Pacific'}

# RTOFS on the Rutgers THREDDS is two regional grids. Guam is on neither: its RTOFS comes from the
# binary-converted files on baffin (run_guam_rtofs.sh). Hawaii's west edge is 157.9W.
RTOFS_SOURCE = {'caribbean': 'east', 'hawaii': 'west', 'mexico_pacific': 'west'}


def glider_csv(region):
    return os.path.join(ROOT, f'{YEAR}_{region}_gliders.csv')
