"""Shared settings for the hurricane-season glider/Argo vs model comparison (see README.md).

Override with environment variables:
    SEASON_PLATFORM  glider (default) or argo
    SEASON_YEAR      default: current year
    SEASON_START     default: <year>-06-01
    GLIDER_OHC_ROOT  default: ~/Documents/glider_ohc_<year>   (data, logs and plots; not in git)
                     Argo runs use <GLIDER_OHC_ROOT>/argo.

Steps 1-3 are platform-agnostic: they read <ROOT>/<year>_<region>_<platform>s.csv, whose platform id
column is called 'glider' for both platforms.
"""
import os

import pandas as pd

PLATFORM = os.environ.get('SEASON_PLATFORM', 'glider').lower()
if PLATFORM not in ('glider', 'argo'):
    raise ValueError(f'SEASON_PLATFORM must be glider or argo, not {PLATFORM!r}')
LABEL = {'glider': 'Glider', 'argo': 'Argo'}[PLATFORM]

YEAR = int(os.environ.get('SEASON_YEAR', pd.Timestamp.now().year))
T0 = pd.Timestamp(os.environ.get('SEASON_START', f'{YEAR}-06-01'))
BASE = os.path.expanduser(os.environ.get('GLIDER_OHC_ROOT', f'~/Documents/glider_ohc_{YEAR}'))
ROOT = BASE if PLATFORM == 'glider' else os.path.join(BASE, PLATFORM)
PLOTS = os.path.join(ROOT, 'plots')

# region key (config/regions/<key>.yml) -> plot title
REGIONS = {'caribbean': 'Caribbean', 'hawaii': 'Hawaii', 'guam': 'Guam', 'mexico_pacific': 'Mexico Pacific'}

# RTOFS on the Rutgers THREDDS is two regional grids. Guam is on neither: its RTOFS comes from the
# binary-converted files on baffin (run_guam_rtofs.sh). Hawaii's west edge is 157.9W.
RTOFS_SOURCE = {'caribbean': 'east', 'hawaii': 'west', 'mexico_pacific': 'west'}


def mexico_pacific_mask(df):
    """Pacific side only: the region box (-126..-92 E) also covers the western Gulf of Mexico,
    so drop north of 18N and east of 100W; and drop north of 32.5N (US-Mexico border)."""
    return ~((df.lat > 18) & (df.lon > -100)) & (df.lat < 32.5)


def profile_csv(region):
    return os.path.join(ROOT, f'{YEAR}_{region}_{PLATFORM}s.csv')


glider_csv = profile_csv   # older name used by steps 1 and the Guam scripts
