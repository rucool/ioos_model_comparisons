from glob import glob
import os
import shutil

gdir = "/Volumes/rucool/hurricane/Hurricane_season_2021"
sdir = "/Users/mikesmith/Documents/plots/profiles/"
glider = "SG663-20210615T1202"

gsdir = os.path.join(sdir, glider)
os.makedirs(gsdir, exist_ok=True)

for f in sorted(glob(os.path.join(gdir, f"*/{glider}*.png"), recursive=True)):
    fname = os.path.basename(f)
    shutil.copy(f, os.path.join(gsdir, fname))
