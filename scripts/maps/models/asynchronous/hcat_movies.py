import cv2
from glob import glob
import os
from pathlib import Path
import pandas as pd

# Set main path of data and plot location
pdir = Path("/Users/mikesmith/Documents/plots/glider_map_vs_maps_active_assets_tds")
mname = "puerto_rico-us_virgin_islands"
depths = [100, 150]
# Subdirectory name and depths to concatenate
# mname = "2021_hurricane_season"
# depths = [0, 100, 200]

# mname = "2021_hurricane_season_mab"
# depths = [0, 30, 200] 

variables = ["temperature"]

# Define subdirectories for GOFS and RTOFS
mdir = pdir / mname
sdir = mdir / "comparisons"
gdir = (mdir  / "gofs")
rdir = (mdir / "rtofs")

def files2df(flist, cname="file"):
    df = pd.DataFrame(sorted(flist), columns=[cname])
    df['date'] = df[cname].str.extract(r'(\d{4}-\d{2}-\d{2})')
    return df.set_index('date')

# Loop through depths
for d in depths:
    for v in variables:
        tsdir = sdir / f"{v}_{d}m"
        os.makedirs(tsdir, exist_ok=True)
        gdf = files2df(sorted(glob(str(gdir / str(d) / v / "*.png"))), cname="gofs")
        rdf = files2df(sorted(glob(str(rdir / str(d) / v / "*.png"))), cname="rtofs")
        df = pd.concat([gdf, rdf], axis=1)
        for index, row in df.iterrows():
            img1 = cv2.imread(row["rtofs"])
            img2 = cv2.imread(row["gofs"])
            # horizontally concatenates images of same height 
            im_h = cv2.hconcat([img1, img2])
            
            # show the output image
            cv2.imwrite(f"{tsdir}/{v}_{d}m_{index}.png", im_h)