"""
Example: process a raw RTOFS .a/.b archive file into a regional, fixed
z-level NetCDF, either on RTOFS's native grid or regridded onto a uniform
lon/lat grid.

Requires a local archive file plus the matching `regional.grid.a/.b` and
`regional.depth.a/.b` files in the same directory.

Uses `n_jobs=-1` below, which runs the vertical interpolation step across
multiple processes -- on macOS/Windows (the `multiprocessing` "spawn" start
method) this requires the `if __name__ == "__main__":` guard at the bottom
of this file, otherwise each worker process re-imports and re-runs this
whole script.
"""
import time

import numpy as np

from ioos_model_comparisons.hycom.rtofs_binary import process_rtofs_region

rtofs_dir = "/Users/mikesmith/Downloads/rtofs"
archv_file = f"{rtofs_dir}/rtofs_glo.t00z.f06.archv.a"
grid_file = f"{rtofs_dir}/regional.grid.a"
depth_file = f"{rtofs_dir}/regional.depth.a"

# z_levels defaults to RTOFS_STANDARD_Z_LEVELS, matching the depths served
# by ioos_model_comparisons.models.rtofs() (the OPeNDAP "scraped" product).


def main():
    # Crop only, native RTOFS grid. process_rtofs_region shows a tqdm
    # progress bar per field while it interpolates onto z-levels (the slow
    # step). n_jobs=-1 uses all CPU cores for that step.
    start = time.time()
    ds_native = process_rtofs_region(
        archv_file=archv_file,
        grid_file=grid_file,
        depth_file=depth_file,
        region_name="guam",
        n_jobs=-1,
    )
    ds_native.to_netcdf("guam_rtofs_f06_native_zlevels.nc")
    print(f"Done in {time.time() - start:.1f}s")

    # # Also regrid horizontally onto a uniform 0.04deg grid.
    # target_grid = dict(
    #     lon=np.arange(-98, -80, 0.04),
    #     lat=np.arange(18, 31, 0.04),
    # )
    # ds_uniform = process_rtofs_region(
    #     archv_file=archv_file,
    #     grid_file=grid_file,
    #     depth_file=depth_file,
    #     region_name="gom",
    #     target_grid=target_grid,
    #     n_jobs=-1,
    # )
    # ds_uniform.to_netcdf("gom_rtofs_f06_uniform_zlevels.nc")

    # # The whole globe, on the native grid (no horizontal crop or regrid).
    # # Expect ~15-20 minutes with n_jobs=-1 on a 10-core machine (vs.
    # # ~80-90 minutes single-process), and a sizeable output file (~10GB
    # # for 4 fields x 40 z-levels x the full 3298x4500 grid as float32).
    # ds_global = process_rtofs_region(
    #     archv_file=archv_file,
    #     grid_file=grid_file,
    #     depth_file=depth_file,
    #     whole_globe=True,
    #     n_jobs=-1,
    # )
    # ds_global.to_netcdf("rtofs_f06_global_zlevels.nc")


if __name__ == "__main__":
    main()
