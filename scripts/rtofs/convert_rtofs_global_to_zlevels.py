"""
Convert a raw RTOFS global archive (.a/.b) to a fixed z-level NetCDF on the
native RTOFS grid (no horizontal crop or regrid).

Expects the archive file plus regional.grid.a/.b and regional.depth.a/.b in
the same directory. Edit the paths below before running.

Uses method='linear' to match NOAA's archv2ncdf3z tool (and therefore the
RTOFS OPeNDAP product served by ioos_model_comparisons.models.rtofs()).

Run directly:
    python3 scripts/rtofs/convert_rtofs_global_to_zlevels.py

Or keep running after closing the terminal:
    nohup python3 scripts/rtofs/convert_rtofs_global_to_zlevels.py \
        > /tmp/rtofs_global.log 2>&1 &
    tail -f /tmp/rtofs_global.log
"""
import os
import time

from ioos_model_comparisons.hycom.rtofs_binary import process_rtofs_region

rtofs_dir = "/Users/mikesmith/Downloads/rtofs"
archv_file = f"{rtofs_dir}/rtofs_glo.t00z.f06.archv.a"
grid_file = f"{rtofs_dir}/regional.grid.a"
depth_file = f"{rtofs_dir}/regional.depth.a"
out_file = f"{rtofs_dir}/rtofs_glo.t00z.f06_global_zlevels.nc"


def main():
    start = time.time()
    ds = process_rtofs_region(
        archv_file=archv_file,
        grid_file=grid_file,
        depth_file=depth_file,
        whole_globe=True,
        n_jobs=-1,
        method='linear',
    )
    print(f"Processing done in {time.time() - start:.1f}s, writing netcdf...")
    if os.path.exists(out_file):
        os.remove(out_file)
    encoding = {
        v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
        for v in ds.data_vars
    }
    ds.to_netcdf(out_file, encoding=encoding)
    print(f"Wrote {out_file}, total elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
