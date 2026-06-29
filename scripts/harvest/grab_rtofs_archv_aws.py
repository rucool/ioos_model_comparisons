"""
Download raw RTOFS binary archive files (.a/.b) from the NOAA AWS bucket,
convert them to fixed z-level NetCDFs, and organize by date.

Downloads the production run for the most recent available day (f06, f12,
f18, f24), then processes each forecast hour into either a global NetCDF
or per-region NetCDFs (or both).

Output directory structure:
    rtofs_global/
        YYYY/
            YYYYMMDD/
                rtofs_glo_YYYYMMDDTHH_global.nc
                rtofs_glo_YYYYMMDDTHH_hawaii.nc
                rtofs_glo_YYYYMMDDTHH_guam.nc
                ...

Files are named by their valid time (parsed from the .b header), not by
the forecast hour label, so they sort chronologically.

Usage:
    # Global only (default)
    python3 scripts/harvest/grab_rtofs_archv_aws.py

    # Global + specific regions
    python3 scripts/harvest/grab_rtofs_archv_aws.py --regions hawaii guam south_africa

    # Regions only, no global
    python3 scripts/harvest/grab_rtofs_archv_aws.py --regions hawaii guam --no-global

    # Keep the raw .a/.b files after processing
    python3 scripts/harvest/grab_rtofs_archv_aws.py --keep-binary

    # Custom output directory
    python3 scripts/harvest/grab_rtofs_archv_aws.py --output-dir /data/rtofs_global

    # Custom grid/depth file location
    python3 scripts/harvest/grab_rtofs_archv_aws.py --grid-dir /data/rtofs_static
"""
import argparse
import os
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

from ioos_model_comparisons.hycom.rtofs_binary import (
    process_rtofs_region,
    read_archive_time,
)

BASE_URL = "https://noaa-nws-rtofs-pds.s3.amazonaws.com"
FORECAST_HOURS = ["f06", "f12", "f18", "f24"]


def download_file(url, destination, retries=3):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  {destination.name} already exists. Skipping.")
        return True

    attempt = 0
    while attempt < retries:
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(destination, "wb") as f, tqdm(
                    total=total_size, unit="B", unit_scale=True,
                    desc=f"  {destination.name}",
                ) as bar:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                        bar.update(len(chunk))

                if total_size and os.path.getsize(destination) != total_size:
                    print(f"  Incomplete download, retrying... ({attempt + 1}/{retries})")
                    attempt += 1
                    continue
                return True
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}, retrying... ({attempt + 1}/{retries})")
            attempt += 1

    print(f"  Failed after {retries} attempts: {url}")
    if destination.exists():
        destination.unlink()
    return False


def extract_tgz(tgz_path, output_dir):
    print(f"  Extracting {tgz_path.name}...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    tgz_path.unlink()


def find_latest_date(max_lookback=5):
    for days_ago in range(max_lookback):
        date = datetime.utcnow() - timedelta(days=days_ago)
        date_str = date.strftime("%Y%m%d")
        test_url = f"{BASE_URL}/rtofs.{date_str}/rtofs_glo.t00z.{FORECAST_HOURS[0]}.archv.b"
        try:
            r = requests.head(test_url, timeout=10)
            if r.status_code == 200:
                print(f"Latest available date: {date_str}")
                return date_str
        except requests.exceptions.RequestException:
            continue
    raise RuntimeError(f"No RTOFS archive data found in the last {max_lookback} days")


def process_archive(archv_file, grid_file, depth_file, output_dir,
                    do_global, regions, n_jobs):
    valid_time = read_archive_time(archv_file)
    time_str = valid_time.strftime("%Y%m%dT%H")
    date_str = valid_time.strftime("%Y%m%d")
    year_str = valid_time.strftime("%Y")
    month_str = valid_time.strftime("%m")

    day_dir = output_dir / year_str / month_str / date_str
    os.makedirs(day_dir, exist_ok=True)

    encoding_fn = lambda ds: {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for v in ds.data_vars
    }

    if do_global:
        out_file = day_dir / f"rtofs_glo_{time_str}_global.nc"
        if out_file.exists():
            print(f"  {out_file.name} already exists. Skipping.")
        else:
            t0 = time.time()
            print(f"  Processing global -> {out_file.name}")
            ds = process_rtofs_region(
                archv_file=str(archv_file),
                grid_file=str(grid_file),
                depth_file=str(depth_file),
                whole_globe=True,
                n_jobs=n_jobs,
                method="linear",
            )
            if out_file.exists():
                out_file.unlink()
            ds.to_netcdf(out_file, encoding=encoding_fn(ds))
            del ds
            print(f"  Wrote {out_file.name} in {time.time() - t0:.0f}s")

    for region_name in regions:
        out_file = day_dir / f"rtofs_glo_{time_str}_{region_name}.nc"
        if out_file.exists():
            print(f"  {out_file.name} already exists. Skipping.")
            continue
        t0 = time.time()
        print(f"  Processing {region_name} -> {out_file.name}")
        ds = process_rtofs_region(
            archv_file=str(archv_file),
            grid_file=str(grid_file),
            depth_file=str(depth_file),
            region_name=region_name,
            n_jobs=n_jobs,
            method="linear",
        )
        if out_file.exists():
            out_file.unlink()
        ds.to_netcdf(out_file, encoding=encoding_fn(ds))
        del ds
        print(f"  Wrote {out_file.name} in {time.time() - t0:.0f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process RTOFS binary archives from AWS",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/Users/mikesmith/Downloads/rtofs_global"),
        help="Root output directory (default: /Users/mikesmith/Downloads/rtofs_global)",
    )
    parser.add_argument(
        "--grid-dir", type=Path, default=Path("/Users/mikesmith/Downloads/rtofs"),
        help="Directory containing regional.grid.a/.b and regional.depth.a/.b",
    )
    parser.add_argument(
        "--regions", nargs="*", default=[],
        help="Region names to process (from region_config), e.g. hawaii guam gom",
    )
    parser.add_argument(
        "--no-global", action="store_true",
        help="Skip the global file (only produce regional files)",
    )
    parser.add_argument(
        "--keep-binary", action="store_true",
        help="Keep the raw .a/.b files after processing (default: delete them)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of CPU cores for vertical interpolation (-1 = all, default: -1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    grid_dir = args.grid_dir
    grid_file = grid_dir / "regional.grid.a"
    depth_file = grid_dir / "regional.depth.a"
    do_global = not args.no_global

    if not grid_file.exists() or not depth_file.exists():
        raise FileNotFoundError(
            f"Grid/depth files not found in {grid_dir}. "
            "Set --grid-dir to the directory containing regional.grid.a and regional.depth.a"
        )

    tmp_dir = output_dir / ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    date_str = find_latest_date()
    day_url = f"{BASE_URL}/rtofs.{date_str}"

    for fhr in FORECAST_HOURS:
        base = f"rtofs_glo.t00z.{fhr}.archv"
        print(f"\n{'='*60}")
        print(f"{base}")
        print(f"{'='*60}")

        b_path = tmp_dir / f"{base}.b"
        a_path = tmp_dir / f"{base}.a"
        a_tgz_path = tmp_dir / f"{base}.a.tgz"

        # Download .b
        if not download_file(f"{day_url}/{base}.b", b_path):
            print(f"  Skipping {fhr} — .b download failed")
            continue

        # Download and extract .a.tgz
        if not (a_path.exists() and a_path.stat().st_size > 0):
            if download_file(f"{day_url}/{base}.a.tgz", a_tgz_path):
                extract_tgz(a_tgz_path, tmp_dir)
            else:
                print(f"  Skipping {fhr} — .a download failed")
                continue

        # Process
        process_archive(
            a_path, grid_file, depth_file, output_dir,
            do_global=do_global, regions=args.regions, n_jobs=args.n_jobs,
        )

        # Clean up binary files
        if not args.keep_binary:
            for f in [a_path, b_path]:
                if f.exists():
                    f.unlink()
                    print(f"  Removed {f.name}")

    # Clean up tmp dir if empty
    if tmp_dir.exists() and not list(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print(f"\nDone. Output in {output_dir}")


if __name__ == "__main__":
    main()
