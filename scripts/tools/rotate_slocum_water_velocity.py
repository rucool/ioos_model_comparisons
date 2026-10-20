#!/usr/bin/env python3
"""Rotate native Slocum water velocity from magnetic LMC axes to true ENU.

Install:
    python -m pip install numpy xarray netCDF4 pygeomag

Example:
    python rotate_slocum_water_velocity.py input.nc output.nc

The input is never modified. By default, the script rotates
m_final_water_vx/m_final_water_vy and writes water_u_true/water_v_true plus the
calculated magnetic_declination. Variable names can be overridden on the CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from pygeomag import GeoMag


def decimal_year(value: np.datetime64) -> float:
    """Convert numpy datetime64 to decimal UTC year."""
    seconds = value.astype("datetime64[s]").astype(np.int64)
    dt = datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    next_start = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return dt.year + (dt - year_start).total_seconds() / (
        next_start - year_start
    ).total_seconds()


def filled_1d(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Fill internal/edge gaps, as raw glider position can be sparse."""
    if da.ndim != 1 or da.dims[0] != dim:
        raise ValueError(f"{da.name!r} must be one-dimensional on {dim!r}")
    return da.interpolate_na(dim=dim, method="linear").ffill(dim).bfill(dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input glider NetCDF")
    parser.add_argument("output", type=Path, help="New output NetCDF")
    parser.add_argument("--vx", default="m_final_water_vx")
    parser.add_argument("--vy", default="m_final_water_vy")
    parser.add_argument("--latitude", default="latitude")
    parser.add_argument("--longitude", default="longitude")
    parser.add_argument("--time", default="time")
    parser.add_argument("--u-output", default="water_u_true")
    parser.add_argument("--v-output", default="water_v_true")
    parser.add_argument("--declination-output", default="magnetic_declination")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacing output file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.output}. Use --overwrite to replace it."
        )

    ds = xr.open_dataset(args.input, decode_times=True)
    required = [args.vx, args.vy, args.latitude, args.longitude, args.time]
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"Missing required variables: {', '.join(missing)}")

    vx = ds[args.vx]
    vy = ds[args.vy]
    if vx.dims != vy.dims or vx.shape != vy.shape or vx.ndim != 1:
        raise ValueError("vx and vy must be matching one-dimensional arrays")
    dim = vx.dims[0]

    lat = filled_1d(ds[args.latitude].broadcast_like(vx), dim)
    lon = filled_1d(ds[args.longitude].broadcast_like(vx), dim)
    times = ds[args.time].broadcast_like(vx)

    declination_deg = np.full(vx.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(vx.values)
        & np.isfinite(vy.values)
        & np.isfinite(lat.values)
        & np.isfinite(lon.values)
        & ~np.isnat(times.values)
    )

    model = GeoMag()
    for i in np.flatnonzero(valid):
        result = model.calculate(
            glat=float(lat.values[i]),
            glon=float(lon.values[i]),
            alt=0.0,
            time=decimal_year(times.values[i]),
        )
        declination_deg[i] = result.d

    dec = np.deg2rad(declination_deg)
    # Native x/y are magnetic east/north. WMM declination is east-positive.
    u_true = vx.values * np.cos(dec) + vy.values * np.sin(dec)
    v_true = -vx.values * np.sin(dec) + vy.values * np.cos(dec)

    coords = vx.coords
    ds[args.declination_output] = xr.DataArray(
        declination_deg, dims=vx.dims, coords=coords
    )
    ds[args.declination_output].attrs.update(
        long_name="World Magnetic Model declination",
        units="degrees_east",
        comment="Positive east of true north; calculated from position and time.",
    )
    ds[args.u_output] = xr.DataArray(u_true, dims=vx.dims, coords=coords)
    ds[args.u_output].attrs.update(
        standard_name="eastward_sea_water_velocity",
        long_name="True-eastward depth-averaged sea-water velocity",
        units=vx.attrs.get("units", "m s-1"),
        source=f"Rotated from {args.vx} and {args.vy} using WMM declination",
    )
    ds[args.v_output] = xr.DataArray(v_true, dims=vy.dims, coords=coords)
    ds[args.v_output].attrs.update(
        standard_name="northward_sea_water_velocity",
        long_name="True-northward depth-averaged sea-water velocity",
        units=vy.attrs.get("units", "m s-1"),
        source=f"Rotated from {args.vx} and {args.vy} using WMM declination",
    )
    ds.attrs["water_velocity_rotation"] = (
        "Native magnetic-LMC velocity rotated to true east/north using pygeomag WMM"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output)
    ds.close()
    print(f"Wrote {args.output}")
    print(f"Rotated {valid.sum()} velocity observations")


if __name__ == "__main__":
    main()
