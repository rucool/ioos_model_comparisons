#!/usr/bin/env python3
"""
Step 1 of the glider-coverage analysis: download and standardize inputs.

Writes daily 2-D fields to <workdir>/<region>/<model>/, on a common regular
1/12 deg grid over the region's bbox (gc_regions.py), so every later step treats
the models identically. One file per variable group (--vars):

    temp -> fields.nc + mask.nc
             sst   temperature at the top model level
             t100  temperature near 100 m (CMEMS 109.7 m, RTOFS 100 m)
             t200  temperature near 200 m (CMEMS 186.1 m, RTOFS 200 m)
             heat  thickness-weighted mean temperature, surface to t200's depth
                   (upper-ocean heat content proxy)
             ssh   sea surface height (CMEMS only; not in the RTOFS aggregation)
           mask.nc: wet / deep200 / deep1000 booleans from one full-depth snapshot
    salt -> fields_salt.nc
             sss, s100, s200, salt (thickness-weighted mean salinity, surface to ~200 m)
    uv   -> fields_uv.nc
             u200/v200, u500/v500, u1000/v1000: thickness-weighted mean eastward /
             northward current from the surface to ~200, ~500 and ~1000 m (NaN where
             the bottom is shallower). These match glider depth-averaged currents,
             which average over the glider's dive depth.

    ssh  -> fields_ssh.nc (RTOFS from --rtofs-dir only; CMEMS sea surface height is in the temp group)
             ssh   sea surface height from the RTOFS 2-D diagnostic files

Models:
    cmems  Copernicus GLO 1/12 deg analysis, daily means (needs CMEMS_USERNAME /
           CMEMS_PASSWORD in the shell or repo .env). Raw subsets are kept in
           <workdir>/<region>/cmems/raw/ (~2.7 GB per 0-200 m variable for 2 yr in
           the Gulf, ~4 GB per current component to 1000 m) and reused if present.
    espc   ESPC-D-V02 (Navy global HYCOM with NCODA assimilation), 00Z snapshot each day
           from the HYCOM THREDDS yearly aggregations, on a reduced set of depth levels
           (see ESPC_DEPTHS), interpolated from its 0.08 x 0.04 deg grid.
    rtofs  RTOFS US East, 00Z snapshot each day, linearly interpolated from its
           0.08 deg Mercator grid. Read from the Rutgers THREDDS aggregation
           (models.rtofs()) by default, or, much faster, from a local archive of
           NCEP files with --rtofs-dir (e.g. /Volumes/hurricaneadm/data/rtofs),
           laid out as rtofs.YYYYMMDD/rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc.
           The 00Z field for day D is the f024 forecast from day D-1's cycle,
           which is exactly what the THREDDS aggregation serves for that time.

Then run scales.py, fleet_coverage.py, figure.py, validate_obs.py with the same --model and --region.

Usage:
    python scripts/tools/glider_coverage/download.py --model cmems
    python scripts/tools/glider_coverage/download.py --model rtofs --start 2024-10-01 --end 2026-09-30
    python scripts/tools/glider_coverage/download.py --model rtofs --region caribbean_pr --vars salt uv
    python scripts/tools/glider_coverage/download.py --model rtofs --vars salt uv ssh --rtofs-dir /Volumes/hurricaneadm/data/rtofs
"""
import argparse
import os
import sys

# HDF5 file locking fails on network shares (e.g. the RTOFS archive on the fileserver) when more
# than one reader opens the same file; the files are only read, so locking is not needed.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gc_regions import REGIONS, grid
from ioos_model_comparisons import configs

DEFAULT_WORKDIR = configs.path_data / "glider_coverage"


def depth_mean(da, zmax):
    """Thickness-weighted mean from the surface to zmax (NaN where the bottom is shallower)."""
    da = da.sel(depth=slice(None, zmax))
    return da.integrate("depth") / float(da.depth[-1] - da.depth[0])


def level(da, z):
    return da.sel(depth=z, method="nearest").drop_vars("depth")


# Variable groups: which model variables to fetch, how deep, the valid range used to
# catch corrupted days, and how to reduce the 3-D profiles to the stored 2-D fields.
# CMEMS levels are not round numbers, so its levels/limits are chosen per model.
GROUPS = {
    "temp": dict(
        file="fields.nc", valid=(-3, 40), zmax=200,
        rtofs=["temperature"], cmems=("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"], "thetao.nc"),
        espc=[("t3z", "water_temp")],
        make=lambda P, z: dict(sst=P[0].isel(depth=0, drop=True), t100=level(P[0], z["d100"]), t200=level(P[0], z["d200"]),
                               heat=depth_mean(P[0], z["h"])),
    ),
    "salt": dict(
        file="fields_salt.nc", valid=(0, 45), zmax=200,
        rtofs=["salinity"], cmems=("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ["so"], "so.nc"),
        espc=[("s3z", "salinity")],
        make=lambda P, z: dict(sss=P[0].isel(depth=0, drop=True), s100=level(P[0], z["d100"]), s200=level(P[0], z["d200"]),
                               salt=depth_mean(P[0], z["h"])),
    ),
    "uv": dict(
        file="fields_uv.nc", valid=(-5, 5), zmax=1000,
        rtofs=["u", "v"], cmems=("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"], "cur.nc"),
        espc=[("u3z", "water_u"), ("v3z", "water_v")],
        make=lambda P, z: {f"{c}{H}": depth_mean(P[k], zz) for k, c in enumerate("uv") for H, zz in
                           ((200, z["h"]), (500, z["h500"]), (1000, z["h1000"]))},
    ),
    "ssh": dict(
        file="fields_ssh.nc", valid=(-5, 5), zmax=None,
        rtofs=["ssh"], rtofs_file="rtofs_glo_2ds_f024_diag.nc", cmems=None, espc=[("ssh", "surf_el")],
        make=lambda P, z: dict(ssh=P[0]),
    ),
}
# Depth choices per model: d100/d200 = levels, h/h500/h1000 = depth-mean limits.
LEVELS = {
    "rtofs": dict(d100=100, d200=200, h=200, h500=500, h1000=1000),
    "cmems": dict(d100=109.7, d200=186.1, h=190, h500=545, h1000=1065),
    "espc": dict(d100=100, d200=200, h=200, h500=500, h1000=1000),
}
# ESPC is read from HYCOM's THREDDS server, where request time grows with the number of depth
# levels (~5 s/level per day), so it uses a subset of its 40 z-levels. Tested on RTOFS (same
# levels): the RMS change in the depth means is ~0.02 C, ~0.003 salinity, ~0.001 m/s.
ESPC_DEPTHS = {200: [0, 10, 20, 30, 50, 70, 100, 125, 150, 200],
               1000: [0, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
ESPC_URL = "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/{product}/{year}"
ATTRS = {
    "rtofs": dict(model="RTOFS (HYCOM, US East)", sampling="00Z snapshot", t100_depth="100 m", t200_depth="200 m",
                  heat_depth="0-200 m", s100_depth="100 m", s200_depth="200 m", salt_depth="0-200 m",
                  uv_depths="0-200 m, 0-500 m, 0-1000 m"),
    "cmems": dict(model="CMEMS GLO 1/12°", sampling="daily mean", t100_depth="109.7 m", t200_depth="186.1 m",
                  heat_depth="0-186 m", s100_depth="109.7 m", s200_depth="186.1 m", salt_depth="0-186 m",
                  uv_depths="0-186 m, 0-541 m, 0-1062 m"),
    "espc": dict(model="ESPC-D-V02 (Navy HYCOM/NCODA)", sampling="00Z snapshot", t100_depth="100 m", t200_depth="200 m",
                 heat_depth="0-200 m", s100_depth="100 m", s200_depth="200 m", salt_depth="0-200 m",
                 uv_depths="0-200 m, 0-500 m, 0-1000 m"),
}


def write(out, group, fields, attrs, mask=None):
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(fields).astype("float32")
    ds.attrs.update(attrs)
    f = out / GROUPS[group]["file"]
    ds.to_netcdf(f)
    if mask is not None:
        xr.Dataset(mask).to_netcdf(out / "mask.nc")
    print(f"wrote {f} {dict(ds.sizes)}")


def clean(fields, valid, max_gap_days=5):
    """Blank days the archive served corrupted, then fill short gaps in time.

    A handful of RTOFS days come back with unmasked fill values (~1e30) or with
    land not masked. A day is bad if any value is outside the group's valid range
    or its count of valid cells differs from the typical day by more than 1%.
    """
    lo, hi = valid
    for k, da in fields.items():
        v = da.values.reshape(da.sizes["time"], -1)
        n_valid = np.isfinite(v).sum(1)
        with np.errstate(all="ignore"):
            bad = (np.nanmax(v, axis=1) > hi) | (np.nanmin(v, axis=1) < lo) | \
                  (np.abs(n_valid - np.median(n_valid)) > 0.01 * np.median(n_valid))
        if bad.any():
            print(f"  {k}: {bad.sum()} bad day(s) interpolated: {[str(t)[:10] for t in da.time.values[bad]]}")
        da = da.where(~xr.DataArray(bad, dims="time", coords={"time": da.time}))
        typical = np.isfinite(da.isel(time=np.flatnonzero(~bad)[0]))  # land mask from a good day
        fields[k] = da.interpolate_na("time", max_gap=pd.Timedelta(days=max_gap_days)).where(typical)
    return fields


# --- CMEMS ----------------------------------------------------------------
def cmems(args, out, group):
    import copernicusmarine as cm
    from ioos_model_comparisons.env import load_env

    G = GROUPS[group]
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = REGIONS[args.region]["bbox"]
    LON, LAT = grid(args.region)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dataset_id, variables, raw_name = G["cmems"]
    need = [f for f in ([raw_name] + (["zos.nc", "mask_full_depth.nc"] if group == "temp" else [])) if not (raw / f).exists()]
    if need:
        load_env()
        user, pw = os.getenv("CMEMS_USERNAME"), os.getenv("CMEMS_PASSWORD")
        if not user or not pw:
            # Fall back to the credentials models.CMEMS ships with, as the rest of the repo does.
            import inspect
            from ioos_model_comparisons.models import CMEMS
            sig = inspect.signature(CMEMS.__init__).parameters
            user, pw = user or sig["username"].default, pw or sig["password"].default
        common = dict(minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
                      minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
                      username=user, password=pw, output_directory=str(raw), overwrite=True)
        if raw_name in need:
            cm.subset(dataset_id=dataset_id, variables=variables, start_datetime=args.start, end_datetime=args.end,
                      minimum_depth=0, maximum_depth=G["zmax"] + 100, output_filename=raw_name, **common)
        if "zos.nc" in need:
            cm.subset(dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m", variables=["zos"],
                      start_datetime=args.start, end_datetime=args.end, output_filename="zos.nc", **common)
        if "mask_full_depth.nc" in need:
            cm.subset(dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", variables=["thetao"],
                      start_datetime=args.mask_date, end_datetime=args.mask_date, minimum_depth=0,
                      maximum_depth=1100, output_filename="mask_full_depth.nc", **common)

    on_grid = dict(latitude=LAT, longitude=LON)
    R = xr.open_dataset(raw / raw_name).sel(time=slice(args.start, args.end)).assign_coords(**on_grid)
    P = [R[v] for v in variables]
    fields = {k: v.load() for k, v in G["make"](P, LEVELS["cmems"]).items()}
    mask = None
    if group == "temp":
        fields["ssh"] = xr.open_dataset(raw / "zos.nc").zos.sel(time=slice(args.start, args.end)).assign_coords(**on_grid).load()
        M = xr.open_dataset(raw / "mask_full_depth.nc").thetao.isel(time=0).assign_coords(**on_grid)
        mask = dict(wet=np.isfinite(M.isel(depth=0)).drop_vars("depth"),
                    deep200=np.isfinite(M.sel(depth=222.5, method="nearest")).drop_vars("depth"),
                    deep1000=np.isfinite(M.sel(depth=1062.4, method="nearest")).drop_vars("depth"))
    write(out, group, fields, ATTRS["cmems"], mask)


# --- RTOFS ----------------------------------------------------------------
def rtofs(args, out, group):
    from ioos_model_comparisons.models import rtofs as open_rtofs

    G = GROUPS[group]
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = REGIONS[args.region]["bbox"]
    LON, LAT = grid(args.region)
    ds = open_rtofs()
    lon, lat = ds.lon.values, ds.lat.values
    inside = (lon >= LON_MIN - 0.2) & (lon <= LON_MAX + 0.2) & (lat >= LAT_MIN - 0.2) & (lat <= LAT_MAX + 0.2)
    yy, xx = np.where(inside)
    box = dict(y=slice(yy.min(), yy.max() + 1), x=slice(xx.min(), xx.max() + 1))
    x1, y1 = ds.lon.isel(**box).isel(y=0).values, ds.lat.isel(**box).isel(x=0).values
    # South of ~35N the RTOFS Atlantic grid is rectilinear (Mercator), so 1-D coords suffice.
    if np.ptp(np.diff(x1)) > 0.01:
        sys.exit("RTOFS grid is not rectilinear over this region; regridding assumption broken")
    subs = [ds[v].isel(**box).assign_coords(x=x1, y=y1).drop_vars(["lon", "lat"]).rename(x="longitude", y="latitude")
            for v in G["rtofs"]]

    t = ds.get_index("time")
    days = pd.date_range(args.start, args.end, freq="D")
    want = [d for d in days if d in t]
    if len(want) < len(days):
        print(f"warning: {len(days) - len(want)} days have no 00Z RTOFS record and are skipped")
    target = dict(longitude=LON, latitude=LAT)
    drop = lambda v: v.drop_vars([c for c in ("depth", "time", "Date") if c in v.coords])

    def one(day):
        for attempt in range(4):
            try:
                P = [s.sel(time=day).sel(depth=slice(0, G["zmax"])).load() for s in subs]
                return {k: drop(v).interp(**target).values for k, v in G["make"](P, LEVELS["rtofs"]).items()}
            except Exception as e:  # THREDDS hiccups
                err = e
        raise RuntimeError(f"{day}: {err}")

    with ThreadPoolExecutor(args.workers) as ex:
        res = []
        for i, r in enumerate(ex.map(one, want)):
            res.append(r)
            if i % 50 == 0:
                print(f"  {group} {i + 1}/{len(want)}", flush=True)
    coords = dict(time=want, latitude=LAT, longitude=LON)
    fields = clean({k: xr.DataArray(np.stack([r[k] for r in res]), coords=coords, dims=("time", "latitude", "longitude"))
                    for k in res[0]}, G["valid"])
    mask = None
    if group == "temp":
        M = drop(subs[0].sel(time=pd.Timestamp(args.mask_date)).sel(depth=[0, 200, 1000]).load()).interp(**target)
        mask = dict(wet=np.isfinite(M.isel(depth=0)), deep200=np.isfinite(M.isel(depth=1)), deep1000=np.isfinite(M.isel(depth=2)))
    write(out, group, fields, ATTRS["rtofs"], mask)


def _local_read(f, group, box, x1, y1, depths=None):
    """Read one local NCEP RTOFS file: the group's variables over the region box, on 1-D lon/lat."""
    G = GROUPS[group]
    with xr.open_dataset(f) as ds:
        P = []
        for v in G["rtofs"]:
            a = ds[v].isel(MT=0, **box).assign_coords(X=x1, Y=y1).rename(X="longitude", Y="latitude")
            a = a.drop_vars([c for c in ("Longitude", "Latitude", "MT", "Date") if c in a.coords])
            if "Depth" in a.dims:
                a = a.rename(Depth="depth")
                a = a.sel(depth=depths) if depths is not None else a.sel(depth=slice(0, G["zmax"]))
            P.append(a.load())
    return P


def _local_day(task):
    """Worker: one day's 2-D fields on the analysis grid, or None if the file is missing."""
    f, group, box, x1, y1, LON, LAT = task
    if not Path(f).exists():
        return None
    for attempt in range(3):
        try:
            P = _local_read(f, group, box, x1, y1)
            break
        except OSError:  # transient network-share read errors
            if attempt == 2:
                raise
    return {k: v.drop_vars([c for c in ("depth",) if c in v.coords]).interp(longitude=LON, latitude=LAT).values
            for k, v in GROUPS[group]["make"](P, LEVELS["rtofs"]).items()}


def rtofs_local(args, out, group):
    """Same as rtofs(), but reading the NCEP files from a local archive (--rtofs-dir)."""
    G = GROUPS[group]
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = REGIONS[args.region]["bbox"]
    LON, LAT = grid(args.region)
    fname = G.get("rtofs_file", "rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc")
    path = lambda D: Path(args.rtofs_dir) / f"rtofs.{D - pd.Timedelta(days=1):%Y%m%d}" / fname
    days = pd.date_range(args.start, args.end, freq="D")

    with xr.open_dataset(path(days[0])) as first:
        lon = (first.Longitude.values + 180) % 360 - 180  # the global 2-D files store 74..1019
        lat = first.Latitude.values
    inside = (lon >= LON_MIN - 0.2) & (lon <= LON_MAX + 0.2) & (lat >= LAT_MIN - 0.2) & (lat <= LAT_MAX + 0.2)
    yy, xx = np.where(inside)
    box = dict(Y=slice(yy.min(), yy.max() + 1), X=slice(xx.min(), xx.max() + 1))
    x1, y1 = lon[yy.min(), box["X"]], lat[box["Y"], xx.min()]
    if np.ptp(np.diff(x1)) > 0.01 or np.ptp(lat[box["Y"], box["X"]] - y1[:, None]) > 0.01:
        sys.exit("RTOFS grid is not rectilinear over this region; regridding assumption broken")
    target = dict(longitude=LON, latitude=LAT)
    # HDF5 is not safe with several threads opening files, so days are read in separate processes.
    tasks = [(str(path(D)), group, box, x1, y1, LON, LAT) for D in days]
    with ProcessPoolExecutor(args.workers) as ex:
        res = []
        for i, r in enumerate(ex.map(_local_day, tasks, chunksize=4)):
            res.append(r)
            if i % 50 == 0:
                print(f"  {group} {i + 1}/{len(days)}", flush=True)
    missing = [str(D.date()) for D, r in zip(days, res) if r is None]
    if missing:
        print(f"warning: {len(missing)} day(s) missing locally, filled by interpolation where gaps <= 5 days: {missing[:10]}")
    keys = next(r for r in res if r is not None).keys()
    blank = np.full((len(LAT), len(LON)), np.nan)
    coords = dict(time=days, latitude=LAT, longitude=LON)
    fields = clean({k: xr.DataArray(np.stack([r[k] if r is not None else blank for r in res]), coords=coords,
                                    dims=("time", "latitude", "longitude")) for k in keys}, G["valid"])
    mask = None
    if group == "temp":
        M = _local_read(str(path(pd.Timestamp(args.mask_date))), group, box, x1, y1, depths=[0, 200, 1000])[0].interp(**target)
        mask = dict(wet=np.isfinite(M.isel(depth=0, drop=True)), deep200=np.isfinite(M.isel(depth=1, drop=True)),
                    deep1000=np.isfinite(M.isel(depth=2, drop=True)))
    write(out, group, fields, ATTRS["rtofs"], mask)


# --- ESPC -----------------------------------------------------------------
def espc(args, out, group):
    """ESPC-D-V02 from the HYCOM THREDDS yearly aggregations, 00Z each day, on reduced depth levels."""
    from ioos_model_comparisons.models import _open_cf_dataset_sanitize_time

    G = GROUPS[group]
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = REGIONS[args.region]["bbox"]
    LON, LAT = grid(args.region)
    days = pd.date_range(args.start, args.end, freq="D")
    depths = ESPC_DEPTHS[G["zmax"]] if G["zmax"] else None
    years = sorted({d.year for d in days} | {(d - pd.Timedelta(days=1)).year for d in days})
    # one lazy dataset per (product, year); 00Z on 1 Jan lives at the end of the previous year's file
    dsets = {}
    for product, var in G["espc"]:
        for y in years:
            try:
                dsets[(product, y)] = _open_cf_dataset_sanitize_time(ESPC_URL.format(product=product, year=y), drop_variables="tau")
            except Exception as e:
                print(f"warning: could not open ESPC {product} {y}: {e}")
    any_ds = next(iter(dsets.values()))
    lon = any_ds.lon.values
    lat = any_ds.lat.values
    xi = np.flatnonzero((lon >= LON_MIN + 360 - 0.2) & (lon <= LON_MAX + 360 + 0.2))
    yi = np.flatnonzero((lat >= LAT_MIN - 0.2) & (lat <= LAT_MAX + 0.2))
    box = dict(lon=slice(xi.min(), xi.max() + 1), lat=slice(yi.min(), yi.max() + 1))
    target = dict(longitude=LON, latitude=LAT)

    def fetch(product, var, t, dep):
        for y in (t.year, t.year - 1):
            ds = dsets.get((product, y))
            if ds is not None and t in ds.get_index("time"):
                a = ds[var].sel(time=t).isel(**box)
                if dep is not None and "depth" in a.dims:
                    a = a.sel(depth=dep)
                a = a.load()
                a = a.assign_coords(lon=a.lon.values - 360).rename(lon="longitude", lat="latitude")
                return a.drop_vars([c for c in a.coords if c not in ("longitude", "latitude", "depth")])
        return None

    def one(day, dep=depths):
        t = pd.Timestamp(day)
        for attempt in range(4):
            try:
                P = [fetch(product, var, t, dep) for product, var in G["espc"]]
                if any(p is None for p in P):
                    return None
                return {k: v.drop_vars([c for c in ("depth",) if c in v.coords]).interp(**target).values
                        for k, v in G["make"](P, LEVELS["espc"]).items()}
            except Exception as e:  # THREDDS hiccups
                err = e
        print(f"  {day:%Y-%m-%d}: giving up ({err})")
        return None

    with ThreadPoolExecutor(args.workers) as ex:
        res = []
        for i, r in enumerate(ex.map(one, days)):
            res.append(r)
            if i % 25 == 0:
                print(f"  {group} {i + 1}/{len(days)}", flush=True)
    missing = [str(D.date()) for D, r in zip(days, res) if r is None]
    if missing:
        print(f"warning: {len(missing)} day(s) unavailable, filled by interpolation where gaps <= 5 days: {missing[:10]}")
    keys = next(r for r in res if r is not None).keys()
    blank = np.full((len(LAT), len(LON)), np.nan)
    coords = dict(time=days, latitude=LAT, longitude=LON)
    fields = clean({k: xr.DataArray(np.stack([r[k] if r is not None else blank for r in res]), coords=coords,
                                    dims=("time", "latitude", "longitude")) for k in keys}, G["valid"])
    mask = None
    if group == "temp":
        M = fetch("t3z", "water_temp", pd.Timestamp(args.mask_date), [0, 200, 1000]).interp(**target)
        mask = dict(wet=np.isfinite(M.isel(depth=0, drop=True)), deep200=np.isfinite(M.isel(depth=1, drop=True)),
                    deep1000=np.isfinite(M.isel(depth=2, drop=True)))
    write(out, group, fields, ATTRS["espc"], mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cmems", "rtofs", "espc"], required=True)
    ap.add_argument("--region", choices=list(REGIONS), default="gulf")
    ap.add_argument("--vars", nargs="+", choices=list(GROUPS), default=["temp"], help="variable groups to download")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--mask-date", default="2026-01-01", help="day used for the static depth mask")
    ap.add_argument("--workers", type=int, default=8, help="parallel reads (rtofs, espc)")
    ap.add_argument("--rtofs-dir", help="local archive of NCEP RTOFS files (rtofs.YYYYMMDD/...); default: THREDDS")
    args = ap.parse_args()
    for group in args.vars:
        if args.model == "cmems" and GROUPS[group]["cmems"] is None:
            print(f"skipping {group}: not a separate CMEMS group (CMEMS ssh is written with temp)")
            continue
        if args.model == "rtofs" and group == "ssh" and not args.rtofs_dir:
            ap.error("RTOFS ssh is only in the local NCEP 2-D files: pass --rtofs-dir")
        reader = rtofs_local if (args.model == "rtofs" and args.rtofs_dir) else {"cmems": cmems, "rtofs": rtofs, "espc": espc}[args.model]
        reader(args, args.workdir / args.region / args.model, group)


if __name__ == "__main__":
    main()
