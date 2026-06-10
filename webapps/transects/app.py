"""
Flask app for geodesic ocean-model transects
---------------------------------------------
Run with:
    python app.py
or:
    flask --app app run --debug
"""
from __future__ import annotations

import base64
import io
import os
import time
import datetime as dt
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import threading

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from flask import Flask, jsonify, render_template, request
from pyproj import Geod
import cmocean  # noqa: F401 – registers cmo.* colormaps with matplotlib
import copernicusmarine as cm

# Suppress the HDF5 error stack that netCDF4 prints to stderr when it probes
# an OPeNDAP URL to detect the file format.  The probe always "fails" for
# remote URLs (they are not local HDF5 files), but xarray handles it fine.
try:
    import h5py
    h5py.h5e.set_auto(None, None)
except Exception:
    pass

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    _PC   = ccrs.PlateCarree()
    _MERC = ccrs.Mercator()
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    _PC = _MERC = None

try:
    from gsw import SA_from_SP, CT_from_t, rho as gsw_rho, p_from_z
    from gsw import sound_speed as gsw_sound_speed_fn
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

app = Flask(__name__)

# ─────────────────────────────────────────────────────────── globals ──────────
GEOD = Geod(ellps="WGS84")
DEFAULT_START = (-74.0, 40.0)
DEFAULT_END = (-70.0, 40.0)
DEFAULT_DATE = (
    pd.Timestamp.now().round("D").tz_localize("UTC") - pd.Timedelta(days=1)
)
MODEL_LABELS = ["RTOFS", "ESPC", "CMEMS"]

# Simple in-memory model cache: key -> (dataset, timestamp)
_model_cache: dict = {}
CACHE_TTL = 86_400  # 24 hours

# Platform (glider / Argo) cache
_platforms_cache: dict = {}
PLATFORMS_CACHE_TTL = 3_600  # 1 hour

GLIDER_ERDDAP = "https://gliders.ioos.us/erddap"
ARGO_ERDDAP   = "https://erddap.ifremer.fr/erddap"

REDWING_DEPLOYMENT = "redwing-20251011T1511"
REDWING_API_URL    = "https://marine.rutgers.edu/cool/data/gliders/api/surfacings/"
_redwing_cache: dict = {}

# matplotlib is not thread-safe; serialise all figure creation/rendering
_mpl_lock = threading.Lock()


# ─────────────────────────────────────────────────── JSON helpers ─────────────

def _to_list(arr):
    """1-D numpy array → JSON-safe Python list (NaN → None)."""
    return [None if np.isnan(x) else float(x) for x in np.asarray(arr, dtype=float).flat]


def _to_2d(arr):
    """2-D numpy array → JSON-safe nested list (NaN → None)."""
    return [[None if np.isnan(x) else float(x) for x in row]
            for row in np.asarray(arr, dtype=float)]


# ─────────────────────────────────────────────────────── scientific helpers ───

def lon180to360(v):
    return np.mod(np.asarray(v, dtype=float), 360)


def lon360to180(v):
    return np.mod(np.asarray(v, dtype=float) + 180, 360) - 180


def _density(temperature, depth, salinity, latitude, longitude):
    if not HAS_GSW:
        return np.full_like(temperature, np.nan)
    pressure = p_from_z(-np.abs(depth), latitude)
    absolute_salinity = SA_from_SP(salinity, pressure, longitude, latitude)
    conservative_temperature = CT_from_t(absolute_salinity, temperature, pressure)
    return gsw_rho(absolute_salinity, conservative_temperature, pressure)


def _sound_speed(temperature, depth, salinity, latitude, longitude):
    if not HAS_GSW:
        return np.full_like(temperature, np.nan)
    pressure = p_from_z(-np.abs(depth), latitude)
    absolute_salinity = SA_from_SP(salinity, pressure, longitude, latitude)
    conservative_temperature = CT_from_t(absolute_salinity, temperature, pressure)
    return gsw_sound_speed_fn(absolute_salinity, conservative_temperature, pressure)


def add_density_and_sound_speed(ds: xr.Dataset) -> xr.Dataset:
    lat = float(ds.lat.values.flat[0])
    lon = float(ds.lon.values.flat[0])
    depths = ds.depth.values
    temp = ds["temperature"].values
    sal = ds["salinity"].values
    ds["density"] = xr.DataArray(
        _density(temp, depths, sal, lat, lon), dims=["depth"]
    )
    ds["sound_speed"] = xr.DataArray(
        _sound_speed(temp, depths, sal, lat, lon), dims=["depth"]
    )
    return ds


def uv2spdir(u, v, mag=0, rot=0):
    """Convert u/v components to speed and direction (geographic convention)."""
    u, v, mag, rot = list(map(np.asarray, (u, v, mag, rot)))
    vec = u + 1j * v
    spd = np.abs(vec)
    ang = np.angle(vec, deg=True)
    ang = ang - mag + rot
    ang = np.mod(90.0 - ang, 360.0)
    return ang, spd


def rtofs(rename=None, source="east", chunks=None):
    if source == "east":
        url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_scraped"
        model = "RTOFS"
    elif source == "west":
        url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_west_scraped"
        model = "RTOFS (West Coast)"
    elif source == "parallel":
        url = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/rtofs/rtofs_us_east_parallel_scraped"
        model = "RTOFS-P"

    ds = xr.open_dataset(url, chunks={"MT": 1})
    ds = ds.rename({
        "Longitude": "lon",
        "Latitude": "lat",
        "MT": "time",
        "Depth": "depth",
        "X": "x",
        "Y": "y",
    })
    ds = ds.set_coords(["lon", "lat"])
    ds.attrs["model"] = model
    return ds


def espc_uv(rename=False):
    url_uv = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd"
    ds = xr.open_dataset(url_uv, drop_variables="tau")[["water_u", "water_v"]]
    ds.attrs["model"] = "ESPC"
    if rename:
        ds = ds.rename({"water_u": "u", "water_v": "v"})
    return ds


def espc_ts(rename=False, chunks=None):
    url_ts = "https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_ts3z/FMRC_ESPC-D-V02_ts3z_best.ncd"
    if chunks:
        ds = xr.open_dataset(url_ts, drop_variables="tau", chunks=chunks)
    else:
        ds = xr.open_dataset(url_ts, drop_variables="tau")
    ds.attrs["model"] = "ESPC"
    if rename:
        ds = ds.rename({"water_temp": "temperature"})
    return ds


class CMEMS:
    def __init__(self, username=None, password=None) -> None:
        self.username = username or os.getenv("CMEMS_USERNAME")
        self.password = password or os.getenv("CMEMS_PASSWORD")
        if not self.username or not self.password:
            raise ValueError("CMEMS credentials are required. Provide them in the UI or set CMEMS_USERNAME/CMEMS_PASSWORD env vars.")
        self.datasets = {}
        self._load_data()

    def _load_data(self):
        datasets = {
            "temperature": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
            "salinity": "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
            "currents": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        }
        for var, dataset_id in datasets.items():
            self.datasets[var] = cm.open_dataset(
                dataset_id=dataset_id,
                username=self.username,
                password=self.password,
                chunk_size_limit=0,
                service="arco-geo-series",
            )

    def get_combined_subset(self, lon_extent, lat_extent, time=None):
        def _sel(ds, key):
            da = ds.sel(
                longitude=slice(lon_extent[0], lon_extent[1]),
                latitude=slice(lat_extent[0], lat_extent[1]),
            )
            if time:
                da = da.sel(time=time)
            return da

        temperature = _sel(self.datasets["temperature"], "thetao")["thetao"]
        salinity = _sel(self.datasets["salinity"], "so")["so"]
        u = _sel(self.datasets["currents"], "uo")["uo"]
        v = _sel(self.datasets["currents"], "vo")["vo"]

        ds = xr.Dataset({"temperature": temperature, "salinity": salinity, "u": u, "v": v})
        ds.attrs["model"] = "CMEMS"
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        return ds


def subset_rtofs(ds: xr.Dataset, lon_extent, lat_extent):
    grid_lons = ds.lon.values[0, :]
    grid_lats = ds.lat.values[:, 0]
    grid_x = ds.x.values
    grid_y = ds.y.values

    lons_ind = np.interp(lon_extent, grid_lons, grid_x)
    lats_ind = np.interp(lat_extent, grid_lats, grid_y)

    extent = (
        int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1])),
        int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1])),
    )
    return ds.isel(x=slice(extent[0], extent[1]), y=slice(extent[2], extent[3]))


def _load_model_fresh(label: str, lon_extent: tuple, lat_extent: tuple, depth_slice: slice,
                      cmems_username: str | None = None, cmems_password: str | None = None):
    if label == "RTOFS":
        ds = rtofs(rename=True).sel(depth=depth_slice)
        return subset_rtofs(ds, list(lon_extent), list(lat_extent))
    if label == "RTOFS-P":
        ds = rtofs(source="parallel", rename=True).sel(depth=depth_slice)
        return subset_rtofs(ds, list(lon_extent), list(lat_extent))
    if label == "ESPC":
        curr = espc_uv(rename=True).sel(depth=depth_slice)
        ts   = espc_ts(rename=True).sel(depth=depth_slice)
        # Convert requested extent to 0-360 (ESPC native) and add a small pad
        # so the transect endpoints are never clipped at the boundary.
        pad = 2.0
        lon360 = sorted(float(lon180to360(x)) for x in lon_extent)
        lat_sel = slice(lat_extent[0] - pad, lat_extent[1] + pad)
        lon_sel = slice(lon360[0] - pad, lon360[1] + pad)
        curr = curr.sel(lat=lat_sel, lon=lon_sel)
        ts   = ts.sel(lat=lat_sel,   lon=lon_sel)
        merged = xr.merge([curr, ts], compat="override")
        merged["lon"] = (merged["lon"] + 180) % 360 - 180
        return merged
    if label == "CMEMS":
        cmems_obj = CMEMS(username=cmems_username, password=cmems_password)
        return cmems_obj.get_combined_subset(list(lon_extent), list(lat_extent)).sel(depth=depth_slice)
    raise ValueError(f"Unknown model: {label}")


def load_model(label: str, lon_extent: tuple, lat_extent: tuple, depth_slice: slice,
               cmems_username: str | None = None, cmems_password: str | None = None):
    # Credentials are not included in the cache key — use env vars for server-side caching of CMEMS.
    key = (label, lon_extent, lat_extent, depth_slice.start, depth_slice.stop)
    now = time.time()
    if key in _model_cache:
        ds, ts = _model_cache[key]
        if now - ts < CACHE_TTL:
            return ds
    ds = _load_model_fresh(label, lon_extent, lat_extent, depth_slice, cmems_username, cmems_password)
    _model_cache[key] = (ds, now)
    return ds


# ─────────────────────────────────────────────────────── profile helpers ──────

_espc_uv_year_cache: dict = {}


def _load_espc_uv_year(year: int):
    now = time.time()
    if year in _espc_uv_year_cache:
        ds, ts = _espc_uv_year_cache[year]
        if now - ts < CACHE_TTL:
            return ds
    url_u = f"https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/{year}"
    url_v = f"https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z/{year}"
    ds_u = xr.open_dataset(url_u, drop_variables="tau")
    ds_v = xr.open_dataset(url_v, drop_variables="tau")
    ds = xr.Dataset({"u": ds_u["water_u"], "v": ds_v["water_v"]})
    ds.attrs["model"] = "ESPC"
    _espc_uv_year_cache[year] = (ds, now)
    return ds


def extract_rtofs_profile(lon: float, lat: float, target_time, max_depth: float,
                          interp_method: str = "linear"):
    rds = rtofs(rename=True).sel(depth=slice(0, max_depth))
    rlons = rds.lon.values[0, :]
    rlats = rds.lat.values[:, 0]
    rx = rds.x.values
    ry = rds.y.values
    rlonI = np.interp(lon, rlons, rx)
    rlatI = np.interp(lat, rlats, ry)
    ds = rds.sel(time=target_time, method="nearest").interp(x=rlonI, y=rlatI, method=interp_method)
    ds = add_density_and_sound_speed(ds)
    ds.load()
    return ds


def extract_espc_profile(lon: float, lat: float, target_time, max_depth: float,
                         interp_method: str = "nearest"):
    gds_ts = espc_ts(rename=True)
    year = pd.Timestamp(target_time).year
    gds_uv = _load_espc_uv_year(year)

    lon360 = float(lon180to360(lon))

    if interp_method == "linear":
        # Select nearest time step first, then interpolate spatially.
        ds_ts = gds_ts.sel(time=target_time, method="nearest").interp(lon=lon360, lat=lat)
        ds_uv = gds_uv.sel(time=target_time, method="nearest").interp(lon=lon360, lat=lat)
    else:
        kw = dict(method="nearest")
        ds_ts = gds_ts.sel(lon=lon360, lat=lat, time=target_time, **kw)
        ds_uv = gds_uv.sel(lon=lon360, lat=lat, time=target_time, **kw)

    # Build the result dataset explicitly from individual variables.
    # Avoids xr.merge entirely so there is no alignment pass and no risk of
    # pulling the full remote arrays into memory before .load() is called.
    depth = ds_ts.depth.values          # coordinate array — already in memory
    depth_mask = depth <= max_depth

    ds = xr.Dataset(
        {
            "temperature": ds_ts["temperature"].isel(depth=depth_mask),
            "salinity":    ds_ts["salinity"].isel(depth=depth_mask),
            "u":           ds_uv["u"].isel(depth=depth_mask),
            "v":           ds_uv["v"].isel(depth=depth_mask),
        },
        coords={
            "depth": depth[depth_mask],
            "lon":   lon360to180(float(ds_ts.lon.values)),
            "lat":   float(ds_ts.lat.values),
            "time":  ds_ts.time.values,
        },
    )
    ds.attrs["model"] = "ESPC"
    ds = add_density_and_sound_speed(ds)
    ds.load()   # single OPeNDAP request for the 4 depth profiles
    return ds


def extract_cmems_profile(lon: float, lat: float, target_time, username: str, password: str,
                          max_depth: float, interp_method: str = "nearest"):
    if not username or not password:
        raise ValueError("CMEMS credentials required.")
    cmems_obj = CMEMS(username=username, password=password)

    if interp_method == "linear":
        kw_t = dict(method="nearest")
        temperature = cmems_obj.datasets["temperature"]["thetao"].sel(
            time=target_time, **kw_t).interp(longitude=lon, latitude=lat)
        salinity = cmems_obj.datasets["salinity"]["so"].sel(
            time=target_time, **kw_t).interp(longitude=lon, latitude=lat)
        u = cmems_obj.datasets["currents"]["uo"].sel(
            time=target_time, **kw_t).interp(longitude=lon, latitude=lat)
        v = cmems_obj.datasets["currents"]["vo"].sel(
            time=target_time, **kw_t).interp(longitude=lon, latitude=lat)
    else:
        kw = dict(method="nearest")
        temperature = cmems_obj.datasets["temperature"]["thetao"].sel(
            time=target_time, longitude=lon, latitude=lat, **kw)
        salinity = cmems_obj.datasets["salinity"]["so"].sel(
            time=target_time, longitude=lon, latitude=lat, **kw)
        u = cmems_obj.datasets["currents"]["uo"].sel(
            time=target_time, longitude=lon, latitude=lat, **kw)
        v = cmems_obj.datasets["currents"]["vo"].sel(
            time=target_time, longitude=lon, latitude=lat, **kw)

    ds = xr.Dataset({"temperature": temperature, "salinity": salinity, "u": u, "v": v})
    ds.attrs["model"] = "CMEMS"
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = ds.sel(depth=ds.depth[ds.depth <= max_depth])
    ds = add_density_and_sound_speed(ds)
    ds.load()
    return ds


def create_profile_plot(profiles: dict, lon: float, lat: float, max_depth: float,
                        include_u: bool = True, include_v: bool = True,
                        include_sound_speed: bool = True):
    colors = {"RTOFS": "red", "ESPC": "green", "CMEMS": "magenta"}

    # Build variable list — temp/sal/density always shown; u/v/sound_speed optional
    vars_to_plot = [
        ("temperature", "Temperature (°C)"),
        ("salinity",    "Salinity (PSU)"),
        ("density",     "Density (kg/m³)"),
    ]
    if include_u:           vars_to_plot.append(("u",           "U (m/s)"))
    if include_v:           vars_to_plot.append(("v",           "V (m/s)"))
    if include_sound_speed: vars_to_plot.append(("sound_speed", "Sound Speed (m/s)"))

    n = len(vars_to_plot)
    fig = plt.figure(figsize=((n + 1) * 3, 6))
    gs  = fig.add_gridspec(3, n + 1, width_ratios=[1] * n + [1.5], height_ratios=[1, 2, 1])

    axes = []
    for i in range(n):
        ax = fig.add_subplot(gs[:, i], sharey=axes[0] if i else None)
        if i > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)

    ax_info = fig.add_subplot(gs[0, -1])
    if HAS_CARTOPY:
        ax_map = fig.add_subplot(gs[1, -1], projection=ccrs.Mercator())
    else:
        ax_map = fig.add_subplot(gs[1, -1])
    ax_leg = fig.add_subplot(gs[2, -1])

    info_lines = []
    for model_name, ds in profiles.items():
        c = colors.get(model_name, "blue")[0]
        label = f"{model_name} [{float(ds.lon.values.flat[0]):.2f}, {float(ds.lat.values.flat[0]):.2f}]"
        for ax, (var, _) in zip(axes, vars_to_plot):
            ax.plot(ds[var].values, ds["depth"].values, f"{c}-o", label=label, markersize=4)
        info_lines.append(f"{model_name}: {pd.to_datetime(ds.time.values)}")

    for ax, (_, xlabel) in zip(axes, vars_to_plot):
        ax.set_ylim([max_depth, 0])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Depth (m)", fontsize=12, fontweight="bold")

    ax_info.text(0.1, 0.9, "\n".join(info_lines) or "No data",
                 ha="left", va="top", size=10, fontweight="bold", transform=ax_info.transAxes)
    ax_info.set_axis_off()

    if HAS_CARTOPY:
        pad = 7
        ax_map.set_extent([lon - pad, lon + pad, lat - 5, lat + 5], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor="0.9")
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", color="gray")
        gl.top_labels = False
        gl.right_labels = False
        ax_map.plot(lon, lat, "ro", markersize=8, transform=ccrs.PlateCarree())
    else:
        lon_dir = "W" if lon < 0 else "E"
        lat_dir = "N" if lat >= 0 else "S"
        ax_map.text(0.5, 0.5, f"{abs(lat):.2f}°{lat_dir}\n{abs(lon):.2f}°{lon_dir}",
                    ha="center", va="center", fontsize=12, fontweight="bold", transform=ax_map.transAxes)
        ax_map.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        ax_leg.legend(handles, labels, ncol=1, loc="center", fontsize=11)
    ax_leg.set_axis_off()

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────── transect helpers ──────

def calculate_transect(start, end, npts):
    """Return (lon/lat points, cumulative distance in km) along geodesic."""
    line = GEOD.inv_intermediate(
        start[0], start[1], end[0], end[1], npts, return_back_azimuth=False
    )
    lons = np.array(line.lons)
    lats = np.array(line.lats)
    _, _, dist_m = GEOD.inv(
        np.full(npts, start[0]),
        np.full(npts, start[1]),
        lons,
        lats,
    )
    return np.column_stack([lons, lats]), dist_m / 1000  # km


def transect2rtofs(pts, grid_lons, grid_lats, grid_x, grid_y):
    lonidx = np.interp(pts[:, 0], grid_lons, grid_x)
    latidx = np.interp(pts[:, 1], grid_lats, grid_y)
    return lonidx, latidx


def build_section(ds: xr.Dataset, pts, distances, depth_arr, interp_method: str = "linear"):
    if {"x", "y"}.issubset(ds.dims):
        lonidx, latidx = transect2rtofs(
            pts,
            ds.lon.values[0, :],
            ds.lat.values[:, 0],
            ds.x.values,
            ds.y.values,
        )
        section = ds.interp(
            x=xr.DataArray(lonidx, dims="point"),
            y=xr.DataArray(latidx, dims="point"),
            depth=xr.DataArray(depth_arr, dims="depth"),
            method=interp_method,
        )
    else:
        section = ds.interp(
            lon=xr.DataArray(pts[:, 0], dims="point"),
            lat=xr.DataArray(pts[:, 1], dims="point"),
            depth=xr.DataArray(depth_arr, dims="depth"),
            method=interp_method,
        )
        section.load()

    _, spd = uv2spdir(section["u"].data, section["v"].data)
    section["speed"] = (("depth", "point"), spd)
    section = section.assign_coords(
        longitude=("point", pts[:, 0]),
        distance=("point", distances),
    )
    return section


# ─────────────────────────────────────────────────────────── Flask routes ─────

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_labels=MODEL_LABELS,
        default_start=DEFAULT_START,
        default_end=DEFAULT_END,
        default_date=DEFAULT_DATE.date().isoformat(),
    )


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)

        model_label = data["model"]
        date_str = data["date"]          # "YYYY-MM-DD"
        hour = int(data["hour"])
        start = (float(data["start_lon"]), float(data["start_lat"]))
        end = (float(data["end_lon"]), float(data["end_lat"]))
        npts = int(data.get("npts", 500))
        x_axis = data.get("x_axis", "distance")
        depth_min = float(data.get("depth_min", 0))
        depth_max = float(data.get("depth_max", 1000))
        depth_spacing = float(data.get("depth_spacing", 10))
        temp_min = float(data.get("temp_min", 4.0))
        temp_max = float(data.get("temp_max", 28.0))
        temp_step = float(data.get("temp_step", 1.0))
        sal_min = float(data.get("sal_min", 34.9))
        sal_max = float(data.get("sal_max", 36.6))
        sal_step = float(data.get("sal_step", 0.1))
        spd_max = float(data.get("spd_max", 1.0))
        spd_step = float(data.get("spd_step", 0.1))
        interp_method = data.get("interp_method", "linear")
        cmems_username = data.get("cmems_username") or None
        cmems_password = data.get("cmems_password") or None

        analysis_time = pd.Timestamp.combine(
            pd.Timestamp(date_str).date(), dt.time(hour, 0)
        )

        lon_extent = tuple(sorted([start[0], end[0]]))
        lat_extent = tuple(sorted([start[1], end[1]]))
        depth_slice = slice(depth_min, depth_max)
        depth_arr = np.arange(depth_min, depth_max + depth_spacing, depth_spacing)

        model_ds = load_model(model_label, lon_extent, lat_extent, depth_slice,
                              cmems_username, cmems_password).sel(
            time=analysis_time, method="nearest"
        )

        pts, dists = calculate_transect(start, end, npts)
        section = build_section(model_ds, pts, dists, depth_arr, interp_method=interp_method)

        # ── plot ──────────────────────────────────────────────────────────────
        with _mpl_lock:
            from matplotlib.gridspec import GridSpecFromSubplotSpec

            fig    = plt.figure(figsize=(18, 10))
            gs     = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.15, wspace=0.3)
            ax0    = fig.add_subplot(gs[0, 0])
            ax1    = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax2    = fig.add_subplot(gs[2, 0], sharex=ax0)
            axes   = [ax0, ax1, ax2]

            # Right column: info panel (top) + location map (bottom, ~3× taller)
            gs_right = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[:, 1],
                height_ratios=[1, 3], hspace=0.15,
            )
            ax_info = fig.add_subplot(gs_right[0])
            if HAS_CARTOPY:
                ax_map = fig.add_subplot(gs_right[1], projection=ccrs.Mercator())
            else:
                ax_map = fig.add_subplot(gs_right[1])

            def _plot(var, ax, cmap, vmin, vmax, vstep, label, title):
                levels = np.arange(vmin, vmax + vstep, vstep)
                h = section[var].plot(
                    ax=ax,
                    x=x_axis,
                    y="depth",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    levels=levels,
                    add_labels=False,
                    add_colorbar=False,
                    extend="both",
                )
                cbar = plt.colorbar(h, ax=ax, orientation="vertical", pad=0.02)
                cbar.set_label(label)
                ax.invert_yaxis()
                ax.set_ylabel("Depth (m)", fontweight="bold")
                ax.set_title(title, fontsize=12, fontweight="bold")

            _plot("temperature", axes[0], plt.get_cmap("cmo.thermal"), temp_min, temp_max, temp_step, "°C", "Temperature")
            _plot("salinity",    axes[1], plt.get_cmap("cmo.haline"),  sal_min,  sal_max,  sal_step,  "PSU", "Salinity")
            _plot("speed",       axes[2], plt.get_cmap("cmo.speed"),   0,        spd_max,  spd_step,  "m/s", "Speed")

            xlabel_map = {
                "distance": "Distance along transect (km)",
                "lon": "Longitude (°W)",
                "lat": "Latitude (°N)",
            }
            axes[2].set_xlabel(xlabel_map.get(x_axis, x_axis), fontweight="bold")

            # ── info panel ────────────────────────────────────────────────────
            def _fmt_coord(lon, lat):
                lo_dir = "W" if lon < 0 else "E"
                la_dir = "N" if lat >= 0 else "S"
                return f"{abs(lat):.2f}°{la_dir}, {abs(lon):.2f}°{lo_dir}"

            total_dist_km = float(dists[-1])
            info_text = (
                f"{model_label}\n"
                f"{analysis_time.strftime('%Y-%m-%d %H UTC')}\n\n"
                f"Start: {_fmt_coord(*start)}\n"
                f"End:   {_fmt_coord(*end)}\n"
                f"Dist:  {total_dist_km:.0f} km"
            )
            ax_info.text(0.05, 0.95, info_text,
                         ha="left", va="top", fontsize=9, fontweight="bold",
                         transform=ax_info.transAxes, linespacing=1.5)
            ax_info.set_axis_off()

            # ── location map ──────────────────────────────────────────────────
            if HAS_CARTOPY:
                lon_mid  = (start[0] + end[0]) / 2
                lat_mid  = (start[1] + end[1]) / 2
                lon_span = abs(end[0] - start[0])
                lat_span = abs(end[1] - start[1])
                pad      = max(lon_span, lat_span) * 0.5 + 3.0
                ax_map.set_extent(
                    [lon_mid - pad, lon_mid + pad, lat_mid - pad * 0.7, lat_mid + pad * 0.7],
                    crs=ccrs.PlateCarree(),
                )
                ax_map.add_feature(cfeature.LAND, facecolor="0.9")
                ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6)
                gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", color="gray")
                gl.top_labels   = False
                gl.right_labels = False
                gl.xlabel_style = {"size": 7}
                gl.ylabel_style = {"size": 7}
                _pc = ccrs.PlateCarree()
                ax_map.plot([start[0], end[0]], [start[1], end[1]],
                            "r-", linewidth=2, transform=_pc, zorder=5)
                ax_map.plot(start[0], start[1], "go", markersize=8, transform=_pc,
                            zorder=6, label="Start")
                ax_map.plot(end[0],   end[1],   "rs", markersize=8, transform=_pc,
                            zorder=6, label="End")
                ax_map.legend(loc="lower left", fontsize=8, framealpha=0.7)
            else:
                ax_map.text(
                    0.5, 0.5,
                    f"Start: {_fmt_coord(*start)}\nEnd: {_fmt_coord(*end)}",
                    ha="center", va="center", fontsize=9,
                    transform=ax_map.transAxes,
                )
                ax_map.set_axis_off()
            ax_map.set_title("Location", fontsize=9, fontweight="bold", pad=3)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        filename = f"{model_label}_transect_{analysis_time.strftime('%Y%m%d_%HZ')}.png"

        try:
            transect_data = {
                "depth": depth_arr.tolist(),
                "distance_km": _to_list(section["distance"].values),
                "longitude": _to_list(section["longitude"].values),
                "temperature": _to_2d(section["temperature"].values),
                "salinity": _to_2d(section["salinity"].values),
                "speed": _to_2d(section["speed"].values),
            }
        except Exception:
            transect_data = None

        return jsonify({"status": "ok", "image": img_b64, "filename": filename, "data": transect_data})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/generate_profile", methods=["POST"])
def generate_profile():
    try:
        data = request.get_json(force=True)

        include_rtofs       = bool(data.get("include_rtofs", False))
        include_espc        = bool(data.get("include_espc",  False))
        include_cmems       = bool(data.get("include_cmems", False))
        include_u           = bool(data.get("include_u",           True))
        include_v           = bool(data.get("include_v",           True))
        include_sound_speed = bool(data.get("include_sound_speed", True))
        lon           = float(data["lon"])
        lat           = float(data["lat"])
        date_str      = data["date"]
        hour          = int(data["hour"])
        max_depth     = float(data.get("max_depth", 400))
        interp_method  = data.get("interp_method", "linear")
        cmems_username = data.get("cmems_username") or None
        cmems_password = data.get("cmems_password") or None

        if not any([include_rtofs, include_espc, include_cmems]):
            return jsonify({"status": "error", "message": "Select at least one model."}), 400

        if include_cmems and (not cmems_username or not cmems_password):
            return jsonify({"status": "error", "message": "CMEMS credentials required."}), 400

        target_time = pd.Timestamp.combine(
            pd.Timestamp(date_str).date(), dt.time(hour, 0)
        )

        profiles = {}
        errors = []

        if include_rtofs:
            try:
                profiles["RTOFS"] = extract_rtofs_profile(lon, lat, target_time, max_depth,
                                                           interp_method=interp_method)
            except Exception as e:
                errors.append(f"RTOFS: {e}")

        if include_espc:
            try:
                profiles["ESPC"] = extract_espc_profile(lon, lat, target_time, max_depth,
                                                         interp_method=interp_method)
            except Exception as e:
                errors.append(f"ESPC: {e}")

        if include_cmems:
            try:
                profiles["CMEMS"] = extract_cmems_profile(
                    lon, lat, target_time, cmems_username, cmems_password, max_depth,
                    interp_method=interp_method,
                )
            except Exception as e:
                errors.append(f"CMEMS: {e}")

        if not profiles:
            return jsonify({"status": "error", "message": "All models failed. " + " | ".join(errors)}), 500

        with _mpl_lock:
            fig = create_profile_plot(profiles, lon, lat, max_depth,
                                      include_u=include_u,
                                      include_v=include_v,
                                      include_sound_speed=include_sound_speed)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        filename = f"profile_{lat:.2f}N_{abs(lon):.2f}W_{target_time.strftime('%Y%m%d_%HZ')}.png"

        profile_data = {}
        for model_name, ds in profiles.items():
            profile_data[model_name] = {
                "depth":       ds["depth"].values.tolist(),
                "temperature": _to_list(ds["temperature"].values) if "temperature" in ds else None,
                "salinity":    _to_list(ds["salinity"].values)    if "salinity"    in ds else None,
                "density":     _to_list(ds["density"].values)     if "density"     in ds else None,
                "sound_speed": _to_list(ds["sound_speed"].values) if "sound_speed" in ds else None,
                "u":           _to_list(ds["u"].values)           if "u"           in ds else None,
                "v":           _to_list(ds["v"].values)           if "v"           in ds else None,
            }

        result = {"status": "ok", "image": img_b64, "filename": filename, "data": profile_data}
        if errors:
            result["warnings"] = errors
        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/map_overlay", methods=["POST"])
def map_overlay():
    """Generate a transparent PNG of surface model data for the Leaflet imageOverlay."""
    try:
        import copy as _copy
        data = request.get_json(force=True)

        model          = data["model"]
        variable       = data["variable"]   # "temperature" | "salinity" | "speed"
        date_str       = data["date"]
        hour           = int(data["hour"])
        west           = float(data["west"])
        east           = float(data["east"])
        south          = float(data["south"])
        north          = float(data["north"])
        vmin           = float(data.get("vmin", 4.0))
        vmax           = float(data.get("vmax", 28.0))
        cmap_name      = data.get("cmap", "cmo.thermal")
        depth_val      = float(data.get("depth", 0.0))
        cmems_username = data.get("cmems_username") or None
        cmems_password = data.get("cmems_password") or None

        lon_extent  = (west, east)
        lat_extent  = (south, north)
        # Load a tight depth window around the requested level so the cache key
        # stays small; 50 m of margin is enough to hit the nearest model level.
        margin      = 50.0
        depth_slice = slice(max(0.0, depth_val - margin), depth_val + margin)

        target_time = pd.Timestamp.combine(
            pd.Timestamp(date_str).date(), dt.time(hour, 0)
        )

        ds   = load_model(model, lon_extent, lat_extent, depth_slice,
                          cmems_username, cmems_password)
        ds_t = ds.sel(time=target_time, method="nearest").sel(depth=depth_val, method="nearest")

        # Resolve 2-D lon/lat arrays (RTOFS has curvilinear coords)
        if {"x", "y"}.issubset(ds_t.dims):
            lons = ds_t.lon.values   # shape (y, x)
            lats = ds_t.lat.values
        else:
            lon_1d = ds_t.lon.values
            lat_1d = ds_t.lat.values
            lons, lats = np.meshgrid(lon_1d, lat_1d)

        # Compute requested variable
        if variable == "speed":
            u_vals = np.asarray(ds_t["u"].values, dtype=float)
            v_vals = np.asarray(ds_t["v"].values, dtype=float)
            if u_vals.ndim > 2:
                u_vals = u_vals.squeeze()
                v_vals = v_vals.squeeze()
            _, data_vals = uv2spdir(u_vals, v_vals)
        else:
            u_vals = v_vals = None
            data_vals = np.asarray(ds_t[variable].values, dtype=float)
            if data_vals.ndim > 2:
                data_vals = data_vals.squeeze()

        # Build transparent PNG in Web Mercator projection so pixels align with
        # Leaflet tiles.  L.imageOverlay stretches the image linearly in Mercator
        # space, so rendering in Mercator eliminates the lat/lon distortion that
        # makes features appear shifted vs. the land tiles at higher latitudes.
        dpi = 100
        cmap = _copy.copy(plt.get_cmap(cmap_name))
        cmap.set_bad(alpha=0)   # NaN / masked cells → transparent

        with _mpl_lock:
            if HAS_CARTOPY:
                # Size the figure to match the Mercator aspect ratio of the bbox so
                # subplots_adjust(0,0,1,1) gives us exactly the right pixel grid.
                xy_sw = _MERC.transform_point(west,  south, _PC)
                xy_ne = _MERC.transform_point(east,  north, _PC)
                aspect = (xy_ne[0] - xy_sw[0]) / max(xy_ne[1] - xy_sw[1], 1e-6)
                fig_h  = 6.0
                fig    = plt.figure(figsize=(fig_h * aspect, fig_h), dpi=dpi)
                ax     = fig.add_subplot(1, 1, 1, projection=_MERC)
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                ax.set_extent([west, east, south, north], crs=_PC)

                ax.pcolormesh(lons, lats, data_vals,
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              shading="nearest", rasterized=True,
                              transform=_PC)

                if variable == "speed" and u_vals is not None and v_vals is not None:
                    ny, nx = data_vals.shape
                    stride = max(1, min(ny, nx) // 40)
                    qs     = np.s_[::stride, ::stride]
                    lon_q  = lons[qs]; lat_q = lats[qs]
                    u_q    = u_vals[qs]; v_q  = v_vals[qs]
                    mask_q = np.isfinite(u_q) & np.isfinite(v_q) & np.isfinite(lat_q)
                    if mask_q.any():
                        um, vm = u_q[mask_q], v_q[mask_q]
                        mag = np.sqrt(um**2 + vm**2)
                        mag = np.where(mag == 0, 1, mag)
                        ax.quiver(lon_q[mask_q], lat_q[mask_q],
                                  um / mag, vm / mag,
                                  color="black", alpha=0.7, zorder=5,
                                  transform=_PC,
                                  pivot="mid", scale=45, scale_units="width",
                                  width=0.002, headwidth=4, headlength=5)
            else:
                # Fallback (no cartopy): flat lat/lon render — will be misaligned at
                # high latitudes because Leaflet uses Mercator.
                fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                ax.pcolormesh(lons, lats, data_vals,
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              shading="nearest", rasterized=True)
                if variable == "speed" and u_vals is not None and v_vals is not None:
                    ny, nx = data_vals.shape
                    stride = max(1, min(ny, nx) // 40)
                    qs     = np.s_[::stride, ::stride]
                    lon_q  = lons[qs]; lat_q = lats[qs]
                    u_q    = u_vals[qs]; v_q  = v_vals[qs]
                    mask_q = np.isfinite(u_q) & np.isfinite(v_q) & np.isfinite(lat_q)
                    if mask_q.any():
                        um, vm = u_q[mask_q], v_q[mask_q]
                        mag = np.sqrt(um**2 + vm**2)
                        mag = np.where(mag == 0, 1, mag)
                        ax.quiver(lon_q[mask_q], lat_q[mask_q],
                                  um / mag, vm / mag,
                                  color="black", alpha=0.7, zorder=5,
                                  pivot="mid", scale=45, scale_units="width",
                                  width=0.002, headwidth=4, headlength=5)
                ax.set_xlim(west, east)
                ax.set_ylim(south, north)

            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
            plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "status": "ok",
            "image":  img_b64,
            "bounds": {"west": west, "east": east, "south": south, "north": north},
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


# ─────────────────────────────────────────────────── platform helpers ─────────

def _fetch_glider_track(dataset_id: str, days: int):
    """Return track coords + last position for one glider over the past N days."""
    try:
        now = pd.Timestamp.now(tz="UTC")
        cutoff = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"{GLIDER_ERDDAP}/tabledap/{dataset_id}.json"
            f"?time,latitude,longitude&time>={cutoff}&orderBy(%22time%22)"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = data["table"]["rows"]
        if not rows:
            return None
        coords = []
        last_time = None
        for t, lat, lon in rows:
            if lat is not None and lon is not None:
                coords.append([float(lon), float(lat)])
                last_time = t
        if not coords:
            return None
        return {
            "id": dataset_id,
            "coords": coords,
            "last_lon": coords[-1][0],
            "last_lat": coords[-1][1],
            "last_time": last_time,
        }
    except Exception:
        return None


def fetch_gliders(days: int = 30) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{GLIDER_ERDDAP}/tabledap/allDatasets.json"
        f"?datasetID,maxTime"
        f"&maxTime>={cutoff}"
        f"&datasetID!=%22allDatasets%22"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = data["table"]["columnNames"]
    rows = data["table"]["rows"]
    dataset_ids = [row[cols.index("datasetID")] for row in rows]

    features = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(_fetch_glider_track, did, days): did for did in dataset_ids}
        for fut in as_completed(futures):
            result = fut.result()
            if result and result["coords"]:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": result["coords"]},
                    "properties": {"id": result["id"], "n": len(result["coords"])},
                })
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [result["last_lon"], result["last_lat"]]},
                    "properties": {"id": result["id"], "time": result["last_time"]},
                })
    return {"type": "FeatureCollection", "features": features}


def fetch_argo(days: int = 30) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    cutoff = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{ARGO_ERDDAP}/tabledap/ArgoFloats.json"
        f"?platform_number,time,latitude,longitude"
        f"&time>={cutoff}"
        f"&orderByMax(%22platform_number,time%22)"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    cols = data["table"]["columnNames"]
    rows = data["table"]["rows"]

    features = []
    for row in rows:
        platform = str(row[cols.index("platform_number")])
        t        = row[cols.index("time")]
        lat      = row[cols.index("latitude")]
        lon      = row[cols.index("longitude")]
        if lat is not None and lon is not None:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {"id": platform, "time": t},
            })
    return {"type": "FeatureCollection", "features": features}


@app.route("/platforms")
def platforms():
    try:
        ptype = request.args.get("type", "gliders")
        days  = int(request.args.get("days", 30))
        key   = (ptype, days)
        now   = time.time()
        if key in _platforms_cache:
            cached, ts = _platforms_cache[key]
            if now - ts < PLATFORMS_CACHE_TTL:
                return jsonify(cached)
        if ptype == "gliders":
            result = fetch_gliders(days)
        elif ptype == "argo":
            result = fetch_argo(days)
        else:
            return jsonify({"status": "error", "message": "Unknown type"}), 400
        _platforms_cache[key] = (result, now)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/redwing")
def redwing():
    """Return Redwing glider track + latest position as a GeoJSON FeatureCollection."""
    try:
        now = time.time()
        if "data" in _redwing_cache:
            cached, ts = _redwing_cache["data"]
            if now - ts < PLATFORMS_CACHE_TTL:
                return jsonify(cached)

        r = requests.get(
            REDWING_API_URL,
            params={"deployment": REDWING_DEPLOYMENT},
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not records:
            return jsonify({"status": "error", "message": "No surfacings returned"}), 404

        df = pd.json_normalize(records)
        df["time"] = pd.to_datetime(df.get("gps_timestamp_epoch"), unit="s", utc=True, errors="coerce")

        if "gps_lat_degrees" in df.columns:
            df["lat"] = pd.to_numeric(df["gps_lat_degrees"], errors="coerce")
        if "gps_lon_degrees" in df.columns:
            df["lon"] = pd.to_numeric(df["gps_lon_degrees"], errors="coerce")

        df = df.dropna(subset=["time", "lat", "lon"]).sort_values("time")
        if df.empty:
            return jsonify({"status": "error", "message": "No valid GPS positions in response"}), 404

        coords = [[float(row.lon), float(row.lat)] for _, row in df.iterrows()]
        first, last = df.iloc[0], df.iloc[-1]

        result = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "id":         REDWING_DEPLOYMENT,
                        "n":          len(coords),
                        "start_time": first["time"].isoformat(),
                        "end_time":   last["time"].isoformat(),
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(last.lon), float(last.lat)]},
                    "properties": {
                        "id":   REDWING_DEPLOYMENT,
                        "time": last["time"].isoformat(),
                    },
                },
            ],
        }
        _redwing_cache["data"] = (result, now)
        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
