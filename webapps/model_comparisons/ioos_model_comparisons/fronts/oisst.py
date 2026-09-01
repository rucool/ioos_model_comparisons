"""
oisst.py — NOAA OISST v2 high-resolution SST, fetched on demand for the map.

https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html

Daily 0.25-degree global analysis, 1981-09 to near-present, served by PSL over
OPeNDAP as one file per year. Unlike the GOES-19 overlays (rendered once by the
nightly digitizer and stored in Mongo), this is rendered per request so the
date and colour scale can be changed interactively.

Two facts about the source shape the code:

* Longitude is 0-360, so a Gulf Stream extent of [-77.25, -49.75] has to be
  requested as [282.75, 310.25]. An extent that crosses the antimeridian
  becomes two requests that are then concatenated.
* Opening a year file costs ~2 s but a regional slice is ~0.4 s, so the opened
  dataset is cached per year and the extracted slice per (date, extent). With
  both warm, changing the colormap or limits is a pure re-render (~50 ms) and
  never touches the network.

Caches are per process. Under gunicorn each worker keeps its own, which is
fine for a handful of editors — it costs a little duplicated memory, not
correctness.
"""

from __future__ import annotations

import datetime
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

URL_TEMPLATE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
                "noaa.oisst.v2.highres/sst.day.mean.{year}.nc")

FIRST_YEAR = 1981

# Curated colormaps: the project's matplotlib figures use turbo and cmocean
# thermal, so an overlay can be made to match one. Names are validated against
# this list before reaching resolve_cmap so a query string cannot probe for
# arbitrary attributes.
COLORMAPS = [
    {"id": "turbo",        "label": "turbo"},
    {"id": "cmo.thermal",  "label": "cmocean thermal"},
    {"id": "cmo.balance",  "label": "cmocean balance (diverging)"},
    {"id": "cmo.haline",   "label": "cmocean haline"},
    {"id": "viridis",      "label": "viridis"},
    {"id": "plasma",       "label": "plasma"},
    {"id": "inferno",      "label": "inferno"},
    {"id": "magma",        "label": "magma"},
    {"id": "Spectral_r",   "label": "Spectral (reversed)"},
    {"id": "RdYlBu_r",     "label": "RdYlBu (reversed)"},
    {"id": "coolwarm",     "label": "coolwarm"},
    {"id": "jet",          "label": "jet"},
]
VALID_CMAPS = {c["id"] for c in COLORMAPS}

_ds_cache = {}          # year -> xr.Dataset
_slice_cache = {}       # (date, extent) -> (lats, lons, sst)
_lock = threading.Lock()
_MAX_SLICES = 64


def valid_cmap(name):
    return name if name in VALID_CMAPS else "turbo"


def _open_year(year):
    with _lock:
        if year in _ds_cache:
            return _ds_cache[year]
    import xarray as xr
    url = URL_TEMPLATE.format(year=int(year))
    ds = xr.open_dataset(url)           # lazy; only coords are read here
    with _lock:
        _ds_cache[year] = ds
    logger.info(f"opened OISST {year}")
    return ds


def available_range():
    """(first_date, last_date) actually served, or None if PSL is unreachable."""
    try:
        import pandas as pd
        year = datetime.date.today().year
        for y in (year, year - 1):      # early January the current file may not exist
            try:
                ds = _open_year(y)
            except Exception:
                continue
            t = pd.to_datetime(ds.time.values)
            return f"{FIRST_YEAR}-09-01", str(t[-1].date())
    except Exception as exc:
        logger.warning(f"OISST availability check failed: {exc}")
    return None


def fetch(date, extent):
    """(lats, lons, sst) for `date` over `extent` = [lon0, lon1, lat0, lat1].

    Longitudes in `extent` are the usual -180..180; the 0-360 conversion the
    source needs happens here so callers never have to think about it.
    """
    import pandas as pd

    key = (str(date), tuple(round(float(v), 4) for v in extent))
    with _lock:
        hit = _slice_cache.get(key)
    if hit is not None:
        return hit

    day = pd.Timestamp(date)
    ds = _open_year(day.year)
    lon0, lon1, lat0, lat1 = [float(v) for v in extent]

    def to360(x):
        return x % 360.0

    sel = dict(lat=slice(min(lat0, lat1), max(lat0, lat1)))
    a0, a1 = to360(lon0), to360(lon1)
    if a0 <= a1:
        sub = ds["sst"].sel(time=day, method="nearest").sel(lon=slice(a0, a1), **sel)
    else:
        # crosses the antimeridian: two reads, stitched west-to-east
        import xarray as xr
        left = ds["sst"].sel(time=day, method="nearest").sel(lon=slice(a0, 360.0), **sel)
        right = ds["sst"].sel(time=day, method="nearest").sel(lon=slice(0.0, a1), **sel)
        sub = xr.concat([left, right], dim="lon")

    sub = sub.load()
    lons = sub["lon"].values.astype(float)
    # back to -180..180 so the values line up with the map's extent
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    order = np.argsort(lons)
    out = (sub["lat"].values.astype(float), lons[order],
           sub.values.astype(float)[:, order])

    with _lock:
        if len(_slice_cache) >= _MAX_SLICES:
            _slice_cache.pop(next(iter(_slice_cache)))
        _slice_cache[key] = out
    return out


def render(date, extent, *, cmap="turbo", vmin=None, vmax=None, stride=None,
           height=900):
    """PNG bytes + metadata for a date/extent, ready for L.imageOverlay."""
    from ioos_model_comparisons.fronts.webmap import render_overlay_png
    lats, lons, sst = fetch(date, extent)
    png, meta = render_overlay_png(sst, lats, lons, extent,
                                   cmap=valid_cmap(cmap), vmin=vmin, vmax=vmax,
                                   stride=stride, height=height)
    meta.update(date=str(date), source="NOAA OISST v2 high-res",
                grid=f"{sst.shape[0]}x{sst.shape[1]}")
    return png, meta


def render_colorbar(*, cmap="turbo", vmin=0.0, vmax=1.0, stride=None,
                    width=240, height=18):
    """A horizontal ramp of the SAME colormap/limits/stride as the map layer.

    Built from the identical norm the data render uses, so the legend cannot
    drift from what is drawn — a legend that is merely a similar-looking
    gradient is worse than none.
    """
    import io
    import numpy as np
    import matplotlib.image as mpimage
    from matplotlib import cm, colors
    from ioos_model_comparisons.fronts.webmap import resolve_cmap

    base = resolve_cmap(valid_cmap(cmap))
    if stride and stride > 0 and (vmax - vmin) / stride <= 256:
        levels = np.arange(vmin, vmax + stride / 2.0, stride)
        norm = colors.BoundaryNorm(levels, ncolors=base.N, clip=True)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    ramp = np.linspace(vmin, vmax, int(width))[None, :].repeat(int(height), 0)
    rgba = cm.ScalarMappable(norm=norm, cmap=base).to_rgba(ramp, bytes=True)
    buf = io.BytesIO()
    mpimage.imsave(buf, rgba, format="png")
    return buf.getvalue()


def stats(date, extent):
    """Percentiles for the current view, so the UI can suggest sensible limits
    instead of making someone guess vmin/vmax for an unfamiliar region."""
    try:
        _, _, sst = fetch(date, extent)
        v = sst[np.isfinite(sst)]
        if not v.size:
            return None
        return {"min": float(v.min()), "max": float(v.max()),
                "p2": float(np.percentile(v, 2)), "p98": float(np.percentile(v, 98)),
                "suggest_vmin": float(np.floor(np.percentile(v, 2))),
                "suggest_vmax": float(np.ceil(np.percentile(v, 98)))}
    except Exception as exc:
        logger.warning(f"OISST stats failed: {exc}")
        return None
