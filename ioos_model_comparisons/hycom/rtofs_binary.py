"""
Process raw RTOFS/HYCOM binary (.a/.b) archive files into regional,
fixed-z-level output.

RTOFS archive files store the global ocean on a hybrid/isopycnal vertical
coordinate: layer thickness (and therefore the physical depth of a given
layer index) varies per grid column. This module crops a region out of the
global grid, converts the hybrid layers to physical depths, and interpolates
each water column onto a set of fixed z-levels -- optionally also
regridding horizontally onto a uniform target lon/lat grid.

The vertical interpolation (pchip + bottom/land fill) ports the approach
used in https://github.com/DmitryDukhovskoy/hycom_TSIS_GoM/tree/main/interp2grid.
"""
import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, griddata, interp1d
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from ioos_model_comparisons.hycom.info import read_hycom_coords, read_hycom_fields
from ioos_model_comparisons.regions import region_config
from pyhycom.pyhycom import thickness2depths, getBathymetry

RG = 9806.0  # HYCOM pressure -> meters conversion factor
DEEP_PAD = 12000.0  # synthetic deep padding depth (m), matches matlab reference

# Standard z-levels used by the Rutgers RTOFS OPeNDAP "scraped" product
# (ioos_model_comparisons.models.rtofs() -> ds['Depth']), so output from this
# module lines up with data pulled from that source.
RTOFS_STANDARD_Z_LEVELS = [
    0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
    100, 125, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000,
    1250, 1500, 2000, 2500, 3000, 4000, 5000,
]


def read_archive_time(archv_or_b_file):
    """
    Parse the model day from an archv .b file header and convert it to a
    real-world UTC datetime: the inverse of `pyhycom.forday` (yrflag=3, where
    model day 1.0 = 1901-01-01 00:00 UTC).
    """
    b_file = str(archv_or_b_file)
    if b_file.endswith('.a') or b_file.endswith('.b'):
        b_file = b_file[:-2]
    b_file = b_file + '.b'

    lines = open(b_file).readlines()
    model_day = float(lines[10].split()[3])
    return dt.datetime(1901, 1, 1) + dt.timedelta(days=model_day - 1.0)


def get_grid_indices(grid_file, extent):
    """
    Find the (j0, j1, i0, i1) index window of `grid_file` (regional.grid.a/.b)
    covering `extent` = [lonmin, lonmax, latmin, latmax]. If `extent` is
    `None`, returns the full global grid (no crop) -- this works correctly
    even in the curvilinear Arctic tripolar patch above ~47N, since no
    regularity assumption is needed when there's nothing to search for.

    The cropping search itself is only valid below ~47N where the RTOFS grid
    is regular Mercator (plat constant along rows, plon constant along
    columns). `plon` is stored unwrapped (monotonically increasing past
    360), so the requested longitudes are shifted into that frame before
    searching.
    """
    coords = read_hycom_coords(grid_file, ['plon', 'plat'])
    plon, plat = coords['plon'], coords['plat']

    if extent is None:
        jdm, idm = plon.shape
        lon_out = ((plon + 180) % 360) - 180
        return 0, jdm, 0, idm, lon_out, plat

    lon1d = plon[0, :]
    lat1d = plat[:, 0]

    lonmin, lonmax, latmin, latmax = extent

    # Shift requested longitudes into the grid's unwrapped frame.
    lon0 = lon1d[0]
    lonmin_shifted = lon0 + ((lonmin - lon0) % 360)
    lonmax_shifted = lon0 + ((lonmax - lon0) % 360)
    if lonmax_shifted <= lonmin_shifted:
        lonmax_shifted += 360

    i0, i1 = np.searchsorted(lon1d, [lonmin_shifted, lonmax_shifted])
    j0, j1 = np.searchsorted(lat1d, [latmin, latmax])

    pad = 15
    i0, i1 = max(i0 - pad, 0), min(i1 + pad, lon1d.size)
    j0, j1 = max(j0 - pad, 0), min(j1 + pad, lat1d.size)

    # Wrap longitudes back to the standard [-180, 180) convention for output
    # (the unwrapped frame above is only needed for the index search).
    lon_out = ((plon[j0:j1, i0:i1] + 180) % 360) - 180

    return j0, j1, i0, i1, lon_out, plat[j0:j1, i0:i1]


def read_region(archv_file, grid_file, depth_file, extent,
                 fields=('temp', 'salin', 'u-vel.', 'v-vel.', 'thknss')):
    """
    Read a region out of an RTOFS archive file and compute the per-column
    hybrid-layer depths. `extent = None` reads the whole global grid (see
    `get_grid_indices`).

    Returns a dict with `lon`, `lat`, `bathymetry` (2-D, meters positive
    down), `z_center` (per-column layer-center depths, meters positive
    down), and one entry per requested field (excluding `thknss`), all
    cropped to `extent`.

    Fields are downcast to float32 right after cropping (`read_hycom_fields`
    itself always reads full-precision float64) to roughly halve memory use
    for the rest of the pipeline -- this matters at global scale, where each
    full-precision 3-D field is close to 5GB.
    """
    j0, j1, i0, i1, lon2d, lat2d = get_grid_indices(grid_file, extent)

    raw = read_hycom_fields(archv_file, list(fields))
    cropped = {field: data[:, j0:j1, i0:i1].astype(np.float32) for field, data in raw.items()}

    bathy = getBathymetry(depth_file)[j0:j1, i0:i1].astype(np.float32)

    dz_m = cropped['thknss'] / np.float32(RG)
    _, z_center, _ = thickness2depths(dz_m)

    out = {'lon': lon2d, 'lat': lat2d, 'bathymetry': bathy, 'z_center': z_center.astype(np.float32)}
    for field, data in cropped.items():
        if field != 'thknss':
            out[field] = data
    return out


def read_grid_and_depths(archv_file, grid_file, depth_file, extent):
    """
    Like `read_region`, but reads only `thknss` (not the data fields) --
    returns `lon`, `lat`, `bathymetry`, `z_center`, and the crop index
    window `(j0, j1, i0, i1)`. Used by `process_rtofs_region` to read each
    data field separately afterward, so at most one data field (plus
    `thknss`) is ever in memory at once -- this matters at global scale,
    where each full field is close to 5GB before downcasting.
    """
    j0, j1, i0, i1, lon2d, lat2d = get_grid_indices(grid_file, extent)

    thknss = read_hycom_fields(archv_file, ['thknss'])['thknss'][:, j0:j1, i0:i1].astype(np.float32)
    bathy = getBathymetry(depth_file)[j0:j1, i0:i1].astype(np.float32)

    dz_m = thknss / np.float32(RG)
    _, z_center, _ = thickness2depths(dz_m)

    return {
        'lon': lon2d, 'lat': lat2d, 'bathymetry': bathy,
        'z_center': z_center.astype(np.float32),
        'window': (j0, j1, i0, i1),
    }


def read_field(archv_file, field, window):
    """Read one field from an archv file, cropped to `window = (j0, j1, i0, i1)`."""
    j0, j1, i0, i1 = window
    return read_hycom_fields(archv_file, [field])[field][:, j0:j1, i0:i1].astype(np.float32)


def fill_bottom_nans(field, z_center):
    """
    Forward-fill NaNs down each water column (axis 0) with the last valid
    value above them, then fill any fully-NaN (land) column from its left
    neighbor. Port of `sub_fill_bottom_nans.m`, vectorized over the
    horizontal dimensions (no per-point Python loop).
    """
    field = field.copy()
    valid = ~np.isnan(field)

    # Forward-fill along axis 0: at each level, carry the index of the
    # most recent valid level seen so far.
    idx = np.where(valid, np.arange(field.shape[0])[:, None, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    filled = np.take_along_axis(field, idx, axis=0)
    filled = np.where(valid, field, filled)

    # Fully-NaN columns (land) have no valid level to forward-fill from;
    # pull them from the nearest valid column to the left along axis 2.
    fully_land = ~valid.any(axis=0)
    if fully_land.any():
        jj, ii = np.where(fully_land)
        for j, i in zip(jj, ii):
            src_i = i
            while src_i > 0 and fully_land[j, src_i]:
                src_i -= 1
            filled[:, j, i] = filled[:, j, src_i]

    return filled


def _interp_column(z_pad, v_pad, z_levels, method):
    order = np.argsort(z_pad)
    z_pad, v_pad = z_pad[order], v_pad[order]
    # Interpolators below require strictly increasing x.
    keep = np.concatenate(([True], np.diff(z_pad) > 0))
    z_pad, v_pad = z_pad[keep], v_pad[keep]
    if z_pad.size < 2:
        return np.full(z_levels.size, np.nan)

    if method == 'pchip':
        interpolator = PchipInterpolator(z_pad, v_pad, extrapolate=False)
    else:
        interpolator = interp1d(z_pad, v_pad, bounds_error=False)
    return interpolator(z_levels)


def _interp_to_z_chunk(filled, z_center, z_levels, method):
    """Process every column in a (kdm, jdm_chunk, idm) slice. Module-level
    so it can be pickled and sent to worker processes."""
    kdm, jdm, idm = filled.shape
    out = np.full((z_levels.size, jdm, idm), np.nan, dtype=np.float32)
    for j in range(jdm):
        for i in range(idm):
            col_v = filled[:, j, i]
            if np.isnan(col_v).all():
                continue
            col_z = z_center[:, j, i]
            z_pad = np.concatenate(([0.0], col_z, [DEEP_PAD]))
            v_pad = np.concatenate(([col_v[0]], col_v, [col_v[-1]]))
            out[:, j, i] = _interp_column(z_pad, v_pad, z_levels, method)
    return out


def interp_to_z(field, z_center, z_levels, bathymetry, progress=True, desc=None,
                 method='pchip', n_jobs=1):
    """
    Interpolate `field` (kdm, jdm, idm) from per-column hybrid-layer depths
    `z_center` (kdm, jdm, idm) onto the fixed `z_levels` (1-D, meters
    positive down), per grid column.

    Port of `sub_interp2z_2D.m`: each column is first bottom-filled (no NaN
    gap between the last valid layer and the seafloor), padded with a
    synthetic surface point (z=0) and a synthetic deep point (z=DEEP_PAD),
    then interpolated. Output is masked NaN wherever the requested z-level
    is deeper than the column's bathymetry.

    `method` is `'pchip'` (default, shape-preserving cubic, matches the
    matlab reference this module is based on) or `'linear'`. NOAA's own
    `archv2ncdf3z` tool (which produces the RTOFS OPeNDAP product served by
    `ioos_model_comparisons.models.rtofs()`) interpolates linearly between
    layer centers, so `method='linear'` reproduces that product almost
    exactly; `'pchip'` gives smoother profiles but can diverge from it
    (by up to ~0.5 degC/0.2 PSU in this region) wherever the water column
    curves sharply with depth, e.g. across eddy/Loop Current edges.

    This loops over every grid column in Python (one interpolator call per
    column, since each column has its own depth axis) and can take tens of
    seconds for a regional crop, or over an hour for the full globe.
    `n_jobs` controls parallelism: `1` (default) runs single-process with a
    `tqdm` progress bar over rows; `-1` uses all CPU cores; any other
    positive int uses that many worker processes, splitting the grid into
    row-chunks (one `ProcessPoolExecutor` task per chunk, progress bar over
    completed chunks instead of rows).
    """
    if method not in ('pchip', 'linear'):
        raise ValueError("method must be 'pchip' or 'linear'")

    filled = fill_bottom_nans(field, z_center)
    kdm, jdm, idm = filled.shape
    z_levels = np.asarray(z_levels, dtype=float)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, jdm))

    if n_jobs == 1:
        out = np.full((z_levels.size, jdm, idm), np.nan, dtype=np.float32)
        rows = tqdm(range(jdm), desc=desc or f"interp_to_z ({method})", disable=not progress)
        for j in rows:
            for i in range(idm):
                col_v = filled[:, j, i]
                if np.isnan(col_v).all():
                    continue
                col_z = z_center[:, j, i]
                z_pad = np.concatenate(([0.0], col_z, [DEEP_PAD]))
                v_pad = np.concatenate(([col_v[0]], col_v, [col_v[-1]]))
                out[:, j, i] = _interp_column(z_pad, v_pad, z_levels, method)
    else:
        bounds = np.linspace(0, jdm, n_jobs + 1, dtype=int)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _interp_to_z_chunk,
                    filled[:, bounds[k]:bounds[k + 1]],
                    z_center[:, bounds[k]:bounds[k + 1]],
                    z_levels, method,
                ): k
                for k in range(n_jobs)
            }
            chunks = [None] * n_jobs
            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=n_jobs, desc=desc or f"interp_to_z ({method}, {n_jobs} workers)")
            for future in iterator:
                chunks[futures[future]] = future.result()
        out = np.concatenate(chunks, axis=1)

    bottom_mask = z_levels[:, None, None] > bathymetry[None, :, :]
    out[bottom_mask] = np.nan
    out[:, np.isnan(bathymetry)] = np.nan
    return out


def fill_land_nearest(slice2d):
    """
    Fill NaN (land) points in a 2-D lat/lon slice with the value of the
    nearest non-NaN (ocean) point. Vectorized port of `sub_fill_land.m`
    using a Euclidean distance transform instead of a per-point search.
    """
    mask = np.isnan(slice2d)
    if not mask.any():
        return slice2d
    _, indices = distance_transform_edt(mask, return_indices=True)
    return slice2d[tuple(indices)]


def regrid_horizontal(lon2d, lat2d, field_at_z, z_levels, target_lon, target_lat, bathymetry):
    """
    Bilinearly regrid `field_at_z` (nz, jdm, idm) from the native cropped
    grid (`lon2d`, `lat2d`) onto a uniform target grid (`target_lon`,
    `target_lat`, both 1-D), one z-level at a time. Port of the `interp2`
    step in `sub_interp3D_nas.m`: land/below-bottom is filled before
    regridding so it doesn't contaminate nearby ocean values, then
    re-masked NaN afterward using the bathymetry interpolated onto the
    same target grid.
    """
    lon1d, lat1d = lon2d[0, :], lat2d[:, 0]
    rectilinear = (
        np.allclose(lon2d, lon1d[None, :]) and np.allclose(lat2d, lat1d[:, None])
    )

    target_lon2d, target_lat2d = np.meshgrid(target_lon, target_lat)
    bathy_filled = fill_land_nearest(np.where(bathymetry <= 0, np.nan, bathymetry))
    bathy_target = griddata(
        (lon2d.ravel(), lat2d.ravel()), bathy_filled.ravel(),
        (target_lon2d, target_lat2d), method='linear',
    )

    nz = field_at_z.shape[0]
    out = np.full((nz, target_lat.size, target_lon.size), np.nan)
    for k in range(nz):
        filled = fill_land_nearest(field_at_z[k])
        if rectilinear:
            interpolator = RegularGridInterpolator(
                (lat1d, lon1d), filled, bounds_error=False, fill_value=np.nan,
            )
            out[k] = interpolator((target_lat2d, target_lon2d))
        else:
            out[k] = griddata(
                (lon2d.ravel(), lat2d.ravel()), filled.ravel(),
                (target_lon2d, target_lat2d), method='linear',
            )

    below_bottom = np.asarray(z_levels)[:, None, None] > bathy_target[None, :, :]
    out[below_bottom | np.isnan(bathy_target)] = np.nan
    return out, target_lon2d, target_lat2d, bathy_target


def process_rtofs_region(archv_file, grid_file, depth_file,
                          z_levels=RTOFS_STANDARD_Z_LEVELS,
                          region_name=None, extent=None, target_grid=None,
                          fields=('temp', 'salin', 'u-vel.', 'v-vel.'),
                          progress=True, method='pchip', n_jobs=1, whole_globe=False):
    """
    Read a region out of an RTOFS archive file and interpolate it onto fixed
    z-levels, returning an xarray.Dataset.

    `z_levels` defaults to `RTOFS_STANDARD_Z_LEVELS`, matching the depths
    served by the Rutgers RTOFS OPeNDAP product
    (`ioos_model_comparisons.models.rtofs()`), so output from this module is
    directly comparable to data pulled from that source.

    Either `region_name` (looked up via `region_config`), an explicit
    `extent = [lonmin, lonmax, latmin, latmax]`, or `whole_globe=True` must
    be given. `whole_globe=True` skips cropping entirely and processes the
    full native grid (idm=4500, jdm=3298), including the curvilinear Arctic
    patch above ~47N -- that's fine here since, unlike the region-crop path,
    nothing in this mode assumes a regular lat/lon grid.

    If `target_grid` is provided (dict with 1-D `lon`/`lat` arrays), the
    result is also horizontally regridded onto that uniform grid; otherwise
    the result stays on RTOFS's native cropped grid (2-D lon/lat coords).
    Combining `whole_globe=True` with `target_grid` works but falls back to
    slow `griddata` regridding in the curvilinear Arctic region.

    `method` is passed through to `interp_to_z` -- `'pchip'` (default) or
    `'linear'`. `'linear'` reproduces NOAA's own `archv2ncdf3z` tool (and
    therefore the OPeNDAP product) almost exactly; `'pchip'` gives smoother
    profiles but can diverge from it across sharp vertical gradients.

    `n_jobs` is passed through to `interp_to_z` to parallelize the
    column-by-column interpolation across CPU cores (`1` = single process,
    `-1` = all cores). This is the dominant cost at global scale (roughly
    80-90 minutes single-process for one global archive file on a 10-core
    machine, vs. ~15-20 minutes with `n_jobs=-1`), and matters far less for
    small regional crops.

    `progress` (default True) shows a `tqdm` progress bar, one per field,
    while interpolating onto `z_levels` -- this step loops over every grid
    column and can take tens of seconds (regional) to well over an hour
    (global, single-process), so the bar is there to show the call is still
    working rather than hung.
    """
    if whole_globe:
        extent = None
    elif extent is None:
        if region_name is None:
            raise ValueError("Provide region_name, extent, or whole_globe=True")
        extent = region_config(regions=[region_name])['extent']

    data = read_grid_and_depths(archv_file, grid_file, depth_file, extent)
    z_levels = np.asarray(z_levels, dtype=float)

    z_data = {}
    for field in fields:
        raw_field = read_field(archv_file, field, data['window'])
        z_data[field] = interp_to_z(
            raw_field, data['z_center'], z_levels, data['bathymetry'],
            progress=progress, desc=f"interp {field} to z ({method})", method=method,
            n_jobs=n_jobs,
        )
        del raw_field

    if target_grid is not None:
        target_lon = np.asarray(target_grid['lon'])
        target_lat = np.asarray(target_grid['lat'])
        out_vars = {}
        for field, arr in z_data.items():
            regridded, _, _, _ = regrid_horizontal(
                data['lon'], data['lat'], arr, z_levels, target_lon, target_lat, data['bathymetry'],
            )
            out_vars[field] = (('z', 'lat', 'lon'), regridded)
        ds = xr.Dataset(
            out_vars,
            coords={
                'z': z_levels,
                'lon': target_lon,
                'lat': target_lat,
            },
        )
    else:
        lon1d = data['lon'][0, :]
        lat1d = data['lat'][:, 0]
        out_vars = {field: (('z', 'lat', 'lon'), arr) for field, arr in z_data.items()}
        ds = xr.Dataset(
            out_vars,
            coords={
                'z': z_levels,
                'lon': lon1d,
                'lat': lat1d,
            },
        )

    if not ds.indexes['lon'].is_monotonic_increasing:
        ds = ds.sortby('lon')

    return ds
