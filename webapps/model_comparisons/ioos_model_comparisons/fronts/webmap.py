"""
webmap.py — helpers for putting the digitized front on a Leaflet map.

Two jobs:

1. `write_overlay_png` — render a field as a bare, georeferenced PNG suitable
   for `L.imageOverlay`. Deliberately NOT done with cartopy: Leaflet places an
   image overlay by mapping its corners to lat/lon in *screen* space, and
   Leaflet's screen space is Web Mercator. An equirectangular image stretched
   to lat/lon bounds is therefore misregistered in latitude — across a 32-45N
   domain the error is tens of km, worst at the edges. So the field is
   explicitly resampled onto a grid that is uniform in Mercator y before the
   PNG is written, which makes the overlay pixel-correct.

2. `simplify_lines` — Douglas-Peucker decimation for browser editing. A traced
   wall carries ~3000 vertices; handed to a map editor that is ~3000 draggable
   handles and the page is unusable. Douglas-Peucker is used rather than
   uniform decimation because it keeps vertices where the line actually bends
   and drops them along straight runs.

Requires: numpy, matplotlib. `simplify_lines` prefers shapely and falls back to
a pure-numpy implementation if it is missing.
"""

from __future__ import annotations

import numpy as np

# Web Mercator is only defined to ~85.05 deg; the Gulf Stream domain is far
# from that, but clamp anyway so a bad extent fails visibly rather than
# producing infinities.
_MERC_LAT_LIMIT = 85.05112878


def _merc_y(lat):
    """Web Mercator y for a latitude in degrees (unitless, y increases north)."""
    lat = np.clip(np.asarray(lat, float), -_MERC_LAT_LIMIT, _MERC_LAT_LIMIT)
    return np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))


def _merc_y_inverse(y):
    """Latitude in degrees from a Web Mercator y."""
    return np.degrees(2.0 * np.arctan(np.exp(np.asarray(y, float))) - np.pi / 2.0)


def render_overlay_png(field, lats, lons, extent, cmap="turbo", vmin=None,
                       vmax=None, height=1400, stride=None):
    """Render to PNG BYTES in Web Mercator. See write_overlay_png for the
    projection reasoning; this is the same routine without a file.

    `stride`, if given, quantises the colour scale into discrete bands of that
    width instead of a continuous ramp — matching how the project's matplotlib
    figures use `levels = arange(vmin, vmax + stride, stride)`, so a web
    overlay and a published figure of the same field look the same.
    """
    import io
    import matplotlib.image as mpimage
    from matplotlib import cm, colors

    lon0, lon1, lat0, lat1 = [float(v) for v in extent]
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    Z = np.asarray(field, float)

    if lats.size > 1 and lats[0] > lats[-1]:
        lats, Z = lats[::-1], Z[::-1, :]
    if lons.size > 1 and lons[0] > lons[-1]:
        lons, Z = lons[::-1], Z[:, ::-1]

    y0, y1 = _merc_y(lat0), _merc_y(lat1)
    width = max(2, int(round(height * (lon1 - lon0) / np.degrees(y1 - y0))))
    ys = np.linspace(y1, y0, height)
    target_lats = _merc_y_inverse(ys)
    target_lons = np.linspace(lon0, lon1, width)

    def _nearest(grid, want):
        i = np.clip(np.searchsorted(grid, want), 0, grid.size - 1)
        lo = np.clip(i - 1, 0, grid.size - 1)
        return np.where(np.abs(grid[lo] - want) < np.abs(grid[i] - want), lo, i)

    out = Z[np.ix_(_nearest(lats, target_lats), _nearest(lons, target_lons))]
    finite = np.isfinite(out)
    if vmin is None:
        vmin = float(np.nanmin(out)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(out)) if finite.any() else 1.0

    base = resolve_cmap(cmap)
    if stride and stride > 0 and (vmax - vmin) / stride <= 256:
        levels = np.arange(vmin, vmax + stride / 2.0, stride)
        norm = colors.BoundaryNorm(levels, ncolors=base.N, clip=True)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    rgba = cm.ScalarMappable(norm=norm, cmap=base).to_rgba(
        np.where(finite, out, vmin), bytes=True)
    rgba[..., 3] = np.where(finite, 255, 0)

    buf = io.BytesIO()
    mpimage.imsave(buf, rgba, format="png")
    return buf.getvalue(), {"extent": [lon0, lon1, lat0, lat1],
                            "width": int(width), "height": int(height),
                            "vmin": float(vmin), "vmax": float(vmax),
                            "stride": float(stride) if stride else None,
                            "cmap": str(cmap)}


def resolve_cmap(name):
    """Look up a colormap by name, including cmocean's `cmo.*` names.

    Falls back to turbo rather than raising: a bad name in a URL query string
    should not 500 an image request.
    """
    from matplotlib import cm, colors
    if isinstance(name, colors.Colormap):
        return name
    name = str(name or "turbo")
    if name.startswith("cmo."):
        try:
            import cmocean
            return getattr(cmocean.cm, name[4:])
        except Exception:
            return cm.get_cmap("turbo")
    try:
        import matplotlib.pyplot as plt
        return plt.get_cmap(name)
    except Exception:
        return cm.get_cmap("turbo")


def write_overlay_png(field, lats, lons, extent, path, cmap="turbo",
                      vmin=None, vmax=None, height=1400):
    """Write `field` as a transparent-background PNG in Web Mercator.

    Parameters
    ----------
    field : 2-D array (lat, lon), NaN where missing
    lats, lons : 1-D coordinate arrays matching `field`
    extent : [lon0, lon1, lat0, lat1] — the bounds the PNG will span exactly,
        i.e. what you pass to L.imageOverlay as [[lat0, lon0], [lat1, lon1]]
    path : output .png path
    height : output pixel height; width is derived to keep Mercator aspect
        square (so the image is not pre-distorted).

    Returns the extent actually written, for the caller to record.
    """
    # import the submodule explicitly: `import matplotlib` alone does not bind
    # matplotlib.image, and this module must not depend on some caller having
    # imported pyplot first
    import matplotlib.image as mpimage
    from matplotlib import cm, colors

    lon0, lon1, lat0, lat1 = [float(v) for v in extent]
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    Z = np.asarray(field, float)

    # source may be stored either N->S or S->N; normalise to ascending
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        Z = Z[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        Z = Z[:, ::-1]

    # Target grid: uniform in Mercator y (rows) and in longitude (columns).
    # Row 0 is the TOP of the image, i.e. the northern edge, matching how PNG
    # rows are read and how Leaflet anchors the overlay.
    y0, y1 = _merc_y(lat0), _merc_y(lat1)
    width = max(2, int(round(height * (lon1 - lon0) / np.degrees(y1 - y0))))
    ys = np.linspace(y1, y0, height)          # north -> south
    target_lats = _merc_y_inverse(ys)
    target_lons = np.linspace(lon0, lon1, width)

    # Nearest-neighbour lookup. Deliberate: the field already carries NaN holes
    # (cloud), and bilinear interpolation would bleed those NaNs outward and
    # eat real data along every hole edge.
    #
    # searchsorted alone gives a CEILING index, which biases every row half a
    # grid cell north (measured ~1.4 km before this correction); step back one
    # index where the lower neighbour is actually closer.
    def _nearest(grid, want):
        i = np.clip(np.searchsorted(grid, want), 0, grid.size - 1)
        lo = np.clip(i - 1, 0, grid.size - 1)
        take_lo = np.abs(grid[lo] - want) < np.abs(grid[i] - want)
        return np.where(take_lo, lo, i)

    ri = _nearest(lats, target_lats)
    ci = _nearest(lons, target_lons)
    out = Z[np.ix_(ri, ci)]

    finite = np.isfinite(out)
    if vmin is None:
        vmin = float(np.nanmin(out)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(out)) if finite.any() else 1.0

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(np.where(finite, out, 0.0), bytes=True)
    rgba[..., 3] = np.where(finite, 255, 0)   # missing data fully transparent

    mpimage.imsave(str(path), rgba, format="png")
    return {"extent": [lon0, lon1, lat0, lat1],
            "width": int(width), "height": int(height),
            "vmin": float(vmin), "vmax": float(vmax), "cmap": str(cmap)}


def _dp_mask(pts, tol):
    """Douglas-Peucker keep-mask, iterative (no recursion limit worries)."""
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[a:b + 1]
        p0, p1 = pts[a], pts[b]
        d = p1 - p0
        L = np.hypot(*d)
        if L == 0:
            dist = np.hypot(*(seg - p0).T)
        else:
            # perpendicular distance to the chord
            dist = np.abs(np.cross(np.broadcast_to(d, seg.shape), seg - p0)) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + i] = True
            stack.append((a, a + i))
            stack.append((a + i, b))
    return keep


def simplify_lines(lines, tolerance_deg=0.02, max_points=None):
    """Douglas-Peucker each (N, 2) lon/lat line.

    `max_points`, if given, loosens the tolerance until the total vertex count
    across all lines fits — the editor needs a bound it can actually render,
    and a fixed tolerance gives wildly different counts on a smooth day versus
    a meandering one.
    """
    try:
        from shapely.geometry import LineString

        def _simplify(arr, tol):
            if len(arr) < 3:
                return arr
            g = LineString(arr).simplify(tol, preserve_topology=False)
            return np.asarray(g.coords)
    except ImportError:
        def _simplify(arr, tol):
            if len(arr) < 3:
                return arr
            return arr[_dp_mask(np.asarray(arr, float), tol)]

    tol = float(tolerance_deg)
    for _ in range(24):
        out = [_simplify(np.asarray(l, float), tol) for l in lines]
        total = sum(len(o) for o in out)
        if max_points is None or total <= max_points or total <= 2 * len(out):
            return out, tol
        tol *= 1.4
    return out, tol
