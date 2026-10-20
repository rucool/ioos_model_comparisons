"""
storm_track.py — storm track + NHC forecast-cone lookup for the Profiles map.

A single tropycal TrackDataset (HURDAT2 + best-track) per basin supplies both
a storm's full position history AND, via Storm.get_nhc_forecast_dict(time),
the NHC forecast advisory nearest a given date — for archived storms, not
just live ones. That's enough to cover "which storm, and what did its
forecast look like, on day X" uniformly for historical and recent dates
without a separate live-NHC-scraping code path.

Basins: north_atlantic + east_pacific only (v1) — covers Gulf of Mexico,
Caribbean, Mid/South Atlantic Bight, Tropical Western Atlantic, West Florida
Shelf, Mexico Pacific, and Hawaii's approach. Guam/Fiji (west/south Pacific)
show no storms for now; tropycal's non-HURDAT2 basins are less mature and
those two regions need this least.

Known gap: best-track finalization lags a storm actively in progress by up
to a day or two, so a storm from the last day or so may not appear yet. The
Profiles tab is an archive-browsing tool ("Navigate Days"), not a live
tracker, so this is an acceptable v1 limitation rather than a live
tropycal.realtime path (see scripts/tools/overview_powerpoint.py for how
that would look if it's ever needed).
"""

from __future__ import annotations

import datetime as dt
import time as _time
import warnings

import numpy as np

BASINS = ("north_atlantic", "east_pacific")
_BASIN_CACHE_TTL = 12 * 3600  # seconds — long-lived; rebuilt lazily, not on a timer

_basin_cache = {}  # basin_name -> (timestamp, TrackDataset)

# Saffir-Simpson category -> label. -1/0 come from tropycal's wind_to_category().
_CATEGORY_LABELS = {
    -1: "Tropical Depression",
    0: "Tropical Storm",
    1: "Category 1",
    2: "Category 2",
    3: "Category 3",
    4: "Category 4",
    5: "Category 5",
}


def _get_basin(basin_name):
    now = _time.time()
    cached = _basin_cache.get(basin_name)
    if cached and now - cached[0] < _BASIN_CACHE_TTL:
        return cached[1]

    from tropycal import tracks  # deferred: heavy import, only needed once a request lands here

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = tracks.TrackDataset(basin=basin_name, include_btk=True)
    _basin_cache[basin_name] = (now, dataset)
    return dataset


def active_storms_for_date(date_obj, lookback_days=5, lookahead_days=1):
    """Storms from BASINS whose track overlapped a window around date_obj.

    Adapted from ioos_model_comparisons/plotting.py::get_active_storms, minus
    its map-extent intersection filter — the Profiles map is one merged,
    un-filtered world view (confirmed: no per-region selection exists there
    any more), so date overlap is the only filter that makes sense here.

    Returns a list of (basin_name, Storm) tuples.
    """
    time_start = dt.datetime.combine(date_obj, dt.time.min) - dt.timedelta(days=lookback_days)
    time_end = dt.datetime.combine(date_obj, dt.time.min) + dt.timedelta(days=lookahead_days)

    years = {date_obj.year}
    if date_obj.month == 1:
        years.add(date_obj.year - 1)
    if date_obj.month == 12:
        years.add(date_obj.year + 1)

    found = []
    for basin_name in BASINS:
        try:
            basin = _get_basin(basin_name)
        except Exception as e:
            print(f"storm_track: couldn't load basin {basin_name}: {e}")
            continue

        for year in years:
            try:
                season = basin.get_season(year)
            except Exception:
                continue

            for storm_id in season.summary()["id"]:
                try:
                    storm = basin.get_storm(storm_id)
                    storm_times = storm.dict["time"]
                    if not storm_times:
                        continue
                    if max(storm_times) < time_start or min(storm_times) > time_end:
                        continue
                    found.append((basin_name, storm))
                except Exception as e:
                    print(f"storm_track: error checking storm {storm_id}: {e}")
                    continue

    return found


def _get_forecast(storm, date_obj):
    """NHC forecast advisory dict nearest date_obj, or None if none exists
    (true for most of a historical storm's lifetime — NHC only issues
    forecasts while a storm is active)."""
    try:
        forecast = storm.get_nhc_forecast_dict(dt.datetime.combine(date_obj, dt.time(hour=12)))
    except Exception:
        return None
    return forecast or None


def _forecast_points(forecast):
    """[{time, lat, lon, vmax, fhr}, ...] — the discrete advisory points
    (fhr=0,3,12,24,...) a forecast dict is built from, distinct from the
    smoothed/gridded cone polygon derived from the same dict below."""
    init = forecast["init"]
    points = []
    for fhr, lat, lon, vmax in zip(forecast["fhr"], forecast["lat"], forecast["lon"], forecast["vmax"]):
        if lat is None or lon is None:
            continue
        points.append({
            "time": (init + dt.timedelta(hours=fhr)).isoformat(),
            "lat": float(lat),
            "lon": float(lon),
            "vmax": None if vmax is None or np.isnan(vmax) else float(vmax),
            "fhr": fhr,
        })
    return points


def _cone_polygon(forecast, basin_name, storm_name):
    """[[lat, lon], ...] boundary of the NHC forecast cone for an already-
    fetched forecast dict, or None if it can't be generated."""
    from tropycal.utils.generic_utils import generate_nhc_cone

    try:
        cone = generate_nhc_cone(forecast, basin_name)
    except Exception as e:
        print(f"storm_track: couldn't generate cone for {storm_name}: {e}")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        cs = plt.contour(cone["lon2d"], cone["lat2d"], cone["cone"], levels=[0.5])
        segs = cs.allsegs[0] if hasattr(cs, "allsegs") else [p.vertices for p in cs.collections[0].get_paths()]
        if not segs:
            return None
        # Largest ring only, and decimated — a full-resolution cone boundary
        # is 1000+ points, far more detail than a web map needs.
        boundary = max(segs, key=len)
        return [[float(lat), float(lon)] for lon, lat in boundary[::4]]
    except Exception as e:
        print(f"storm_track: couldn't extract cone polygon for {storm_name}: {e}")
        return None
    finally:
        plt.close(fig)


def storm_payload(storm, basin_name, date_obj):
    """JSON-ready dict for one storm, trimmed to track points through date_obj."""
    d = storm.dict
    cutoff = dt.datetime.combine(date_obj, dt.time.max)

    track = [
        {"time": t.isoformat(), "lat": float(lat), "lon": float(lon), "vmax": (None if vmax is None or np.isnan(vmax) else float(vmax)), "type": typ}
        for t, lat, lon, vmax, typ in zip(d["time"], d["lat"], d["lon"], d["vmax"], d["type"])
        if t <= cutoff
    ]
    if not track:
        return None

    from tropycal.utils.generic_utils import wind_to_category
    from tropycal.utils.colors import get_colors_sshws

    peak_vmax = max((p["vmax"] for p in track if p["vmax"] is not None), default=0)
    category = wind_to_category(peak_vmax)

    forecast = _get_forecast(storm, date_obj)

    return {
        "name": storm.name,
        "basin": basin_name,
        "category": category,
        "category_label": _CATEGORY_LABELS.get(category, "Unknown"),
        "color": get_colors_sshws(peak_vmax),
        "track": track,
        "forecast": _forecast_points(forecast) if forecast else [],
        "cone": _cone_polygon(forecast, basin_name, storm.name) if forecast else None,
    }


def storms_for_date(date_obj):
    """{storm_id: storm_payload} for every storm active near date_obj."""
    out = {}
    for basin_name, storm in active_storms_for_date(date_obj):
        payload = storm_payload(storm, basin_name, date_obj)
        if payload:
            out[storm.id] = payload
    return out
