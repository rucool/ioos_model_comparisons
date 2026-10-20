"""
storm_track.py — storm track + NHC forecast-cone lookup for the Profiles map.

A single tropycal TrackDataset (HURDAT2 + best-track) per basin supplies both
a storm's full position history AND, via Storm.get_nhc_forecast_dict(time),
the NHC forecast advisory nearest a given date — for archived storms, not
just live ones. That's enough to cover "which storm, and what did its
forecast look like, on day X" uniformly for historical and recent dates
without a separate live-NHC-scraping code path.

Archive basins: north_atlantic + east_pacific — covers Gulf of Mexico,
Caribbean, Mid/South Atlantic Bight, Tropical Western Atlantic, West Florida
Shelf, Mexico Pacific, and Hawaii's approach.

Live storms: for dates within REALTIME_DAYS of today, storms from tropycal's
real-time feed are added — every basin NHC or JTWC tracks (West Pacific
typhoons near Guam, South Pacific/Indian cyclones, invests), and storms still
in progress whose best track hasn't been finalized yet. Same approach as the
OHC maps (scripts/maps/models/synchronous/ocean_heat_content.py). The feed
takes ~30 s to load, so it is loaded once and then refreshed in the
background every _REALTIME_TTL seconds; requests never wait on a refresh.

Known gap: past West Pacific / Southern Hemisphere storms (older than
REALTIME_DAYS) don't appear — they'd need the much heavier IBTrACS archive.
"""

from __future__ import annotations

import datetime as dt
import threading
import time as _time
import warnings

import numpy as np

BASINS = ("north_atlantic", "east_pacific")
_BASIN_CACHE_TTL = 12 * 3600  # seconds — long-lived; rebuilt lazily, not on a timer

_basin_cache = {}  # basin_name -> (timestamp, TrackDataset)

REALTIME_DAYS = 2          # dates this recent also get live storms (all basins)
_REALTIME_TTL = 30 * 60    # seconds between background refreshes of the live feed
_realtime = {"ts": 0.0, "storms": None, "refreshing": False}  # storms: [(storm, forecast)]
_realtime_lock = threading.Lock()

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


def _load_realtime_storms():
    """[(RealtimeStorm, forecast dict or None)] for every active NHC/JTWC storm.
    Forecasts are fetched here, once per refresh, not per web request."""
    from tropycal import realtime  # deferred: heavy import

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt = realtime.Realtime(jtwc=True)
        for key in rt.list_active_storms(basin="all"):
            try:
                storm = rt.get_storm(key)
            except Exception as e:
                print(f"storm_track: couldn't load live storm {key}: {e}")
                continue
            forecast = None
            if not getattr(storm, "invest", False):  # invests have no advisories
                try:
                    # No argument: tropycal 1.3's first parameter is an SSL
                    # certificate path, so passing True breaks every JTWC fetch.
                    forecast = storm.get_forecast_realtime() or None
                except Exception as e:
                    print(f"storm_track: no live forecast for {key} ({e}); track only")
            out.append((storm, forecast))
    return out


def _refresh_realtime():
    try:
        _realtime["storms"] = _load_realtime_storms()
    except Exception as e:
        print(f"storm_track: live storm feed failed: {e}")
        if _realtime["storms"] is None:
            _realtime["storms"] = []  # retry after _REALTIME_TTL rather than on every request
    finally:
        _realtime["ts"] = _time.time()
        _realtime["refreshing"] = False


def realtime_storms():
    """[(storm, forecast)] from the live feed. The first call in a process
    loads it (concurrent first callers wait for that one load); later calls
    return the last list immediately and refresh it in the background once
    it is older than _REALTIME_TTL."""
    with _realtime_lock:
        if _realtime["storms"] is None:
            _refresh_realtime()
            return _realtime["storms"]
        if _time.time() - _realtime["ts"] > _REALTIME_TTL and not _realtime["refreshing"]:
            _realtime["refreshing"] = True
            threading.Thread(target=_refresh_realtime, daemon=True).start()
        return _realtime["storms"]


def is_recent(date_obj, today=None):
    today = today or dt.datetime.now(dt.timezone.utc).date()
    return (today - date_obj).days <= REALTIME_DAYS


def active_storms_for_date(date_obj, lookback_days=5, lookahead_days=1):
    """Storms from BASINS whose track overlapped a window around date_obj.

    Adapted from ioos_model_comparisons/plotting.py::get_active_storms, minus
    its map-extent intersection filter — the Profiles map is one merged,
    un-filtered world view (confirmed: no per-region selection exists there
    any more), so date overlap is the only filter that makes sense here.

    Returns a list of (basin_name, Storm, forecast-or-None) tuples; forecast is
    None for archive storms (looked up per date in storm_payload). For recent
    dates, live-feed storms are merged in and replace the archive copy of the
    same storm.
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
                    found.append((basin_name, storm, None))
                except Exception as e:
                    print(f"storm_track: error checking storm {storm_id}: {e}")
                    continue

    if is_recent(date_obj):
        by_id = {entry[1].id: entry for entry in found}
        for storm, forecast in realtime_storms():
            times = storm.dict.get("time") or []
            if not times or max(times) < time_start or min(times) > time_end:
                continue
            by_id[storm.id] = (storm.basin, storm, forecast or {})
        found = list(by_id.values())

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


def storm_payload(storm, basin_name, date_obj, forecast=None):
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

    if forecast is None:            # archive storm: NHC advisory nearest date_obj
        forecast = _get_forecast(storm, date_obj)

    return {
        "name": storm.name,
        "basin": basin_name,
        "invest": bool(getattr(storm, "invest", False)),
        "category": category,
        "category_label": ("Invest" if getattr(storm, "invest", False)
                           else _CATEGORY_LABELS.get(category, "Unknown")),
        "color": get_colors_sshws(peak_vmax),
        "track": track,
        "forecast": _forecast_points(forecast) if forecast else [],
        "cone": _cone_polygon(forecast, basin_name, storm.name) if forecast else None,
    }


def storms_for_date(date_obj):
    """{storm_id: storm_payload} for every storm active near date_obj."""
    out = {}
    for basin_name, storm, forecast in active_storms_for_date(date_obj):
        payload = storm_payload(storm, basin_name, date_obj, forecast)
        if payload:
            out[storm.id] = payload
    return out
