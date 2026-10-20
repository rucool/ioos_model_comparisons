"""
discovery.py — how the portal finds published figures.

Two backends answer the same questions ("which comparison images exist for
this region/variable/time?", "what's the newest date with Argo profiles?"):

  catalog  one MongoDB query against the figure catalog (hurricanes.figures,
           see ioos_model_comparisons/catalog.py) — no HTTP round trips;
  legacy   the original approach: predict each image URL from the naming
           convention and HEAD-request it, and regex rucool's Apache
           directory listings for dates and IDs.

DISCOVERY_BACKEND (environment) picks:
  legacy   (default) legacy only
  shadow   answer from legacy, but also query the catalog and log any
           difference to the "discovery.shadow" logger — run this for a while
           after the backfill to confirm the catalog is complete
  auto     catalog, falling back to legacy per request when MongoDB is down,
           a query fails, or the catalog is stale (site.yml stale_after_hours)
  catalog  catalog only

Response shapes are identical either way; app.py's routes don't care which
backend answered.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

import requests

from ioos_model_comparisons import catalog
from ioos_model_comparisons.registry import load_registry

log = logging.getLogger(__name__)
shadow_log = logging.getLogger("discovery.shadow")

MODES = ("legacy", "shadow", "auto", "catalog")


def mode() -> str:
    m = (os.getenv("DISCOVERY_BACKEND") or "legacy").strip().lower()
    return m if m in MODES else "legacy"


_health = {"ts": 0.0, "ok": False}
_HEALTH_TTL = 60


def catalog_healthy() -> bool:
    now = time.time()
    if now - _health["ts"] < _HEALTH_TTL:
        return _health["ok"]
    try:
        fresh = catalog.freshness()
        ok = fresh is not None and not catalog.is_stale(fresh)
    except Exception as e:
        log.warning("catalog health check failed: %s", e)
        ok = False
    _health.update(ts=now, ok=ok)
    return ok


def use_catalog() -> bool:
    m = mode()
    return m == "catalog" or (m == "auto" and catalog_healthy())


def backend_name() -> str:
    return "catalog" if use_catalog() else "legacy"


def check_image(url) -> bool:
    if not url:
        return False
    try:
        return requests.head(url, timeout=5).status_code == 200
    except Exception:
        return False


def _check_many(urls) -> dict:
    urls = [u for u in dict.fromkeys(urls) if u]
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as ex:
        return dict(zip(urls, ex.map(check_image, urls)))


def _rel_of(url):
    base = load_registry().site.public_base_url.rstrip("/") + "/"
    return unquote(url[len(base):]) if url and url.startswith(base) else None


def run(name, catalog_fn, legacy_fn, compare=None):
    """Run the configured backend; in auto mode fall back to legacy on any
    catalog error; in shadow mode also run the catalog and log differences."""
    m = mode()
    if m in ("catalog", "auto") and use_catalog():
        try:
            return catalog_fn()
        except Exception as e:
            log.warning("catalog %s failed (%s); using legacy discovery", name, e)
            if m == "catalog":
                raise
    result = legacy_fn()
    if m == "shadow":
        try:
            other = catalog_fn()
            a, b = (compare(result), compare(other)) if compare else (result, other)
            if a != b:
                shadow_log.info("%s differs: legacy=%r catalog=%r", name, a, b)
        except Exception as e:
            shadow_log.info("%s: catalog error %s", name, e)
    return result


# --------------------------------------------------------------------------
# Comparison maps (Maps Archive + Overview)
# --------------------------------------------------------------------------
def _valid_time(date_obj, time_str):
    hour = int(str(time_str).replace("Z", "") or 0)
    return dt.datetime(date_obj.year, date_obj.month, date_obj.day, hour)


def map_panels(region_folder, region_name, var_depth, date_obj, time_str) -> dict:
    """{panel_key: {"url", "available", "label"[, "rtofs"]}} for every panel
    in comparisons.yml (inapplicable panels: url None, available False)."""
    reg = load_registry()
    valid = _valid_time(date_obj, time_str)
    applicable = reg.map_panels(region_folder, var_depth)

    def _base():
        return {p.key: {"url": None, "available": False, "label": p.label} for p in reg.panels}

    def _candidates(panel):
        return [c for c in panel.comparisons
                if reg.comparison_applies(c, region_folder, var_depth)]

    def from_catalog():
        out = _base()
        slot = catalog.find_for_slot(region_folder, var_depth, valid)
        for p in applicable:
            cands = _candidates(p)
            hit = next((slot[c] for c in cands if c in slot), None)
            if hit:
                out[p.key].update(url=hit["url"], available=True)
                if hit.get("rtofs"):
                    out[p.key]["rtofs"] = hit["rtofs"]
            elif cands:
                rel = reg.predict_rel(cands[0], region_folder, region_name, var_depth, valid)
                out[p.key]["url"] = reg.public_url(rel) if rel else None
        return out

    def from_legacy():
        out = _base()
        urls = {}
        for p in applicable:
            urls[p.key] = [reg.public_url(r) for r in
                           (reg.predict_rel(c, region_folder, region_name, var_depth, valid)
                            for c in _candidates(p)) if r]
        ok = _check_many(u for us in urls.values() for u in us)
        for key, us in urls.items():
            hit = next((u for u in us if ok.get(u)), None)
            out[key].update(url=hit or (us[0] if us else None), available=bool(hit))
        return out

    return run(
        "map_panels", from_catalog, from_legacy,
        compare=lambda r: {k: v["available"] for k, v in r.items()})


def latest_map(region_folder, var_depth, comparisons=None):
    """(YYYY-MM-DD, 'HHZ') of the newest map for this region/var_depth."""
    def from_catalog():
        t = catalog.latest_valid_time(region_folder, var_depth, comparisons)
        return (t.strftime("%Y-%m-%d"), f"{t:%H}Z") if t else None

    def from_legacy():
        return _scrape_latest(f"maps/{region_folder}/{var_depth}/",
                              r"(\d{4}-\d{2}-\d{2})T(\d{6})Z", with_hour=True)

    return run("latest_map", from_catalog, from_legacy)


def _scrape_latest(prefix, pattern, with_hour=False):
    """Newest timestamp under <base>/<prefix>/YYYY/MM/ from Apache listings."""
    base = load_registry().site.public_base_url.rstrip("/") + "/" + prefix
    try:
        r = requests.get(base, timeout=6)
        years = sorted(re.findall(r'href="(\d{4})/"', r.text), reverse=True)
        if not years:
            return None
        r = requests.get(f"{base}{years[0]}/", timeout=6)
        months = sorted(re.findall(r'href="(\d{2})/"', r.text), reverse=True)
        if not months:
            return None
        r = requests.get(f"{base}{years[0]}/{months[0]}/", timeout=6)
        matches = sorted(set(re.findall(pattern, r.text)), reverse=True)
        if not matches:
            return None
        if with_hour:
            return matches[0][0], matches[0][1][:2] + "Z"
        return matches[0]
    except Exception:
        return None


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
def _day_bounds(date_obj):
    start = dt.datetime(date_obj.year, date_obj.month, date_obj.day)
    return start, start + dt.timedelta(days=1)


def _scrape_pngs(url):
    try:
        r = requests.get(url, timeout=8)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    return re.findall(r'href="([^"]+\.png)"', r.text)


def glider_ids(date_obj) -> list:
    def from_catalog():
        docs = catalog.platform_figures("glider", *_day_bounds(date_obj), extra={"group": "gliders"})
        return sorted({d["platform"]["id"] for d in docs})

    def from_legacy():
        base = load_registry().public_url(f"profiles/gliders/{date_obj.year}/{date_obj:%m-%d}/")
        return sorted({f.split("_20")[0] for f in _scrape_pngs(base)})

    return run("glider_ids", from_catalog, from_legacy)


def platform_ids(ptype, date_obj, folders) -> dict:
    """{platform_id: {"filename", "region"}} for argo/fvon across folders."""
    def from_catalog():
        docs = catalog.platform_figures(ptype, *_day_bounds(date_obj), regions=folders)
        return {d["platform"]["id"]: {"filename": d["_id"].rsplit("/", 1)[1], "region": d["region"]}
                for d in docs}

    def from_legacy():
        reg = load_registry()

        def one(folder):
            url = reg.public_url(f"profiles/{ptype}/{folder}/{date_obj:%Y/%m/%d}/")
            split = (lambda f: f.split("-profile-")[0]) if ptype == "fvon" else (lambda f: f.split("-")[0])
            return {split(f): {"filename": f, "region": folder} for f in _scrape_pngs(url)}

        out = {}
        with ThreadPoolExecutor(max_workers=max(1, len(folders))) as ex:
            for part in ex.map(one, folders):
                out.update(part)
        return out

    return run(f"{ptype}_ids", from_catalog, from_legacy,
                          compare=lambda r: sorted(r))


def latest_platform_date(ptype, folders=None, legacy_fn=None):
    """Newest YYYY-MM-DD with profile figures for ptype."""
    def from_catalog():
        extra = {"group": "gliders"} if ptype == "glider" else None
        d = catalog.latest_platform_day(ptype, regions=folders, extra=extra)
        return d.strftime("%Y-%m-%d") if d else None

    return run(f"{ptype}_latest_date", from_catalog, legacy_fn or (lambda: None))


def url_available(url) -> bool:
    """Is this published figure there? Catalog lookup by path, else HEAD."""
    def from_catalog():
        rel = _rel_of(url)
        return bool(rel) and catalog.find_one({"_id": rel}) is not None

    return run("url_available", from_catalog, lambda: check_image(url))


# --------------------------------------------------------------------------
# Adaptive sampling guidance
# --------------------------------------------------------------------------
def asg_latest_date(legacy_fn):
    def from_catalog():
        d = catalog.find_one({"family": "adaptive"}, sort=[("valid_time", -1)])
        return d["valid_time"].strftime("%Y-%m-%d") if d else None

    return run("asg_latest_date", from_catalog, legacy_fn)


# --------------------------------------------------------------------------
# Profile positions and glider tracks (from the positions plot scripts record)
# --------------------------------------------------------------------------
_ID_KEY = {"glider": "glider_id", "argo": "wmo", "fvon": "wigos"}


def catalog_positions(date_obj, argo_folders, fvon_folders):
    """({type: {filename: entry}}, {type: set(filenames without a position)})
    in /api/profile-positions' shape, from the catalog. Raises if unavailable."""
    start, end = _day_bounds(date_obj)
    day = date_obj.strftime("%Y-%m-%d")
    queries = {
        "glider": dict(extra={"group": "gliders"}),
        "argo": dict(regions=argo_folders),
        "fvon": dict(regions=fvon_folders),
    }
    out, missing = {}, {}
    for ptype, kw in queries.items():
        out[ptype], missing[ptype] = {}, set()
        for d in catalog.platform_figures(ptype, start, end, **kw):
            plat, fname = d.get("platform") or {}, d["_id"].rsplit("/", 1)[1]
            if plat.get("lat") is None or plat.get("lon") is None:
                missing[ptype].add(fname)
                continue
            out[ptype][fname] = {"lat": plat["lat"], "lon": plat["lon"],
                                 _ID_KEY[ptype]: plat.get("id"), "time": day,
                                 "url": d["url"], "type": ptype}
    return out, missing


def catalog_glider_tracks(end_date, days=5):
    """{glider_id: [{lat, lon, time}, ...]} for the `days` days ending on
    end_date, from catalog glider profile positions. Raises if unavailable."""
    start = _day_bounds(end_date - dt.timedelta(days=days - 1))[0]
    end = _day_bounds(end_date)[1]
    tracks = {}
    for d in catalog.platform_figures("glider", start, end, extra={"group": "gliders"}):
        plat = d.get("platform") or {}
        if plat.get("lat") is None or plat.get("lon") is None:
            continue
        tracks.setdefault(plat["id"], []).append(
            {"lat": plat["lat"], "lon": plat["lon"], "time": d["valid_time"].strftime("%Y-%m-%d")})
    for pts in tracks.values():
        pts.sort(key=lambda p: p["time"])
    return tracks
