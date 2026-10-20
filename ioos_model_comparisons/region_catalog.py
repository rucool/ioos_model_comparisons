"""
region_catalog.py — the canonical list of regions.py-defined regions, and a
helper that builds the effective per-region config (regions.py defaults with
any MongoDB region_configs override layered on top) for every one of them.

Why this exists: webapps/model_comparisons/app.py used to keep its own
hand-typed region_info/argo_regions dicts — a frozen snapshot of whatever
regions.py looked like whenever someone last copied it over by hand. Neither
a regions.py edit nor a save through the new /regions web editor ever showed
up on the website as a result. build_region_info() instead reads the same
live source every offline plotting script already reads — regions.region_config()
with db.apply_colorbar_overrides() layered on top — so the website, the region
editor, and the plotting scripts can no longer drift apart.

scripts/tools/seed_region_configs.py imports ALL_REGIONS from here too, so
there is exactly one list of "the regions that exist" to keep in sync with
regions.py, not two.
"""

from __future__ import annotations

import logging
import time

from ioos_model_comparisons.db import apply_colorbar_overrides
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.registry import load_registry

logger = logging.getLogger(__name__)

# Every region defined in config/regions/*.yml, in their `order`. Derived, not
# hand-typed: adding a region file adds it to MongoDB seeding, the website's
# region_info/argo_regions, and every offline plotting script.
ALL_REGIONS = load_registry().region_keys()


def build_region_config(key):
    """The effective config for *key*: regions.py defaults with any
    MongoDB region_configs override applied — the same document every
    offline plotting script sees via db.apply_colorbar_overrides()."""
    return apply_colorbar_overrides(key, region_config([key]))


def _fmt_depths(nums):
    """["0m", "150m", ...] from an iterable of numeric depths, deduped and
    numerically (not lexically) sorted."""
    return sorted({f"{int(n)}m" for n in nums}, key=lambda s: int(s[:-1]))


# regions.py's ocean_heat_content field is a *colorbar-limit override* flag
# (set once someone tunes custom limits for that region), not an "OHC maps
# exist" flag — it's None for most regions until tuned. Fiji is a confirmed
# exception: its OHC maps are fully published (verified live 2026-08-24,
# https://.../maps/fiji/ocean_heat_content/ returns real files) despite
# never having been tuned, unlike amazon_plume/hurricane_alley/south_africa/
# gulf_stream, whose ocean_heat_content directories genuinely 404. Force it
# on for known exceptions like this rather than trusting the config field.
_OHC_AVAILABLE_OVERRIDES = {"fiji"}


def build_region_info():
    """{display_name: {"variables": [...], "depths": {var: ["0m", ...]},
    "key": regions key, "folder": region folder}}
    for every region in ALL_REGIONS, live from regions.py + MongoDB.

    Mirrors the webapps/model_comparisons UI contract exactly (verified
    against its previous hand-typed region_info): temperature and salinity
    are always listed, even with an empty depth list (mastr's temperature
    has no depth entries in regions.py but is still a selectable variable);
    ocean_heat_content has no depth dimension and is only listed when
    configured (or overridden, see _OHC_AVAILABLE_OVERRIDES); currents is
    only listed when currents.bool is True, with its own depth list —
    frequently different from temperature/salinity's.
    """
    info = {}
    for key in ALL_REGIONS:
        try:
            cfg = build_region_config(key)
        except Exception as exc:
            logger.warning(f"build_region_info: skipping '{key}': {exc}")
            continue

        name = cfg.get("name")
        if not name:
            logger.warning(f"build_region_info: '{key}' has no name, skipping")
            continue

        variables_cfg = cfg.get("variables") or {}
        temperature = [d["depth"] for d in (variables_cfg.get("temperature") or []) if "depth" in d]
        salinity = [d["depth"] for d in (variables_cfg.get("salinity") or []) if "depth" in d]

        variables = ["temperature", "salinity"]
        depths = {
            "temperature": _fmt_depths(temperature),
            "salinity": _fmt_depths(salinity),
        }

        if cfg.get("ocean_heat_content") or key in _OHC_AVAILABLE_OVERRIDES:
            variables.append("ocean_heat_content")

        currents = cfg.get("currents") or {}
        if currents.get("bool"):
            variables.append("currents")
            depths["currents"] = _fmt_depths(currents.get("depths") or [])

        info[name] = {"variables": variables, "depths": depths,
                      "key": key, "folder": cfg.get("folder")}

    return info


# ---------------------------------------------------------------------------
# Cache — building region_info means one Mongo round-trip per region
# (27, currently), which is fine once per request but wasteful on every one.
# A short TTL means a /regions edit shows up on the website within minutes,
# with no code change or restart, while normal page loads stay cheap.
# ---------------------------------------------------------------------------
_cache = {"ts": 0.0, "data": None}
CACHE_TTL = 300  # seconds


def get_region_info(force=False):
    now = time.time()
    if force or _cache["data"] is None or now - _cache["ts"] > CACHE_TTL:
        _cache["data"] = build_region_info()
        _cache["ts"] = now
    return _cache["data"]
