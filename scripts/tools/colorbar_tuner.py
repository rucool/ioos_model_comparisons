#!/usr/bin/env python3
"""
colorbar_tuner.py
-----------------
Interactive GUI that walks through every (region, variable, depth) combination
defined in region_config, lets you adjust colorbar limits via sliders, and
saves everything to MongoDB in one batch when you click "Save All".

Navigation
----------
    < Back   – store current settings and go to previous combo
    Next >   – store current settings and go to next combo
    Finish   – label on "Next >" when on the last combo; auto-saves all
    Save All – batch-save all stored settings to MongoDB at any time
    Reset    – restore auto-computed limits for the current combo

Usage
-----
    python scripts/tools/colorbar_tuner.py
    python scripts/tools/colorbar_tuner.py --regions mab sab --vars temperature
    python scripts/tools/colorbar_tuner.py --regions guam --vars currents
        (tunes the current-speed colorbar, i.e. currents.limits_by_depth)
    python scripts/tools/colorbar_tuner.py --regions guam --vars currents \
        --rtofs-file /path/to/rtofs_glo_20260724T06_guam.nc
        (for regions not served by the THREDDS OPeNDAP source -- e.g. guam,
        which is binary-archive-only -- point at a regional NetCDF produced
        by scripts/harvest/grab_rtofs_archv_aws.py instead)

Requirements
------------
    - ioos_model_comparisons conda env
    - MONGODB_URI env var (only for Save All)
    - Interactive matplotlib backend (TkAgg / Qt5Agg / MacOSX)

MongoDB access from a laptop
-----------------------------
mongodb.marine.rutgers.edu isn't reachable directly -- tunnel through
boardwalk.marine.rutgers.edu and point MONGODB_URI at localhost so the GUI
still runs locally (it needs a real display). Use a local port other than
27017 if you run MongoDB locally too (it'll otherwise fail to bind):

    ssh -N -L 27018:mongodb.marine.rutgers.edu:27017 michaesm@boardwalk.marine.rutgers.edu &
    export MONGODB_URI="mongodb://localhost:27018/hurricanes"

("Save regions.py" doesn't touch Mongo at all, so it works without the tunnel.)
"""

import argparse
import os
import re
import sys

import numpy as np
import xarray as xr

import matplotlib
for _backend in ("TkAgg", "Qt5Agg", "MacOSX", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from ioos_model_comparisons.regions import region_config
from ioos_model_comparisons.models import rtofs, espc_ts, espc_uv, CMEMS
from ioos_model_comparisons.calc import lon180to360, lon360to180

# ---------------------------------------------------------------------------
P_LO, P_HI = 5, 95

REGIONS_ALL = [
    "gom",
    "sab",
    "bahamas",
    "mab", 
    "west_florida_shelf", 
    "caribbean", 
    "windward",  
    "tropical_western_atlantic",
    "guam",
    ]

STRIDE_CANDIDATES = {
    "temperature":        [0.25, 0.5, 1.0, 2.0],
    "salinity":           [0.05, 0.1, 0.25, 0.5],
    "sea_surface_height": [0.05, 0.1, 0.2],
    "currents":           [0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
}
CMAPS = {
    "temperature":        cmocean.cm.thermal,
    "salinity":           cmocean.cm.haline,
    "sea_surface_height": cmocean.cm.balance,
    "currents":           cmocean.cm.speed,
}
RTOFS_VAR = {
    "temperature":        "temperature",
    "salinity":           "salinity",
    "sea_surface_height": "ssh",
}
UNITS = {
    "temperature":        "C",
    "salinity":           "PSU",
    "sea_surface_height": "m",
    "currents":           "m/s",
}
VAR_LABELS = {
    "currents": "Current Speed",
}

import matplotlib as _mpl
_SLIDER_EXTRA = {}
if tuple(int(x) for x in _mpl.__version__.split(".")[:2]) >= (3, 7):
    _SLIDER_EXTRA["track_color"] = "#45475a"

# ---------------------------------------------------------------------------

def build_combo_list(regions=None, vars_=None):
    """Return [(region_key, var_name, depth), ...] for all configured combos."""
    regions = regions or REGIONS_ALL
    vars_   = vars_   or ["temperature", "salinity"]
    combos  = []
    for r in regions:
        try:
            cfg = region_config(r)
        except Exception as e:
            print(f"  skip region {r!r}: {e}")
            continue
        for v in vars_:
            if v == "currents":
                cdict = cfg.get("currents")
                if not cdict or not cdict.get("bool"):
                    continue
                by_depth = cdict.get("limits_by_depth") or {}
                base_limits = cdict.get("limits")
                for depth in cdict.get("depths", []):
                    limits = by_depth.get(depth, base_limits)
                    combos.append((r, v, float(depth), limits))
                continue
            entries = (cfg.get("sea_surface_height", []) if v == "sea_surface_height"
                       else cfg.get("variables", {}).get(v, []))
            for entry in entries:
                combos.append((r, v, float(entry["depth"]), entry.get("limits")))
    return combos


def _subset_rtofs(ds, extent, buf=1.0):
    lons_ind = np.interp(
        [extent[0] - buf, extent[1] + buf],
        ds.lon.values[0, :], ds.x.values,
    )
    lats_ind = np.interp(
        [extent[2] - buf, extent[3] + buf],
        ds.lat.values[:, 0], ds.y.values,
    )
    return ds.isel(
        x=slice(int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1]))),
        y=slice(int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1]))),
    )


def _load_rtofs_local(path):
    """Open a regional RTOFS NetCDF produced by grab_rtofs_archv_aws.py.

    These files come from process_rtofs_region() (binary .a/.b archives
    interpolated onto standard z-levels) for regions -- e.g. guam -- that
    the Rutgers THREDDS OPeNDAP aggregation doesn't serve at all. Variables
    use raw HYCOM field names ('temp', 'salin', 'u-vel.', 'v-vel.') on a
    'z' vertical coord instead of 'depth'; rename to match what load_data's
    rtofs-local branch (and the rest of the tuner) expects. There is no
    'time' dimension -- one file is one archive snapshot.
    """
    ds = xr.open_dataset(path)
    return ds.rename({
        "temp": "temperature",
        "salin": "salinity",
        "u-vel.": "u",
        "v-vel.": "v",
        "z": "depth",
    })


def _auto_limits(data, stride_candidates):
    flat = data.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    lo = float(np.percentile(flat, P_LO))
    hi = float(np.percentile(flat, P_HI))
    rng = hi - lo
    stride = stride_candidates[-1]
    for s in stride_candidates:
        if 8 <= rng / s <= 20:
            stride = s
            break
    cmin = np.floor(lo / stride) * stride
    cmax = np.ceil(hi / stride) * stride
    return float(cmin), float(cmax), float(stride)


def load_data(region_key, var_name, depth, model="rtofs", model_ds=None):
    print(f"Loading {model.upper()} for {region_key} / {var_name} @ {depth} m ...")
    cfg    = region_config(region_key)
    extent = cfg["extent"]

    if model == "rtofs":
        ds    = model_ds
        rds_t = ds.sel(time=ds.time.values[-1])
        sub   = _subset_rtofs(rds_t, extent)
        if var_name == "currents":
            da_u, da_v = sub["u"], sub["v"]
            if "depth" in da_u.dims:
                da_u = da_u.sel(depth=depth, method="nearest")
                da_v = da_v.sel(depth=depth, method="nearest")
            u, v = da_u.compute().values, da_v.compute().values
            data = np.sqrt(u ** 2 + v ** 2)
        else:
            da = sub[RTOFS_VAR.get(var_name, var_name)]
            if "depth" in da.dims:
                da = da.sel(depth=depth, method="nearest")
            data = da.compute().values
        lons = sub.lon.values
        lats = sub.lat.values

    elif model == "rtofs-local":
        buf = 1.0
        sub = model_ds.sel(
            lon=slice(extent[0] - buf, extent[1] + buf),
            lat=slice(extent[2] - buf, extent[3] + buf),
        )
        if var_name == "currents":
            da_u, da_v = sub["u"], sub["v"]
            if "depth" in da_u.dims:
                da_u = da_u.sel(depth=depth, method="nearest")
                da_v = da_v.sel(depth=depth, method="nearest")
            u, v = da_u.compute().values, da_v.compute().values
            data = np.sqrt(u ** 2 + v ** 2)
        else:
            da = sub[var_name]
            if "depth" in da.dims:
                da = da.sel(depth=depth, method="nearest")
            data = da.compute().values
        lons = sub.lon.values
        lats = sub.lat.values

    elif model == "espc":
        ds  = model_ds["uv"] if var_name == "currents" else model_ds["ts"]
        buf = 1.0
        lon360 = lon180to360([extent[0] - buf, extent[1] + buf])
        t   = ds.time.values[-1]
        sub = ds.sel(
            time=t,
            lon=slice(lon360[0], lon360[1]),
            lat=slice(extent[2] - buf, extent[3] + buf),
        )
        if var_name == "currents":
            da_u, da_v = sub["u"], sub["v"]
            if "depth" in da_u.dims:
                da_u = da_u.sel(depth=depth, method="nearest")
                da_v = da_v.sel(depth=depth, method="nearest")
            u, v = da_u.compute().values, da_v.compute().values
            data = np.sqrt(u ** 2 + v ** 2)
        else:
            da = sub[var_name]
            if "depth" in da.dims:
                da = da.sel(depth=depth, method="nearest")
            data = da.compute().values
        lons = lon360to180(sub.lon.values)
        lats = sub.lat.values

    elif model == "cmems":
        buf = 1.0
        lon_extent = [extent[0] - buf, extent[1] + buf]
        lat_extent = [extent[2] - buf, extent[3] + buf]
        if var_name == "currents":
            da_u = model_ds.get_subset("uo", lon_extent=lon_extent, lat_extent=lat_extent)
            da_v = model_ds.get_subset("vo", lon_extent=lon_extent, lat_extent=lat_extent)
            if "time" in da_u.dims:
                da_u = da_u.isel(time=-1)
                da_v = da_v.isel(time=-1)
            if "depth" in da_u.dims:
                da_u = da_u.sel(depth=depth, method="nearest")
                da_v = da_v.sel(depth=depth, method="nearest")
            u, v = da_u.compute().values, da_v.compute().values
            data = np.sqrt(u ** 2 + v ** 2)
            lons = da_u.longitude.values
            lats = da_u.latitude.values
        else:
            da  = model_ds.get_subset(
                var_name,
                lon_extent=lon_extent,
                lat_extent=lat_extent,
            )
            if "time" in da.dims:
                da = da.isel(time=-1)
            if "depth" in da.dims:
                da = da.sel(depth=depth, method="nearest")
            data = da.compute().values
            lons = da.longitude.values
            lats = da.latitude.values

    else:
        raise ValueError(f"Unknown model: {model!r}")

    print("  done.")
    return data, lons, lats, extent


def _save_to_mongo(region_key, var_name, depth, limits, model="rtofs"):
    """Save a tuned [min, max, stride] into hurricanes.region_configs.

    region_configs is the single source of truth for MongoDB-driven region
    config (see db.apply_colorbar_overrides) — this reads the existing depth
    list for *var_name*, updates just the matching depth entry, and $sets
    that one field back, leaving the rest of the document (extent, folder,
    currents, ...) untouched. *model* is only used for the print/log message;
    region_configs has no per-model split.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("  MONGODB_URI not set -- NOT saved.")
        return
    if var_name not in ("temperature", "salinity", "sea_surface_height", "currents"):
        print(f"  Unknown var_name {var_name!r} -- NOT saved.")
        return
    try:
        import pymongo
        client     = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        collection = client["hurricanes"]["region_configs"]
        query      = {"region": region_key}
        doc        = collection.find_one(query) or {"region": region_key}

        if var_name == "currents":
            # currents is otherwise fully owned by regions.py (seeded whole
            # via seed_region_configs.py) -- apply_colorbar_overrides()
            # replaces the *entire* top-level 'currents' key wholesale, so
            # if this region's Mongo doc has never been seeded we backfill
            # bool/depths/limits/coarsen/kwargs from regions.py first, to
            # avoid clobbering them down to just limits_by_depth.
            cur = doc.get("currents")
            if not isinstance(cur, dict):
                base_cfg = region_config(region_key).get("currents")
                cur = dict(base_cfg) if isinstance(base_cfg, dict) else {}
            by_depth = {str(k): v for k, v in (cur.get("limits_by_depth") or {}).items()}
            by_depth[str(int(depth) if float(depth).is_integer() else depth)] = limits
            cur["limits_by_depth"] = by_depth
            collection.update_one(query, {"$set": {"currents": cur}}, upsert=True)
            print(f"  Saved -> hurricanes.region_configs[{region_key}].currents.limits_by_depth[{depth}] (model={model})")
            return

        if var_name in ("temperature", "salinity"):
            lst = doc.setdefault("variables", {}).setdefault(var_name, [])
            set_field = f"variables.{var_name}"
        else:
            lst = doc.setdefault("sea_surface_height", [])
            set_field = "sea_surface_height"

        for entry in lst:
            if entry.get("depth", 0) == depth:
                entry["limits"] = limits
                break
        else:
            lst.append({"depth": depth, "limits": limits})

        collection.update_one(query, {"$set": {set_field: lst}}, upsert=True)
        print(f"  Saved -> hurricanes.region_configs[{region_key}].{set_field} (model={model})")
    except Exception as exc:
        print(f"  MongoDB save failed: {exc}")

def _update_regions_py(pending):
    import re
    regions_file = os.path.join(os.path.dirname(__file__), "..", "..", "ioos_model_comparisons", "regions.py")
    with open(regions_file, "r") as f:
        lines = f.readlines()

    var_name_map = {
        "temperature": "sea_water_temperature",
        "salinity": "salinity",
        "sea_surface_height": "sea_surface_height",
    }

    by_region = {}
    currents_by_region = {}
    for (region, var, depth), limits in pending.items():
        if var == "currents":
            currents_by_region.setdefault(region, []).append((depth, limits))
        else:
            by_region.setdefault(region, []).append((var_name_map.get(var, var), depth, limits))

    def _find_block(region):
        block_start = -1
        for i, line in enumerate(lines):
            # Find key = "region" or key='region'
            if re.match(rf"^\s*key\s*=\s*['\"]{region}['\"]\s*", line):
                block_start = i
                break

        if block_start == -1:
            return -1, -1

        block_end = len(lines)
        for i in range(block_start + 1, len(lines)):
            if re.match(r"^\s*key\s*=\s*['\"].*['\"]\s*", lines[i]):
                if not lines[i].strip().startswith("#"):
                    block_end = i
                    break
        return block_start, block_end

    def _fmt_depth(depth):
        return f"{int(depth)}" if float(depth).is_integer() else f"{depth}"

    for region, edits in currents_by_region.items():
        block_start, block_end = _find_block(region)
        if block_start == -1:
            print(f"  Could not find block for region {region!r} in regions.py")
            continue

        cur_start = -1
        for i in range(block_start, block_end):
            if re.match(r"^\s*currents\s*=\s*dict\s*\(", lines[i]):
                cur_start = i
                break
        if cur_start == -1:
            print(f"  No currents block found for region {region!r} -- skipping currents save")
            continue

        paren_depth = 0
        cur_end = cur_start
        for i in range(cur_start, block_end):
            paren_depth += lines[i].count("(") - lines[i].count(")")
            if paren_depth <= 0:
                cur_end = i
                break

        lbd_line = -1
        for i in range(cur_start, cur_end + 1):
            if re.match(r"^\s*limits_by_depth\s*=\s*\{", lines[i]):
                lbd_line = i
                break

        if lbd_line != -1:
            line = lines[lbd_line]
            for depth, limits in edits:
                dstr = _fmt_depth(depth)
                new_val = f"[{limits[0]}, {limits[1]}, {limits[2]}]"
                pattern = re.compile(rf"(?<![\d.]){re.escape(dstr)}\s*:\s*\[[^\]]*\]")
                if pattern.search(line):
                    line = pattern.sub(f"{dstr}: {new_val}", line, count=1)
                else:
                    line = re.sub(r"\}\s*,?\s*\n?$", "", line.rstrip("\n"))
                    line = f"{line}, {dstr}: {new_val}}},\n"
            lines[lbd_line] = line
        else:
            indent = "            "
            insert_at = cur_start + 1
            for i in range(cur_start, cur_end + 1):
                m = re.match(r"^(\s*)limits\s*=\s*\[", lines[i])
                if m:
                    indent = m.group(1)
                    insert_at = i + 1
                    break
            entries = ", ".join(
                f"{_fmt_depth(depth)}: [{limits[0]}, {limits[1]}, {limits[2]}]"
                for depth, limits in edits
            )
            lines.insert(insert_at, f"{indent}limits_by_depth = {{{entries}}},\n")

    for region, edits in by_region.items():
        block_start, block_end = _find_block(region)
        if block_start == -1:
            print(f"  Could not find block for region {region!r} in regions.py")
            continue

        for var, depth, limits in edits:
            var_start = -1
            for i in range(block_start, block_end):
                if lines[i].strip().startswith(f"{var} = ["):
                    var_start = i
                    break
            
            if var_start != -1:
                # Search for the depth line within the block
                for i in range(var_start + 1, block_end):
                    line = lines[i]
                    if line.strip() == "]" or any(line.strip().startswith(f"{v} = ") for v in ["sea_water_temperature", "salinity", "sea_surface_height", "ocean_heat_content", "salinity_max", "currents"]):
                        break
                    
                    # Match dict(depth=DEPTH, ...)
                    # Also handle float depth like 0.0 vs 0
                    # For safety, let's parse the string to float
                    match = re.search(r'dict\s*\(\s*depth\s*=\s*([0-9.]+)', line)
                    if match and float(match.group(1)) == float(depth):
                        # Replace limits=[...] with the new limits
                        new_line = re.sub(r'limits\s*=\s*\[[^\]]*\]', f'limits=[{limits[0]}, {limits[1]}, {limits[2]}]', line)
                        if new_line == line:
                            # It didn't have limits before, append it
                            new_line = re.sub(r'dict\s*\(\s*depth\s*=\s*' + match.group(1) + r'\s*\)', f'dict(depth={match.group(1)}, limits=[{limits[0]}, {limits[1]}, {limits[2]}])', line)
                        lines[i] = new_line
                        break

    with open(regions_file, "w") as f:
        f.writelines(lines)
    print("  Saved limits directly to regions.py")

# ---------------------------------------------------------------------------

class ColorbarWalker:
    """Walk through all (region, var, depth) combos; batch-save at the end."""

    def __init__(self, combos, model="rtofs", model_ds=None):
        self.combos    = combos
        self.model     = model
        self.model_ds  = model_ds
        self.idx       = 0
        self.pending   = {}    # (region, var, depth) -> [vmin, vmax, stride]
        self._cache    = {}    # same key -> (data, lons, lats, extent)
        self._cf       = None
        self._cb       = None
        self._snapping = False
        self.stride    = 1.0

        self._build_figure()
        self._load_current()
        plt.show()

    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.patch.set_facecolor("#1e1e2e")

        self.ax_map = self.fig.add_axes(
            (0.04, 0.32, 0.78, 0.63),
            projection=ccrs.PlateCarree(),
        )
        self.ax_map.set_facecolor("#0d1b2a")

        self.ax_cb   = self.fig.add_axes((0.84, 0.32, 0.025, 0.63))
        self._cb_pos = self.ax_cb.get_position().bounds

        self.ax_vmin = self.fig.add_axes((0.08, 0.21, 0.58, 0.025))
        self.ax_vmax = self.fig.add_axes((0.08, 0.15, 0.58, 0.025))

        # Placeholder sliders — replaced in _load_current via _refresh_widgets
        self.sl_vmin = Slider(self.ax_vmin, "Min", 0, 1, valinit=0.0,
                              color="#89b4fa", **_SLIDER_EXTRA)
        self.sl_vmax = Slider(self.ax_vmax, "Max", 0, 1, valinit=1.0,
                              color="#f38ba8", **_SLIDER_EXTRA)
        for sl in (self.sl_vmin, self.sl_vmax):
            sl.label.set_color("white")
            sl.valtext.set_color("white")

        # Placeholder radio — replaced in _refresh_widgets
        self.ax_radio = self.fig.add_axes((0.70, 0.12, 0.26, 0.14))
        self.ax_radio.set_facecolor("#1e1e2e")
        self.radio = RadioButtons(self.ax_radio, ["0.5", "1.0"])
        self.ax_radio.set_title("Stride", color="white", fontsize=9, pad=2)

        # Free-entry box for a stride finer than any preset above -- some
        # deep/weak currents combos need well below 0.05 to show contrast.
        self.ax_stride_custom = self.fig.add_axes((0.855, 0.075, 0.105, 0.035))
        self.ax_stride_custom.set_facecolor("#1e1e2e")
        self.tb_stride = TextBox(
            self.ax_stride_custom, "Custom ", initial="",
            color="#313244", hovercolor="#45475a",
        )
        self.tb_stride.label.set_color("white")
        self.tb_stride.label.set_fontsize(8)
        self.tb_stride.on_submit(self._on_custom_stride)

        self.ax_btn_reset    = self.fig.add_axes((0.08, 0.07, 0.09, 0.045))
        self.ax_btn_back     = self.fig.add_axes((0.19, 0.07, 0.09, 0.045))
        self.ax_btn_next     = self.fig.add_axes((0.30, 0.07, 0.09, 0.045))
        self.ax_btn_save_all = self.fig.add_axes((0.41, 0.07, 0.12, 0.045))
        self.ax_btn_save_reg = self.fig.add_axes((0.55, 0.07, 0.15, 0.045))

        self.btn_reset    = Button(self.ax_btn_reset,    "Reset",       color="#313244", hovercolor="#45475a")
        self.btn_back     = Button(self.ax_btn_back,     "< Back",      color="#313244", hovercolor="#45475a")
        self.btn_next     = Button(self.ax_btn_next,     "Next >",      color="#313244", hovercolor="#45475a")
        self.btn_save_all = Button(self.ax_btn_save_all, "Save Mongo",  color="#313244", hovercolor="#45475a")
        self.btn_save_reg = Button(self.ax_btn_save_reg, "Save regions.py", color="#313244", hovercolor="#45475a")

        for b in (self.btn_reset, self.btn_back, self.btn_next, self.btn_save_all, self.btn_save_reg):
            b.label.set_color("white")
            b.label.set_fontsize(9)

        self.info_text = self.fig.text(
            0.08, 0.025, "", color="#cdd6f4", fontsize=9, fontfamily="monospace",
        )
        self.progress_text = self.fig.text(
            0.73, 0.04, "", color="#a6e3a1", fontsize=10, fontweight="bold",
        )

        for ax in (self.ax_vmin, self.ax_vmax, self.ax_radio, self.ax_stride_custom,
                   self.ax_btn_reset, self.ax_btn_back,
                   self.ax_btn_next, self.ax_btn_save_all, self.ax_btn_save_reg):
            ax.set_facecolor("#1e1e2e")

        self.btn_reset.on_clicked(self._on_reset)
        self.btn_back.on_clicked(self._on_back)
        self.btn_next.on_clicked(self._on_next)
        self.btn_save_all.on_clicked(self._on_save_all)
        self.btn_save_reg.on_clicked(self._on_save_reg)
        self.fig.canvas.mpl_connect(
            "draw_event",
            lambda _e: self.ax_cb.set_position(self._cb_pos),
        )

    # ------------------------------------------------------------------
    def _load_current(self):
        combo = self.combos[self.idx]
        if len(combo) == 3:
            region_key, var_name, depth = combo
            existing_limits = None
        else:
            region_key, var_name, depth, existing_limits = combo
        n = len(self.combos)

        # Update progress label and navigation button labels
        self.progress_text.set_text(f"{self.idx + 1} / {n}")
        self.btn_back.label.set_alpha(0.35 if self.idx == 0 else 1.0)
        is_last = self.idx == n - 1
        self.btn_next.label.set_text("Finish" if is_last else "Next >")

        self.fig.suptitle(
            f"[{self.model.upper()}]  {region_key}  /  {var_name}  /  {depth} m",
            color="#cdd6f4", fontsize=12, fontweight="bold", y=0.97,
        )

        # Fetch or retrieve cached data
        cache_key = (region_key, var_name, depth)
        if cache_key not in self._cache:
            self.info_text.set_text("Loading data ...")
            self.fig.canvas.draw_idle()
            plt.pause(0.05)
            try:
                result = load_data(region_key, var_name, depth,
                                   model=self.model, model_ds=self.model_ds)
            except Exception as exc:
                self.info_text.set_text(f"Load error: {exc}")
                self.fig.canvas.draw_idle()
                return
            self._cache[cache_key] = result

        data, lons, lats, extent = self._cache[cache_key]
        self.data   = data
        self.lons   = lons
        self.lats   = lats
        self.extent = extent

        strides      = STRIDE_CANDIDATES.get(var_name, [0.5, 1.0])
        self.strides = strides
        self.cmap    = CMAPS.get(var_name, cmocean.cm.thermal)
        self.units   = UNITS.get(var_name, "")

        auto = _auto_limits(data, strides)
        if auto is None:
            self.info_text.set_text("No valid data for this combo.")
            self.fig.canvas.draw_idle()
            return
        auto_vmin, auto_vmax, auto_stride = auto
        self.auto_vmin   = auto_vmin
        self.auto_vmax   = auto_vmax
        self.auto_stride = auto_stride

        # Restore previously stored limits if revisiting
        if cache_key in self.pending:
            vmin, vmax, stride = self.pending[cache_key]
        elif existing_limits is not None and len(existing_limits) == 3:
            vmin, vmax, stride = existing_limits
        else:
            vmin, vmax, stride = auto_vmin, auto_vmax, auto_stride
        self.stride = stride

        flat   = data[np.isfinite(data)]
        p01    = float(np.percentile(flat, 1))
        p99    = float(np.percentile(flat, 99))
        
        span_data = p99 - p01
        span_config = (vmax - vmin) if (vmax > vmin) else 0.0
        span = max(span_data, span_config)
        if span <= 0: span = 1.0

        sl_min = round(min(vmin, p01 - 1.0 * span), 4)
        sl_max = round(max(vmax, p99 + 1.0 * span), 4)

        self._refresh_widgets(sl_min, sl_max, vmin, vmax, stride, strides)

        # Redraw map with cleared state
        self._cf = None
        self._cb = None
        self.ax_map.cla()
        self.ax_cb.cla()
        self.ax_map.set_facecolor("#0d1b2a")
        self._draw_map(vmin, vmax, stride)

    # ------------------------------------------------------------------
    def _refresh_widgets(self, sl_min, sl_max, vmin, vmax, stride, strides):
        """Recreate sliders and radio buttons for the new combo's range."""
        self.ax_vmin.cla()
        self.ax_vmax.cla()
        self.ax_vmin.set_facecolor("#1e1e2e")
        self.ax_vmax.set_facecolor("#1e1e2e")

        self.sl_vmin = Slider(
            self.ax_vmin, "Min", sl_min, sl_max,
            valinit=vmin, color="#89b4fa", **_SLIDER_EXTRA,
        )
        self.sl_vmax = Slider(
            self.ax_vmax, "Max", sl_min, sl_max,
            valinit=vmax, color="#f38ba8", **_SLIDER_EXTRA,
        )
        for sl in (self.sl_vmin, self.sl_vmax):
            sl.label.set_color("white")
            sl.valtext.set_color("white")
        self.sl_vmin.on_changed(self._on_slider)
        self.sl_vmax.on_changed(self._on_slider)

        self.ax_radio.cla()
        self.ax_radio.set_facecolor("#1e1e2e")
        stride_labels = [str(s) for s in strides]
        active_idx    = strides.index(stride) if stride in strides else 0
        self.radio    = RadioButtons(self.ax_radio, stride_labels, active=active_idx)
        self.ax_radio.set_title("Stride", color="white", fontsize=9, pad=2)
        self.tb_stride.set_val("")
        for lbl in self.radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_stride)

    # ------------------------------------------------------------------
    def _draw_map(self, vmin, vmax, stride):
        ax = self.ax_map
        ax.set_extent(self.extent, crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.NaturalEarthFeature("physical", "land", "110m"),
            facecolor="#45475a", edgecolor="#cdd6f4", linewidth=0.5, zorder=10,
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#6c7086", zorder=11)
        ax.coastlines(resolution="110m", linewidth=0.6, color="#cdd6f4", zorder=12)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="#585b70",
                     alpha=0.7, linestyle="--")
        self._update_contourf(vmin, vmax, stride)

    # ------------------------------------------------------------------
    def _update_contourf(self, vmin, vmax, stride):
        if self._cf is not None:
            try:
                self._cf.remove()
            except AttributeError:
                for coll in self._cf.collections:
                    coll.remove()

        levels = np.arange(vmin, vmax + stride * 0.5, stride)
        if len(levels) < 2:
            return

        self._cf = self.ax_map.contourf(
            self.lons, self.lats, self.data,
            levels=levels,
            cmap=self.cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        var_name = self.combos[self.idx][1]
        if self._cb is None:
            self._cb = self.fig.colorbar(self._cf, cax=self.ax_cb)
            label = VAR_LABELS.get(var_name, var_name.replace(chr(95), ' ').title())
            self._cb.set_label(
                f"{label} ({self.units})",
                color="#cdd6f4", fontsize=9,
            )
            self._cb.outline.set_edgecolor("#585b70")
        else:
            self._cb.update_normal(self._cf)

        max_ticks = 12
        step      = max(1, len(levels) // max_ticks)
        tick_vals = levels[::step]
        self._cb.set_ticks(tick_vals)
        self._cb.set_ticklabels([f"{t:.4g}" for t in tick_vals])
        self._cb.ax.yaxis.set_tick_params(color="#cdd6f4", labelcolor="#cdd6f4")

        n_bands = len(levels) - 1
        self.info_text.set_text(
            f"limits = [{vmin:.4g},  {vmax:.4g},  stride {stride:.4g}]"
            f"   ->   {n_bands} contour bands"
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_slider(self, _val):
        if self._snapping:
            return
        stride = self.stride

        def _snap(v):
            return round(round(v / stride) * stride, 10)

        vmin_s = _snap(self.sl_vmin.val)
        vmax_s = _snap(self.sl_vmax.val)

        self._snapping = True
        try:
            if abs(vmin_s - self.sl_vmin.val) > 1e-12:
                self.sl_vmin.set_val(vmin_s)
            if abs(vmax_s - self.sl_vmax.val) > 1e-12:
                self.sl_vmax.set_val(vmax_s)
        finally:
            self._snapping = False

        if vmin_s >= vmax_s:
            return
        self._update_contourf(vmin_s, vmax_s, stride)

    def _on_stride(self, label):
        self.stride = float(label)
        self._on_slider(None)

    def _on_custom_stride(self, text):
        try:
            val = float(text)
        except ValueError:
            return
        if val <= 0:
            return
        self.stride = val
        self._on_slider(None)

    def _on_reset(self, _event):
        self.stride = self.auto_stride
        self.sl_vmin.set_val(self.auto_vmin)
        self.sl_vmax.set_val(self.auto_vmax)
        for i, s in enumerate(self.strides):
            if s == self.auto_stride:
                self.radio.set_active(i)
                break

    # ------------------------------------------------------------------
    def _store_current(self):
        key = self.combos[self.idx][:3]
        self.pending[key] = [
            round(self.sl_vmin.val, 6),
            round(self.sl_vmax.val, 6),
            self.stride,
        ]

    def _on_back(self, _event):
        if self.idx == 0:
            return
        self._store_current()
        self.idx -= 1
        self._load_current()

    def _on_next(self, _event):
        self._store_current()
        if self.idx < len(self.combos) - 1:
            self.idx += 1
            self._load_current()
        else:
            self._on_save_all(_event)

    def _on_save_all(self, _event):
        self._store_current()
        n = len(self.pending)
        print(f"\n[Save Mongo] Saving {n} entries ...")
        for (region, var, depth), limits in self.pending.items():
            print(f"  {region} / {var} @ {depth} m  ->  {limits}")
            _save_to_mongo(region, var, depth, limits, model=self.model)
        print("[Save Mongo] Done.")
        self.info_text.set_text(f"Saved {n} entries to MongoDB.")
        self.fig.canvas.draw_idle()

    def _on_save_reg(self, _event):
        self._store_current()
        n = len(self.pending)
        print(f"\n[Save regions.py] Saving {n} entries ...")
        for (region, var, depth), limits in self.pending.items():
            print(f"  {region} / {var} @ {depth} m  ->  {limits}")
        _update_regions_py(self.pending)
        print("[Save regions.py] Done.")
        self.info_text.set_text(f"Saved {n} entries directly to regions.py.")
        self.fig.canvas.draw_idle()

# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Walk through all colorbar combos and batch-save to MongoDB."
    )
    p.add_argument("--regions", "-r", nargs="+",
                   help="Regions to walk (default: all)")
    p.add_argument("--vars", "-v", nargs="+",
                   choices=["temperature", "salinity", "currents"],
                   help="Variables to walk (default: temperature and salinity). "
                        "'currents' tunes the speed (m/s) colorbar.")
    p.add_argument("--model", "-m", default="rtofs",
                   choices=["rtofs", "espc", "cmems"],
                   help="Model to load data from (default: rtofs)")
    p.add_argument("--rtofs-file", type=str, default=None,
                   help="Path to a regional RTOFS NetCDF produced by "
                        "scripts/harvest/grab_rtofs_archv_aws.py, for regions "
                        "(e.g. guam) not served by the THREDDS OPeNDAP source. "
                        "Overrides --model. If --regions is omitted, the region "
                        "is inferred from the filename (rtofs_glo_*_<region>.nc).")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.rtofs_file:
        regions = args.regions
        if not regions:
            m = re.search(r"_([A-Za-z0-9]+)\.nc$", os.path.basename(args.rtofs_file))
            if m:
                regions = [m.group(1)]
                print(f"Inferred region {regions[0]!r} from filename.")
        combos = build_combo_list(regions=regions, vars_=args.vars)
        if not combos:
            print("No region/variable/depth combinations found.")
            return
        print(f"Found {len(combos)} combinations to walk through.")
        print(f"Loading local RTOFS file: {args.rtofs_file}")
        model_ds = _load_rtofs_local(args.rtofs_file)
        ColorbarWalker(combos, model="rtofs-local", model_ds=model_ds)
        return

    combos = build_combo_list(regions=args.regions, vars_=args.vars)
    if not combos:
        print("No region/variable/depth combinations found.")
        return
    print(f"Found {len(combos)} combinations to walk through.")

    model = args.model
    vars_requested = args.vars or ["temperature", "salinity"]
    print(f"Loading {model.upper()} dataset (once) ...")
    if model == "rtofs":
        model_ds = rtofs()
    elif model == "espc":
        # ts3z (temperature/salinity) and uv3z (currents) are separate HYCOM
        # endpoints -- only load what this walk actually needs.
        model_ds = {}
        if any(v != "currents" for v in vars_requested):
            model_ds["ts"] = espc_ts(rename=True)
        if "currents" in vars_requested:
            model_ds["uv"] = espc_uv(rename=True)
    elif model == "cmems":
        model_ds = CMEMS()

    ColorbarWalker(combos, model=model, model_ds=model_ds)


if __name__ == "__main__":
    main()
