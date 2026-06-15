#!/usr/bin/env python3
"""
update_colorbar_limits.py
─────────────────────────
Weekly cron script that samples live RTOFS data for every active region
defined in regions.py and updates the colorbar limits stored in MongoDB
(hurricanes.colorbar_configs) so that map plots rendered with contourf
show enough color gradations without flooding the colorbar.

Algorithm per region / depth layer
───────────────────────────────────
1. Subset the most-recent RTOFS snapshot to the region extent (+1° buffer).
2. Select the nearest available depth level.
3. Compute robust percentiles (P_LO / P_HI, defaulting to 2 / 98) of the
   field values to clip outliers.
4. Round the percentile bounds *outward* to the nearest multiple of the
   desired stride, keeping the total number of contour levels between
   N_LEVELS_MIN and N_LEVELS_MAX.
5. Write the resulting [min, max, stride] triple back to MongoDB via an
   upsert, preserving all other fields in the document.

Cron example (every Sunday at 02:00 local):
    0 2 * * 0  /path/to/python /path/to/update_colorbar_limits.py

Requires:
  - MONGODB_URI environment variable
  - ioos_model_comparisons package installed / on PYTHONPATH
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import cmocean

# NOTE: matplotlib / cartopy / cmocean / cool_maps are intentionally NOT
# imported here at module level.  They are lazy-imported inside
# plot_check_maps() so that the normal (no-plot) cron run pays zero cost
# from these heavyweight packages.

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ensure package is importable when run from any CWD ───────────────────────
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

from ioos_model_comparisons.regions import region_config  # noqa: E402
from ioos_model_comparisons.models import rtofs           # noqa: E402

# ── Tuneable parameters ───────────────────────────────────────────────────────

# Percentiles used when computing the colorbar min/max from data.
# These apply to ALL variables: temperature, salinity, and SSH.
#
# The 5 / 95 split is intentional and symmetric:
#
#   P_LO = 5  → ~5% of pixels fall BELOW the computed minimum and render
#               as the pure "cold" colour (e.g. dark blue on cmocean.thermal,
#               dark purple on cmocean.haline).  Cold shelf water, deep
#               upwelling, or fresh river plumes should stand out at the
#               low end of the colorbar just as clearly as the Gulf Stream
#               stands out at the high end.
#
#   P_HI = 95 → ~5% of pixels exceed the computed maximum and render as
#               the pure "hot" colour (e.g. yellow on cmocean.thermal).
#               Energetic features (Gulf Stream core, warm-core eddies,
#               high-salinity tongues) show as a saturated max colour
#               rather than stretching the entire colorbar to fit them.
#
# This mirrors the manually tuned limits in regions.py (e.g. temperature
# clipped at 29 °C for MAB, currents capped at 1.5 m/s) and produces
# colorbars that are visually informative across the full domain.
P_LO = 5
P_HI = 95

# Desired stride (colour interval) for each variable.
# The auto-computed stride is always snapped to these canonical values so
# that the colours are human-readable round numbers.
TEMP_STRIDES   = [0.25, 0.5, 1.0, 2.0]   # °C candidates – smallest first
SAL_STRIDES    = [0.05, 0.1, 0.25, 0.5]   # PSU candidates – smallest first
SSH_STRIDES    = [0.05, 0.1, 0.2]          # m candidates

# Colourbar should show at least this many and at most this many contour bands.
N_LEVELS_MIN = 8
N_LEVELS_MAX = 20

# Region list – all regions defined in regions.py that should be updated.
# Add or remove names here to control which regions are processed.
ALL_REGIONS = [
    "mastr",
    "yucatan",
    "leeward",
    "loop_current",
    "gom",
    "gom_east",
    "gom_west",
    "east_coast",
    "sab",
    "mab",
    "west_florida_shelf",
    "caribbean",
    "windward",
    "amazon",
    "hurricane",
    "tropical_western_atlantic",
    "passengers",
    "mexico_pacific",
    "hawaii",
    "wmo_v_south",
    "bahamas",
    "ru29",
    "philippines",
    "guam",
    "fiji",
]

# Variable name map: regions.py key → RTOFS dataset variable name
RTOFS_VAR_MAP = {
    "temperature": "temperature",
    "salinity":    "salinity",
}

# SSH is a top-level list in the region dict (key "sea_surface_height")
# and lives in a separate RTOFS variable
SSH_RTOFS_VAR = "ssh"   # adjust if your RTOFS dataset uses a different name


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _round_floor(value: float, stride: float) -> float:
    """Round *value* down to the nearest multiple of *stride*."""
    return np.floor(value / stride) * stride


def _round_ceil(value: float, stride: float) -> float:
    """Round *value* up to the nearest multiple of *stride*."""
    return np.ceil(value / stride) * stride


def _pick_stride(data_range: float, stride_candidates: list) -> float:
    """
    Choose the smallest stride from *stride_candidates* such that the number
    of levels (data_range / stride) stays within [N_LEVELS_MIN, N_LEVELS_MAX].

    Falls back to the largest candidate if nothing satisfies the constraint.
    """
    for stride in stride_candidates:
        n = data_range / stride
        if N_LEVELS_MIN <= n <= N_LEVELS_MAX:
            return stride
    return stride_candidates[-1]


def compute_limits(
    values: np.ndarray,
    stride_candidates: list,
) -> Optional[list]:
    """
    Compute [min, max, stride] limits from a 1-D or N-D array of values.

    Returns None if the array is empty or all-NaN.
    """
    flat = values.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None

    lo = float(np.percentile(flat, P_LO))
    hi = float(np.percentile(flat, P_HI))
    data_range = hi - lo
    if data_range <= 0:
        return None

    stride = _pick_stride(data_range, stride_candidates)
    cmin = _round_floor(lo, stride)
    cmax = _round_ceil(hi, stride)

    # Safety: ensure we still have at least N_LEVELS_MIN levels
    while (cmax - cmin) / stride < N_LEVELS_MIN:
        cmin -= stride
        cmax += stride

    return [round(cmin, 6), round(cmax, 6), stride]


def subset_rtofs(ds: xr.Dataset, extent: list) -> xr.Dataset:
    """
    Subset an RTOFS xarray.Dataset to *extent* = [lonmin, lonmax, latmin, latmax].

    Uses the RTOFS native x/y index approach (same as the main plotting scripts).
    A 1° buffer is added on each side so that contourf at the region edge has
    data to interpolate from.
    """
    buf = 1.0
    lon_min, lon_max = extent[0] - buf, extent[1] + buf
    lat_min, lat_max = extent[2] - buf, extent[3] + buf

    grid_lons = ds.lon.values[0, :]
    grid_lats = ds.lat.values[:, 0]
    grid_x    = ds.x.values
    grid_y    = ds.y.values

    lons_ind = np.interp([lon_min, lon_max], grid_lons, grid_x)
    lats_ind = np.interp([lat_min, lat_max], grid_lats, grid_y)

    x_sl = slice(int(np.floor(lons_ind[0])), int(np.ceil(lons_ind[1])))
    y_sl = slice(int(np.floor(lats_ind[0])), int(np.ceil(lats_ind[1])))

    return ds.isel(x=x_sl, y=y_sl)


def get_depth_slice(ds: xr.Dataset, var_name: str, target_depth: float) -> np.ndarray:
    """
    Return a 2-D numpy array of *var_name* at the depth level nearest to
    *target_depth* metres.  Returns an empty array on failure.

    .compute() is called explicitly to materialise the lazy OPeNDAP-backed
    DataArray before converting to numpy.
    """
    try:
        da = ds[var_name]
        # RTOFS uses 'depth' as the vertical coordinate name
        if "depth" in da.dims:
            da = da.sel(depth=target_depth, method="nearest")
        # Force data download from OPeNDAP server
        da = da.compute()
        return da.values
    except Exception as exc:
        logger.debug(f"  Could not extract {var_name} at depth {target_depth}: {exc}")
        return np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_mongo_collection():
    """Return the hurricanes.colorbar_configs collection, or None on failure."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        logger.error("MONGODB_URI environment variable is not set — cannot write limits")
        return None
    try:
        import pymongo
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client["hurricanes"]["colorbar_configs"]
    except Exception as exc:
        logger.error(f"MongoDB connection failed: {exc}")
        return None


def upsert_colorbar_doc(collection, region_key: str, doc: dict):
    """Upsert *doc* into the collection keyed by *region*."""
    result = collection.replace_one({"region": region_key}, doc, upsert=True)
    action = "inserted" if result.upserted_id else "updated"
    logger.info(f"  [{region_key}] MongoDB document {action}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-region processing
# ─────────────────────────────────────────────────────────────────────────────

def process_region(
    region_key: str,
    rds_time: xr.Dataset,
) -> Optional[dict]:
    """
    Compute updated colorbar limits for *region_key* using the RTOFS snapshot
    *rds_time* (already time-selected).

    Returns a MongoDB document dict or None if the region could not be processed.
    """
    try:
        cfg = region_config(region_key)
    except Exception as exc:
        logger.warning(f"[{region_key}] region_config failed: {exc}")
        return None

    extent = cfg.get("extent")
    if extent is None:
        logger.warning(f"[{region_key}] no extent defined — skipping")
        return None

    logger.info(f"[{region_key}] Processing extent {extent}")

    try:
        rds_sub = subset_rtofs(rds_time, extent)
    except Exception as exc:
        logger.warning(f"[{region_key}] Failed to subset RTOFS: {exc}")
        return None

    doc = {"region": region_key}

    # ── Temperature & Salinity ────────────────────────────────────────────────
    variables_cfg = cfg.get("variables", {})

    vars_doc = {}
    for var_key, rtofs_var in RTOFS_VAR_MAP.items():
        depth_list = variables_cfg.get(var_key, [])
        if not depth_list:
            continue

        entries = []
        for entry in depth_list:
            depth = entry.get("depth")
            if depth is None:
                continue

            stride_candidates = TEMP_STRIDES if var_key == "temperature" else SAL_STRIDES
            values = get_depth_slice(rds_sub, rtofs_var, depth)
            limits = compute_limits(values, stride_candidates)

            if limits is None:
                logger.warning(
                    f"  [{region_key}] {var_key} @ {depth} m — "
                    "insufficient data; keeping existing limits"
                )
                # Carry forward the original limits if they exist
                if "limits" in entry:
                    entries.append({"depth": depth, "limits": entry["limits"]})
            else:
                logger.info(
                    f"  [{region_key}] {var_key} @ {depth} m → {limits}"
                )
                entries.append({"depth": depth, "limits": limits})

        if entries:
            vars_doc[var_key] = entries

    if vars_doc:
        doc["variables"] = vars_doc

    # ── Sea Surface Height ────────────────────────────────────────────────────
    ssh_cfg = cfg.get("sea_surface_height", [])
    ssh_entries = []
    for entry in ssh_cfg:
        depth = entry.get("depth", 0)
        if SSH_RTOFS_VAR in rds_sub.data_vars:
            values = rds_sub[SSH_RTOFS_VAR].values
            limits = compute_limits(values, SSH_STRIDES)
            if limits is not None:
                logger.info(f"  [{region_key}] ssh @ surface → {limits}")
                ssh_entries.append({"depth": depth, "limits": limits})
                continue
        # Fall back to existing limits
        if "limits" in entry:
            ssh_entries.append({"depth": depth, "limits": entry["limits"]})

    if ssh_entries:
        doc["sea_surface_height"] = ssh_entries

    # ── ocean_heat_content / salinity_max — keep existing limits ─────────────
    # These are derived / integrated quantities that RTOFS surface snapshots
    # cannot estimate directly.  Carry the regions.py defaults forward so that
    # the MongoDB document remains complete.
    for top_key in ("ocean_heat_content", "salinity_max"):
        val = cfg.get(top_key)
        if isinstance(val, dict) and "limits" in val:
            doc[top_key] = {"limits": val["limits"]}

    # ── Currents — keep existing limits ──────────────────────────────────────
    cur = cfg.get("currents")
    if isinstance(cur, dict) and "limits" in cur:
        doc["currents"] = {"limits": cur["limits"]}

    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Check-plot helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colourmap selection matching the production plotting.py convention
_CMAPS = {
    "temperature":       cmocean.cm.thermal,
    "salinity":          cmocean.cm.haline,
    "sea_surface_height": cmocean.cm.balance,
}


def _cmap_for(var_key: str):
    return _CMAPS.get(var_key, cmocean.cm.thermal)


def plot_check_maps(
    doc: dict,
    rds_sub: xr.Dataset,
    extent: list,
    region_key: str,
    plot_dir: Path,
) -> None:
    """
    Render one PNG per (variable, depth) using the limits that were just
    computed, so you can visually verify the colorbar before committing.

    Output files are saved as::

        <plot_dir>/<region_key>/<var>_<depth>m.png

    Parameters
    ----------
    doc      : MongoDB document produced by process_region()
    rds_sub  : RTOFS subset for this region (time already selected)
    extent   : [lonmin, lonmax, latmin, latmax]
    region_key : short region name, used for the sub-directory
    plot_dir : root directory for output PNG files
    """
    # ── Lazy imports (only paid when --plot is used) ────────────────────────────
    import cmocean
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe for cron
    import matplotlib.pyplot as plt
    from cool_maps.plot import add_features, create
    out_dir = plot_dir / region_key
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_map  = ccrs.Mercator()
    proj_data = ccrs.PlateCarree()

    # ── Colormap lookup (matches production plotting.py convention) ───────────
    cmaps = {
        "temperature":        cmocean.cm.thermal,
        "salinity":           cmocean.cm.haline,
        "sea_surface_height": cmocean.cm.balance,
    }
    def _cmap_for(var_key):
        return cmaps.get(var_key, cmocean.cm.thermal)

    # ── Temperature & Salinity ────────────────────────────────────────────────
    for var_key, rtofs_var in RTOFS_VAR_MAP.items():
        entries = doc.get("variables", {}).get(var_key, [])
        for entry in entries:
            depth  = entry["depth"]
            limits = entry["limits"]          # [min, max, stride]

            values = get_depth_slice(rds_sub, rtofs_var, depth)
            if values.size == 0:
                logger.warning(f"  plot: no data for {var_key}@{depth}m — skipping")
                continue

            # Pull the matching lon/lat grid (2-D)
            try:
                lons = rds_sub.lon.values
                lats = rds_sub.lat.values
            except Exception:
                logger.warning("  plot: could not retrieve lon/lat — skipping")
                continue

            levels = np.arange(limits[0], limits[1] + limits[2] * 0.5, limits[2])

            fig, ax = create(extent, proj=proj_map, figsize=(10, 7))
            # coast='low' → 110m NaturalEarth land (already cached, much faster
            # than the default GSHHS full-resolution coastline)
            add_features(ax, coast="low")

            h = ax.contourf(
                lons, lats, values,
                levels=levels,
                cmap=_cmap_for(var_key),
                extend="both",
                transform=proj_data,
            )
            ax.contour(
                lons, lats, values,
                levels=levels,
                colors="k",
                linewidths=0.3,
                alpha=0.4,
                transform=proj_data,
            )

            cb = fig.colorbar(h, ax=ax, orientation="horizontal",
                              pad=0.04, fraction=0.046, shrink=0.85)
            cb.set_label(f"{var_key.replace('_', ' ').title()} (depth {depth} m)",
                         fontsize=11)
            cb.ax.tick_params(labelsize=9)
            # Mark the auto-computed min/max with triangles
            cb.ax.axvline(limits[0], color="red",  linewidth=1.5, linestyle="--")
            cb.ax.axvline(limits[1], color="blue", linewidth=1.5, linestyle="--")

            ax.set_title(
                f"{region_key}  ·  {var_key}  ·  {depth} m\n"
                f"limits = [{limits[0]:.4g}, {limits[1]:.4g}, stride {limits[2]:.4g}]  "
                f"({len(levels)-1} bands)",
                fontsize=10,
            )

            fname = out_dir / f"{var_key}_{depth}m.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            logger.info(f"  plot saved → {fname}")

    # ── Sea Surface Height ────────────────────────────────────────────────────
    ssh_entries = doc.get("sea_surface_height", [])
    if SSH_RTOFS_VAR in rds_sub.data_vars and ssh_entries:
        for entry in ssh_entries:
            limits = entry["limits"]
            depth  = entry.get("depth", 0)

            ssh_vals = rds_sub[SSH_RTOFS_VAR].compute().values
            lons     = rds_sub.lon.values
            lats     = rds_sub.lat.values
            levels   = np.arange(limits[0], limits[1] + limits[2] * 0.5, limits[2])

            fig, ax = create(extent, proj=proj_map, figsize=(10, 7))
            add_features(ax, coast="low")

            h = ax.contourf(
                lons, lats, ssh_vals,
                levels=levels,
                cmap=_cmap_for("sea_surface_height"),
                extend="both",
                transform=proj_data,
            )
            cb = fig.colorbar(h, ax=ax, orientation="horizontal",
                              pad=0.04, fraction=0.046, shrink=0.85)
            cb.set_label(f"Sea Surface Height (m)", fontsize=11)
            cb.ax.tick_params(labelsize=9)
            cb.ax.axvline(limits[0], color="red",  linewidth=1.5, linestyle="--")
            cb.ax.axvline(limits[1], color="blue", linewidth=1.5, linestyle="--")

            ax.set_title(
                f"{region_key}  ·  sea_surface_height\n"
                f"limits = [{limits[0]:.4g}, {limits[1]:.4g}, stride {limits[2]:.4g}]  "
                f"({len(levels)-1} bands)",
                fontsize=10,
            )

            fname = out_dir / f"sea_surface_height_{depth}m.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            logger.info(f"  plot saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update MongoDB colorbar limits from live RTOFS data."
    )
    p.add_argument(
        "-p", "--plot",
        action="store_true",
        default=False,
        help="After computing limits, save a check-plot PNG for every "
             "(region, variable, depth) so you can visually verify the colorbar.",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("colorbar_check_plots"),
        metavar="DIR",
        help="Directory in which check-plot PNGs are saved "
             "(default: ./colorbar_check_plots/).",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=None,
        metavar="REGION",
        help="Process only these region keys instead of ALL_REGIONS.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="Skip MongoDB writes; only generate check-plots. "
             "Implies --plot. Useful for dry-runs without a DB connection.",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # --plot-only implies --plot and skips DB writes
    do_plot  = args.plot or args.plot_only
    do_db    = not args.plot_only
    regions  = args.regions or ALL_REGIONS
    plot_dir = args.plot_dir

    logger.info("=" * 60)
    logger.info("update_colorbar_limits.py — started")
    if do_plot:
        logger.info(f"Check-plots → {plot_dir.resolve()}")
    if not do_db:
        logger.info("DB writes skipped (--plot-only mode)")
    logger.info("=" * 60)

    # ── Connect to MongoDB (skip if --plot-only) ───────────────────────────────
    collection = None
    if do_db:
        collection = _get_mongo_collection()
        if collection is None:
            sys.exit(1)
        # Ensure unique index on 'region' (idempotent)
        collection.create_index("region", unique=True)

    # ── Load RTOFS ────────────────────────────────────────────────────────────
    logger.info("Loading RTOFS …")
    try:
        # rename=True only renames coordinate dimensions (Longitude→lon etc.),
        # which happens unconditionally in the function, so no need to pass it.
        rds = rtofs()
    except Exception as exc:
        logger.error(f"Failed to open RTOFS dataset: {exc}")
        sys.exit(1)

    # Select the most recent available time step
    try:
        latest_time = rds.time.values[-1]
        rds_time = rds.sel(time=latest_time)
        logger.info(f"Using RTOFS snapshot: {latest_time}")
    except Exception as exc:
        logger.error(f"Failed to select RTOFS time: {exc}")
        sys.exit(1)

    # ── Process regions ───────────────────────────────────────────────────────
    n_ok = 0
    n_fail = 0
    for region_key in regions:
        logger.info(f"── {region_key} ──────────────────────────────────")
        doc = process_region(region_key, rds_time)
        if doc is None:
            n_fail += 1
            continue

        # ── DB upsert ─────────────────────────────────────────────────────────
        if do_db:
            try:
                upsert_colorbar_doc(collection, region_key, doc)
                n_ok += 1
            except Exception as exc:
                logger.error(f"[{region_key}] MongoDB upsert failed: {exc}")
                n_fail += 1
        else:
            n_ok += 1   # counted as processed even without DB write

        # ── Check plots ───────────────────────────────────────────────────────
        if do_plot:
            try:
                cfg     = region_config(region_key)
                extent  = cfg["extent"]
                rds_sub = subset_rtofs(rds_time, extent)
                plot_check_maps(doc, rds_sub, extent, region_key, plot_dir)
            except Exception as exc:
                logger.warning(f"[{region_key}] Check-plot failed: {exc}")

    logger.info("=" * 60)
    logger.info(
        f"Done — {n_ok} regions processed, {n_fail} failed/skipped"
    )
    if do_plot:
        logger.info(f"Check-plots saved to: {plot_dir.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
