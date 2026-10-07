#!/usr/bin/env python
"""
Download the latest ECCOFS output to the local data/eccofs/ cache.

Pulls both products this project reads from ioos_model_comparisons.models:
  - "his" (full-depth, 50 s-levels) -- used by ECCOFSFullDepth for the OHC,
    Argo/glider profile, and RTOFS-vs-ECCOFS currents comparisons. Every
    file in the latest published cycle (3-day analysis + forecast) is
    downloaded, since a comparison script may ask for any date within it.
  - "qck" (quicksave, 3 fixed depths) -- used by ECCOFS for the higher-
    cadence temperature/salinity/currents comparisons.

This only downloads -- it doesn't build the full-depth z-level
interpolation that ECCOFSFullDepth.sel() does on top of a downloaded "his"
file (each comparison script still does that itself, in-process, on first
use). Splitting the download out into its own script means:
  - it can run on its own schedule (e.g. cron), well ahead of whatever
    comparison scripts need the data, so those scripts hit a warm cache
    instead of each independently kicking off a ~5 min-per-file download
    and racing each other to write the same file on a cold cache
  - a slow/flaky download doesn't block or delay an actual comparison run

Both ensure_eccofs_cached() and ensure_eccofs_his_cached() are safe to
re-run: each checks the local file's size against S3 before downloading
anything, so a file already fetched by this script (or by a comparison
script that ran first) is a no-op here.

Usage:
    python3 scripts/harvest/grab_eccofs.py
    python3 scripts/harvest/grab_eccofs.py --qck-only
    python3 scripts/harvest/grab_eccofs.py --his-only
"""
import argparse
import logging

from ioos_model_comparisons.models import (
    _eccofs_his_index,
    ensure_eccofs_cached,
    ensure_eccofs_his_cached,
    prune_eccofs_his_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def grab_qck():
    """Download the latest ECCOFS qck (quicksave) file."""
    path = ensure_eccofs_cached()
    logger.info(f"qck ready: {path}")


def grab_his():
    """Download every file in the latest ECCOFS his cycle, then prune any
    cached files left over from a now-superseded older cycle."""
    index = _eccofs_his_index()
    logger.info(
        f"his: latest cycle spans {index[0][0].date()} to {index[-1][0].date()} "
        f"({len(index)} files)"
    )
    for t, key, size in index:
        path = ensure_eccofs_his_cached(key, size)
        logger.info(f"his ready: {t.date()} -> {path}")

    prune_eccofs_his_cache(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qck-only", action="store_true", help="Only download the qck product")
    parser.add_argument("--his-only", action="store_true", help="Only download the his product")
    args = parser.parse_args()

    if args.qck_only and args.his_only:
        parser.error("--qck-only and --his-only are mutually exclusive")

    if not args.his_only:
        grab_qck()
    if not args.qck_only:
        grab_his()


if __name__ == "__main__":
    main()
