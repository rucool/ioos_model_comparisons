#!/usr/bin/env python3
"""
Audit the model_plots MongoDB collection to verify record counts match
the expected plot-call vs individual-file breakdown.

Usage:
    python scripts/tools/audit_model_plots.py [--script wfs]

Requires MONGODB_URI environment variable.
"""
import argparse
import os
import sys
from collections import defaultdict

import pymongo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="wfs", help="SCRIPT_ID to audit (default: wfs)")
    args = parser.parse_args()

    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("ERROR: MONGODB_URI environment variable is not set")
        sys.exit(1)

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    coll = client["hurricanes"]["model_plots"]

    docs = list(coll.find({"script": args.script}, {"_id": 0}))
    total = len(docs)
    print(f"\nTotal records for script='{args.script}': {total}\n")

    # ── Records per plot_type ─────────────────────────────────────────────────
    by_type = defaultdict(int)
    for d in docs:
        by_type[d["plot_type"]] += 1

    print("Records by plot_type:")
    for pt, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {pt:<25} {count:>5}")

    # ── Records per (region, timestamp, plot_type, model1, model2) ────────────
    # This groups by "function call" — the unit the summary counts as one plot.
    call_counts = defaultdict(int)
    for d in docs:
        key = (d["region"], str(d["timestamp"]), d["plot_type"], d["model1"], d["model2"])
        call_counts[key] += 1

    num_calls = len(call_counts)
    print(f"\nUnique plot function calls (region × timestamp × type × models): {num_calls}")
    print(f"Average records per call: {total / num_calls:.2f}")

    # ── Records per call breakdown ────────────────────────────────────────────
    records_per_call = defaultdict(int)
    for count in call_counts.values():
        records_per_call[count] += 1

    print("\nDistribution of records-per-call:")
    for n, freq in sorted(records_per_call.items()):
        print(f"  {n} record(s) per call: {freq} call(s)")

    # ── Timestamps seen ───────────────────────────────────────────────────────
    timestamps = sorted({str(d["timestamp"]) for d in docs})
    print(f"\nTimestamps covered ({len(timestamps)}):")
    for ts in timestamps:
        print(f"  {ts}")

    # ── Regions seen ─────────────────────────────────────────────────────────
    regions = sorted({d["region"] for d in docs})
    print(f"\nRegions covered ({len(regions)}):")
    for r in regions:
        print(f"  {r}")


if __name__ == "__main__":
    main()
