#!/usr/bin/env python
"""
Diagnostic plots showing that FishSOOP profiles have a fixed lat/lon per tow
but a unique timestamp per depth observation.

Runs for each WIGOS ID in wigos_ids_filter, using the most recent tow available.
"""
import io
import re
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Config — mirror wigos_ids_filter from the main comparisons script
# ---------------------------------------------------------------------------

wigos_ids_filter = [
    "0-22000-0-QNE94JY",
    "0-22000-0-DZK2PNH",
    "0-22000-0-PECEQRL",
    "0-22000-0-FYNTR3N",
]

days = 14   # look back window to find a recent tow for each ID

_THREDDS  = "https://thredds.aodn.org.au/thredds"
_CATALOG  = _THREDDS + "/catalog/IMOS/SOOP/SOOP-FishSOOP/REALTIME/{year}/{month:02d}/catalog.xml"
_FILE_URL = _THREDDS + "/fileServer/IMOS/SOOP/SOOP-FishSOOP/REALTIME/{year}/{month:02d}/{fname}"
_NS       = "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def catalog_fnames(year, month):
    url = _CATALOG.format(year=year, month=month)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Catalog failed {year}/{month:02d}: {e}")
        return []
    root = ET.fromstring(r.content)
    return [
        el.attrib["name"]
        for el in root.iter(f"{{{_NS}}}dataset")
        if el.attrib.get("name", "").endswith(".nc")
    ]


def fname_dt(fname):
    m = re.search(r"_(\d{8}T\d{6})Z_", fname)
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")) if m else None


def wigos_id_from_ds(ds):
    abstract = ds.attrs.get("abstract", "")
    m = re.search(r"WIGOS ID ([\w-]+)", abstract)
    return m.group(1) if m else "unknown"


def fetch_ds(year, month, fname):
    url = _FILE_URL.format(year=year, month=month, fname=fname)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return xr.open_dataset(io.BytesIO(r.content)), fname
    except Exception as e:
        print(f"Download failed {fname}: {e}")
        return None, fname


# ---------------------------------------------------------------------------
# Find the most recent file for each WIGOS ID
# ---------------------------------------------------------------------------

date_end   = pd.Timestamp.utcnow().tz_localize(None)
date_start = (date_end - pd.Timedelta(days=days)).floor("1d")

months = pd.period_range(pd.Period(date_start, "M"), pd.Period(date_end, "M"), freq="M")
in_range = [
    (p.year, p.month, f)
    for p in months
    for f in catalog_fnames(p.year, p.month)
    if (dt := fname_dt(f)) and date_start <= dt <= date_end
]

print(f"Scanning {len(in_range)} files from {date_start.date()} to {date_end.date()}…")

# Download all files in parallel, keep only wanted WIGOS IDs
wanted = set(wigos_ids_filter)
best = {}   # wigos_id -> (dt, ds, fname)

with ThreadPoolExecutor(max_workers=16) as ex:
    futures = {ex.submit(fetch_ds, y, mo, f): (y, mo, f) for y, mo, f in in_range}
    for fut in as_completed(futures):
        ds, fname = fut.result()
        if ds is None:
            continue
        wid = wigos_id_from_ds(ds)
        if wid not in wanted:
            continue
        dt = fname_dt(fname)
        if wid not in best or dt > best[wid][0]:
            best[wid] = (dt, ds, fname)

print(f"Found recent tows for: {list(best.keys())}")

missing = wanted - set(best.keys())
if missing:
    print(f"No data found for: {missing}")

# ---------------------------------------------------------------------------
# Diagnostic plot function
# ---------------------------------------------------------------------------

def make_diagnostic(wigos_id, ds, fname):
    df = ds[["TIME", "LATITUDE", "LONGITUDE", "DEPTH", "TEMPERATURE"]].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["TIME"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)

    n   = len(df)
    idx = np.arange(n)

    print(f"\n{wigos_id}")
    print(f"  Rows        : {n}")
    print(f"  Lat unique  : {df['LATITUDE'].nunique()}")
    print(f"  Lon unique  : {df['LONGITUDE'].nunique()}")
    print(f"  Time unique : {df['time'].nunique()}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"FishSOOP data structure diagnostic\nWIGOS {wigos_id}  |  {fname}",
        fontsize=11, fontweight="bold",
    )

    # Panel A — observation index per datetime
    ax = axes[0, 0]
    ax.plot(df["time"], idx, color="steelblue", linewidth=1.5)
    ax.set_title("A — Observation index per datetime", fontweight="bold")
    ax.set_xlabel("Datetime (UTC)")
    ax.set_ylabel("Observation index")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.annotate(
        f"{df['time'].nunique()} unique timestamps\n(out of {n} observations)",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    # Panel B — LATITUDE & LONGITUDE per observation index
    ax = axes[0, 1]
    lat_vals = df["LATITUDE"].values
    lon_vals = df["LONGITUDE"].values
    ax.plot(idx, lat_vals, color="tomato",   linewidth=2, label="LATITUDE")
    ax.plot(idx, lon_vals, color="seagreen", linewidth=2, label="LONGITUDE", linestyle="--")
    ax.set_title("B — LATITUDE & LONGITUDE per observation index", fontweight="bold")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Degrees")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.annotate(
        f"Lat: {lat_vals.min():.5f} → {lat_vals.max():.5f}\n"
        f"Lon: {lon_vals.min():.5f} → {lon_vals.max():.5f}",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    # Panel C — DEPTH vs TIME coloured by temperature
    ax = axes[1, 0]
    sc = ax.scatter(df["time"], df["DEPTH"], c=df["TEMPERATURE"],
                    cmap="RdYlBu_r", s=10, linewidths=0)
    ax.invert_yaxis()
    ax.set_title("C — DEPTH vs TIME (colour = temperature)", fontweight="bold")
    ax.set_xlabel("Datetime (UTC)")
    ax.set_ylabel("Depth (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.colorbar(sc, ax=ax, label="Temperature (°C)", pad=0.02)

    # Panel D — unique value counts per variable
    ax = axes[1, 1]
    labels = ["TIME", "LATITUDE", "LONGITUDE", "DEPTH"]
    unique = [
        df["time"].nunique(),
        df["LATITUDE"].nunique(),
        df["LONGITUDE"].nunique(),
        df["DEPTH"].nunique(),
    ]
    colors = ["steelblue" if u > 1 else "tomato" for u in unique]
    bars = ax.bar(labels, unique, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(n, color="gray", linestyle="--", linewidth=1, label=f"Total rows ({n})")
    ax.set_title("D — Unique values per variable", fontweight="bold")
    ax.set_ylabel("# unique values")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    for bar, u in zip(bars, unique):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n * 0.01,
            str(u),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.tight_layout()
    out = f"fishsoop_diagnostic_{wigos_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

for wigos_id in wigos_ids_filter:
    if wigos_id in best:
        _, ds, fname = best[wigos_id]
        make_diagnostic(wigos_id, ds, fname)
    else:
        print(f"\n{wigos_id}: no data found — skipping")
