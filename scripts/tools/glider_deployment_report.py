#!/usr/bin/env python3
"""
glider_deployment_report.py — weekly glider deployment report to Google Sheets.

Every Friday night this queries the IOOS Glider DAC (NGDAC ERDDAP) for every
glider dataset with data in the last 24 hours and writes one row per glider
into a new, dated tab of a shared Google Sheet:

    Glider Deployment | Group | Location | Model Evaluation | Glider Evaluation

The last two columns are left blank on purpose — glider operators and the
model comparisons team fill those in by hand over the following week. That's
also why each run creates a brand-new tab instead of overwriting last week's:
overwriting would wipe out whatever had already been filled in.

"Group" comes straight from the DAC's institution metadata (with a couple of
long/inconsistent names shortened — see INSTITUTION_LABELS). "Location" is a
best-effort label from a small set of named-region bounding boxes tuned to
match how this has been labeled by hand in the past (see REGION_BOXES) — it's
plain text in the sheet, so just correct it by hand for the occasional glider
that lands near a boundary.

Setup
-----
1. Create a Google Cloud service account, enable the Sheets and Drive APIs
   for its project, download its JSON key, and share the target spreadsheet
   with the service account's client_email as an Editor.
2. Point GOOGLE_SHEETS_CREDENTIALS_FILE (in .env or the environment) at that
   JSON key file's path.
3. GLIDER_REPORT_SPREADSHEET_ID defaults to the existing "Glider Deployments"
   sheet below; override via env var or --spreadsheet-id for a different one.

Usage
-----
    python scripts/tools/glider_deployment_report.py               # normal weekly run
    python scripts/tools/glider_deployment_report.py --dry-run      # print rows, touch nothing
    python scripts/tools/glider_deployment_report.py --hours 48     # widen the active window
    python scripts/tools/glider_deployment_report.py --force        # overwrite today's tab if it exists

Cron (Friday 8pm local time)
-----------------------------
    0 20 * * 5 cd /path/to/ioos_model_comparisons && /path/to/python scripts/tools/glider_deployment_report.py >> logs/glider_deployment_report.log 2>&1
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd
from erddapy import ERDDAP

from ioos_model_comparisons.env import load_env, require

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GLIDER_DAC_SERVER = "https://gliders.ioos.us/erddap"

DEFAULT_SPREADSHEET_ID = "1y_R_H0NYDV9cE2aBoI_cDFYB4kuKZusIR58U1cAGbco"

HEADER = ["Glider Deployment", "Group", "Location", "Model Evaluation", "Glider Evaluation"]

# A handful of DAC institution strings are long/inconsistent enough to be
# worth shortening for a report meant to be scanned quickly. Anything not
# listed here is passed through unchanged — add to this as new operators
# show up.
INSTITUTION_LABELS = {
    "Naval Oceanographic Office": "Navy",
}

# Ordered (name, lon_min, lon_max, lat_min, lat_max) boxes, checked in order —
# first match wins, so more specific/local boxes come before broad
# ocean-basin fallbacks. A longitude range "wraps" (lon_min > lon_max) to
# cross the antimeridian.
#
# These are hand-tuned to label gliders the way they've been labeled by hand
# in the "Glider Deployments" sheet, not derived from an authoritative
# boundary data set. The western Atlantic offshore boxes in particular
# (South Atlantic Bight / Bahamas / Mid-Atlantic Bight / Gulf Stream /
# Sargasso Sea) sit only a couple of degrees apart, so a glider near a
# boundary can land in the "wrong" one — Location is plain text, so just
# correct it by hand when that happens.
REGION_BOXES = [
    ("Chukchi Sea",           -180.0, -155.0,  65.0,  75.0),
    ("Guam",                   140.0,  150.0,   8.0,  20.0),
    ("Hawaii",                -162.0, -152.0,  15.0,  25.0),
    ("Northeast Pacific",     -135.0, -122.0,  30.0,  50.0),
    ("Eastern Pacific",       -115.0,  -90.0,   0.0,  20.0),
    ("Gulf of Mexico",         -98.0,  -80.5,  18.0,  31.0),
    ("South Atlantic Bight",   -82.0,  -78.0,  27.0,  33.0),
    ("Bahamas",                -80.5,  -72.0,  20.0,  28.0),
    ("Mid-Atlantic Bight",     -77.0,  -65.0,  35.0,  41.5),
    ("Gulf Stream",            -68.0,  -58.0,  35.5,  40.0),
    ("Sargasso Sea",           -78.0,  -50.0,  25.0,  36.0),
    ("Caribbean",              -89.0,  -50.0,   7.0,  27.0),
    # Broad ocean-basin fallbacks for anything outside the named boxes above.
    ("Arctic Ocean",          -180.0,  180.0,  66.0,  90.0),
    ("Southern Ocean",        -180.0,  180.0, -90.0, -55.0),
    ("North Atlantic",         -80.0,   20.0,   0.0,  66.0),
    ("South Atlantic",         -70.0,   20.0, -55.0,   0.0),
    ("Indian Ocean",            20.0,  110.0, -55.0,  30.0),
    ("North Pacific",          110.0,  -90.0,   0.0,  66.0),
    ("South Pacific",          110.0,  -70.0, -55.0,   0.0),
]


def _lon_in_range(lon, lo, hi):
    if lo <= hi:
        return lo <= lon <= hi
    return lon >= lo or lon <= hi  # wraps across the antimeridian


def classify_location(lat, lon):
    """Best-effort named-region label for a (lat, lon) point. See REGION_BOXES."""
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    for name, lon_min, lon_max, lat_min, lat_max in REGION_BOXES:
        if lat_min <= lat <= lat_max and _lon_in_range(lon, lon_min, lon_max):
            return name
    return "Unknown"


def fetch_active_gliders(hours=24):
    """One row per glider dataset with data in the last *hours* hours, read
    from the DAC's allDatasets metadata table — a single lightweight request
    rather than downloading each glider's full track."""
    t0 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)

    e = ERDDAP(server=GLIDER_DAC_SERVER, protocol="tabledap")
    e.dataset_id = "allDatasets"
    e.variables = [
        "datasetID", "institution",
        "minLongitude", "maxLongitude", "minLatitude", "maxLatitude",
        "minTime", "maxTime",
    ]
    e.constraints = {"maxTime>=": t0.strftime("%Y-%m-%dT%H:%M:%SZ")}

    df = e.to_pandas(response="csv", skiprows=(1,))
    df = df[df["datasetID"] != "allDatasets"]
    df = df[~df["datasetID"].str.contains("delayed", case=False, na=False)]

    if df.empty:
        logger.warning(f"No active glider datasets found in the last {hours} hours")
        return pd.DataFrame(columns=["Glider Deployment", "Group", "Location"])

    df["Group"] = df["institution"].map(lambda i: INSTITUTION_LABELS.get(i, i))
    mid_lat = (df["minLatitude"] + df["maxLatitude"]) / 2
    mid_lon = (df["minLongitude"] + df["maxLongitude"]) / 2
    df["Location"] = [classify_location(la, lo) for la, lo in zip(mid_lat, mid_lon)]
    df = df.rename(columns={"datasetID": "Glider Deployment"})

    df = df.sort_values(["Location", "Group", "Glider Deployment"]).reset_index(drop=True)
    logger.info(f"Found {len(df)} active glider datasets")
    return df[["Glider Deployment", "Group", "Location"]]


def build_rows(df):
    rows = [HEADER]
    for _, r in df.iterrows():
        rows.append([r["Glider Deployment"], r["Group"], r["Location"], "", ""])
    return rows


def write_worksheet(spreadsheet_id, credentials_file, tab_name, rows, force=False):
    import gspread

    gc = gspread.service_account(filename=credentials_file)
    sh = gc.open_by_key(spreadsheet_id)

    existing_titles = [ws.title for ws in sh.worksheets()]
    if tab_name in existing_titles:
        if not force:
            raise SystemExit(
                f"Tab '{tab_name}' already exists in the spreadsheet — refusing to "
                f"overwrite it (it may already have operator input). Pass --force to "
                f"overwrite it anyway."
            )
        ws = sh.worksheet(tab_name)
        ws.clear()
        logger.info(f"Cleared existing tab '{tab_name}' (--force)")
    else:
        ws = sh.add_worksheet(title=tab_name, rows=len(rows) + 5, cols=len(HEADER) + 2, index=0)
        logger.info(f"Created new tab '{tab_name}' (placed first)")

    ws.update(rows, range_name="A1", value_input_option="USER_ENTERED")
    ws.format("A1:E1", {"textFormat": {"bold": True}})
    ws.freeze(rows=1)
    try:
        ws.columns_auto_resize(0, len(HEADER) - 1)
    except Exception as e:
        logger.debug(f"Column auto-resize skipped: {e}")

    return ws.url


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=int, default=24, help="Active window in hours (default: 24)")
    p.add_argument("--tab-name", default=None, help="Worksheet tab name (default: today's date, YYYY-MM-DD)")
    p.add_argument("--spreadsheet-id", default=None, help="Target spreadsheet ID (default: $GLIDER_REPORT_SPREADSHEET_ID or the built-in default)")
    p.add_argument("--credentials", default=None, help="Path to the service account JSON key (default: $GOOGLE_SHEETS_CREDENTIALS_FILE)")
    p.add_argument("--force", action="store_true", help="Overwrite the tab if it already exists")
    p.add_argument("--dry-run", action="store_true", help="Fetch and print the rows without touching Google Sheets")
    return p.parse_args()


def main():
    args = parse_args()
    load_env()

    tab_name = args.tab_name or dt.date.today().isoformat()

    gliders = fetch_active_gliders(hours=args.hours)
    rows = build_rows(gliders)

    if args.dry_run:
        for row in rows:
            print("\t".join(row))
        return

    spreadsheet_id = args.spreadsheet_id or require(
        "GLIDER_REPORT_SPREADSHEET_ID",
        hint=f"Falling back to the built-in default ({DEFAULT_SPREADSHEET_ID}).",
    ) or DEFAULT_SPREADSHEET_ID

    credentials_file = args.credentials or require(
        "GOOGLE_SHEETS_CREDENTIALS_FILE",
        hint="Set this to the path of your Google service account JSON key.",
    )
    if not credentials_file:
        raise SystemExit(
            "No service account credentials file given. Set GOOGLE_SHEETS_CREDENTIALS_FILE "
            "(in .env or the environment) or pass --credentials."
        )

    url = write_worksheet(spreadsheet_id, credentials_file, tab_name, rows, force=args.force)
    logger.info(f"Wrote {len(rows) - 1} gliders to tab '{tab_name}': {url}")


if __name__ == "__main__":
    main()
