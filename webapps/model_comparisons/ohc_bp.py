"""
ohc_bp.py — daily OHC export table.

Built for Lev's request: Platform ID / date / OHC-at-platform /
OHC-at-model(s) / OHC-NESDIS, read from Argo and glider profiles and rolled
up into a daily table.

The numbers themselves are already computed by the profile-comparison
scripts (daily_glider_profiles_comparisons_*.py, argo_profile_model_
comparisons*.py) — this blueprint just reads what db.log_ohc_metrics wrote
and renders/exports it. No new computation happens here.

Public, read-only — same access model as the main dashboard (/). There is
nothing to edit; unlike /fronts and /regions this has no before_request gate.
"""

import csv
import io

from flask import Blueprint, Response, jsonify, render_template, request

from ioos_model_comparisons.db import OHC_MODEL_FIELDS, fetch_ohc_metrics

ohc_bp = Blueprint("ohc", __name__, url_prefix="/ohc-table")

# Column order for both the JSON rows and the CSV export.
_COLUMNS = (
    ["platform_type", "platform_id", "date", "region", "lat", "lon", "ohc_platform"]
    + [f"ohc_{m}" for m in OHC_MODEL_FIELDS]
)


def _parse_filters():
    date_from = (request.args.get("date_from") or "").strip() or None
    date_to = (request.args.get("date_to") or "").strip() or None
    platform_type = (request.args.get("platform_type") or "").strip() or None
    if platform_type not in (None, "argo", "glider"):
        platform_type = None
    return date_from, date_to, platform_type


def _rows():
    date_from, date_to, platform_type = _parse_filters()
    docs = fetch_ohc_metrics(date_from=date_from, date_to=date_to, platform_type=platform_type)
    return [{col: doc.get(col) for col in _COLUMNS} for doc in docs]


@ohc_bp.route("")
@ohc_bp.route("/")
def table():
    return render_template("ohc_table.html")


@ohc_bp.route("/api/data")
def api_data():
    rows = _rows()
    return jsonify({"rows": rows, "columns": _COLUMNS})


@ohc_bp.route("/api/csv")
def api_csv():
    rows = _rows()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    date_from, date_to, platform_type = _parse_filters()
    name_bits = ["ohc_export", platform_type or "all"]
    if date_from or date_to:
        name_bits.append(f"{date_from or 'earliest'}_to_{date_to or 'latest'}")
    filename = "_".join(name_bits) + ".csv"

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
