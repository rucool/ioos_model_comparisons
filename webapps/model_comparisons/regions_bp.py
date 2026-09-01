"""
regions_bp.py — web CRUD for hurricanes.region_configs.

Gated exactly like the front editor: everything under /regions needs a
session, writes need the `editor` role and a CSRF token. Region configs drive
every production plot, so a bad edit here is more expensive than a bad wall
edit — hence validation on every write and full version history.

The live document in region_configs keeps the shape db.apply_colorbar_overrides
expects; history lives alongside in region_config_versions.
"""

from __future__ import annotations

import datetime

from flask import Blueprint, abort, jsonify, render_template, request
from flask_login import current_user

from ioos_model_comparisons import regions_store as rs
from ioos_model_comparisons.regions import region_config

regions_bp = Blueprint("regions", __name__, url_prefix="/regions")


@regions_bp.before_request
def _require_editor():
    from auth import auth_disabled, check_csrf, login_manager
    if auth_disabled():
        abort(503, "The region editor is not configured on this server.")
    if not current_user.is_authenticated:
        if request.is_json or request.path.startswith("/regions/api/"):
            abort(401)
        return login_manager.unauthorized()
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        if not current_user.has_role("editor"):
            abort(403, "Your account cannot edit region configs.")
        check_csrf()


def _iso(v):
    return v.isoformat() if isinstance(v, datetime.datetime) else v


@regions_bp.route("")
@regions_bp.route("/")
def editor():
    return render_template("regions.html")


@regions_bp.route("/api/regions")
def api_regions():
    rows = [{**r, "updated_at": _iso(r.get("updated_at"))}
            for r in rs.list_regions()]
    return jsonify({"regions": rows})


@regions_bp.route("/api/region")
def api_region():
    region = (request.args.get("region") or "").strip()
    if not region:
        return jsonify({"error": "region is required"}), 400
    doc = rs.fetch(region)
    if doc is None:
        return jsonify({"error": f"no config for {region}"}), 404

    # seed_region_configs.py deliberately never writes variables/
    # sea_surface_height (those belong to update_colorbar_limits.py's weekly
    # tuning and colorbar_tuner.py's manual tuning) — so a never-tuned
    # region's live document has neither key at all, and the editor's
    # colorbar-limits section rendered as empty with no way to tell there
    # was ever anything to show. Fill in regions.py's defaults for display
    # only when Mongo doesn't have the key; leave an existing (even empty)
    # tuned value alone.
    try:
        defaults = region_config([region])
    except Exception:
        defaults = {}
    for key in ("variables", "sea_surface_height"):
        doc.setdefault(key, defaults.get(key))

    errors, warnings = rs.validate(doc)
    return jsonify({"region": region, "doc": doc,
                    "errors": errors, "warnings": warnings,
                    "versions": [{**v, "created_at": _iso(v.get("created_at"))}
                                 for v in rs.list_versions(region)]})


@regions_bp.route("/api/version")
def api_version():
    region = (request.args.get("region") or "").strip()
    version = request.args.get("version", type=int)
    if not region or version is None:
        return jsonify({"error": "region and version are required"}), 400
    v = rs.fetch_version(region, version)
    if not v:
        return jsonify({"error": "no such version"}), 404
    return jsonify({**v, "created_at": _iso(v.get("created_at"))})


@regions_bp.route("/api/validate", methods=["POST"])
def api_validate():
    """Dry-run: what would this document do, without writing it."""
    body = request.get_json(silent=True) or {}
    doc = body.get("doc") or {}
    region = (body.get("region") or doc.get("region") or "").strip()
    doc = dict(doc, region=region)
    errors, warnings = rs.validate(rs.normalize(doc))
    return jsonify({"errors": errors, "warnings": warnings,
                    "diff": rs.diff(rs.fetch(region) or {}, rs.normalize(doc))})


@regions_bp.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(silent=True) or {}
    doc = body.get("doc") or {}
    region = (body.get("region") or doc.get("region") or "").strip()
    if not region:
        return jsonify({"error": "region is required"}), 400
    try:
        version = rs.save(region, doc, edited_by=current_user.username,
                          note=body.get("note"))
    except ValueError as exc:
        # validation failure is the caller's fault, and the message names the
        # offending field(s) — surface it rather than a generic 400
        return jsonify({"error": str(exc), "invalid": True}), 400
    if version is None:
        return jsonify({"error": "could not save (is MongoDB reachable?)"}), 503
    return jsonify({"ok": True, "version": version})


@regions_bp.route("/api/revert", methods=["POST"])
def api_revert():
    body = request.get_json(silent=True) or {}
    region = (body.get("region") or "").strip()
    version = body.get("version")
    if not region or version is None:
        return jsonify({"error": "region and version are required"}), 400
    try:
        new_version = rs.revert(region, int(version),
                                edited_by=current_user.username)
    except ValueError as exc:
        return jsonify({"error": str(exc), "invalid": True}), 400
    if new_version is None:
        return jsonify({"error": "could not revert"}), 503
    return jsonify({"ok": True, "version": new_version, "restored_from": version})


@regions_bp.route("/api/delete", methods=["POST"])
def api_delete():
    """Drop the override so regions.py defaults apply again (history kept)."""
    body = request.get_json(silent=True) or {}
    region = (body.get("region") or "").strip()
    if not region:
        return jsonify({"error": "region is required"}), 400
    if not rs.delete(region, edited_by=current_user.username):
        return jsonify({"error": "nothing to delete"}), 404
    return jsonify({"ok": True})
