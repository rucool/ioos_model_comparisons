"""
fronts_bp.py — the Gulf Stream front viewer/editor, backed by MongoDB.

Read routes under /fronts are public: anyone can see the Gulf Stream front,
history, rings and overlays. Mutation routes stay behind a session and the
editor role.

Reads are public; the WRITE gate is a blueprint-level before_request
rather than per-route
decorators, because the failure mode of decorators is someone adding a write
route and forgetting one.

Versioning replaces the old file-based auto_backup/ dance: an edit appends a
new version, so the automatic full-resolution geometry is never overwritten
and "revert" is just another append.
"""

from __future__ import annotations

import datetime
import io
import json
import re

from flask import (Blueprint, abort, current_app, jsonify, render_template,
                   request, send_file)
from flask_login import current_user

from ioos_model_comparisons.fronts import oisst, store
from ioos_model_comparisons.fronts.webmap import simplify_lines

fronts_bp = Blueprint("fronts", __name__, url_prefix="/fronts")

# What the browser can actually render as draggable handles. A traced wall is
# ~3000 vertices; hand it that many and the page is unusable.
MAX_EDIT_VERTICES = 220
BASE_TOLERANCE_DEG = 0.01

_SAFE_FIELDS = ("sst", "sla")


def _extent_for(stamp=None):
    """Map extent for a scene: whatever the stored overlay used, else the
    region default. Keeps OISST aligned with the GOES overlay it sits under."""
    if stamp:
        got = store.fetch_overlay(stamp, "sst")
        if got and got[1].get("extent"):
            return [float(v) for v in got[1]["extent"]]
    from ioos_model_comparisons.regions import region_config
    return [float(v) for v in region_config("gulf_stream")["extent"]]


def _safe_stamp(stamp):
    """Validate by SHAPE, not by sanitising. Anything that is not exactly a
    YYYYmmddTHHMM token is rejected outright."""
    s = (stamp or "").strip()
    return s if store.STAMP_RE.match(s) else None


@fronts_bp.before_request
def _require_editor():
    """Reads are public; writes require a signed-in editor.

    Deliberate, and worth stating because it differs from /regions (which is
    gated end to end): anyone may LOOK at the digitized wall, the rings, the
    stored overlays and the OISST layer, including loading the editor page
    itself. Only saving, reverting and any other non-safe method needs a
    session, the `editor` role and a CSRF token.

    The consequence to keep in mind: every GET here is world-readable, so do
    not add a route under /fronts that returns something you would not publish
    (user lists, tokens, unpublished analysis). Put anything like that behind
    its own gate.
    """
    from auth import auth_disabled, check_csrf, login_manager
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if auth_disabled():
        abort(503, "Front editing is not configured on this server.")
    if not current_user.is_authenticated:
        # A JSON caller should get a status code, not a 302 to an HTML page.
        if request.is_json or request.path.startswith("/fronts/api/"):
            abort(401)
        return login_manager.unauthorized()
    if not current_user.has_role("editor"):
        abort(403, "Your account cannot edit the wall.")
    check_csrf()


# ---------------------------------------------------------------------------
@fronts_bp.route("")
@fronts_bp.route("/")
def editor():
    can_edit = (
        current_user.is_authenticated
        and current_user.has_role("editor")
    )
    return render_template("fronts.html", can_edit=can_edit)


@fronts_bp.route("/api/days")
def api_days():
    days = []
    for d in store.list_wall_days():
        days.append({
            "stamp": d.get("stamp"),
            "day": d.get("day"),
            "time": d["time"].isoformat() if isinstance(
                d.get("time"), datetime.datetime) else d.get("time"),
            "version": d.get("version"),
            "n_versions": d.get("n_versions"),
            "origin": d.get("origin"),
            "resolution": d.get("resolution"),
            "qc_pass": d.get("qc_pass"),
            "qc_stale": d.get("qc_stale"),
            "n_pieces": d.get("n_pieces"),
            "n_vertices": d.get("n_vertices"),
            "edited_by": d.get("edited_by"),
        })
    return jsonify({"days": days})


@fronts_bp.route("/api/versions")
def api_versions():
    stamp = _safe_stamp(request.args.get("stamp"))
    if not stamp:
        return jsonify({"error": "bad or missing stamp"}), 400
    out = []
    for v in store.list_wall_versions(stamp):
        out.append({k: (v[k].isoformat() if isinstance(v.get(k), datetime.datetime)
                        else v.get(k))
                    for k in ("version", "origin", "resolution", "n_pieces",
                              "n_vertices", "edited_by", "created_at", "note",
                              "qc_stale", "parent_version")})
    return jsonify({"stamp": stamp, "versions": out})


@fronts_bp.route("/api/features")
def api_features():
    stamp = _safe_stamp(request.args.get("stamp"))
    if not stamp:
        return jsonify({"error": "bad or missing stamp"}), 400
    version = request.args.get("version", type=int)

    doc = store.fetch_wall(stamp, version=version)
    if doc is None:
        return jsonify({"error": f"no wall for {stamp}"}), 404

    lines = store.geometry_to_lines(doc.get("geometry"))
    original_n = int(sum(len(l) for l in lines))
    simplified, tol = simplify_lines(lines, BASE_TOLERANCE_DEG,
                                     max_points=MAX_EDIT_VERTICES)

    wall_features = [{
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[float(x), float(y)] for x, y in l]},
        "properties": {"kind": "wall", "piece": i},
    } for i, l in enumerate(simplified) if len(l) >= 2]

    rings_doc = store.fetch_rings(stamp) or {}
    ring_features = []
    for r in store.normalize_rings(rings_doc.get("rings")):
        if not isinstance(r.get("geometry"), dict):
            continue
        ring_features.append({
            "type": "Feature",
            "geometry": r["geometry"],
            "properties": {"kind": "ring", "ring_kind": r.get("kind"),
                           "days_tracked": r.get("days_tracked"),
                           "radius_km": r.get("radius_km"),
                           "amplitude_cm": r.get("amplitude_cm")},
        })

    props = dict(doc.get("properties") or {})
    props.update(version=doc.get("version"), origin=doc.get("origin"),
                 resolution=doc.get("resolution"), qc_stale=doc.get("qc_stale"),
                 edited_by=doc.get("edited_by"))
    return jsonify({
        "stamp": stamp,
        "wall": {"type": "FeatureCollection", "features": wall_features},
        "rings": {"type": "FeatureCollection", "features": ring_features},
        "rings_version": rings_doc.get("version"),
        "wall_properties": props,
        "overlay": {"extent": (store.fetch_overlay(stamp, "sst") or (None, {}))[1]
                    .get("extent")},
        "simplify": {"original_n_vertices": original_n,
                     "edit_n_vertices": int(sum(len(l) for l in simplified)),
                     "tolerance_deg": round(float(tol), 5)},
    })


@fronts_bp.route("/api/overlay")
def api_overlay():
    stamp = _safe_stamp(request.args.get("stamp"))
    field = request.args.get("field", "sst")
    if not stamp or field not in _SAFE_FIELDS:
        return jsonify({"error": "bad stamp or field"}), 400
    got = store.fetch_overlay(stamp, field)
    if got is None:
        return jsonify({"error": "no overlay for that day"}), 404
    png, meta = got
    resp = send_file(io.BytesIO(png), mimetype="image/png",
                     download_name=f"{field}_{stamp}.png")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@fronts_bp.route("/api/oisst/meta")
def api_oisst_meta():
    """Date range, colormap choices and suggested limits for the current view."""
    stamp = _safe_stamp(request.args.get("stamp"))
    extent = _extent_for(stamp)
    rng = oisst.available_range()
    date = request.args.get("date")
    return jsonify({
        "available": {"first": rng[0], "last": rng[1]} if rng else None,
        "colormaps": oisst.COLORMAPS,
        "extent": extent,
        "stats": oisst.stats(date, extent) if date else None,
    })


@fronts_bp.route("/api/oisst")
def api_oisst():
    """Render an OISST field on the fly.

    Rendered per request rather than pre-generated because the point is to
    change date and colour scale interactively; the underlying slice is cached
    so only the first request for a date touches the network.
    """
    date = (request.args.get("date") or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    stamp = _safe_stamp(request.args.get("stamp"))
    extent = _extent_for(stamp)

    def num(name):
        v = request.args.get(name)
        try:
            return float(v) if v not in (None, "") else None
        except ValueError:
            return None

    vmin, vmax, stride = num("vmin"), num("vmax"), num("stride")
    if request.args.get("bar"):
        # legend strip, built from the same norm as the data render
        png = oisst.render_colorbar(cmap=request.args.get("cmap", "turbo"),
                                    vmin=vmin if vmin is not None else 0.0,
                                    vmax=vmax if vmax is not None else 1.0,
                                    stride=stride)
        resp = send_file(io.BytesIO(png), mimetype="image/png",
                         download_name="oisst_colorbar.png")
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp
    if vmin is not None and vmax is not None and vmin >= vmax:
        return jsonify({"error": "vmin must be less than vmax"}), 400
    try:
        png, meta = oisst.render(date, extent,
                                 cmap=request.args.get("cmap", "turbo"),
                                 vmin=vmin, vmax=vmax, stride=stride)
    except Exception as exc:
        # PSL being unreachable or a date outside the archive should read as a
        # missing layer, not a broken editor
        current_app.logger.warning(f"OISST render failed for {date}: {exc}")
        return jsonify({"error": f"could not load OISST for {date}"}), 502
    resp = send_file(io.BytesIO(png), mimetype="image/png",
                     download_name=f"oisst_{date}.png")
    # same URL always yields the same image, so let the browser keep it
    resp.headers["Cache-Control"] = "private, max-age=86400"
    resp.headers["X-OISST-Meta"] = json.dumps(meta)
    return resp


@fronts_bp.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(silent=True) or {}
    stamp = _safe_stamp(body.get("stamp"))
    if not stamp:
        return jsonify({"error": "bad or missing stamp"}), 400

    cur = store.fetch_wall(stamp)
    if cur is None:
        return jsonify({"error": f"no wall for {stamp}"}), 404

    lines = []
    for feat in (body.get("wall") or {}).get("features", []):
        g = feat.get("geometry") or {}
        if g.get("type") != "LineString":
            continue
        coords = [[float(x), float(y)] for x, y in g.get("coordinates", [])]
        if len(coords) >= 2:
            lines.append(coords)
    if not lines:
        return jsonify({"error": "refusing to save a wall with no line left"}), 400

    geometry = ({"type": "LineString", "coordinates": lines[0]} if len(lines) == 1
                else {"type": "MultiLineString", "coordinates": lines})

    props = dict(cur.get("properties") or {})
    version = store.save_wall_version(
        stamp, geometry, props, origin="manual", resolution="simplified",
        source=cur.get("source"), edited_by=current_user.username,
        parent_version=cur.get("version"),
        simplify={"original_n_vertices": body.get("original_n_vertices"),
                  "edit_n_vertices": int(sum(len(c) for c in lines)),
                  "tolerance_deg": body.get("tolerance_deg")},
        note=body.get("note"),
        # The stored QC describes the automatic geometry and is no longer true
        # of this line. Flag it rather than leave numbers that look current.
        qc_stale=True, qc_pass=cur.get("qc_pass"))
    if version is None:
        return jsonify({"error": "could not save (is MongoDB reachable?)"}), 503

    result = {"ok": True, "version": version, "n_pieces": len(lines),
              "n_vertices": int(sum(len(c) for c in lines))}

    rings = body.get("rings")
    if rings is not None:
        prev = store.fetch_rings(stamp) or {}
        prev_rings = store.normalize_rings(prev.get("rings"))
        out = []
        for feat in rings.get("features", []):
            g = feat.get("geometry") or {}
            if g.get("type") != "Polygon" or not g.get("coordinates"):
                continue
            ring = g["coordinates"][0]
            if len(ring) < 4:
                continue
            fp = feat.get("properties") or {}
            clon = sum(p[0] for p in ring) / len(ring)
            clat = sum(p[1] for p in ring) / len(ring)
            src = next((p for p in prev_rings
                        if p.get("kind") == fp.get("ring_kind")
                        and abs((p.get("lon") or 1e9) - clon) < 0.4
                        and abs((p.get("lat") or 1e9) - clat) < 0.4), None)
            kind = fp.get("ring_kind") or (src or {}).get("kind") or "warm"
            out.append({
                "feature": f"{kind}_core_ring", "kind": kind,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                # amplitude is a measurement from the SLA field and cannot be
                # recomputed from a hand-drawn outline: carry the original
                # through, leave it null for a newly drawn ring
                "amplitude_cm": (src or {}).get("amplitude_cm"),
                "radius_km": (src or {}).get("radius_km"),
                "compactness": (src or {}).get("compactness"),
                "centroid_lon": round(clon, 4), "centroid_lat": round(clat, 4),
                "days_tracked": (src or {}).get("days_tracked"),
                "edited_by_hand": src is None or bool(fp.get("dirty")),
            })
        rv = store.save_rings_version(stamp, out, origin="manual",
                                      edited_by=current_user.username,
                                      parent_version=prev.get("version"))
        result["rings_version"] = rv
    return jsonify(result)


@fronts_bp.route("/api/revert", methods=["POST"])
def api_revert():
    """Append a version copied from the newest automatic one.

    Not a delete: the hand edit stays in history and remains viewable, it is
    simply no longer current.
    """
    body = request.get_json(silent=True) or {}
    stamp = _safe_stamp(body.get("stamp"))
    if not stamp:
        return jsonify({"error": "bad or missing stamp"}), 400
    auto = store.fetch_wall(stamp, origin="auto")
    if auto is None:
        return jsonify({"error": "no automatic version to revert to"}), 404
    version = store.save_wall_version(
        stamp, auto.get("geometry"), auto.get("properties") or {},
        origin="auto_restore", resolution=auto.get("resolution", "full"),
        source=auto.get("source"), edited_by=current_user.username,
        parent_version=auto.get("version"),
        note=f"reverted to automatic v{auto.get('version')}",
        qc_pass=auto.get("qc_pass"), qc_stale=False)
    if version is None:
        return jsonify({"error": "could not revert"}), 503
    return jsonify({"ok": True, "version": version,
                    "restored_from": auto.get("version")})
