"""
auth.py — sign-in for the front editor.

The dashboard is public and stays public. This gates only the editor: read
routes on `/` and `/api/*` are untouched.

Design notes worth keeping:

* **A missing FLASK_SECRET_KEY must not take the site down.** It disables
  sign-in and /fronts (503) while `/` keeps serving. Failing closed on the
  feature, open on the public site.
* **The secret must come from the environment, not be generated at import.**
  Flask's session is a client-side signed cookie, so it is stateless and works
  across gunicorn workers — but only if every worker signs with the SAME key.
  A per-process random key makes users appear randomly logged out as requests
  land on different workers. That is also why no Flask-Session/Redis/sticky
  sessions are needed.
* **CSRF is hand-rolled** (~30 lines) rather than pulling Flask-WTF + WTForms
  in for two JSON endpoints and one form. Layers, in order of what actually
  stops the attack: SameSite=Lax on the cookie; an X-CSRF-Token header that a
  cross-origin fetch cannot set without a CORS preflight the app never
  answers; and hmac.compare_digest on the value.
"""

from __future__ import annotations

import hmac
import os
import secrets
from datetime import timedelta

from flask import (Blueprint, abort, current_app, flash, redirect,
                   render_template, request, session, url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)

from ioos_model_comparisons import users as user_store

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to use the front editor."
login_manager.session_protection = "strong"

auth_bp = Blueprint("auth", __name__)


class EditorUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc or {}
        self.id_ = str(self.doc.get("_id"))
        self.username = self.doc.get("username")
        self.name = self.doc.get("name") or self.username
        self.roles = list(self.doc.get("roles") or [])
        self.session_token = self.doc.get("session_token") or ""

    def get_id(self):
        # The token is half the identity: rotating it in the database
        # invalidates every outstanding cookie for this user, everywhere,
        # which is what makes "sign out" and "change password" mean something
        # with a stateless cookie.
        return f"{self.id_}:{self.session_token}"

    def has_role(self, role):
        return role in self.roles

    @property
    def is_active(self):
        return bool(self.doc.get("active", True))


@login_manager.user_loader
def _load_user(composite_id):
    try:
        uid, token = str(composite_id).split(":", 1)
    except ValueError:
        return None
    doc = user_store.get_user_by_id(uid)      # None also means "Mongo down"
    if not doc or not doc.get("active", True):
        return None
    if not hmac.compare_digest(str(doc.get("session_token") or ""), token):
        return None                            # password changed / signed out
    return EditorUser(doc)


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------
def csrf_token():
    tok = session.get("_csrf")
    if not tok:
        tok = secrets.token_urlsafe(32)
        session["_csrf"] = tok
    return tok


def check_csrf():
    """abort(400) unless the request carries the session's CSRF token.

    Minted per session rather than per request: per-request rotation breaks
    multi-tab editing and buys nothing here.
    """
    sent = request.headers.get("X-CSRF-Token")
    if not sent and request.form:
        sent = request.form.get("csrf_token")
    if not sent or not hmac.compare_digest(str(sent), str(session.get("_csrf", ""))):
        abort(400, "CSRF token missing or invalid")


def auth_disabled():
    return bool(current_app.config.get("AUTH_DISABLED"))


def init_auth(app):
    secret = os.getenv("FLASK_SECRET_KEY")
    if not secret:
        app.logger.warning(
            "FLASK_SECRET_KEY is not set — sign-in and the front editor are "
            "disabled (503). The public dashboard is unaffected. Set it in the "
            "systemd/gunicorn unit's Environment=, or in .env for local dev.")
        app.config["AUTH_DISABLED"] = True
        secret = secrets.token_hex(32)          # lets the app boot; nobody can log in
    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_NAME="imc_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Correct because ProxyFix(x_proto=1) is configured, so Flask sees the
        # external scheme. Over plain HTTP in dev the cookie is dropped and
        # login will appear to silently fail — set AUTH_INSECURE_COOKIE=1 then.
        SESSION_COOKIE_SECURE=not os.getenv("AUTH_INSECURE_COOKIE"),
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    )
    login_manager.init_app(app)
    app.jinja_env.globals["csrf_token"] = csrf_token
    app.register_blueprint(auth_bp)
    return app


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if auth_disabled():
        abort(503, "Sign-in is not configured on this server.")
    if current_user.is_authenticated:
        return redirect(url_for("fronts.editor"))

    if request.method == "POST":
        check_csrf()
        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        doc = user_store.get_user(username)

        # doc is None for BOTH "no such user" and "database unreachable".
        # Distinguish only for the message; never for the decision.
        if doc is None and user_store._coll() is None:
            flash("Sign-in is temporarily unavailable. Try again shortly.", "warning")
            return render_template("login.html"), 503

        if user_store.verify_password(doc, password):
            user_store.record_login_success(username)
            login_user(EditorUser(user_store.get_user(username)), remember=False)
            session.permanent = True
            nxt = request.args.get("next")
            # only ever redirect within this site
            if not nxt or not nxt.startswith("/") or nxt.startswith("//"):
                nxt = url_for("fronts.editor")
            return redirect(nxt)

        if doc is not None:
            user_store.record_login_failure(username)
        if doc is not None and user_store.is_locked(user_store.get_user(username)):
            flash(f"Too many failed attempts. Locked for "
                  f"{user_store.LOCKOUT_MINUTES} minutes.", "danger")
        else:
            flash("Invalid username or password.", "danger")
        return render_template("login.html"), 401

    return render_template("login.html")


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    check_csrf()
    logout_user()
    session.pop("_csrf", None)
    return redirect(url_for("index"))
