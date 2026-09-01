"""
users.py — accounts for the web editors, stored in MongoDB.

The collection is named `front_users` for historical reasons but gates every
signed-in section of webapps/model_comparisons — /fronts and /regions alike.

Lives at package level rather than under fronts/ because users are not a
fronts concept; a second gated feature would want the same table.

**This module deliberately breaks db.py's contract, and that is the point.**
db.py degrades on failure: every function returns None and callers carry on
with defaults. For authentication, "carrying on" would mean letting people in
without a database. So:

    get_user() returns None for BOTH "no such user" and "Mongo unreachable",
    and every caller MUST treat None as DENY.

The login route distinguishes the two cases only to render the right message
(503 "temporarily unavailable" vs 401 "invalid credentials") — never to grant
access. Do not "fix" this inconsistency with db.py.

Passwords are bcrypt (cost 12). `session_token` is what makes logout and
password changes meaningful with a stateless signed cookie: Flask-Login's
user id embeds it, so rotating the token invalidates every outstanding cookie
for that user across all gunicorn workers at once.

Lockout counters live in the document, not in process memory, because
gunicorn runs several workers — an in-process counter is defeated by simply
retrying until you land on a different one.
"""

from __future__ import annotations

import datetime
import logging
import secrets

from ioos_model_comparisons.db import get_client

logger = logging.getLogger(__name__)

DB_NAME = "hurricanes"
USERS_COLL = "front_users"

BCRYPT_ROUNDS = 12
MAX_PASSWORD_BYTES = 72          # bcrypt truncates beyond this
MIN_PASSWORD_LEN = 10
MAX_FAILED_LOGINS = 8
LOCKOUT_MINUTES = 15

# A real bcrypt hash of a value nobody knows, compared against when the
# username does not exist so that a missing account and a wrong password take
# the same wall-clock time. Without it the login endpoint is a free username
# oracle.
_DUMMY_HASH = b"$2b$12$C6UzMDM.H6dfI/f/IKcEe.eDjhV0RIkNKAlEBhtOfHkYqyBB1yPYy"


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _coll():
    client = get_client()
    if client is None:
        return None
    return client[DB_NAME][USERS_COLL]


def ensure_user_indexes():
    client = get_client()
    if client is None:
        return
    try:
        import pymongo
        client[DB_NAME][USERS_COLL].create_index(
            "username", unique=True, background=True)
    except Exception as exc:
        logger.warning(f"ensure_user_indexes failed: {exc}")


# ---------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------
def hash_password(password):
    """bcrypt hash as str. Raises ValueError on an unusable password."""
    import bcrypt
    if password is None:
        raise ValueError("password is required")
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        # Reject rather than silently truncate: a user who set a 100-char
        # passphrase would otherwise be authenticated by its first 72 bytes.
        raise ValueError(
            f"password is {len(raw)} bytes; bcrypt ignores everything past "
            f"{MAX_PASSWORD_BYTES}. Choose a shorter one.")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(user, password):
    """True only for an existing, active, unlocked user with the right password.

    Always performs a bcrypt comparison — against a dummy hash when `user` is
    None — so timing does not reveal whether the account exists.
    """
    import bcrypt
    raw = (password or "").encode("utf-8")[:MAX_PASSWORD_BYTES]
    stored = None
    if user:
        stored = (user.get("password_hash") or "").encode() or None
    try:
        ok = bcrypt.checkpw(raw, stored or _DUMMY_HASH)
    except Exception:
        ok = False
    if not user or not stored:
        return False
    if not user.get("active", True) or is_locked(user):
        return False
    return bool(ok)


def is_locked(user):
    lu = (user or {}).get("locked_until")
    return bool(lu and lu > _utcnow())


# ---------------------------------------------------------------------------
# reads  — None means DENY, whatever the reason
# ---------------------------------------------------------------------------
def get_user(username):
    coll = _coll()
    if coll is None:
        logger.warning("front_users unavailable — denying (Mongo unreachable?)")
        return None
    try:
        return coll.find_one({"username": (username or "").strip().lower()})
    except Exception as exc:
        logger.warning(f"get_user failed: {exc}")
        return None


def get_user_by_id(user_id):
    coll = _coll()
    if coll is None:
        return None
    try:
        from bson import ObjectId
        return coll.find_one({"_id": ObjectId(str(user_id))})
    except Exception as exc:
        logger.warning(f"get_user_by_id failed: {exc}")
        return None


def list_users():
    coll = _coll()
    if coll is None:
        return []
    try:
        return list(coll.find({}, {"password_hash": 0, "session_token": 0}
                              ).sort("username", 1))
    except Exception as exc:
        logger.warning(f"list_users failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------
def create_user(username, password, *, name=None, email=None, roles=("editor",)):
    coll = _coll()
    if coll is None:
        return False
    username = (username or "").strip().lower()
    if not username:
        raise ValueError("username is required")
    try:
        ensure_user_indexes()
        now = _utcnow()
        coll.insert_one({
            "username": username,
            "name": name or username,
            "email": email,
            "password_hash": hash_password(password),
            "roles": list(roles),
            "active": True,
            "session_token": secrets.token_urlsafe(32),
            "failed_logins": 0,
            "locked_until": None,
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        })
        return True
    except Exception as exc:
        logger.warning(f"create_user failed: {exc}")
        return False


def set_password(username, password):
    """Change a password AND rotate session_token, logging that user out
    everywhere — otherwise old cookies stay valid after a compromise."""
    coll = _coll()
    if coll is None:
        return False
    try:
        r = coll.update_one(
            {"username": (username or "").strip().lower()},
            {"$set": {"password_hash": hash_password(password),
                      "session_token": secrets.token_urlsafe(32),
                      "failed_logins": 0, "locked_until": None,
                      "updated_at": _utcnow()}})
        return r.matched_count == 1
    except Exception as exc:
        logger.warning(f"set_password failed: {exc}")
        return False


def set_active(username, active):
    coll = _coll()
    if coll is None:
        return False
    try:
        r = coll.update_one(
            {"username": (username or "").strip().lower()},
            {"$set": {"active": bool(active),
                      "session_token": secrets.token_urlsafe(32),
                      "updated_at": _utcnow()}})
        return r.matched_count == 1
    except Exception as exc:
        logger.warning(f"set_active failed: {exc}")
        return False


def set_roles(username, roles):
    coll = _coll()
    if coll is None:
        return False
    try:
        r = coll.update_one({"username": (username or "").strip().lower()},
                            {"$set": {"roles": list(roles),
                                      "updated_at": _utcnow()}})
        return r.matched_count == 1
    except Exception as exc:
        logger.warning(f"set_roles failed: {exc}")
        return False


def record_login_success(username):
    coll = _coll()
    if coll is None:
        return
    try:
        coll.update_one({"username": username},
                        {"$set": {"failed_logins": 0, "locked_until": None,
                                  "last_login_at": _utcnow()}})
    except Exception as exc:
        logger.warning(f"record_login_success failed: {exc}")


def record_login_failure(username):
    """Count a failure and lock the account once it crosses the threshold."""
    coll = _coll()
    if coll is None:
        return
    try:
        doc = coll.find_one_and_update(
            {"username": (username or "").strip().lower()},
            {"$inc": {"failed_logins": 1}},
            return_document=True) if hasattr(coll, "find_one_and_update") else None
        if doc and int(doc.get("failed_logins", 0)) >= MAX_FAILED_LOGINS:
            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {"locked_until": _utcnow() + datetime.timedelta(
                    minutes=LOCKOUT_MINUTES)}})
            logger.warning(f"locked {doc['username']} for {LOCKOUT_MINUTES} min "
                           f"after {doc['failed_logins']} failed logins")
    except Exception as exc:
        logger.warning(f"record_login_failure failed: {exc}")
