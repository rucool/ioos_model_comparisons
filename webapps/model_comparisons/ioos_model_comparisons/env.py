"""
env.py — optional .env loading for local development.

The repo's convention is bare `os.getenv` at call sites, with values exported
from the shell (locally) or set in the systemd/gunicorn unit's `Environment=`
(on the server). This adds a convenience layer only:

    load_env()   # reads <repo root>/.env if it exists

**Real environment variables always win.** `load_dotenv(override=False)` means
a value already exported in your shell, or set by systemd, is never replaced by
a stale line in a .env someone forgot about. That ordering is the whole point —
production should not be able to pick up a developer's file.

Safe to call unconditionally: if python-dotenv is missing or there is no .env,
it does nothing and says so at debug level.

.env is gitignored. Never commit real values; .env.example documents the names.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_loaded = False


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_env(path=None, force=False):
    """Load <repo root>/.env into the environment if present. Returns bool."""
    global _loaded
    if _loaded and not force:
        return True
    env_path = Path(path) if path else repo_root() / ".env"
    if not env_path.is_file():
        logger.debug(f"no .env at {env_path} — using the ambient environment")
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed — ignoring %s", env_path)
        return False
    # override=False: an already-set variable (shell export, systemd
    # Environment=) beats the file. Do not change this.
    load_dotenv(env_path, override=False)
    _loaded = True
    logger.debug(f"loaded {env_path} (existing environment takes precedence)")
    return True


def require(name, hint=""):
    """Return os.environ[name], or None with a clear warning naming the fix."""
    load_env()
    val = os.getenv(name)
    if not val:
        logger.warning(f"{name} is not set. {hint}".strip())
        return None
    return val
