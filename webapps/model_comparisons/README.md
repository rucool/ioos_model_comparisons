# IOOS Model Comparisons Archive Explorer

A Flask web application that serves an interactive ocean model comparison
viewer. The plot images themselves are served by Apache from
`rucool.marine.rutgers.edu`; this app works out which images exist and
presents them.

---

## Releasing (the normal path)

Releases are built from git and deployed by script; never zip this folder by
hand. From the repo root:

```bash
make check      # tests + vendored-package/requirements check
make deploy     # build dist/model_comparisons-<version>.zip, install on baffin, verify /healthz
make rollback   # back to the previous release
```

See `docs/deployment.md` for the one-time server setup
(`scripts/release/deploy.conf`, the uwsgi vassal, `shared/.env`).

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r ../../requirements-dev.txt
python app.py                 # http://localhost:5001
python -m pytest              # the web app test suite (offline)
```

## Settings (environment, or `.env` in this directory)

| Variable | Purpose |
|---|---|
| `MONGODB_URI` | MongoDB with the figure catalog, region overrides, front editor and users. Without it the public site still works with legacy discovery. |
| `DISCOVERY_BACKEND` | How images are found. `legacy` (default) probes rucool over HTTP. `shadow` runs both and logs differences. `auto` uses the catalog with automatic fallback. `catalog` uses the catalog only. See `discovery.py`. |
| `FLASK_SECRET_KEY`, `AUTH_INSECURE_COOKIE` | Sign-in for the `/fronts` and `/regions` editors (below). |
| `LOCAL_PLOTS_DIR` | Optional local mirror of the plot tree, read first for `locations.json`. |

`GET /healthz` reports the release version, MongoDB, the discovery backend in
use, catalog freshness and the config hash. `?strict=1` returns 503 when
degraded.

## What's in here

| Path | Purpose |
|---|---|
| `app.py`, `wsgi.py` | Flask app and WSGI entry point. Importing has no side effects. |
| `discovery.py` | Finds images and profile positions, from the catalog or the legacy way. |
| `fronts_bp.py`, `regions_bp.py`, `ohc_bp.py`, `auth.py` | Editors, OHC table, sign-in |
| `ioos_model_comparisons/` | **Generated** copy of library modules and `config/*.yml`. Never edit it here; edit the top level and run `make sync-vendor`. |
| `templates/`, `static/` | Pages, CSS, JS and images |
| `tests/` | pytest suite (not shipped) |
| `dev_checks/` | Ad hoc network scripts (not shipped) |

Regions, models, comparison panels and portal lists are configuration
(`ioos_model_comparisons/config/` at the repo root); see
`docs/configuration.md`.

## Troubleshooting

- **Images not loading:** confirm the server can reach
  `rucool.marine.rutgers.edu` over HTTPS, and check `/healthz`.
- **A page 500s with no traceback after a deploy:** usually a missing
  dependency. `make check-vendor` compares imports against `requirements.txt`.

---

## Gulf Stream front editor (`/fronts`)

Hand-correct the digitized north wall and altimetry rings.

**Access:** reads under `/fronts` are **public** — the editor page, the wall and
ring geometry, the stored SST/SLA overlays and the live OISST layer are all
viewable without signing in. **Writes** (save, revert) require a signed-in
account with the `editor` role plus a CSRF token. Because every GET here is
world-readable, do not add a `/fronts` route returning anything you would not
publish.

The region-config editor at `/regions` is different: it is gated end to end,
reads included.

### Configuration

Two environment variables, read via `os.getenv` (a local `.env` at the repo
root is also honoured — real environment variables always win):

    MONGODB_URI        store of record for walls, rings and map overlays
    FLASK_SECRET_KEY   signs the session cookie

`FLASK_SECRET_KEY` **must be the same in every gunicorn worker**, so set it in
the systemd/gunicorn unit's `Environment=` — not generated per process, or
users appear randomly signed out as requests land on different workers. This
is also why no Flask-Session, Redis or sticky sessions are needed: the session
is a stateless signed cookie. Rotating the key signs everyone out, which is
the intended way to revoke all sessions.

If `FLASK_SECRET_KEY` is unset the editor self-disables (503) and the public
dashboard is unaffected.

Over plain HTTP in local dev, set `AUTH_INSECURE_COOKIE=1` — otherwise the
`Secure` cookie is dropped and sign-in appears to silently fail.

### Accounts

    python scripts/tools/manage_front_users.py add --username msmith
    python scripts/tools/manage_front_users.py list

No self-signup and no password reset by design. Passwords are bcrypt; changing
one (or disabling an account) rotates a `session_token` that invalidates that
user's existing cookies everywhere.

### Editing model

Saving appends a **version** rather than overwriting. The automatic
full-resolution wall (~3000 vertices) is a separate document from a hand edit
(~200 vertices, because a browser cannot usefully offer 3000 draggable
handles), so editing can never destroy it. "Revert to auto" appends another
version copied from the newest automatic one — the hand edit stays in history.

A saved wall is marked `qc_stale`: the stored support/displacement numbers
describe the automatic geometry, not the edited line.

The nightly digitizer refuses to write an automatic version for a day that
already has a hand edit (`--force-auto` overrides), so a routine re-run cannot
silently supersede someone's work.
