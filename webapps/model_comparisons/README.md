# IOOS Model Comparisons Archive Explorer — Deployment Guide

A Flask web application that serves an interactive ocean model comparison viewer.
It fetches image data from `rucool.marine.rutgers.edu` and requires no local database.

---

## Requirements

- Python 3.10 or newer
- Network access to `rucool.marine.rutgers.edu`

---

## Installation

### 1. Extract the archive

```bash
unzip ioos-model-comparisons-webapp.zip
cd webapps
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the app

### Development / quick test

```bash
python app.py
```

Then open `http://localhost:5001` in a browser.

### Production (recommended)

Use **Gunicorn** (included in `requirements.txt`):

```bash
gunicorn --workers 4 --bind 0.0.0.0:8000 wsgi:app
```

The app will be available at `http://<server-ip>:8000`.

To run on a different port:
```bash
gunicorn --workers 4 --bind 0.0.0.0:80 wsgi:app
```

---

## Optional: Running behind a reverse proxy (nginx / Apache)

If you put the app behind nginx or Apache, proxy requests to the Gunicorn port
and serve it at your preferred URL. Example nginx snippet:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## Optional: Local plots directory

If the server has a local mirror of the plot files (e.g. an NFS mount), set the
`LOCAL_PLOTS_DIR` environment variable to its root path. The app will check
there first before fetching from the remote server.

```bash
export LOCAL_PLOTS_DIR=/mnt/plots/model_comparisons
gunicorn --workers 4 --bind 0.0.0.0:8000 wsgi:app
```

Expected directory structure under `LOCAL_PLOTS_DIR`:

```
LOCAL_PLOTS_DIR/
└── profiles/
    ├── gliders/
    │   └── YYYY/MM-DD/locations.json
    ├── argo/
    │   └── <region>/last_14_days/locations.json
    └── fvon/
        └── <region>/last_14_days/locations.json
```

Leave `LOCAL_PLOTS_DIR` unset (the default) to always fetch from the remote server.

---

## Files included

| File | Purpose |
|------|---------|
| `app.py` | Flask application |
| `wsgi.py` | Gunicorn entry point |
| `requirements.txt` | Python dependencies |
| `templates/index.html` | Main page template |
| `static/` | CSS, JS, and images |

Test files (`test_*.py`) and `app copy.py` are not needed for deployment and can
be excluded from the zip.

---

## Troubleshooting

**Port already in use:** Change the `--bind` port in the gunicorn command.

**Images not loading:** Confirm the server can reach `rucool.marine.rutgers.edu`
on port 443 (HTTPS).

**`ModuleNotFoundError`:** Make sure the virtual environment is activated before
running gunicorn/python.


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
