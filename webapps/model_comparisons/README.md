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
