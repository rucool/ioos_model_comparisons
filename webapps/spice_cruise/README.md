# SPICE Cruise Model Comparisons — Deployment Guide

A Flask web application that serves an interactive ocean model comparison viewer
scoped to the **Tropical Western Atlantic**, in support of the SPICE Cruise.
It fetches image data from `rucool.marine.rutgers.edu` and requires no local database.

This is a single-region fork of the general IOOS Model Comparisons Archive Explorer —
there is no region selector; every view is locked to Tropical Western Atlantic.

---

## Requirements

- Python 3.10 or newer
- Network access to `rucool.marine.rutgers.edu`

---

## Installation

```bash
cd webapps/spice_cruise
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the app

### Development / quick test

```bash
python app.py
```

Then open `http://localhost:5002` in a browser.

### Production (recommended)

```bash
gunicorn --workers 4 --bind 0.0.0.0:8001 wsgi:app
```

Pick a port that doesn't collide with the main model_comparisons deployment if
both run on the same host.

---

## Optional: Local plots directory

If the server has a local mirror of the plot files (e.g. an NFS mount), set the
`LOCAL_PLOTS_DIR` environment variable to its root path. The app will check
there first before fetching from the remote server.

```bash
export LOCAL_PLOTS_DIR=/mnt/plots/model_comparisons
gunicorn --workers 4 --bind 0.0.0.0:8001 wsgi:app
```

Expected directory structure under `LOCAL_PLOTS_DIR`:

```
LOCAL_PLOTS_DIR/
└── profiles/
    ├── gliders/
    │   └── YYYY/MM-DD/locations.json
    └── argo/
        └── tropical_western_atlantic/last_14_days/locations.json
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

---

## Troubleshooting

**Port already in use:** Change the `--bind` port in the gunicorn command.

**Images not loading:** Confirm the server can reach `rucool.marine.rutgers.edu`
on port 443 (HTTPS).

**`ModuleNotFoundError`:** Make sure the virtual environment is activated before
running gunicorn/python.
