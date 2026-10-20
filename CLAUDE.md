# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rutgers (RUCOOL) ocean-model comparison toolkit. It does three things:

1. **Plot generation** — scripts compare ocean models (RTOFS, ESPC/GOFS, CMEMS, AMSEAS, CNAPS, Doppio, ECCOFS, Lusitania) against each other and against in-situ platforms (gliders, Argo, FVON/FishSOOP, GOES SST) by region and publish PNGs.
2. **Gulf Stream front digitizer** — traces the north wall and detects warm-core rings from SST (`ioos_model_comparisons/fronts/`).
3. **Flask web apps** — browse the plot archive, edit region configs, and edit/view Gulf Stream fronts (`webapps/model_comparisons/`).

The README is a stub. There is no test suite, linter config, or build step; `pytest` is not set up. `test_update.py` (repo root) and `webapps/model_comparisons/test_*.py` are ad hoc scripts, not a suite.

## Setup and running

```bash
conda env create -f environment.yml   # env name: model_comps (python 3.12); requirements.txt is only a partial pip list
pip install -e .                      # scripts import the package as `ioos_model_comparisons`
cp .env.example .env                  # MONGODB_URI, FLASK_SECRET_KEY, AUTH_INSECURE_COOKIE=1, CMEMS_USERNAME/PASSWORD
```

- Plot scripts are run directly: `python scripts/maps/models/synchronous/rtofs-gofs-cmems-amseas.py --regions gom mab --espc --cmems` (see each script's docstring for flags such as `--start/--end/--freq`, `--no-overwrite`, `--serial`). On the server they are invoked by cron through `scripts/shell/*.sh` (conda env `hurricanes`, `MPLBACKEND=agg`, `PYPROJ_GLOBAL_CONTEXT=ON`).
- Web app (dev): `cd webapps/model_comparisons && python app.py` (port 5001). Production: `gunicorn --workers 4 --bind 0.0.0.0:8000 wsgi:app` (served by uwsgi on host `baffin`). See `webapps/model_comparisons/README.md`.
- Fronts: `scripts/fronts/run_digitizer_goes19.py` (one scene) and `run_digitizer_batch.py` (N days) both call `fronts/pipeline.py:process_scene`; the nightly wrapper is `scripts/shell/gulf_stream_fronts.sh`; `scripts/tools/fronts_preflight.py` and `backfill_fronts_to_mongo.py` support it.

## Architecture

### Shared library: `ioos_model_comparisons/`
- `models.py` — one loader/class per model (`rtofs`, `ESPC`, `CMEMS`, `amseas`, `cnaps`, `Doppio`, `ECCOFS*`, ...). Remote data via OPeNDAP/THREDDS/S3; ECCOFS and Lusitania cache locally.
- `platforms.py` — fetchers for gliders, Argo, FVON, GOES, NESDIS OHC (OHC comes from THREDDS/OpenDAP, not ERDDAP).
- `plotting.py` (~10k lines) — all map/profile plot functions; scripts call into it rather than plotting directly. `region_transform()` handles antimeridian regions (e.g. Fiji, lon > 180) — use it for Pacific regions instead of the default Mercator.
- `configs.py` — platform-dependent paths (`linux` = server paths under `/www/web/...` and `/home/hurricaneadm`, `darwin` = local `~/Documents/plots/model_comparisons`), plus default plot settings. Output location differs per OS.
- `calc.py`, `limits.py`, `common.py` — numerics, colorbar/limit helpers, shared utilities.

### Region config: two sources, MongoDB wins
`regions.py:region_config(regions, model)` holds the static per-region definition (extent, folder, per-variable/depth colorbar limits, currents, EEZ, OHC flags). At runtime scripts pass the result through `db.apply_colorbar_overrides()`, which **replaces whole top-level keys** with the document in MongoDB `hurricanes.region_configs`. Consequences:
- Editing `regions.py` has no effect on scripts that use MongoDB until `python scripts/tools/seed_region_configs.py` is re-run. The seed deliberately skips `variables` and `sea_surface_height` (owned by `update_colorbar_limits.py` weekly tuning and `colorbar_tuner.py`), and uses a partial `$set`.
- `regions_store.py` is the validated CRUD + append-only `region_config_versions` history used by the `/regions` editor. `currents.limits_by_depth` has int keys in Python but must be string keys in BSON (`db.py` re-ints on read). Validation raises `ValueError`; don't bypass it.
- `region_catalog.py` derives the web app's region list live from `regions.py` + MongoDB (no hand-typed copy). Note regions.py's `ocean_heat_content` is a colorbar flag, not an availability flag.
- `db.get_client()` returns `None` rather than raising when `MONGODB_URI` is missing/unreachable; plotting degrades to `regions.py` defaults. Auth (`users.py`) does the opposite and treats `None` as deny. Also logs generated plots (`model_plots` collection, used for `--no-overwrite` / `needs_replot`) and OHC metrics.
- A shell-exported `MONGODB_URI` silently overrides `.env` (`env.py` uses `override=False` intentionally).

### Fronts: `ioos_model_comparisons/fronts/`
`digitizer.py` (gradient anchors → calibrated isotherm trace), `eddies.py` (ring detection from SLA, run *before* the wall trace so rings can influence it), `oisst.py`, `webmap.py`, `store.py` (MongoDB, highest version = current), `pipeline.py` (per-scene sequence shared by both runners; files are written before MongoDB so a DB outage never loses the artifact). Cartopy `set_extent()` on Mercator axes needs an explicit `crs=` or it overshoots.

### Web app: `webapps/model_comparisons/`
Flask app (`app.py`) with blueprints `fronts_bp` (public read, editor-only write → 401/403), `regions_bp`, `ohc_bp`; `auth.py` + `ioos_model_comparisons/users.py` for login (Flask-Login, MongoDB users; manage with `scripts/tools/manage_front_users.py`). `/api/*` routes proxy plot images and `locations.json` from `rucool.marine.rutgers.edu`, or from `LOCAL_PLOTS_DIR` first if set. Templates share `base.html`; pages that need the viewport-locked layout opt in with `body.app-shell` (others must scroll).

**The web app ships as a standalone zip** and carries its own vendored subset of the library in `webapps/model_comparisons/ioos_model_comparisons/` (db, env, region_catalog, regions, regions_store, users, fronts/). Changes to those modules in the top-level package must be mirrored there, and any new import (e.g. `shapely`) must be added to `webapps/model_comparisons/requirements.txt` — a missing dependency causes a silent 500 with no traceback in the error log.

### Scripts: `scripts/`
Grouped by product: `maps/models/{synchronous,asynchronous}`, `profiles/{argo,gliders,fvon}`, `redwing/` (mission-specific glider maps), `rtofs/` (RTOFS binary → z-level conversion), `harvest/` (data grabbers), `tools/` (seeding, colorbar tuning, PowerPoint/overview decks), `shell/` (cron wrappers). Many scripts are near-duplicates per region/basin (e.g. `_pacific`, `_guam`, `_fiji`); check for sibling copies when changing shared behavior. Per-region plotting jobs use `ProcessPoolExecutor` unless `--serial`.

## Repo notes
- `scratches/`, `notebooks/`, `outputs/`, `cache*/`, `accidentally_deleted/`, `deprecated/`, `*.png`, `*.nc`, `*.zip` are gitignored or scratch — ignore them when searching.
- `regions_backup.py` and `scripts/tools/update_colorbar_limits copy.py` are stale copies.
