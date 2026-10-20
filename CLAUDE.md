# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Rutgers (RUCOOL) ocean-model comparison toolkit. It does three things:

1. **Plot generation** — scripts compare ocean models (RTOFS, ESPC/GOFS, CMEMS, AMSEAS, CNAPS, Doppio, ECCOFS, Lusitania) against each other and against in-situ platforms (gliders, Argo, FVON/FishSOOP, GOES SST) by region and publish PNGs.
2. **Gulf Stream front digitizer** — traces the north wall and detects warm-core rings from SST (`ioos_model_comparisons/fronts/`).
3. **Flask web apps** — browse the plot archive, edit region configs, and edit/view Gulf Stream fronts (`webapps/model_comparisons/`).

The README is a stub; see `docs/` (configuration.md, deployment.md, rtofs_upgrade.md). Tests: `make test` (library, `tests/`) and `make test-portal` (`webapps/model_comparisons/tests/`), both offline (mongomock + `responses`); `make check` runs both plus the vendored-copy check, and GitHub Actions (`.github/workflows/ci.yml`) runs the same. `webapps/model_comparisons/dev_checks/` are ad hoc network scripts, not tests.

## Setup and running

```bash
conda env create -f environment.yml   # env name: model_comps (python 3.12); requirements.txt is only a partial pip list
pip install -e .                      # scripts import the package as `ioos_model_comparisons`
cp .env.example .env                  # MONGODB_URI, FLASK_SECRET_KEY, AUTH_INSECURE_COOKIE=1, CMEMS_USERNAME/PASSWORD
```

- Plot scripts are run directly: `python scripts/maps/models/synchronous/rtofs-gofs-cmems-amseas.py --regions gom mab --espc --cmems` (see each script's docstring for flags such as `--start/--end/--freq`, `--no-overwrite`, `--serial`). On the server they are invoked by cron through `scripts/shell/*.sh` (conda env `hurricanes`, `MPLBACKEND=agg`, `PYPROJ_GLOBAL_CONTEXT=ON`).
- Web app (dev): `cd webapps/model_comparisons && python app.py` (port 5001). Production: uwsgi vassal `aquaman` on host `baffin` (run by a colleague). Releases are zips from `make build`. `scripts/release/install_release.sh` (shipped inside each zip) installs on the server, with `--adopt` for the one-time takeover of the old hand-installed folder, and `--rollback`. `make deploy`/`make rollback` run it over ssh, using the git-ignored `scripts/release/deploy.conf`. This covers the web app only; the plot host updates by `git pull`. Importing `app.py` has no side effects; the NGDAC glider thread starts on the first request.
- Fronts: `scripts/fronts/run_digitizer_goes19.py` (one scene) and `run_digitizer_batch.py` (N days) both call `fronts/pipeline.py:process_scene`; the nightly wrapper is `scripts/shell/gulf_stream_fronts.sh`; `scripts/tools/fronts_preflight.py` and `backfill_fronts_to_mongo.py` support it.

## Architecture

### Shared library: `ioos_model_comparisons/`
- `models.py` — one loader/class per model (`rtofs`, `ESPC`, `CMEMS`, `amseas`, `cnaps`, `Doppio`, `ECCOFS*`, ...). Remote data via OPeNDAP/THREDDS/S3; ECCOFS and Lusitania cache locally.
- `platforms.py` — fetchers for gliders, Argo, FVON, GOES, NESDIS OHC (OHC comes from THREDDS/OpenDAP, not ERDDAP).
- `plotting.py` (~10k lines) — all map/profile plot functions; scripts call into it rather than plotting directly. `region_transform()` handles antimeridian regions (e.g. Fiji, lon > 180) — use it for Pacific regions instead of the default Mercator.
- `configs.py` — platform-dependent paths (`linux` = server paths under `/www/web/...` and `/home/hurricaneadm`, `darwin` = local `~/Documents/plots/model_comparisons`), plus default plot settings. Output location differs per OS.
- `calc.py`, `limits.py`, `common.py` — numerics, colorbar/limit helpers, shared utilities.

### Configuration: YAML in git, MongoDB overrides
`ioos_model_comparisons/config/` holds the configuration, loaded and validated (pydantic, cross-references) by `registry.py:load_registry()`; any bad edit raises `RegistryError`:
- `regions/<key>.yml`: region defaults.
- `models.yml`: models and their filename tokens.
- `comparisons.yml`: comparisons and the portal panels, with region/var_depth rules.
- `rtofs.yml`: RTOFS streams, URLs, harvester settings and version history.
- `portal.yml`: portal lists.
- `site.yml`: plot roots and the published URL.

Adding a region, a portal comparison or an RTOFS version is a YAML edit (see `docs/configuration.md`, `docs/rtofs_upgrade.md`).
- `regions.py:region_config()` is now a thin wrapper over the region YAML. `tests/test_region_parity.py` + `tests/golden/region_config.json` hold it to the legacy function's exact output (tuples for `figsize`, int keys in `limits_by_depth`, key order). The legacy source was exported by `scripts/tools/export_regions_yaml.py`.
- At runtime scripts pass the result through `db.apply_colorbar_overrides()`, which **replaces whole top-level keys** with MongoDB `hurricanes.region_configs`. `regions_store.save()` (the `/regions` editor) stores only keys that differ from the YAML. Old full-document seeds hide YAML edits until `scripts/tools/unseed_region_overrides.py --apply` is run. `seed_region_configs.py` is deprecated (`--force` only).
- `currents.limits_by_depth` has int keys in Python/YAML but string keys in BSON (`db.py` re-ints on read). Validation raises `ValueError`; don't bypass it.
- `colorbar_tuner.py` writes limits into the YAML via `region_yaml.update_limits` (comment-preserving), not into Python source.
- `region_catalog.ALL_REGIONS` derives from the YAML. Note that a region's `ocean_heat_content` is a colorbar flag, not an availability flag.
- `db.get_client()` returns `None` rather than raising when `MONGODB_URI` is missing or unreachable, and is fork-safe (it reconnects in `ProcessPoolExecutor` children). Auth (`users.py`) treats `None` as deny.
- A shell-exported `MONGODB_URI` silently overrides `.env` (`env.py` uses `override=False` intentionally). CMEMS credentials come only from `CMEMS_USERNAME`/`CMEMS_PASSWORD` (`env.cmems_credentials()`); never hardcode them.

### Figure catalog: `catalog.py`, `figure_names.py`
MongoDB `hurricanes.figures` holds one small metadata document per published PNG, keyed by its path relative to the plots root; the images stay on the file server.
- **Filename conventions:** `figure_names.parse()`/`build()` are the single source of truth, round-trip tested against real paths in `tests/fixtures/sample_paths.txt`.
- **Writes:**
  - `plotting.py` calls `_register()` after every `savefig` (including `export_fig`), and the profile scripts call `catalog.register_figure(path, platform={lat, lon})`. It never raises: there is a circuit breaker, it is a no-op outside `site.yml publish_root` or `$IMC_PLOTS_ROOT`, and `CATALOG_DISABLED=1` turns it off.
  - `scripts/tools/index_figures.py` (cron `scripts/shell/index_figures.sh`) backfills, catches missed figures, merges `locations.json` positions and marks deleted files `status: missing`.
- **`model_plots` (`db.log_plots` / `needs_replot`)** is still the replot ledger for the map scripts; it is separate from the catalog.

### Fronts: `ioos_model_comparisons/fronts/`
`digitizer.py` (gradient anchors → calibrated isotherm trace), `eddies.py` (ring detection from SLA, run *before* the wall trace so rings can influence it), `oisst.py`, `webmap.py`, `store.py` (MongoDB, highest version = current), `pipeline.py` (per-scene sequence shared by both runners; files are written before MongoDB so a DB outage never loses the artifact). Cartopy `set_extent()` on Mercator axes needs an explicit `crs=` or it overshoots.

### Web app: `webapps/model_comparisons/`
Flask app (`app.py`) with blueprints:
- `fronts_bp`: public read, editor-only write (401/403).
- `regions_bp`: works on the effective doc, YAML + overrides, and reports which keys are overridden.
- `ohc_bp`.

Login is `auth.py` + `ioos_model_comparisons/users.py`.

**Figure discovery goes through `discovery.py`**, selected by `DISCOVERY_BACKEND`:
- `legacy` (default): predict URLs from config, HEAD them, scrape Apache listings.
- `shadow`: legacy answers; diffs against the catalog go to the `discovery.shadow` log.
- `auto`: catalog, falling back to legacy when MongoDB is down or the catalog is stale.
- `catalog`.

Response shapes are identical across backends. Panel rules come from `comparisons.yml`, injected into the page as `PANEL_DEFS`/`MAP_PANELS`; `app.js` has no hardcoded model or region rules. `/healthz` reports version, MongoDB, catalog freshness and config hash. Templates share `base.html`; pages that need the viewport-locked layout opt in with `body.app-shell` (others must scroll).

**The web app ships as a standalone zip** with a generated copy of part of the library in `webapps/model_comparisons/ioos_model_comparisons/` (the modules plus `config/`).
- Never edit that copy. Edit the top-level module and run `make sync-vendor`; the file list is `scripts/release/vendor_manifest.txt`.
- `make check-vendor` (CI and the build) fails on drift, or on a third-party import missing from `webapps/model_comparisons/requirements.txt`. A missing dependency otherwise shows up only as a silent 500 with no traceback.
- The build zips only files git knows about, so commit new web app files before building.

### Scripts: `scripts/`
Grouped by product: `maps/models/{synchronous,asynchronous}`, `profiles/{argo,gliders,fvon}`, `redwing/` (mission-specific glider maps), `rtofs/` (RTOFS binary → z-level conversion), `harvest/` (data grabbers), `tools/` (seeding, colorbar tuning, PowerPoint/overview decks), `shell/` (cron wrappers). Many scripts are near-duplicates per region/basin (e.g. `_pacific`, `_guam`, `_fiji`); check for sibling copies when changing shared behavior. Per-region plotting jobs use `ProcessPoolExecutor` unless `--serial`.

## Repo notes
- `scratches/`, `notebooks/`, `outputs/`, `cache*/`, `accidentally_deleted/`, `deprecated/`, `*.png`, `*.nc`, `*.zip` are gitignored or scratch — ignore them when searching.
