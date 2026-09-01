# Vendored subset of `ioos_model_comparisons`

This directory is **not** a package install — it's a manually copied subset of
the top-level `ioos_model_comparisons/` package from the repo root, placed
here so `webapps/model_comparisons/` can be zipped and deployed on its own,
without installing the rest of the monorepo (which pulls in cartopy, oceans,
pydap, etc. that this webapp never needs).

Because `webapps/model_comparisons/` is already the app's own import root
(see `auth.py`, `fronts_bp.py`, `regions_bp.py` — all imported as bare
names), dropping a real `ioos_model_comparisons/` package folder here makes
every existing `from ioos_model_comparisons.X import Y` line in the webapp
resolve to *this* copy automatically. No import lines were changed to make
this work.

## Files here, and why

| File | Used by |
|---|---|
| `env.py` | `app.py` — optional `.env` loading |
| `db.py` | everything below — the shared Mongo client helper |
| `users.py` | `auth.py` — login/session user store |
| `regions.py` | `region_catalog.py`, `regions_bp.py`, `fronts_bp.py` — region config defaults |
| `regions_store.py` | `regions_bp.py` — Mongo-backed region config overrides |
| `region_catalog.py` | `app.py` — merges `regions.py` + Mongo into `get_region_info()` |
| `fronts/__init__.py`, `fronts/store.py`, `fronts/oisst.py`, `fronts/webmap.py` | `fronts_bp.py` — Gulf Stream front editor |

Deliberately **not** copied: `fronts/digitizer.py`, `fronts/eddies.py`,
`fronts/pipeline.py`, `platforms.py`, `models.py`, `plotting.py`, etc. —
those belong to the offline batch pipeline (`scripts/fronts/*.py`) and are
never imported by the live webapp.

## Keeping this in sync

There is no automation for this — if the source files change at the repo
root, re-copy them here by hand:

```bash
# from the repo root
cp ioos_model_comparisons/env.py                webapps/model_comparisons/ioos_model_comparisons/env.py
cp ioos_model_comparisons/db.py                 webapps/model_comparisons/ioos_model_comparisons/db.py
cp ioos_model_comparisons/users.py              webapps/model_comparisons/ioos_model_comparisons/users.py
cp ioos_model_comparisons/regions.py            webapps/model_comparisons/ioos_model_comparisons/regions.py
cp ioos_model_comparisons/regions_store.py      webapps/model_comparisons/ioos_model_comparisons/regions_store.py
cp ioos_model_comparisons/region_catalog.py     webapps/model_comparisons/ioos_model_comparisons/region_catalog.py
cp ioos_model_comparisons/fronts/__init__.py    webapps/model_comparisons/ioos_model_comparisons/fronts/__init__.py
cp ioos_model_comparisons/fronts/store.py       webapps/model_comparisons/ioos_model_comparisons/fronts/store.py
cp ioos_model_comparisons/fronts/oisst.py       webapps/model_comparisons/ioos_model_comparisons/fronts/oisst.py
cp ioos_model_comparisons/fronts/webmap.py      webapps/model_comparisons/ioos_model_comparisons/fronts/webmap.py
```

Note `regions.py` here is mostly a fallback: MongoDB overrides it at
runtime for anything seeded via `scripts/tools/seed_region_configs.py`, so a
stale copy only matters for *new* regions/fields that haven't been seeded
yet.
