# Vendored subset of `ioos_model_comparisons`

**Generated — do not edit files here.** This directory is a copy of the
modules (and YAML config) the web app needs from the top-level
`ioos_model_comparisons/` package, so `webapps/model_comparisons/` can be
zipped and deployed on its own without the plotting stack (cartopy, oceans,
pydap, ...).

Because `webapps/model_comparisons/` is the app's import root, every
`from ioos_model_comparisons.X import Y` in the web app resolves to this copy.

## Keeping it in sync

The file list is `scripts/release/vendor_manifest.txt`. After changing any of
those modules or the config at the repo root:

```bash
python scripts/release/sync_vendor.py          # copy (also: make sync-vendor)
python scripts/release/sync_vendor.py --check  # what CI and the build run
```

`--check` fails if a file here differs from its source, if a file here isn't
in the manifest, or if the web app imports a third-party package that is
missing from `webapps/model_comparisons/requirements.txt` (a missing
dependency otherwise shows up only as a silent 500 in production).
`VENDORED_FROM` records the git commit and file hashes of the last sync.

A new import from the package in the web app means: add the module to the
manifest, re-sync, and add any new third-party requirement.
