# Testing, building and deploying the portal

## Tests

```bash
pip install -r requirements-dev.txt
make test          # library: config registry, region parity, figure catalog, indexer
make test-portal   # web app: API contracts on both backends, legacy URL parity, health check
make check         # both, plus the vendored-copy / requirements check
```

No test reaches the network or a real MongoDB. GitHub Actions
(`.github/workflows/ci.yml`) runs the same on every push and pull request,
and builds the zip. On `v*` tags it attaches the zip to the GitHub release.

## Releasing the portal to baffin

This covers the web app only: the Flask portal in `webapps/model_comparisons/`.
The plot scripts, cron jobs and figure indexer run from the git checkout on
the plot host; see "Updating the plot host" below.

A release is a zip built from a commit: `make build` locally, or the CI
artifact. Every zip contains `install_release.sh`, which does all the
server-side work on baffin:
1. Unpacks the zip into `releases/<version>`.
2. Reuses or builds a package environment (only when `requirements.txt` changed).
3. Checks that the app imports.
4. Atomically switches the `current` link and touches the uwsgi vassal ini.
5. Waits for `/healthz` to report the new version, and rolls back automatically if it doesn't.

It refuses dirty builds. `--rollback` returns to the previous release, and
`--status` shows what's live.

### Who runs it

- **The server admin on baffin**, with no ssh from anyone else:
  ```bash
  unzip -o -j model_comparisons-<version>.zip '*/install_release.sh' -d /tmp
  bash /tmp/install_release.sh --app-root <APP_ROOT> model_comparisons-<version>.zip
  ```
  Put `VASSAL_INI`, `PYTHON`, `HEALTH_URL` and `KEEP` in `<APP_ROOT>/install.conf`
  once, so only `--app-root` is needed.
- **Or from a laptop with ssh access:** `make deploy` / `make rollback` copy the
  zip and the installer over and run the same installer remotely. The
  settings go in `scripts/release/deploy.conf` (copy `deploy.conf.example`).

### First time: taking over the existing hand-installed deployment

The current site runs from a hand-unzipped folder that the `aquaman` vassal
points at directly. Switch over without downtime:

1. `install_release.sh --app-root <APP_ROOT> --adopt <current app folder>`
   - registers the running app as release `model_comparisons-adopted`;
   - copies its `.env` to `<APP_ROOT>/shared/.env`;
   - changes nothing that's serving.
2. Edit the vassal ini, then touch it. The same old code is now served through `current`; check the site.
   - `chdir = <APP_ROOT>/current`
   - `virtualenv = <APP_ROOT>/current/venv`
   - `module = wsgi:app`
3. Add `DISCOVERY_BACKEND=legacy` (the default) to `shared/.env` for now.
4. Install the new release zip. If anything is wrong, `--rollback` returns to the adopted app.

The new release finds figures exactly the old way until `DISCOVERY_BACKEND`
is changed (see the rollout below), so this first deploy changes how the
code is managed, not what the site shows.

## The plot host

The plot host is the Linux server where the plotting cron jobs run as
`hurricaneadm` and write PNGs into `/www/web/rucool/hurricane/model_comparisons/`.
The toolbox checkout is `/home/hurricaneadm/scripts/temp/ioos_model_comparisons`.
Note that some older cron wrappers still point at
`~/scripts/ioos_model_comparisons` or `~/scripts/hurricanes`.

The figure indexer can run from its own clone, e.g. `~/scripts/figure_catalog`;
`index_figures.sh` uses its own checkout's code. That way the catalog doesn't
require updating the toolbox. To update the toolbox itself:
1. `git pull` there.
2. Install `pyyaml` and `pydantic` once into its conda env, `model_comps`.
   Some older wrappers still activate `hurricanes`.
3. Set up CMEMS credentials first: `copernicusmarine login`, or
   `CMEMS_USERNAME`/`CMEMS_PASSWORD`. The code no longer contains them.

## The figure catalog: rollout

| Step | Where | Command / setting |
|---|---|---|
| 1. See what would be indexed | plot host | `python scripts/tools/index_figures.py --survey` |
| 2. Backfill the archive (metadata only, about 1 KB per figure) | plot host | `scripts/shell/index_figures.sh --full` |
| 3. Keep it current | plot host crontab | `*/30` `index_figures.sh`; weekly `index_figures.sh --mark-missing` |
| 4. Optional: plot scripts register as they save | plot host | update the toolbox checkout (see above); until then the 30-minute indexer catches everything |
| 5. Compare old and new discovery | baffin `.env` | `DISCOVERY_BACKEND=shadow`, then watch the `discovery.shadow` log for differences |
| 6. Switch | baffin `.env` | `DISCOVERY_BACKEND=auto` (catalog, with automatic fallback to the old discovery when MongoDB is down or the catalog is stale) |

`/healthz` reports the discovery mode, the backend actually in use, and how
fresh the catalog is.
