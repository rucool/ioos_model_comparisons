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

## Releasing to baffin

One-time setup:

1. `cp scripts/release/deploy.conf.example scripts/release/deploy.conf` and
   fill in the values from baffin: app root, vassal ini path, Python, health
   URL.
2. On baffin, point the `aquaman` uwsgi vassal at `chdir = <app root>/current`,
   `virtualenv = <app root>/current/venv` and `module = wsgi:app`.
3. Put the app's environment in `<app root>/shared/.env`:
   - `MONGODB_URI`
   - `FLASK_SECRET_KEY`
   - `DISCOVERY_BACKEND`

Each release:

```bash
git tag v2026.10.1 && git push --tags     # optional; untagged builds use the commit id
make deploy                                # check -> build -> upload -> switch -> verify
make rollback                              # previous release, if needed
```

`deploy.sh`:
1. Unpacks the zip into `releases/<version>`.
2. Reuses or creates a venv keyed by the requirements hash.
3. Checks that the app imports.
4. Switches the `current` symlink atomically and touches the vassal ini.
5. Waits for `/healthz` to report the new version, and rolls back
   automatically if it doesn't.

It refuses dirty builds.

## The figure catalog: rollout

| Step | Where | Command / setting |
|---|---|---|
| 1. See what would be indexed | plot host | `python scripts/tools/index_figures.py --survey` |
| 2. Backfill the archive (metadata only, about 1 KB per figure) | plot host | `scripts/shell/index_figures.sh --full` |
| 3. Keep it current | plot host crontab | `*/30` `index_figures.sh`; weekly `index_figures.sh --mark-missing` |
| 4. Plot scripts register as they save | plot host | `git pull` (hooks in `plotting.py` and the profile scripts); install `pyyaml` and `pydantic` in the `hurricanes` env first |
| 5. Compare old and new discovery | baffin `.env` | `DISCOVERY_BACKEND=shadow`, then watch the `discovery.shadow` log for differences |
| 6. Switch | baffin `.env` | `DISCOVERY_BACKEND=auto` (catalog, with automatic fallback to the old discovery when MongoDB is down or the catalog is stale) |

`/healthz` reports the discovery mode, the backend actually in use, and how
fresh the catalog is.
