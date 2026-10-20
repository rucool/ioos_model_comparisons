# Common tasks. Run from the repo root.
#   make test          library tests (no network, no MongoDB needed)
#   make test-portal   web app tests (vendored package, mocked HTTP/Mongo)
#   make check         everything CI runs
#   make build         dist/model_comparisons-<version>.zip (clean tree required)
#   make deploy        build + deploy to baffin (scripts/release/deploy.conf)
#   make rollback      return baffin to the previous release

PYTHON ?= python

.PHONY: test test-portal check sync-vendor check-vendor build deploy rollback survey

test:
	$(PYTHON) -m pytest

test-portal:
	cd webapps/model_comparisons && $(PYTHON) -m pytest

sync-vendor:
	$(PYTHON) scripts/release/sync_vendor.py

check-vendor:
	$(PYTHON) scripts/release/sync_vendor.py --check

check: check-vendor test test-portal

build: check
	$(PYTHON) scripts/release/build_webapp.py

deploy: check
	scripts/release/deploy.sh

rollback:
	scripts/release/rollback.sh

# What the figure indexer would catalog under the plots root (no writes).
survey:
	$(PYTHON) scripts/tools/index_figures.py --survey
