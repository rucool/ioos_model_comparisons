"""Web app test fixtures. No test reaches the network or a real MongoDB:
`responses` intercepts every requests.* call (unregistered URLs raise), and
MongoDB is either mongomock or "down"."""

import os

import mongomock
import pytest
import responses

from ioos_model_comparisons import catalog, db, region_catalog
from ioos_model_comparisons.registry import reload_registry

BASE = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for var in ("MONGODB_URI", "DISCOVERY_BACKEND", "LOCAL_PLOTS_DIR", "IMC_PLOTS_ROOT",
                "IMC_CONFIG_DIR", "FLASK_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(db, "_client", None)
    monkeypatch.setattr(db, "_client_tried", True)
    monkeypatch.setattr(db, "_client_pid", os.getpid())
    monkeypatch.setattr(catalog, "_breaker_until", 0.0)
    region_catalog._cache.update(ts=0.0, data=None)
    reload_registry()
    import discovery
    discovery._health.update(ts=0.0, ok=False)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


@pytest.fixture
def rsps(_isolate):
    return _isolate


@pytest.fixture
def fake_mongo(monkeypatch):
    client = mongomock.MongoClient()
    monkeypatch.setattr(db, "_client", client)
    return client


@pytest.fixture
def app_module():
    import app
    app.app.config.update(TESTING=True, BACKGROUND_TASKS=False)
    app._asg_latest_cache.update(ts=0.0, date=None)
    app._glider_latest_date_cache.update(ts=0.0, date=None)
    return app


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture
def seed(fake_mongo):
    """Insert catalog entries for relative paths (like the indexer would)."""
    def _seed(*rels):
        catalog.register_figures([{"rel": r, "stat": {"size": 1}} for r in rels])
    return _seed
