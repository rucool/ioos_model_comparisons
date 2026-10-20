"""Shared fixtures. Nothing here touches the network or a real MongoDB."""

import mongomock
import pytest

from ioos_model_comparisons import catalog, db
from ioos_model_comparisons.registry import reload_registry


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No test sees a developer's MONGODB_URI, .env or catalog state."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("IMC_CONFIG_DIR", raising=False)
    monkeypatch.delenv("IMC_PLOTS_ROOT", raising=False)
    monkeypatch.delenv("CATALOG_DISABLED", raising=False)
    monkeypatch.setattr(catalog, "_breaker_until", 0.0)
    monkeypatch.setattr(catalog, "_ctx", {})
    reload_registry()
    yield
    reload_registry()


@pytest.fixture
def fake_mongo(monkeypatch):
    """db.get_client() returns an in-memory mongomock client."""
    import os
    client = mongomock.MongoClient()
    monkeypatch.setattr(db, "_client", client)
    monkeypatch.setattr(db, "_client_tried", True)
    monkeypatch.setattr(db, "_client_pid", os.getpid())
    return client


@pytest.fixture
def mongo_down(monkeypatch):
    """db.get_client() returns None (MONGODB_URI unset / unreachable)."""
    import os
    monkeypatch.setattr(db, "_client", None)
    monkeypatch.setattr(db, "_client_tried", True)
    monkeypatch.setattr(db, "_client_pid", os.getpid())


@pytest.fixture
def plots_root(tmp_path, monkeypatch):
    """A temporary published plots root (catalog only registers under it)."""
    root = tmp_path / "plots"
    root.mkdir()
    monkeypatch.setenv("IMC_PLOTS_ROOT", str(root))
    return root
