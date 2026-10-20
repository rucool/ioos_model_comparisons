import shutil

import pytest
import yaml

from ioos_model_comparisons.region_yaml import update_limits
from ioos_model_comparisons.registry import CONFIG_DIR, Registry


@pytest.fixture
def cfg(tmp_path):
    dst = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, dst)
    return dst


def _load(cfg, key):
    return yaml.safe_load((cfg / "regions" / f"{key}.yml").read_text())


def test_updates_variable_limits_and_keeps_comments(cfg):
    before = (cfg / "regions" / "mexico_pacific.yml").read_text()
    assert "RTOFS west dataset" in before
    msgs = update_limits({("mexico_pacific", "temperature", 50): [16, 31.5, 0.5]}, cfg)
    after = (cfg / "regions" / "mexico_pacific.yml").read_text()
    assert msgs == ["mexico_pacific.temperature[50]: updated"]
    assert "RTOFS west dataset" in after and "CICESE" in after
    t50 = [d for d in _load(cfg, "mexico_pacific")["variables"]["temperature"] if d["depth"] == 50][0]
    assert t50 == {"depth": 50, "limits": [16, 31.5, 0.5], "pinned": True}
    # Only that one line changed.
    diff = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(diff) == 1


def test_currents_limits_by_depth_added_then_updated(cfg):
    update_limits({("gom", "currents", 150): [0, 1.2, 0.1]}, cfg)
    assert _load(cfg, "gom")["currents"]["limits_by_depth"] == {150: [0, 1.2, 0.1]}
    update_limits({("gom", "currents", 0): [0, 2, 0.1]}, cfg)
    assert _load(cfg, "gom")["currents"]["limits_by_depth"] == {150: [0, 1.2, 0.1], 0: [0, 2, 0.1]}
    update_limits({("gom", "currents", 150): [0, 1.5, 0.1]}, cfg)
    assert _load(cfg, "gom")["currents"]["limits_by_depth"][150] == [0, 1.5, 0.1]
    Registry(cfg)  # still valid config


def test_block_style_limits_by_depth_append(cfg):
    p = cfg / "regions" / "gom.yml"
    p.write_text(p.read_text().replace(
        "currents:\n  bool: true\n",
        "currents:\n  bool: true\n  limits_by_depth:\n    0: [0, 1, 0.1]\n"))
    update_limits({("gom", "currents", 200): [0, 0.8, 0.1]}, cfg)
    assert _load(cfg, "gom")["currents"]["limits_by_depth"] == {0: [0, 1, 0.1], 200: [0, 0.8, 0.1]}


def test_sea_surface_height_and_missing_depth(cfg):
    msgs = update_limits({("mexico_pacific", "sea_surface_height", 0): [-0.5, 0.5, 0.1],
                          ("mexico_pacific", "salinity", 999): [30, 35, 0.1]}, cfg)
    assert "updated" in msgs[0] and "skipped" in msgs[1]
    assert _load(cfg, "mexico_pacific")["sea_surface_height"][0]["limits"] == [-0.5, 0.5, 0.1]
