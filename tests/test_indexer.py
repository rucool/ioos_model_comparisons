import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "index_figures", Path(__file__).parents[1] / "scripts" / "tools" / "index_figures.py")
index_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(index_figures)

MAP = ("maps/gulf_of_mexico/temperature_0m/2026/10/"
       "gulf-of-mexico_2026-10-12T060000Z_temperature-0m_rtofs-vs-espc.png")
OLD_MAP = MAP.replace("2026/10", "2024/01").replace("2026-10-12", "2024-01-12")
ARGO = "profiles/argo/gulf_of_mexico/2026/10/11/7902352-profile-2026-10-11T0714Z.png"
GLIDER = "profiles/gliders/2026/10-11/ru29-20260101T0000_20261011_to_20261012_400m.png"


@pytest.fixture
def tree(plots_root):
    for rel in (MAP, OLD_MAP, ARGO, GLIDER, "maps/atlantic_hurricanes/summary.png"):
        p = plots_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    rolling = plots_root / "profiles/argo/gulf_of_mexico/last_14_days"
    rolling.mkdir(parents=True)
    (rolling / Path(ARGO).name).symlink_to(plots_root / ARGO)
    (rolling / "locations.json").write_text(json.dumps(
        {Path(ARGO).name: {"lat": 25.0, "lon": -90.0, "time": "2026-10-11T07:14", "wmo": "7902352"}}))
    (plots_root / "profiles/gliders/2026/10-11/locations.json").write_text(json.dumps(
        {Path(GLIDER).name: {"lat": 39.0, "lon": -73.0, "glider_id": "ru29"}}))
    return plots_root


def _coll(client):
    return client["hurricanes"]["figures"]


def test_walk_skips_symlinks_and_reads_locations(tree):
    found = {rel: is_json for rel, _p, is_json in index_figures.walk(tree)}
    assert MAP in found and found[MAP] is False
    assert "profiles/argo/gulf_of_mexico/last_14_days/locations.json" in found
    assert not any("last_14_days" in r and r.endswith(".png") for r in found)


def test_recent_months_prunes_old_dated_dirs(tree):
    min_month = index_figures._month_index(2026, 9)
    rels = {rel for rel, _p, _j in index_figures.walk(tree, min_month=min_month)}
    assert MAP in rels and ARGO in rels and GLIDER in rels and OLD_MAP not in rels


def test_full_index_and_positions(fake_mongo, tree):
    stats = index_figures.index(tree)
    coll = _coll(fake_mongo)
    assert stats["written"] == 4 and stats["unparsed"] == 1
    assert coll.find_one({"_id": ARGO})["platform"]["lat"] == 25.0
    assert coll.find_one({"_id": GLIDER})["platform"]["lon"] == -73.0
    assert coll.find_one({"_id": MAP})["source"]["via"] == "indexer"
    # Second run: nothing changed on disk, nothing rewritten.
    again = index_figures.index(tree)
    assert again["written"] == 0 and again["unchanged"] == 4


def test_mark_missing(fake_mongo, tree):
    index_figures.index(tree)
    (tree / MAP).unlink()
    assert index_figures.mark_missing(tree) == {"checked": 4, "missing": 1}
    assert _coll(fake_mongo).find_one({"_id": MAP})["status"] == "missing"
    # Re-appearing file is restored on the next index.
    (tree / MAP).write_bytes(b"xy")
    index_figures.index(tree)
    assert _coll(fake_mongo).find_one({"_id": MAP})["status"] == "present"


def test_dry_run_writes_nothing(fake_mongo, tree):
    stats = index_figures.index(tree, dry_run=True)
    assert stats["would_write"] == 4 and _coll(fake_mongo).count_documents({}) == 0


def test_positions_from_rolling_glider_locations():
    pos = index_figures.positions_from_locations(
        "profiles/gliders/last_14_days/locations.json",
        {"ru29-x_20261011_to_20261012_400m.png": {"lat": 1, "lon": 2}})
    assert pos == {"profiles/gliders/2026/10-11/ru29-x_20261011_to_20261012_400m.png": {"lat": 1, "lon": 2}}


def test_retag_rtofs(fake_mongo, tree, monkeypatch):
    index_figures.index(tree)
    _coll(fake_mongo).update_one({"_id": MAP}, {"$unset": {"rtofs": ""}})
    out = index_figures.retag_rtofs(dt.datetime(2026, 1, 1), dt.datetime(2027, 1, 1))
    assert out["retagged"] >= 1 and "rtofs" in _coll(fake_mongo).find_one({"_id": MAP})
