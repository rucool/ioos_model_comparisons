import datetime as dt

import pytest

from ioos_model_comparisons import catalog

MAP = ("maps/gulf_of_mexico/temperature_0m/2026/10/"
       "gulf-of-mexico_2026-10-12T060000Z_temperature-0m_rtofs-vs-espc.png")
ARGO = "profiles/argo/gulf_of_mexico/2026/07/30/7902352-profile-2026-07-30T0714Z.png"


def _touch(root, rel, data=b"png"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _coll(client):
    return client["hurricanes"]["figures"]


def test_register_map_figure(fake_mongo, plots_root):
    path = _touch(plots_root, MAP)
    catalog.set_run_context(script="rtofs-gofs-cmems-amseas.py", has_argo=True)
    out = catalog.register_figure(path)
    doc = _coll(fake_mongo).find_one({"_id": MAP})
    assert out is not None and doc is not None
    assert doc["region"] == "gulf_of_mexico" and doc["var_depth"] == "temperature_0m"
    assert doc["comparison"] == "rtofs-vs-espc" and doc["valid_time"] == dt.datetime(2026, 10, 12, 6)
    assert doc["url"] == ("https://rucool.marine.rutgers.edu/hurricane/model_comparisons/" + MAP)
    assert doc["source"] == {"script": "rtofs-gofs-cmems-amseas.py", "via": "hook"}
    assert doc["has_argo"] is True and doc["status"] == "present" and doc["size"] == 3
    assert "rtofs" in doc  # tagged with the RTOFS stream(s) involved


def test_register_is_idempotent_and_keeps_created_at(fake_mongo, plots_root):
    path = _touch(plots_root, MAP)
    catalog.register_figure(path)
    first = _coll(fake_mongo).find_one({"_id": MAP})["created_at"]
    catalog.register_figure(path)
    assert _coll(fake_mongo).count_documents({}) == 1
    assert _coll(fake_mongo).find_one({"_id": MAP})["created_at"] == first


def test_path_without_extension_is_registered_as_png(fake_mongo, plots_root):
    path = _touch(plots_root, MAP)
    catalog.register_figure(str(path)[:-4])  # plot_sst saves without ".png"
    assert _coll(fake_mongo).find_one({"_id": MAP}) is not None


def test_outside_publish_root_is_ignored(fake_mongo, plots_root, tmp_path):
    other = _touch(tmp_path / "elsewhere", MAP)
    assert catalog.register_figure(other) is None
    assert _coll(fake_mongo).count_documents({}) == 0


def test_unrecognized_name_is_ignored(fake_mongo, plots_root):
    path = _touch(plots_root, "maps/atlantic_hurricanes/summary.png")
    assert catalog.register_figure(path) is None
    assert _coll(fake_mongo).count_documents({}) == 0


def test_profile_platform_position(fake_mongo, plots_root):
    path = _touch(plots_root, ARGO)
    catalog.register_figure(path, platform={"lat": 26.5, "lon": 271.5})
    doc = _coll(fake_mongo).find_one({"_id": ARGO})
    assert doc["platform"] == {"type": "argo", "id": "7902352", "lat": 26.5, "lon": -88.5}


def test_mongo_down_returns_none(mongo_down, plots_root):
    assert catalog.register_figure(_touch(plots_root, MAP)) is None


def test_disabled_by_env(fake_mongo, plots_root, monkeypatch):
    monkeypatch.setenv("CATALOG_DISABLED", "1")
    assert catalog.register_figure(_touch(plots_root, MAP)) is None
    assert _coll(fake_mongo).count_documents({}) == 0


def test_write_error_never_raises_and_opens_breaker(fake_mongo, plots_root, monkeypatch):
    calls = []

    class Boom:
        def update_one(self, *a, **k):
            calls.append(1)
            raise RuntimeError("connection reset")

    monkeypatch.setattr(catalog, "_collection", lambda: Boom())
    path = _touch(plots_root, MAP)
    assert catalog.register_figure(path) is None
    assert catalog.register_figure(path) is None  # breaker open: no second attempt
    assert len(calls) == 1


def test_bulk_register_does_not_overwrite_hook_fields(fake_mongo, plots_root):
    path = _touch(plots_root, ARGO)
    catalog.register_figure(path, platform={"lat": 1.0, "lon": 2.0}, script="argo.py")
    n = catalog.register_figures([{"rel": ARGO, "stat": {"size": 99}}])
    doc = _coll(fake_mongo).find_one({"_id": ARGO})
    assert n == 1 and doc["size"] == 99
    assert doc["platform"]["lat"] == 1.0 and doc["source"]["script"] == "argo.py"


def test_read_side(fake_mongo, plots_root):
    for rel in (MAP, MAP.replace("06", "12").replace("rtofs-vs-espc", "rtofs-vs-cmems"),
                MAP.replace("060000Z", "180000Z")):
        catalog.register_figure(_touch(plots_root, rel))
    slot = catalog.find_for_slot("gulf_of_mexico", "temperature_0m", dt.datetime(2026, 10, 12, 6))
    assert set(slot) == {"rtofs-vs-espc"}
    assert catalog.latest_valid_time("gulf_of_mexico", "temperature_0m") == dt.datetime(2026, 10, 12, 18)
    assert catalog.latest_valid_time("gulf_of_mexico", "temperature_0m", ["rtofs-vs-cmems"]) \
        == dt.datetime(2026, 10, 12, 12)
    fresh = catalog.freshness()
    assert fresh["count_24h"] == 3 and not catalog.is_stale(fresh)


def test_missing_entries_are_hidden(fake_mongo, plots_root):
    catalog.register_figure(_touch(plots_root, MAP))
    _coll(fake_mongo).update_one({"_id": MAP}, {"$set": {"status": "missing"}})
    assert catalog.latest_valid_time("gulf_of_mexico", "temperature_0m") is None


def test_read_side_raises_when_unavailable(mongo_down):
    with pytest.raises(RuntimeError):
        catalog.find_for_slot("gulf_of_mexico", "temperature_0m", dt.datetime(2026, 1, 1))
    assert catalog.freshness() is None and catalog.is_stale(None)


def test_indexes(fake_mongo):
    assert catalog.ensure_catalog_indexes()
    names = set(_coll(fake_mongo).index_information())
    assert {"slot", "platform_id_time", "created"} <= names
