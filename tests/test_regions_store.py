import importlib.util
from pathlib import Path

import pytest

from ioos_model_comparisons import db, regions_store as rs
from ioos_model_comparisons.regions import region_config


def _live(client):
    return client["hurricanes"]["region_configs"]


def test_save_stores_only_differences(fake_mongo):
    doc = dict(region_config("gom"))
    doc["extent"] = [-98, -80, 18, 31]
    assert rs.save("gom", doc, edited_by="t") is not None
    stored = _live(fake_mongo).find_one({"region": "gom"}, {"_id": 0})
    assert stored == {"region": "gom", "extent": [-98, -80, 18, 31]}
    # Plotting sees the override plus YAML for everything else.
    eff = db.apply_colorbar_overrides("gom", region_config("gom"))
    assert eff["extent"] == [-98, -80, 18, 31] and eff["name"] == "Gulf of Mexico"


def test_partial_doc_is_merged_and_validated(fake_mongo):
    with pytest.raises(ValueError, match="lat_min"):
        rs.save("gom", {"extent": [-98, -80, 40, 30]}, edited_by="t")
    rs.save("gom", {"eez": True}, edited_by="t")
    assert _live(fake_mongo).find_one({"region": "gom"}, {"_id": 0}) == {"region": "gom", "eez": True}


def test_history_keeps_full_documents(fake_mongo):
    rs.save("gom", {"eez": True}, edited_by="t")
    v = rs.fetch_version("gom", 1)
    assert v["doc"]["name"] == "Gulf of Mexico" and v["doc"]["eez"] is True
    assert v["changed"] == {"eez": [False, True]}


def test_fetch_effective_and_listing(fake_mongo):
    rs.save("gom", {"eez": True}, edited_by="t")
    doc, overridden = rs.fetch_effective("gom")
    assert overridden == ["eez"] and doc["folder"] == "gulf_of_mexico"
    doc, overridden = rs.fetch_effective("sargasso")  # never saved: YAML only
    assert overridden == [] and doc["name"] == "Sargasso Sea"
    assert rs.fetch_effective("atlantis") is None
    keys = [r["region"] for r in rs.list_regions()]
    assert "gom" in keys and "sargasso" in keys


def test_unseed_removes_copies_of_defaults(fake_mongo, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "unseed", Path(__file__).parents[1] / "scripts/tools/unseed_region_overrides.py")
    unseed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unseed)

    seeded = rs.normalize(dict(region_config("gom")))
    seeded["region"] = "gom"
    seeded["variables"]["temperature"][0]["limits"] = [25, 31, 0.5]  # a tuned override
    _live(fake_mongo).insert_one(seeded)

    unseed.main([])  # dry run
    assert set(_live(fake_mongo).find_one({"region": "gom"})) > {"name", "folder"}

    unseed.main(["--apply", "--backup-dir", str(tmp_path)])
    stored = _live(fake_mongo).find_one({"region": "gom"}, {"_id": 0})
    assert set(stored) == {"region", "variables"}
    assert len(list(tmp_path.glob("region_configs_backup_*.json"))) == 1
