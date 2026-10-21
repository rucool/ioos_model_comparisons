import datetime as dt
import shutil

import pytest
import yaml

from ioos_model_comparisons.registry import CONFIG_DIR, RegistryError, Registry, load_registry

# Python port of the panel rules that were hardcoded in static/js/app.js
# (updateMapModelsConfig) before they moved to comparisons.yml, with the one
# deliberate change since: ESPC-vs-CMEMS removed from Guam/Fiji (2026-10-21),
# Fiji OHC keeping its combined three-model image as `three_model`.
_LEGACY_ECCOFS_EXCLUDED = {"Tropical Western Atlantic", "Eastern Pacific - Mexico", "Hawaii",
                           "WMO V - South Pacific", "Philippines Sea", "Guam", "Fiji"}


def legacy_panels(region_name, vd):
    if region_name == "Fiji" and vd == "ocean_heat_content":
        return ["three_model"]
    out = ["copernicus", "espc"]
    if region_name not in _LEGACY_ECCOFS_EXCLUDED:
        out.append("eccofs")
    if vd == "ocean_heat_content":
        out.append("nesdis")
    if vd == "temperature_0m" and region_name != "Fiji":
        out += ["goes16", "goes19"]
    return out


VAR_DEPTHS = ["temperature_0m", "temperature_150m", "salinity_0m", "salinity_200m",
              "currents_0m", "currents_150m", "ocean_heat_content"]


def test_shipped_config_loads():
    reg = load_registry()
    assert "gom" in reg.region_keys() and "sargasso" in reg.region_keys()
    assert [p.key for p in reg.panels] == ["copernicus", "espc", "three_model", "eccofs",
                                          "nesdis", "goes16", "goes19"]


@pytest.mark.parametrize("vd", VAR_DEPTHS)
def test_panels_reproduce_legacy_js_rules(vd):
    reg = load_registry()
    for key in reg.portal.map_regions:
        d = reg.region_defaults(key)
        got = [p.key for p in reg.map_panels(d["folder"], vd)]
        assert got == legacy_panels(d["name"], vd), (d["name"], vd)


def test_comparison_rules():
    reg = load_registry()
    assert not reg.comparison_applies("rtofs-vs-eccofs", "hawaii")
    assert reg.comparison_applies("rtofs-vs-eccofs", "gulf_of_mexico")
    assert not reg.comparison_applies("rtofs-vs-goes16", "gulf_of_mexico", "salinity_0m")
    assert not reg.comparison_applies("nope", "gulf_of_mexico")


def test_boxes_resolve_extent_from_and_labels():
    reg = load_registry()
    boxes = {b["key"]: b for b in reg.argo_boxes()}
    assert boxes["mexico_pacific"] == {"key": "mexico_pacific", "label": "Eastern Pacific - Mexico",
                                       "extent": [-126, -92, 10.1, 33]}
    assert boxes["south_atlantic_bight"]["extent"] == [-82.25, -63.75, 24.75, 42.25]


def test_rtofs_version_history(tmp_path):
    cfg = _copy_config(tmp_path)
    doc = yaml.safe_load((cfg / "rtofs.yml").read_text())
    doc["history"] = [
        {"stream": "operational", "version": "v2.4", "from": None},
        {"stream": "operational", "version": "v2.5", "from": "2026-06-01"},
        {"stream": "parallel", "version": "v2.6", "from": "2026-09-01"},
    ]
    (cfg / "rtofs.yml").write_text(yaml.safe_dump(doc))
    reg = Registry(cfg)
    assert reg.rtofs_version_at("operational", dt.date(2026, 5, 31)) == "v2.4"
    assert reg.rtofs_version_at("operational", dt.datetime(2026, 6, 1, 0)) == "v2.5"
    assert reg.rtofs_version_at("parallel", dt.date(2026, 8, 1)) is None
    assert reg.rtofs_tag_for_models(["rtofs", "rtofs-p"], dt.date(2026, 9, 2)) == \
        {"operational": "v2.5", "parallel": "v2.6"}
    assert reg.rtofs_tag_for_models(["espc", "cmems"]) is None


def _copy_config(tmp_path):
    dst = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, dst)
    return dst


def _edit(path, fn):
    doc = yaml.safe_load(path.read_text())
    fn(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


@pytest.mark.parametrize("filename,mutate,expect", [
    ("comparisons.yml", lambda d: d["comparisons"]["rtofs-vs-espc"].update(models=["rtofs", "espcc"]),
     "unknown model 'espcc'"),
    ("comparisons.yml", lambda d: d["map_panels"][0].update(comparisons=["rtofs-vs-nothing"]),
     "unknown comparison"),
    ("comparisons.yml", lambda d: d["comparisons"]["rtofs-vs-eccofs"]["regions"].update(exclude=["hawai"]),
     "unknown region folder 'hawai'"),
    ("comparisons.yml", lambda d: d.update(overview_order=["espc"]), "overview_order"),
    ("models.yml", lambda d: d["models"]["espc"].update(lable="x"), "Extra inputs"),
    ("models.yml", lambda d: d["models"]["espc"].update(filename_token="rtofs"), "already used"),
    ("portal.yml", lambda d: d["map_regions"].append("atlantis"), "unknown region 'atlantis'"),
    ("portal.yml", lambda d: d["argo_regions"]["gulf_of_mexico"].update(extent=[1, 2, 3, 4]),
     "exactly one of extent"),
    ("rtofs.yml", lambda d: d["history"].append({"stream": "beta", "version": "x"}), "unknown stream"),
    ("regions/gom.yml", lambda d: d["extent"].__setitem__(2, 50), "lat_min"),
    ("regions/gom.yml", lambda d: d["variables"]["temperature"][0].update(limits=[30, 20, 1]), "limits"),
    ("regions/gom.yml", lambda d: d.pop("folder"), "folder: required"),
])
def test_bad_config_is_rejected(tmp_path, filename, mutate, expect):
    cfg = _copy_config(tmp_path)
    _edit(cfg / filename, mutate)
    with pytest.raises(RegistryError, match=expect):
        Registry(cfg)


def test_yaml_syntax_error_is_reported(tmp_path):
    cfg = _copy_config(tmp_path)
    (cfg / "models.yml").write_text("models: [unclosed\n")
    with pytest.raises(RegistryError, match="models.yml: YAML syntax error"):
        Registry(cfg)


def test_new_region_is_config_only(tmp_path):
    cfg = _copy_config(tmp_path)
    (cfg / "regions" / "atlantis.yml").write_text(yaml.safe_dump({
        "order": 99, "name": "Atlantis", "folder": "atlantis", "extent": [-30, -20, 30, 40],
        "figure": {"figsize": [12, 8], "legend": {"columns": 5}},
        "sea_surface_height": [], "eez": False,
        "variables": {"salinity": [{"depth": 0, "limits": [35, 37, 0.1]}],
                      "temperature": [{"depth": 0, "limits": [15, 25, 0.5]}]},
    }, sort_keys=False))
    _edit(cfg / "portal.yml", lambda d: d["map_regions"].append("atlantis"))
    reg = Registry(cfg)
    assert reg.region_keys()[-1] == "atlantis"
    assert [p.key for p in reg.map_panels("atlantis", "temperature_0m")] == \
        ["copernicus", "espc", "eccofs", "goes16", "goes19"]
