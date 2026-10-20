"""The portal no longer hardcodes image URL patterns: in legacy mode it
predicts them from comparisons.yml + models.yml (registry.predict_rel). This
holds that prediction to the old hand-written build_map_urls() — copied
verbatim below — for every map region, variable and depth, so turning the
rules into config didn't change a single URL (except the Windward Islands
folder, which the old code got wrong)."""

import datetime as dt

import pytest

BASE = "https://rucool.marine.rutgers.edu/hurricane/model_comparisons/maps/"
_OLD_FOLDER_MAP = {"Eastern Pacific - Mexico": "mexico_pacific"}
_OLD_ECCOFS_EXCLUDED = {"Tropical Western Atlantic", "Eastern Pacific - Mexico", "Hawaii",
                        "WMO V - South Pacific", "Philippines Sea", "Guam", "Fiji"}


def old_build_map_urls(region, variable_depth, date_obj, time_str):
    variable_depth_mod = variable_depth.replace("_", "-")
    year, month, day = date_obj.year, f"{date_obj.month:02d}", f"{date_obj.day:02d}"
    time_code = f"{time_str.replace('Z', '').zfill(2)}0000"
    region_url_key = _OLD_FOLDER_MAP.get(region, region.replace(" ", "_").lower())
    region_file_slug = region_url_key if variable_depth.startswith("currents") else region_url_key.replace("_", "-")
    base_url = f"{BASE}{region_url_key}/{variable_depth}/{year}/{month}/"
    out = dict.fromkeys(["copernicus", "espc", "espc_cmems", "goes16", "goes19", "eccofs", "nesdis"])
    eccofs_ok = region not in _OLD_ECCOFS_EXCLUDED
    if variable_depth == "ocean_heat_content":
        if region == "Fiji":
            out["espc_cmems"] = f"{base_url}{region}_{year}-{month}-{day}T{time_code}Z_ohc_rtofs-espc-cmems.png"
        else:
            f = f"{region_url_key}_{year}-{month}-{day}T{time_code}Z"
            out["copernicus"] = f"{base_url}{f}_heat_content_rtofs-cmems.png"
            out["espc"] = f"{base_url}{f}_heat_content_rtofs-espc.png"
            out["espc_cmems"] = f"{base_url}{f}_heat_content_espc-cmems.png"
            if eccofs_ok:
                out["eccofs"] = f"{base_url}{f}_heat_content_rtofs-eccofs.png"
            out["nesdis"] = f"{base_url}{f}_heat_content_rtofs-nesdis.png"
    else:
        f = f"{region_file_slug}_{year}-{month}-{day}T{time_code}Z_{variable_depth_mod}"
        out["copernicus"] = f"{base_url}{f}_rtofs-vs-cmems.png"
        out["espc"] = f"{base_url}{f}_rtofs-vs-espc.png"
        out["espc_cmems"] = f"{base_url}{f}_espc-vs-cmems.png"
        if eccofs_ok:
            out["eccofs"] = f"{base_url}{f}_rtofs-vs-eccofs.png"
        if variable_depth == "temperature_0m":
            out["goes16"] = f"{base_url}{f}_rtofs-vs-GOES16.png"
            out["goes19"] = f"{base_url}{f}_rtofs-vs-GOES19.png"
    return out


def _cases():
    from ioos_model_comparisons.region_catalog import build_region_info
    from ioos_model_comparisons.registry import load_registry
    reg = load_registry()
    info = build_region_info()
    names = {v["key"]: n for n, v in info.items()}
    for key in reg.portal.map_regions:
        name = names[key]
        vds = [f"{v}_{d}" for v, ds in info[name]["depths"].items() for d in ds] + ["ocean_heat_content"]
        for vd in vds:
            yield name, info[name]["folder"], vd


@pytest.mark.parametrize("name,folder,vd", list(_cases()))
def test_predicted_urls_match_old_builder(name, folder, vd):
    from ioos_model_comparisons.registry import load_registry
    reg = load_registry()
    when = dt.datetime(2026, 10, 12, 6)
    old = old_build_map_urls(name, vd, when.date(), "06Z")
    if name == "Caribbean: Windward Islands":
        old = {k: v and v.replace("caribbean:_windward_islands", "caribbean-windward")
               .replace("caribbean:-windward-islands", "caribbean-windward") for k, v in old.items()}
    for panel in reg.map_panels(folder, vd):
        predicted = [reg.public_url(reg.predict_rel(c, folder, name, vd, when))
                     for c in panel.comparisons if reg.comparison_applies(c, folder, vd)]
        assert old[panel.key] in predicted, (panel.key, old[panel.key], predicted)
