import datetime as dt
from pathlib import Path

import pytest

from ioos_model_comparisons import figure_names as fn

SAMPLES = [
    line.strip()
    for line in (Path(__file__).parent / "fixtures" / "sample_paths.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]


def test_samples_present():
    assert len(SAMPLES) > 100


@pytest.mark.parametrize("rel", SAMPLES)
def test_real_paths_parse_and_round_trip(rel):
    meta = fn.parse(rel)
    assert meta is not None, rel
    assert fn.build(meta) == rel


def test_model_comparison():
    m = fn.parse("maps/gulf_of_mexico/temperature_0m/2026/10/"
                 "gulf-of-mexico_2026-10-12T060000Z_temperature-0m_rtofs-vs-espc.png")
    assert (m.family, m.product, m.region) == ("map", "model_comparison", "gulf_of_mexico")
    assert (m.variable, m.depth, m.var_depth) == ("temperature", 0, "temperature_0m")
    assert m.valid_time == dt.datetime(2026, 10, 12, 6)
    assert m.models == ("rtofs", "espc") and m.comparison == "rtofs-vs-espc"


def test_goes_is_sst_with_lowercase_models_but_exact_rebuild():
    rel = ("maps/caribbean/temperature_0m/2026/08/"
           "caribbean_2026-08-06T000000Z_temperature-0m_rtofs-vs-GOES16.png")
    m = fn.parse(rel)
    assert m.family == "sst" and m.comparison == "rtofs-vs-goes16"
    assert m.var_depth == "temperature_0m"
    assert fn.build(m) == rel


def test_currents_use_underscored_slug():
    m = fn.parse("maps/south_atlantic_bight/currents_200m/2026/07/"
                 "south_atlantic_bight_2026-07-30T120000Z_currents-200m_rtofs-vs-eccofs.png")
    assert m.product == "streamplot" and m.var_depth == "currents_200m"


def test_windward_folder_is_kept_verbatim():
    m = fn.parse("maps/caribbean-windward/salinity_0m/2026/09/"
                 "caribbean-windward_2026-09-01T000000Z_salinity-0m_rtofs-vs-cmems.png")
    assert m.region == "caribbean-windward"


def test_ohc_splits_rtofs_parallel_token():
    m = fn.parse("maps/gulf_of_mexico/ocean_heat_content/2026/09/"
                 "gulf_of_mexico_2026-09-18T000000Z_heat_content_rtofs-p-espc.png")
    assert m.models == ("rtofs-p", "espc") and m.var_depth == "ocean_heat_content"


def test_ohc_nesdis():
    m = fn.parse("maps/hawaii/ocean_heat_content/2026/09/"
                 "hawaii_2026-09-18T000000Z_heat_content_rtofs-nesdis.png")
    assert m.comparison == "rtofs-vs-nesdis"


def test_fiji_combined_ohc():
    rel = "maps/fiji/ocean_heat_content/2026/09/Fiji_2026-09-18T000000Z_ohc_rtofs-espc-cmems.png"
    m = fn.parse(rel)
    assert m.product == "ohc_combined" and m.comparison == "rtofs-vs-espc-vs-cmems"
    assert m.region == "fiji" and fn.build(m) == rel


def test_fvon_hyphenated_wigos_id():
    m = fn.parse("profiles/fvon/south_pacific/2026/08/20/"
                 "0-22000-0-QNE94JY-profile-2026-08-20T040044Z.png")
    assert m.platform_type == "fvon" and m.platform_id == "0-22000-0-QNE94JY"
    assert m.valid_time == dt.datetime(2026, 8, 20, 4, 0, 44)


def test_argo_minute_timestamp():
    m = fn.parse("profiles/argo/gulf_of_mexico/2026/07/30/7902352-profile-2026-07-30T0714Z.png")
    assert m.platform_id == "7902352" and m.valid_time == dt.datetime(2026, 7, 30, 7, 14)


def test_glider_profile():
    m = fn.parse("profiles/gliders/2026/06-08/sp213-20260317T1523_20260608_to_20260609_400m.png")
    assert m.platform_type == "glider" and m.platform_id == "sp213-20260317T1523"
    assert m.valid_time == dt.datetime(2026, 6, 8) and m.depth == 400


def test_asg_with_spaces_and_percent_encoding():
    rel = ("adaptive_sampling_guidance/maps/gom_east/sea_surface_salinity/three_panel/espc/2026/06/"
           "Eastern Gulf of Mexico_2026-06-14T180000Z_sss_rtofs-espc-diff.png")
    m = fn.parse(rel)
    assert m.family == "adaptive" and m.extra["name"] == "Eastern Gulf of Mexico"
    assert fn.parse(rel.replace(" ", "%20")) is not None


@pytest.mark.parametrize("rel", [
    "profiles/argo/gulf_of_mexico/last_14_days/7902352-profile-2026-07-30T0714Z.png",
    "maps/atlantic_hurricanes/gulf_category4plus_summary_figure.png",
    "maps/gulf_stream/temperature_0m_turbo/2026/08/gulf-stream_2026-08-08T000000Z_temperature-0m_rtofs-vs-goes-16-turbo.png",
    "maps/gulf_of_mexico/temperature_0m/2026/10/notes.txt",
    "",
    None,
])
def test_unrecognized_returns_none(rel):
    assert fn.parse(rel) is None


def test_to_doc_profile_has_platform():
    d = fn.to_doc(fn.parse("profiles/argo/fiji/2026/04/24/5904628-profile-2026-04-24T2211Z.png"))
    assert d["platform"] == {"type": "argo", "id": "5904628"} and d["var_depth"] is None
