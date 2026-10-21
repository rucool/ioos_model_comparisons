"""/api/maps and /api/overview-latest: same JSON contract from either backend."""

import datetime as dt

import pytest
import responses

from conftest import BASE

KEYS = {"copernicus", "espc", "three_model", "eccofs", "nesdis", "goes16", "goes19"}
GOM = "maps/gulf_of_mexico/temperature_0m/2026/10/gulf-of-mexico_2026-10-12T060000Z_temperature-0m_"


def _maps(client, **kw):
    q = {"region": "Gulf of Mexico", "variable_depth": "temperature_0m",
         "date": "2026-10-12", "time": "06Z", **kw}
    return client.get("/api/maps", query_string=q)


def test_catalog_backend(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(GOM + "rtofs-vs-espc.png", GOM + "rtofs-vs-GOES16.png")
    data = _maps(client).get_json()
    assert set(data) == KEYS
    assert data["espc"] == {"url": BASE + GOM + "rtofs-vs-espc.png", "available": True,
                            "label": "RTOFS vs. ESPC", "rtofs": {"operational": None}}
    assert data["goes16"]["available"] and data["goes16"]["url"].endswith("rtofs-vs-GOES16.png")
    # Applicable but not (yet) published: predicted URL, unavailable.
    assert data["copernicus"]["available"] is False
    assert data["copernicus"]["url"] == BASE + GOM + "rtofs-vs-cmems.png"
    # Not applicable to this region: no URL at all.
    assert data["three_model"] == {"url": None, "available": False, "label": "RTOFS vs. ESPC vs. CMEMS"}


def test_legacy_backend_heads_only_applicable_panels(client, rsps):
    rsps.add(responses.HEAD, BASE + GOM + "rtofs-vs-espc.png", status=200)
    for suffix in ("rtofs-vs-cmems", "rtofs-vs-eccofs", "rtofs-vs-GOES16", "rtofs-vs-GOES19"):
        rsps.add(responses.HEAD, BASE + GOM + suffix + ".png", status=404)
    data = _maps(client).get_json()
    assert set(data) == KEYS
    assert data["espc"]["available"] and not data["copernicus"]["available"]
    assert data["nesdis"]["url"] is None  # not applicable to temperature: never requested
    assert len(rsps.calls) == 5


def test_windward_islands_uses_real_folder(client, rsps):
    # The old URL builder produced maps/caribbean:_windward_islands/... (404).
    prefix = (BASE + "maps/caribbean-windward/salinity_0m/2026/10/"
              "caribbean-windward_2026-10-12T060000Z_salinity-0m_")
    rsps.add(responses.HEAD, prefix + "rtofs-vs-cmems.png", status=200)
    rsps.add(responses.HEAD, prefix + "rtofs-vs-espc.png", status=404)
    rsps.add(responses.HEAD, prefix + "rtofs-vs-eccofs.png", status=404)
    data = _maps(client, region="Caribbean: Windward Islands", variable_depth="salinity_0m").get_json()
    assert data["copernicus"]["available"] is True


def test_fiji_ohc_combined_image(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    rel = "maps/fiji/ocean_heat_content/2026/10/Fiji_2026-10-12T000000Z_ohc_rtofs-espc-cmems.png"
    seed(rel)
    data = _maps(client, region="Fiji", variable_depth="ocean_heat_content", time="00Z").get_json()
    assert data["three_model"]["available"] and data["three_model"]["url"] == BASE + rel
    assert data["espc"]["url"] is None


def test_auto_falls_back_to_legacy_when_mongo_down(client, rsps, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "auto")
    rsps.add(responses.HEAD, BASE + GOM + "rtofs-vs-espc.png", status=200)
    for suffix in ("rtofs-vs-cmems", "rtofs-vs-eccofs", "rtofs-vs-GOES16", "rtofs-vs-GOES19"):
        rsps.add(responses.HEAD, BASE + GOM + suffix + ".png", status=404)
    assert _maps(client).get_json()["espc"]["available"] is True


def test_auto_uses_legacy_while_catalog_is_stale(client, seed, rsps, monkeypatch, fake_mongo):
    monkeypatch.setenv("DISCOVERY_BACKEND", "auto")
    seed(GOM + "rtofs-vs-espc.png")
    fake_mongo["hurricanes"]["figures"].update_many(
        {}, {"$set": {"created_at": dt.datetime(2020, 1, 1)}})
    rsps.add(responses.HEAD, BASE + GOM + "rtofs-vs-espc.png", status=404)
    for suffix in ("rtofs-vs-cmems", "rtofs-vs-eccofs", "rtofs-vs-GOES16", "rtofs-vs-GOES19"):
        rsps.add(responses.HEAD, BASE + GOM + suffix + ".png", status=404)
    assert _maps(client).get_json()["espc"]["available"] is False  # legacy answered


def test_shadow_logs_differences(client, seed, rsps, monkeypatch, caplog):
    monkeypatch.setenv("DISCOVERY_BACKEND", "shadow")
    seed(GOM + "rtofs-vs-espc.png")
    for suffix in ("rtofs-vs-espc", "rtofs-vs-cmems", "rtofs-vs-eccofs", "rtofs-vs-GOES16", "rtofs-vs-GOES19"):
        rsps.add(responses.HEAD, BASE + GOM + suffix + ".png", status=404)
    with caplog.at_level("WARNING", logger="discovery.shadow"):
        data = _maps(client).get_json()
    assert data["espc"]["available"] is False  # legacy is authoritative in shadow mode
    assert any("map_panels differs" in r.message for r in caplog.records)


def test_overview_latest_from_catalog(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(GOM + "rtofs-vs-espc.png", GOM.replace("060000Z", "180000Z") + "rtofs-vs-cmems.png")
    data = client.get("/api/overview-latest", query_string={
        "region": "Gulf of Mexico", "variable_depth": "temperature_0m"}).get_json()
    assert data["date"] == "2026-10-12" and data["time"] == "18Z"
    assert data["copernicus"]["available"] and not data["espc"]["available"]
    assert KEYS <= set(data)


def test_overview_latest_none(client, fake_mongo, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    r = client.get("/api/overview-latest", query_string={"region": "Hawaii"})
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["2026-13-01", "nope"])
def test_bad_date(client, bad):
    assert _maps(client, date=bad).status_code == 400
