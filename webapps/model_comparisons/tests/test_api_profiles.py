"""Profile ID / latest-date / availability routes, both backends."""

import json

import responses

from conftest import BASE

ARGO = "profiles/argo/gulf_of_mexico/2026/10/11/7902352-profile-2026-10-11T0714Z.png"
ARGO_FIJI = "profiles/argo/fiji/2026/10/11/5904628-profile-2026-10-11T2211Z.png"
FVON = "profiles/fvon/south_pacific/2026/10/11/0-22000-0-QNE94JY-profile-2026-10-11T040044Z.png"
GLIDER = "profiles/gliders/2026/10-11/ru29-20260101T0000_20261011_to_20261012_400m.png"
GULF_1000M = "profiles/gulf_gliders/2026/10-11/sg622-20250514T0000_20261011_to_20261012_1000m.png"


def test_argo_ids_and_latest_from_catalog(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(ARGO, ARGO_FIJI)
    ids = client.get("/api/argo-ids?date=2026-10-11").get_json()["ids"]
    assert ids == {"7902352": {"filename": ARGO.rsplit("/", 1)[1], "region": "gulf_of_mexico"},
                   "5904628": {"filename": ARGO_FIJI.rsplit("/", 1)[1], "region": "fiji"}}
    assert client.get("/api/argo-latest-date").get_json() == {"date": "2026-10-11"}
    assert client.get("/api/argo-ids?date=2026-10-10").get_json() == {"ids": {}}


def test_argo_profile_availability_from_catalog(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(ARGO)
    q = {"region": "gulf_of_mexico", "filename": ARGO.rsplit("/", 1)[1], "date": "2026-10-11"}
    assert client.get("/api/argo-profile", query_string=q).get_json() == {"url": BASE + ARGO, "available": True}
    q["filename"] = "123-profile-2026-10-11T0000Z.png"
    assert client.get("/api/argo-profile", query_string=q).get_json()["available"] is False


def test_glider_ids_exclude_gulf_1000m_group(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(GLIDER, GULF_1000M)
    assert client.get("/api/glider-ids?date=2026-10-11").get_json() == {"ids": ["ru29-20260101T0000"]}
    assert client.get("/api/glider-latest-date").get_json() == {"date": "2026-10-11"}
    r = client.get("/api/glider-profile", query_string={"glider_id": "ru29-20260101T0000",
                                                        "date": "2026-10-11"}).get_json()
    assert r == {"url": BASE + GLIDER, "available": True}


def test_fvon_ids_from_catalog(client, seed, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(FVON)
    ids = client.get("/api/fvon-ids?date=2026-10-11").get_json()["ids"]
    assert ids == {"0-22000-0-QNE94JY": {"filename": FVON.rsplit("/", 1)[1], "region": "south_pacific"}}
    assert client.get("/api/fvon-latest-date").get_json() == {"date": "2026-10-11"}


def test_argo_ids_legacy_scrapes_listings(client, rsps):
    listing = f'<a href="{ARGO.rsplit("/", 1)[1]}">x</a>'
    rsps.add(responses.GET, BASE + "profiles/argo/gulf_of_mexico/2026/10/11/", body=listing)
    for folder in ("fiji", "mid_atlantic_bight", "south_atlantic_bight", "caribbean",
                   "tropical_western_atlantic", "west_florida_shelf", "guam", "hawaii", "mexico_pacific"):
        rsps.add(responses.GET, BASE + f"profiles/argo/{folder}/2026/10/11/", status=404)
    ids = client.get("/api/argo-ids?date=2026-10-11").get_json()["ids"]
    assert ids == {"7902352": {"filename": ARGO.rsplit("/", 1)[1], "region": "gulf_of_mexico"}}


def test_profile_locations_with_local_plots_dir(client, tmp_path, monkeypatch, app_module):
    # Used to raise UnboundLocalError (a local `import os` shadowed the module).
    d = tmp_path / "profiles" / "argo" / "mid_atlantic_bight" / "last_14_days"
    d.mkdir(parents=True)
    (d / "locations.json").write_text(json.dumps(
        {"1-profile-2026-10-11T0000Z.png": {"lat": 1, "lon": 2}}))
    monkeypatch.setattr(app_module, "_LOCAL_PLOTS_BASE", str(tmp_path))
    r = client.get("/api/profile-locations?type=argo&region=Mid Atlantic Bight")
    assert r.status_code == 200 and "1-profile-2026-10-11T0000Z.png" in r.get_json()


def test_download_rejects_foreign_urls_and_sets_timeout(client, rsps):
    assert client.get("/api/download?url=https://example.com/x.png").status_code == 400
    assert client.get("/api/download?url=https://rucool.marine.rutgers.edu/other/x.png").status_code == 400
    url = BASE + ARGO
    rsps.add(responses.GET, url, body=b"png", content_type="image/png")
    r = client.get("/api/download", query_string={"url": url})
    assert r.status_code == 200 and r.data == b"png"
    assert rsps.calls[-1].request.req_kwargs["timeout"] == (5, 30)


def _with_position(fake_mongo, rel, lat, lon):
    fake_mongo["hurricanes"]["figures"].update_one(
        {"_id": rel}, {"$set": {"platform.lat": lat, "platform.lon": lon}})


def test_profile_positions_from_catalog(client, seed, fake_mongo, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(GLIDER, ARGO, FVON)
    _with_position(fake_mongo, GLIDER, 39.1, -73.2)
    _with_position(fake_mongo, ARGO, 25.0, -90.0)
    _with_position(fake_mongo, FVON, -17.0, 178.0)
    data = client.get("/api/profile-positions?date=2026-10-11").get_json()
    g = data["glider"][GLIDER.rsplit("/", 1)[1]]
    assert g == {"lat": 39.1, "lon": -73.2, "glider_id": "ru29-20260101T0000",
                 "time": "2026-10-11", "url": BASE + GLIDER, "type": "glider"}
    assert data["argo"][ARGO.rsplit("/", 1)[1]]["wmo"] == "7902352"
    assert data["fvon"][FVON.rsplit("/", 1)[1]]["wigos"] == "0-22000-0-QNE94JY"


def test_profile_positions_fill_only_missing_from_legacy(client, seed, fake_mongo, monkeypatch, app_module):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    seed(GLIDER, ARGO)
    _with_position(fake_mongo, GLIDER, 39.1, -73.2)   # ARGO has no position recorded
    called = []

    def fake_fetch_argo_floats(**kw):
        called.append(kw)
        import pandas as pd
        return pd.DataFrame({"argo": ["7902352"], "time": ["2026-10-11"], "lat": [24.0],
                             "lon": [-91.0]}).set_index(["argo", "time"])

    monkeypatch.setattr(app_module, "get_argo_floats_by_time", fake_fetch_argo_floats)
    data = client.get("/api/profile-positions?date=2026-10-12").get_json()  # nothing that day
    assert data == {"glider": {}, "argo": {}, "fvon": {}} and not called
    app_module._pos_cache.clear()
    data = client.get("/api/profile-positions?date=2026-10-11").get_json()
    assert data["argo"][ARGO.rsplit("/", 1)[1]]["lat"] == 24.0 and called
    assert data["glider"][GLIDER.rsplit("/", 1)[1]]["lat"] == 39.1


def test_glider_tracks_from_catalog(client, seed, fake_mongo, monkeypatch):
    monkeypatch.setenv("DISCOVERY_BACKEND", "catalog")
    day2 = GLIDER.replace("10-11", "10-10").replace("20261011_to_20261012", "20261010_to_20261011")
    seed(GLIDER, day2)
    _with_position(fake_mongo, GLIDER, 39.1, -73.2)
    _with_position(fake_mongo, day2, 39.0, -73.0)
    tracks = client.get("/api/glider-tracks?date=2026-10-11").get_json()
    assert tracks == {"ru29-20260101T0000": [
        {"lat": 39.0, "lon": -73.0, "time": "2026-10-10"},
        {"lat": 39.1, "lon": -73.2, "time": "2026-10-11"}]}
