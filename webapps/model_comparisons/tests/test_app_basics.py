import threading

import ioos_model_comparisons


def test_vendored_package_is_used():
    # pytest.ini puts the app directory first, as in production.
    assert "webapps" in ioos_model_comparisons.__file__


def test_import_has_no_side_effects():
    before = threading.active_count()
    import app  # noqa: F401
    from ioos_model_comparisons import db
    assert threading.active_count() == before
    assert db._client is None  # no MongoDB connection attempted by importing


def test_index_renders_with_config_driven_panels(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "const PANEL_DEFS" in html and "const MAP_PANELS" in html
    # Overview radios come from comparisons.yml, ESPC first (overview_order).
    assert html.index('id="ov_mod_espc"') < html.index('id="ov_mod_copernicus"')
    assert 'id="ov_mod_goes19"' in html


def test_map_panels_injected_per_region(app_module, client):
    import json
    import re
    html = client.get("/").get_data(as_text=True)
    panels = json.loads(re.search(r"const MAP_PANELS = (\{.*?\});\n", html).group(1))
    assert panels["Fiji"]["ocean_heat_content"] == ["espc_cmems"]
    assert panels["Guam"]["temperature_0m"] == ["copernicus", "espc", "espc_cmems", "goes16", "goes19"]
    assert "Caribbean: Windward Islands" in panels


def test_healthz(client):
    r = client.get("/healthz")
    body = r.get_json()
    assert r.status_code == 200
    assert body["version"] == "dev" and body["mongo"]["ok"] is False
    assert body["discovery"] == {"mode": "legacy", "backend": "legacy"}
    assert body["unknown_map_regions"] == [] and len(body["config"]["hash"]) == 12
    assert client.get("/healthz?strict=1").status_code == 503


def test_healthz_ok_with_fresh_catalog(client, seed, monkeypatch):
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    seed(f"maps/gulf_of_mexico/temperature_0m/{now:%Y/%m}/"
         f"gulf-of-mexico_{now:%Y-%m-%d}T000000Z_temperature-0m_rtofs-vs-espc.png")
    monkeypatch.setenv("DISCOVERY_BACKEND", "auto")
    body = client.get("/healthz?strict=1").get_json()
    assert body["status"] == "ok" and body["catalog"]["stale"] is False
    assert body["discovery"]["backend"] == "catalog"


def test_ohc_table_columns_follow_db_fields(client):
    html = client.get("/ohc-table").get_data(as_text=True)
    assert "OHC RTOFS Parallel" in html and '"ohc_rtofs_parallel"' in html


def test_storm_track_served_from_shared_cache(client, fake_mongo, app_module, monkeypatch):
    import datetime as dt
    from ioos_model_comparisons import storm_track
    monkeypatch.setattr(storm_track, "storms_for_date",
                        lambda d: (_ for _ in ()).throw(AssertionError("tropycal should not run")))
    fake_mongo["hurricanes"]["storm_tracks"].insert_one({
        "_id": "2025-09-01", "computed_at": dt.datetime(2025, 9, 2), "v": storm_track.CACHE_VERSION,
        "storms": {"AL052025": {"name": "ERIN", "track": []}}})
    app_module._storm_cache.clear()
    r = client.get("/api/storm-track?date=2025-09-01")
    assert r.status_code == 200 and r.get_json()["storms"]["AL052025"]["name"] == "ERIN"
