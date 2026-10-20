"""storm_track: live (NHC + JTWC) storms are merged in for recent dates."""

import datetime as dt
import types

import pytest

from ioos_model_comparisons import storm_track as st


def _storm(sid, name, basin, days, invest=False):
    t0 = dt.datetime(2026, 10, 20, 12)
    times = [t0 - dt.timedelta(hours=6 * i) for i in range(days * 4)][::-1]
    return types.SimpleNamespace(
        id=sid, name=name, basin=basin, invest=invest,
        dict={"time": times, "lat": [15.0] * len(times), "lon": [140.0] * len(times),
              "vmax": [90.0] * len(times), "type": ["TY"] * len(times)})


@pytest.fixture
def live(monkeypatch):
    champi = _storm("WP292026", "CHAMPI", "west_pacific", 6)
    invest = _storm("AL932026", "93L", "north_atlantic", 2, invest=True)
    monkeypatch.setattr(st, "BASINS", ())                       # no archive downloads
    monkeypatch.setattr(st, "_realtime", {"ts": 0.0, "storms": None, "refreshing": False})
    monkeypatch.setattr(st, "_load_realtime_storms", lambda: [(champi, None), (invest, None)])
    return champi


def test_recent_date_includes_live_west_pacific_storm(live, monkeypatch):
    monkeypatch.setattr(st, "is_recent", lambda d, today=None: True)
    out = st.storms_for_date(dt.date(2026, 10, 20))
    assert set(out) == {"WP292026", "AL932026"}
    champi = out["WP292026"]
    assert champi["name"] == "CHAMPI" and champi["basin"] == "west_pacific"
    assert champi["forecast"] == [] and champi["cone"] is None   # no forecast: track only
    assert out["AL932026"]["invest"] is True and out["AL932026"]["category_label"] == "Invest"


def test_old_date_does_not_use_live_feed(live, monkeypatch):
    monkeypatch.setattr(st, "is_recent", lambda d, today=None: False)
    assert st.storms_for_date(dt.date(2026, 9, 1)) == {}


def test_live_feed_is_loaded_once_then_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(st, "_realtime", {"ts": 0.0, "storms": None, "refreshing": False})
    monkeypatch.setattr(st, "_load_realtime_storms", lambda: calls.append(1) or [])
    st.realtime_storms(); st.realtime_storms()
    assert len(calls) == 1


def test_live_feed_failure_is_not_fatal(monkeypatch):
    def boom():
        raise RuntimeError("NHC down")
    monkeypatch.setattr(st, "_realtime", {"ts": 0.0, "storms": None, "refreshing": False})
    monkeypatch.setattr(st, "_load_realtime_storms", boom)
    assert st.realtime_storms() == []


def test_is_recent():
    today = dt.date(2026, 10, 20)
    assert st.is_recent(dt.date(2026, 10, 18), today) and not st.is_recent(dt.date(2026, 10, 17), today)


# ---------------------------------------------------------------- shared cache
@pytest.fixture
def computed(monkeypatch):
    calls = []

    def fake(date_obj):
        calls.append(date_obj)
        return {"WP292026": {"name": "CHAMPI", "computed_for": date_obj.isoformat()}}

    monkeypatch.setattr(st, "storms_for_date", fake)
    monkeypatch.setattr(st, "_recomputing", set())
    return calls


def _coll(client):
    return client["hurricanes"]["storm_tracks"]


def test_new_date_is_computed_once_then_served_from_mongo(fake_mongo, computed):
    d = dt.date(2025, 9, 1)                           # old: final once stored
    first = st.cached_storms_for_date(d)
    again = st.cached_storms_for_date(d)
    assert first == again and computed == [d]
    assert _coll(fake_mongo).find_one({"_id": "2025-09-01"})["storms"]["WP292026"]["name"] == "CHAMPI"


def test_stale_entry_is_served_and_recomputed_in_background(fake_mongo, computed, monkeypatch):
    started = []
    monkeypatch.setattr(st, "_recompute_in_background", lambda d: started.append(d))
    d = st._today()
    _coll(fake_mongo).insert_one({"_id": d.isoformat(), "storms": {"OLD": {}},
                                  "computed_at": st._utcnow() - dt.timedelta(hours=3)})
    assert st.cached_storms_for_date(d) == {"OLD": {}}    # instant, no recompute in the request
    assert computed == [] and started == [d]


def test_fresh_entry_is_not_recomputed(fake_mongo, computed, monkeypatch):
    monkeypatch.setattr(st, "_recompute_in_background", lambda d: pytest.fail("should not recompute"))
    d = st._today()
    _coll(fake_mongo).insert_one({"_id": d.isoformat(), "storms": {"X": {}}, "computed_at": st._utcnow()})
    assert st.cached_storms_for_date(d) == {"X": {}}


def test_without_mongo_it_computes_in_process(mongo_down, computed):
    assert st.cached_storms_for_date(dt.date(2026, 10, 20))["WP292026"]["name"] == "CHAMPI"


def test_refresh_recent_skips_fresh_older_days_but_always_redoes_live_days(fake_mongo, computed):
    today = dt.date(2026, 10, 20)
    now = st._utcnow()
    for back in range(7):
        d = today - dt.timedelta(days=back)
        _coll(fake_mongo).insert_one({"_id": d.isoformat(), "storms": {}, "computed_at": now})
    done = st.refresh_recent(days=7, coll=_coll(fake_mongo), today=today)
    assert done == [today, today - dt.timedelta(days=1), today - dt.timedelta(days=2)]


def test_only_one_process_gets_the_lease(fake_mongo):
    coll = _coll(fake_mongo)
    assert st._acquire_lease(coll, 600) is True
    assert st._acquire_lease(coll, 600) is False           # held
    coll.update_one({"_id": st._LEASE_ID}, {"$set": {"until": st._utcnow() - dt.timedelta(seconds=1)}})
    assert st._acquire_lease(coll, 600) is True            # expired -> next one takes it


def test_max_age():
    today = dt.date(2026, 10, 20)
    assert st.max_age_seconds(today, today) == st.REFRESH_SECONDS + 900
    assert st.max_age_seconds(dt.date(2026, 10, 10), today) == 24 * 3600
    assert st.max_age_seconds(dt.date(2026, 8, 1), today) is None
