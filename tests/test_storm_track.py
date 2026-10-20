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
