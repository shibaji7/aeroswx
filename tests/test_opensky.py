import pandas as pd
import pytest
from conftest import FakeResponse

from swe_airline_delay import http, storage
from swe_airline_delay.adsb import opensky


def test_bbox():
    lamin, lomin, lamax, lomax = opensky.airport_bbox(40.0, -75.0, 111.0)
    assert lamin == pytest.approx(39.0) and lamax == pytest.approx(41.0)
    assert lomax - lomin > 2.0  # longitude degrees are shorter at 40 N


def test_live_states_anonymous(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, **kw):
        seen.update(url=url, params=params, headers=headers)
        state = ["abc123", "DAL1  ", "United States", 1715400000, 1715400001, -84.4, 33.6, 1000.0,
                 False, 120.0, 90.0, 0.0, None, 1050.0, "1200", False, 0]  # fmt: skip
        return FakeResponse(json_data={"time": 1715400002, "states": [state]})

    monkeypatch.setattr(http, "get", fake_get)
    df = opensky.fetch_live("ATL")
    assert seen["url"].endswith("/states/all") and seen["headers"] == {}  # no auth header
    assert list(df["icao24"]) == ["abc123"] and df["callsign"].iloc[0] == "DAL1"
    assert df["time_position"].dt.tz is not None and df["airport"].iloc[0] == "ATL"
    assert storage.Manifest().entries


def test_live_with_free_credentials_uses_oauth(monkeypatch):
    posted = {}

    def fake_post(url, data=None, **kw):
        posted.update(url=url, data=data)
        return FakeResponse(json_data={"access_token": "tok", "expires_in": 1800})

    got = {}

    def fake_get(url, params=None, headers=None, **kw):
        got["headers"] = headers
        return FakeResponse(json_data={"time": 1, "states": None})

    monkeypatch.setattr(http, "post", fake_post)
    monkeypatch.setattr(http, "get", fake_get)
    client = opensky.OpenSkyClient("id", "secret")
    df = client.states_in_bbox((0, 0, 1, 1))
    assert df.empty and got["headers"] == {"Authorization": "Bearer tok"}
    assert posted["data"]["grant_type"] == "client_credentials"


def test_rate_limit_retry(monkeypatch):
    responses = [
        FakeResponse(429, headers={"X-Rate-Limit-Retry-After-Seconds": "0"}),
        FakeResponse(json_data={"ok": 1}),
    ]
    monkeypatch.setattr(http, "get", lambda *a, **k: responses.pop(0))
    assert opensky.OpenSkyClient().get_json("/x") == {"ok": 1}


def _msgs():
    t0 = pd.Timestamp("2024-05-11 00:00", tz="UTC")
    rows = []
    for rx, secs in (("r1", [1, 3, 5, 61, 62]), ("r2", [100])):
        for s in secs:
            rows.append((t0 + pd.Timedelta(seconds=s), "ATL", "ab", rx))
    return pd.DataFrame(rows, columns=["time", "airport", "icao24", "receiver"])


def test_metrics_basic_and_gap_fill():
    m = opensky.compute_receiver_metrics(_msgs(), window="2min", subbin_s=10)
    r1 = m[m["receiver"] == "r1"].iloc[0]
    assert r1["n_msgs"] == 5 and r1["msg_rate_hz"] == pytest.approx(5 / 120)
    assert r1["max_gap_s"] == pytest.approx(58.0)  # trailing edge: 62 s -> 120 s (> 56 s mid gap)
    assert r1["coverage_frac"] == pytest.approx(2 / 12)  # 10 s sub-bins {0, 6} of 12 occupied
    # window grid with zero-filled outages
    full = opensky.compute_receiver_metrics(
        _msgs(), window="1min", start="2024-05-11 00:00", end="2024-05-11 00:05"
    )
    r2 = full[full["receiver"] == "r2"].set_index("window_start")
    assert len(r2) == 5 and r2["n_msgs"].sum() == 1
    assert (r2.loc[r2["n_msgs"] == 0, "max_gap_s"] == 60).all()


def test_metrics_without_receiver_column():
    m = opensky.compute_receiver_metrics(_msgs().drop(columns="receiver"), window="5min")
    assert list(m["receiver"]) == ["ALL"] and m["n_msgs"].iloc[0] == 6


def test_fetch_historical_chunks_by_day_and_is_idempotent():
    calls = []

    def fake_history(day, bbox):
        calls.append(day)
        return pd.DataFrame(
            {
                "time": [day + pd.Timedelta(seconds=5)] * 2,
                "icao24": ["a", "b"],
                "serials": [[1, 2], [2]],
            }
        )

    out = opensky.fetch_historical(
        "2024-05-11", "2024-05-12", "ATL", history_fn=fake_history, window="1h"
    )
    assert len(calls) == 2  # one call per day
    assert set(out["receiver"]) == {"1", "2"}
    again = opensky.fetch_historical("2024-05-11", "2024-05-12", "ATL", history_fn=fake_history)
    assert len(calls) == 2 and again.empty


def test_fetch_mode_validation():
    with pytest.raises(ValueError):
        opensky.fetch(mode="historical")
    with pytest.raises(ValueError):
        opensky.fetch(mode="bogus")
