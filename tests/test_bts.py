import io
import zipfile

import pandas as pd

from swe_airline_delay import http, storage
from swe_airline_delay.flight_delay import bts

RAW = pd.DataFrame(
    {
        "FlightDate": ["2024-05-11", "2024-05-11", "2024-05-11", "2024-05-11"],
        "Reporting_Airline": ["DL", "AA", "UA", "WN"],
        "Flight_Number_Reporting_Airline": [100, 200, 300, 400],
        "Origin": ["ATL", "ORD", "ATL", "XYZ"],  # XYZ is not a configured airport
        "Dest": ["JFK", "LAX", "DEN", "ATL"],
        "CRSDepTime": [600, 2400, 1500, 900],
        "DepDelay": [30.0, -5.0, None, 10.0],
        "CRSElapsedTime": [150.0, 240.0, 200.0, 100.0],
        "ArrDelay": [45.0, -10.0, None, 5.0],
        "Cancelled": [0.0, 0.0, 1.0, 0.0],
        "CancellationCode": [None, None, "B", None],
        "CarrierDelay": [10.0, None, None, None],
        "WeatherDelay": [0.0, None, None, None],
        "NASDelay": [35.0, None, None, None],
        "SecurityDelay": [0.0, None, None, None],
        "LateAircraftDelay": [0.0, None, None, None],
        "Unused": [1, 2, 3, 4],
    }
)


def _zip_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("On_Time_2024_5.csv", df.to_csv(index=False))
    return buf.getvalue()


def test_urls():
    urls = bts.discover_urls("2024-05-10", "2024-06-02")
    assert [(y, m) for y, m, _ in urls] == [(2024, 5), (2024, 6)]
    assert urls[0][2].endswith(
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_5.zip"
    )


def test_normalize_schema_and_values():
    out = bts.normalize(RAW)
    assert list(out.columns) == bts.TIDY_COLUMNS
    assert set(out["airport"]) == {"ATL", "ORD"}  # XYZ dropped
    atl = out[out["flight_id"].str.contains("DL100")].iloc[0]
    # 06:00 EDT == 10:00 UTC
    assert atl["scheduled_dep"] == pd.Timestamp("2024-05-11 10:00", tz="UTC")
    assert atl["actual_dep"] == pd.Timestamp("2024-05-11 10:30", tz="UTC")
    assert atl["scheduled_arr"] == pd.Timestamp("2024-05-11 12:30", tz="UTC")
    assert atl["actual_arr"] == pd.Timestamp("2024-05-11 13:15", tz="UTC")
    assert atl["delay_cause_code"] == "N" and atl["nas_delay_min"] == 35
    assert not atl["cancelled"]
    # CRSDepTime 2400 -> midnight at the *end* of 2024-05-11, CDT (UTC-5) => 05:00 UTC on the 12th
    ord_ = out[out["airport"] == "ORD"].iloc[0]
    assert ord_["scheduled_dep"] == pd.Timestamp("2024-05-12 05:00", tz="UTC")
    assert pd.isna(ord_["delay_cause_code"])
    canx = out[out["cancelled"]].iloc[0]
    assert (
        canx["cancel_code"] == "B" and pd.isna(canx["actual_dep"]) and pd.isna(canx["actual_arr"])
    )
    assert out["dep_delay_min"].dtype.kind == "f" and out["cancelled"].dtype == bool


def test_fetch_downloads_once_then_cached(monkeypatch):
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_zip_bytes(RAW))
        return True

    monkeypatch.setattr(http, "download", fake_download)
    df = bts.fetch("2024-05-01", "2024-05-31", airports="ATL,ORD")
    assert len(calls) == 1 and len(df) == 3
    assert bts.raw_csv_path(2024, 5).exists()  # data/raw/bts/2024/5.csv
    assert not bts.raw_csv_path(2024, 5).with_suffix(".zip").exists()
    assert storage.Manifest().has("bts", "ATL", "2024-05")
    assert len(bts.load(airports="ATL,ORD")) == 3

    again = bts.fetch("2024-05-01", "2024-05-31", airports="ATL,ORD")  # idempotent
    assert len(calls) == 1 and again.empty

    forced = bts.fetch("2024-05-01", "2024-05-31", airports="ATL,ORD", force=True)  # raw cache hit
    assert len(calls) == 1 and len(forced) == 3


def test_unpublished_month_is_skipped(monkeypatch):
    monkeypatch.setattr(http, "download", lambda url, dest: False)
    assert bts.fetch("2030-01-01", "2030-01-31").empty
    assert not storage.Manifest().entries
