import h5py
import numpy as np
import pandas as pd
from conftest import FakeResponse

from swe_airline_delay import http, storage
from swe_airline_delay.space_weather import goes_xray as gx


def _flare_series():
    t = pd.date_range("2024-05-11 00:00", periods=360, freq="1min", tz="UTC")
    f = np.full(len(t), 2e-7)
    # an M5.2 flare peaking at 01:23 and a smaller C3 later; fast rise, slow decay
    for peak_idx, peak in ((83, 5.2e-5), (250, 3e-6)):
        for k in range(-15, 60):
            i = peak_idx + k
            if 0 <= i < len(f):
                shape = (k + 15) / 15 if k < 0 else np.exp(-k / 15)
                f[i] = max(f[i], 2e-7 + (peak - 2e-7) * shape)
    return pd.DataFrame(
        {"xrsa_flux": f / 10, "xrsb_flux": f}, index=pd.DatetimeIndex(t, name="time")
    )


def test_flare_classes():
    assert gx.flux_to_class(5.2e-5) == "M5.2"
    assert gx.flux_to_class(1.2e-4) == "X1.2"
    assert gx.flux_to_class(3e-6) == "C3.0"
    assert gx.flux_to_class(float("nan")) is None
    assert gx.class_to_flux("X2.5") == 2.5e-4


def test_extract_flares_and_crosscheck():
    flux = _flare_series()
    fl = gx.extract_flares(flux, min_class="C")
    assert list(fl["flare_class"]) == ["M5.2", "C3.0"]
    assert fl.loc[0, "peak_time"] == pd.Timestamp("2024-05-11 01:23", tz="UTC")
    assert fl.loc[0, "begin_time"] < fl.loc[0, "peak_time"] < fl.loc[0, "end_time"]
    assert len(gx.extract_flares(flux, min_class="M")) == 1

    official = pd.DataFrame(
        {"peak_time": [pd.Timestamp("2024-05-11 01:25", tz="UTC")], "flare_class": ["M5.8"]}
    )
    cc = gx.crosscheck_flares(fl, official, tolerance="10min")
    assert list(cc["matched"]) == [True, False] and cc.loc[0, "official_class"] == "M5.8"
    assert not gx.crosscheck_flares(fl, pd.DataFrame())["matched"].any()


def test_parse_swpc_json():
    recs = []
    for t in ("2024-05-11T00:00:00Z", "2024-05-11T00:01:00Z"):
        recs.append({"time_tag": t, "satellite": 16, "flux": 1e-6, "energy": "0.1-0.8nm"})
        recs.append({"time_tag": t, "satellite": 16, "flux": 1e-7, "energy": "0.05-0.4nm"})
    df = gx.parse_swpc_json(recs)
    assert list(df[["xrsa_flux", "xrsb_flux"]].iloc[0]) == [1e-7, 1e-6]
    assert df.index.tz is not None and len(df) == 2 and (df["source"] == "swpc").all()


EPOCH_OFFSET_S = (
    pd.Timestamp("2024-05-11", tz="UTC") - pd.Timestamp("2000-01-01 12:00", tz="UTC")
).total_seconds()


def _write_nc(path, n=1440):
    with h5py.File(path, "w") as f:
        t = f.create_dataset("time", data=np.arange(n) * 60.0 + EPOCH_OFFSET_S)
        t.attrs["units"] = "seconds since 2000-01-01 12:00:00"
        for name, base in (("xrsa_flux", 1e-8), ("xrsb_flux", 1e-6)):
            d = np.full(n, base, dtype="float32")
            d[5] = -9999.0
            ds = f.create_dataset(name, data=d)
            ds.attrs["_FillValue"] = np.float32(-9999.0)


LISTING = (
    '<a href="sci_xrsf-l2-avg1m_g16_d20240510_v2-2-0.nc">x</a>'
    '<a href="sci_xrsf-l2-avg1m_g16_d20240511_v2-2-0.nc">x</a>'
)


def test_ncei_discovery_download_parse_and_idempotency(monkeypatch):
    calls = {"get": [], "dl": []}

    def fake_get(url, **kw):
        calls["get"].append(url)
        return FakeResponse(text=LISTING) if "/goes16/" in url else FakeResponse(404)

    def fake_download(url, dest):
        calls["dl"].append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_nc(dest)
        return True

    gx._ncei_listing.cache_clear()
    monkeypatch.setattr(http, "get", fake_get)
    monkeypatch.setattr(http, "download", fake_download)
    df = gx.fetch("2024-05-11", "2024-05-11")
    assert calls["dl"][0].endswith("sci_xrsf-l2-avg1m_g16_d20240511_v2-2-0.nc")
    assert len(df) == 1440 and df.index.tz is not None
    assert df["xrsb_flux"].isna().sum() == 1  # fill value masked
    assert df["satellite"].iloc[0] == 16
    assert storage.Manifest().has("goes_xrs", "global", "2024-05-11")
    assert gx.fetch("2024-05-11", "2024-05-11").empty and len(calls["dl"]) == 1
    assert len(gx.load("2024-05-11", "2024-05-11")) == 1440
    assert len(gx.resample_5min(df)) == 288


def test_missing_old_day_is_skipped(monkeypatch):
    gx._ncei_listing.cache_clear()
    monkeypatch.setattr(http, "get", lambda url, **kw: FakeResponse(404))
    assert gx.fetch("2015-01-01", "2015-01-01").empty
