import numpy as np
import pandas as pd
import xarray as xr

from swe_airline_delay import storage
from swe_airline_delay.space_weather import omni


class FakeCdas:
    def __init__(self):
        self.calls = []

    def get_data(self, dataset, variables, t0, t1, **kw):
        self.calls.append((dataset, tuple(variables), t0, t1))
        t = pd.date_range(
            t0, periods=4, freq="1h" if "MRG1HR" in dataset else "1min", tz=None
        ).tz_localize(None)
        if "MRG1HR" in dataset:
            data = {
                "BZ_GSM1800": [-5.0, -10.0, 9999.99, 2.0],
                "BY_GSM1800": [1.0, 2.0, 3.0, 4.0],
                "V1800": [400.0, 650.0, 700.0, 500.0],
                "N1800": [5.0, 6.0, 7.0, 8.0],
                "KP1800": [20, 50, 60, 30],  # Kp*10
                "AP1800": [7, 48, 80, 15],
                "DST1800": [-10, -80, -120, -30],
                "AE1800": [100, 800, 1200, 300],
            }
        else:
            data = {
                "BZ_GSM": [-5.0, -6.0, -7.0, -8.0],
                "BY_GSM": [1.0] * 4,
                "flow_speed": [500.0, 510.0, 99999.9, 530.0],
                "proton_density": [5.0] * 4,
                "AE_INDEX": [100.0] * 4,
                "SYM_H": [-20.0] * 4,
            }
        ds = xr.Dataset(
            {k: ("Epoch", np.array(v, dtype=float)) for k, v in data.items()}, coords={"Epoch": t}
        )
        return 200, ds


def _patch_cdas_enum(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("cdasws.datarepresentation")
    mod.DataRepresentation = types.SimpleNamespace(XARRAY="xarray")
    monkeypatch.setitem(sys.modules, "cdasws", types.ModuleType("cdasws"))
    monkeypatch.setitem(sys.modules, "cdasws.datarepresentation", mod)


def test_fetch_long_format_and_idempotent(monkeypatch):
    _patch_cdas_enum(monkeypatch)
    client = FakeCdas()
    df = omni.fetch("2024-05-10", "2024-05-12", client=client)
    assert len(client.calls) == 2  # hourly + 1-min for the single month
    assert list(df.columns) == ["variable", "value", "resolution"]
    assert df.index.tz is not None and df.index.name == "time"
    assert set(df["resolution"]) == {"1h", "1min"}
    h = df[df["resolution"] == "1h"]
    assert set(h["variable"]) == {
        "bz_gsm",
        "by_gsm",
        "flow_speed",
        "proton_density",
        "kp",
        "ap",
        "dst",
        "ae",
    }
    assert h[h["variable"] == "kp"]["value"].max() == 6.0  # scaled from Kp*10
    assert not ((h["variable"] == "bz_gsm") & (h["value"] > 1000)).any()  # fill value dropped
    assert 99999.9 not in df["value"].to_numpy()
    assert omni.consolidated_path().exists()
    assert storage.Manifest().has("omni", "global", "2024-05_1h")

    omni.fetch("2024-05-10", "2024-05-12", client=client)
    assert len(client.calls) == 2  # no new requests


def test_to_wide_and_load(monkeypatch):
    _patch_cdas_enum(monkeypatch)
    omni.fetch("2024-05-10", "2024-05-12", high_res=False, client=FakeCdas())
    long = omni.load(resolution="1h")
    wide = omni.to_wide(long, "1h")
    assert {"kp", "dst"} <= set(wide.columns) and wide.index.tz is not None
    assert omni.load("2030-01-01", "2030-01-02").empty
