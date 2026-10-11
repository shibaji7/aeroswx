import pandas as pd
import pytest
from conftest import FakeResponse

from swe_airline_delay import http
from swe_airline_delay.space_weather import events, ionosphere

SWPC_TEXT = """:Product: 20240511events.txt
:Created: 2024 May 12 0302 UT
:Date: 2024 05 11
#Event    Begin    Max       End  Obs  Q  Type  Loc/Frq   Particulars       Reg#
#-------------------------------------------------------------------------------
 5170 +    0123     0123     0127  G16  5  XRA  1-8A      M5.8    1.2E-02   3664
 5180      0200     0210     0220  G16  5  XRA  1-8A      C2.1    1.0E-03   3664
 5190      0300     0310     0315  G18  5  RBR  245       1000
"""

ICME_HTML = """<html><body><table>
<tr><th>Disturbance</th><th>ICME start</th><th>ICME end</th></tr>
<tr><td>2024/05/10 1706</td><td>2024/05/10 1706</td><td>05/12 1200</td><td>x</td></tr>
<tr><td>2023/12/31 2200</td><td>2024/01/01 0100</td><td>01/02 0600</td><td>x</td></tr>
</table></body></html>"""


def test_parse_swpc_events():
    df = events.parse_swpc_events(SWPC_TEXT)
    assert list(df["flare_class"]) == ["M5.8", "C2.1"]  # RBR line ignored
    assert df.loc[0, "peak_time"] == pd.Timestamp("2024-05-11 01:23", tz="UTC")
    assert df.loc[0, "region"] == "3664"
    assert events.filter_class(df, "M")["flare_class"].tolist() == ["M5.8"]


def test_fetch_flare_list_mocked_and_cached(monkeypatch):
    n = []

    def fake_get(url, **kw):
        n.append(url)
        return FakeResponse(text=SWPC_TEXT) if "20240511" in url else FakeResponse(404)

    monkeypatch.setattr(http, "get", fake_get)
    out = events.fetch_flare_list("2024-05-10", "2024-05-11", min_class="M")
    assert out["flare_class"].tolist() == ["M5.8"]
    events.fetch_flare_list("2024-05-10", "2024-05-11")
    assert len([u for u in n if "20240511" in u]) == 1  # cached by manifest


def test_parse_swpc_flares_json():
    rec = [{"begin_time": "2024-05-11T01:20:00Z", "max_time": "2024-05-11T01:23:00Z", "end_time": "2024-05-11T01:27:00Z",
            "max_class": "M5.8", "max_xrlong": 5.8e-5, "satellite": 16}]  # fmt: skip
    df = events.parse_swpc_flares_json(rec)
    assert df.loc[0, "flare_class"] == "M5.8" and df.loc[0, "peak_time"].tz is not None


def test_parse_icme():
    df = events.parse_icme_html(ICME_HTML)
    assert len(df) == 2
    assert df.loc[0, "icme_start"] == pd.Timestamp("2024-05-10 17:06", tz="UTC")
    assert df.loc[0, "icme_end"] == pd.Timestamp("2024-05-12 12:00", tz="UTC")
    assert df.loc[1, "icme_end"] == pd.Timestamp("2024-01-02 06:00", tz="UTC")


def test_spe_manual(tmp_path):
    f = tmp_path / "spe.csv"
    f.write_text(
        "# comment\nstart_time,max_time,end_time,peak_flux_pfu,noaa_scale\n2024-05-11T02:00,2024-05-11T05:00,2024-05-12T00:00,500,S3\n"
    )
    df = events.load_spe_manual(f)
    assert df.loc[0, "peak_flux_pfu"] == 500 and df["start_time"].dt.tz is not None
    assert events.load_spe_manual().empty  # shipped template has only a header


def test_identify_sir():
    t = pd.date_range("2024-05-01", periods=48, freq="1h", tz="UTC")
    w = pd.DataFrame({"flow_speed": 400.0, "kp": 2.0}, index=t)
    w.loc[t[5:9], ["flow_speed", "kp"]] = [650.0, 5.0]
    w.loc[t[10], ["flow_speed", "kp"]] = [700.0, 6.0]  # within merge gap -> same event
    w.loc[t[30], ["flow_speed", "kp"]] = [610.0, 5.3]  # separate event
    w.loc[t[20], ["flow_speed", "kp"]] = [800.0, 3.0]  # fast but Kp < 5 -> not SIR
    sir = events.identify_sir(w)
    assert len(sir) == 2
    assert (
        sir.loc[0, "start"] == t[5]
        and sir.loc[0, "end"] == t[10]
        and sir.loc[0, "peak_speed"] == 700
    )
    assert events.identify_sir(w.assign(kp=0.0)).empty


def test_ionosphere_stubs_raise():
    with pytest.raises(NotImplementedError):
        ionosphere.fetch_tec("2024-05-10", "2024-05-11")
    with pytest.raises(NotImplementedError):
        ionosphere.fetch_fof2("2024-05-10", "2024-05-11")
