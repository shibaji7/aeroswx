import matplotlib.pyplot as plt
import pandas as pd
import pytest
from typer.testing import CliRunner

from swe_airline_delay import align, cli
from swe_airline_delay.config import load_airports
from swe_airline_delay.plotting import figures, style
from swe_airline_delay.plotting.sample_data import make_synthetic


def test_build_panel_nearest_backward_and_tolerance():
    flights = pd.DataFrame(
        {
            "airport": ["ATL"] * 3,
            "scheduled_dep": pd.to_datetime(
                ["2024-05-11 00:20", "2024-05-11 00:40", "2024-05-11 05:00"], utc=True
            ),
            "dep_delay_min": [1.0, 2.0, 3.0],
        }
    )
    idx = pd.DataFrame(
        {"kp": [5.0, 6.0]}, index=pd.date_range("2024-05-11 00:00", periods=2, freq="1h", tz="UTC")
    )
    nearest = align.build_panel(flights, {"omni": idx}, tolerance="1h")
    assert list(nearest["omni_kp"].iloc[:2]) == [5.0, 6.0] and pd.isna(nearest["omni_kp"].iloc[2])
    assert nearest["omni_lag_min"].iloc[0] == pytest.approx(20.0)
    back = align.build_panel(flights, {"omni": idx}, direction="backward")
    assert list(back["omni_kp"].iloc[:2]) == [5.0, 5.0]
    assert len(nearest) == 3  # all flights kept (left join)
    bare = align.build_panel(flights, idx)  # bare frame -> "swx" prefix
    assert "swx_kp" in bare


def test_figures_run_headless(tmp_path):
    d, i, ev = make_synthetic(days=10, n_airports=4)
    style.use_style()
    figs = [
        figures.plot_timeseries(d, i, "kp")[0],
        figures.plot_timeseries(d, i, "kp", layout="twin")[0],
        figures.plot_superposed_epoch(ev, d)[0],
    ]
    panel = d.assign(t=d["scheduled_dep"].dt.floor("1h")).merge(
        i[["kp"]], left_on="t", right_index=True
    )
    figs.append(figures.plot_dose_response(panel, "kp", bins=5)[0])
    figs.append(figures.plot_dose_response(panel, "kp", bins=[0, 2, 4, 9], metric="cancelled")[0])
    for k, f in enumerate(figs):
        f.savefig(tmp_path / f"f{k}.png")
        assert (tmp_path / f"f{k}.png").stat().st_size > 1000
    plt.close("all")


def test_airport_map_headless():
    a = load_airports().assign(burden=lambda x: x["lat"])
    fig, (ax, inset) = figures.plot_airport_map(a, color_by="burden", draw_features=False)
    fig.canvas.draw()
    fig2, _ = figures.plot_airport_map(a, draw_features=False)
    fig2.canvas.draw()
    plt.close("all")


def test_palette():
    assert style.PALETTE == {
        "deep_blue": "#003366",
        "golden_yellow": "#FFCC00",
        "white": "#FFFFFF",
        "light_grey": "#F0F0F0",
    }


def test_cli_status_and_flights(monkeypatch):
    runner = CliRunner()
    r = runner.invoke(cli.fetch_app, ["status"])
    assert r.exit_code == 0 and "manifest is empty" in r.output

    seen = {}
    from swe_airline_delay.flight_delay import bts

    monkeypatch.setattr(
        bts, "fetch", lambda s, e, a, force=False: seen.update(s=s, e=e, a=a) or pd.DataFrame()
    )
    r = runner.invoke(
        cli.fetch_app,
        ["flights", "--airports", "top20", "--start", "2024-05-11", "--end", "2024-05-12"],
    )
    assert r.exit_code == 0, r.output
    assert seen["a"] == "top20" and seen["s"].year == 2024

    r = runner.invoke(cli.fetch_app, ["flights", "--start", "nonsense", "--end", "2024-05-12"])
    assert r.exit_code != 0


def test_cli_adsb_modes(monkeypatch):
    from swe_airline_delay.adsb import opensky

    calls = []
    monkeypatch.setattr(opensky, "fetch", lambda *a, **k: calls.append((a, k)) or pd.DataFrame())
    runner = CliRunner()
    assert (
        runner.invoke(cli.fetch_app, ["adsb", "--mode", "live", "--airports", "ATL"]).exit_code == 0
    )
    assert calls[0][1]["mode"] == "live"
    assert (
        runner.invoke(
            cli.fetch_app, ["adsb", "--start", "2024-05-11", "--end", "2024-05-11"]
        ).exit_code
        == 0
    )
    assert calls[1][1]["mode"] == "historical"


def test_cli_align_end_to_end(tmp_path, monkeypatch):
    from swe_airline_delay import storage

    t = pd.date_range("2024-05-11", periods=6, freq="1h", tz="UTC")
    flights = pd.DataFrame({"airport": "ATL", "scheduled_dep": t, "dep_delay_min": range(6)})
    storage.write_parquet(flights, "interim", "bts", 2024, "ATL", "2024-05")
    long = pd.DataFrame(
        {"variable": "kp", "value": [3.0] * 6, "resolution": "1h"},
        index=pd.DatetimeIndex(t, name="time"),
    )
    storage.write_parquet(long, "interim", "omni", 2024, "global", "2024-05_1h")
    from swe_airline_delay.space_weather import omni

    omni.consolidate()
    out = tmp_path / "panel.parquet"
    r = CliRunner().invoke(cli.align_app, ["--out", str(out), "--airports", "ATL"])
    assert r.exit_code == 0, r.output
    panel = pd.read_parquet(out)
    assert len(panel) == 6 and (panel["omni1h_kp"] == 3.0).all()


def test_cli_align_without_data_fails():
    r = CliRunner().invoke(cli.align_app, ["--out", "x.parquet"])
    assert r.exit_code == 1
