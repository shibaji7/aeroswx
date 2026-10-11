"""Command-line interface.

    swe-fetch flights --airports top20 --start 2024-05-01 --end 2024-05-31
    swe-fetch adsb    --airports top20 --start ... --end ... --mode {historical,live}
    swe-fetch omni    --start ... --end ...
    swe-fetch xray    --start ... --end ...
    swe-fetch events
    swe-fetch all     --start ... --end ...
    swe-fetch status
    swe-align --out data/processed/panel.parquet

Every command is idempotent (chunks already in ``data/data_manifest.yaml`` are skipped)
and logs through loguru.
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from swe_airline_delay import __version__

fetch_app = typer.Typer(help="Fetch raw data (free/open sources only).", no_args_is_help=True)
align_app = typer.Typer(
    help="Align flight delays with space-weather indices.", add_completion=False
)

DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]
Start = typer.Option(..., "--start", formats=DATE_FORMATS, help="UTC start date (YYYY-MM-DD)")
End = typer.Option(..., "--end", formats=DATE_FORMATS, help="UTC end date, inclusive")
Airports = typer.Option(
    "top20", "--airports", help="'top20' or comma list of IATA codes, e.g. ATL,ORD"
)


class Mode(StrEnum):
    historical = "historical"
    live = "live"


def _setup_logging(verbose: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}",
    )


@fetch_app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    _setup_logging(verbose)


@fetch_app.command()
def flights(
    start: datetime = Start, end: datetime = End, airports: str = Airports, force: bool = False
):
    """BTS On-Time Performance (monthly zips; ~1-2 month reporting lag)."""
    from swe_airline_delay.flight_delay import bts

    df = bts.fetch(start, end, airports, force=force)
    logger.info("flights: {} new rows", len(df))


@fetch_app.command()
def adsb(
    start: datetime = typer.Option(None, "--start", formats=DATE_FORMATS),
    end: datetime = typer.Option(None, "--end", formats=DATE_FORMATS),
    airports: str = Airports,
    mode: Mode = typer.Option(Mode.historical, "--mode"),
    force: bool = False,
):
    """OpenSky ADS-B: per-receiver coverage metrics (historical) or live state vectors."""
    from swe_airline_delay.adsb import opensky

    kwargs = {"force": force} if mode == Mode.historical else {}
    df = opensky.fetch(start, end, airports, mode=mode.value, **kwargs)
    logger.info("adsb ({}): {} rows", mode.value, len(df))


@fetch_app.command()
def omni(
    start: datetime = Start,
    end: datetime = End,
    high_res: bool = typer.Option(True, "--high-res/--no-high-res", help="also pull 1-min OMNI"),
    force: bool = False,
):
    """NASA OMNIWeb via CDAWeb (IMF, solar wind, Kp/Ap/Dst/AE)."""
    from swe_airline_delay.space_weather import omni as omni_mod

    df = omni_mod.fetch(start, end, high_res=high_res, force=force)
    logger.info("omni: {} long-format rows total", len(df))


@fetch_app.command()
def xray(start: datetime = Start, end: datetime = End, force: bool = False):
    """GOES XRS X-ray flux (NOAA NCEI archive + SWPC recent)."""
    from swe_airline_delay.space_weather import goes_xray

    df = goes_xray.fetch(start, end, force=force)
    logger.info("xray: {} new 1-min rows", len(df))


@fetch_app.command()
def events(
    start: datetime = typer.Option(None, "--start", formats=DATE_FORMATS),
    end: datetime = typer.Option(None, "--end", formats=DATE_FORMATS),
):
    """Reference event tables: flares (if dates given), ICMEs, SPEs (manual CSV)."""
    from swe_airline_delay.space_weather import events as ev

    out = ev.fetch(start, end)
    for k, v in out.items():
        logger.info("events[{}]: {} rows", k, len(v))


@fetch_app.command("all")
def all_(start: datetime = Start, end: datetime = End, airports: str = Airports):
    """flights + omni + xray + events (ADS-B is separate: it is large and needs OpenSky access)."""
    flights(start, end, airports, False)
    omni(start, end, True, False)
    xray(start, end, False)
    events(start, end)


@fetch_app.command()
def status():
    """Show what has been pulled (from data/data_manifest.yaml)."""
    from swe_airline_delay.storage import Manifest

    s = Manifest().summary()
    typer.echo(s.to_string(index=False) if not s.empty else "manifest is empty")


@align_app.command()
def align(
    out: Path = typer.Option(Path("data/processed/panel.parquet"), "--out"),
    start: datetime = typer.Option(None, "--start", formats=DATE_FORMATS),
    end: datetime = typer.Option(None, "--end", formats=DATE_FORMATS),
    airports: str = Airports,
    tolerance: str = typer.Option("1h", help="max time offset for a match"),
    direction: str = typer.Option("nearest", help="nearest | backward | forward"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build the analysis-ready panel from fetched BTS + OMNI + GOES data."""
    _setup_logging(verbose)
    from swe_airline_delay.align import build_panel
    from swe_airline_delay.flight_delay import bts
    from swe_airline_delay.space_weather import goes_xray
    from swe_airline_delay.space_weather import omni as omni_mod

    flights_df = bts.load(start, end, airports)
    if flights_df.empty:
        raise typer.Exit(code=_fail("no BTS data found; run `swe-fetch flights` first"))
    long = omni_mod.load(start, end)
    frames = {}
    for res in ("1h", "1min"):
        w = omni_mod.to_wide(long, res) if not long.empty else pd.DataFrame()
        if not w.empty:
            frames[f"omni{res}"] = w
    xr = goes_xray.load(start, end)
    if not xr.empty:
        frames["xrs"] = xr[[c for c in goes_xray.FLUX_COLUMNS if c in xr]]
    if not frames:
        logger.warning("no space-weather data found; panel will contain flights only")
    panel = build_panel(flights_df, frames, tolerance=tolerance, direction=direction)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    logger.info("wrote {} rows x {} cols -> {}", len(panel), panel.shape[1], out)


def _fail(msg: str) -> int:
    logger.error(msg)
    return 1


@fetch_app.command()
def version():
    """Print the package version."""
    typer.echo(__version__)
