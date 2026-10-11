"""BTS On-Time Performance fetcher -- the PRIMARY and ONLY delay/cancellation source.

Data come from the Bureau of Transportation Statistics TranStats pre-zipped
monthly files (free, public domain, **no key**)::

    https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip

.. note::
   BTS publishes with roughly a **1-2 month reporting lag**. This is the
   authoritative *historical* source but not a live feed -- the OpenSky live
   layer in :mod:`swe_airline_delay.adsb.opensky` is the near-real-time
   complement.

Flow: discover URLs -> download zip -> unzip to ``data/raw/bts/{year}/{month}.csv``
(cached; never re-downloaded) -> normalise + filter to the configured airports ->
``data/interim/bts/{year}/{airport}/{YYYY-MM}.parquet`` + manifest entry.

Tidy schema (one row per scheduled departure from a configured airport):

=================  ==========================================================
airport            IATA code of the *origin* airport (the airport "of record")
flight_id          ``YYYY-MM-DD_<carrier><flightno>_<orig>-<dest>_<crsdep>``
carrier            BTS ``Reporting_Airline``
dest               destination IATA
scheduled_dep      UTC, tz-aware (from local CRS time + airport tz)
actual_dep         UTC; ``scheduled_dep + DepDelay`` (NaT if cancelled)
scheduled_arr      UTC; ``scheduled_dep + CRSElapsedTime``
actual_arr         UTC; ``scheduled_arr + ArrDelay`` (NaT if cancelled/diverted)
dep_delay_min      minutes, may be negative (early)
arr_delay_min      minutes, may be negative
cancelled          bool
cancel_code        A carrier / B weather / C NAS / D security (else null)
delay_cause_code   dominant cause of the *arrival* delay: C carrier, W weather,
                   N NAS, S security, L late-aircraft (null if BTS gave none;
                   BTS only attributes delays >= 15 min)
*_delay_min        carrier/weather/nas/security/late_aircraft minutes (raw split)
=================  ==========================================================

Arrival timestamps are derived from elapsed time rather than parsing local
arrival clock times, which avoids needing destination time zones and
midnight-rollover bugs.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from swe_airline_delay import http, storage
from swe_airline_delay.config import data_dir, resolve_airports
from swe_airline_delay.timeutil import to_utc

SOURCE = "bts"
BASE_URL = "https://transtats.bts.gov/PREZIP/"
FILE_PATTERN = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

# BTS columns we need (raw CSV names).
USECOLS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepDelay",
    "CRSElapsedTime",
    "ArrDelay",
    "Cancelled",
    "CancellationCode",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

CAUSE_COLS = {
    "carrier_delay_min": ("CarrierDelay", "C"),
    "weather_delay_min": ("WeatherDelay", "W"),
    "nas_delay_min": ("NASDelay", "N"),
    "security_delay_min": ("SecurityDelay", "S"),
    "late_aircraft_delay_min": ("LateAircraftDelay", "L"),
}

TIDY_COLUMNS = [
    "airport",
    "flight_id",
    "carrier",
    "dest",
    "scheduled_dep",
    "actual_dep",
    "scheduled_arr",
    "actual_arr",
    "dep_delay_min",
    "arr_delay_min",
    "cancelled",
    "cancel_code",
    "delay_cause_code",
    *CAUSE_COLS,
]


# --------------------------------------------------------------------------- URLs / cache
def zip_url(year: int, month: int) -> str:
    return BASE_URL + FILE_PATTERN.format(year=year, month=month)


def discover_urls(start, end) -> list[tuple[int, int, str]]:
    """``[(year, month, url), ...]`` for every month touching ``[start, end]``.

    The PREZIP URL pattern is deterministic, so "discovery" is generation; whether a
    month is actually published yet (reporting lag) is determined at download time.
    """
    return [(y, m, zip_url(y, m)) for y, m in storage.month_chunks(start, end)]


def raw_csv_path(year: int, month: int) -> Path:
    return data_dir() / "raw" / SOURCE / str(year) / f"{month}.csv"


def download_month(year: int, month: int) -> Path | None:
    """Download+unzip one month into the raw cache. Returns the CSV path or None if unpublished."""
    csv_path = raw_csv_path(year, month)
    if csv_path.exists():
        logger.debug("BTS {}-{:02d}: raw CSV cached", year, month)
        return csv_path
    zpath = csv_path.with_suffix(".zip")
    logger.info("BTS {}-{:02d}: downloading {}", year, month, zip_url(year, month))
    if not http.download(zip_url(year, month), zpath):
        logger.warning(
            "BTS {}-{:02d} not available (likely inside the 1-2 month reporting lag)", year, month
        )
        return None
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"no CSV inside {zpath}")
            tmp = csv_path.with_suffix(".csv.part")
            with zf.open(names[0]) as src, open(tmp, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
            tmp.replace(csv_path)
    finally:
        zpath.unlink(missing_ok=True)  # keep only the CSV, as per the raw-cache convention
    return csv_path


# --------------------------------------------------------------------------- normalisation
def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    """BTS ``hhmm`` (e.g. 1435; 2400 == end of day) -> minutes since local midnight."""
    v = pd.to_numeric(s, errors="coerce")
    return (v // 100) * 60 + (v % 100)


def normalize(raw: pd.DataFrame, airports: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw BTS frame -> tidy schema (see module docstring), filtered to ``airports``."""
    airports = resolve_airports("top20") if airports is None else airports
    tz_by_iata = dict(zip(airports["iata"], airports["tz"], strict=True))
    df = raw[raw["Origin"].isin(tz_by_iata)].copy()
    if df.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    date = pd.to_datetime(df["FlightDate"])
    local_naive = date + pd.to_timedelta(_hhmm_to_minutes(df["CRSDepTime"]), unit="min")

    sched = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for iata, tz in tz_by_iata.items():
        m = (df["Origin"] == iata).to_numpy()
        if not m.any():
            continue
        loc = local_naive[m].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        sched.loc[m] = loc.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")

    dep_delay = pd.to_numeric(df["DepDelay"], errors="coerce")
    arr_delay = pd.to_numeric(df["ArrDelay"], errors="coerce")
    elapsed = pd.to_numeric(df["CRSElapsedTime"], errors="coerce")
    cancelled = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0).astype(bool)

    out = pd.DataFrame(index=df.index)
    out["airport"] = df["Origin"]
    out["carrier"] = df["Reporting_Airline"]
    out["dest"] = df["Dest"]
    flightno = pd.to_numeric(df["Flight_Number_Reporting_Airline"], errors="coerce")
    out["flight_id"] = (
        date.dt.strftime("%Y-%m-%d")
        + "_"
        + df["Reporting_Airline"].astype(str)
        + flightno.fillna(0).astype(int).astype(str)
        + "_"
        + df["Origin"].astype(str)
        + "-"
        + df["Dest"].astype(str)
        + "_"
        + pd.to_numeric(df["CRSDepTime"], errors="coerce").fillna(0).astype(int).astype(str)
    )
    out["scheduled_dep"] = sched
    out["actual_dep"] = sched + pd.to_timedelta(dep_delay, unit="min")
    out["scheduled_arr"] = sched + pd.to_timedelta(elapsed, unit="min")
    out["actual_arr"] = out["scheduled_arr"] + pd.to_timedelta(arr_delay, unit="min")
    out["dep_delay_min"] = dep_delay
    out["arr_delay_min"] = arr_delay
    out["cancelled"] = cancelled
    code = df["CancellationCode"].astype("object").where(cancelled, None)
    out["cancel_code"] = code.where(code.isin(list("ABCD")), None)

    causes = pd.DataFrame(
        {name: pd.to_numeric(df[col], errors="coerce") for name, (col, _) in CAUSE_COLS.items()}
    )
    for name in CAUSE_COLS:
        out[name] = causes[name]
    letters = np.array([c for _, c in CAUSE_COLS.values()], dtype=object)
    vals = causes.to_numpy(dtype=float)
    has_cause = np.nan_to_num(vals, nan=0.0).sum(axis=1) > 0
    dominant = letters[np.nanargmax(np.nan_to_num(vals, nan=-1.0), axis=1)]
    out["delay_cause_code"] = pd.Series(np.where(has_cause, dominant, None), index=df.index)
    return out[TIDY_COLUMNS].reset_index(drop=True)


def read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)


# --------------------------------------------------------------------------- public API
def fetch(start, end, airports="top20", force: bool = False) -> pd.DataFrame:
    """Fetch BTS On-Time Performance for ``[start, end]`` (month granularity), idempotently.

    Months already recorded in the manifest are skipped. Returns the concatenation of the
    *newly* normalised months (empty if everything was cached).
    """
    ap = resolve_airports(airports)
    manifest = storage.Manifest()
    new_frames = []
    for year, month, _url in discover_urls(start, end):
        chunk = f"{year}-{month:02d}"
        if not force and all(manifest.has(SOURCE, a, chunk) for a in ap["iata"]):
            logger.info("BTS {}: already in manifest, skipping", chunk)
            continue
        csv = download_month(year, month)
        if csv is None:
            continue
        tidy = normalize(read_raw_csv(csv), ap)
        month_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(1)
        for iata in ap["iata"]:
            sub = tidy[tidy["airport"] == iata]
            path = storage.write_parquet(sub, "interim", SOURCE, year, iata, chunk)
            manifest.record(SOURCE, iata, chunk, month_start, month_end, len(sub), path)
        new_frames.append(tidy)
        logger.info("BTS {}: {} rows across {} airports", chunk, len(tidy), len(ap))
    return (
        pd.concat(new_frames, ignore_index=True)
        if new_frames
        else pd.DataFrame(columns=TIDY_COLUMNS)
    )


def load(start=None, end=None, airports="top20") -> pd.DataFrame:
    """Load already-fetched tidy BTS data from ``data/interim/bts`` (optionally date-clipped)."""
    ap = resolve_airports(airports)
    df = storage.read_parquet("interim", SOURCE, scope=list(ap["iata"]))
    if df.empty:
        return df
    if start is not None:
        df = df[df["scheduled_dep"] >= to_utc(start)]
    if end is not None:
        df = df[df["scheduled_dep"] < to_utc(end) + pd.Timedelta(days=1)]
    return df.reset_index(drop=True)
