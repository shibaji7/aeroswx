"""GOES XRS X-ray flux -- NOAA SWPC (recent / real-time) + NOAA NCEI archive (historical).

Free, no key. Output is 1-min UTC-indexed ``xrsa_flux`` (0.5-4 A) and ``xrsb_flux``
(1-8 A) in W m^-2, stored per day in ``data/interim/goes_xrs/{year}/global/{YYYY-MM-DD}.parquet``.
:func:`resample_5min` gives the 5-min cadence.

Sources
-------
* **NCEI archive** (final, science-quality, GOES-R era 2017+, also reprocessed GOES-13/14/15)::

      https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes{NN}/l2/data/
          xrsf-l2-avg1m_science/{YYYY}/{MM}/sci_xrsf-l2-avg1m_g{NN}_d{YYYYMMDD}_v{x-y-z}.nc

  The file version suffix varies, so the month directory listing is scraped to find the
  file (:func:`ncei_find_file`). TODO: confirm directory layout / satellite coverage with live
  access; satellites are tried in ``_SAT_CANDIDATES`` order.
* **SWPC real-time** (last ~7 days, provisional)::

      https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json

  Used only for days that are not (yet) in the NCEI archive. Provisional days are *not*
  logged in the manifest, so they are refreshed on the next run.

A ``backend="sunpy"`` alternative (``sunpy.net.Fido`` + ``sunpy.timeseries``) is provided
as a best-guess fallback (:func:`fetch_day_fido`); it is untested without live access.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from loguru import logger

from swe_airline_delay import http, storage
from swe_airline_delay.config import data_dir
from swe_airline_delay.timeutil import day_chunks, to_utc

SOURCE = "goes_xrs"
SWPC_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
NCEI_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
FLUX_COLUMNS = ["xrsa_flux", "xrsb_flux"]

# flare class lower bounds, W m^-2 (XRS long channel)
CLASS_THRESHOLDS = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


# --------------------------------------------------------------------------- flare classes
def flux_to_class(flux: float) -> str | None:
    """``2.3e-5 -> "M2.3"``; None for NaN or flux < A-class."""
    if flux is None or not np.isfinite(flux) or flux < CLASS_THRESHOLDS["A"]:
        return None
    for letter in ("X", "M", "C", "B", "A"):
        if flux >= CLASS_THRESHOLDS[letter]:
            return f"{letter}{flux / CLASS_THRESHOLDS[letter]:.1f}"
    return None


def class_to_flux(flare_class: str) -> float:
    """``"M5.2" -> 5.2e-5``."""
    m = re.fullmatch(r"\s*([ABCMX])\s*(\d+(?:\.\d+)?)\s*", str(flare_class))
    if not m:
        return float("nan")
    return float(m.group(2)) * CLASS_THRESHOLDS[m.group(1)]


# --------------------------------------------------------------------------- SWPC
def parse_swpc_json(records: list[dict]) -> pd.DataFrame:
    """SWPC ``xrays-*.json`` records (one per time x energy band) -> wide 1-min frame."""
    if not records:
        return pd.DataFrame(columns=FLUX_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time_tag"], utc=True)
    flux_col = "flux" if "flux" in df else "observed_flux"
    band = df["energy"].map({"0.05-0.4nm": "xrsa_flux", "0.1-0.8nm": "xrsb_flux"})
    wide = (
        df.assign(band=band)
        .dropna(subset=["band"])
        .pivot_table(index="time", columns="band", values=flux_col, aggfunc="mean")
        .reindex(columns=FLUX_COLUMNS)
    )
    wide[wide <= 0] = np.nan
    wide["satellite"] = df.groupby("time")["satellite"].first().reindex(wide.index).to_numpy()
    wide["source"] = "swpc"
    return wide.sort_index()


def fetch_swpc_recent() -> pd.DataFrame:
    resp = http.get(SWPC_URL)
    resp.raise_for_status()
    return parse_swpc_json(resp.json())


# --------------------------------------------------------------------------- NCEI
def _sat_candidates(year: int) -> list[int]:
    return [16, 18, 19, 17] if year >= 2017 else [15, 14, 13]


def ncei_month_url(sat: int, year: int, month: int) -> str:
    return f"{NCEI_BASE}goes{sat}/l2/data/xrsf-l2-avg1m_science/{year}/{month:02d}/"


@lru_cache(maxsize=256)
def _ncei_listing(sat: int, year: int, month: int) -> dict[str, str]:
    """{YYYYMMDD: filename} for one month directory ({} if the directory does not exist)."""
    resp = http.get(ncei_month_url(sat, year, month))
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    files = re.findall(r"(sci_xrsf-l2-avg1m_g\d+_d(\d{8})_v[\d-]+\.nc)", resp.text)
    return {d: f for f, d in dict.fromkeys(files)}


def ncei_find_file(day: pd.Timestamp) -> tuple[int, str] | None:
    """(satellite, URL) of the archive file for ``day`` or None."""
    for sat in _sat_candidates(day.year):
        name = _ncei_listing(sat, day.year, day.month).get(day.strftime("%Y%m%d"))
        if name:
            return sat, ncei_month_url(sat, day.year, day.month) + name
    return None


def _decode_attr(v):
    v = np.asarray(v).ravel()[0] if np.ndim(v) else v
    return v.decode() if isinstance(v, bytes) else v


def parse_ncei_netcdf(path) -> pd.DataFrame:
    """Read an ``xrsf-l2-avg1m_science`` netCDF4 file with h5py -> wide 1-min frame (UTC)."""
    import h5py

    with h5py.File(path, "r") as f:
        t = f["time"][:].astype("float64")
        units = str(_decode_attr(f["time"].attrs.get("units", "seconds since 2000-01-01 12:00:00")))
        epoch = pd.Timestamp(re.search(r"since\s+(.+)$", units).group(1).strip(), tz="UTC")
        cols = {}
        for name in FLUX_COLUMNS:
            if name not in f:
                cols[name] = np.full(t.shape, np.nan)
                continue
            arr = f[name][:].astype("float64")
            fill = f[name].attrs.get("_FillValue")
            if fill is not None:
                arr[arr == float(np.asarray(fill).ravel()[0])] = np.nan
            arr[arr <= 0] = np.nan
            cols[name] = arr
    idx = epoch + pd.to_timedelta(t, unit="s")
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(idx, name="time"))
    df["source"] = "ncei"
    return df.sort_index()


def fetch_day_ncei(day: pd.Timestamp) -> pd.DataFrame | None:
    found = ncei_find_file(day)
    if found is None:
        return None
    sat, url = found
    dest = data_dir() / "raw" / SOURCE / str(day.year) / url.rsplit("/", 1)[1]
    if not dest.exists():
        if not http.download(url, dest):
            return None
    df = parse_ncei_netcdf(dest)
    df["satellite"] = sat
    return df


def fetch_day_fido(
    day: pd.Timestamp,
) -> pd.DataFrame | None:  # pragma: no cover - needs live access
    """Best-guess sunpy/Fido alternative to the direct NCEI download (untested)."""
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from sunpy.timeseries import TimeSeries

    res = Fido.search(a.Time(day, day + pd.Timedelta(days=1)), a.Instrument("XRS"))
    files = Fido.fetch(res, path=str(data_dir() / "raw" / SOURCE / "{file}"))
    ts = TimeSeries(files, concatenate=True)
    df = ts.to_dataframe().rename(columns={"xrsa": "xrsa_flux", "xrsb": "xrsb_flux"})
    df.index = pd.DatetimeIndex(df.index, tz="UTC", name="time")
    df["source"] = "sunpy"
    return df[[c for c in FLUX_COLUMNS + ["source"] if c in df]]


# --------------------------------------------------------------------------- public API
def fetch(start, end, backend: str = "ncei", force: bool = False) -> pd.DataFrame:
    """Fetch 1-min GOES XRS flux for ``[start, end]`` (daily chunks, idempotent)."""
    manifest = storage.Manifest()
    today = pd.Timestamp.now(tz="UTC").normalize()
    swpc_cache: pd.DataFrame | None = None
    out = []
    for day in day_chunks(start, end):
        chunk = day.strftime("%Y-%m-%d")
        if not force and manifest.has(SOURCE, "global", chunk):
            logger.info("GOES XRS {}: in manifest, skipping", chunk)
            continue
        df = fetch_day_fido(day) if backend == "sunpy" else fetch_day_ncei(day)
        final = df is not None
        if df is None and (today - day) <= pd.Timedelta(days=7):
            swpc_cache = fetch_swpc_recent() if swpc_cache is None else swpc_cache
            df = swpc_cache[
                (swpc_cache.index >= day) & (swpc_cache.index < day + pd.Timedelta(days=1))
            ]
            logger.info("GOES XRS {}: using provisional SWPC real-time data", chunk)
        if df is None or df.empty:
            logger.warning(
                "GOES XRS {}: no data available (NCEI lag / outside SWPC 7-day window)", chunk
            )
            continue
        path = storage.write_parquet(df, "interim", SOURCE, day.year, "global", chunk)
        if final:
            manifest.record(SOURCE, "global", chunk, day, day + pd.Timedelta(days=1), len(df), path)
        out.append(df)
    return pd.concat(out).sort_index() if out else pd.DataFrame(columns=FLUX_COLUMNS)


def load(start=None, end=None) -> pd.DataFrame:
    df = storage.read_parquet("interim", SOURCE, scope="global", ignore_index=False)
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if start is not None:
        df = df[df.index >= to_utc(start)]
    if end is not None:
        df = df[df.index < to_utc(end) + pd.Timedelta(days=1)]
    return df


def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """5-min cadence (mean of the 1-min fluxes)."""
    return df[[c for c in FLUX_COLUMNS if c in df]].resample("5min").mean()


# --------------------------------------------------------------------------- flare extraction
def extract_flares(
    flux: pd.DataFrame | pd.Series,
    min_class: str = "C",
    min_separation: str = "30min",
    col: str = "xrsb_flux",
) -> pd.DataFrame:
    """Detect flares (>= ``min_class``) from the long-channel flux series.

    A flare peak is the maximum of the series within +-``min_separation``/2 and above the
    class threshold. Begin = minimum flux in the 60 min before the peak; end = first time
    after the peak when the flux has decayed to halfway between the peak and the begin flux
    (NOAA's end definition), capped at 3 h. Returns ``peak_time, peak_flux, flare_class,
    begin_time, end_time`` (UTC).
    """
    s = flux[col] if isinstance(flux, pd.DataFrame) else flux
    s = s.dropna().sort_index()
    cols = ["peak_time", "peak_flux", "flare_class", "begin_time", "end_time"]
    if s.empty:
        return pd.DataFrame(columns=cols)
    thr = CLASS_THRESHOLDS[min_class.upper()]
    half = pd.Timedelta(min_separation) / 2
    # rolling max over a centred time window
    rmax = s.rolling(2 * half + pd.Timedelta("1min"), center=True, min_periods=1).max()
    peaks = s[(s >= thr) & (s == rmax)]
    # collapse plateaus (equal maxima) keeping the first
    peaks = peaks[~(peaks.index.to_series().diff() < half * 2).fillna(False)]
    rows = []
    for t, f in peaks.items():
        pre = s[(s.index >= t - pd.Timedelta("60min")) & (s.index <= t)]
        t_begin = pre.idxmin()
        f_begin = pre.min()
        post = s[(s.index > t) & (s.index <= t + pd.Timedelta("3h"))]
        decayed = post[post <= f_begin + (f - f_begin) / 2]
        t_end = decayed.index[0] if len(decayed) else post.index[-1] if len(post) else t
        rows.append((t, f, flux_to_class(f), t_begin, t_end))
    return pd.DataFrame(rows, columns=cols)


def crosscheck_flares(
    detected: pd.DataFrame,
    official: pd.DataFrame,
    tolerance: str = "10min",
) -> pd.DataFrame:
    """Annotate detected flares with the nearest NOAA-listed flare within ``tolerance``.

    ``official`` needs ``peak_time`` and ``flare_class`` (see ``events.fetch_flare_list``).
    Adds ``matched`` (bool), ``official_peak_time`` and ``official_class``.
    """
    det = detected.sort_values("peak_time").copy()
    if official is None or official.empty or det.empty:
        det["matched"] = False
        det["official_peak_time"] = pd.NaT
        det["official_class"] = None
        return det
    off = (
        official[["peak_time", "flare_class"]]
        .rename(columns={"peak_time": "official_peak_time", "flare_class": "official_class"})
        .sort_values("official_peak_time")
    )
    off["_ot"] = off["official_peak_time"]
    merged = pd.merge_asof(
        det,
        off,
        left_on="peak_time",
        right_on="official_peak_time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    merged["matched"] = merged["_ot"].notna()
    return merged.drop(columns="_ot")
