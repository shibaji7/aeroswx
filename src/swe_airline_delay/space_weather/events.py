"""Reference event tables: flares, ICMEs, SPEs, and a simple SIR identifier. All free, no key.

========  =====================================================================
flares    NOAA SWPC daily event reports (``XRA`` lines) ->
          ``https://ftp.swpc.noaa.gov/pub/indices/events/{YYYYMMDD}events.txt``;
          recent 7 days also from
          ``services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json``
          (field names unverified).
          M/X filter by default.
ICMEs     Richardson & Cane near-Earth ICME catalogue (HTML table, ACE Science Center).
SPEs      NOAA SPE list -- no plain-table URL confirmed -> **manual-entry CSV**
          ``config/manual/spe_manual.csv`` + :func:`load_spe_manual`.
SIRs      :func:`identify_sir` (solar wind speed >= 600 km/s and Kp >= 5, following the
          Luo et al. 2026 methodology as specified for this project).
========  =====================================================================

TODO (all unverified without live access): the SWPC event-report column layout, the
Richardson & Cane table layout (parsed with regexes, not column positions), and the
SWPC 7-day flare JSON field names. Parsers are written against the documented formats
and covered by synthetic fixtures in ``tests/``; confirm against real files on first run.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from loguru import logger

from swe_airline_delay import http, storage
from swe_airline_delay.config import config_dir
from swe_airline_delay.space_weather.goes_xray import class_to_flux
from swe_airline_delay.timeutil import day_chunks, to_utc

SOURCE = "events"
SWPC_EVENTS_URL = "https://ftp.swpc.noaa.gov/pub/indices/events/{ymd}events.txt"
SWPC_FLARES_7D_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
ICME_URL = "https://www.srl.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm"

FLARE_COLUMNS = [
    "begin_time",
    "peak_time",
    "end_time",
    "flare_class",
    "flux",
    "satellite",
    "region",
]


# --------------------------------------------------------------------------- flares
def parse_swpc_events(text: str, date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Parse a SWPC daily events report, keeping X-ray (``XRA``) events.

    Data line layout (whitespace separated)::

        5170 +  0030 0034 0036  G16 5  XRA  1-8A  B2.1  3.0E-05  3456
        event[+] begin max end  obs  q  type freq  class fluence   region
    """
    if date is None:
        m = re.search(r":Date:\s+(\d{4})\s+(\d{2})\s+(\d{2})", text)
        if not m:
            raise ValueError("no ':Date:' header in SWPC events text and no date given")
        date = pd.Timestamp(year=int(m[1]), month=int(m[2]), day=int(m[3]), tz="UTC")
    date = to_utc(date).normalize()
    rows = []
    for line in text.splitlines():
        if line.startswith(("#", ":")) or "XRA" not in line:
            continue
        tok = line.replace("+", " ").split()
        try:
            i = tok.index("XRA")
        except ValueError:
            continue
        # tok: [event, begin, max, end, obs, q, XRA, freq, class, fluence, (region)]
        begin, peak, end = tok[1], tok[2], tok[3]
        cls = next((t for t in tok[i + 1 :] if re.fullmatch(r"[ABCMX]\d+(\.\d+)?", t)), None)
        if cls is None:
            continue

        def hhmm(s: str, _d=date):
            if not re.fullmatch(r"\d{4}", s):
                return pd.NaT
            return _d + pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))

        region = next((t for t in tok[tok.index(cls) + 1 :] if re.fullmatch(r"\d{4}", t)), None)
        rows.append((hhmm(begin), hhmm(peak), hhmm(end), cls, class_to_flux(cls), tok[4], region))
    return pd.DataFrame(rows, columns=FLARE_COLUMNS)


def parse_swpc_flares_json(records: list[dict]) -> pd.DataFrame:
    """SWPC ``xray-flares-7-day.json`` -> flare table (field names per SWPC docs; TODO verify)."""
    if not records:
        return pd.DataFrame(columns=FLARE_COLUMNS)
    df = pd.DataFrame(records)
    out = pd.DataFrame(
        {
            "begin_time": pd.to_datetime(df.get("begin_time"), utc=True),
            "peak_time": pd.to_datetime(df.get("max_time"), utc=True),
            "end_time": pd.to_datetime(df.get("end_time"), utc=True),
            "flare_class": df.get("max_class"),
            "flux": df.get("max_xrlong"),
            "satellite": df.get("satellite"),
            "region": df.get("active_region", pd.Series([None] * len(df))),
        }
    )
    return out


def fetch_flare_list(start, end, min_class: str = "M", force: bool = False) -> pd.DataFrame:
    """NOAA flare list (class >= ``min_class``) for ``[start, end]``; daily raw cache."""
    manifest = storage.Manifest()
    frames = []
    for day in day_chunks(start, end):
        chunk = day.strftime("%Y-%m-%d")
        if not force and manifest.has(SOURCE + "_flares", "global", chunk):
            continue
        resp = http.get(SWPC_EVENTS_URL.format(ymd=day.strftime("%Y%m%d")))
        if resp.status_code == 404:
            logger.warning("SWPC events {}: not found", chunk)
            continue
        resp.raise_for_status()
        df = parse_swpc_events(resp.text, day)
        path = storage.write_parquet(df, "interim", SOURCE + "_flares", day.year, "global", chunk)
        manifest.record(
            SOURCE + "_flares", "global", chunk, day, day + pd.Timedelta(days=1), len(df), path
        )
        frames.append(df)
    all_ = storage.read_parquet("interim", SOURCE + "_flares", scope="global")
    if not all_.empty:
        lo, hi = to_utc(start), to_utc(end) + pd.Timedelta(days=1)
        all_ = all_[(all_["peak_time"] >= lo) & (all_["peak_time"] < hi)]
    return filter_class(all_, min_class)


def filter_class(df: pd.DataFrame, min_class: str = "M") -> pd.DataFrame:
    if df.empty:
        return df
    thr = class_to_flux(min_class + "1.0")
    return (
        df[df["flare_class"].map(class_to_flux) >= thr]
        .sort_values("peak_time")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- ICMEs
_DT = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*(\d{4})")
_MDT = re.compile(r"(\d{2})/(\d{2})\s*(\d{4})")


def parse_icme_html(html: str) -> pd.DataFrame:
    """Parse the Richardson & Cane table into ``disturbance_time, icme_start, icme_end``.

    Regex-based (robust to column drift): the 1st ``YYYY/MM/DD HHMM`` in a row is the
    shock/disturbance time, the 2nd the ICME start, and the first ``MM/DD HHMM`` after
    that is the ICME end (year inferred; rolled over if before the start).
    TODO: confirm against the live page.
    """
    tables = pd.read_html(io.StringIO(html))
    rows = []
    for tbl in tables:
        for rec in tbl.astype(str).itertuples(index=False):
            text = "  ".join(rec)
            dts = list(_DT.finditer(text))
            if not dts:
                continue

            def ts(m):
                return pd.Timestamp(
                    year=int(m[1]),
                    month=int(m[2]),
                    day=int(m[3]),
                    hour=int(m[4][:2]),
                    minute=int(m[4][2:]),
                    tz="UTC",
                )

            dist = ts(dts[0])
            start = ts(dts[1]) if len(dts) > 1 else pd.NaT
            end = pd.NaT
            tail = text[dts[1].end() :] if len(dts) > 1 else ""
            em = _MDT.search(tail)
            if em and start is not pd.NaT:
                end = pd.Timestamp(
                    year=start.year,
                    month=int(em[1]),
                    day=int(em[2]),
                    hour=int(em[3][:2]),
                    minute=int(em[3][2:]),
                    tz="UTC",
                )
                if end < start:
                    end = end.replace(year=start.year + 1)
            rows.append((dist, start, end))
    return (
        pd.DataFrame(rows, columns=["disturbance_time", "icme_start", "icme_end"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def fetch_icme_catalog(force: bool = False) -> pd.DataFrame:
    path = storage.partition_path(
        "raw", SOURCE, "catalogs", "global", "richardson_cane_icme", ext="html"
    )
    if force or not path.exists():
        resp = http.get(ICME_URL)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text)
    return parse_icme_html(path.read_text(errors="replace"))


# --------------------------------------------------------------------------- SPEs (manual)
def load_spe_manual(path=None) -> pd.DataFrame:
    """Load the manual-entry NOAA SPE table (see header of ``config/manual/spe_manual.csv``)."""
    path = path or config_dir() / "manual" / "spe_manual.csv"
    df = pd.read_csv(path, comment="#")
    for c in ("start_time", "max_time", "end_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


# --------------------------------------------------------------------------- SIRs
def identify_sir(
    omni_wide: pd.DataFrame,
    speed_col: str = "flow_speed",
    kp_col: str = "kp",
    v_min: float = 600.0,
    kp_min: float = 5.0,
    merge_gap: str = "6h",
) -> pd.DataFrame:
    """Flag SIR-like intervals: solar wind speed >= ``v_min`` km/s AND Kp >= ``kp_min``.

    ``omni_wide`` is hourly wide OMNI (UTC index; ``omni.to_wide(long, "1h")``). Consecutive
    flagged samples separated by less than ``merge_gap`` are merged into one event. Returns
    ``start, end, peak_speed, max_kp, n_samples``.
    """
    cols = ["start", "end", "peak_speed", "max_kp", "n_samples"]
    flag = (omni_wide[speed_col] >= v_min) & (omni_wide[kp_col] >= kp_min)
    t = omni_wide.index[flag.to_numpy()]
    if len(t) == 0:
        return pd.DataFrame(columns=cols)
    ev = (pd.Series(t).diff() > pd.Timedelta(merge_gap)).cumsum().to_numpy()
    rows = []
    for k in np.unique(ev):
        idx = t[ev == k]
        sub = omni_wide.loc[idx]
        rows.append((idx[0], idx[-1], sub[speed_col].max(), sub[kp_col].max(), len(idx)))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- CLI entry
def fetch(start=None, end=None, min_class: str = "M") -> dict[str, pd.DataFrame]:
    """Pull every reference table that has a confirmed-format public source.

    Flares require ``start``/``end``; the ICME catalogue is a single page; SPEs come from the
    manual CSV (empty by default).
    """
    out: dict[str, pd.DataFrame] = {}
    if start is not None and end is not None:
        out["flares"] = fetch_flare_list(start, end, min_class)
    else:
        logger.info("events: no --start/--end given, skipping flare list")
    out["icme"] = fetch_icme_catalog()
    storage.write_parquet(out["icme"], "interim", SOURCE, "catalogs", "global", "icme")
    out["spe"] = load_spe_manual()
    return out
