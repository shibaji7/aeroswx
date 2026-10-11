"""OpenSky Network: ADS-B integrity inputs (historical) + near-real-time tracking (live).

Everything here is **free**. Higher rate limits require a *free* OpenSky account
(``OPENSKY_CLIENT_ID`` / ``OPENSKY_CLIENT_SECRET`` in ``.env``); the pipeline works
anonymously at a slower pace. Nothing is gated behind a paid plan.

Two modes
---------
``live``
    Current state vectors inside a bounding box around each airport via the REST
    endpoint ``GET /api/states/all?lamin&lomin&lamax&lomax`` (anonymous allowed).
    A free near-real-time complement to BTS's lagged records. Snapshots are
    appended (one parquet per airport per pull, not idempotent by nature).

``historical``
    Per-day pulls for arbitrary time windows (to be centred on SWE events later),
    via the ``traffic`` library's OpenSky/Impala integration. This needs a free
    OpenSky account *approved for research data access* configured in traffic's
    own config file (see README). Requests are chunked **by day** to respect
    OpenSky query-size limits. Raw state vectors go to ``data/raw/adsb``;
    per-receiver message-rate/coverage-gap metrics to ``data/interim/adsb``.

    TODO (unverified without live access): the exact ``traffic`` call signature and
    whether ``state_vectors_data4`` exposes the ``serials`` (receiver) column in the
    installed version. The code below is a best-guess; if ``serials`` is missing the
    metric falls back to a single pseudo-receiver per airport ("ALL").

Metric
------
:func:`compute_receiver_metrics` -- per (airport, receiver, time window): message
count, message rate (Hz), distinct aircraft, covered-fraction of 10 s sub-bins, and
the longest gap with no messages. No integrity-proxy analysis is attempted here.
"""

from __future__ import annotations

import math
import os
import time as _time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from loguru import logger

from swe_airline_delay import http, storage
from swe_airline_delay.config import load_env, resolve_airports
from swe_airline_delay.timeutil import day_chunks, to_utc

API_BASE = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)

STATE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "lon",
    "lat",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
]

DEFAULT_RADIUS_KM = 100.0


# --------------------------------------------------------------------------- helpers
def airport_bbox(
    lat: float, lon: float, radius_km: float = DEFAULT_RADIUS_KM
) -> tuple[float, float, float, float]:
    """(lamin, lomin, lamax, lomax) box of +-``radius_km`` around a point."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


class OpenSkyClient:
    """Minimal REST client with optional free OAuth2 client-credentials auth."""

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        load_env()
        self.client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID") or None
        self.client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET") or None
        self._token: str | None = None
        self._token_expiry = 0.0

    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _headers(self) -> dict[str, str]:
        if not self.authenticated:
            return {}
        if self._token is None or _time.time() > self._token_expiry - 30:
            resp = http.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            self._token = body["access_token"]
            self._token_expiry = _time.time() + float(body.get("expires_in", 300))
        return {"Authorization": f"Bearer {self._token}"}

    def get_json(self, path: str, params: dict | None = None, max_retries: int = 3):
        """GET with 429 handling (honours ``X-Rate-Limit-Retry-After-Seconds``)."""
        for attempt in range(max_retries + 1):
            resp = http.get(f"{API_BASE}{path}", params=params, headers=self._headers())
            if resp.status_code == 429:
                wait = float(resp.headers.get("X-Rate-Limit-Retry-After-Seconds", 2**attempt * 5))
                logger.warning(
                    "OpenSky rate limit hit (anonymous={}); sleeping {:.0f}s",
                    not self.authenticated,
                    wait,
                )
                _time.sleep(min(wait, 3600))
                continue
            if resp.status_code == 404:  # OpenSky returns 404 for "no data"
                return None
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(
            "OpenSky: rate limit retries exhausted; register for free to raise limits"
        )

    def states_in_bbox(
        self, bbox: tuple[float, float, float, float], time: int | None = None
    ) -> pd.DataFrame:
        lamin, lomin, lamax, lomax = bbox
        params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
        if time is not None:
            params["time"] = int(time)
        body = self.get_json("/states/all", params) or {}
        rows = [r[: len(STATE_COLUMNS)] for r in (body.get("states") or [])]
        df = pd.DataFrame(rows, columns=STATE_COLUMNS)
        df["snapshot_time"] = pd.to_datetime(body.get("time"), unit="s", utc=True)
        for c in ("time_position", "last_contact"):
            df[c] = pd.to_datetime(df[c], unit="s", utc=True)
        df["callsign"] = df["callsign"].str.strip()
        return df


# --------------------------------------------------------------------------- metric
def compute_receiver_metrics(
    msgs: pd.DataFrame,
    window: str = "10min",
    subbin_s: int = 10,
    start=None,
    end=None,
    time_col: str = "time",
    receiver_col: str = "receiver",
    airport_col: str = "airport",
) -> pd.DataFrame:
    """Per-receiver message-rate / coverage-gap metric per time window.

    ``msgs``: one row per received message (or state vector) with ``time`` (UTC), ``airport``,
    optional ``receiver`` and ``icao24``. Returns one row per (airport, receiver, window):

    ================  =============================================================
    n_msgs            messages in the window
    msg_rate_hz       n_msgs / window length
    n_aircraft        distinct ``icao24`` (NaN if column absent)
    coverage_frac     fraction of ``subbin_s``-second sub-bins with >= 1 message
    max_gap_s         longest message-free stretch inside the window (incl. edges)
    ================  =============================================================

    With ``start``/``end`` given, windows with *no* messages are emitted as zeros for
    every (airport, receiver) seen at all, so receiver outages show up as explicit gaps.
    """
    cols = [
        "airport",
        "receiver",
        "window_start",
        "n_msgs",
        "msg_rate_hz",
        "n_aircraft",
        "coverage_frac",
        "max_gap_s",
    ]
    if msgs.empty:
        return pd.DataFrame(columns=cols)
    df = msgs.rename(
        columns={time_col: "time", receiver_col: "receiver", airport_col: "airport"}
    ).copy()
    if "receiver" not in df:
        df["receiver"] = "ALL"
    df["receiver"] = df["receiver"].astype(str)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    win = pd.Timedelta(window)
    win_s = win.total_seconds()
    df["window_start"] = df["time"].dt.floor(window)
    df["_t"] = (df["time"] - df["window_start"]).dt.total_seconds()
    n_sub = max(int(round(win_s / subbin_s)), 1)
    has_ac = "icao24" in df

    rows = []
    for (ap, rx, w), g in df.groupby(["airport", "receiver", "window_start"], sort=True):
        t = np.sort(g["_t"].to_numpy())
        edges = np.concatenate([[0.0], t, [win_s]])
        sub = np.unique(np.minimum((t // subbin_s).astype(int), n_sub - 1)).size
        rows.append(
            (
                ap,
                rx,
                w,
                len(g),
                len(g) / win_s,
                g["icao24"].nunique() if has_ac else np.nan,
                sub / n_sub,
                float(np.diff(edges).max()),
            )
        )
    out = pd.DataFrame(rows, columns=cols)

    if start is not None and end is not None:
        grid = pd.date_range(
            to_utc(start).floor(window), to_utc(end), freq=window, inclusive="left"
        )
        keys = out[["airport", "receiver"]].drop_duplicates()
        full = keys.merge(pd.DataFrame({"window_start": grid}), how="cross")
        out = full.merge(out, on=["airport", "receiver", "window_start"], how="left")
        out["n_msgs"] = out["n_msgs"].fillna(0).astype(int)
        out["msg_rate_hz"] = out["msg_rate_hz"].fillna(0.0)
        out["coverage_frac"] = out["coverage_frac"].fillna(0.0)
        out["max_gap_s"] = out["max_gap_s"].fillna(win_s)
    return out.sort_values(["airport", "receiver", "window_start"]).reset_index(drop=True)


# --------------------------------------------------------------------------- historical
def _history_via_traffic(day: pd.Timestamp, bbox) -> pd.DataFrame:
    """One day of state vectors in ``bbox`` via traffic's OpenSky/Impala integration.

    TODO: verify against the installed ``traffic`` version (selected_columns / bounds API).
    """
    from traffic.data import opensky  # lazy: heavy optional dependency

    lamin, lomin, lamax, lomax = bbox
    result = opensky.history(
        day.to_pydatetime(),
        (day + pd.Timedelta(days=1)).to_pydatetime(),
        bounds=(lomin, lamin, lomax, lamax),  # traffic: (west, south, east, north)
        selected_columns=(
            "time",
            "icao24",
            "lat",
            "lon",
            "geoaltitude",
            "lastposupdate",
            "lastcontact",
            "serials",
        ),
        cached=True,
    )
    if result is None:
        return pd.DataFrame()
    return result.data if hasattr(result, "data") else pd.DataFrame(result)


def _explode_receivers(df: pd.DataFrame, airport: str) -> pd.DataFrame:
    """Normalise a traffic state-vector frame to one row per (message, receiver)."""
    if df.empty:
        return pd.DataFrame(columns=["time", "airport", "icao24", "receiver"])
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["airport"] = airport
    if "serials" in out:
        out = out.explode("serials").rename(columns={"serials": "receiver"})
        out["receiver"] = out["receiver"].astype("object").where(out["receiver"].notna(), "ALL")
    else:
        out["receiver"] = "ALL"
    return out[["time", "airport", "icao24", "receiver"]].reset_index(drop=True)


def fetch_historical(
    start,
    end,
    airports="top20",
    window: str = "10min",
    radius_km: float = DEFAULT_RADIUS_KM,
    history_fn=None,
    force: bool = False,
) -> pd.DataFrame:
    """Chunk by day x airport; store raw + per-receiver metrics; skip chunks in the manifest."""
    history_fn = history_fn or _history_via_traffic
    ap = resolve_airports(airports)
    manifest = storage.Manifest()
    metrics = []
    for day in day_chunks(start, end):
        chunk = day.strftime("%Y-%m-%d")
        for r in ap.itertuples():
            if not force and manifest.has("adsb", r.iata, chunk):
                logger.info("ADS-B {} {}: in manifest, skipping", r.iata, chunk)
                continue
            raw = history_fn(day, airport_bbox(r.lat, r.lon, radius_km))
            msgs = _explode_receivers(raw, r.iata)
            storage.write_parquet(msgs, "raw", "adsb", day.year, r.iata, chunk)
            m = compute_receiver_metrics(
                msgs, window=window, start=day, end=day + pd.Timedelta(days=1)
            )
            path = storage.write_parquet(m, "interim", "adsb", day.year, r.iata, chunk)
            manifest.record("adsb", r.iata, chunk, day, day + pd.Timedelta(days=1), len(m), path)
            metrics.append(m)
            logger.info(
                "ADS-B {} {}: {} raw msgs -> {} metric rows", r.iata, chunk, len(msgs), len(m)
            )
    return pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()


# --------------------------------------------------------------------------- live
def fetch_live(
    airports="top20", radius_km: float = DEFAULT_RADIUS_KM, client: OpenSkyClient | None = None
) -> pd.DataFrame:
    """Snapshot of current state vectors around each airport (one REST call per airport)."""
    client = client or OpenSkyClient()
    if not client.authenticated:
        logger.info("OpenSky: anonymous access (free registration only raises rate limits)")
    ap = resolve_airports(airports)
    manifest = storage.Manifest()
    now = datetime.now(UTC)
    chunk = now.strftime("%Y-%m-%dT%H%M%SZ")
    frames = []
    for r in ap.itertuples():
        df = client.states_in_bbox(airport_bbox(r.lat, r.lon, radius_km))
        df.insert(0, "airport", r.iata)
        df["sensors"] = (
            df["sensors"].astype("object").map(lambda v: list(v) if v is not None else None)
        )
        path = storage.write_parquet(df, "raw", "adsb_live", now.year, r.iata, chunk)
        manifest.record("adsb_live", r.iata, chunk, now, now, len(df), path, stage="raw")
        frames.append(df)
        logger.info("OpenSky live {}: {} aircraft", r.iata, len(df))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch(
    start=None, end=None, airports="top20", mode: str = "historical", **kwargs
) -> pd.DataFrame:
    """CLI entry point. ``mode`` is ``"historical"`` (needs start/end) or ``"live"``."""
    if mode == "live":
        return fetch_live(airports, **kwargs)
    if mode == "historical":
        if start is None or end is None:
            raise ValueError("historical mode needs --start and --end")
        return fetch_historical(start, end, airports, **kwargs)
    raise ValueError(f"mode must be 'historical' or 'live', got {mode!r}")
