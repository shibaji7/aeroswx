"""Synthetic data for building/testing figures. NOT real observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swe_airline_delay.config import load_airports


def make_synthetic(
    days: int = 30,
    seed: int = 0,
    n_airports: int = 8,
    airports: list[str] | None = None,
    start: str = "2024-05-01",
    n_events: int = 4,
):
    """Return ``(delay_df, index_df, event_times)`` with the real tidy-schema column names.

    delay_df : hourly-ish flights with ``airport, scheduled_dep, dep_delay_min, cancelled``
    index_df : hourly wide frame (UTC index) with ``kp`` and ``flow_speed``
    event_times : DatetimeIndex of synthetic "events" (Kp spikes)
    """
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * 24, freq="1h", tz="UTC")
    kp = np.clip(
        2 + 1.2 * rng.standard_normal(len(t)).cumsum() / 8 + rng.normal(0, 0.5, len(t)), 0, 9
    )
    margin = min(48, len(t) // 4)
    ev_idx = rng.choice(
        np.arange(margin, len(t) - margin), size=min(n_events, len(t) - 2 * margin), replace=False
    )
    for i in ev_idx:
        kp[i : i + 6] += np.array([2, 3, 3.5, 2.5, 1.5, 1.0])
    kp = np.clip(kp, 0, 9)
    index_df = pd.DataFrame(
        {"kp": kp, "flow_speed": 380 + 40 * kp + rng.normal(0, 25, len(t))}, index=t
    )
    index_df.index.name = "time"

    ap = load_airports()
    airports = ap[ap["iata"].isin(airports)] if airports else ap.head(n_airports)
    rows = []
    for a in airports.itertuples():
        n = len(t) * 3
        ts = t[rng.integers(0, len(t), n)] + pd.to_timedelta(rng.integers(0, 60, n), unit="min")
        k = np.interp(ts.as_unit("ns").asi8, t.as_unit("ns").asi8, kp)
        base = rng.gamma(2.0, 6.0, n) - 8
        delay = base + 1.8 * k + (rng.random(n) < 0.04) * rng.exponential(60, n)
        rows.append(
            pd.DataFrame(
                {
                    "airport": a.iata,
                    "scheduled_dep": ts,
                    "dep_delay_min": delay,
                    "cancelled": rng.random(n) < (0.01 + 0.004 * k),
                }
            )
        )
    delay_df = (
        pd.concat(rows, ignore_index=True).sort_values("scheduled_dep").reset_index(drop=True)
    )
    delay_df.loc[delay_df["cancelled"], "dep_delay_min"] = np.nan
    return delay_df, index_df, t[ev_idx]
