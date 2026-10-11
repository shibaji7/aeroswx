"""Small UTC helpers (naive inputs are interpreted as UTC)."""

from __future__ import annotations

import pandas as pd


def to_utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def day_chunks(start, end) -> list[pd.Timestamp]:
    """UTC midnights for every day touching ``[start, end]``."""
    s, e = to_utc(start).normalize(), to_utc(end).normalize()
    return list(pd.date_range(s, e, freq="D"))
