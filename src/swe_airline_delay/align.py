"""Align BTS flight records with space-weather indices on UTC time.

Final pipeline output -- a merge only, no statistics.

For each flight the nearest (or most recent) sample of every index frame is attached
via :func:`pandas.merge_asof`. ``direction="nearest"`` follows the project spec;
use ``"backward"`` for a strictly causal join (never use an index sample later than
the flight's scheduled departure).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from loguru import logger


def _prep_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = (
            out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
        )
        return out.rename_axis("_index_time").reset_index()
    if "time" in df:
        out = df.copy()
        out["_index_time"] = pd.to_datetime(out["time"], utc=True)
        return out.drop(columns="time")
    raise ValueError("index frames need a UTC DatetimeIndex or a 'time' column")


def build_panel(
    delay_df: pd.DataFrame,
    index_frames: Mapping[str, pd.DataFrame] | pd.DataFrame,
    time_col: str = "scheduled_dep",
    tolerance: str | pd.Timedelta = "1h",
    direction: str = "nearest",
) -> pd.DataFrame:
    """Return the analysis-ready panel: one row per flight + matched space-weather columns.

    Parameters
    ----------
    delay_df : tidy BTS frame (``time_col`` must be tz-aware UTC).
    index_frames : ``{name: wide_frame}`` (UTC DatetimeIndex, one column per index). Columns are
        renamed ``{name}_{column}``; a ``{name}_lag_min`` column records
        ``flight time - matched index time`` in minutes. A bare DataFrame is
        treated as ``{"swx": df}``.
    tolerance : max |time difference| for a match (else NaN).
    direction : ``"nearest"`` | ``"backward"`` | ``"forward"``.
    """
    if isinstance(index_frames, pd.DataFrame):
        index_frames = {"swx": index_frames}
    panel = delay_df.copy()
    panel[time_col] = pd.to_datetime(panel[time_col], utc=True)
    panel = panel.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    tol = pd.Timedelta(tolerance)
    for name, frame in index_frames.items():
        idx = _prep_index(frame).sort_values("_index_time")
        idx = idx.rename(columns={c: f"{name}_{c}" for c in idx.columns if c != "_index_time"})
        # merge_asof requires identical datetime resolutions on both keys
        idx["_index_time"] = idx["_index_time"].astype(panel[time_col].dtype)
        merged = pd.merge_asof(
            panel,
            idx,
            left_on=time_col,
            right_on="_index_time",
            direction=direction,
            tolerance=tol,
        )
        merged[f"{name}_lag_min"] = (
            merged[time_col] - merged["_index_time"]
        ).dt.total_seconds() / 60.0
        panel = merged.drop(columns="_index_time")
        n_miss = panel[f"{name}_lag_min"].isna().sum()
        logger.info(
            "align[{}]: {} / {} flights without an index match within {}",
            name,
            n_miss,
            len(panel),
            tol,
        )
    return panel
