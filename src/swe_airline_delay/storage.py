"""Storage convention + manifest.

Layout (``data/`` is gitignored)::

    data/{raw|interim|processed}/{source}/{year}/{airport_or_global}/{chunk}.parquet
    data/data_manifest.yaml

* **raw**       -- as-downloaded (BTS CSVs live in ``raw/bts/{year}/{month}.csv``;
                   other sources keep their native files or minimally parsed parquet).
* **interim**   -- cleaned/tidy per-source tables (one parquet per chunk).
* **processed** -- analysis-ready outputs (e.g. the aligned panel).

The manifest logs every pulled chunk (source, date range, rows, timestamp), which
makes re-runs idempotent (``manifest.has(...)`` -> skip) and gaps visible.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from swe_airline_delay.config import data_dir

STAGES = ("raw", "interim", "processed")


def _check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return stage


def partition_dir(stage: str, source: str, year: int | str, scope: str = "global") -> Path:
    return data_dir() / _check_stage(stage) / source / str(year) / scope


def partition_path(
    stage: str, source: str, year: int | str, scope: str, chunk: str, ext: str = "parquet"
) -> Path:
    return partition_dir(stage, source, year, scope) / f"{chunk}.{ext}"


def write_parquet(
    df: pd.DataFrame, stage: str, source: str, year: int | str, scope: str, chunk: str
) -> Path:
    path = partition_path(stage, source, year, scope, chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.part")
    df.to_parquet(tmp)
    tmp.replace(path)
    return path


def read_parquet(
    stage: str,
    source: str,
    year: int | str | None = None,
    scope: str | Iterable[str] | None = None,
    ignore_index: bool = True,
) -> pd.DataFrame:
    """Concatenate all parquet chunks under ``{stage}/{source}`` (optionally filtered).

    Pass ``ignore_index=False`` to keep a (UTC) DatetimeIndex stored in the chunks.
    """
    root = data_dir() / _check_stage(stage) / source
    years = [str(year)] if year is not None else None
    scopes = None
    if scope is not None:
        scopes = {scope} if isinstance(scope, str) else set(scope)
    files = []
    for p in sorted(root.glob("*/*/*.parquet")):
        if years and p.parent.parent.name not in years:
            continue
        if scopes and p.parent.name not in scopes:
            continue
        files.append(p)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=ignore_index)


class Manifest:
    """YAML log of pulled chunks, keyed by ``source|scope|chunk``."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else data_dir() / "data_manifest.yaml"
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as fh:
                loaded = yaml.safe_load(fh) or {}
            self.entries = loaded.get("entries", {})

    @staticmethod
    def key(source: str, scope: str, chunk: str) -> str:
        return f"{source}|{scope}|{chunk}"

    def has(self, source: str, scope: str, chunk: str) -> bool:
        """True if the chunk was pulled AND its output file still exists."""
        e = self.entries.get(self.key(source, scope, chunk))
        return bool(e) and (e.get("path") is None or (data_dir() / e["path"]).exists())

    def record(
        self,
        source: str,
        scope: str,
        chunk: str,
        start,
        end,
        rows: int,
        path: Path | None = None,
        stage: str = "interim",
    ) -> None:
        rel = None
        if path is not None:
            try:
                rel = str(Path(path).resolve().relative_to(data_dir()))
            except ValueError:
                rel = str(path)
        self.entries[self.key(source, scope, chunk)] = {
            "source": source,
            "scope": scope,
            "chunk": chunk,
            "stage": stage,
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
            "rows": int(rows),
            "pulled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "path": rel,
        }
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".yaml.part")
        with open(tmp, "w") as fh:
            yaml.safe_dump({"entries": self.entries}, fh, sort_keys=True)
        tmp.replace(self.path)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.entries.values()))

    def gaps(self, source: str, scope: str, expected_chunks: Iterable[str]) -> list[str]:
        """Chunks in ``expected_chunks`` that have not been pulled for (source, scope)."""
        return [c for c in expected_chunks if not self.has(source, scope, c)]

    def summary(self) -> pd.DataFrame:
        """One row per (source, scope): #chunks, rows, first/last chunk, last pull time."""
        df = self.to_frame()
        if df.empty:
            return df
        g = df.groupby(["source", "scope"])
        out = g.agg(
            chunks=("chunk", "count"),
            rows=("rows", "sum"),
            first=("chunk", "min"),
            last=("chunk", "max"),
            last_pull=("pulled_at", "max"),
        ).reset_index()
        logger.debug("manifest summary:\n{}", out)
        return out


def month_chunks(start, end) -> list[tuple[int, int]]:
    """``[(year, month), ...]`` covering ``[start, end]`` inclusive."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    periods = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")
    return [(p.year, p.month) for p in periods]
