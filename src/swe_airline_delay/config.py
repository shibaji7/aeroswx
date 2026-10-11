"""Project paths and config-file loaders."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    """Repo root: ``$SWE_ROOT`` if set, else the first parent that contains ``config/``."""
    env = os.environ.get("SWE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "airports_top20.yaml").exists():
            return parent
    return Path.cwd()


def data_dir() -> Path:
    return Path(os.environ.get("SWE_DATA_DIR", project_root() / "data")).resolve()


def config_dir() -> Path:
    return project_root() / "config"


def load_env() -> None:
    """Load ``.env`` (if present). Only free OpenSky credentials live there."""
    load_dotenv(project_root() / ".env", override=False)


@lru_cache(maxsize=1)
def _airports_raw() -> tuple[dict, ...]:
    with open(config_dir() / "airports_top20.yaml") as fh:
        return tuple(yaml.safe_load(fh)["airports"])


def load_airports() -> pd.DataFrame:
    """All configured airports as a DataFrame (iata, icao, name, lat, lon, tz, notes)."""
    df = pd.DataFrame(list(_airports_raw()))
    if "notes" not in df:
        df["notes"] = None
    return df


def resolve_airports(spec: str | list[str] | None = "top20") -> pd.DataFrame:
    """``"top20"`` -> every configured airport; otherwise a comma list of IATA codes."""
    df = load_airports()
    if spec is None:
        return df
    if isinstance(spec, str):
        if spec.lower() in {"top20", "all"}:
            return df
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    codes = [s.upper() for s in spec]
    unknown = sorted(set(codes) - set(df["iata"]))
    if unknown:
        raise ValueError(f"Unknown airport code(s) {unknown}; known: {sorted(df['iata'])}")
    return df[df["iata"].isin(codes)].reset_index(drop=True)


def load_data_sources() -> dict:
    with open(config_dir() / "data_sources.yaml") as fh:
        return yaml.safe_load(fh)["sources"]
