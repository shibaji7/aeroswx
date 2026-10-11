"""NASA OMNIWeb data through CDAWeb web services (``cdasws``) -- free, no key.

Datasets
--------
* ``OMNI2_H0_MRG1HR``  -- low-resolution hourly (IMF, solar wind, Kp, Ap, Dst, AE)
* ``OMNI_HRO_1MIN``    -- high-resolution 1-min (IMF, solar wind, AE, SYM-H; no Kp/Ap/Dst)

Output: a single tidy **long-format**, UTC-indexed parquet
(``data/interim/omni/omni_long.parquet``) with columns ``variable, value, resolution``
(``resolution`` in ``{"1h", "1min"}``), assembled from monthly chunks in
``data/interim/omni/{year}/global/{YYYY-MM}_{res}.parquet`` (chunks are skipped when
already in the manifest).

TODO: CDAWeb variable names below are best-guess from the CDAWeb dataset listings and
could not be verified without live access. They live in ``VARIABLES`` so a mismatch is a
one-line fix; ``_standardise`` matches on a case-insensitive *prefix* to tolerate suffixes
such as ``BZ_GSM1800``.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from swe_airline_delay import storage
from swe_airline_delay.timeutil import to_utc

SOURCE = "omni"

# canonical name -> CDAWeb variable name, per dataset
VARIABLES: dict[str, dict[str, str]] = {
    "1h": {
        "bz_gsm": "BZ_GSM",
        "by_gsm": "BY_GSM",
        "flow_speed": "V",
        "proton_density": "N",
        "kp": "KP",
        "ap": "AP",
        "dst": "DST",
        "ae": "AE",
    },
    "1min": {
        "bz_gsm": "BZ_GSM",
        "by_gsm": "BY_GSM",
        "flow_speed": "flow_speed",
        "proton_density": "proton_density",
        "ae": "AE_INDEX",
        "sym_h": "SYM_H",
    },
}
DATASETS = {"1h": "OMNI2_H0_MRG1HR", "1min": "OMNI_HRO_1MIN"}
FILL_THRESHOLD = (
    9999.0  # OMNI fill values are 9999.99, 99999.9, 999.99 ...; real values are below this
)
KP_SCALE = {"1h": 0.1}  # hourly OMNI2 Kp is stored as Kp*10


def get_client():
    """Lazy CDAS client (cdasws is an optional dependency)."""
    from cdasws import CdasWs

    return CdasWs()


def _standardise(ds, resolution: str) -> pd.DataFrame:
    """CDAS xarray.Dataset -> long frame (UTC index; variable, value, resolution)."""
    wanted = VARIABLES[resolution]
    df = ds.to_dataframe() if hasattr(ds, "to_dataframe") else pd.DataFrame(ds)
    df = df.reset_index()
    tcol = next(
        (c for c in df.columns if str(c).lower() in {"epoch", "time", "epoch_1"}), df.columns[0]
    )
    time = pd.to_datetime(df[tcol], utc=True)
    frames = []
    for canon, cdas_name in wanted.items():
        col = next(
            (c for c in df.columns if str(c).upper().startswith(cdas_name.upper()) and c != tcol),
            None,
        )
        if col is None:
            logger.warning(
                "OMNI {}: variable {} ({}) not found in response", resolution, canon, cdas_name
            )
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals.mask(vals.abs() >= FILL_THRESHOLD)
        if canon == "kp" and resolution in KP_SCALE:
            vals = vals * KP_SCALE[resolution]
        frames.append(
            pd.DataFrame({"time": time, "variable": canon, "value": vals, "resolution": resolution})
        )
    if not frames:
        return pd.DataFrame(
            columns=["variable", "value", "resolution"],
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    return out.set_index("time").sort_index()


def fetch_month(client, year: int, month: int, resolution: str) -> pd.DataFrame:
    t0 = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    t1 = t0 + pd.offsets.MonthBegin(1)
    from cdasws.datarepresentation import DataRepresentation  # lazy

    names = list(VARIABLES[resolution].values())
    status, data = client.get_data(
        DATASETS[resolution],
        names,
        t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dataRepresentation=DataRepresentation.XARRAY,
    )
    if status != 200 or data is None:
        logger.warning("OMNI {} {}-{:02d}: CDAS status {}", resolution, year, month, status)
        return _standardise(pd.DataFrame(), resolution)
    return _standardise(data, resolution)


def consolidated_path():
    from swe_airline_delay.config import data_dir

    return data_dir() / "interim" / SOURCE / "omni_long.parquet"


def load(start=None, end=None, resolution: str | None = None) -> pd.DataFrame:
    """Long-format OMNI (UTC index), optionally clipped / limited to one resolution."""
    path = consolidated_path()
    if not path.exists():
        return pd.DataFrame(columns=["variable", "value", "resolution"])
    df = pd.read_parquet(path)
    if resolution:
        df = df[df["resolution"] == resolution]
    if start is not None:
        df = df[df.index >= to_utc(start)]
    if end is not None:
        df = df[df.index < to_utc(end) + pd.Timedelta(days=1)]
    return df


def to_wide(long_df: pd.DataFrame, resolution: str = "1h") -> pd.DataFrame:
    """Pivot long -> wide for one resolution: UTC index, one column per variable."""
    sub = long_df[long_df["resolution"] == resolution]
    return sub.reset_index().pivot_table(
        index="time", columns="variable", values="value", aggfunc="mean"
    )


def fetch(start, end, high_res: bool = True, force: bool = False, client=None) -> pd.DataFrame:
    """Fetch OMNI hourly (+ 1-min) for ``[start, end]``, month chunks, idempotent."""
    manifest = storage.Manifest()
    resolutions = ["1h", "1min"] if high_res else ["1h"]
    for year, month in storage.month_chunks(start, end):
        for res in resolutions:
            chunk = f"{year}-{month:02d}_{res}"
            if not force and manifest.has(SOURCE, "global", chunk):
                logger.info("OMNI {}: in manifest, skipping", chunk)
                continue
            client = client or get_client()
            df = fetch_month(client, year, month, res)
            path = storage.write_parquet(df, "interim", SOURCE, year, "global", chunk)
            t0 = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            manifest.record(SOURCE, "global", chunk, t0, t0 + pd.offsets.MonthEnd(1), len(df), path)
            logger.info("OMNI {}: {} rows", chunk, len(df))
    return consolidate()


def consolidate() -> pd.DataFrame:
    """Merge all monthly chunks into the single long-format parquet and return it."""
    df = storage.read_parquet("interim", SOURCE, scope="global", ignore_index=False)
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.set_index(["variable", "resolution"], append=True).index.duplicated(keep="last")]
    path = consolidated_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
