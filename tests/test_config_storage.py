import pandas as pd
import pytest

from swe_airline_delay import config, storage


def test_airports_schema():
    df = config.load_airports()
    assert len(df) == 20
    assert {"iata", "icao", "name", "lat", "lon", "tz"} <= set(df.columns)
    assert df["iata"].is_unique
    anc = df[df["iata"] == "ANC"].iloc[0]
    assert "high-latitude" in anc["notes"]
    assert (df["lat"].between(20, 70) & df["lon"].between(-170, -60)).all()


def test_resolve_airports():
    assert len(config.resolve_airports("top20")) == 20
    assert list(config.resolve_airports("atl, ord")["iata"]) == ["ATL", "ORD"]
    with pytest.raises(ValueError):
        config.resolve_airports("XXX")


def test_data_sources_all_free():
    src = config.load_data_sources()
    assert src and all(s["free_for_research"] is True for s in src.values())


def test_month_chunks():
    assert storage.month_chunks("2023-11-15", "2024-02-01") == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_parquet_roundtrip_and_manifest():
    df = pd.DataFrame({"a": [1, 2]})
    p = storage.write_parquet(df, "interim", "bts", 2024, "ATL", "2024-05")
    assert p.name == "2024-05.parquet" and p.parent.name == "ATL" and p.parent.parent.name == "2024"
    assert len(storage.read_parquet("interim", "bts", scope="ATL")) == 2
    assert storage.read_parquet("interim", "bts", scope="ORD").empty

    m = storage.Manifest()
    assert not m.has("bts", "ATL", "2024-05")
    m.record("bts", "ATL", "2024-05", "2024-05-01", "2024-05-31", 2, p)
    m2 = storage.Manifest()  # reload from disk
    assert m2.has("bts", "ATL", "2024-05")
    assert m2.gaps("bts", "ATL", ["2024-05", "2024-06"]) == ["2024-06"]
    s = m2.summary()
    assert s.loc[0, "rows"] == 2
    p.unlink()  # file gone -> not considered pulled
    assert not storage.Manifest().has("bts", "ATL", "2024-05")


def test_bad_stage():
    with pytest.raises(ValueError):
        storage.partition_dir("bronze", "x", 2024)
