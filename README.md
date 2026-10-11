# swe-airline-delay

Data-acquisition, storage, alignment and plotting infrastructure for a study of **space-weather-event (SWE)
impacts on flight delays and cancellations at the top-20 US airports**, with an empirical ADS-B / GNSS-integrity
component.

> **Scope:** data infrastructure only — no statistical or econometric analysis is implemented.
>
> **Free/open only:** every source below is free for research use; nothing is gated behind a paid plan or key.
> [`config/data_sources.yaml`](config/data_sources.yaml) is the project's record of that.

## Setup

```bash
make setup                       # conda env "swe-airline-delay" (Python 3.11) + pip install -e .
conda activate swe-airline-delay
cp .env.example .env             # optional, see "OpenSky credentials"
make test                        # all HTTP mocked, no network
```

Without conda: `pip install -e ".[dev]"` installs the core + dev dependencies; add `.[adsb]` / `.[spacewx]` for the
optional `traffic`/`opensky-api` and `cdasws`/`sunpy` back-ends (imported lazily).

### OpenSky credentials (optional, free)

The OpenSky Network works **anonymously** (slower: small daily request budget, short history window). A *free* account
raises those limits; it never costs money and is **not required**:

1. Register at <https://opensky-network.org> (free) and create an API client in your account page.
2. Put the OAuth2 client credentials in `.env`: `OPENSKY_CLIENT_ID=...`, `OPENSKY_CLIENT_SECRET=...`.

*Historical* ADS-B pulls use the `traffic` library's OpenSky/Impala integration, which additionally needs a free OpenSky
account **approved for research data access**, configured in traffic's own config file (`traffic.conf`, `[opensky]`),
not in `.env`. Requests are chunked by day to stay inside OpenSky's query-size limits.

## Usage

```bash
swe-fetch flights --airports top20 --start 2024-05-01 --end 2024-05-31   # BTS On-Time Performance
swe-fetch adsb    --airports top20 --start 2024-05-10 --end 2024-05-12 --mode historical
swe-fetch adsb    --airports ATL,ORD --mode live                         # current state vectors
swe-fetch omni    --start 2024-05-01 --end 2024-05-31                    # NASA OMNIWeb (add --no-high-res to skip 1-min)
swe-fetch xray    --start 2024-05-01 --end 2024-05-31                    # GOES XRS 1-min flux
swe-fetch events  [--start ... --end ...]                                # flares, ICMEs, SPEs (manual CSV)
swe-fetch all     --start 2024-05-01 --end 2024-05-31                    # flights + omni + xray + events
swe-fetch status                                                         # what has been pulled (manifest)
swe-align --out data/processed/panel.parquet                             # flights x space-weather panel
```

* **Idempotent:** each chunk (BTS month, OMNI month × resolution, GOES day, ADS-B airport-day) is logged in
  `data/data_manifest.yaml` and skipped on re-runs; `--force` re-pulls. Gaps are visible via `swe-fetch status`.
* **Storage convention** (`data/` is gitignored): `data/{raw,interim,processed}/{source}/{year}/{airport_or_global}/{chunk}.parquet`.
  BTS raw CSVs are cached at `data/raw/bts/{year}/{month}.csv`. OMNI is additionally consolidated into one tidy
  long-format, UTC-indexed file `data/interim/omni/omni_long.parquet`.
* `--airports` accepts `top20` or a comma list of IATA codes. `--start/--end` are UTC dates (inclusive).
* Plot demo: `python -m swe_airline_delay.plotting.figures` writes sample figures built from **synthetic** data.
* `notebooks/00_pipeline_smoke_test.ipynb` runs the whole pipeline on 10–12 May 2024 (set `OFFLINE_DEMO=1` to run it
  on synthetic data without network).

### Alignment

`swe_airline_delay.align.build_panel(delay_df, {name: index_frame}, tolerance="1h", direction="nearest")` attaches the
nearest (or `"backward"` = strictly causal) space-weather sample to each flight's scheduled departure on UTC time with
`pandas.merge_asof`. Columns are prefixed (`omni1h_kp`, `xrs_xrsb_flux`, …) and a `*_lag_min` column records the offset.

### Tidy flight schema (`flight_delay/bts.py`)

`airport, flight_id, carrier, dest, scheduled_dep, actual_dep, scheduled_arr, actual_arr, dep_delay_min, arr_delay_min,
cancelled, cancel_code, delay_cause_code` plus the raw BTS cause split (`carrier|weather|nas|security|late_aircraft_delay_min`).
`delay_cause_code` (C/W/N/S/L) is the dominant BTS-attributed cause — essential for separating SWE signal from ordinary
tropospheric weather. Times are UTC; `airport` is the origin. BTS publishes with a **~1–2 month lag**, so it is the
authoritative *historical* source, not a live feed (that is the OpenSky live layer's role).

## Data sources (all free/open — details and licences in `config/data_sources.yaml`)

| Source | Used for | Auth | Terms of use |
|---|---|---|---|
| **BTS TranStats** On-Time Performance | delays / cancellations / BTS cause split | none | US-government public domain — <https://www.bts.gov/privacy-policy-and-terms-of-use> |
| **OpenSky Network** REST + historical (via `traffic`) | ADS-B coverage / message-rate metrics; live state vectors | none, or free registration for higher limits | free for research/non-commercial — <https://opensky-network.org/about/terms-of-use> |
| **NASA OMNIWeb / CDAWeb** (`cdasws`) | IMF Bz/By, solar wind, Kp, Ap, Dst, AE (hourly + 1-min) | none | NASA open data; acknowledge OMNI/SPDF — <https://omniweb.gsfc.nasa.gov/html/ow_data.html> |
| **NOAA SWPC** | real-time GOES XRS flux, flare list / daily event reports | none | US-government public domain — <https://www.weather.gov/disclaimer> |
| **NOAA NCEI/NGDC** GOES XRS L2 archive | historical 1-min X-ray flux | none | US-government public domain — <https://www.ngdc.noaa.gov/ngdcinfo/privacy.html> |
| **Richardson & Cane** ICME catalogue | ICME reference table | none | freely hosted by the ACE Science Center; cite Cane & Richardson (2003), Richardson & Cane (2010) |
| NOAA SPE list | SPE reference table | none | manual-entry CSV (`config/manual/spe_manual.csv`) — no plain-table URL confirmed |
| Madrigal (TEC), ionosondes (foF2) | **not yet implemented** — stubs in `space_weather/ionosphere.py` | free registration | needs source selection (no single CONUS ionosonde network exists) |

## Status / caveats — please read

This code was written **without live access to the data hosts** (the authoring sandbox blocked them). Therefore:

* The pipeline logic, parsers and storage are covered by **mocked** pytest tests (40 tests; Python 3.11/pandas 2 and
  Python 3.13/pandas 3), and the notebook has been executed in *offline/synthetic* mode only. **No live fetch has been run.**
* URLs and field names that could not be confirmed are marked `TODO` in the module docstrings and `last_verified: null`
  in `data_sources.yaml`. Most uncertain: CDAWeb variable names (`space_weather/omni.py::VARIABLES`), the NCEI GOES-XRS
  directory layout, the SWPC daily-event text layout, the Richardson & Cane table layout, and the `traffic`
  `opensky.history(...)` call/`serials` column. Expect to adjust these on first real run.
* `config/airports_top20.yaml` is a **placeholder ranking** (`# TODO: verify against latest FAA ranking`): 19 airports by
  operations plus ANC (kept for high-latitude geomagnetic exposure), so the study set is 20 airports.
* Time handling: BTS local clock times are converted to UTC with each origin airport's IANA zone; arrival times are
  derived from scheduled elapsed time + reported arrival delay.

## Layout

```
config/            airports_top20.yaml, data_sources.yaml, manual/spe_manual.csv
src/swe_airline_delay/
  flight_delay/bts.py          adsb/opensky.py
  space_weather/{omni,goes_xray,events,ionosphere}.py
  storage.py  align.py  cli.py  config.py  http.py  timeutil.py
  plotting/{style.py,swe.mplstyle,figures.py,sample_data.py}
notebooks/00_pipeline_smoke_test.ipynb      tests/      data/ (gitignored)
```

Plot palette: deep blue `#003366`, golden yellow `#FFCC00`, white `#FFFFFF`, light grey `#F0F0F0`.
`plot_timeseries` stacks delay and index in two panels on a shared time axis by default (avoiding a dual-axis chart);
pass `layout="twin"` for the overlaid variant.

## License

MIT (see `LICENSE`).
