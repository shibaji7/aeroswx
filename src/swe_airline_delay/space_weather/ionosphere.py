"""Ionospheric context (TEC, foF2) -- STUB, source selection pending.

Both functions deliberately raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import pandas as pd


def fetch_tec(start, end, airports="top20") -> pd.DataFrame:
    """Fetch GNSS vertical TEC near the study airports.

    TODO (decisions needed before implementing):
      * Source: Madrigal (free/open, ``madrigalWeb`` client, free name/email registration)
        -- e.g. the MIT Haystack global GNSS TEC product ("gps TEC", 1 deg x 1 deg,
        5-min). Alternatives: NOAA SWPC US-TEC, IGS GIM maps (IONEX, CDDIS needs a free
        Earthdata login).
      * Spatial handling: nearest grid cell to each airport vs. a regional mean; pierce-
        point geometry vs. pre-gridded vertical TEC.
      * Cadence/latency and whether to store the grid (xarray/HDF5) or only per-airport series.
    """
    raise NotImplementedError("fetch_tec: TEC source (Madrigal vs. alternatives) not yet selected")


def fetch_fof2(start, end, airports="top20") -> pd.DataFrame:
    """Fetch ionosonde foF2 for CONUS.

    TODO (decisions needed before implementing):
      * There is no single CONUS ionosonde/Digisonde network equivalent to the GIRO
        coverage used by Luo et al. in China. Candidates: GIRO/DIDBase (free, registration
        for some data), NOAA NCEI SPIDR-successor archives, individual stations
        (e.g. Boulder, Wallops, Eglin, Austin) -- station selection and their distance
        to each airport must be decided, plus ARTIST autoscaling quality filtering (confidence
        score / manual vs. auto-scaled values).
      * Cadence (5-15 min) and how to handle missing stations for airports far from any sounder.
    """
    raise NotImplementedError("fetch_foF2: CONUS ionosonde network/station selection pending")
