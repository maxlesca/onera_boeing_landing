# -*- coding: utf-8 -*-
"""Turn the 85-run delivery into the canonical csv the build step reads.

Source files (read-only, in datasets/):
  - all_simulations_complete_with_wind_004s_NED_corrected.csv  (223739 x 42)
  - airports_info_complete.csv                                 (30 runways)

The delivery is already usable: the position comes as a local NED at the landing
threshold, so there is no geodesy step here (contrast with ils_aligned_cfc, which
had to convert lat/lon). All this module does is rename the columns the repo
already knows under another name, and check the delivery against the airport
table. Values are never touched.

The airport table is not a source of features here -- the position is used
exactly as delivered. It is read to verify that every (airport, runway) flown
exists in the navigation database, which is the cheap way to catch a truncated
or mismatched delivery before training on it. It also holds the 4 runway corners
that objective 2 (YOLO) needs.

    make dataset CONFIG=ned_wind_cfc     # runs this step, then builds the npz
"""

from __future__ import annotations

import pandas as pd

# Delivery name -> repo name. Only columns the repo already knows under a
# different name are renamed; everything else keeps the delivered name, so the
# canonical csv stays readable next to the source. Lowercase simulationindex and
# time are the build step's contract (data/build_dataset.clean).
COLUMN_MAP = {
    "SimulationIndex": "simulationindex",
    "Time": "time",
    "Airport": "airport",
    "Runway": "runway",
    # linear wind: the repo's WIND group, and the names the physical bounds use
    "wind_vx": "wind_velocity_x",
    "wind_vy": "wind_velocity_y",
    "wind_vz": "wind_velocity_z",
    # rotational gusts: features.WIND_RATE
    "wind_wx": "wind_rate_x",
    "wind_wy": "wind_rate_y",
    "wind_wz": "wind_rate_z",
}

# Columns dropped on the way out: the ILS errors and the absolute GPS fix are
# not in the input set, and `timestamps` duplicates `Time` exactly.
DROPPED = ["timestamps", "latitude", "longitude", "altitude",
           "localizer_error_DDM", "glideslope_error_DDM",
           "localizer_error_M", "glideslope_error_M"]


def rename_columns(df: pd.DataFrame, column_map: dict = COLUMN_MAP) -> pd.DataFrame:
    """Give the delivery's columns the names the repo already uses.

    Args:
        df: the delivered frame.
        column_map: delivery name -> repo name.
    Returns:
        A renamed copy; values are never touched.
    Raises:
        SystemExit: a source column is absent -- a silent rename miss would
            surface much later as a cryptic build error.
    """
    missing = [c for c in column_map if c not in df.columns]
    if missing:
        raise SystemExit(f"the simulation csv lacks the expected columns {missing}")
    return df.rename(columns=column_map)


def drop_unused(df: pd.DataFrame, columns: list[str] = DROPPED) -> pd.DataFrame:
    """Drop what no input set uses, explicitly rather than implicitly: the
    canonical csv should show what was deliberately left out.

    Args:
        df: the renamed frame.
        columns: columns to drop; those already absent are ignored.
    Returns:
        A copy without them.
    """
    return df.drop(columns=[c for c in columns if c in df.columns])


def runway_key(value) -> str:
    """Canonical runway designator, so '7' and '07' are one key whichever way
    pandas happened to type the column in each of the two csv.

    Args:
        value: the designator as read.
    Returns:
        Two digits plus the L/R/C suffix ('07R'); the text upper-cased and
        stripped when it holds no digit at all.
    """
    text = str(value).strip().upper()
    digits = "".join(c for c in text if c.isdigit())
    suffix = "".join(c for c in text if c.isalpha())
    return f"{int(digits):02d}{suffix}" if digits else text


def airport_keys(df: pd.DataFrame, airport="airport", runway="runway") -> set[tuple[str, str]]:
    """The (airport, runway) pairs a table holds.

    Args:
        df: any table with an airport and a runway column.
        airport, runway: their names in that table.
    Returns:
        The canonicalised pairs, comparable across tables.
    """
    return {(str(a).strip().upper(), runway_key(r))
            for a, r in zip(df[airport], df[runway])}


def missing_runways(sims: pd.DataFrame, airports: pd.DataFrame) -> list[tuple[str, str]]:
    """The cheap check that catches a truncated or mismatched delivery.

    Args:
        sims: the prepared simulation frame.
        airports: the airport table.
    Returns:
        The (airport, runway) pairs flown but absent from the table, sorted.
    """
    return sorted(airport_keys(sims) - airport_keys(airports, "airport", "runway"))


def prepare(sims: pd.DataFrame, airports: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """The contract data/prepare.py dispatches to.

    Args:
        sims: the delivered simulation frame.
        airports: the airport table -- a check only, no column of it enters the
            dataset.
    Returns:
        (canonical frame sorted by run then time, uncovered runways).
    """
    df = drop_unused(rename_columns(sims))
    return df.sort_values(["simulationindex", "time"]).reset_index(drop=True), \
        missing_runways(df, airports)
