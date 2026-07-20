# -*- coding: utf-8 -*-
"""Augment a landing CSV with runway-frame coordinates of the aircraft.

Reads a raw ldg_*.csv (25 Hz landing frames, GPS lat/lon in radians) and a
navigation database (NavDB json: airport -> QFU -> LTP/FTP, FPAP, ...), then
writes a NEW csv -- the inputs are never modified -- with 7 extra columns:

    poi_latitude, poi_longitude, poi_altitude   the landing runway's LTP/FTP
                                                (rad, rad, m -- the aircraft-GPS
                                                representation), the new origin
    poi_course                                  true course of the approach at
                                                the LTP/FTP (rad, from north)
    pos_along, pos_cross, pos_up                aircraft position in the runway
                                                frame at that origin (m)

The runway frame makes every approach look the same wherever the runway is on
Earth and whichever QFU is flown, and its signs match the ILS deviations of
the dataset: `along` points down the runway in the LANDING direction
(negative on final, 0 at the threshold), `cross` points to the LEFT of that
direction (same sign as localizer_error_m, checked corr ~ +1), `up` points up
(same sign as the above-glide deviation, glideslope_error_m, corr ~ +1). The
triad is right-handed. It is the local NED frame at the LTP/FTP rotated
around Down by the approach course (bearing LTP -> FPAP, the alignment point
of the FAS data block), with the lateral and vertical axes mirrored.

The LTP/FTP (Landing Threshold Point / Fictitious Threshold Point) is the
threshold point of the approach itself: every QFU has its own entry in the
database, so 16R and 34L -- the same physical runway flown in opposite
directions -- resolve to different origins and opposite courses.

Rows whose (airport, runway) is missing from the database keep NaN in the new
columns and are reported, never dropped.

    python -m boeing_landing.pipelines.ils_aligned_cfc.augment_ned datasets/ldg_dataset_images.csv \\
        datasets/NavDB_MFS.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from boeing_landing.data.geodesy import (approach_course, geodetic_to_ned,
                                         load_navdb, norm_qfu)

POI_COLUMNS = ["poi_latitude", "poi_longitude", "poi_altitude", "poi_course"]
POS_COLUMNS = ["pos_along", "pos_cross", "pos_up"]


def _course_spin(course: float) -> np.ndarray:
    """NED -> (along, left, up): rotation around Down by the approach course,
    then lateral and vertical axes mirrored to match the ILS deviation signs
    (cross positive LEFT like localizer_error_m, up positive like
    glideslope_error_m). Determinant +1: still right-handed."""
    c, s = np.cos(course), np.sin(course)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _positions(rows: pd.DataFrame, entry: dict) -> np.ndarray:
    """(n, 3) along/cross/up of the rows' GPS in the entry's runway frame: local
    NED at the LTP (pymap3d), then spun about Down by the approach course."""
    lat0, lon0, alt0 = entry["ltp"]
    ned = geodetic_to_ned(rows["latitude"].to_numpy(), rows["longitude"].to_numpy(),
                          rows["altitude"].to_numpy(), lat0, lon0, alt0)
    return ned @ _course_spin(approach_course(entry)).T


def augment(df: pd.DataFrame, navdb: dict) -> tuple[pd.DataFrame, list]:
    """Copy of df with the 7 new columns; also returns the (airport, runway)
    pairs absent from the database (their rows keep NaN)."""
    df = df.copy()
    for col in POI_COLUMNS + POS_COLUMNS:
        df[col] = np.nan
    missing = []
    for (airport, runway), rows in df.groupby(["airport", "runway"]):
        entry = navdb.get((airport.strip(), norm_qfu(runway)))
        if entry is None:
            missing.append((airport, runway, len(rows)))
            continue
        df.loc[rows.index, POI_COLUMNS] = (*entry["ltp"], approach_course(entry))
        df.loc[rows.index, POS_COLUMNS] = _positions(rows, entry)
    return df, missing


def _report(df: pd.DataFrame, missing: list, out: Path) -> None:
    done = df[POS_COLUMNS[0]].notna()
    print(f"{done.sum()}/{len(df)} rows augmented -> {out}")
    for airport, runway, n in missing:
        print(f"  WARNING: {airport} {runway} not in the nav database "
              f"({n} rows left NaN)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="raw ldg_*.csv (read-only)")
    ap.add_argument("navdb", type=Path, help="NavDB json (read-only)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output csv (default: <source>_ned.csv)")
    a = ap.parse_args()
    out = a.output or a.source.with_name(f"{a.source.stem}_ned.csv")
    if out.resolve() in (a.source.resolve(), a.navdb.resolve()):
        raise SystemExit("refusing to overwrite an input file")

    df = pd.read_csv(a.source, sep=";", low_memory=False)
    augmented, missing = augment(df, load_navdb(a.navdb))
    augmented.to_csv(out, sep=";", index=False)
    _report(augmented, missing, out)


if __name__ == "__main__":
    main()
