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
    """The rotation taking local NED to the runway frame.

    Args:
        course: true course of the approach, radians from north.
    Returns:
        The (3, 3) matrix: a spin around Down by `course`, with the lateral and
        vertical axes mirrored so the signs match the ILS deviations (cross
        positive LEFT like localizer_error_m, up positive like
        glideslope_error_m). Determinant +1: still right-handed.
    """
    c, s = np.cos(course), np.sin(course)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _positions(rows: pd.DataFrame, entry: dict) -> np.ndarray:
    """Aircraft positions in one runway's frame.

    Args:
        rows: frames of a single (airport, runway), read for their GPS fix.
        entry: that runway's nav database entry.
    Returns:
        (n, 3) along/cross/up in meters: local NED at the LTP (pymap3d), then
        spun about Down by the approach course.
    """
    lat0, lon0, alt0 = entry["ltp"]
    ned = geodetic_to_ned(rows["latitude"].to_numpy(), rows["longitude"].to_numpy(),
                          rows["altitude"].to_numpy(), lat0, lon0, alt0)
    return ned @ _course_spin(approach_course(entry)).T


def _group_columns(rows: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """The 7 new columns for the frames of one runway.

    Args:
        rows: frames of a single (airport, runway).
        entry: that runway's nav database entry.
    Returns:
        A frame indexed like `rows`, holding the constant LTP/course columns
        and the per-frame runway-frame position.
    """
    origin = np.tile([*entry["ltp"], approach_course(entry)], (len(rows), 1))
    return pd.DataFrame(np.hstack([origin, _positions(rows, entry)]),
                        index=rows.index, columns=POI_COLUMNS + POS_COLUMNS)


def augment(df: pd.DataFrame, navdb: dict) -> tuple[pd.DataFrame, list]:
    """Add the runway-frame coordinates to a landing csv.

    Args:
        df: the raw frames, left untouched.
        navdb: the nav database (geodesy.load_navdb).
    Returns:
        (augmented frame, missing) -- a copy of df carrying the 7 new columns,
        and the (airport, runway, row count) triples absent from the database.
        Their rows keep NaN: a run is reported, never silently dropped here.
    """
    groups = [(airport, runway, rows, navdb.get((airport.strip(), norm_qfu(runway))))
              for (airport, runway), rows in df.groupby(["airport", "runway"])]
    known = [_group_columns(rows, entry) for _, _, rows, entry in groups if entry is not None]
    columns = (pd.concat(known).reindex(df.index) if known
               else pd.DataFrame(np.nan, index=df.index, columns=POI_COLUMNS + POS_COLUMNS))
    missing = [(airport, runway, len(rows))
               for airport, runway, rows, entry in groups if entry is None]
    return df.assign(**{name: columns[name] for name in columns}), missing


def _report(df: pd.DataFrame, missing: list, out: Path) -> None:
    """Print what the augmentation covered.

    Args:
        df: the augmented frame.
        missing: the triples augment returned.
        out: where the csv was written.
    Returns:
        Nothing; a missing runway is a WARNING line, since its rows will be
        dropped later by the build's NaN filter.
    """
    done = df[POS_COLUMNS[0]].notna()
    print(f"{done.sum()}/{len(df)} rows augmented -> {out}")
    for airport, runway, n in missing:
        print(f"  WARNING: {airport} {runway} not in the nav database "
              f"({n} rows left NaN)")


def main() -> None:
    """CLI entrypoint: augment a csv on its own (the pipeline normally goes
    through data/augment.py, which reads the paths from the config).

    Returns:
        Nothing; writes the augmented csv and prints the coverage.
    """
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
