# -*- coding: utf-8 -*-
"""Augment a landing CSV with magnetic-north NED coordinates of the aircraft.

Sibling of the ILS-aligned augmentation (ils_aligned_cfc/augment_ned.py): same geodetic chain, the same origin
(the runway threshold LTP/FTP) and the same GPS-derived position -- only the
horizontal axes differ, pointing to MAGNETIC north instead of down the runway.
The runway frame's axis comes from the ILS localiser (a ground installation); the
magnetic frame's axis is the one an onboard magnetometer defines. Comparing the
two isolates the effect of that azimuth choice (the position source stays GPS in
both -- this is not yet a GPS-free setup).

Reads a raw ldg_*.csv (25 Hz frames, GPS lat/lon in radians) and the NavDB, then
writes a NEW csv -- the inputs are never modified -- with 6 extra columns:

    poi_latitude, poi_longitude, poi_altitude   the LTP/FTP origin (rad, rad, m)
    poi_declination                             magnetic declination used (rad)
    pos_north_mag, pos_east_mag, pos_up_mag     aircraft position in the
                                                magnetic-NED frame at the LTP (m)

The frame is the local NED at the LTP rotated around Down by the magnetic
declination: pos_north_mag / pos_east_mag are the offsets along magnetic north /
magnetic east, pos_up_mag is height above the threshold (up positive, and
numerically identical to the runway frame's pos_up -- a spin about Down leaves
the vertical component untouched).

The declination comes for free from the NavDB: the QFU is the runway's MAGNETIC
bearing, so declination = true_course(LTP->FPAP) - QFU*10deg. It is coarse (the
QFU is rounded to 10deg, so +/-5deg); refine with a WMM/IGRF model later if the
comparison warrants it.

Rows whose (airport, runway) is missing from the database keep NaN in the new
columns and are reported, never dropped.

    python -m boeing_landing.pipelines.magnetic_north_cfc.augment_magnetic \\
        datasets/ldg_dataset_images.csv datasets/NavDB_MFS.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from boeing_landing.data.geodesy import (approach_course, geodetic_to_ned,
                                         load_navdb, norm_qfu, runway_heading)

POI_COLUMNS = ["poi_latitude", "poi_longitude", "poi_altitude", "poi_declination"]
POS_COLUMNS = ["pos_north_mag", "pos_east_mag", "pos_up_mag"]


def magnetic_declination(entry: dict) -> float:
    """The declination the NavDB gives for free: the QFU is the runway's
    MAGNETIC bearing, so the gap to the true approach course is the local
    declination. Coarse to +/-5deg, the QFU being rounded to 10deg.

    Args:
        entry: one nav database entry (ltp, fpap, designator).
    Returns:
        East-positive declination in radians, folded to (-pi, pi]. The fold
        matters: approach_course is already in (-pi, pi] while the QFU bearing
        is in [0, 2pi), so their raw difference can sit near -2pi for the
        high-numbered QFU of a runway pair.
    """
    d = approach_course(entry) - runway_heading(entry["designator"])
    return float((d + np.pi) % (2 * np.pi) - np.pi)


def _declination_spin(decl: float) -> np.ndarray:
    """The rotation taking local NED to the magnetic frame.

    Args:
        decl: magnetic declination in radians, east positive.
    Returns:
        The (3, 3) matrix: a spin around Down by `decl`, so the horizontal axes
        point to magnetic north, with the vertical mirrored to report height
        up-positive (like the runway frame's pos_up).
    """
    c, s = np.cos(decl), np.sin(decl)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, -1.0]])


def _positions(rows: pd.DataFrame, entry: dict) -> np.ndarray:
    """Aircraft positions in one runway's magnetic frame.

    Args:
        rows: frames of a single (airport, runway), read for their GPS fix.
        entry: that runway's nav database entry.
    Returns:
        (n, 3) north_mag/east_mag/up in meters: local NED at the LTP (pymap3d),
        then spun about Down by the magnetic declination.
    """
    lat0, lon0, alt0 = entry["ltp"]
    ned = geodetic_to_ned(rows["latitude"].to_numpy(), rows["longitude"].to_numpy(),
                          rows["altitude"].to_numpy(), lat0, lon0, alt0)
    return ned @ _declination_spin(magnetic_declination(entry)).T


def _group_columns(rows: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """The 6 new columns for the frames of one runway.

    Args:
        rows: frames of a single (airport, runway).
        entry: that runway's nav database entry.
    Returns:
        A frame indexed like `rows`, holding the constant LTP/declination
        columns and the per-frame magnetic-frame position.
    """
    origin = np.tile([*entry["ltp"], magnetic_declination(entry)], (len(rows), 1))
    return pd.DataFrame(np.hstack([origin, _positions(rows, entry)]),
                        index=rows.index, columns=POI_COLUMNS + POS_COLUMNS)


def augment(df: pd.DataFrame, navdb: dict) -> tuple[pd.DataFrame, list]:
    """Add the magnetic-frame coordinates to a landing csv.

    Args:
        df: the raw frames, left untouched.
        navdb: the nav database (geodesy.load_navdb).
    Returns:
        (augmented frame, missing) -- a copy of df carrying the 6 new columns,
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
                    help="output csv (default: <source>_mag.csv)")
    a = ap.parse_args()
    out = a.output or a.source.with_name(f"{a.source.stem}_mag.csv")
    if out.resolve() in (a.source.resolve(), a.navdb.resolve()):
        raise SystemExit("refusing to overwrite an input file")

    df = pd.read_csv(a.source, sep=";", low_memory=False)
    augmented, missing = augment(df, load_navdb(a.navdb))
    augmented.to_csv(out, sep=";", index=False)
    _report(augmented, missing, out)


if __name__ == "__main__":
    main()
