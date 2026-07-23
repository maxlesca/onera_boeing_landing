# -*- coding: utf-8 -*-
"""Geodetic helpers shared by the local-frame augmentations.

A raw GPS fix travels through this chain before it becomes a local Cartesian
position:

    geodetic (lat, lon, h)  --(1)+(2)-->  local NED  --(3)-->  frame
       WGS84, not Cartesian             at a reference     (runway- or
                                        point (lat0,lon0)   magnetic-aligned)

Steps (1) geodetic->ECEF and (2) ECEF->NED are done by pymap3d (tested, WGS84 by
default) -- `geodetic_to_ned` and `bearing` are thin radians-in wrappers. Step
(3) -- the spin about Down by an azimuth -- is what distinguishes the runway- and
magnetic-frame pipelines, so it lives with each of them, not here; no library
provides it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pymap3d as pm


def geodetic_to_ned(lat, lon, h, lat0, lon0, h0) -> np.ndarray:
    """Geodetic coordinates expressed in the local NED frame of a reference point.

    Wraps pymap3d.geodetic2ned (steps 1 and 2 above).

    Args:
        lat, lon: WGS84 latitude/longitude in radians, scalars or arrays.
        h: ellipsoidal height in meters, broadcastable with lat/lon.
        lat0, lon0, h0: the reference point the frame is centred on (rad, rad, m).
    Returns:
        Array of shape (..., 3): north, east and down offsets in meters.
    """
    n, e, d = pm.geodetic2ned(lat, lon, h, lat0, lon0, h0, deg=False)
    return np.stack([n, e, d], axis=-1)


def bearing(lat0, lon0, h0, lat, lon, h) -> float:
    """True course from one geodetic point to another.

    Args:
        lat0, lon0, h0: origin of the course (rad, rad, m).
        lat, lon, h: point aimed at (rad, rad, m).
    Returns:
        The azimuth of pymap3d.geodetic2aer in radians from north, folded to
        (-pi, pi].
    """
    az, _, _ = pm.geodetic2aer(lat, lon, h, lat0, lon0, h0, deg=False)
    return float((az + np.pi) % (2 * np.pi) - np.pi)


def norm_qfu(qfu: str) -> str:
    """Runway designator stripped of its zero padding, so the two spellings the
    database mixes ('07R' and '7R') resolve to one key.

    Args:
        qfu: designator as written in a csv or in the nav database.
    Returns:
        The digits without leading zeros, plus the L/R/C suffix if any.
    Raises:
        SystemExit: the designator does not parse.
    """
    m = re.fullmatch(r"0*(\d+)\s*([LRC]?)", str(qfu).strip().upper())
    if not m:
        raise SystemExit(f"unparseable runway designator: {qfu!r}")
    return m.group(1) + m.group(2)


def runway_heading(designator: str) -> float:
    """The magnetic bearing a runway number encodes: '02' -> 020deg,
    '34L' -> 340deg. The QFU is a magnetic heading rounded to 10deg.

    Args:
        designator: runway designator, padded or not, with or without suffix.
    Returns:
        That bearing in radians.
    Raises:
        SystemExit: the designator does not parse.
    """
    m = re.fullmatch(r"0*(\d+)\s*[LRC]?", str(designator).strip().upper())
    if not m:
        raise SystemExit(f"unparseable runway designator: {designator!r}")
    return np.radians(int(m.group(1)) * 10.0)


def load_navdb(path: Path) -> dict[tuple[str, str], dict]:
    """Flatten the NavDB json into one entry per approach.

    Args:
        path: NavDB json, nested airport -> QFU -> runway record (degrees).
    Returns:
        {(airport, normalised qfu): {"ltp": (lat rad, lon rad, alt m),
        "fpap": (lat rad, lon rad), "designator": the qfu as spelled}}.
    """
    db = json.loads(path.read_text())
    return {(airport.strip(), norm_qfu(qfu)):
            {"ltp": (np.radians(e["lat_ltp_ftp"]), np.radians(e["long_ltp_ftp"]),
                     float(e["alt_ltp_ftp"])),
             "fpap": (np.radians(e["lat_fpap"]), np.radians(e["long_fpap"])),
             "designator": str(qfu)}
            for airport, runways in db.items() for qfu, e in runways.items()}


def approach_course(entry: dict) -> float:
    """Direction the aircraft travels at the threshold, whichever QFU is flown:
    the FPAP is the alignment point of the approach, so the LTP -> FPAP bearing
    is the approach axis itself.

    Args:
        entry: one nav database entry (see load_navdb), read for ltp and fpap.
    Returns:
        That true course in radians from north.
    """
    lat0, lon0, alt0 = entry["ltp"]
    lat_f, lon_f = entry["fpap"]
    return bearing(lat0, lon0, alt0, lat_f, lon_f, alt0)
