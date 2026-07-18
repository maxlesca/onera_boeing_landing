# -*- coding: utf-8 -*-
"""Runway geometry: published threshold coordinates -> the 4 runway corners.

Corners are returned in the aircraft-GPS representation of the dataset --
latitude/longitude in RADIANS and altitude in meters, exactly like the CSV's
latitude/longitude/altitude channels -- so the network compares like with
like. ECEF is only used internally to apply the metric width/offset geometry.
Corners are named relative to the LANDING direction: thr_* = the threshold
the aircraft crosses, end_* = the opposite end, left/right as seen from the
approaching aircraft. The same physical runway therefore yields different
corner ordering for its two landing directions, which is what the controller
cares about.

Coordinates: OpenStreetMap runway geometry (physical runway ends, WGS84).
Validated against the simulator's own localizer: the axis direction matches
localizer_error_m with correlation -1.0 on every run, and the run-7 touchdown
falls ~300 m past the 16R threshold (the standard aim point). Note that the
sim uses the PHYSICAL end of 16R as its threshold, ignoring the real-world
400 m displaced threshold. Elevations are the published MSL values; the sim's
altitude datum sits ~19 m above them (constant offset, harmless for constant
input channels -- to be pinned down against the sim scene database).
"""

from __future__ import annotations

import numpy as np

# WGS84 ellipsoid
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

# (lat deg, lon deg, elevation m, width m) of each runway end (OSM geometry).
# ZBTJ Tianjin Binhai: 16R/34L 3600x60 m, 16L/34R 3200x45 m.
THRESHOLDS = {
    "ZBTJ": {
        "16R": (39.1406451, 117.3361953, 2.4, 60.0),
        "34L": (39.1113561, 117.3541536, 2.4, 60.0),
        "16L": (39.1414550, 117.3625882, 3.7, 45.0),
        "34R": (39.1154544, 117.3785457, 3.7, 45.0),
    },
}

# Sim-scene calibration: the simulator's centerline sits a constant few meters
# off the OSM centerline (measured on the dataset runs against the sim's own
# localizer_error_m, correlation -1.0). Meters toward the RIGHT of the keyed
# runway's landing direction; the reciprocal end flips the sign automatically.
LATERAL_OFFSETS = {"ZBTJ": {"16R": -5.2, "16L": 2.2}}

# 12 input channels: 4 corners x (lat rad, lon rad, alt m) -- the same
# units/representation as the aircraft latitude/longitude/altitude channels.
CORNERS = [f"{corner}_{field}"
           for corner in ("thr_left", "thr_right", "end_left", "end_right")
           for field in ("lat", "lon", "alt")]


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:
    """WGS84 geodetic (degrees, meters) -> ECEF meters."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - _E2) + h) * np.sin(lat)])


def ecef_to_geodetic(p: np.ndarray) -> np.ndarray:
    """ECEF meters -> WGS84 (lat rad, lon rad, alt m), Bowring's closed form
    (sub-mm accurate near the surface)."""
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    b = _A * (1 - _F)
    u = np.arctan2(z * _A, r * b)
    lat = np.arctan2(z + (_A**2 - b**2) / b * np.sin(u) ** 3,
                     r - _E2 * _A * np.cos(u) ** 3)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.array([lat, lon, r / np.cos(lat) - n])


def _up(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Local ellipsoidal 'up' unit vector, in ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def opposite_end(runway: str) -> str:
    """The reciprocal runway designator (16R -> 34L, 34R -> 16L)."""
    number = (int(runway[:2]) + 17) % 36 + 1
    side = {"L": "R", "R": "L", "C": "C"}.get(runway[2:].strip(), "")
    return f"{number:02d}{side}"


def _lateral_offset(airport: str, runway: str, opposite: str) -> float:
    """Sim-scene centerline offset in this runway's landing frame."""
    offsets = LATERAL_OFFSETS.get(airport, {})
    return offsets[runway] if runway in offsets else -offsets.get(opposite, 0.0)


def runway_corners(airport: str, runway: str) -> np.ndarray:
    """(4, 3) corners of the landing runway as (lat rad, lon rad, alt m),
    ordered as in CORNERS. The width/offset geometry is applied in ECEF."""
    airport, runway = airport.strip(), runway.strip()
    opposite = opposite_end(runway)
    thr_geo = THRESHOLDS[airport][runway]
    end_geo = THRESHOLDS[airport][opposite]
    width = thr_geo[3]

    thr, end = geodetic_to_ecef(*thr_geo[:3]), geodetic_to_ecef(*end_geo[:3])
    along = (end - thr) / np.linalg.norm(end - thr)
    right = np.cross(along, _up(thr_geo[0], thr_geo[1]))  # right of the landing aircraft
    right /= np.linalg.norm(right)
    half = 0.5 * width * right
    corners = np.stack([thr - half, thr + half, end - half, end + half])
    corners += _lateral_offset(airport, runway, opposite) * right
    return np.stack([ecef_to_geodetic(c) for c in corners])


def corner_features(airport: str, runway: str) -> dict[str, float]:
    """The corners as {channel name: value}, ready to append as dataset columns."""
    values = runway_corners(airport, runway).ravel()
    return dict(zip(CORNERS, values.tolist()))
