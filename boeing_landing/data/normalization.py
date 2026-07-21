# -*- coding: utf-8 -*-
"""Single source of truth for normalisation.

Everything that decides how a raw value becomes a [0,1] model input lives here:
the fixed physical bounds, the circular-angle (sin/cos) encodings, and the
min-max helpers. build_dataset and loader only call into this module -- no
normalisation constant is defined anywhere else.

Two kinds of bounds:
- fixed physical bounds (below): airport-independent, chosen from the approach
  envelope to CONTAIN the data with margin so they never clip and stay stable
  across airports/deliveries. Used when a pipeline sets `build.physical_bounds`.
- data-driven: the train-split min/max, computed for any channel without a fixed
  bound (and for every channel when physical_bounds is off, e.g. gps_cfc).
"""

from __future__ import annotations

import numpy as np

# Names of the sin/cos channels a heading expands into (kept here so the encoding
# and its bounds stay together); features.py imports this to lay out the channels.
HEADING_SINCOS = ["heading_sin", "heading_cos"]

# Circular-angle encodings: encoded name -> (source column, numpy function).
# A true heading wraps at +-pi, so a min-max on it is meaningless; a sin/cos pair
# gives the conv a smooth 2-vector instead. pitch/bank do NOT wrap (bounded, far
# from +-pi) so they stay raw.
ANGLE_ENCODINGS = {
    "heading_sin": ("heading", np.sin),
    "heading_cos": ("heading", np.cos),
}

# Fixed physical bounds (min, max). Position is in metres at the runway threshold;
# attitude in radians. Ranges observed on the ZBTJ data are noted for reference;
# the bounds are the approach envelope (wider), not a dataset fit -- tune here.
#
# Two tiers, selected by build.physical_bounds:
#   true / "core" -> _CORE_BOUNDS only (position + attitude);
#   "all"         -> _CORE_BOUNDS + _EXTENDED_BOUNDS (also wind, velocities, rates).
# The extended tier matters for out-of-distribution robustness: with data-driven
# bounds a held-out run whose wind is outside the train range normalises OUTSIDE
# [0,1] (extrapolation); a fixed operational envelope keeps it in range.
_CORE_BOUNDS = {
    # attitude (rad) -- sourced envelope: pitch approach, bank CS-25 40 deg max
    "pitch": (-0.35, 0.35),   # ~20 deg   (data ~ +-0.12)
    "bank":  (-0.70, 0.70),   # ~40 deg   (data ~ +-0.14)
    "heading_sin": (-1.0, 1.0),   # sin/cos bounded by construction
    "heading_cos": (-1.0, 1.0),
    # runway-frame position (m): threshold origin, runway-aligned axes
    "pos_along": (-15000.0, 500.0),   # along runway   (data -11933 .. -54)
    "pos_cross": (-500.0, 500.0),     # cross track    (data -113 .. 181)
    "pos_up":    (0.0, 1000.0),       # height a/g     (data 21 .. 621)
    # magnetic-frame position (m): geographic axes. North and east SHARE the same
    # bound -- a runway of arbitrary QFU projects its full length onto both, so
    # fixing them symmetrically is what makes the frame airport-independent.
    "pos_north_mag": (-15000.0, 15000.0),
    "pos_east_mag":  (-15000.0, 15000.0),
    "pos_up_mag":    (0.0, 1000.0),
}

# Extended tier (build.physical_bounds: all) -- sourced operational envelope from
# the 737 FCTM / CS-25 (see DOC 8.13), wide enough to contain a held-out run's
# wind/velocities so they normalise inside [0,1] instead of extrapolating.
_EXTENDED_BOUNDS = {
    "wind_velocity_x": (-20.0, 20.0),   # crosswind/tailwind envelope (m/s)
    "wind_velocity_y": (-20.0, 20.0),
    "wind_velocity_z": (-5.0, 5.0),
    "u": (0.0, 120.0),                  # body velocities (m/s)
    "v": (-15.0, 15.0),
    "w": (-10.0, 10.0),
    "northsouth_velocity": (-120.0, 120.0),   # NEU velocities (m/s)
    "eastwest_velocity":   (-120.0, 120.0),
    "vertical_velocity":   (-10.0, 10.0),
    "p": (-0.5, 0.5),                   # body rates (rad/s)
    "q": (-0.5, 0.5),
    "r": (-0.5, 0.5),
}

# Kept as a public name (data_report and older imports) = the core tier.
PHYSICAL_BOUNDS = _CORE_BOUNDS


def bounds_table(mode) -> dict:
    """Fixed-bounds table for build.physical_bounds `mode`: 'all' adds the
    extended tier (wind/velocities/rates); anything else truthy = core only."""
    if mode == "all":
        return {**_CORE_BOUNDS, **_EXTENDED_BOUNDS}
    return _CORE_BOUNDS


def add_angle_encodings(df, columns):
    """Add every *_sin/*_cos channel requested in `columns`, each derived from its
    source angle (ANGLE_ENCODINGS). No-op for channels that need no encoding, so
    input sets that keep the raw heading (gps) are untouched."""
    for name in columns:
        spec = ANGLE_ENCODINGS.get(name)
        if spec and name not in df.columns:
            src, fn = spec
            if src not in df.columns:
                raise SystemExit(f"cannot encode {name!r}: source column {src!r} is missing")
            df[name] = fn(df[src].astype(float))
    return df


def resolve_bounds(train, columns, physical) -> tuple[list, list]:
    """Per-column (min, max): the fixed physical bound when `physical` is truthy
    and the column has one in the selected tier (see bounds_table), else the
    train-split min/max. Returns two aligned lists."""
    table = bounds_table(physical) if physical else {}
    lo, hi = [], []
    for c in columns:
        if c in table:
            a, b = table[c]
        else:
            a, b = float(train[c].min()), float(train[c].max())
        lo.append(a)
        hi.append(b)
    return lo, hi


def resolve_norm(train, columns, method: str = "minmax", physical=False) -> tuple[list, list]:
    """Per-column (a, b) normalisation params for `method`:
    - 'minmax': (min, max) -- the fixed physical bound where `physical` selects one;
    - 'zscore': (mean, std) from the train split (physical is ignored -- z-score is
      a mean/std rescale, not a min/max one).
    The pair is stored in the npz; `normalize` reads `method` back to apply it."""
    if method == "zscore":
        return ([float(train[c].mean()) for c in columns],
                [float(train[c].std()) for c in columns])
    return resolve_bounds(train, columns, physical)


def normalize(arr, a, b, method: str = "minmax"):
    """Apply `method` with the two per-channel params (a, b) from resolve_norm:
    - 'minmax': (x - a) / (b - a)  -> [0,1]      (a=min, b=max);
    - 'zscore': (x - a) / b        -> mean 0, std 1  (a=mean, b=std).
    The +1e-10 guards a channel that is constant on train."""
    a, b = np.asarray(a), np.asarray(b)
    if method == "zscore":
        return (arr - a) / (b + 1e-10)
    return (arr - a) / (b - a + 1e-10)
