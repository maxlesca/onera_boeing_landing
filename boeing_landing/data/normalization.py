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
PHYSICAL_BOUNDS = {
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


def resolve_bounds(train, columns, use_physical: bool) -> tuple[list, list]:
    """Per-column (min, max): the fixed physical bound when `use_physical` and the
    column has one, else the train-split min/max. Returns two aligned lists."""
    lo, hi = [], []
    for c in columns:
        if use_physical and c in PHYSICAL_BOUNDS:
            a, b = PHYSICAL_BOUNDS[c]
        else:
            a, b = float(train[c].min()), float(train[c].max())
        lo.append(a)
        hi.append(b)
    return lo, hi


def normalize(arr, lo, hi):
    """Min-max to [0,1]; the +1e-10 guards a channel that is constant on train."""
    return (arr - lo) / (hi - lo + 1e-10)
