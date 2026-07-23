# -*- coding: utf-8 -*-
"""Single source of truth for normalisation.

Everything that decides how a raw value becomes a [0,1] model input lives here:
the fixed physical bounds, the circular-angle (sin/cos) encodings, and the
min-max helpers. build_dataset and loader only call into this module -- no
normalisation constant is defined anywhere else.

Two kinds of bounds:
- fixed physical bounds: airport-independent, chosen from the approach envelope
  to CONTAIN the data with margin so they never clip and stay stable across
  airports/deliveries. Used when a pipeline sets `build.physical_bounds`. The
  numbers live in physical_bounds.yaml next to this file -- they are data, not
  code, so they are tuned there and reviewed without reading Python.
- data-driven: the train-split min/max, computed for any channel without a fixed
  bound (and for every channel when physical_bounds is off, e.g. gps_cfc).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.config import load_yaml

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

# The two tiers of fixed bounds, read from physical_bounds.yaml (channel ->
# [min, max]) -- the numbers are data, not code, so they are tuned and reviewed
# in the yaml. Loaded once at import: the table is constant, so no pipeline can
# silently ship bounds other than the ones the yaml documents.
#
# Two tiers, selected by build.physical_bounds:
#   true / "core" -> the `core` tier only (position + attitude);
#   "all"         -> `core` + `extended` (also wind, velocities, rates).
# The extended tier matters for out-of-distribution robustness: with data-driven
# bounds a held-out run whose wind is outside the train range normalises OUTSIDE
# [0,1] (extrapolation); a fixed operational envelope keeps it in range.
BOUNDS_FILE = Path(__file__).with_name("physical_bounds.yaml")


def load_bounds_file(path: Path = BOUNDS_FILE) -> tuple[dict, dict]:
    """The (core, extended) tiers of `path`, each as {channel: (min, max)}."""
    raw = load_yaml(path)
    return tuple({channel: (float(lo), float(hi))
                  for channel, (lo, hi) in (raw.get(tier) or {}).items()}
                 for tier in ("core", "extended"))


_CORE_BOUNDS, _EXTENDED_BOUNDS = load_bounds_file()

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
