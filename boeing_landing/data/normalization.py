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
    """Read the fixed-bounds yaml.

    Args:
        path: yaml holding a `core:` and an `extended:` mapping of
            channel -> [min, max] (defaults to physical_bounds.yaml).
    Returns:
        The (core, extended) tiers, each as {channel: (min, max)} floats; a tier
        the file omits comes back empty.
    """
    raw = load_yaml(path)
    return tuple({channel: (float(lo), float(hi))
                  for channel, (lo, hi) in (raw.get(tier) or {}).items()}
                 for tier in ("core", "extended"))


_CORE_BOUNDS, _EXTENDED_BOUNDS = load_bounds_file()

# Kept as a public name (data_report and older imports) = the core tier.
PHYSICAL_BOUNDS = _CORE_BOUNDS


def bounds_table(mode) -> dict:
    """The fixed bounds a pipeline selected.

    Args:
        mode: the `build.physical_bounds` value -- 'all' takes both tiers,
            anything else truthy takes the core tier.
    Returns:
        {channel: (min, max)} for the channels that tier fixes; every other
        channel falls back to its train statistic in resolve_bounds.
    """
    if mode == "all":
        return {**_CORE_BOUNDS, **_EXTENDED_BOUNDS}
    return _CORE_BOUNDS


def _encoded_column(df, name: str):
    """The values of one *_sin/*_cos channel, from its source angle.

    Args:
        df: frame holding the source column (ANGLE_ENCODINGS).
        name: the encoded channel to compute.
    Returns:
        The encoded Series.
    Raises:
        SystemExit: the source angle is not in the frame.
    """
    src, fn = ANGLE_ENCODINGS[name]
    if src not in df.columns:
        raise SystemExit(f"cannot encode {name!r}: source column {src!r} is missing")
    return fn(df[src].astype(float))


def add_angle_encodings(df, columns):
    """Derive the sin/cos channels an input set asks for.

    Args:
        df: source frame, left untouched.
        columns: the channels the input set wants; those that need no encoding
            are ignored, so a set keeping the raw heading (gps) changes nothing.
    Returns:
        A frame with the missing *_sin/*_cos columns appended (df itself when
        there is nothing to add).
    """
    encoded = {name: _encoded_column(df, name) for name in columns
               if name in ANGLE_ENCODINGS and name not in df.columns}
    return df.assign(**encoded) if encoded else df


def _column_bounds(train, column: str, table: dict) -> tuple[float, float]:
    """(min, max) of one column: the fixed bound when the table has one for it,
    else the train-split extrema.

    Args:
        train: the training frame.
        column: column to bound.
        table: fixed bounds in play (see bounds_table); empty = data-driven only.
    Returns:
        The (min, max) pair.
    """
    if column in table:
        return table[column]
    return float(train[column].min()), float(train[column].max())


def resolve_bounds(train, columns, physical) -> tuple[list, list]:
    """Min-max normalisation params of several columns.

    Args:
        train: the training frame (bounds are never fit on validation).
        columns: columns to resolve, in order.
        physical: `build.physical_bounds` -- falsy for data-driven bounds only.
    Returns:
        Two lists aligned with `columns`: the mins and the maxes.
    """
    table = bounds_table(physical) if physical else {}
    pairs = [_column_bounds(train, c, table) for c in columns]
    return [lo for lo, _ in pairs], [hi for _, hi in pairs]


def resolve_norm(train, columns, method: str = "minmax", physical=False) -> tuple[list, list]:
    """Normalisation params of several columns, whatever the method.

    The pair is stored in the npz; `normalize` reads `method` back to apply it.

    Args:
        train: the training frame.
        columns: columns to resolve, in order.
        method: 'minmax' or 'zscore'.
        physical: fixed-bounds selector, honoured by minmax only -- a z-score is
            a mean/std rescale, not a min/max one.
    Returns:
        Two lists aligned with `columns`: (min, max) per column for minmax,
        (mean, std) for zscore.
    """
    if method == "zscore":
        return ([float(train[c].mean()) for c in columns],
                [float(train[c].std()) for c in columns])
    return resolve_bounds(train, columns, physical)


def normalize(arr, a, b, method: str = "minmax"):
    """Apply the normalisation whose params resolve_norm computed.

    Args:
        arr: raw values, last axis being the channels.
        a, b: the per-channel params -- (min, max) for minmax, (mean, std) for
            zscore -- broadcast against that last axis.
        method: 'minmax' -> (x - a) / (b - a), roughly [0,1];
            'zscore' -> (x - a) / b, mean 0 and std 1.
    Returns:
        The normalised array. The +1e-10 keeps a channel that is constant on
        train from dividing by zero.
    """
    a, b = np.asarray(a), np.asarray(b)
    if method == "zscore":
        return (arr - a) / (b + 1e-10)
    return (arr - a) / (b - a + 1e-10)
