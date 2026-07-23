# -*- coding: utf-8 -*-
"""Load a step-1 npz into Tudor's array format: (samples, features, timesteps).

Drop-in replacement for utils.data.get_data. Our runs are long continuous logs,
while Tudor trained on many short fixed-length trajectories (100k x 200 steps).
So we cut each run into short fixed-length portions (a few seconds, never
crossing a run) -- one portion plays the role of one Tudor trajectory (their
200-step ~2 s trajectory; the default here is ~5 s). The CfC keeps memory
within a portion and resets between portions, exactly like Tudor.

The input channel order is chosen here (not at build time), so the conv-order
study just passes a different `input_order` -- no dataset rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boeing_landing.data.features import CANONICAL_INPUTS, extend_order
from boeing_landing.data.normalization import normalize


def _reorder_inputs(npz, input_order: list[str]):
    """Return X and its (min, max) bounds reordered to `input_order` (extended
    with any dataset-only channels, e.g. extra_columns)."""
    names = [str(n) for n in npz["input_names"]]
    order = extend_order(input_order, names)
    idx = [names.index(n) for n in order]
    return npz["X"][:, idx], npz["x_min"][idx], npz["x_max"][idx]


def _portion_starts(n_frames: int, portion_len: int, stride: int) -> range:
    """Start index of every portion cut from a run of `n_frames`. Single source
    for the cutting contract: _cut_portions and portion_runs both read it, so
    the run ids can never drift out of sync with the tensors."""
    return range(0, n_frames - portion_len + 1, stride)


def _cut_portions(x, y, portion_len: int, stride: int):
    """Slide fixed-length portions over one run. x, y are (features, T_run)."""
    xs, ys = [], []
    for start in _portion_starts(x.shape[1], portion_len, stride):
        xs.append(x[:, start:start + portion_len])
        ys.append(y[:, start:start + portion_len])
    return xs, ys


def portion_runs(npz_path: str | Path, portion_len: int = 125, stride: int = 25) -> np.ndarray:
    """Run id of every portion `load_portions` returns, in the same order.

    Only the run column is read, so this is cheap. It is what lets evaluation
    report one score per validation run instead of one score over all of them
    mixed together -- the difference between "the recipe does not extrapolate"
    and "the recipe has not converged".
    """
    runs = np.load(npz_path, allow_pickle=True)["run"]
    return np.array([run for run in np.unique(runs)
                     for _ in _portion_starts(int((runs == run).sum()), portion_len, stride)],
                    dtype=np.int32)


def _dt_channel(t_run: np.ndarray) -> np.ndarray:
    """Per-frame time step of one run, from the real timestamps (so a skipped
    frame or a variable rate is carried to the CfC). First frame copies the second."""
    dt = np.diff(t_run, prepend=t_run[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    return dt


def _add_noise(X: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Gaussian noise on the normalised inputs, i.i.d. per frame and channel.
    Behavioural cloning only sees the states the expert visited; perturbing the
    inputs covers a thin tube around them. Labels stay untouched -- the target is
    still the command the expert issued in the true state."""
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, std, X.shape).astype(X.dtype)


def load_portions(npz_path: str | Path,
                  input_order: list[str] = CANONICAL_INPUTS,
                  portion_len: int = 125,
                  stride: int = 25,
                  normalized: bool = True,
                  use_dt: bool = False,
                  noise_std: float = 0.0,
                  seed: int = 42):
    """Load one split as (input_array, output_array), each (n_portions, feat, portion_len).

    use_dt appends the raw per-frame time step as a LAST channel (after any
    reordering); Lightning_Model splits it off as the CfC timespans -- the
    conv_cfc baseline recipe. Conservative default: opt-in via the config.

    noise_std > 0 perturbs the normalised inputs (see _add_noise); the caller
    passes it for the training split only. The dt channel is appended afterwards,
    so timespans stay exact.
    """
    npz = np.load(npz_path, allow_pickle=True)
    X, x_min, x_max = _reorder_inputs(npz, input_order)
    Y, y_min, y_max = npz["Y"], npz["y_min"], npz["y_max"]
    method = str(npz["norm_method"]) if "norm_method" in npz else "minmax"
    if normalized:
        X = normalize(X, x_min, x_max, method)
        Y = normalize(Y, y_min, y_max, method)
    if noise_std > 0:
        X = _add_noise(X, noise_std, seed)

    xs, ys = [], []
    for run in np.unique(npz["run"]):
        m = npz["run"] == run
        order = np.argsort(npz["t"][m])
        x_run = X[m][order].T
        if use_dt:
            x_run = np.vstack([x_run, _dt_channel(npz["t"][m][order])[None, :]])
        cx, cy = _cut_portions(x_run, Y[m][order].T, portion_len, stride)
        xs += cx
        ys += cy

    input_array = np.stack(xs).astype(np.float32)
    output_array = np.stack(ys).astype(np.float32)
    noise = f", noise sigma={noise_std}" if noise_std > 0 else ""
    print(f"{Path(npz_path).name}: {len(input_array)} portions of {portion_len} "
          f"({input_array.shape[1]} inputs, {output_array.shape[1]} outputs{noise})")
    return input_array, output_array
