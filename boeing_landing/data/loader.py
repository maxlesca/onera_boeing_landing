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
    """Permute the input channels of a split to the requested order.

    Args:
        npz: the loaded npz (X, input_names, x_min, x_max).
        input_order: the named channel order, extended here with any
            dataset-only channel (e.g. extra_columns).
    Returns:
        (X, x_min, x_max), all three permuted the same way so a bound never
        drifts away from its channel.
    """
    names = [str(n) for n in npz["input_names"]]
    order = extend_order(input_order, names)
    idx = [names.index(n) for n in order]
    return npz["X"][:, idx], npz["x_min"][idx], npz["x_max"][idx]


def _portion_starts(n_frames: int, portion_len: int, stride: int) -> range:
    """Single source for the cutting contract: _cut_portions and portion_runs
    both read it, so the run ids can never drift out of sync with the tensors.

    Args:
        n_frames: length of the run being cut.
        portion_len: frames per portion.
        stride: step between two consecutive starts.
    Returns:
        The start index of every whole portion (empty when the run is shorter
        than one portion -- such a run contributes nothing).
    """
    return range(0, n_frames - portion_len + 1, stride)


def _cut_portions(x, y, portion_len: int, stride: int) -> list[tuple]:
    """Slide fixed-length portions over one run.

    Args:
        x, y: that run's inputs and labels, both (features, T_run).
        portion_len: frames per portion.
        stride: step between two consecutive portions (< portion_len overlaps).
    Returns:
        [(x_portion, y_portion), ...], each (features, portion_len) and each a
        view into `x`/`y` -- no copy until the caller stacks them.
    """
    return [(x[:, start:start + portion_len], y[:, start:start + portion_len])
            for start in _portion_starts(x.shape[1], portion_len, stride)]


def portion_runs(npz_path: str | Path, portion_len: int = 125, stride: int = 25) -> np.ndarray:
    """Run id of every portion `load_portions` returns, in the same order.

    It is what lets evaluation report one score per validation run instead of
    one score over all of them mixed together -- the difference between "the
    recipe does not extrapolate" and "the recipe has not converged".

    Args:
        npz_path: the split to enumerate; only its run column is read, so this
            stays cheap next to a full load.
        portion_len, stride: the cutting the tensors will use -- pass what
            load_portions was given, or the ids will not line up.
    Returns:
        int32 array with one run id per portion.
    """
    runs = np.load(npz_path, allow_pickle=True)["run"]
    return np.array([run for run in np.unique(runs)
                     for _ in _portion_starts(int((runs == run).sum()), portion_len, stride)],
                    dtype=np.int32)


def _dt_channel(t_run: np.ndarray) -> np.ndarray:
    """Per-frame time step of one run, taken from the real timestamps so a
    skipped frame or a variable rate is carried through to the CfC.

    Args:
        t_run: the run's timestamps, already sorted.
    Returns:
        The steps, same length as `t_run`; the first frame copies the second,
        having no predecessor of its own.
    """
    dt = np.diff(t_run, prepend=t_run[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    return dt


def _add_noise(X: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Gaussian noise on the normalised inputs, i.i.d. per frame and channel.

    Behavioural cloning only sees the states the expert visited; perturbing the
    inputs covers a thin tube around them. Labels stay untouched -- the target is
    still the command the expert issued in the true state.

    Args:
        X: normalised inputs (frames, channels).
        std: sigma in normalised units, so it reads as a fraction of a channel's
            range whatever that channel's physical unit is.
        seed: draw seed; the same seed gives the same perturbation.
    Returns:
        A perturbed copy of X, same shape and dtype.
    """
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, std, X.shape).astype(X.dtype)


def _run_channels(run_ids: np.ndarray, times: np.ndarray, X: np.ndarray,
                  Y: np.ndarray, run: int, use_dt: bool) -> tuple:
    """One run's frames, in chronological order and in Tudor's channel-major
    layout, ready to be cut into portions.

    Args:
        run_ids, times: the split's run and time columns.
        X, Y: the split's inputs and labels, already normalised and reordered.
        run: the run to extract.
        use_dt: append the per-frame time step as a LAST input channel.
    Returns:
        (x_run, y_run), both (features, T_run).
    """
    m = run_ids == run
    order = np.argsort(times[m])
    x_run = X[m][order].T
    if use_dt:
        x_run = np.vstack([x_run, _dt_channel(times[m][order])[None, :]])
    return x_run, Y[m][order].T


def load_portions(npz_path: str | Path,
                  input_order: list[str] = CANONICAL_INPUTS,
                  portion_len: int = 125,
                  stride: int = 25,
                  normalized: bool = True,
                  use_dt: bool = False,
                  noise_std: float = 0.0,
                  seed: int = 42):
    """Load one split as training tensors, cut into fixed-length portions.

    Args:
        npz_path: the split written by build_dataset.
        input_order: channel order for the conv, extended with the channels the
            dataset holds beyond it.
        portion_len: frames per portion (~5 s at 25 Hz by default).
        stride: step between portions; below portion_len they overlap, which is
            the augmentation the short-run count needs.
        normalized: apply the bounds embedded in the npz (they were fit on the
            train split, so validation is normalised with the training bounds).
        use_dt: append the raw per-frame time step as a LAST channel, after any
            reordering; Lightning_Model splits it off as the CfC timespans --
            the conv_cfc baseline recipe. Conservative default: opt-in.
        noise_std: sigma of the gaussian perturbation on the normalised inputs
            (see _add_noise); 0 disables it. Callers pass it for the training
            split only. The dt channel is appended afterwards, so timespans
            stay exact.
        seed: seed of that perturbation.
    Returns:
        (input_array, output_array), each (n_portions, features, portion_len)
        float32, portions ordered by run then by start frame -- the order
        portion_runs reproduces.
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

    # a portion never crosses a run: the CfC keeps memory within one and resets
    # between them, exactly like Tudor's short trajectories
    portions = [portion
                for run in np.unique(npz["run"])
                for portion in _cut_portions(
                    *_run_channels(npz["run"], npz["t"], X, Y, run, use_dt),
                    portion_len, stride)]
    input_array = np.stack([x for x, _ in portions]).astype(np.float32)
    output_array = np.stack([y for _, y in portions]).astype(np.float32)
    noise = f", noise sigma={noise_std}" if noise_std > 0 else ""
    print(f"{Path(npz_path).name}: {len(input_array)} portions of {portion_len} "
          f"({input_array.shape[1]} inputs, {output_array.shape[1]} outputs{noise})")
    return input_array, output_array
