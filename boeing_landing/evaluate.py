# -*- coding: utf-8 -*-
"""Evaluate a trained landing run and optionally ablate feature groups.

Mirrors quadrotor_baseline/test.py: reload the run's resolved config and
checkpoint, evaluate on the validation split, then rerun with feature groups
masked to measure each group's contribution. All the machinery lives in
utils/evaluation.py; only the data loading is landing-specific.

    python -m boeing_landing.evaluate --run runs/gps_cfc/grouped/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from boeing_landing.data.features import (ANGULAR_RATES, ATTITUDE, BODY_VELOCITY,
                                          GPS, LABELS, NED_VELOCITY, POS_MAGNETIC,
                                          POS_NED, POS_RUNWAY, TOUCHDOWN, WIND,
                                          WIND_RATE)
from boeing_landing.data.loader import portion_runs
from boeing_landing.data.normalization import HEADING_SINCOS
from boeing_landing.train import _load_split, _resolve_order
from utils.config import load_yaml
from utils.evaluation import (evaluate_arrays, metrics, plot_predictions,
                              regression_metrics, run_ablation_suite)

# Masked one group at a time to measure each group's contribution. A run keeps
# whichever channels of a group it holds; the rest are filtered out per run, so
# the same table serves the gps set (absolute lat/lon, raw heading) and the
# local-frame sets (converted position, sin/cos heading). "attitude" lists both
# the raw heading and its sin/cos pair; only the ones present are masked.
ABLATION_GROUPS = {
    "gps": GPS,
    "position_runway": POS_RUNWAY,
    "position_magnetic": POS_MAGNETIC,
    "position_ned": POS_NED,
    "attitude": ATTITUDE + HEADING_SINCOS,
    "angular_rate": ANGULAR_RATES,
    "body_velocity": BODY_VELOCITY,
    "ned_velocity": NED_VELOCITY,
    "wind": WIND,
    "wind_rate": WIND_RATE,
    "touchdown": TOUCHDOWN,
}


def _ablation_groups(input_labels: list[str]) -> dict:
    """Restrict the ablation table to what one run can actually mask.

    Args:
        input_labels: the run's input channels, in model order.
    Returns:
        {group: channels present in this run}, groups it holds none of being
        dropped -- that is what lets one table serve the gps set and the
        local-frame sets.
    """
    present = set(input_labels)
    groups = {name: [c for c in channels if c in present]
              for name, channels in ABLATION_GROUPS.items()}
    return {name: chans for name, chans in groups.items() if chans}


def _find_run_files(run_dir: Path) -> tuple[dict, Path]:
    """Locate what a run dir must hold to be evaluated.

    Args:
        run_dir: the directory a training run wrote.
    Returns:
        (its archived config, its best checkpoint).
    Raises:
        SystemExit: no checkpoint there.
    """
    config = load_yaml(run_dir / "config.yaml")
    checkpoints = sorted(run_dir.glob("*.ckpt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint found in {run_dir}")
    return config, checkpoints[-1]


def _val_arrays(config: dict):
    """Rebuild the validation tensors with the exact preprocessing the run was
    trained with -- the archived config is read for all of it, so a later change
    of defaults cannot rescore an old run under a new recipe.

    Args:
        config: the run's archived config.
    Returns:
        (inputs, outputs) as _load_split returns them. No input noise: the score
        must measure the model, not the seed.
    """
    d = config["dataset"]
    seq_len = int(config["sequencing"]["seq_len"]) if config.get("sequencing", {}).get("value") else 0
    return _load_split(d["val_npz"], _resolve_order(d), int(d["portion_len"]), int(d["stride"]),
                       seq_len, bool(d.get("use_dt", False)))


def _val_portion_runs(config: dict) -> np.ndarray:
    """Run id of every validation portion, in tensor order.

    Args:
        config: the run's archived config (val_npz, portion_len, stride).
    Returns:
        The int array data.loader.portion_runs produces for that split.
    """
    d = config["dataset"]
    return portion_runs(d["val_npz"], int(d["portion_len"]), int(d["stride"]))


def mean_r2(regression: dict) -> float:
    """Summarise a regression report in one scale-invariant number -- unlike
    val_loss it compares across normalisation methods.

    Args:
        regression: what regression_metrics returned.
    Returns:
        The mean R2 over the command channels that have one. A channel whose
        target is constant (the dead `directional`, `flap`, `speedbreak`) gives
        NaN and is skipped: averaging it in would average in an undefined
        number. NaN when no channel has an R2 at all.
    """
    finite = [m["r2"] for m in regression["per_channel"].values() if math.isfinite(m["r2"])]
    return float(statistics.mean(finite)) if finite else float("nan")


def _run_regression(yhat, target, mask, labels: list[str]) -> dict:
    """Score the portions of a single held-out run.

    Args:
        yhat, target: predictions and truth, (portions, time, channels).
        mask: boolean selector of that run's portions.
        labels: command channel names.
    Returns:
        Its portion count, global MSE, mean R2 and per-channel metrics.
    """
    reg = regression_metrics(yhat[mask], target[mask], labels)
    return {"n_portions": int(mask.sum()),
            "mse": reg["global"]["mse"],
            "mean_r2": mean_r2(reg),
            "per_channel": reg["per_channel"]}


def _per_run_regression(yhat, target, runs, labels: list[str]) -> dict:
    """Regression metrics computed run by run instead of over the whole split.

    Why it matters: with several held-out runs chosen for different reasons (an
    extreme crosswind, an extreme headwind, a median control), a single averaged
    score cannot tell "the recipe does not extrapolate" from "the recipe has not
    converged". Per run, it can.

    Args:
        yhat, target: predictions and truth, batched or flat.
        runs: run id of every portion, in tensor order -- the evaluation
            dataloader keeps that order (shuffle=False) and now scores every
            portion, so the two line up exactly.
        labels: command channel names.
    Returns:
        {run id: its _run_regression report}.
    """
    yhat = yhat.reshape(-1, *yhat.shape[-2:])
    target = target.reshape(-1, *target.shape[-2:])
    runs = np.asarray(runs)
    if len(runs) != len(yhat):
        # said out loud: the portions lost are all at the end, hence all from
        # the same run, which would be scored on a truncated subset in silence
        print(f"  note: {len(runs) - len(yhat)} portions were not scored; "
              f"the per-run table below is incomplete for the last run")
        runs = runs[:len(yhat)]
    return {int(run): _run_regression(yhat, target, runs == run, labels)
            for run in np.unique(runs)}


def _print_metrics(name: str, m: dict) -> None:
    """Print one MSE/runtime line.

    Args:
        name: what is being scored ("Baseline", "Ablation[wind]", ...).
        m: the metrics dict from utils.evaluation.metrics.
    Returns:
        Nothing.
    """
    print(f"{name}: MSE={m['mse_mean']:.8f} ± {m['mse_std']:.8f} | "
          f"runtime={m['runtime_mean']:.6f}s ± {m['runtime_std']:.6f}s")


def _labels(config: dict) -> list[str]:
    """The run's own command labels.

    Args:
        config: the run's archived config.
    Returns:
        Its output_labels, falling back to LABELS -- a run trained before a
        LABELS change must be evaluated with the labels it was trained on.
    """
    return list(config["dataset"].get("output_labels") or LABELS)


def _baseline_results(config: dict, checkpoint: Path, inputs, outputs):
    """Evaluate the run as trained.

    Args:
        config: the run's archived config.
        checkpoint: its best checkpoint.
        inputs, outputs: the validation tensors.
    Returns:
        (results, yhat, target), results holding the loss metrics, the
        per-command regression and the per-run breakdown.
    """
    yhat, target, runtime = evaluate_arrays(config, config["dataloader"], checkpoint, inputs, outputs)
    labels = _labels(config)
    results = {"baseline": metrics(yhat, target, runtime),
               "regression": regression_metrics(yhat, target, labels),
               "per_run": _per_run_regression(yhat, target, _val_portion_runs(config), labels)}
    return results, yhat, target


def _ablation_results(config: dict, checkpoint: Path, inputs, outputs) -> dict:
    """Re-evaluate once per feature group, that group masked.

    Args:
        config: the run's archived config (evaluation.fill_value, input_labels).
        checkpoint: its best checkpoint.
        inputs, outputs: the validation tensors, left untouched -- each pass
            masks a copy.
    Returns:
        {group: metrics with that group masked}. expand_labels=False: our
        labels are one channel each ("u" is a body velocity here, not the
        quadrotor's 4-motor command vector).
    """
    ablation_cfg = {"enabled": True,
                    "fill_value": config.get("evaluation", {}).get("fill_value", 0.0),
                    "feature_sets": _ablation_groups(config["dataset"].get("input_labels", []))}
    return dict(run_ablation_suite(config, config["dataloader"], checkpoint,
                                   inputs, outputs, ablation_cfg, expand_labels=False))


def _live_channels(per_run: dict) -> list[str]:
    """The command channels worth a column in the per-run table.

    Args:
        per_run: what _per_run_regression returned.
    Returns:
        Those with a defined R2 on at least one run -- a constant command would
        otherwise print a wall of nan.
    """
    return [name for name in next(iter(per_run.values()))["per_channel"]
            if any(math.isfinite(r["per_channel"][name]["r2"]) for r in per_run.values())]


def _print_per_run(per_run: dict) -> None:
    """Print one line per held-out run, R2 by command.

    Args:
        per_run: what _per_run_regression returned; fewer than two runs prints
            nothing, the global table already saying it all.
    Returns:
        Nothing.
    """
    if len(per_run) < 2:
        return
    live = _live_channels(per_run)
    print(f"\n{'run':>6s} {'portions':>9s} {'mse':>10s} {'mean_r2':>8s}  "
          + " ".join(f"{name[:12]:>12s}" for name in live))
    for run, r in sorted(per_run.items()):
        cells = " ".join(f"{r['per_channel'][name]['r2']:12.4f}" for name in live)
        print(f"{run:6d} {r['n_portions']:9d} {r['mse']:10.6f} {r['mean_r2']:8.4f}  {cells}")


def _print_results(results: dict) -> None:
    """Print the whole report: loss line, per-command table, per-run table,
    then one line per ablated group.

    Args:
        results: the dict evaluate_run assembled.
    Returns:
        Nothing.
    """
    _print_metrics("Baseline", results["baseline"])
    reg = results["regression"]
    print(f"{'command':14s} {'r2':>8s} {'rmse':>10s} {'mae':>10s} {'max_err':>10s}")
    for name, m in reg["per_channel"].items():
        print(f"{name:14s} {m['r2']:8.4f} {m['rmse']:10.6f} {m['mae']:10.6f} {m['max_abs_error']:10.6f}")
    _print_per_run(results.get("per_run", {}))
    for name, m in results["ablation"].items():
        _print_metrics(f"Ablation[{name}]", m)


def _save_results(run_dir: Path, results: dict) -> None:
    """Archive the report next to the checkpoint.

    Args:
        run_dir: the run directory.
        results: the dict evaluate_run assembled.
    Returns:
        Nothing; writes evaluation.json, which report.py later plots from.
    """
    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {run_dir / 'evaluation.json'}")


def _wants_ablation(config: dict, override: bool | None) -> bool:
    """Whether to run the ablation suite.

    Args:
        config: the run's archived config (its evaluation.ablation flag).
        override: the CLI's answer, None meaning "let the config decide".
    Returns:
        True to ablate.
    """
    return config.get("evaluation", {}).get("ablation", False) if override is None else override


def evaluate_run(run_dir: Path, with_ablation: bool | None = None, plot: bool = False) -> dict:
    """Score one finished run.

    Args:
        run_dir: the directory holding its config and checkpoint.
        with_ablation: force the ablation suite on/off; None follows the config.
        plot: also show the first predicted portion.
    Returns:
        The results dict, also written to run_dir/evaluation.json and printed.
    """
    config, checkpoint = _find_run_files(run_dir)
    print(f"Loading checkpoint from {checkpoint}")
    inputs, outputs = _val_arrays(config)

    results, yhat, target = _baseline_results(config, checkpoint, inputs, outputs)
    results["ablation"] = (_ablation_results(config, checkpoint, inputs, outputs)
                           if _wants_ablation(config, with_ablation) else {})

    _print_results(results)
    _save_results(run_dir, results)
    if plot:
        plot_predictions(yhat, target, labels=_labels(config))
    return results


def main() -> None:
    """CLI entrypoint: evaluate the run dir given by --run.

    Returns:
        Nothing; see evaluate_run for what is printed and written.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="run directory (config.yaml + .ckpt)")
    ap.add_argument("--ablation", action="store_true",
                    help="force feature-group ablations (default: evaluation section of the run config)")
    ap.add_argument("--plot", action="store_true", help="plot the first predicted portion")
    a = ap.parse_args()
    evaluate_run(a.run, with_ablation=True if a.ablation else None, plot=a.plot)


if __name__ == "__main__":
    main()
