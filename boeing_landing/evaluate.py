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
    """Each group restricted to the channels this run actually has; empty groups
    (a group the run holds none of) are dropped."""
    present = set(input_labels)
    groups = {name: [c for c in channels if c in present]
              for name, channels in ABLATION_GROUPS.items()}
    return {name: chans for name, chans in groups.items() if chans}


def _find_run_files(run_dir: Path) -> tuple[dict, Path]:
    """A run dir holds exactly one config.yaml and one best checkpoint."""
    config = load_yaml(run_dir / "config.yaml")
    checkpoints = sorted(run_dir.glob("*.ckpt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint found in {run_dir}")
    return config, checkpoints[-1]


def _val_arrays(config: dict):
    """Validation tensors with the exact preprocessing the run was trained with."""
    d = config["dataset"]
    seq_len = int(config["sequencing"]["seq_len"]) if config.get("sequencing", {}).get("value") else 0
    return _load_split(d["val_npz"], _resolve_order(d), int(d["portion_len"]), int(d["stride"]),
                       seq_len, bool(d.get("use_dt", False)))


def _val_portion_runs(config: dict) -> np.ndarray:
    """Run id of every validation portion, in tensor order (see loader)."""
    d = config["dataset"]
    return portion_runs(d["val_npz"], int(d["portion_len"]), int(d["stride"]))


def mean_r2(regression: dict) -> float:
    """Mean R2 over the command channels that have one. Channels whose target is
    constant (the dead `directional`, `flap`, `speedbreak`) give R2 = NaN and are
    skipped -- averaging them in would be averaging in an undefined number.
    Scale-invariant, so unlike val_loss it compares across normalisation methods."""
    finite = [m["r2"] for m in regression["per_channel"].values() if math.isfinite(m["r2"])]
    return float(statistics.mean(finite)) if finite else float("nan")


def _per_run_regression(yhat, target, runs, labels: list[str]) -> dict:
    """Regression metrics computed run by run instead of over the whole split.

    Why it matters: with several held-out runs chosen for different reasons (an
    extreme crosswind, an extreme headwind, a median control), a single averaged
    score cannot tell "the recipe does not extrapolate" from "the recipe has not
    converged". Per run, it can.

    The evaluation dataloader keeps the portion order (shuffle=False) but drops
    the last incomplete batch, so the ids are truncated to what was scored.
    """
    yhat = yhat.reshape(-1, *yhat.shape[-2:])
    target = target.reshape(-1, *target.shape[-2:])
    runs = np.asarray(runs)[:len(yhat)]
    per_run = {}
    for run in np.unique(runs):
        mask = runs == run
        reg = regression_metrics(yhat[mask], target[mask], labels)
        per_run[int(run)] = {"n_portions": int(mask.sum()),
                             "mse": reg["global"]["mse"],
                             "mean_r2": mean_r2(reg),
                             "per_channel": reg["per_channel"]}
    return per_run


def _print_metrics(name: str, m: dict) -> None:
    print(f"{name}: MSE={m['mse_mean']:.8f} ± {m['mse_std']:.8f} | "
          f"runtime={m['runtime_mean']:.6f}s ± {m['runtime_std']:.6f}s")


def _labels(config: dict) -> list[str]:
    """The run's own command labels (a run trained before a LABELS change must
    be evaluated with the labels it was trained on)."""
    return list(config["dataset"].get("output_labels") or LABELS)


def _baseline_results(config: dict, checkpoint: Path, inputs, outputs):
    """Evaluate the run as-is: loss metrics, per-command and per-run regression."""
    yhat, target, runtime = evaluate_arrays(config, config["dataloader"], checkpoint, inputs, outputs)
    labels = _labels(config)
    results = {"baseline": metrics(yhat, target, runtime),
               "regression": regression_metrics(yhat, target, labels),
               "per_run": _per_run_regression(yhat, target, _val_portion_runs(config), labels)}
    return results, yhat, target


def _ablation_results(config: dict, checkpoint: Path, inputs, outputs) -> dict:
    """Re-evaluate with each feature group masked (fill_value from the config).
    expand_labels=False: our labels are one channel each ("u" is a body
    velocity here, not the quadrotor's 4-motor command vector)."""
    ablation_cfg = {"enabled": True,
                    "fill_value": config.get("evaluation", {}).get("fill_value", 0.0),
                    "feature_sets": _ablation_groups(config["dataset"].get("input_labels", []))}
    return dict(run_ablation_suite(config, config["dataloader"], checkpoint,
                                   inputs, outputs, ablation_cfg, expand_labels=False))


def _print_per_run(per_run: dict) -> None:
    """One line per held-out run. Only the channels that have a defined R2
    somewhere get a column -- a constant command would print a wall of nan."""
    if len(per_run) < 2:
        return
    live = [name for name in next(iter(per_run.values()))["per_channel"]
            if any(math.isfinite(r["per_channel"][name]["r2"]) for r in per_run.values())]
    print(f"\n{'run':>6s} {'portions':>9s} {'mse':>10s} {'mean_r2':>8s}  "
          + " ".join(f"{name[:12]:>12s}" for name in live))
    for run, r in sorted(per_run.items()):
        cells = " ".join(f"{r['per_channel'][name]['r2']:12.4f}" for name in live)
        print(f"{run:6d} {r['n_portions']:9d} {r['mse']:10.6f} {r['mean_r2']:8.4f}  {cells}")


def _print_results(results: dict) -> None:
    _print_metrics("Baseline", results["baseline"])
    reg = results["regression"]
    print(f"{'command':14s} {'r2':>8s} {'rmse':>10s} {'mae':>10s} {'max_err':>10s}")
    for name, m in reg["per_channel"].items():
        print(f"{name:14s} {m['r2']:8.4f} {m['rmse']:10.6f} {m['mae']:10.6f} {m['max_abs_error']:10.6f}")
    _print_per_run(results.get("per_run", {}))
    for name, m in results["ablation"].items():
        _print_metrics(f"Ablation[{name}]", m)


def _save_results(run_dir: Path, results: dict) -> None:
    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(f"saved -> {run_dir / 'evaluation.json'}")


def evaluate_run(run_dir: Path, with_ablation: bool | None = None, plot: bool = False) -> dict:
    """Orchestrate: load run, baseline, optional ablations, print, save, plot."""
    config, checkpoint = _find_run_files(run_dir)
    print(f"Loading checkpoint from {checkpoint}")
    inputs, outputs = _val_arrays(config)

    results, yhat, target = _baseline_results(config, checkpoint, inputs, outputs)
    # ablation on/off comes from the config's evaluation: section; CLI forces it
    if config.get("evaluation", {}).get("ablation", False) if with_ablation is None else with_ablation:
        results["ablation"] = _ablation_results(config, checkpoint, inputs, outputs)
    else:
        results["ablation"] = {}

    _print_results(results)
    _save_results(run_dir, results)
    if plot:
        plot_predictions(yhat, target, labels=_labels(config))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="run directory (config.yaml + .ckpt)")
    ap.add_argument("--ablation", action="store_true",
                    help="force feature-group ablations (default: evaluation section of the run config)")
    ap.add_argument("--plot", action="store_true", help="plot the first predicted portion")
    a = ap.parse_args()
    evaluate_run(a.run, with_ablation=True if a.ablation else None, plot=a.plot)


if __name__ == "__main__":
    main()
