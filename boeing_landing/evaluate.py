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
from pathlib import Path

from boeing_landing.data.features import (ANGULAR_RATES, ATTITUDE, BODY_VELOCITY,
                                          GPS, LABELS, NED_VELOCITY, WIND)
from boeing_landing.train import _load_split, _resolve_order
from utils.config import load_yaml
from utils.evaluation import (evaluate_arrays, metrics, plot_predictions,
                              regression_metrics, run_ablation_suite)

# Masked one group at a time to measure each group's contribution. Groups the
# run's dataset does not have are filtered out per run.
ABLATION_GROUPS = {
    "gps": GPS,
    "attitude": ATTITUDE,
    "angular_rate": ANGULAR_RATES,
    "body_velocity": BODY_VELOCITY,
    "ned_velocity": NED_VELOCITY,
    "wind": WIND,
    "touchdown": ["touchdown_flag"],
}


def _ablation_groups(input_labels: list[str]) -> dict:
    """Only the groups whose channels this run actually has."""
    return {name: channels for name, channels in ABLATION_GROUPS.items()
            if set(channels) <= set(input_labels)}


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


def _print_metrics(name: str, m: dict) -> None:
    print(f"{name}: MSE={m['mse_mean']:.8f} ± {m['mse_std']:.8f} | "
          f"runtime={m['runtime_mean']:.6f}s ± {m['runtime_std']:.6f}s")


def _labels(config: dict) -> list[str]:
    """The run's own command labels (a run trained before a LABELS change must
    be evaluated with the labels it was trained on)."""
    return list(config["dataset"].get("output_labels") or LABELS)


def _baseline_results(config: dict, checkpoint: Path, inputs, outputs):
    """Evaluate the run as-is: loss metrics + per-command regression metrics."""
    yhat, target, runtime = evaluate_arrays(config, config["dataloader"], checkpoint, inputs, outputs)
    results = {"baseline": metrics(yhat, target, runtime),
               "regression": regression_metrics(yhat, target, _labels(config))}
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


def _print_results(results: dict) -> None:
    _print_metrics("Baseline", results["baseline"])
    reg = results["regression"]
    print(f"{'command':14s} {'r2':>8s} {'rmse':>10s} {'mae':>10s} {'max_err':>10s}")
    for name, m in reg["per_channel"].items():
        print(f"{name:14s} {m['r2']:8.4f} {m['rmse']:10.6f} {m['mae']:10.6f} {m['max_abs_error']:10.6f}")
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
