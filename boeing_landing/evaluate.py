# -*- coding: utf-8 -*-
"""Evaluate a trained landing run and optionally ablate feature groups.

Mirrors quadrotor_baseline/test.py: reload the run's resolved config and
checkpoint, evaluate on the validation split, then rerun with feature groups
masked to measure each group's contribution. All the machinery lives in
utils/evaluation.py; only the data loading is landing-specific.

    python -m boeing_landing.evaluate --run runs/step1_landing_cfc_grouped/<timestamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from boeing_landing.data.features import (ANGULAR_RATES, ATTITUDE, BODY_VELOCITY,
                                          GPS, LABELS, NED_VELOCITY)
from boeing_landing.train import _load_split, _resolve_order
from utils.config import load_yaml
from utils.evaluation import evaluate_arrays, metrics, plot_predictions, run_ablation_suite

# Masked one group at a time to measure each group's contribution.
ABLATION_GROUPS = {
    "gps": GPS,
    "attitude": ATTITUDE,
    "angular_rate": ANGULAR_RATES,
    "body_velocity": BODY_VELOCITY,
    "ned_velocity": NED_VELOCITY,
}


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
    return _load_split(d["val_npz"], _resolve_order(d), int(d["portion_len"]), int(d["stride"]), seq_len)


def _print_metrics(name: str, m: dict) -> None:
    print(f"{name}: MSE={m['mse_mean']:.8f} ± {m['mse_std']:.8f} | "
          f"runtime={m['runtime_mean']:.6f}s ± {m['runtime_std']:.6f}s")


def evaluate_run(run_dir: Path, with_ablation: bool = False, plot: bool = False) -> dict:
    config, checkpoint = _find_run_files(run_dir)
    print(f"Loading checkpoint from {checkpoint}")
    inputs, outputs = _val_arrays(config)

    yhat, target, runtime = evaluate_arrays(config, config["dataloader"], checkpoint, inputs, outputs)
    baseline = metrics(yhat, target, runtime)
    _print_metrics("Baseline", baseline)

    if with_ablation:
        ablation_cfg = {"enabled": True, "fill_value": 0.0, "feature_sets": ABLATION_GROUPS}
        # expand_labels=False: our labels are one channel each ("u" is a body
        # velocity here, not the quadrotor's 4-motor command vector).
        for name, m in run_ablation_suite(config, config["dataloader"], checkpoint,
                                          inputs, outputs, ablation_cfg, expand_labels=False):
            _print_metrics(f"Ablation[{name}]", m)

    if plot:
        plot_predictions(yhat, target, labels=LABELS)
    return baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="run directory (config.yaml + .ckpt)")
    ap.add_argument("--ablation", action="store_true", help="mask each feature group and re-evaluate")
    ap.add_argument("--plot", action="store_true", help="plot the first predicted portion")
    a = ap.parse_args()
    evaluate_run(a.run, with_ablation=a.ablation, plot=a.plot)


if __name__ == "__main__":
    main()
