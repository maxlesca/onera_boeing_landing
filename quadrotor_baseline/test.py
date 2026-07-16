#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained controller and optionally run automated ablations.

The testing path mirrors training: it reloads the saved config for a checkpoint,
rebuilds the model via the shared builder, evaluates on the requested test set,
and can then rerun the same evaluation under feature-ablation masks.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.config import load_yaml, resolve_checkpoint, resolve_saved_config
from utils.data import get_data, transform_to_sequence
from utils.evaluation import evaluate_arrays, metrics, plot_predictions, run_ablation_suite


def _prepare_arrays(config_model: dict, config_test: dict) -> tuple[np.ndarray, np.ndarray]:
    if config_test.get("sensitivity_analysis", False):
        starting_trajectory = None
        desired_trajectories = None
    else:
        starting_trajectory = random.randint(0, 999)
        desired_trajectories = config_test["dataset"]["desired_trajectories"]

    # Test-set loading intentionally uses the same label ordering and
    # normalization rules as training so ablations operate on the exact
    # feature tensor the model expects.
    inputs, outputs = get_data(
        input_labels=config_model["dataset"]["input_labels"],
        output_labels=config_model["dataset"]["output_labels"],
        path=config_test["dataset"]["test_path"],
        starting_trajectory=starting_trajectory,
        desired_trajectories=desired_trajectories,
        normalized=config_model["dataset"].get("normalized", True),
    )

    if config_model.get("sequencing", {}).get("value", False):
        seq_len = int(config_model["sequencing"]["seq_len"])
        # Sequence models predict the last step in each extracted history window.
        inputs = transform_to_sequence(inputs, seq_len)
        outputs = outputs[:, :, (seq_len - 1):]

    return inputs, outputs


def parse_args(cli_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained controller.")
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--test-config", type=Path, default=default_root / "test_config.yaml")
    parser.add_argument("--config-dir", type=Path, default=default_root / "configs")
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--plot", action="store_true", help="Plot the first predicted trajectory.")
    return parser.parse_args(cli_args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    config_test = load_yaml(args.test_config)
    config_model = load_yaml(resolve_saved_config(config_test["model_path"], args.config_dir))
    checkpoint_path = resolve_checkpoint(config_test["model_path"], args.project_root)
    print(f"Loading checkpoint from {checkpoint_path}")

    baseline_inputs, baseline_outputs = _prepare_arrays(config_model, config_test)
    yhat, target, runtime = evaluate_arrays(config_model, config_test["dataloader"], checkpoint_path, baseline_inputs, baseline_outputs)
    baseline_metrics = metrics(yhat, target, runtime)
    print(f"Baseline MSE: {baseline_metrics['mse_mean']:.8f} ± {baseline_metrics['mse_std']:.8f}")
    print(f"Baseline runtime: {baseline_metrics['runtime_mean']:.8f}s ± {baseline_metrics['runtime_std']:.8f}s")

    if config_test.get("ablation", {}).get("enabled", False):
        ablation_results = run_ablation_suite(config_model, config_test["dataloader"], checkpoint_path,
                                              baseline_inputs, baseline_outputs, config_test["ablation"])
        for name, metrics in ablation_results:
            print(
                f"Ablation[{name}] MSE={metrics['mse_mean']:.8f} ± {metrics['mse_std']:.8f} | "
                f"runtime={metrics['runtime_mean']:.8f}s ± {metrics['runtime_std']:.8f}s"
            )

    if args.plot or config_test.get("trajectory_plotter", False):
        plot_predictions(yhat, target)


if __name__ == "__main__":
    main()
