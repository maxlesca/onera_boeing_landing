# -*- coding: utf-8 -*-
"""
Shared evaluation helpers used by the test/evaluate entrypoints.

Extracted from test.py so the quadrotor baseline and the boeing_landing
pipeline reuse the same checkpoint-reload, evaluation, metrics, and
feature-ablation machinery instead of duplicating it.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.ablation import apply_feature_ablation, iter_ablation_specs
from utils.data import DatasetController
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network


def dataloader_from_arrays(inputs: np.ndarray, outputs: np.ndarray, loader_cfg: dict):
    dataset = DatasetController(inputs, outputs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        drop_last=loader_cfg["drop_last"],
        shuffle=False,
    )


def load_model(config_model: dict, checkpoint_path: Path) -> Lightning_Model:
    input_dim = int(config_model["dataset"].get("input_dim", config_model["dataset"].get("input_size")))
    output_dim = int(config_model["dataset"].get("output_dim", config_model["dataset"].get("output_size")))
    network = build_controller_network(config_model, input_dim, output_dim)
    model = Lightning_Model(network, config_model)
    # Checkpoints only store the state dict; the full model topology comes from
    # the archived YAML produced during training.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate_arrays(config_model: dict,
                    loader_cfg: dict,
                    checkpoint_path: Path,
                    inputs: np.ndarray,
                    outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = dataloader_from_arrays(inputs, outputs, loader_cfg)
    model = load_model(config_model, checkpoint_path)
    # Logging/model-summary are disabled here because this path is typically
    # used for repeated benchmark and ablation sweeps.
    trainer = L.Trainer(logger=False, enable_model_summary=False, devices=1, num_nodes=1)
    start = time.time()
    trainer.test(model, dataloaders=loader)
    duration = time.time() - start
    print(f"Inference wall time: {duration:.2f}s")
    # cat, not stack: the batches are concatenated back into one (samples, time,
    # channels) array, so `metrics` really averages per sample and the per-channel
    # metrics see every portion of the split.
    return (
        torch.cat(model.all_yhat).cpu().numpy(),
        torch.cat(model.all_target).cpu().numpy(),
        np.asarray(model.all_runtime),
    )


def metrics(yhat: np.ndarray, target: np.ndarray, runtime: np.ndarray) -> dict:
    mse_per_sample = np.mean((yhat - target) ** 2, axis=(1, 2))
    return {
        "mse_mean": float(mse_per_sample.mean()),
        "mse_std": float(mse_per_sample.std()),
        "runtime_mean": float(runtime.mean()) if runtime.size else 0.0,
        "runtime_std": float(runtime.std()) if runtime.size else 0.0,
    }


def _channel_metrics(err: np.ndarray, truth: np.ndarray) -> dict:
    """MSE/MAE/RMSE/R2/max-abs-error of one output channel."""
    mse = float((err ** 2).mean())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return {
        "mse": mse,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "max_abs_error": float(np.abs(err).max()),
    }


def regression_metrics(yhat: np.ndarray, target: np.ndarray,
                       labels: list[str] | None = None) -> dict:
    """Global and per-channel regression metrics, keyed by channel name."""
    yhat2 = yhat.reshape(-1, yhat.shape[-1])
    target2 = target.reshape(-1, target.shape[-1])
    labels = labels or [f"y{i}" for i in range(target2.shape[1])]
    return {
        "global": _channel_metrics(yhat2 - target2, target2),
        "per_channel": {name: _channel_metrics(yhat2[:, i] - target2[:, i], target2[:, i])
                        for i, name in enumerate(labels)},
    }


def plot_predictions(yhat: np.ndarray, target: np.ndarray, labels: list[str] | None = None) -> None:
    if yhat.ndim == 3:
        yhat = yhat[0]
    if target.ndim == 3:
        target = target[0]

    n_outputs = target.shape[-1]
    if labels is None:
        labels = [f"u_{i + 1}" for i in range(n_outputs)]

    n_cols = 2
    n_rows = (n_outputs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i in range(n_outputs):
        axes[i].plot(target[:, i], label="Target")
        axes[i].plot(yhat[:, i], label="Prediction")
        axes[i].set_title(labels[i])
        axes[i].grid(True)
        axes[i].legend()
    for ax in axes[n_outputs:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def run_ablation_suite(config_model: dict,
                       loader_cfg: dict,
                       checkpoint_path: Path,
                       baseline_inputs: np.ndarray,
                       baseline_outputs: np.ndarray,
                       ablation_cfg: dict,
                       expand_labels: bool = True) -> list[tuple[str, dict]]:
    fill_value = float(ablation_cfg.get("fill_value", 0.0))
    results: list[tuple[str, dict]] = []

    for name, features in iter_ablation_specs(ablation_cfg, config_model["dataset"]["input_labels"],
                                              expand=expand_labels):
        # Each ablation reruns the full evaluation with a masked copy of the
        # original test tensor, leaving the baseline arrays untouched.
        ablated_inputs = apply_feature_ablation(
            baseline_inputs,
            config_model["dataset"]["input_labels"],
            features,
            fill_value=fill_value,
            expand=expand_labels,
        )
        yhat, target, runtime = evaluate_arrays(config_model, loader_cfg, checkpoint_path, ablated_inputs, baseline_outputs)
        results.append((name, metrics(yhat, target, runtime)))

    return results
