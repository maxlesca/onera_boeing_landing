# -*- coding: utf-8 -*-
"""Train a landing controller from a pipeline config (default: gps_cfc).

Reuses the shared building blocks (build_controller_network, Lightning_Model,
transform_to_sequence, DatasetController) unchanged; only the data source is
landing-specific (boeing_landing.data.loader). Run from the repo root:

    python -m boeing_landing.train --config boeing_landing/pipelines/gps_cfc/base.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from boeing_landing.data.features import (CANONICAL_INPUTS, FEATURE_ORDERS, LABELS,
                                          extend_order)
from boeing_landing.data.loader import load_portions
from boeing_landing.config import load_pipeline_config
from utils.config import ensure_dir, save_yaml
from utils.data import DatasetController, transform_to_sequence
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network

# Single source for the repo root and the default pipeline config; every
# entrypoint (train, experiments, build) imports these instead of hardcoding.
# Pipelines live in boeing_landing/pipelines/<name>/: base.yaml + variants
# that `extends` it, overriding only the knobs they change.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "boeing_landing" / "pipelines" / "gps_cfc" / "base.yaml"


def _resolve_order(dataset_cfg: dict) -> list[str]:
    """The named channel order, extended with the dataset's own extra channels
    (extra_columns) so labels always match the real tensors."""
    order = FEATURE_ORDERS.get(dataset_cfg.get("input_order", "grouped"), CANONICAL_INPUTS)
    names = np.load(dataset_cfg["train_npz"], allow_pickle=True)["input_names"]
    return extend_order(order, [str(n) for n in names])


def _sequence(x, y, seq_len: int):
    """Tudor's sliding-window sequencing; seq_len=1 keeps one conv frame."""
    if seq_len <= 0:
        return x, y
    return transform_to_sequence(x, seq_len), y[:, :, seq_len - 1:]


def _load_split(npz_path: str, order: list[str], portion_len: int, stride: int,
                seq_len: int, use_dt: bool = False):
    x, y = load_portions(npz_path, order, portion_len=portion_len, stride=stride, use_dt=use_dt)
    return _sequence(x, y, seq_len)


def _dataloaders(config: dict):
    d = config["dataset"]
    order = _resolve_order(d)
    seq_len = int(config["sequencing"]["seq_len"]) if config.get("sequencing", {}).get("value") else 0
    use_dt = bool(d.get("use_dt", False))
    xtr, ytr = _load_split(d["train_npz"], order, int(d["portion_len"]), int(d["stride"]), seq_len, use_dt)
    xva, yva = _load_split(d["val_npz"], order, int(d["portion_len"]), int(d["stride"]), seq_len, use_dt)

    train_set, val_set = DatasetController(xtr, ytr), DatasetController(xva, yva)
    lc = config["dataloader"]
    kw = dict(batch_size=lc["batch_size"], num_workers=lc["num_workers"],
              pin_memory=lc["pin_memory"], drop_last=lc["drop_last"])
    # the dt channel is split off as timespans before the model sees the data,
    # so it does not count as a model input (same convention as the baseline)
    input_dim = int(train_set.input.shape[-1]) - (1 if use_dt else 0)
    return (torch.utils.data.DataLoader(train_set, shuffle=True, **kw),
            torch.utils.data.DataLoader(val_set, shuffle=False, **kw),
            input_dim, int(train_set.output.shape[-1]))


def _inject_labels(config: dict) -> None:
    """Lightning_Model reads input_labels/output_labels; 'dt' at the end turns
    on its with_time path (the dt channel becomes the CfC timespans)."""
    labels = list(_resolve_order(config["dataset"]))
    if config["dataset"].get("use_dt", False):
        labels.append("dt")
    config["dataset"]["input_labels"] = labels
    config["dataset"]["output_labels"] = LABELS


def _run_dir(project_root: Path, config: dict) -> Path:
    """One folder per run: runs/<pipeline>/<variant>/<timestamp>/ -- one
    subfolder per pipeline yaml (checkpoint_name), one per variant inside it
    (input order, optionally prefixed by a run_tag such as seed43). Never
    shared, never overwritten across pipelines/iterations."""
    base = config.get("checkpoint_name") or "run"
    order = config["dataset"].get("input_order", "grouped")
    tag = config.get("run_tag")
    variant = f"{tag}_{order}" if tag else order
    # microseconds: parallel jobs (e.g. a SLURM array) must never share a dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ensure_dir(project_root / "runs" / base / variant / stamp)


def assemble(config: dict):
    """Build (model, train_loader, val_loader) from a config, filling I/O dims."""
    train_loader, val_loader, input_dim, output_dim = _dataloaders(config)
    config["dataset"]["input_dim"] = config["dataset"]["input_size"] = input_dim
    config["dataset"]["output_dim"] = config["dataset"]["output_size"] = output_dim
    network = build_controller_network(config, input_dim, output_dim)
    return Lightning_Model(network, config), train_loader, val_loader


class GradNorm(L.Callback):
    """Log the total L2 gradient norm each step (training-stability signal)."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        grads = [p.grad.norm(2) for p in pl_module.parameters() if p.grad is not None]
        if grads:
            pl_module.log("grad_norm", torch.stack(grads).norm(2))


class EpochTimer(L.Callback):
    """Log the wall time of every training epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("epoch_time_s", time.time() - self._t0)


def _callbacks(config: dict, run_dir: Path):
    """Best-val checkpoint into run_dir, metric loggers, optional early stopping."""
    checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=str(run_dir),
                                 filename="{epoch:02d}_{val_loss:.6f}",
                                 save_top_k=1, mode="min")
    cbs = [checkpoint, GradNorm(), EpochTimer()]
    patience = int(config["training"].get("early_stopping_patience", 0))
    if patience > 0:
        cbs.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))
    return cbs, checkpoint


def _trainer(config: dict, run_dir: Path, callbacks) -> L.Trainer:
    """Trainer from the config knobs (hardware, precision, gradient clipping)."""
    t = config["training"]
    # log_every_n_steps > 0: metrics.csv written at that rate; 0: no logging at
    # all (checkpointing/early stopping still work, they read metrics in memory)
    n_log = int(t.get("log_every_n_steps", 1))
    logging = {"logger": None, "log_every_n_steps": n_log} if n_log > 0 else {"logger": False}
    return L.Trainer(
        max_epochs=int(t["max_epochs"]),
        accelerator=t.get("accelerator", "auto"),
        devices=t.get("devices", 1),
        precision=t.get("precision", 32),
        gradient_clip_val=float(t.get("gradient_clip_val", 0.0)),
        callbacks=callbacks, default_root_dir=str(run_dir), **logging)


def _summary(model, trainer, checkpoint, wall_time_s: float) -> dict:
    """Key facts of a finished run, saved as summary.json next to the checkpoint."""
    best = Path(checkpoint.best_model_path)
    epoch = re.search(r"epoch=(\d+)", best.stem)
    return {
        "best_val_loss": float(checkpoint.best_model_score),
        "best_epoch": int(epoch.group(1)) if epoch else None,
        "epochs_run": int(trainer.current_epoch),
        "wall_time_s": round(wall_time_s, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "best_checkpoint": best.name,
    }


def fit_and_save(model, train_loader, val_loader, config: dict, project_root: Path) -> Path:
    """Train `model` into its own run dir (checkpoint + config + summary).
    Generic: any pipeline's entrypoint can reuse this."""
    run_dir = _run_dir(project_root, config)
    callbacks, checkpoint = _callbacks(config, run_dir)
    trainer = _trainer(config, run_dir, callbacks)
    start = time.time()
    trainer.fit(model, train_loader, val_loader)

    save_yaml(run_dir / "config.yaml", config)
    summary = _summary(model, trainer, checkpoint, time.time() - start)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    best = Path(checkpoint.best_model_path)
    print(f"run dir: {run_dir}\nbest checkpoint: {best}")
    return best


def train_config(config: dict, project_root: Path) -> Path:
    """Config dict -> trained checkpoint. Orchestrates assemble + fit_and_save."""
    torch.manual_seed(int(config.get("training", {}).get("seed", 42)))
    _inject_labels(config)
    model, train_loader, val_loader = assemble(config)
    return fit_and_save(model, train_loader, val_loader, config, project_root)


def train(config_path: Path, project_root: Path = PROJECT_ROOT,
          input_order: str | None = None, max_epochs: int | None = None) -> Path:
    """Same from a YAML path, with optional launch-time overrides."""
    config = load_pipeline_config(config_path)
    if input_order:
        config["dataset"]["input_order"] = input_order
    if max_epochs:
        config["training"]["max_epochs"] = max_epochs
    return train_config(config, project_root)


def val_loss_from_checkpoint(ckpt: Path) -> float:
    """Read the val_loss back from the checkpoint filename (our naming contract)."""
    m = re.search(r"val_loss=([0-9.]+)", ckpt.stem)
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--input-order", default=None,
                    help="override the config channel order (see features.FEATURE_ORDERS)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override the config epoch count (quick trials)")
    a = ap.parse_args()
    train(a.config, a.project_root, a.input_order, a.max_epochs)


if __name__ == "__main__":
    main()
