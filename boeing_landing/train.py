# -*- coding: utf-8 -*-
"""Train the step-1 landing controller (Conv1D -> CfC) on inertial + GPS.

Reuses Tudor's build_controller_network, Lightning_Model, transform_to_sequence
and DatasetController unchanged. Only the data source differs
(boeing_landing.data.loader). Run from the repo root:

    python -m boeing_landing.train --config boeing_landing/configs/step1_cfc.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from boeing_landing.data.features import CANONICAL_INPUTS, FEATURE_ORDERS, LABELS
from boeing_landing.data.loader import load_portions
from utils.config import ensure_dir, load_yaml, save_yaml
from utils.data import DatasetController, transform_to_sequence
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network


def _resolve_order(dataset_cfg: dict) -> list[str]:
    return FEATURE_ORDERS.get(dataset_cfg.get("input_order", "grouped"), CANONICAL_INPUTS)


def _sequence(x, y, seq_len: int):
    """Tudor's sliding-window sequencing; seq_len=1 keeps one conv frame."""
    if seq_len <= 0:
        return x, y
    return transform_to_sequence(x, seq_len), y[:, :, seq_len - 1:]


def _load_split(npz_path: str, order: list[str], portion_len: int, stride: int, seq_len: int):
    x, y = load_portions(npz_path, order, portion_len=portion_len, stride=stride)
    return _sequence(x, y, seq_len)


def _dataloaders(config: dict):
    d = config["dataset"]
    order = _resolve_order(d)
    seq_len = int(config["sequencing"]["seq_len"]) if config.get("sequencing", {}).get("value") else 0
    xtr, ytr = _load_split(d["train_npz"], order, int(d["portion_len"]), int(d["stride"]), seq_len)
    xva, yva = _load_split(d["val_npz"], order, int(d["portion_len"]), int(d["stride"]), seq_len)

    train_set, val_set = DatasetController(xtr, ytr), DatasetController(xva, yva)
    lc = config["dataloader"]
    kw = dict(batch_size=lc["batch_size"], num_workers=lc["num_workers"],
              pin_memory=lc["pin_memory"], drop_last=lc["drop_last"])
    return (torch.utils.data.DataLoader(train_set, shuffle=True, **kw),
            torch.utils.data.DataLoader(val_set, shuffle=False, **kw),
            int(train_set.input.shape[-1]), int(train_set.output.shape[-1]))


def _inject_labels(config: dict) -> None:
    """Lightning_Model reads input_labels/output_labels; keep them consistent
    with the chosen channel order (no 't'/'dt' -> with_time stays False)."""
    config["dataset"]["input_labels"] = _resolve_order(config["dataset"])
    config["dataset"]["output_labels"] = LABELS


def _run_dir(project_root: Path, config: dict) -> Path:
    """One folder per run: runs/<name>_<order>/<timestamp>/ -- never shared,
    never overwritten across pipelines/iterations."""
    base = config.get("checkpoint_name") or "step1"
    order = config["dataset"].get("input_order", "grouped")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(project_root / "runs" / f"{base}_{order}" / stamp)


def assemble(config: dict):
    """Build (model, train_loader, val_loader) from a config, filling I/O dims."""
    train_loader, val_loader, input_dim, output_dim = _dataloaders(config)
    config["dataset"]["input_dim"] = config["dataset"]["input_size"] = input_dim
    config["dataset"]["output_dim"] = config["dataset"]["output_size"] = output_dim
    network = build_controller_network(config, input_dim, output_dim)
    return Lightning_Model(network, config), train_loader, val_loader


def _callbacks(config: dict, run_dir: Path):
    """Best-val checkpoint into run_dir, plus optional early stopping."""
    checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=str(run_dir),
                                 filename="{epoch:02d}_{val_loss:.6f}",
                                 save_top_k=1, mode="min")
    cbs = [checkpoint]
    patience = int(config["training"].get("early_stopping_patience", 0))
    if patience > 0:
        cbs.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))
    return cbs, checkpoint


def _trainer(config: dict, run_dir: Path, callbacks) -> L.Trainer:
    """Trainer from the config knobs (hardware, precision, gradient clipping)."""
    t = config["training"]
    return L.Trainer(
        max_epochs=int(t["max_epochs"]),
        accelerator=t.get("accelerator", "auto"),
        devices=t.get("devices", 1),
        precision=t.get("precision", 32),
        gradient_clip_val=float(t.get("gradient_clip_val", 0.0)),
        callbacks=callbacks, logger=None, default_root_dir=str(run_dir))


def fit_and_save(model, train_loader, val_loader, config: dict, project_root: Path) -> Path:
    """Train `model` into its own run dir (checkpoint + resolved config).
    Generic: any pipeline's entrypoint can reuse this."""
    run_dir = _run_dir(project_root, config)
    callbacks, checkpoint = _callbacks(config, run_dir)
    _trainer(config, run_dir, callbacks).fit(model, train_loader, val_loader)

    save_yaml(run_dir / "config.yaml", config)
    best = Path(checkpoint.best_model_path)
    print(f"run dir: {run_dir}\nbest checkpoint: {best}")
    return best


def train(config_path: Path, project_root: Path, input_order: str | None = None) -> Path:
    """Config -> trained checkpoint. Orchestrates assemble + fit_and_save."""
    config = load_yaml(config_path)
    if input_order:
        config["dataset"]["input_order"] = input_order
    torch.manual_seed(int(config.get("training", {}).get("seed", 42)))
    _inject_labels(config)
    model, train_loader, val_loader = assemble(config)
    return fit_and_save(model, train_loader, val_loader, config, project_root)


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # onera_boeing_landing/
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=root / "boeing_landing/configs/step1_cfc.yaml")
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument("--input-order", default=None,
                    help="override the config channel order (see features.FEATURE_ORDERS)")
    a = ap.parse_args()
    train(a.config, a.project_root, a.input_order)


if __name__ == "__main__":
    main()
