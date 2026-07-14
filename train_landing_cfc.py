#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement du CfC ATTERRISSAGE : (coins YOLO + capteurs internes) -> gouvernes.

Variante minimale de train.py (Tudor) : SEULE la préparation des données change
(utils/landing_data.py remplace utils/data.get_data). Tout le reste est
réutilisé tel quel : DatasetController, build_controller_network,
Lightning_Model, et le câblage checkpoint/Trainer.

Prérequis :
    1. le grand run YOLO : detections.csv couvrant les 16 184 images
       (yolo_runway_corners.py --n 1 avec le modèle pose_v8 retenu) ;
    2. pip install ncps pytorch-lightning lightning.

Usage :
    python train_landing_cfc.py --config train_landing_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

# --- réutilisation directe du code de Tudor -------------------------------
from train import _default_checkpoint_name, _set_seed
from utils.config import ensure_dir, load_yaml, save_yaml
from utils.data import DatasetController
from utils.landing_data import get_landing_data
from utils.lightning import Lightning_Model
from utils.model_builder import build_controller_network


def _build_dataloaders(config: dict):
    """Équivalent de train._build_dataloaders, données atterrissage.
    Le split train/val est porté par les npz eux-mêmes (csv_to_npz.py)."""
    d = config["dataset"]
    commun = dict(
        detections_csv=d["detections_csv"],
        seq_len=int(d.get("seq_len", 64)),
        stride=int(d.get("stride", 16)),
        image_size=tuple(d.get("image_size", [1920, 991])),
        normalized=d.get("normalized", True),
    )
    train_in, train_out = get_landing_data(npz_path=d["train_npz"], **commun)
    val_in, val_out = get_landing_data(npz_path=d["val_npz"], **commun)

    train_set = DatasetController(train_in, train_out)     # Tudor, tel quel
    val_set = DatasetController(val_in, val_out)

    loader_cfg = config["dataloader"]
    kwargs = dict(batch_size=loader_cfg["batch_size"],
                  num_workers=loader_cfg["num_workers"],
                  pin_memory=loader_cfg["pin_memory"],
                  drop_last=loader_cfg["drop_last"])
    return (torch.utils.data.DataLoader(train_set, shuffle=True, **kwargs),
            torch.utils.data.DataLoader(val_set, shuffle=False, **kwargs),
            int(train_set.input.shape[-1]), int(train_set.output.shape[-1]))


def train_landing(config_path: Path, project_root: Path) -> Path:
    config = load_yaml(config_path)
    _set_seed(int(config.get("training", {}).get("seed", 42)))

    train_loader, val_loader, input_dim, output_dim = _build_dataloaders(config)
    config["dataset"]["input_dim"] = config["dataset"]["input_size"] = input_dim
    config["dataset"]["output_dim"] = config["dataset"]["output_size"] = output_dim

    network = build_controller_network(config, input_dim, output_dim)   # Tudor
    model = Lightning_Model(network, config)                            # Tudor

    checkpoint_dir = ensure_dir(project_root / "checkpoints")
    nom = config.get("checkpoint_name") or ("landing_" + _default_checkpoint_name(config))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=str(checkpoint_dir),
                                          filename=nom + "_{epoch:02d}_{val_loss:.6f}",
                                          save_top_k=1, mode="min")

    trainer = L.Trainer(max_epochs=int(config["training"]["max_epochs"]),
                        callbacks=[checkpoint_callback], logger=None, devices=1)
    trainer.fit(model, train_loader, val_loader)

    best = Path(checkpoint_callback.best_model_path)
    save_yaml(ensure_dir(project_root / "configs") / f"{best.stem}.yaml", config)
    print(f"Meilleur checkpoint : {best}")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    racine = Path(__file__).resolve().parent
    ap.add_argument("--config", type=Path, default=racine / "train_landing_config.yaml")
    ap.add_argument("--project-root", type=Path, default=racine)
    a = ap.parse_args()
    train_landing(a.config, a.project_root)
