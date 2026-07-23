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
from functools import lru_cache
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from boeing_landing.data.features import FEATURE_ORDERS, LABELS, extend_order
from boeing_landing.data.loader import load_portions
from boeing_landing.config import load_pipeline_config
from utils.scheduler import model_for, wants_schedule
from utils.config import ensure_dir, save_yaml
from utils.data import DatasetController, transform_to_sequence
from utils.model_builder import build_controller_network

# Single source for the repo root and the default pipeline config; every
# entrypoint (train, experiments, build) imports these instead of hardcoding.
# Pipelines live in boeing_landing/pipelines/<name>/: base.yaml + variants
# that `extends` it, overriding only the knobs they change.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "boeing_landing" / "pipelines" / "gps_cfc" / "base.yaml"


@lru_cache(maxsize=8)
def _read_npz_meta(path: str, mtime_ns: int, size: int) -> tuple[list[str], list[str]]:
    """Read a split's channel names, memoised on the file's identity.

    Args:
        path: the npz.
        mtime_ns, size: its stat, part of the key -- a rebuilt npz is a
            different entry, so the cache can never serve stale names.
    Returns:
        (input_names, label_names), the latter falling back to LABELS for a npz
        built before label_names existed.
    """
    npz = np.load(path, allow_pickle=True)
    labels = [str(n) for n in npz["label_names"]] if "label_names" in npz else list(LABELS)
    return [str(n) for n in npz["input_names"]], labels


def _npz_meta(npz_path: str) -> tuple[list[str], list[str]]:
    """The channel names of a split.

    Args:
        npz_path: the split.
    Returns:
        (input_names, label_names). The archive is opened once per file and per
        process: a seed sweep or a LORO fold resolves the order and the labels
        several times, and each open inflates the members again.
    """
    stat = Path(npz_path).stat()
    return _read_npz_meta(str(npz_path), stat.st_mtime_ns, stat.st_size)


def _resolve_order(dataset_cfg: dict) -> list[str]:
    """The channel order a config asks for, as actual channel names.

    Args:
        dataset_cfg: the config's dataset section, read for `input_order`
            (default 'grouped') and `train_npz`.
    Returns:
        The named order extended with the dataset's own extra channels
        (extra_columns), so the labels always match the real tensors.
    Raises:
        SystemExit: the name is not in FEATURE_ORDERS. Falling back to the
            default would train a second copy of the grouped arm, filed under
            the name that was asked for.
    """
    name = dataset_cfg.get("input_order", "grouped")
    if name not in FEATURE_ORDERS:
        raise SystemExit(f"unknown input_order {name!r}; choose from {sorted(FEATURE_ORDERS)}")
    return extend_order(FEATURE_ORDERS[name], _npz_meta(dataset_cfg["train_npz"])[0])


def _npz_labels(dataset_cfg: dict) -> list[str]:
    """The command channels the npz was actually built with.

    Args:
        dataset_cfg: the config's dataset section, read for `train_npz`.
    Returns:
        Its label_names -- the npz is the source of truth here, so a pipeline
        that changed build.label_set cannot be trained against a stale list.
    """
    return _npz_meta(dataset_cfg["train_npz"])[1]


def _sequence(x, y, seq_len: int):
    """Tudor's sliding-window sequencing.

    Args:
        x, y: the portion tensors.
        seq_len: window length; 1 keeps one conv frame (the baseline recipe),
            <= 0 skips the transform entirely.
    Returns:
        (x, y) windowed, the labels trimmed to the frames a full window covers.
    """
    if seq_len <= 0:
        return x, y
    return transform_to_sequence(x, seq_len), y[:, :, seq_len - 1:]


def _load_split(npz_path: str, order: list[str], portion_len: int, stride: int,
                seq_len: int, use_dt: bool = False):
    """One split, from npz to sequenced tensors.

    Args:
        npz_path: the split to load.
        order: channel order (see _resolve_order).
        portion_len, stride: the portion cutting (see data.loader).
        seq_len: sequencing window (see _sequence).
        use_dt: append the per-frame dt channel for the CfC timespans.
    Returns:
        (x, y), ready for DatasetController. Never perturbed: input noise is
        applied per fetch by NoisyInputs, not baked into the tensors.
    """
    x, y = load_portions(npz_path, order, portion_len=portion_len, stride=stride,
                         use_dt=use_dt)
    return _sequence(x, y, seq_len)


class NoisyInputs(torch.utils.data.Dataset):
    """Gaussian noise on the normalised inputs, redrawn on every fetch.

    Behavioural cloning only sees the states the expert visited; perturbing the
    inputs covers a thin tube around them. That only works if the perturbation
    is resampled: noise drawn once and baked into the tensors is just a second
    fixed dataset, which the network memorises exactly like the clean one.

    Labels stay untouched -- the target is still the command the expert issued
    in the true state -- and so does the dt channel, whose timespans must stay
    exact.
    """

    def __init__(self, dataset, std: float, n_channels: int, seed: int):
        """Wrap a dataset.

        Args:
            dataset: the clean DatasetController.
            std: sigma in normalised units.
            n_channels: how many leading channels to perturb, i.e. everything
                but the trailing dt channel when the config appends one.
            seed: seed of the draw generator. With num_workers > 0 each worker
                copies it, so the noise stays random but is no longer
                reproducible sample by sample.
        Returns:
            Nothing.
        """
        self.dataset = dataset
        self.std = std
        self.n_channels = n_channels
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        """Dataset size.

        Returns:
            The wrapped dataset's length, unchanged.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Fetch one sample and perturb it.

        Args:
            idx: its index.
        Returns:
            (inputs + fresh noise on their first n_channels, labels unchanged).
        """
        x, y = self.dataset[idx]
        x = x.clone()
        noise = torch.randn(x[..., :self.n_channels].shape, generator=self.generator)
        x[..., :self.n_channels] += noise * self.std
        return x, y


def _dataloaders(config: dict):
    """Both dataloaders and the model's I/O dimensions, from a config alone.

    Args:
        config: the resolved pipeline config (dataset + dataloader sections).
    Returns:
        (train_loader, val_loader, input_dim, output_dim). The dt channel is
        split off as timespans before the model sees the data, so it does not
        count in input_dim (same convention as the baseline). Only the training
        loader is perturbed: the score must measure the model, not the seed.
    """
    d = config["dataset"]
    order = _resolve_order(d)
    seq_len = int(config["sequencing"]["seq_len"]) if config.get("sequencing", {}).get("value") else 0
    use_dt = bool(d.get("use_dt", False))
    noise = float(d.get("noise_std", 0.0)) if d.get("with_noise") else 0.0
    seed = int(config.get("training", {}).get("seed", 42))
    xtr, ytr = _load_split(d["train_npz"], order, int(d["portion_len"]), int(d["stride"]),
                           seq_len, use_dt)
    xva, yva = _load_split(d["val_npz"], order, int(d["portion_len"]), int(d["stride"]), seq_len, use_dt)

    train_set, val_set = DatasetController(xtr, ytr), DatasetController(xva, yva)
    lc = config["dataloader"]
    kw = dict(batch_size=lc["batch_size"], num_workers=lc["num_workers"],
              pin_memory=lc["pin_memory"])
    input_dim = int(train_set.input.shape[-1]) - (1 if use_dt else 0)
    output_dim = int(train_set.output.shape[-1])
    train_data = NoisyInputs(train_set, noise, input_dim, seed) if noise > 0 else train_set
    # drop_last applies to TRAINING only. On validation it would throw away the
    # tail portions, and a split shorter than one batch would yield NO batch at
    # all -- every leave-one-run-out fold of ned_wind_cfc is in that case
    # (17-29 portions held out, batch 32), leaving no val_loss to checkpoint on.
    return (torch.utils.data.DataLoader(train_data, shuffle=True,
                                        drop_last=lc["drop_last"], **kw),
            torch.utils.data.DataLoader(val_set, shuffle=False,
                                        drop_last=False, **kw),
            input_dim, output_dim)


def _with_dataset(config: dict, **fields) -> dict:
    """Set fields in a config's dataset section without touching the caller's.

    Args:
        config: the resolved config.
        fields: dataset keys to add or replace.
    Returns:
        A shallow copy carrying them, so a sweep can derive one config per arm
        from a single base without the arms bleeding into each other.
    """
    return {**config, "dataset": {**config["dataset"], **fields}}


def with_labels(config: dict) -> dict:
    """Name the channels for Lightning_Model, which reads input_labels and
    output_labels off the config.

    Args:
        config: the resolved config.
    Returns:
        A copy whose dataset section carries both lists; a trailing 'dt' in the
        inputs is what turns on the model's with_time path (that channel then
        becomes the CfC timespans).
    """
    labels = list(_resolve_order(config["dataset"]))
    if config["dataset"].get("use_dt", False):
        labels.append("dt")
    return _with_dataset(config, input_labels=labels,
                         output_labels=_npz_labels(config["dataset"]))


def _run_dir(project_root: Path, config: dict) -> Path:
    """Reserve this run's own output folder.

    Args:
        project_root: repo root holding runs/.
        config: read for `checkpoint_name` (the pipeline), the dataset's
            `input_order` and the optional `run_tag` (e.g. seed43).
    Returns:
        A fresh runs/<pipeline>/<[tag_]order>/<timestamp>/ directory, created.
        Never shared, never overwritten across pipelines or iterations.
    """
    base = config.get("checkpoint_name") or "run"
    order = config["dataset"].get("input_order", "grouped")
    tag = config.get("run_tag")
    variant = f"{tag}_{order}" if tag else order
    # microseconds: parallel jobs (e.g. a SLURM array) must never share a dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ensure_dir(project_root / "runs" / base / variant / stamp)


def assemble(config: dict):
    """Turn a config into everything a fit needs.

    Args:
        config: the resolved config, labels already named (see with_labels).
    Returns:
        (model, train_loader, val_loader, config), the returned config being a
        copy carrying the measured I/O dimensions -- the model was built from
        that copy, and it is the one to archive next to the checkpoint.
    """
    train_loader, val_loader, input_dim, output_dim = _dataloaders(config)
    resolved = _with_dataset(config, input_dim=input_dim, input_size=input_dim,
                             output_dim=output_dim, output_size=output_dim)
    network = build_controller_network(resolved, input_dim, output_dim)
    return model_for(resolved, network), train_loader, val_loader, resolved


class GradNorm(L.Callback):
    """Log the total L2 gradient norm each step (training-stability signal)."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Log `grad_norm` just before the weights move.

        Args:
            trainer: the Lightning trainer (unused).
            pl_module: the model whose gradients are read.
            optimizer: the optimizer about to step (unused).
        Returns:
            Nothing; the norm goes to the metrics log.
        """
        grads = [p.grad.norm(2) for p in pl_module.parameters() if p.grad is not None]
        if grads:
            pl_module.log("grad_norm", torch.stack(grads).norm(2))


class EpochTimer(L.Callback):
    """Log the wall time of every training epoch."""

    def on_train_epoch_start(self, trainer, pl_module):
        """Start the stopwatch.

        Args:
            trainer, pl_module: Lightning's hook arguments (unused).
        Returns:
            Nothing.
        """
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Log `epoch_time_s`.

        Args:
            trainer: the Lightning trainer (unused).
            pl_module: the model the metric is logged on.
        Returns:
            Nothing.
        """
        pl_module.log("epoch_time_s", time.time() - self._t0)


def _logging_enabled(config: dict) -> bool:
    """Whether this run will have a logger at all.

    Args:
        config: read for training.log_every_n_steps -- 0 means no metrics file
            (checkpointing and early stopping still work, they read the metrics
            in memory).
    Returns:
        True when a logger is attached. Single source for the decision, because
        a callback that writes to the logger must not be added without one.
    """
    return int(config["training"].get("log_every_n_steps", 1)) > 0


def _callbacks(config: dict, run_dir: Path):
    """The callbacks a run needs.

    Args:
        config: read for the scheduler block and
            training.early_stopping_patience.
        run_dir: where the best checkpoint is written.
    Returns:
        (callbacks, checkpoint_callback) -- the second is returned separately
        because the summary reads the best score off it.
    """
    checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=str(run_dir),
                                 filename="{epoch:02d}_{val_loss:.6f}",
                                 save_top_k=1, mode="min")
    cbs = [checkpoint, GradNorm(), EpochTimer()]
    # only when a schedule is on: with a constant lr the logged curve is a flat
    # line that says nothing, and metrics.csv stays comparable to older runs.
    # And only with a logger: Lightning refuses this callback without one.
    if wants_schedule(config) and _logging_enabled(config):
        cbs.append(LearningRateMonitor(logging_interval="epoch"))
    patience = int(config["training"].get("early_stopping_patience", 0))
    if patience > 0:
        cbs.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))
    return cbs, checkpoint


def _trainer(config: dict, run_dir: Path, callbacks) -> L.Trainer:
    """Build the Lightning trainer from the config's hardware and logging knobs.

    Args:
        config: read for the training section (epochs, accelerator, devices,
            precision, gradient clipping, logging rate).
        run_dir: default root dir, so the logs land with the checkpoint.
        callbacks: what _callbacks returned.
    Returns:
        The configured trainer.
    """
    t = config["training"]
    n_log = int(t.get("log_every_n_steps", 1))
    logging = ({"logger": None, "log_every_n_steps": n_log} if _logging_enabled(config)
               else {"logger": False})
    return L.Trainer(
        max_epochs=int(t["max_epochs"]),
        accelerator=t.get("accelerator", "auto"),
        devices=t.get("devices", 1),
        precision=t.get("precision", 32),
        gradient_clip_val=float(t.get("gradient_clip_val", 0.0)),
        callbacks=callbacks, default_root_dir=str(run_dir), **logging)


def _summary(model, trainer, checkpoint, wall_time_s: float) -> dict:
    """Key facts of a finished run.

    Args:
        model: the trained module, counted for its parameters.
        trainer: the finished trainer, read for the epochs actually run.
        checkpoint: the ModelCheckpoint holding the best score and path.
        wall_time_s: fit duration in seconds.
    Returns:
        The dict saved as summary.json next to the checkpoint -- best val_loss
        and its epoch, epochs run, wall time, parameter count, file name.
    """
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
    """Train a model into its own run dir. Generic: any pipeline's entrypoint
    can reuse this.

    Args:
        model: the Lightning module to fit.
        train_loader, val_loader: its data.
        config: the resolved config, archived as config.yaml -- it must be the
            one the model was built from, or evaluation would later rebuild a
            different network.
        project_root: repo root holding runs/.
    Returns:
        Path of the best checkpoint; the run dir also gets config.yaml and
        summary.json.
    """
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
    """Config dict -> trained checkpoint. Orchestrates assemble + fit_and_save.

    Args:
        config: the resolved config, left untouched -- the labels and dims the
            run needs are added to a copy, which is what gets archived.
        project_root: repo root holding runs/.
    Returns:
        Path of the best checkpoint.
    """
    torch.manual_seed(int(config.get("training", {}).get("seed", 42)))
    model, train_loader, val_loader, resolved = assemble(with_labels(config))
    return fit_and_save(model, train_loader, val_loader, resolved, project_root)


def train(config_path: Path, project_root: Path = PROJECT_ROOT,
          input_order: str | None = None, max_epochs: int | None = None) -> Path:
    """Same, starting from a yaml path.

    Args:
        config_path: the pipeline config (extends resolved by load_pipeline_config).
        project_root: repo root holding runs/.
        input_order: launch-time override of the channel order, which is what
            lets the order sweep run one arm per order off a single yaml.
        max_epochs: launch-time override of the epoch count (quick trials).
    Returns:
        Path of the best checkpoint.
    """
    config = load_pipeline_config(config_path)
    if input_order:
        config["dataset"]["input_order"] = input_order
    if max_epochs:
        config["training"]["max_epochs"] = max_epochs
    return train_config(config, project_root)


def val_loss_from_checkpoint(ckpt: Path) -> float:
    """Read a run's score straight off its checkpoint name (our naming contract
    in _callbacks), so a sweep needs no second inference pass.

    Args:
        ckpt: the checkpoint path.
    Returns:
        The val_loss it encodes, NaN when the name does not carry one.
    """
    m = re.search(r"val_loss=([0-9.]+)", ckpt.stem)
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    """CLI entrypoint: train the config given by --config.

    Returns:
        Nothing; the run dir holds the checkpoint, config and summary.
    """
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
