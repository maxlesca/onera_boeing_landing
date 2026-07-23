# -*- coding: utf-8 -*-
"""Optional learning-rate schedule on top of the shared Lightning model.

Why: a constant lr has to be a compromise -- large enough to make progress early,
small enough not to bounce around the minimum late. A schedule removes the
compromise: move fast at the start, refine at the end. The previous study already
showed the symptom, val_loss reaching its best at epoch 3 then climbing back
(DOC 8.18.1), and that a plain lower lr only stabilised without fixing anything.

Off by default: `training.scheduler.type: none` keeps the constant learning rate,
so every earlier result stays reproducible and a schedule is always an explicit
choice, never something that silently changed underneath a comparison.

    training:
      lr: 0.001
      scheduler:
        type: cosine     # none | cosine | plateau | step
        min_lr: 1.0e-5   # cosine/plateau floor
        factor: 0.5      # plateau/step decay factor
        patience: 3      # plateau: epochs without val improvement before decaying
        step_size: 5     # step: decay every N epochs
        warmup_epochs: 0 # cosine: epochs of linear ramp-up before decaying

Lives in utils/ next to lightning.py and model_builder.py: a schedule is engine
level, not landing specific, so the quadrotor baseline can opt into it the same
way. ScheduledModel subclasses Lightning_Model and overrides one method, so the
engine keeps behaving exactly as before for every config that does not ask.
"""

from __future__ import annotations

import torch

from utils.lightning import Lightning_Model

# Schedules a config may ask for. "none" is not in the table: it means no
# scheduler object at all, i.e. the unchanged constant-lr path.
SCHEDULES = ("none", "cosine", "plateau", "step")


def scheduler_config(config: dict) -> dict:
    """The `training.scheduler` block, tolerating its absence (older configs)."""
    return dict(config.get("training", {}).get("scheduler") or {})


def wants_schedule(config: dict) -> bool:
    """True when the config asks for anything other than a constant lr."""
    return str(scheduler_config(config).get("type", "none")).lower() != "none"


def build_scheduler(optimizer, config: dict, max_epochs: int):
    """(scheduler, monitor) for the requested type, or (None, None) for constant
    lr. `monitor` is the metric a plateau schedule watches; the others ignore it.

    - cosine  : smooth decay lr -> min_lr over the whole run. Predictable and
                epoch-count-driven, so two arms sharing max_epochs follow the
                exact same curve -- the right default when comparing recipes.
    - plateau : decay only when validation stops improving. Adapts to the run,
                but the curve then differs between arms, which makes an A/B
                slightly less clean.
    - step    : decay by `factor` every `step_size` epochs. The blunt baseline.
    """
    cfg = scheduler_config(config)
    kind = str(cfg.get("type", "none")).lower()
    if kind not in SCHEDULES:
        raise SystemExit(f"unknown scheduler type {kind!r}; choose from {SCHEDULES}")
    if kind == "none":
        return None, None

    if kind == "cosine":
        warmup = int(cfg.get("warmup_epochs", 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup), eta_min=float(cfg.get("min_lr", 0.0)))
        if warmup <= 0:
            return cosine, None
        # linear ramp from 10% of lr, then hand over to the cosine decay
        ramp = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [ramp, cosine], milestones=[warmup]), None

    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 3)),
            min_lr=float(cfg.get("min_lr", 0.0))), "val_loss"

    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(cfg.get("step_size", 5)),
        gamma=float(cfg.get("factor", 0.5))), None


class ScheduledModel(Lightning_Model):
    """Lightning_Model with the optional schedule wired into the optimizer.

    Everything else -- losses, open/closed loop, logging -- is inherited
    untouched; only configure_optimizers is overridden.
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler, monitor = build_scheduler(
            optimizer, self.config, int(self.config["training"]["max_epochs"]))
        if scheduler is None:
            return {"optimizer": optimizer}
        lr_scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        if monitor:
            lr_scheduler["monitor"] = monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


def model_for(config: dict, network):
    """The Lightning module a config asks for: the scheduled subclass when a
    schedule is requested, the stock model otherwise. One place decides, so no
    caller has to know the rule."""
    return (ScheduledModel if wants_schedule(config) else Lightning_Model)(network, config)
