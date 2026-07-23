#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared controller construction helpers used by train/test/simulators.

All executable entrypoints call into this module so architecture selection is
defined exactly once. That keeps recurrent, liquid, and feedforward baselines
aligned across training, testing, and closed-loop simulation.
"""

from __future__ import annotations

from typing import Sequence

import ncps as kncp
import torch


from .feedforward import FeedForwardSequenceController
from .liquid_networks import CFC, LTC, ConvCfC, MLPCfC, TimespanCfC
from .standard_networks import CTRNN, GRU, LSTM, SimpleRNN


WIDTH_VARIANTS = {
    "eighth": 1.0 / 8.0,
    "quarter": 1.0 / 4.0,
    "half": 1.0 / 2.0,
    "base": 1.0,
    "double": 2.0,
}

# The families backed by ncps, i.e. the only ones that read the timespans and
# want them broadcast to the state size. The torch recurrent baselines
# (gru/lstm/ctrnn/simplernn) take the argument and ignore it.
NCPS_FAMILIES = {"cfc", "ltc", "ncp"}


def scaled_int(value: int | None, scale: float, minimum: int = 1) -> int | None:
    """Scale one width/depth parameter.

    Args:
        value: the base size, None passing through untouched.
        scale: the multiplier (see resolve_scale_factor).
        minimum: floor, so a small layer never scales down to nothing.
    Returns:
        The scaled size, rounded to an int.
    """
    if value is None:
        return None
    return max(minimum, int(round(float(value) * scale)))


def scaled_layers(values: Sequence[int], scale: float, minimum: int = 1) -> list[int]:
    """Scale a whole list of hidden-layer sizes with one multiplier.

    Args:
        values: the base layer sizes.
        scale: the multiplier.
        minimum: per-layer floor.
    Returns:
        The scaled sizes, same length.
    """
    return [max(minimum, int(round(float(value) * scale))) for value in values]


def resolve_scale_factor(config: dict) -> float:
    """The global width multiplier a config asks for.

    Args:
        config: the resolved config -- `model.scale_factor` wins when present,
            otherwise the named `model_variant.width` is looked up.
    Returns:
        That factor, 1.0 for an unknown variant name.
    """
    model_cfg = config.get("model", {})
    if "scale_factor" in model_cfg:
        return float(model_cfg["scale_factor"])

    model_variant = config.get("model_variant", {})
    variant_name = str(model_variant.get("width", "base")).lower()
    return WIDTH_VARIANTS.get(variant_name, 1.0)


def supports_recurrent_state(config: dict) -> bool:
    """Whether the configured model carries a hidden state.

    Args:
        config: the resolved config.
    Returns:
        False for the feedforward baselines (mlp/ff_mlp/nn), True for every
        recurrent family -- callers use it to decide whether to thread a
        hidden state through a rollout.
    """
    return str(config.get("model", {}).get("type", "ltc")).lower() not in {"mlp", "ff_mlp", "nn"}


def build_ncp_wiring(model_cfg: dict, output_size: int, scale_factor: float = 1.0):
    """Construct an NCP wiring from config.

    Args:
        model_cfg: the config's model section, read for its `ncp` block; the
            defaults match the older hard-coded experiment script, so the
            refactored pipeline reproduces those runs without custom edits.
        output_size: the command count, which fixes the motor neurons.
        scale_factor: the global width multiplier.
    Returns:
        The ncps NCP wiring object.
    """
    ncp_cfg = model_cfg.get("ncp", {})
    inter_neurons = scaled_int(ncp_cfg.get("inter_neurons", 32), scale_factor, minimum=1)
    command_neurons = scaled_int(ncp_cfg.get("command_neurons", 24), scale_factor, minimum=1)
    sensory_fanout = scaled_int(ncp_cfg.get("sensory_fanout", 20), scale_factor, minimum=1)
    inter_fanout = scaled_int(ncp_cfg.get("inter_fanout", 16), scale_factor, minimum=1)
    recurrent_command_synapses = scaled_int(
        ncp_cfg.get("recurrent_command_synapses", 16),
        scale_factor,
        minimum=0,
    )
    motor_fanin = scaled_int(ncp_cfg.get("motor_fanin", 20), scale_factor, minimum=1)

    return kncp.wirings.NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=output_size,
        sensory_fanout=sensory_fanout,
        inter_fanout=inter_fanout,
        recurrent_command_synapses=recurrent_command_synapses,
        motor_fanin=motor_fanin,
    )


def build_recurrent_module(model_cfg: dict, input_size: int, output_size: int, scale_factor: float = 1.0) -> torch.nn.Module:
    """Build the recurrent core alone, without any encoder in front of it.

    Args:
        model_cfg: the config's model section; `type` selects the family (cfc,
            ltc, ncp, simplernn, ctrnn, gru, lstm, or a feedforward mlp) and
            `no_neurons_layer` its width.
        input_size: what actually reaches the core -- the encoder's output
            width when a conv/mlp block precedes it, not the dataset's channel
            count.
        output_size: the command count.
        scale_factor: the global width multiplier.
    Returns:
        The module.
    Raises:
        ValueError: unknown model type.
    """
    model_type = str(model_cfg.get("type", "ltc")).lower()
    hidden_units = scaled_int(model_cfg.get("no_neurons_layer", 64), scale_factor, minimum=1)

    if model_type == "cfc":
        # CfC exposes a few extra backbone options compared with the other
        # recurrent families, so they are threaded through explicitly here.
        backbone_units = scaled_int(model_cfg.get("backbone_units", hidden_units), scale_factor, minimum=1)
        return CFC(
            input_size=input_size,
            units=hidden_units,
            proj_size=output_size,
            batch_first=True,
            mode=model_cfg.get("cfc_mode", "default"),
            mixed_memory=model_cfg.get("mixed_memory", False),
            activation=model_cfg.get("activation", "lecun_tanh"),
            backbone_units=backbone_units,
            backbone_layers=model_cfg.get("backbone_layers", 1),
            backbone_dropout=model_cfg.get("backbone_dropout", 0.0),
        )

    if model_type == "ltc":
        wiring = kncp.wirings.FullyConnected(hidden_units, output_dim=output_size)
        return LTC(
            input_size=input_size,
            units=wiring,
            batch_first=True,
            ode_unfolds=model_cfg.get("ode_unfolds", 6),
            mixed_memory=model_cfg.get("mixed_memory", False),
        )

    if model_type == "ncp":
        wiring = build_ncp_wiring(model_cfg, output_size, scale_factor=scale_factor)
        return CFC(
            input_size=input_size,
            units=wiring,
            proj_size=output_size,
            batch_first=True,
            mixed_memory=model_cfg.get("mixed_memory", False),
            activation=model_cfg.get("activation", "lecun_tanh"),
        )

    if model_type == "simplernn":
        return SimpleRNN(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "ctrnn":
        return CTRNN(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "gru":
        return GRU(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type == "lstm":
        return LSTM(input_dim=input_size, hidden_dim=hidden_units, output_dim=output_size)

    if model_type in {"mlp", "ff_mlp", "nn"}:
        hidden_layers = model_cfg.get("hidden_layers")
        if not hidden_layers:
            # Fall back to a simple two-layer MLP sized from the recurrent width
            # knob when no explicit feedforward widths were provided.
            hidden_layers = [hidden_units, hidden_units]
        return FeedForwardSequenceController(
            input_dim=input_size,
            output_dim=output_size,
            hidden_layers=scaled_layers(hidden_layers, scale_factor, minimum=1),
            activation=model_cfg.get("activation", "relu"),
            clamp_output=model_cfg.get("clamp_output", False),
        )

    raise ValueError(f"Unsupported model type '{model_cfg.get('type')}'.")


def build_controller_network(config: dict, input_dim: int, output_dim: int) -> torch.nn.Module:
    """The single factory every entrypoint calls, so architecture selection is
    defined exactly once.

    Supported knobs: `model.type` (cfc, ltc, ncp, ctrnn, simplernn, gru, lstm,
    mlp), `model.no_neurons_layer`, `model.cfc_mode`, `model.ncp.*`,
    `model.backbone_*`, `model.scale_factor`, and the optional encoder in front
    of the core -- `conv_block.output_dim` or `mlp_block.no_layers`.

    Args:
        config: the resolved config.
        input_dim: the dataset's input channels, dt excluded.
        output_dim: the command count.
    Returns:
        The assembled network: encoder + recurrent core, wrapped in TimespanCfC
        when the dataset feeds a dt channel (ncps needs the timespans expanded
        to its state size; ConvCfC does that itself). The wrapper is gated on
        that flag, so a network built without dt keeps its exact state dict.
    Raises:
        ValueError: an unsupported combination -- a feedforward baseline asked
            to sit behind an encoder, an empty mlp_block, or a conv_block
            without sequencing.
    """
    scale_factor = resolve_scale_factor(config)
    mlp_cfg = config.get("mlp_block", {})
    conv_cfg = config.get("conv_block", {})
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "ltc")).lower()

    if model_type in {"mlp", "ff_mlp", "nn"} and (mlp_cfg.get("value") or conv_cfg.get("value")):
        raise ValueError("Feedforward baselines currently do not support conv_block/mlp_block preprocessing.")

    # sequencing produces a rank-4 tensor (batch, time, window, features), which
    # only an encoder can consume: a bare recurrent core would take it straight
    # into ncps and die there on a dimension count
    if config.get("sequencing", {}).get("value", False) and not (
            mlp_cfg.get("value", False) or conv_cfg.get("value", False)):
        raise ValueError("sequencing.value=true needs an encoder in front of the recurrent "
                         "core (conv_block or mlp_block); set it to false otherwise.")

    if mlp_cfg.get("value", False):
        mlp_layers = scaled_layers(mlp_cfg.get("no_layers", []), scale_factor, minimum=1)
        if not mlp_layers:
            raise ValueError("MLP block requested but `mlp_block.no_layers` is empty.")
        rnn_input_dim = mlp_layers[-1]
    elif conv_cfg.get("value", False):
        rnn_input_dim = scaled_int(conv_cfg.get("output_dim", conv_cfg.get("base_channels", 256),), scale_factor, minimum=8)
    else:
        rnn_input_dim = input_dim

    recurrent_module = build_recurrent_module(model_cfg, rnn_input_dim, output_dim, scale_factor=scale_factor)

    if model_type in {"mlp", "ff_mlp", "nn"}:
        return recurrent_module

    # A dataset that appends a per-frame dt channel feeds timespans to the
    # recurrent core, and ncps needs them expanded to its state size. ConvCfC
    # below does that itself; the other paths need the wrapper. Gated on the flag
    # so networks built without a dt channel keep their exact state dict, and on
    # the ncps families: gru/lstm/ctrnn/simplernn ignore timespans entirely, so
    # wrapping them would buy nothing and rename their parameters
    # (model.gru.* -> model.rnn.gru.*), which no earlier checkpoint could reload.
    with_timespans = (bool(config.get("dataset", {}).get("use_dt", False))
                      and model_type in NCPS_FAMILIES)

    if mlp_cfg.get("value", False):
        # the encoder is sized for ONE frame, so a window longer than one frame
        # would reach it flattened (seq_len x features) and fail deep inside a
        # Linear on a shape mismatch; say so here instead
        if int(config.get("sequencing", {}).get("seq_len", 1)) > 1 and \
                config.get("sequencing", {}).get("value", False):
            raise ValueError("mlp_block reads one frame at a time: use sequencing.seq_len=1 "
                             "(or the conv encoder, whose kernel is what consumes a window).")
        mlp = MLPCfC(
            no_input=input_dim,
            layer_sizes=scaled_layers(mlp_cfg["no_layers"], scale_factor, minimum=1),
            rnn_module=recurrent_module,
        )
        return TimespanCfC(mlp) if with_timespans else mlp

    if conv_cfg.get("value", False):
        if not config.get("sequencing", {}).get("value", False):
            raise ValueError("conv_block requires sequencing.value=true.")
        seq_len = int(config["sequencing"]["seq_len"])
        return ConvCfC(
            no_input=seq_len,
            rnn_module=recurrent_module,
            width="base",
            base_channels=scaled_int(conv_cfg.get("output_dim", 256), scale_factor, minimum=8),
        )

    return TimespanCfC(recurrent_module) if with_timespans else recurrent_module
