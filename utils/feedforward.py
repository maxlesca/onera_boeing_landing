#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedforward controller blocks used as recurrent-free baselines.

The refactor keeps a plain MLP option inside the same train/test/simulator
pipeline so recurrent and non-recurrent controllers can be compared without
maintaining a separate code path.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _activation(name: str) -> type[nn.Module]:
    """Map a config string to the corresponding PyTorch activation class."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name == "mish":
        return nn.Mish
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported feedforward activation '{name}'.")


class FeedForwardSequenceController(nn.Module):
    """
    Plain MLP controller applied independently at every prediction step.

    The module follows the same `(output, hidden_state)` contract used by the
    recurrent models so the Lightning wrapper and simulators can treat it as a
    drop-in baseline.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: Sequence[int],
                 activation: str = "relu",
                 clamp_output: bool = False) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one entry.")

        activation_cls = _activation(activation)
        # The network is intentionally simple: repeated Linear + activation,
        # followed by a final projection to motor commands.
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if clamp_output:
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hx=None, timespans=None):
        if x.ndim == 4:
            # Convolution-style datasets keep an explicit history axis; the
            # feedforward baseline only consumes the latest timestep.
            x = x[:, :, -1, :]
        if x.ndim != 3:
            raise ValueError(f"Expected rank-3 or rank-4 sequence input, got {x.shape}.")

        # Flatten time into the batch dimension, run the MLP independently on
        # each step, then restore the sequence shape expected by Lightning.
        batch_size, seq_len, feat_dim = x.shape
        y = self.network(x.reshape(batch_size * seq_len, feat_dim))
        y = y.reshape(batch_size, seq_len, -1)
        return y, None


__all__ = ["FeedForwardSequenceController"]
