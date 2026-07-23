# -*- coding: utf-8 -*-
"""
Liquid-network helpers used by the quadrotor training pipeline.

The actual liquid recurrent models are now provided directly by the local
`ncps` library:

- `CFC` is an alias for `ncps.torch.CfC`
- `LTC` is an alias for `ncps.torch.LTC`

This module keeps only the project-specific wrappers that sit around those
recurrent blocks:

- convolutional preprocessing before the liquid core
- MLP preprocessing before the liquid core
- a custom multi-layer wiring helper
- the optional spiking-network baseline
"""

from __future__ import annotations

from typing import Iterable, Optional, Type

import ncps as kncp
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC as CFC
from ncps.torch import LTC
from ncps.torch.cfc_cell import LeCun


class ConvBlock(nn.Module):
    """Compact temporal feature extractor used before the recurrent core."""

    def __init__(self, no_input, width="base", base_channels=256):
        """Build the four strided convolutions.

        Args:
            no_input: channels the block reads -- for the landing pipeline this
                is the SEQUENCE length, the convolution sliding over the
                feature axis, which is why the channel order is a real
                hyperparameter.
            width: named multiplier on every layer (eighth to double).
            base_channels: width of the last layer before scaling.
        Returns:
            Nothing.
        Raises:
            ValueError: unknown width name.
        """
        super().__init__()

        mult_map = {
            "eighth": 1 / 8,
            "quarter": 1 / 4,
            "half": 1 / 2,
            "base": 1.0,
            "double": 2.0,
        }
        if width not in mult_map:
            raise ValueError(f"Unknown width: {width}")

        multiplier = mult_map[width]

        c1 = max(8, int(round(64 * multiplier)))
        c2 = max(8, int(round(128 * multiplier)))
        c3 = max(8, int(round(128 * multiplier)))
        c4 = max(8, int(round(base_channels * multiplier)))

        self.out_channels = c4
        self.conv1 = nn.Conv1d(no_input, c1, 5, padding=2, stride=2)
        self.conv2 = nn.Conv1d(c1, c2, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2, stride=2)
        self.conv4 = nn.Conv1d(c3, c4, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm1d(c4)

    def forward(self, x):
        """Encode one frame's features.

        Args:
            x: (batch, no_input, features).
        Returns:
            (batch, out_channels): the four convolutions, then a mean over the
            remaining feature axis.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.mean(-1)


class ConvCfC(nn.Module):
    """Apply convolutional preprocessing before a liquid recurrent module."""

    def __init__(self, no_input, rnn_module, width="base", base_channels=256):
        """Assemble encoder and recurrent core.

        Args:
            no_input: what ConvBlock reads (the sequence length).
            rnn_module: the recurrent core to run on the encoded frames.
            width: named width multiplier of the encoder.
            base_channels: encoder output width before scaling.
        Returns:
            Nothing.
        """
        super().__init__()
        self.conv_block = ConvBlock(no_input, width=width, base_channels=base_channels)
        self.rnn = rnn_module

    def forward(self, x, hx=None, timespans=None):
        """Encode every frame, then run the recurrent core over the sequence.

        Args:
            x: (batch, seq_len, ...) as the sequencing produced it.
            hx: recurrent hidden state, None to start fresh.
            timespans: per-frame dt; expanded here to the core's state size,
                which is what ncps expects (see TimespanCfC).
        Returns:
            Whatever the core returns, usually (predictions, hidden state).
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        
        if timespans is not None:
            timespans = timespans.view(batch_size, seq_len, -1)
            if timespans.shape[-1] == 1:
                timespans = timespans.expand(batch_size, seq_len, self.rnn.state_size)

        return self.rnn(x, hx, timespans)


class TimespanCfC(nn.Module):
    """Give a recurrent module the timespans shape ncps expects.

    `ncps.torch.CfC` wants timespans broadcast to its state size: a (B, T, 1)
    per-frame dt raises "The size of tensor a (units) must match the size of
    tensor b (T)". ConvCfC already expands it, which is why the conv path is the
    only one that ever worked with a dt channel; this wrapper does the same
    expansion for the paths without a convolutional front end (a bare CfC, or
    MLPCfC, which forwards timespans untouched).

    Opt-in: model_builder only inserts it when the dataset appends a dt channel,
    so networks built the way they were before keep the exact same state dict.
    """

    def __init__(self, rnn_module):
        """Wrap a recurrent module and find its state size.

        Args:
            rnn_module: the module to wrap. MLPCfC holds the recurrent core one
                level down, so the state size is looked up through the chain
                rather than on the wrapped module itself.
        Returns:
            Nothing.
        """
        super().__init__()
        self.rnn = rnn_module
        inner = rnn_module
        while not hasattr(inner, "state_size") and hasattr(inner, "rnn"):
            inner = inner.rnn
        self.state_size = getattr(inner, "state_size", None)

    def forward(self, x, hx=None, timespans=None):
        """Expand the timespans, then delegate.

        Args:
            x: the inputs.
            hx: recurrent hidden state.
            timespans: per-frame dt, broadcast here to the core's state size.
        Returns:
            Whatever the wrapped module returns.
        """
        if timespans is not None and self.state_size:
            timespans = timespans.reshape(x.size(0), x.size(1), -1)
            if timespans.shape[-1] == 1:
                timespans = timespans.expand(-1, -1, self.state_size)
        return self.rnn(x, hx, timespans)


class MLPBlock(nn.Module):
    """
    Flexible MLP that accepts `(B, F)`, `(B, T, F)`, or `(B, T, S, F)` inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        activation: Type[nn.Module] = LeCun,
        out_activation: Optional[Type[nn.Module]] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        bias: bool = True,
    ):
        """Build the Linear stack.

        Args:
            input_size: input width.
            hidden_sizes: width of each layer, the last one being the output.
            activation: activation class inserted between layers.
            out_activation: optional activation after the last layer.
            dropout: dropout probability between layers, 0 to disable.
            batch_norm: insert a LayerNorm between layers.
            bias: give the Linear layers a bias.
        Returns:
            Nothing.
        Raises:
            ValueError: no hidden size was given.
        """
        super().__init__()

        sizes = [input_size] + list(hidden_sizes)
        if len(sizes) < 2:
            raise ValueError("You must provide at least one hidden size.")

        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            in_features, out_features = sizes[i], sizes[i + 1]
            is_last = i == len(sizes) - 2

            layers.append(nn.Linear(in_features, out_features, bias=bias))
            if not is_last:
                if batch_norm:
                    layers.append(nn.LayerNorm(out_features))
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            elif out_activation is not None:
                layers.append(out_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP frame by frame, whatever the input rank.

        Args:
            x: (B, F), (B, T, F), or (B, T, S, F) -- the history axis of the
                rank-4 layout is flattened into the features.
        Returns:
            (B, out) or (B, T, out).
        Raises:
            ValueError: any other rank.
        """
        if x.ndim == 2:
            return self.net(x)
        if x.ndim == 3:
            batch_size, seq_len, feat_dim = x.shape
            y = self.net(x.reshape(batch_size * seq_len, feat_dim))
            return y.reshape(batch_size, seq_len, y.shape[-1])
        if x.ndim == 4:
            batch_size, seq_len, history_len, feat_dim = x.shape
            y = self.net(x.reshape(batch_size * seq_len, history_len * feat_dim))
            return y.reshape(batch_size, seq_len, y.shape[-1])
        raise ValueError(f"Expected rank-2/3/4 input, got {x.shape}")



class MLPCfC(nn.Module):
    """Apply MLP preprocessing before the liquid recurrent module."""

    def __init__(self, no_input, layer_sizes, rnn_module):
        """Assemble encoder and recurrent core.

        Args:
            no_input: input channels per frame.
            layer_sizes: the encoder's layer widths.
            rnn_module: the recurrent core.
        Returns:
            Nothing.
        """
        super().__init__()
        self.mlp_block = MLPBlock(no_input, layer_sizes)
        self.rnn = rnn_module

    def forward(self, x, hx=None, timespans=None):
        """Encode, then run the recurrent core.

        Args:
            x: the inputs.
            hx: recurrent hidden state.
            timespans: per-frame dt, forwarded untouched -- model_builder wraps
                this module in TimespanCfC when the dataset provides one.
        Returns:
            Whatever the core returns.
        """
        x = self.mlp_block(x)
        return self.rnn(x, hx, timespans)


class MultiLayerWiring(kncp.wirings.FullyConnected):
    """All-to-all multi-layer wiring helper for NCP-style experiments."""

    def __init__(self, neurons_per_layer, output_dim=None):
        """Lay the neurons out in layers.

        Args:
            neurons_per_layer: size of each layer, in order.
            output_dim: how many neurons of the last layer are motor neurons;
                defaults to the whole last layer.
        Returns:
            Nothing.
        Raises:
            ValueError: output_dim exceeds the last layer.
        """
        self._neurons_per_layer = neurons_per_layer
        self._num_layers = len(neurons_per_layer)
        self._layer_boundaries = np.cumsum(neurons_per_layer)
        total_neurons = np.sum(neurons_per_layer)
        if output_dim is None:
            output_dim = neurons_per_layer[-1]
        super().__init__(total_neurons, output_dim=output_dim)

        self._layer_neurons = []
        start = 0
        for layer_size in neurons_per_layer:
            end = start + layer_size
            self._layer_neurons.append(list(range(start, end)))
            start = end

        if output_dim > neurons_per_layer[-1]:
            raise ValueError(
                f"Output dimension {output_dim} cannot be larger than the last layer size {neurons_per_layer[-1]}."
            )

    @property
    def num_layers(self):
        """How many layers the wiring holds.

        Returns:
            That count.
        """
        return self._num_layers

    def get_neurons_of_layer(self, layer):
        """The neuron indices of one layer.

        Args:
            layer: its index.
        Returns:
            The list of neuron indices.
        Raises:
            ValueError: index out of range.
        """
        if layer < 0 or layer >= self._num_layers:
            raise ValueError("Layer index out of range")
        return self._layer_neurons[layer]

    def build(self, input_dim):
        """Wire the adjacency matrices, once the input width is known.

        Args:
            input_dim: the sensory input width; every input reaches every
                neuron.
        Returns:
            Nothing; fills the adjacency matrices -- each layer fully connected
            to the next and to itself -- and, when the output is narrower than
            the last layer, disconnects the neurons that are not motor ones.
        """
        super().build(input_dim)
        self._input_dim = input_dim
        self.sensory_adjacency_matrix = np.ones((self._input_dim, self.units), dtype=np.float32)
        self._sensory_erev_initializer = lambda: np.ones(self._input_dim) * 0.0
        self.adjacency_matrix = np.zeros((self.units, self.units), dtype=np.float32)

        for layer in range(1, self._num_layers):
            pre_neurons = self.get_neurons_of_layer(layer - 1)
            post_neurons = self.get_neurons_of_layer(layer)
            self.adjacency_matrix[np.ix_(pre_neurons, post_neurons)] = 1.0

        for layer in range(self._num_layers):
            layer_neurons = self.get_neurons_of_layer(layer)
            self.adjacency_matrix[np.ix_(layer_neurons, layer_neurons)] = 1.0

        if self.output_dim < self._neurons_per_layer[-1]:
            last_layer_neurons = self.get_neurons_of_layer(self._num_layers - 1)
            non_output_neurons = last_layer_neurons[:-self.output_dim]
            self._output_neurons = last_layer_neurons[-self.output_dim:]
            self.adjacency_matrix[non_output_neurons, :] = 0.0
            self.adjacency_matrix[:, non_output_neurons] = 0.0
            self.adjacency_matrix[non_output_neurons, non_output_neurons] = 1.0

        self._built = True
        self._sparsity_mask = np.where(self.adjacency_matrix != 0.0, 1.0, 0.0).astype(np.float32)
        self._sensory_sparsity_mask = np.where(self.sensory_adjacency_matrix != 0.0, 1.0, 0.0).astype(np.float32)
        self._num_synapses = int(np.sum(np.abs(self.adjacency_matrix))) + int(np.sum(np.abs(self.sensory_adjacency_matrix)))

    def get_config(self):
        """The wiring's serialisable configuration.

        Returns:
            What the ncps base class produces.
        """
        return super().get_config()


class SNN(nn.Module):
    """Optional spiking-network baseline kept for older experiments."""

    def __init__(
        self,
        input_dim=20,
        hidden_dim=64,
        output_dim=4,
        beta=0.5,
        reset_type="subtract",
        reset_delay=True,
        learn_beta=False,
        learn_threshold=False,
    ):
        """Build the three spiking layers and the readout.

        Args:
            input_dim: input channels per frame.
            hidden_dim: width of each spiking layer.
            output_dim: command count.
            beta: membrane decay of the leaky neurons.
            reset_type: snntorch reset mechanism ('subtract' or 'zero').
            reset_delay: snntorch reset timing flag.
            learn_beta: unused here, kept for call compatibility.
            learn_threshold: unused here, same reason.
        Returns:
            Nothing.
        """
        super().__init__()

        self.policy_fc1 = nn.Linear(input_dim, hidden_dim)
        self.policy_lif1 = snn.Leaky(beta=beta, reset_mechanism=reset_type, reset_delay=reset_delay)
        self.policy_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_lif2 = snn.Leaky(beta=beta, reset_mechanism=reset_type, reset_delay=reset_delay)
        self.policy_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_lif3 = snn.Leaky(beta=beta, reset_mechanism=reset_type, reset_delay=reset_delay)
        self.an = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hx=None, timespans=None, num_steps=10):
        """Run the spiking stack, frame by frame.

        Args:
            x: (batch, seq_len, features).
            hx: unused, present for API consistency.
            timespans: unused, same reason.
            num_steps: spiking steps simulated per input frame; the output is
                their mean spike rate, which is what makes the response
                differentiable.
        Returns:
            (predictions (batch, seq_len, output_dim), the spike rates that
            produced them).
        """
        latent_pi_list = []

        for timestep in range(x.size(1)):
            x_t = x[:, timestep, :]
            policy_mem1 = self.policy_lif1.reset_mem()
            policy_mem2 = self.policy_lif2.reset_mem()
            policy_mem3 = self.policy_lif3.reset_mem()

            spikes = []
            for _ in range(num_steps):
                spk1, policy_mem1 = self.policy_lif1(self.policy_fc1(x_t), policy_mem1)
                spk2, policy_mem2 = self.policy_lif2(self.policy_fc2(spk1), policy_mem2)
                spk3, policy_mem3 = self.policy_lif3(self.policy_fc3(spk2), policy_mem3)
                spikes.append(spk3)

            latent_pi = torch.stack(spikes).mean(dim=0)
            latent_pi_list.append(latent_pi)

        latent_pi = torch.stack(latent_pi_list, dim=1)
        output = self.an(latent_pi)
        return output, latent_pi


__all__ = ["CFC", "LTC", "ConvCfC", "MLPCfC", "MultiLayerWiring", "SNN", "LeCun"]
