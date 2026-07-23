# -*- coding: utf-8 -*-
"""
File: networks.py

Purpose:
    Defines custom neural network building blocks for continuous-time and
    discrete-time recurrent architectures beyond standard LNNs. Includes:
      - CellModel: a single update unit combining input and recurrent weights.
      - CT_RNN_Cell: continuous-time RNN cell with learnable time constants.
      - CT_RNN: multi-layer continuous-time recurrent network using CT_RNN_Cell.
      - SimpleRNN: a wrapper around nn.RNN for discrete-time sequence modeling.

Key Functionality:
    - Implements state updates via Euler-like integration in CT_RNN.
    - Learns per-neuron time scales and gating via parameters `a` and `tau`.
    - Provides flexible sequence-to-sequence output for time-series tasks.
"""

import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    """
    Wrapper around PyTorch's nn.RNN for discrete-time sequence modeling.

    !!! Can be run only with batch_size = 1 for the moment !!! TODO:

    Performs stepwise unrolling and outputs full sequence.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        """Build the RNN and its readout.

        Args:
            input_dim: input channels per frame.
            hidden_dim: recurrent width.
            output_dim: command count.
        Returns:
            Nothing.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Batch-first RNN layer
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        # Linear projection for each time step
        self.fc = nn.Linear(hidden_dim, output_dim)

    @property
    def state_size(self):
        """The hidden state width, which callers need to size a rollout.

        Returns:
            hidden_dim.
        """
        return self.hidden_dim

    def forward(self,
                x: torch.Tensor,
                hx: torch.Tensor = None,
                timespans: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Step through the input sequence one frame at a time.

        Args:
            x: inputs [batch, seq_length, input_dim].
            hx: initial hidden state [batch, 1, hidden_dim], None to start
                from zero.
            timespans: unused, present for API consistency with the CfC.
        Returns:
            (readout [batch, seq_length, output_dim], final hidden state).
        """
        batch_size, seq_len, _ = x.size()
        outputs = []
        h_state = hx
        # Unroll manually to capture hidden at each step
        for t in range(seq_len):
            # RNN expects input [batch, input_dim]
            inp = x[:, t]
            h_out, h_state = self.rnn(inp.unsqueeze(1), h_state)
            # Project hidden to output dimension
            out_t = self.fc(h_out.squeeze(1))
            outputs.append(out_t)
        # Stack outputs along time
        readout = torch.stack(outputs, dim=1)
        return readout, h_state


class RecurrentLayer(torch.nn.Module):
    """The continuous-time recurrent layer inside CTRNN: a leaky update with a
    fixed time constant, plus optional pre/post-activation noise."""

    def __init__(self, input_dim, hidden_size):
        """Build the input and hidden projections.

        Args:
            input_dim: input channels per frame.
            hidden_size: recurrent width.
        Returns:
            Nothing.
        """
        super().__init__()
        self.alpha = 0.95  # time constant
        self.preact_noise, self.postact_noise = 0.0, 0.0
        self.activation = torch.sigmoid
        self.hidden_size = hidden_size
        self.input_layer = torch.nn.Linear(input_dim, hidden_size)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
    
    def recurrence(self, fr_t, v_t, u_t):
        """One update step of the leaky recurrence.

        Args:
            fr_t: current firing rates (the activated state).
            v_t: current membrane potentials (the pre-activation state).
            u_t: this frame's input.
        Returns:
            (fr_t, v_t) after the step: the potentials blend the previous ones
            with the new drive through alpha, and the rates are their
            activation. Noise is added only when the corresponding attribute
            was set above 0.
        """
        # through input layer
        w_in_u_t = self.input_layer(u_t)  # u_t @ W_in
        # through hidden layer
        w_hid_fr_t = self.hidden_layer(fr_t)  # fr_t @ W_hid + b
        # update hidden state
        v_t = (1-self.alpha)*v_t + self.alpha*(w_hid_fr_t+w_in_u_t)
    
        # add pre-activation noise
        if self.preact_noise > 0:
            preact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.preact_noise
            v_t = v_t + self.alpha*preact_epsilon
    
        # apply activation function
        fr_t = self.activation(v_t)
    
        # add post-activation noise
        if self.postact_noise > 0:
            postact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.postact_noise
            fr_t = fr_t + postact_epsilon
        
        return fr_t, v_t
    
    def forward(self, input, fr_t=None, timespans=None):
        """Propagate a whole sequence through the layer.

        Args:
            input: shape (seq_len, batch, input_dim) -- time first here, unlike
                the batch-first modules around it.
            fr_t: initial firing rates, None to start from the activation of a
                zero state.
            timespans: unused, present for API consistency.
        Returns:
            The hidden states stacked over time, (seq_len, batch, hidden_size).
        """
        v_t = torch.zeros((input.size(1), self.hidden_size), device=input.device)
        if fr_t is None:
            fr_t = self.activation(v_t)
        # update hidden state and append to stacked_states
        stacked_states = []
        for i in range(input.size(0)):
            fr_t, v_t = self.recurrence(fr_t, v_t, input[i])
            # append to stacked_states
            stacked_states.append(fr_t)
  
        return torch.stack(stacked_states, dim=0)


class CTRNN(torch.nn.Module):
    """Continuous-time RNN: a RecurrentLayer plus a linear readout."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Build the recurrent layer and its readout.

        Args:
            input_dim: input channels per frame.
            hidden_dim: recurrent width.
            output_dim: command count.
        Returns:
            Nothing.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.recurrent_layer = RecurrentLayer(input_dim, hidden_dim)
        self.readout_layer = torch.nn.Linear(hidden_dim, output_dim)

    @property
    def state_size(self):
        """The hidden state width.

        Returns:
            hidden_dim.
        """
        return self.hidden_dim

    def forward(self, inputs, hx=None, timespans=None):
        """Run the sequence and read the commands out of the hidden states.

        Args:
            inputs: (batch, seq_len, input_dim); transposed internally, the
                recurrent layer working time-first.
            hx: initial firing rates, None to start from zero.
            timespans: unused, present for API consistency.
        Returns:
            (predictions (batch, seq_len, output_dim), the LAST hidden state).
        """
        inputs = inputs.permute(1, 0, 2)  # (seq_len, batch, input_dim)
        hx = self.recurrent_layer(inputs, hx, timespans)
        output = self.readout_layer(hx.float())
        output = output.permute(1, 0, 2)  # (batch, seq_len, output_dim)
        hx = hx.permute(1, 0, 2)  # (batch, seq_len, hidden_dim)  
        hx = hx[:, -1, :]
        return output, hx



class GRU(torch.nn.Module):
    """Gated recurrent unit baseline: torch's GRU plus a linear readout."""

    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True):
        """Build the GRU and its readout.

        Args:
            input_dim: input channels per frame.
            hidden_dim: recurrent width.
            output_dim: command count.
            batch_first: keep the batch on axis 0, as the rest of the pipeline
                does.
        Returns:
            Nothing.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=batch_first)
        self.readout_layer = torch.nn.Linear(hidden_dim, output_dim)

    @property
    def state_size(self):
        """The hidden state width.

        Returns:
            hidden_dim.
        """
        return self.hidden_dim

    def forward(self, inputs, hx=None, timespans=None):
        """Run the sequence.

        Args:
            inputs: (batch, seq_len, input_dim).
            hx: initial hidden state (batch, hidden_dim), None to start from
                zero; it is given the layer axis torch expects.
            timespans: unused, present for API consistency.
        Returns:
            (predictions (batch, seq_len, output_dim), the final hidden state
            without its layer axis).
        """
        if hx is not None:
            hx = hx.unsqueeze(0)  # (1, batch, hidden_dim)
        hidden_states, hx = self.gru(inputs, hx)
        output = self.readout_layer(hidden_states.float())
        return output, hx[-1, :, :]


class LSTM(torch.nn.Module):
    """LSTM baseline: torch's LSTM plus a linear readout."""

    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True):
        """Build the LSTM and its readout.

        Args:
            input_dim: input channels per frame.
            hidden_dim: recurrent width.
            output_dim: command count.
            batch_first: keep the batch on axis 0.
        Returns:
            Nothing.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=batch_first)
        self.readout_layer = torch.nn.Linear(hidden_dim, output_dim)

    @property
    def state_size(self):
        """The hidden state width (the cell state has the same one).

        Returns:
            hidden_dim.
        """
        return self.hidden_dim

    def forward(self, inputs, hx=None, timespans=None):
        """Run the sequence.

        Args:
            inputs: (batch, seq_len, input_dim).
            hx: initial (hidden, cell) pair, None to start from zero.
            timespans: unused, present for API consistency.
        Returns:
            (predictions (batch, seq_len, output_dim), the final (hidden, cell)
            pair) -- a pair, unlike the other families, which is why callers
            must not assume the state is a single tensor.
        """
        if hx is not None:
            h_n, c_n = hx
            h_n = h_n.unsqueeze(0)  # (1, batch, hidden_dim)
            c_n = c_n.unsqueeze(0)  # (1, batch, hidden_dim)
            hx = (h_n, c_n)  
        hidden_states, (h_n, c_n) = self.lstm(inputs, hx)
        output = self.readout_layer(hidden_states.float())
        return output, (h_n[-1, :, :], c_n[-1, :, :])
