# -*- coding: utf-8 -*-
"""
File: lightning.py

Purpose:
    Defines drone dynamics and a PyTorch Lightning training framework for
    testing various neural networks. Includes continuous-time and
    discrete-time integration of drone state equations (`dydt` and
    `dynamics`), and a LightningModule (`Lightning_Model`) orchestrating
    open-loop and closed-loop training, validation, and testing with
    configurable loss components and logging.

Key Functionality:
    - `dydt`: computes state derivatives based on physical constants
      and control inputs.
    - `dynamics`: integrates state over time steps to simulate position
      sequences.
    - `Lightning_Model`: wraps a neural network model, handling data
      normalization, loss computation (command vs. positional), optimizer
      setup, and storing predictions for analysis.
"""

import torch
import torch.nn as nn
import lightning as L  # PyTorch Lightning core API
import numpy as np
import time
import utils.normalization_limits as norm  # physical constants
from utils.data import get_norm_vectors  # normalization helper

class Lightning_Model(L.LightningModule):
    """
    PyTorch LightningModule for training trajectory flight of drones.

    Manages training, validation, and testing in open-loop vs. closed-loop
    modes and logs losses and predictions for analysis.
    """
    def __init__(self, model, config):
        """Wrap a controller network for training.

        Args:
            model: the pytorch network to optimize.
            config: the resolved config as a dictionary -- training.lr sets the
                optimizer, and dataset.input_labels decides whether the last
                input channel is a time channel ('t' or 'dt') to be split off
                as timespans.
        Returns:
            Nothing; also saves the config as hyperparameters and prepares the
            lists that test_step fills with predictions.
        """
        super().__init__()
        # Hyperparameters and model
        self.lr = config['training']['lr']
        self.model = model
        self.criterion = nn.MSELoss()
        self.save_hyperparameters(config)

        # Storage for test-time outputs
        self.all_yhat = []
        self.all_target = []
        self.all_runtime = []

        # Mode determines loss computation
        # Closed-loop training is intentionally retired in this refactor. The
        # controller is trained consistently in open-loop and evaluated in
        # closed-loop inside the simulator scripts.
        self.mode = "open-loop"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        # Check if time features are used
        self.with_time = (('t' in config['dataset']['input_labels'])
                          or ('dt' in config['dataset']['input_labels']))
        # Precompute normalization bounds
        self.global_min, self.global_max = get_norm_vectors(config['dataset']['input_labels'])
        self.global_min, self.global_max = torch.tensor(self.global_min), torch.tensor(self.global_max)
        self.global_min, self.global_max = self.global_min.permute(0, 2, 1), self.global_max.permute(0, 2, 1)

    def forward(self, x, hx=None, timespans=None):
        """Run the wrapped network.

        Args:
            x: inputs, time channel included when the config declared one.
            hx: recurrent hidden state, None to start fresh.
            timespans: explicit CfC timespans; when omitted and the config
                declares a time channel, the last channel of `x` is used.
        Returns:
            Whatever the network returns (a tensor, or a (prediction, state)
            pair for the recurrent ones).
        """
        if self.with_time and timespans is None:
            timespans = x[:, :, -1:]
            x = x[:, :, :-1]
        return self.model(x, hx, timespans)

    def configure_optimizers(self):
        """Lightning hook: the optimizer.

        Returns:
            Adam at the config's lr, with no schedule -- utils.scheduler
            subclasses this to add one.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=self.config["training"].get("lr_factor", 0.5),
        #     patience=self.config["training"].get("lr_patience", 3),
        #     min_lr=self.config["training"].get("lr_min", 0.0),
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val_loss",
            #     "interval": "epoch",
            #     "frequency": 1,
            # },
        }

    def training_step(self, batch):
        """Lightning hook: one optimization step.

        Args:
            batch: the (inputs, labels) pair.
        Returns:
            The training loss, logged as `train_loss`.
        """
        self.model.train()
        return self._training_step_OL(batch, log_text='train_loss')

    def _split_time_channel(self, x):
        """Separate the optional time feature from the model inputs.

        Training, validation and testing all consume the same
        `(x, timespans)` representation, so there is only one place that needs
        to understand how time is packed into the dataset tensors.

        Args:
            x: the batch inputs, rank 3 or 4.
        Returns:
            (inputs without the time channel, timespans), or (x, None) when the
            config declares no time channel.
        Raises:
            ValueError: the input rank is neither 3 nor 4.
        """
        if not self.with_time:
            return x, None

        if x.ndim == 3:
            return x[:, :, :-1], x[:, :, -1:]

        if x.ndim == 4:
            return x[:, :, :, :-1], x[:, :, :, -1:]

        raise ValueError(f"Unsupported input rank for time splitting: {x.shape}")

    def _unwrap_model_output(self, output):
        """Reduce any network's output to a prediction tensor.

        Args:
            output: what the network returned -- recurrent models return a
                (prediction, hidden state) pair, feedforward ones a tensor.
        Returns:
            The prediction alone.
        """
        return output[0] if isinstance(output, tuple) else output

    def _forward_open_loop(self, x, timespans=None, stepwise=False):
        """Run the controller through the same pathway used for optimization.

        Args:
            x: inputs, time channel already split off.
            timespans: the CfC timespans, or None.
            stepwise: True replays the sequence one frame at a time, carrying
                the hidden state -- the rollout used at test time, closer to
                the simulator's execution path. False runs the whole batch at
                once, which is what training does.
        Returns:
            The predictions, identical in shape either way.
        """
        if not stepwise:
            return self._unwrap_model_output(self.model(x, timespans=timespans))

        hidden_state = None
        predictions = []
        for i in range(x.shape[1]):
            step_timespans = None if timespans is None else timespans[:, i:(i + 1), ...]
            output = self.model(x[:, i:(i + 1), ...], hx=hidden_state, timespans=step_timespans)
            y_hat = self._unwrap_model_output(output)
            hidden_state = output[1] if isinstance(output, tuple) else None
            predictions.append(y_hat)

        return torch.cat(predictions, dim=1)

    def _open_loop_loss(self, y_hat, y):
        """Compute the command-space MSE, once for every phase.

        Args:
            y_hat: the predictions.
            y: the expert commands.
        Returns:
            The MSE, replaced by 1e3 when it comes out NaN so a diverged batch
            costs the run a bad score instead of poisoning every weight.
        """
        x_hat = y_hat.reshape(-1, y.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        loss = self.criterion(x_hat, y_flat)
        if torch.isnan(loss):
            loss = torch.tensor(1e3, dtype=loss.dtype, device=loss.device)
        return loss

    def _training_step_OL(self, batch, log_text='train_loss'):
        """Open loop: score the commands only, which is what both training and
        validation do.

        Args:
            batch: the (inputs, labels) pair.
            log_text: metric name, the only thing separating a training step
                from a validation one.
        Returns:
            The loss, logged under that name.
        """
        x, y = batch
        x, timespans = self._split_time_channel(x)
        y_hat = self._forward_open_loop(x, timespans=timespans)
        loss = self._open_loop_loss(y_hat, y)
        self.log(log_text, loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        """Lightning hook: score one validation batch.

        Args:
            batch: the (inputs, labels) pair.
        Returns:
            The loss, logged as `val_loss` -- the metric checkpointing and
            early stopping watch.
        """
        self.model.eval()
        return self._training_step_OL(batch, log_text='val_loss')

    def test_step(self, batch):
        """Lightning hook: run inference and keep the predictions.

        Args:
            batch: the (inputs, labels) pair.
        Returns:
            {'test_loss': the loss}; the predictions, targets and runtime are
            appended to all_yhat/all_target/all_runtime, which is where
            evaluate_arrays reads them from. Every sample of the batch is
            kept -- storing only the first scored one portion per batch, i.e.
            1/batch_size of the split.
        """
        self.model.eval()
        x, y = batch
        x, timespans = self._split_time_channel(x)

        # Test keeps the stepwise rollout because it is closer to the simulator
        # execution path and gives a more representative runtime measurement.
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        y_hat = self._forward_open_loop(x, timespans=timespans, stepwise=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.time()
        self.all_runtime.append((end - start))

        loss = self._open_loop_loss(y_hat, y)
        # Collect every sample of the batch, not just the first one: metrics
        # computed downstream (MSE, per-channel R2, ablations) must see the whole
        # split. Keeping y_hat[0] scored one portion per batch, i.e. 1/batch_size
        # of the validation data.
        self.all_yhat.append(y_hat)
        self.all_target.append(y)
        self.log('test_loss', loss, sync_dist=True)
        return {'test_loss': loss}

    def backward(self, loss):
        """Lightning hook: the backward pass, keeping the graph alive.

        Args:
            loss: the value to differentiate; a loss detached from the graph
                (the NaN fallback above) is made differentiable again first.
        Returns:
            Nothing; gradients land on the parameters.
        """
        if not loss.requires_grad:
            loss.requires_grad=True
        loss.backward(retain_graph=True)
