#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared quadrotor simulation helpers for controller rollouts.

This module centralizes the repeated simulation logic that used to live in
multiple standalone scripts: state transforms, continuous-time dynamics,
discretization schemes, checkpoint-backed model loading, and closed-loop rollout.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .config import dataset_dims, resolve_checkpoint
from .data import get_norm_vectors
from .model_builder import build_controller_network
from .lightning import Lightning_Model
from .normalization_limits import (
    G, IXX, IYY, IZZ, KH, KOMEGA, KP, KPV, KQ, KQV, KR1, KR2, KRR, KX, KY, KZ,
    MASS, OMEGA_MAX, OMEGA_MIN, TAU,
)


STATE_LABELS = [
    "dx", "dy", "dz",
    "vx", "vy", "vz",
    "phi", "theta", "psi",
    "p", "q", "r",
    "Mx_ext", "My_ext", "Mz_ext",
    "omega1", "omega2", "omega3", "omega4",
]
STATE_INDEX = {label: idx for idx, label in enumerate(STATE_LABELS)}
LABEL_ALIASES = {"Mx": "Mx_ext", "My": "My_ext", "Mz": "Mz_ext"}


def input_labels_without_time(input_labels: list[str]) -> list[str]:
    """Drop the time channels when building simulator state vectors.

    Args:
        input_labels: the model's input channels.
    Returns:
        The same list without 't' and 'dt' -- those are fed to the recurrent
        core as timespans, not read off the simulated state.
    """
    return [label for label in input_labels if label not in {"t", "dt"}]


def rotation_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    """Build the body-to-world rotation matrix from Euler angles.

    Args:
        phi, theta, psi: roll, pitch and yaw in radians.
    Returns:
        The (3, 3) matrix, composed Rz @ Ry @ Rx.
    """
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    ry = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]])
    rz = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def world_to_body_state(state_world: np.ndarray) -> np.ndarray:
    """Express a world-frame state in the body frame.

    Args:
        state_world: the 19-entry state; only the position and velocity blocks
            are rotated, the rest passes through.
    Returns:
        A converted copy -- the input is never modified.
    """
    state_world = np.asarray(state_world, dtype=np.float64).copy()
    x, y, z, vx, vy, vz, phi, theta, psi = state_world[:9]
    rotation = rotation_body_to_world(phi, theta, psi)
    state_world[0:3] = -rotation.T @ np.array([x, y, z])
    state_world[3:6] = rotation.T @ np.array([vx, vy, vz])
    return state_world


def body_to_world_state(state_body: np.ndarray) -> np.ndarray:
    """Express a body-frame state back in the world frame.

    Args:
        state_body: the 19-entry state; only the position and velocity blocks
            are rotated.
    Returns:
        A converted copy.
    """
    state_body = np.asarray(state_body, dtype=np.float64).copy()
    dx, dy, dz, vx, vy, vz, phi, theta, psi = state_body[:9]
    rotation = rotation_body_to_world(phi, theta, psi)
    state_body[0:3] = -rotation @ np.array([dx, dy, dz])
    state_body[3:6] = rotation @ np.array([vx, vy, vz])
    return state_body


def body_to_world_trajectory(states_body: np.ndarray) -> np.ndarray:
    """Convert a whole rollout to the world frame, for plotting.

    Args:
        states_body: the states of a rollout, one per row.
    Returns:
        The same trajectory in world coordinates.
    """
    return np.asarray([body_to_world_state(state) for state in states_body], dtype=np.float64)


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Evaluate the continuous-time quadrotor dynamics at one point.

    Args:
        state: the 19 entries -- position, velocity, attitude, rates, external
            moments, and the four rotor speeds.
        action: the four motor commands in [0, 1].
    Returns:
        The state derivative, same layout. The external moments are treated as
        constants (their derivative is 0), which is what makes them a
        disturbance the controller has to compensate rather than drive.
    """
    (
        dx, dy, dz, vx, vy, vz,
        phi, theta, psi, p, q, r,
        mx, my, mz, omega1, omega2, omega3, omega4,
    ) = state
    u1, u2, u3, u4 = action

    d_dx = -q * dz + r * dy - vx
    d_dy = p * dz - r * dx - vy
    d_dz = -p * dy + q * dx - vz

    omegas = omega1 + omega2 + omega3 + omega4
    omegas2 = omega1**2 + omega2**2 + omega3**2 + omega4**2

    d_vx = -q * vz + r * vy - G * np.sin(theta) - KX * omegas * vx
    d_vy = p * vz - r * vx + G * np.cos(theta) * np.sin(phi) - KY * omegas * vy
    d_vz = (
        -p * vy + q * vx + G * np.cos(theta) * np.cos(phi)
        - KZ * omegas * vz - KOMEGA * omegas2 - KH * (vx**2 + vy**2)
    )

    d_phi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
    d_theta = q * np.cos(phi) - r * np.sin(phi)
    d_psi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

    d_omega1 = (OMEGA_MIN + u1 * (OMEGA_MAX - OMEGA_MIN) - omega1) / TAU
    d_omega2 = (OMEGA_MIN + u2 * (OMEGA_MAX - OMEGA_MIN) - omega2) / TAU
    d_omega3 = (OMEGA_MIN + u3 * (OMEGA_MAX - OMEGA_MIN) - omega3) / TAU
    d_omega4 = (OMEGA_MIN + u4 * (OMEGA_MAX - OMEGA_MIN) - omega4) / TAU

    tau_x = KP * (omega1**2 - omega2**2 - omega3**2 + omega4**2) + KPV * vy + mx
    tau_y = KQ * (omega1**2 + omega2**2 - omega3**2 - omega4**2) + KQV * vx + my
    tau_z = (
        KR1 * (-omega1 + omega2 - omega3 + omega4)
        + KR2 * (-d_omega1 + d_omega2 - d_omega3 + d_omega4)
        - KRR * r + mz
    )

    d_p = (q * r * (IYY - IZZ) + tau_x) / IXX
    d_q = (p * r * (IZZ - IXX) + tau_y) / IYY
    d_r = (p * q * (IXX - IYY) + tau_z) / IZZ

    return np.array(
        [
            d_dx, d_dy, d_dz,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            d_p, d_q, d_r,
            0.0, 0.0, 0.0,
            d_omega1, d_omega2, d_omega3, d_omega4,
        ],
        dtype=np.float64,
    )


def integrate_state(method: str,
                    state: np.ndarray,
                    action: np.ndarray,
                    dt: float,
                    prev_deriv: np.ndarray | None = None,
                    implicit_iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Advance the state by one time step.

    Every simulator entrypoint picks its integrator through the yaml, so the
    discretisation is implemented once here.

    Args:
        method: explicit/euler, rk4, adams-bashforth, adams-moulton or
            implicit.
        state: the current state.
        action: the commands held over the step.
        dt: the step in seconds.
        prev_deriv: the previous step's derivative, which the multi-step
            methods need; None makes them fall back to one Euler step.
        implicit_iters: fixed-point iterations of the implicit method.
    Returns:
        (next state, the derivative at the current state -- pass it back as
        prev_deriv on the next call).
    Raises:
        ValueError: unknown method.
    """
    method = method.lower()
    deriv = dynamics(state, action)

    if method in {"explicit", "euler"}:
        return state + dt * deriv, deriv

    if method == "rk4":
        k1 = deriv
        k2 = dynamics(state + 0.5 * dt * k1, action)
        k3 = dynamics(state + 0.5 * dt * k2, action)
        k4 = dynamics(state + dt * k3, action)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), deriv

    if method == "adams-bashforth":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        return state + dt * (1.5 * deriv - 0.5 * prev_deriv), deriv

    if method == "adams-moulton":
        if prev_deriv is None:
            return state + dt * deriv, deriv
        predictor = state + dt * (1.5 * deriv - 0.5 * prev_deriv)
        return state + 0.5 * dt * (dynamics(predictor, action) + deriv), deriv

    if method == "implicit":
        next_state = state.copy()
        for _ in range(max(1, int(implicit_iters))):
            next_state = state + dt * dynamics(next_state, action)
        return next_state, deriv

    raise ValueError(
        f"Unsupported integration method '{method}'. "
        "Choose from explicit, euler, rk4, adams-bashforth, adams-moulton, implicit."
    )


def build_input_vector(state: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """Project a full simulator state down to what the model reads.

    Args:
        state: the 19-entry state.
        input_labels: the model's input channels; `omega` expands to its four
            rotors, and the two error channels are computed here.
    Returns:
        The input vector, in the order of `input_labels`.
    Raises:
        ValueError: a label matches no state entry.
    """
    values: list[float] = []
    for label in input_labels:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved == "omega":
            values.extend(state[STATE_INDEX[f"omega{i}"]] for i in range(1, 5))
        elif resolved == "distance_error":
            values.append(float(np.linalg.norm(state[0:3])))
        elif resolved == "attitude_error":
            values.append(float(np.linalg.norm(state[6:9])))
        else:
            if resolved not in STATE_INDEX:
                raise ValueError(f"Unsupported simulation input label '{label}'.")
            values.append(float(state[STATE_INDEX[resolved]]))
    return np.asarray(values, dtype=np.float64)


def state_from_input_features(values: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """Rebuild a simulator state from dataset features -- the inverse of
    build_input_vector, used to start a rollout from a recorded frame.

    Args:
        values: the feature vector.
        input_labels: what those features are.
    Returns:
        The 19-entry body-frame state. Entries no feature covers stay at zero,
        which keeps this usable for ablated or reduced-feature models while
        still letting the simulator run.
    """
    state = np.zeros(len(STATE_LABELS), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    cursor = 0
    for label in input_labels:
        resolved = LABEL_ALIASES.get(label, label)
        if resolved in {"distance_error", "attitude_error"}:
            cursor += 1
            continue
        if resolved == "omega":
            state[STATE_INDEX["omega1"]:STATE_INDEX["omega4"] + 1] = values[cursor:cursor + 4]
            cursor += 4
            continue
        if resolved in STATE_INDEX:
            state[STATE_INDEX[resolved]] = values[cursor]
        cursor += 1
    return state


def normalize_input(values: np.ndarray, input_labels: list[str]) -> np.ndarray:
    """Normalise simulator inputs with the very bounds training used -- get
    this wrong and the closed loop feeds the network a distribution it never
    saw, which looks like a controller failure.

    Args:
        values: the raw input vector.
        input_labels: what those values are.
    Returns:
        The normalised vector.
    """
    global_min, global_max = get_norm_vectors(input_labels)
    global_min = global_min.reshape(-1)
    global_max = global_max.reshape(-1)
    return (values - global_min) / (global_max - global_min + 1e-10)


def build_normalized_window(state: np.ndarray, input_labels: list[str], seq_len: int) -> np.ndarray:
    """Seed the history window a sequence model needs at the first step.

    Args:
        state: the initial state.
        input_labels: the model's input channels.
        seq_len: window length.
    Returns:
        (seq_len, features), the initial observation repeated -- the rollout
        has no past yet.
    """
    obs = normalize_input(build_input_vector(state, input_labels), input_labels)
    return np.repeat(obs[np.newaxis, :], repeats=max(1, seq_len), axis=0)


def update_observation_window(window: np.ndarray, state: np.ndarray, input_labels: list[str], seq_len: int) -> np.ndarray:
    """Slide the history window forward by one simulated step.

    Args:
        window: the current window.
        state: the newly simulated state.
        input_labels: the model's input channels.
        seq_len: window length; 1 keeps only the new observation.
    Returns:
        The updated window, oldest frame dropped.
    """
    new_obs = normalize_input(build_input_vector(state, input_labels), input_labels)
    if seq_len <= 1:
        return new_obs[np.newaxis, :]
    return np.concatenate([window[-(seq_len - 1):], new_obs[np.newaxis, :]], axis=0)


def make_model_observation(window: np.ndarray, use_sequencing: bool, device: torch.device) -> torch.Tensor:
    """Convert a history window into the tensor layout the model expects.

    Args:
        window: the normalised history.
        use_sequencing: True gives the rank-4 layout with an explicit history
            axis (the conv path); False sends the latest frame only.
        device: where to put the tensor.
    Returns:
        The observation tensor, batch size 1.
    """
    if use_sequencing:
        obs = window[np.newaxis, np.newaxis, :, :]
    else:
        obs = window[-1:, :][np.newaxis, :, :]
    return torch.tensor(obs, dtype=torch.float32, device=device)


def build_lightning_model(config_model: dict, checkpoint_path: str | None, project_root, device: torch.device):
    """Rebuild a controller for simulation.

    Args:
        config_model: the run's archived config, which carries the topology.
        checkpoint_path: weights to load; None leaves the network at its
            initialisation, which is the way to get an untrained control arm.
        project_root: repo root, used to resolve a checkpoint given by stem.
        device: where to put the model.
    Returns:
        The model in eval mode.
    """
    input_dim, output_dim = dataset_dims(config_model)
    network = build_controller_network(config_model, input_dim, output_dim)
    model = Lightning_Model(network, config_model)
    if checkpoint_path is not None:
        ckpt = resolve_checkpoint(checkpoint_path, project_root)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)["state_dict"]
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rollout_controller(model,
                       initial_state: np.ndarray,
                       input_labels: list[str],
                       dt: float,
                       horizon_steps: int,
                       device: torch.device,
                       use_sequencing: bool,
                       seq_len: int,
                       initial_window: np.ndarray | None = None,
                       force_timespans: bool = False,
                       integration_method: str = "explicit",
                       implicit_iters: int = 5,
                       stop_fn: Callable[[np.ndarray, int], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fly the model in closed loop: its own commands drive the state it sees
    next. This is where a controller that scores well in open loop can still
    drift away -- the errors compound instead of being reset every frame.

    Shared by every simulator entrypoint, so recurrent-state handling and
    window updates stay identical everywhere.

    Args:
        model: the controller.
        initial_state: the 19-entry state to start from.
        input_labels: the model's input channels.
        dt: simulation step in seconds.
        horizon_steps: how many steps to fly.
        device: torch device.
        use_sequencing: observation layout (see make_model_observation).
        seq_len: history window length.
        initial_window: a window to start with, e.g. taken from a recorded
            trajectory; None builds one from the initial state.
        force_timespans: feed dt as timespans even if the model does not
            declare a time channel.
        integration_method, implicit_iters: passed to integrate_state.
        stop_fn: called with (state, step) after each step; returning True ends
            the rollout early -- what a touchdown or a crash test uses.
    Returns:
        (states, actions): the trajectory including the initial state, and the
        commands issued, clamped to [0, 1].
    """
    base_labels = input_labels_without_time(input_labels)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    window = np.array(initial_window, copy=True) if initial_window is not None else build_normalized_window(state, base_labels, seq_len)
    dt_tensor = None
    if force_timespans or getattr(model, "with_time", False):
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device).reshape(1, 1, 1)

    hx = None
    prev_deriv = None
    states = [state.copy()]
    actions = []

    for step_idx in range(int(horizon_steps)):
        obs_tensor = make_model_observation(window, use_sequencing=use_sequencing, device=device)
        with torch.no_grad():
            output = model(obs_tensor, hx=hx, timespans=dt_tensor)
            prediction, hx = output if isinstance(output, tuple) else (output, None)
        action = torch.clamp(prediction, min=0.0, max=1.0).squeeze(0).squeeze(0).cpu().numpy()
        state, prev_deriv = integrate_state(
            integration_method,
            state,
            action,
            dt,
            prev_deriv=prev_deriv,
            implicit_iters=implicit_iters,
        )
        states.append(state.copy())
        actions.append(action)
        if stop_fn is not None and stop_fn(state, step_idx + 1):
            break
        window = update_observation_window(window, state, base_labels, seq_len)

    return np.asarray(states), np.asarray(actions)


def generate_starting_conditions(n_conditions: int, seed: int = 42) -> np.ndarray:
    """Sample randomised but physically plausible start conditions, so a
    controller is scored over a spread of situations and not one lucky one.

    Args:
        n_conditions: how many to draw.
        seed: draw seed, so a benchmark is reproducible.
    Returns:
        (n_conditions, 19) states: offset position, small velocities, attitude
        within +-40deg in roll/pitch and any heading, small rates and
        disturbances, rotors at mid-throttle.
    """
    rng = np.random.default_rng(seed)
    starts = np.zeros((n_conditions, 19), dtype=np.float64)
    starts[:, 0:2] = rng.choice([-1, 1], size=(n_conditions, 2)) * rng.uniform(1.0, 5.0, size=(n_conditions, 2))
    starts[:, 2] = rng.uniform(-1.0, 1.0, size=n_conditions)
    starts[:, 3:6] = rng.uniform(-0.5, 0.5, size=(n_conditions, 3))
    starts[:, 6:8] = rng.uniform(-2.0 * np.pi / 9.0, 2.0 * np.pi / 9.0, size=(n_conditions, 2))
    starts[:, 8] = rng.uniform(-np.pi, np.pi, size=n_conditions)
    starts[:, 9:12] = rng.uniform(-1.0, 1.0, size=(n_conditions, 3))
    starts[:, 12:15] = rng.uniform(-0.01, 0.01, size=(n_conditions, 3))
    starts[:, 15:19] = (OMEGA_MAX + OMEGA_MIN) / 2.0
    return starts
