# -*- coding: utf-8 -*-
"""Input/output feature definitions for the step-1 landing pipeline.

Inputs: inertial state + raw GPS (no ILS). Labels: the flight commands.

The 1D conv in ConvCfC slides a kernel over the *feature axis*, so the order of
the input channels is a real hyperparameter. Datasets are built in the
canonical order below; FEATURE_ORDERS holds the variants swept at load time by
experiments/feature_order.py -- no rebuild needed to try a new order.
"""

from __future__ import annotations

import numpy as np

GPS = ["latitude", "longitude", "altitude"]
ATTITUDE = ["pitch", "bank", "heading"]
ANGULAR_RATES = ["p", "q", "r"]
BODY_VELOCITY = ["u", "v", "w"]
NED_VELOCITY = ["northsouth_velocity", "eastwest_velocity", "vertical_velocity"]

# Canonical order used when building the npz. Everything else is a permutation.
CANONICAL_INPUTS = GPS + ATTITUDE + ANGULAR_RATES + BODY_VELOCITY + NED_VELOCITY

LABELS = ["longitudinal", "lateral", "directional", "stabilizer", "throttle_left"]

def check_order(order: list[str]) -> list[str]:
    """Validate that `order` is a permutation of the canonical inputs."""
    if sorted(order) != sorted(CANONICAL_INPUTS):
        missing = set(CANONICAL_INPUTS) - set(order)
        extra = set(order) - set(CANONICAL_INPUTS)
        raise ValueError(f"invalid feature order (missing={missing}, extra={extra})")
    return order


def random_order(seed: int) -> list[str]:
    """A reproducible random permutation of the canonical inputs."""
    rng = np.random.default_rng(seed)
    return [CANONICAL_INPUTS[i] for i in rng.permutation(len(CANONICAL_INPUTS))]


# component-major ("all x, then all y, then all z"): the i-th channel of every
# group side by side, for i = 0, 1, 2.
_GROUPS = [GPS, ATTITUDE, ANGULAR_RATES, BODY_VELOCITY, NED_VELOCITY]
BY_AXIS = [g[i] for i in range(3) for g in _GROUPS]

# Named channel orders for the conv-ordering study. Each is a permutation of
# CANONICAL_INPUTS (validated by check_order at load time).
FEATURE_ORDERS = {
    "grouped": CANONICAL_INPUTS,                                               # by physical group
    "gps_first": GPS + ATTITUDE + ANGULAR_RATES + BODY_VELOCITY + NED_VELOCITY,
    "gps_last": ATTITUDE + ANGULAR_RATES + BODY_VELOCITY + NED_VELOCITY + GPS,
    "pos_vel": GPS + BODY_VELOCITY + NED_VELOCITY + ATTITUDE + ANGULAR_RATES,   # positions, then velocities
    "by_axis": BY_AXIS,                                                        # component-major
    "reversed": list(reversed(CANONICAL_INPUTS)),
    "random_1": random_order(1),
    "random_2": random_order(2),
    "random_3": random_order(3),
}
