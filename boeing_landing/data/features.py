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
# True simulator wind, standing in for the FMS wind estimate. The expert
# compensates a per-run different wind that is invisible in the other inputs
# (u,v,w are ground-relative) -- the aircraft analogue of the quadrotor
# baseline's M_ext disturbance inputs.
WIND = ["wind_velocity_x", "wind_velocity_y", "wind_velocity_z"]

# Canonical order used when building the npz. Everything else is a permutation.
# An optional per-frame dt channel (CfC timespans) is appended at load time by
# the loader (dataset.use_dt in the config); it is not a CSV column.
CANONICAL_INPUTS = GPS + ATTITUDE + ANGULAR_RATES + BODY_VELOCITY + NED_VELOCITY + WIND

# throttle_right mirrors throttle_left exactly in the source data (checked:
# max |left-right| = 0.0); kept as a label anyway so the controller outputs
# the full command set -- it doubles the throttle weight in the MSE.
LABELS = ["longitudinal", "lateral", "directional", "stabilizer",
          "throttle_left", "throttle_right"]

def extend_order(order: list[str], available: list[str]) -> list[str]:
    """`order` completed with the channels the dataset holds beyond it (e.g.
    extra_columns), appended in dataset order. The named orders therefore
    stay dataset-agnostic. Names unknown to the dataset are an error (typo guard)."""
    unknown = set(order) - set(available)
    if unknown:
        raise ValueError(f"channels not in the dataset: {sorted(unknown)}")
    return list(order) + [name for name in available if name not in order]


def random_order(seed: int) -> list[str]:
    """A reproducible random permutation of the canonical inputs."""
    rng = np.random.default_rng(seed)
    return [CANONICAL_INPUTS[i] for i in rng.permutation(len(CANONICAL_INPUTS))]


# component-major ("all x, then all y, then all z"): the i-th channel of every
# group side by side, for i = 0, 1, 2.
_GROUPS = [GPS, ATTITUDE, ANGULAR_RATES, BODY_VELOCITY, NED_VELOCITY, WIND]
BY_AXIS = [g[i] for i in range(3) for g in _GROUPS]

# Named channel orders for the conv-ordering study. Each is a permutation of
# CANONICAL_INPUTS; channels a dataset holds beyond them (extra_columns)
# are appended at the end by extend_order at load time.
FEATURE_ORDERS = {
    "grouped": CANONICAL_INPUTS,                                               # by physical group (GPS first)
    "gps_last": ATTITUDE + ANGULAR_RATES + BODY_VELOCITY + NED_VELOCITY + WIND + GPS,
    "pos_vel": GPS + BODY_VELOCITY + NED_VELOCITY + WIND + ATTITUDE + ANGULAR_RATES,  # positions, then velocities
    "by_axis": BY_AXIS,                                                        # component-major
    "reversed": list(reversed(CANONICAL_INPUTS)),
    "random_1": random_order(1),
    "random_2": random_order(2),
    "random_3": random_order(3),
}
