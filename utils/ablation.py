#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated input-feature ablation utilities for controller evaluation.

The testing script uses these helpers to build a repeatable ablation suite from
the YAML config. Each named ablation resolves to concrete feature indices inside
the expanded model input tensor and is then zero-filled before evaluation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .data import expand_feature_labels


DEFAULT_FEATURE_GROUPS = OrderedDict(
    [
        ("position", ["dx", "dy", "dz"]),
        ("velocity", ["vx", "vy", "vz"]),
        ("attitude", ["phi", "theta", "psi"]),
        ("angular_rate", ["p", "q", "r"]),
        ("disturbance", ["Mx_ext", "My_ext", "Mz_ext"]),
        ("rotor_speed", ["omega"]),
        ("time", ["t", "dt"]),
    ]
)

# Small alias map so config files can use short names without repeating the
# full dataset field names.
ALIASES = {
    "mx": "Mx_ext",
    "my": "My_ext",
    "mz": "Mz_ext",
    "omega1": "omega1",
    "omega2": "omega2",
    "omega3": "omega3",
    "omega4": "omega4",
}


def resolve_ablation_features(requested_items: list[str] | None,
                              feature_sets: dict | None = None) -> list[str]:
    """
    Resolve a mixed list of group names and explicit feature names.

    Example:
    - `["velocity"]` -> `["vx", "vy", "vz"]`
    - `["velocity", "phi"]` -> `["vx", "vy", "vz", "phi"]`
    """
    if not requested_items:
        return []

    feature_sets = feature_sets or DEFAULT_FEATURE_GROUPS
    resolved: list[str] = []
    for item in requested_items:
        key = str(item).strip()
        if not key:
            continue
        if key in feature_sets:
            resolved.extend(feature_sets[key])
        else:
            resolved.append(key)

    deduped: list[str] = []
    seen = set()
    for feature in resolved:
        normalized = str(feature).lower()
        if normalized not in seen:
            deduped.append(feature)
            seen.add(normalized)
    return deduped


def drop_features_from_labels(input_labels: list[str],
                              requested_features: list[str]) -> tuple[list[str], list[str]]:
    """
    Remove requested features from a config label list while preserving order.

    The returned tuple is:
    - filtered label list to store back into the config
    - list of actually removed expanded features for metadata/naming
    """
    if not requested_features:
        return list(input_labels), []

    requested_lower = {str(feature).lower() for feature in requested_features}
    filtered: list[str] = []
    removed_expanded: list[str] = []

    for label in input_labels:
        label_lower = label.lower()
        if label_lower == "omega":
            omega_labels = [f"omega{i}" for i in range(1, 5)]
            remove_omega = (
                "omega" in requested_lower
                or any(omega_label in requested_lower for omega_label in omega_labels)
            )
            if remove_omega:
                removed_expanded.extend(omega_labels)
            else:
                filtered.append(label)
            continue

        canonical = ALIASES.get(label_lower, label)
        if str(canonical).lower() in requested_lower:
            removed_expanded.append(label)
        else:
            filtered.append(label)

    return filtered, removed_expanded


def resolve_feature_indices(input_labels: list[str], features: list[str]) -> list[int]:
    expanded = expand_feature_labels(input_labels)
    normalized_lookup = {
        label.lower(): idx
        for idx, label in enumerate(expanded)
    }

    indices: list[int] = []
    for feature in features:
        feature_key = feature.lower()
        if feature_key == "omega":
            # `omega` is stored as four separate rotor-speed channels.
            indices.extend(idx for idx, label in enumerate(expanded) if label.startswith("omega"))
            continue

        resolved = ALIASES.get(feature_key, feature)
        idx = normalized_lookup.get(str(resolved).lower())
        if idx is not None:
            indices.append(idx)

    return sorted(set(indices))


def apply_feature_ablation(data: np.ndarray,
                           input_labels: list[str],
                           features: list[str],
                           fill_value: float = 0.0) -> np.ndarray:
    indices = resolve_feature_indices(input_labels, features)
    if not indices:
        return np.array(data, copy=True)

    # Work on a copy so repeated ablations always start from the same baseline.
    ablated = np.array(data, copy=True)
    ablated[:, indices, ...] = fill_value
    return ablated


def iter_ablation_specs(ablation_cfg: dict | None, input_labels: list[str]):
    if not ablation_cfg or not ablation_cfg.get("enabled", False):
        return []

    # Invalid or empty ablations are silently skipped so the caller can iterate
    # the resulting list directly without extra filtering.
    feature_sets = ablation_cfg.get("feature_sets") or DEFAULT_FEATURE_GROUPS
    specs = []
    for name, features in feature_sets.items():
        indices = resolve_feature_indices(input_labels, list(features))
        if indices:
            specs.append((name, list(features)))
    return specs
