# -*- coding: utf-8 -*-
"""Pipeline config loading with `extends` inheritance.

A pipeline is a folder under boeing_landing/pipelines/: a base.yaml plus
variants that declare `extends: base.yaml` and override only the knobs they
change. Lives here on purpose: the quadrotor baseline does not use config
inheritance, so this is boeing-specific, not shared utils/ material.
"""

from __future__ import annotations

from pathlib import Path

from utils.config import load_yaml


def deep_merge(base: dict, override: dict) -> dict:
    """base updated by override: dicts merge key by key, anything else is
    replaced. Neither input is mutated."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    """load_yaml plus inheritance: a file may declare `extends: other.yaml`
    (relative to itself) -- variants share one base instead of duplicating it."""
    path = Path(path)
    config = load_yaml(path)
    parent = config.pop("extends", None)
    if parent is None:
        return config
    return deep_merge(load_config((path.parent / parent).resolve()), config)


def load_pipeline_config(config_path: Path) -> dict:
    """Resolved config (extends applied); a non-base variant tags its runs
    (runs/<pipeline>/<variant>_<order>/) through run_tag."""
    config = load_config(config_path)
    stem = Path(config_path).stem
    if stem != "base" and not config.get("run_tag"):
        config["run_tag"] = stem
    return config
