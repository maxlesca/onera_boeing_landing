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
    """Merge two config trees.

    Args:
        base: the inherited config.
        override: the variant's own keys.
    Returns:
        A new dict where nested dicts merge key by key and anything else --
        including lists -- is replaced wholesale. Neither input is mutated, so
        one base can serve any number of variants.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    """Read a yaml and apply `extends` inheritance, so variants stay diffs
    against their base instead of copies of it.

    Args:
        path: the yaml to read; it may declare `extends: other.yaml`, resolved
            relative to itself, and the chain may be any depth.
    Returns:
        The resolved config, without the `extends` key.
    """
    path = Path(path)
    config = load_yaml(path)
    parent = config.pop("extends", None)
    if parent is None:
        return config
    return deep_merge(load_config((path.parent / parent).resolve()), config)


def load_pipeline_config(config_path: Path) -> dict:
    """load_config plus the run-dir tagging every entrypoint expects.

    Args:
        config_path: the pipeline yaml.
    Returns:
        The resolved config, with `run_tag` defaulted to the file stem for any
        variant other than base.yaml -- that is what sends a variant's runs to
        runs/<pipeline>/<variant>_<order>/ instead of mixing them into the
        base's folder, even when both share one npz.
    """
    config = load_config(config_path)
    stem = Path(config_path).stem
    if stem != "base" and not config.get("run_tag"):
        config["run_tag"] = stem
    return config
