#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared filesystem and YAML helpers for the training package.

These helpers keep path resolution logic out of the entry scripts so training,
testing, and simulators all resolve checkpoints and saved configs the same way.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def ensure_dir(path: Path) -> Path:
    """Create a directory tree if needed.

    Args:
        path: the directory, parents included.
    Returns:
        That same path, so the call can be chained into a longer path
        expression.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    """Read a YAML file.

    Args:
        path: the file, read as UTF-8.
    Returns:
        The parsed dictionary.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: dict) -> None:
    """Write a YAML file, creating its parent directories on demand.

    Args:
        path: destination file.
        payload: what to write; key order is preserved (sort_keys=False), so an
            archived config still reads like the one it came from.
    Returns:
        Nothing.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_checkpoint(model_name: str, project_root: Path) -> Path:
    """Resolve a checkpoint from an experiment stem or an explicit path.

    Args:
        model_name: an absolute or relative file path, or a stem with or
            without the `.ckpt` suffix.
        project_root: repo root, whose `checkpoints/` folder is searched when
            the name is not a path that exists.
    Returns:
        The checkpoint path.
    Raises:
        FileNotFoundError: no candidate matched.
    """
    candidate = Path(model_name)
    if candidate.is_file():
        return candidate

    variants = [candidate]
    if candidate.suffix != ".ckpt":
        variants.append(candidate.with_suffix(".ckpt"))

    checkpoint_dir = project_root / "checkpoints"
    for variant in variants:
        direct = checkpoint_dir / variant.name
        if direct.is_file():
            return direct

    raise FileNotFoundError(f"Unable to locate checkpoint '{model_name}'.")


def resolve_saved_config(model_name: str, config_dir: Path) -> Path:
    """Resolve a saved training config by experiment stem or explicit path.

    Args:
        model_name: a file path, or a stem with or without the `.yaml` suffix.
        config_dir: directory of archived configs, searched when the name is
            not a path that exists.
    Returns:
        The config path.
    Raises:
        FileNotFoundError: no candidate matched.
    """
    candidate = Path(model_name)
    if candidate.is_file():
        return candidate

    variants = [candidate]
    if candidate.suffix != ".yaml":
        variants.append(candidate.with_suffix(".yaml"))

    for variant in variants:
        direct = config_dir / variant.name
        if direct.is_file():
            return direct

    raise FileNotFoundError(f"Unable to locate saved config for '{model_name}'.")


def dataset_dims(config: dict) -> tuple[int, int]:
    """Read the I/O dimensions a training run recorded.

    Args:
        config: an archived config; both the old (input_size/output_size) and
            the new (input_dim/output_dim) key names are accepted, so a run
            trained before the rename still reloads.
    Returns:
        (input_dim, output_dim).
    Raises:
        KeyError: the config carries neither pair.
    """
    dataset_cfg = config.get("dataset", {})
    input_dim = dataset_cfg.get("input_dim", dataset_cfg.get("input_size"))
    output_dim = dataset_cfg.get("output_dim", dataset_cfg.get("output_size"))
    if input_dim is None or output_dim is None:
        raise KeyError("Model config is missing dataset input/output dimensions.")
    return int(input_dim), int(output_dim)
