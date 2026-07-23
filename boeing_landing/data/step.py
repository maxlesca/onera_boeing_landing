# -*- coding: utf-8 -*-
"""What the upstream csv steps have in common.

A pipeline declares at most one step that produces the csv the build reads:
`prepare:` (rename a raw delivery) or `augment:` (derive the local-frame
coordinates). The two dispatchers -- data/prepare.py and data/augment.py --
differ only in how many inputs they read and which function of the pipeline
module they call; everything around that lives here, so a third kind of step
costs a call sequence and not a third copy of the same forty lines.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def step_config(config: dict, section: str, instead: str) -> dict:
    """The step block a pipeline declares.

    Args:
        config: the resolved pipeline config.
        section: 'prepare' or 'augment'.
        instead: what a pipeline without that block does, quoted in the error.
    Returns:
        The block.
    Raises:
        SystemExit: the config declares no such step.
    """
    cfg = config.get(section)
    if not cfg:
        raise SystemExit(f"this config has no `{section}:` section -- {instead}")
    return cfg


def resolve_path(cfg: dict, key: str, override: Path | None) -> Path:
    """One of a step's paths.

    Args:
        cfg: the step block.
        key: the key holding the default (e.g. 'raw_csv').
        override: a path given on the command line, which wins.
    Returns:
        The path to use, so the build step can run the whole chain from the
        config alone while a CLI call can still point elsewhere.
    """
    return override or Path(cfg[key])


def load_module(cfg: dict):
    """Import the module a step names.

    Args:
        cfg: the step block, read for its `module` key.
    Returns:
        The imported module; the dispatcher calls its documented entry point.
    """
    return importlib.import_module(cfg["module"])


def write_csv(df, out: Path, sources: list[Path]) -> Path:
    """Write a step's output, refusing to destroy what it read.

    Args:
        df: the frame to write.
        out: destination csv.
        sources: the step's inputs -- a delivery is read-only, and a step that
            overwrote one would leave nothing to rebuild from.
    Returns:
        `out`, its parent directory created.
    Raises:
        SystemExit: the output is one of the sources.
    """
    if out.resolve() in {source.resolve() for source in sources}:
        raise SystemExit("refusing to overwrite an input file")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep=";", index=False)
    return out


def report_warnings(warnings) -> None:
    """Print what a step reported as suspicious.

    Args:
        warnings: the messages the pipeline module returned. They are printed
            as they come: only that module knows what its own check means, so
            the sentence belongs there and not here.
    Returns:
        Nothing; a warning never stops the chain -- the point is to catch a
        truncated or mismatched delivery, not to block one.
    """
    for warning in warnings:
        print(f"  WARNING: {warning}")
