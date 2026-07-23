# -*- coding: utf-8 -*-
"""Convergence study: train the same config under several seeds.

A different seed changes the weight init and the batch shuffling. If the runs
end at similar val_loss, training converges reliably; a wide spread means any
single result (and any comparison built on one) is partly luck.

Seeds come from the config's experiments.seeds.

    python -m boeing_landing.experiments.convergence [--config path.yaml]
"""

from __future__ import annotations

import argparse
import statistics
from copy import deepcopy
from pathlib import Path

from boeing_landing.config import load_pipeline_config
from boeing_landing.train import (DEFAULT_CONFIG, PROJECT_ROOT, train_config,
                                  val_loss_from_checkpoint)


def _with_seed(config: dict, seed: int) -> dict:
    """Derive one arm of the sweep from the base config.

    Args:
        config: the base config, left untouched.
        seed: the seed this arm runs with.
    Returns:
        A deep copy carrying it, tagged so the run dirs stay separate
        (runs/<pipeline>/[<variant>_]seed<seed>_<order>/).
    """
    out = deepcopy(config)
    out["training"]["seed"] = seed
    out["run_tag"] = "_".join(filter(None, [out.get("run_tag"), f"seed{seed}"]))
    return out


def _train_seed(config: dict, project_root: Path, seed: int) -> tuple[float, Path]:
    """Train one arm.

    Args:
        config: the base config.
        project_root: repo root holding runs/.
        seed: the seed this arm runs with.
    Returns:
        (best val_loss, run dir).
    """
    print(f"\n=== seed {seed} ===")
    ckpt = train_config(_with_seed(config, seed), project_root)
    return val_loss_from_checkpoint(ckpt), ckpt.parent


def sweep(config_path: Path, project_root: Path) -> dict[int, tuple[float, Path]]:
    """Train the same config once per seed.

    Args:
        config_path: the pipeline config; its experiments.seeds drives the
            sweep (42, 43, 44 when it declares none).
        project_root: repo root holding runs/.
    Returns:
        {seed: (best val_loss, run dir)}.
    """
    base = load_pipeline_config(config_path)
    return {int(seed): _train_seed(base, project_root, seed)
            for seed in base.get("experiments", {}).get("seeds", [42, 43, 44])}


def report(results: dict[int, tuple[float, Path]]) -> None:
    """Print the per-seed scores and their spread -- the number that says how
    much of a comparison is real and how much is the init lottery.

    Args:
        results: what sweep returned.
    Returns:
        Nothing.
    """
    losses = [loss for loss, _ in results.values()]
    mean, spread = statistics.mean(losses), statistics.stdev(losses) if len(losses) > 1 else 0.0
    print("\n=== convergence across seeds ===")
    for seed, (loss, run_dir) in results.items():
        print(f"  seed {seed}: val_loss={loss:.6f}  ({run_dir})")
    print(f"  mean={mean:.6f}  std={spread:.6f}  rel. spread={100 * spread / mean:.1f}%")
    dirs = " ".join(str(run_dir) for _, run_dir in results.values())
    print(f'\ncompare curves:  make plots RUNS="{dirs}"')


def main() -> None:
    """CLI entrypoint: run the seed sweep of the --config pipeline.

    Returns:
        Nothing; trains every seed, then prints the spread.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    report(sweep(a.config, PROJECT_ROOT))


if __name__ == "__main__":
    main()
