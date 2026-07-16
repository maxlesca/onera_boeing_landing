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

from boeing_landing.train import (DEFAULT_CONFIG, PROJECT_ROOT, train_config,
                                  val_loss_from_checkpoint)
from utils.config import load_yaml


def _with_seed(config: dict, seed: int) -> dict:
    """Copy of the config with this seed, tagged so run dirs stay separate."""
    out = deepcopy(config)
    out["training"]["seed"] = seed
    out["checkpoint_name"] = f"{out.get('checkpoint_name') or 'run'}_seed{seed}"
    return out


def sweep(config_path: Path, project_root: Path) -> dict[int, tuple[float, Path]]:
    """Train once per seed; return {seed: (best val_loss, run dir)}."""
    base = load_yaml(config_path)
    results = {}
    for seed in base.get("experiments", {}).get("seeds", [42, 43, 44]):
        print(f"\n=== seed {seed} ===")
        ckpt = train_config(_with_seed(base, seed), project_root)
        results[int(seed)] = (val_loss_from_checkpoint(ckpt), ckpt.parent)
    return results


def report(results: dict[int, tuple[float, Path]]) -> None:
    losses = [loss for loss, _ in results.values()]
    mean, spread = statistics.mean(losses), statistics.stdev(losses) if len(losses) > 1 else 0.0
    print("\n=== convergence across seeds ===")
    for seed, (loss, run_dir) in results.items():
        print(f"  seed {seed}: val_loss={loss:.6f}  ({run_dir})")
    print(f"  mean={mean:.6f}  std={spread:.6f}  rel. spread={100 * spread / mean:.1f}%")
    dirs = " ".join(str(run_dir) for _, run_dir in results.values())
    print(f'\ncompare curves:  make plots RUNS="{dirs}"')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    report(sweep(a.config, PROJECT_ROOT))


if __name__ == "__main__":
    main()
