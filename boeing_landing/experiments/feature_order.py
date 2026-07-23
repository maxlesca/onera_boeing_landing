# -*- coding: utf-8 -*-
"""Sweep the conv channel orders and compare validation loss.

The 1D conv mixes adjacent input channels, so their order matters. This trains
one run per named order in features.FEATURE_ORDERS and prints a ranking.

    python -m boeing_landing.experiments.feature_order [--config path.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from boeing_landing.data.features import FEATURE_ORDERS
from boeing_landing.train import DEFAULT_CONFIG, PROJECT_ROOT, train, val_loss_from_checkpoint


def _train_order(config_path: Path, project_root: Path, order: str) -> tuple[float, Path]:
    """Train one arm of the sweep.

    Args:
        config_path: the pipeline config, used unchanged apart from the order.
        project_root: repo root holding runs/.
        order: the channel order this arm tests.
    Returns:
        (best val_loss, run dir).
    """
    print(f"\n=== order: {order} ===")
    ckpt = train(config_path, project_root, input_order=order)
    return val_loss_from_checkpoint(ckpt), ckpt.parent


def sweep(config_path: Path, project_root: Path) -> dict[str, tuple[float, Path]]:
    """Train one run per named channel order.

    Args:
        config_path: the pipeline config to sweep.
        project_root: repo root holding runs/.
    Returns:
        {order: (best val_loss, run dir)} -- every order in FEATURE_ORDERS,
        each in its own run dir.
    """
    return {order: _train_order(config_path, project_root, order)
            for order in FEATURE_ORDERS}


def report(scores: dict[str, tuple[float, Path]]) -> None:
    """Print the ranking.

    Args:
        scores: what sweep returned.
    Returns:
        Nothing.
    """
    print("\n=== conv channel order vs val_loss (best first) ===")
    for order, (loss, _) in sorted(scores.items(), key=lambda kv: kv[1][0]):
        print(f"  {order:12s} {loss:.6f}")
    print("\nvisualize:  make plots-orders")


def main() -> None:
    """CLI entrypoint: sweep the orders of the --config pipeline.

    Returns:
        Nothing; trains every arm, then prints the ranking.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    report(sweep(a.config, PROJECT_ROOT))


if __name__ == "__main__":
    main()
