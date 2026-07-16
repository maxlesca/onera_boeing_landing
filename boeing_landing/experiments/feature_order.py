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


def sweep(config_path: Path, project_root: Path) -> dict[str, float]:
    scores = {}
    for order in FEATURE_ORDERS:
        print(f"\n=== order: {order} ===")
        scores[order] = val_loss_from_checkpoint(train(config_path, project_root, input_order=order))
    return scores


def report(scores: dict[str, float]) -> None:
    print("\n=== conv channel order vs val_loss (best first) ===")
    for order, loss in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {order:12s} {loss:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    report(sweep(a.config, PROJECT_ROOT))


if __name__ == "__main__":
    main()
