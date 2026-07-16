# -*- coding: utf-8 -*-
"""Sweep the conv channel orders and compare validation loss.

The 1D conv mixes adjacent input channels, so their order matters. This trains
one run per named order in features.FEATURE_ORDERS and prints a ranking.

    python -m boeing_landing.experiments.feature_order
"""

from __future__ import annotations

import re
from pathlib import Path

from boeing_landing.data.features import FEATURE_ORDERS
from boeing_landing.train import train


def _val_loss(ckpt: Path) -> float:
    """Read the checkpoint's val_loss back from its filename."""
    m = re.search(r"val_loss=([0-9.]+)", ckpt.stem)
    return float(m.group(1)) if m else float("nan")


def sweep(config_path: Path, project_root: Path) -> dict[str, float]:
    scores = {}
    for order in FEATURE_ORDERS:
        print(f"\n=== order: {order} ===")
        scores[order] = _val_loss(train(config_path, project_root, input_order=order))
    return scores


def report(scores: dict[str, float]) -> None:
    print("\n=== conv channel order vs val_loss (best first) ===")
    for order, loss in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {order:12s} {loss:.6f}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]  # onera_boeing_landing/
    report(sweep(root / "boeing_landing/configs/step1_cfc.yaml", root))


if __name__ == "__main__":
    main()
