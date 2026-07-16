# -*- coding: utf-8 -*-
"""Plot training curves for one run, or compare several runs.

    python -m boeing_landing.report --runs runs/<name>/<timestamp> [more runs...]

One run  -> its train/val loss curves.
Several  -> their val_loss curves overlaid for comparison.
--save writes a PNG next to the (first) run instead of opening a window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _metrics_frame(run_dir: Path) -> pd.DataFrame:
    """Lightning's metrics.csv of a run (latest version) as a DataFrame."""
    candidates = sorted(run_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        raise SystemExit(f"no metrics.csv under {run_dir}")
    return pd.read_csv(candidates[-1])


def _run_label(run_dir: Path) -> str:
    """Readable legend label: <pipeline_order>/<timestamp>."""
    return f"{run_dir.parent.name}/{run_dir.name}"


def plot_run(run_dir: Path, ax) -> None:
    """Train and val loss curves of a single run."""
    df = _metrics_frame(run_dir)
    for col, style in [("train_loss", "-"), ("val_loss", "o-")]:
        if col in df:
            points = df.dropna(subset=[col])
            ax.plot(points["step"], points[col], style, label=col)
    ax.set_title(_run_label(run_dir))


def plot_comparison(run_dirs: list[Path], ax) -> None:
    """val_loss of several runs overlaid."""
    for run_dir in run_dirs:
        df = _metrics_frame(run_dir)
        points = df.dropna(subset=["val_loss"])
        ax.plot(points["step"], points["val_loss"], "o-", label=_run_label(run_dir))
    ax.set_title("val_loss comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="one or more run directories")
    ap.add_argument("--save", action="store_true", help="save PNG next to the first run instead of showing")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_run(a.runs[0], ax) if len(a.runs) == 1 else plot_comparison(a.runs, ax)
    ax.set(xlabel="step", ylabel="MSE loss", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if a.save:
        out = a.runs[0] / "learning_curves.png"
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
