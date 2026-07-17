# -*- coding: utf-8 -*-
"""Plot training curves for one run, or compare several runs.

    python -m boeing_landing.report --runs runs/<name>/<timestamp> [more runs...]

One run  -> its train/val loss curves, plus the feature-group ablation bars
            when the run has an evaluation.json (make evaluate).
Several  -> their val_loss curves overlaid for comparison.
--save writes a PNG next to the (first) run instead of opening a window.
"""

from __future__ import annotations

import argparse
import json
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


def _ablation_deltas(run_dir: Path) -> tuple[float, dict[str, float]] | None:
    """(baseline mse, {group: delta-mse}) from evaluation.json; None if absent."""
    path = run_dir / "evaluation.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    if not results.get("ablation"):
        return None
    base = results["baseline"]["mse_mean"]
    return base, {name: m["mse_mean"] - base for name, m in results["ablation"].items()}


def plot_ablation(run_dir: Path, ax, noise: float = 0.0) -> None:
    """Sorted delta-MSE bars: how much the model loses when a group is masked."""
    base, deltas = _ablation_deltas(run_dir)
    names = sorted(deltas, key=deltas.get)
    ax.barh(names, [deltas[n] for n in names])
    if noise > 0:
        ax.axvline(noise, ls="--", color="grey", label=f"seed noise ~{noise:g}")
        ax.legend()
    ax.set_xlabel(f"delta MSE when masked (baseline {base:.4f})")
    ax.set_title("feature-group ablation")
    ax.grid(True, axis="x", alpha=0.3)


def _order_run_dirs() -> list[Path]:
    """Latest run dir of each conv-order sweep entry (for --orders)."""
    from boeing_landing.data.features import FEATURE_ORDERS
    from boeing_landing.train import DEFAULT_CONFIG, PROJECT_ROOT
    from utils.config import load_yaml

    base = load_yaml(DEFAULT_CONFIG).get("checkpoint_name") or "run"
    found = []
    for order in FEATURE_ORDERS:
        stamps = sorted((PROJECT_ROOT / "runs" / f"{base}_{order}").glob("*"))
        if stamps:
            found.append(stamps[-1])
    if not found:
        raise SystemExit("no order-sweep runs found (run `make experiment-order` first)")
    return found


def _best_val_loss(run_dir: Path) -> float:
    """best_val_loss recorded in the run's summary.json."""
    return json.loads((run_dir / "summary.json").read_text())["best_val_loss"]


def plot_best_bars(run_dirs: list[Path], ax, noise: float = 0.0) -> None:
    """Sweep comparison: best val_loss of each run as sorted bars.
    Bars left of the `best + noise` line are indistinguishable from the best."""
    scores = {d.parent.name: _best_val_loss(d) for d in run_dirs}
    names = sorted(scores, key=scores.get, reverse=True)
    ax.barh(names, [scores[n] for n in names])
    if noise > 0:
        ax.axvline(min(scores.values()) + noise, ls="--", color="grey",
                   label=f"best + seed noise {noise:g}")
        ax.legend()
    ax.set_xlabel("best val_loss")
    ax.set_title("sweep comparison")
    ax.grid(True, axis="x", alpha=0.3)


def _style_loss_ax(ax) -> None:
    ax.set(xlabel="step", ylabel="MSE loss", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", help="one or more run directories")
    ap.add_argument("--orders", action="store_true",
                    help="auto-discover the conv-order sweep runs (implies --bars)")
    ap.add_argument("--save", action="store_true", help="save PNG next to the first run instead of showing")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="seed-noise threshold line on the bar charts (0 = none)")
    ap.add_argument("--bars", action="store_true",
                    help="several runs: compare their BEST val_loss as bars (sweeps) instead of curves")
    a = ap.parse_args()

    if a.orders:
        a.runs, a.bars = _order_run_dirs(), True
    if not a.runs:
        ap.error("--runs or --orders is required")

    single = len(a.runs) == 1
    with_ablation = single and _ablation_deltas(a.runs[0]) is not None
    fig, axes = plt.subplots(1, 2 if with_ablation else 1,
                             figsize=(14, 5) if with_ablation else (9, 5))

    if single:
        plot_run(a.runs[0], axes[0] if with_ablation else axes)
        _style_loss_ax(axes[0] if with_ablation else axes)
        if with_ablation:
            plot_ablation(a.runs[0], axes[1], noise=a.noise)
    elif a.bars:
        plot_best_bars(a.runs, axes, noise=a.noise)
    else:
        plot_comparison(a.runs, axes)
        _style_loss_ax(axes)
    fig.tight_layout()

    if a.save:
        out = a.runs[0] / "report.png"
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
