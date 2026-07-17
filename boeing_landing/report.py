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
import numpy as np
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


def _mean_predictor_mse(run_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """Per-command MSE of the naive baseline that always outputs the train-mean
    command, computed from the run's own dataset (paths in its saved config).
    Roughly the val-label variance. None if the npz files are missing."""
    from boeing_landing.train import PROJECT_ROOT
    from utils.config import load_yaml

    cfg = load_yaml(run_dir / "config.yaml").get("dataset", {})
    paths = [PROJECT_ROOT / str(cfg.get(k, "")) for k in ("train_npz", "val_npz")]
    if not all(p.is_file() for p in paths):
        return None
    train, val = (np.load(p, allow_pickle=True) for p in paths)
    lo, hi = train["y_min"], train["y_max"]  # bounds come from the train split
    normalize = lambda y: (y - lo) / (hi - lo + 1e-10)
    mse = ((normalize(val["Y"]) - normalize(train["Y"]).mean(axis=0)) ** 2).mean(axis=0)
    return list(train["label_names"]), mse


def _mean_predictor_loss(run_dir: Path) -> float | None:
    """The same baseline as a single val_loss number (mean over the commands).
    A model is only learning when it sits clearly below this."""
    per_command = _mean_predictor_mse(run_dir)
    return float(per_command[1].mean()) if per_command else None


def _draw_mean_baseline(run_dir: Path, ax, axis: str = "y") -> bool:
    """Dotted line at the predict-the-mean val_loss; False if it can't be computed."""
    base = _mean_predictor_loss(run_dir)
    if base is None:
        return False
    line = ax.axhline if axis == "y" else ax.axvline
    line(base, ls=":", color="tab:red", label=f"predict-the-mean {base:.4f}")
    return True


def plot_run(run_dir: Path, ax) -> None:
    """Train and val loss curves of a single run."""
    df = _metrics_frame(run_dir)
    for col, style in [("train_loss", "-"), ("val_loss", "o-")]:
        if col in df:
            points = df.dropna(subset=[col])
            ax.plot(points["step"], points[col], style, label=col)
    _draw_mean_baseline(run_dir, ax)
    ax.set_title(_run_label(run_dir))


def plot_comparison(run_dirs: list[Path], ax) -> None:
    """val_loss of several runs overlaid."""
    for run_dir in run_dirs:
        df = _metrics_frame(run_dir)
        points = df.dropna(subset=["val_loss"])
        ax.plot(points["step"], points["val_loss"], "o-", label=_run_label(run_dir))
    _draw_mean_baseline(run_dirs[0], ax)
    ax.set_title("val_loss comparison")


def _model_per_command_mse(run_dir: Path) -> dict[str, float] | None:
    """{command: model mse} from the run's evaluation.json; None if not evaluated."""
    path = run_dir / "evaluation.json"
    if not path.exists():
        return None
    per = json.loads(path.read_text()).get("regression", {}).get("per_channel")
    return {name: m["mse"] for name, m in per.items()} if per else None


def plot_per_command(run_dir: Path, ax) -> None:
    """Per-command MSE of the model next to the predict-the-mean baseline
    (~ val-label variance): a command is only learned when its model bar sits
    below the baseline bar. The global val_loss hides these differences."""
    labels, base = _mean_predictor_mse(run_dir)
    model = _model_per_command_mse(run_dir)
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, [model[l] for l in labels], 0.4, label="model")
    ax.bar(pos + 0.2, base, 0.4, label="predict-the-mean (~variance)", color="tab:red", alpha=0.6)
    ax.set_xticks(pos, labels, rotation=20, ha="right")
    ax.set_ylabel("MSE on val")
    ax.set_title("per-command MSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()


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


def _order_run_dirs(config_path: Path, stamp: str | None = None) -> list[Path]:
    """One run dir per conv-order sweep entry of the given pipeline config:
    the latest, or the latest whose timestamp starts with `stamp`."""
    from boeing_landing.data.features import FEATURE_ORDERS
    from boeing_landing.train import PROJECT_ROOT
    from utils.config import load_yaml

    base = load_yaml(config_path).get("checkpoint_name") or "run"
    found = []
    for order in FEATURE_ORDERS:
        stamps = sorted((PROJECT_ROOT / "runs" / f"{base}_{order}").glob(f"{stamp or ''}*"))
        if stamps:
            found.append(stamps[-1])
    if not found:
        raise SystemExit(f"no order-sweep runs for '{base}' (run `make experiment-order` "
                         "first, or loosen STAMP/CONFIG)")
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
    if _draw_mean_baseline(run_dirs[0], ax, axis="x") or noise > 0:
        ax.legend()
    ax.set_xlabel("best val_loss")
    ax.set_title("sweep comparison")
    ax.grid(True, axis="x", alpha=0.3)


def _style_loss_ax(ax) -> None:
    ax.set(xlabel="step", ylabel="MSE loss", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def _report_single(run_dir: Path, noise: float):
    """One run's figure: loss curves, plus the per-command MSE and ablation
    panels when their data exists (make evaluate writes evaluation.json)."""
    with_commands = (_model_per_command_mse(run_dir) is not None
                     and _mean_predictor_mse(run_dir) is not None)
    with_ablation = _ablation_deltas(run_dir) is not None
    n = 1 + with_commands + with_ablation
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    axes = np.atleast_1d(axes)
    plot_run(run_dir, axes[0])
    _style_loss_ax(axes[0])
    if with_commands:
        plot_per_command(run_dir, axes[1])
    if with_ablation:
        plot_ablation(run_dir, axes[-1], noise=noise)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", help="one or more run directories")
    ap.add_argument("--orders", action="store_true",
                    help="auto-discover the conv-order sweep runs (implies --bars)")
    ap.add_argument("--config", type=Path, default=None,
                    help="with --orders: pipeline config whose sweep to show (default: gps_cfc)")
    ap.add_argument("--stamp", default=None,
                    help="with --orders: timestamp prefix selecting a sweep session")
    ap.add_argument("--save", action="store_true", help="save PNG next to the first run instead of showing")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="seed-noise threshold line on the bar charts (0 = none)")
    ap.add_argument("--bars", action="store_true",
                    help="several runs: compare their BEST val_loss as bars (sweeps) instead of curves")
    a = ap.parse_args()

    if a.orders:
        from boeing_landing.train import DEFAULT_CONFIG
        a.runs, a.bars = _order_run_dirs(a.config or DEFAULT_CONFIG, a.stamp), True
    if not a.runs:
        ap.error("--runs or --orders is required")

    if len(a.runs) == 1:
        fig = _report_single(a.runs[0], a.noise)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        if a.bars:
            plot_best_bars(a.runs, ax, noise=a.noise)
        else:
            plot_comparison(a.runs, ax)
            _style_loss_ax(ax)
    fig.tight_layout()

    if a.save:
        out = a.runs[0] / "report.png"
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
