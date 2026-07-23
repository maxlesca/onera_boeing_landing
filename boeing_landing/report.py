# -*- coding: utf-8 -*-
"""Plot training curves for one run, or compare several runs.

    python -m boeing_landing.report --runs runs/<pipeline>/<variant>/<timestamp> [more runs...]

One run  -> its train/val loss curves, plus the feature-group ablation bars
            when the run has an evaluation.json (make evaluate).
Several  -> their val_loss curves overlaid for comparison.
--save writes a PNG into figures/ (one flat folder, named after the runs)
instead of opening a window.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _metrics_frame(run_dir: Path) -> pd.DataFrame:
    """Read a run's training curves.

    Args:
        run_dir: the run directory.
    Returns:
        Lightning's metrics.csv (latest version subfolder) as a DataFrame.
    Raises:
        SystemExit: the run has none, i.e. it trained with logging off.
    """
    candidates = sorted(run_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        raise SystemExit(f"no metrics.csv under {run_dir}")
    return pd.read_csv(candidates[-1])


def _run_label(run_dir: Path) -> str:
    """Legend label of one run.

    Args:
        run_dir: the run directory.
    Returns:
        '<pipeline>/<variant>/<timestamp>'.
    """
    return "/".join(run_dir.parts[-3:])


def _variant_label(run_dir: Path) -> str:
    """Legend label of one variant.

    Args:
        run_dir: the run directory.
    Returns:
        '<pipeline>/<variant>' -- identifies a run without its timestamp, which
        is what a sweep chart needs.
    """
    return f"{run_dir.parent.parent.name}/{run_dir.parent.name}"


def _mean_predictor_mse(run_dir: Path) -> tuple[list[str], np.ndarray] | None:
    """The naive baseline every curve is read against: always output the
    train-mean command. Roughly the val-label variance.

    Args:
        run_dir: the run directory; the dataset comes from its saved config, so
            the baseline is computed on the very data the run was trained on.
    Returns:
        (command names, per-command MSE), or None when the npz are missing --
        both normalised with the train bounds, as at training time.
    """
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
    """The same baseline as a single number.

    Args:
        run_dir: the run directory.
    Returns:
        Its mean over the commands, comparable to val_loss -- a model is only
        learning when it sits clearly below this -- or None if it cannot be
        computed.
    """
    per_command = _mean_predictor_mse(run_dir)
    return float(per_command[1].mean()) if per_command else None


def _draw_mean_baseline(run_dir: Path, ax, axis: str = "y") -> bool:
    """Draw the predict-the-mean reference line.

    Args:
        run_dir: the run whose dataset defines it.
        ax: axes to draw on.
        axis: 'y' for a horizontal line (loss curves), anything else for a
            vertical one (bar charts).
    Returns:
        True when the line was drawn, False when the baseline is unavailable --
        the caller uses it to decide whether a legend is worth showing.
    """
    base = _mean_predictor_loss(run_dir)
    if base is None:
        return False
    line = ax.axhline if axis == "y" else ax.axvline
    line(base, ls=":", color="tab:red", label=f"predict-the-mean {base:.4f}")
    return True


def plot_run(run_dir: Path, ax) -> None:
    """Draw one run's train and val loss curves, with the baseline line.

    Args:
        run_dir: the run directory.
        ax: axes to draw on.
    Returns:
        Nothing.
    """
    df = _metrics_frame(run_dir)
    for col, style in [("train_loss", "-"), ("val_loss", "o-")]:
        if col in df:
            points = df.dropna(subset=[col])
            ax.plot(points["step"], points[col], style, label=col)
    _draw_mean_baseline(run_dir, ax)
    ax.set_title(_run_label(run_dir))


def plot_comparison(run_dirs: list[Path], ax) -> None:
    """Overlay the val_loss curves of several runs.

    Args:
        run_dirs: the runs to compare; the first one's dataset provides the
            baseline line.
        ax: axes to draw on.
    Returns:
        Nothing.
    """
    for run_dir in run_dirs:
        df = _metrics_frame(run_dir)
        points = df.dropna(subset=["val_loss"])
        ax.plot(points["step"], points["val_loss"], "o-", label=_run_label(run_dir))
    _draw_mean_baseline(run_dirs[0], ax)
    ax.set_title("val_loss comparison")


def _model_per_command_mse(run_dir: Path) -> dict[str, float] | None:
    """The model's own per-command scores.

    Args:
        run_dir: the run directory.
    Returns:
        {command: mse} from its evaluation.json, or None when the run has not
        been evaluated (make evaluate writes that file).
    """
    path = run_dir / "evaluation.json"
    if not path.exists():
        return None
    per = json.loads(path.read_text()).get("regression", {}).get("per_channel")
    return {name: m["mse"] for name, m in per.items()} if per else None


def plot_per_command(run_dir: Path, ax) -> None:
    """Draw the model's per-command MSE next to the predict-the-mean baseline
    (~ the val-label variance): a command is only learned when its model bar
    sits below the baseline bar, a difference the global val_loss hides.

    Args:
        run_dir: the run directory; it must have been evaluated.
        ax: axes to draw on.
    Returns:
        Nothing.
    """
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
    """What each feature group is worth to the model.

    Args:
        run_dir: the run directory.
    Returns:
        (baseline mse, {group: mse increase when masked}) from its
        evaluation.json, or None when the run has no ablation results.
    """
    path = run_dir / "evaluation.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    if not results.get("ablation"):
        return None
    base = results["baseline"]["mse_mean"]
    return base, {name: m["mse_mean"] - base for name, m in results["ablation"].items()}


def plot_ablation(run_dir: Path, ax, noise: float = 0.0) -> None:
    """Draw the sorted delta-MSE bars: how much the model loses when a group is
    masked.

    Args:
        run_dir: the run directory; it must have been evaluated with ablation.
        ax: axes to draw on.
        noise: seed-noise threshold; a bar below it is indistinguishable from
            no effect at all. 0 hides the line.
    Returns:
        Nothing.
    """
    base, deltas = _ablation_deltas(run_dir)
    names = sorted(deltas, key=deltas.get)
    ax.barh(names, [deltas[n] for n in names])
    if noise > 0:
        ax.axvline(noise, ls="--", color="grey", label=f"seed noise ~{noise:g}")
        ax.legend()
    ax.set_xlabel(f"delta MSE when masked (baseline {base:.4f})")
    ax.set_title("feature-group ablation")
    ax.grid(True, axis="x", alpha=0.3)


def _latest_order_run(pipeline: str, order: str, stamp: str | None) -> Path | None:
    """The run to show for one channel order.

    Args:
        pipeline: the pipeline folder name under runs/.
        order: the channel order, which names the variant folder.
        stamp: timestamp prefix selecting a sweep session; None takes the
            latest run whatever its session.
    Returns:
        That run dir, or None when the order was never trained.
    """
    from boeing_landing.train import PROJECT_ROOT
    stamps = sorted((PROJECT_ROOT / "runs" / pipeline / order).glob(f"{stamp or ''}*"))
    return stamps[-1] if stamps else None


def _order_run_dirs(config_path: Path, stamp: str | None = None) -> list[Path]:
    """Find a whole conv-order sweep on disk, so the chart needs no run list.

    Args:
        config_path: the pipeline config whose sweep to collect.
        stamp: timestamp prefix selecting a sweep session.
    Returns:
        One run dir per order that has been trained.
    Raises:
        SystemExit: none has.
    """
    from boeing_landing.config import load_config
    from boeing_landing.data.features import FEATURE_ORDERS

    base = load_config(config_path).get("checkpoint_name") or "run"
    found = [d for d in (_latest_order_run(base, order, stamp) for order in FEATURE_ORDERS)
             if d is not None]
    if not found:
        raise SystemExit(f"no order-sweep runs for '{base}' (run `make experiment-order` "
                         "first, or loosen STAMP/CONFIG)")
    return found


def _best_val_loss(run_dir: Path) -> float:
    """A run's score, read from the summary it wrote.

    Args:
        run_dir: the run directory.
    Returns:
        Its best_val_loss.
    """
    return json.loads((run_dir / "summary.json").read_text())["best_val_loss"]


def plot_best_bars(run_dirs: list[Path], ax, noise: float = 0.0) -> None:
    """Draw a sweep comparison: each run's best val_loss as sorted bars.

    Args:
        run_dirs: the runs compared, each labelled by its variant.
        ax: axes to draw on.
        noise: seed-noise threshold; bars left of the `best + noise` line are
            indistinguishable from the best. 0 hides the line.
    Returns:
        Nothing.
    """
    scores = {_variant_label(d): _best_val_loss(d) for d in run_dirs}
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


def _figure_path(run_dirs: list[Path], bars: bool) -> Path:
    """Where a saved figure goes: figures/<pipeline>/, mirroring the runs
    organisation and never inside a run dir. Plots mixing several pipelines
    land in figures/comparisons/.

    Args:
        run_dirs: the runs the figure shows.
        bars: True for a sweep chart, which names the file differently from a
            curve comparison.
    Returns:
        The PNG path, its directory created.
    """
    from boeing_landing.train import PROJECT_ROOT
    from utils.config import ensure_dir

    pipelines = sorted({d.parent.parent.name for d in run_dirs})
    folder = pipelines[0] if len(pipelines) == 1 else "comparisons"
    if len(run_dirs) == 1:
        name = "_".join(run_dirs[0].parts[-2:])  # variant_timestamp
    else:
        kind = "bars" if bars else "comparison"
        name = f"{'_'.join(pipelines)}_{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    return ensure_dir(PROJECT_ROOT / "figures" / folder) / f"{name}.png"


def _style_loss_ax(ax) -> None:
    """Apply the shared look of the loss panels.

    Args:
        ax: axes holding loss curves.
    Returns:
        Nothing; log scale, since what matters late in training is the ratio,
        not the absolute gap.
    """
    ax.set(xlabel="step", ylabel="MSE loss", yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def _report_single(run_dir: Path, noise: float):
    """Assemble one run's figure.

    Args:
        run_dir: the run directory.
        noise: seed-noise threshold for the ablation panel.
    Returns:
        The figure: loss curves, plus the per-command MSE and ablation panels
        when their data exists (make evaluate writes evaluation.json).
    """
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
    """CLI entrypoint: plot one run, or compare several.

    Returns:
        Nothing; shows the figure, or writes it under figures/ with --save.
    """
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
        out = _figure_path(a.runs, a.bars)
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
