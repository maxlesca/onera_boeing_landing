# -*- coding: utf-8 -*-
"""Per-run data distribution: which runs are outliers, and on which channels.

Pools the train+val npz (all runs, whatever the split), normalises with the
dataset's own bounds, then per run:
- a runs x channels heatmap of the per-run **mean** -- a run's fingerprint; an
  outlier stands out as an off-colour row;
- an **extremeness** score = mean over channels of |run mean - other runs' mean|,
  ranking how far each run sits from the rest (+ the channel that drives it);
- a **wind scatter**: wind is ~constant per run and invisible to the other inputs
  (u,v,w are ground-relative), so a wind outlier is genuinely out-of-distribution
  and a good candidate for a "far" validation run.

    python -m boeing_landing.data.run_report --config <pipeline yaml> [--save]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boeing_landing.data.normalization import normalize


def _load_pooled(train_npz: Path, val_npz: Path):
    """Concatenate train+val (same channels and bounds) into one normalised set."""
    zt, zv = np.load(train_npz, allow_pickle=True), np.load(val_npz, allow_pickle=True)
    names = [str(n) for n in zt["input_names"]]
    lo, hi = zt["x_min"].astype(float), zt["x_max"].astype(float)
    method = str(zt["norm_method"]) if "norm_method" in zt else "minmax"
    x = np.vstack([zt["X"].astype(float), zv["X"].astype(float)])
    run = np.concatenate([zt["run"], zv["run"]]).astype(int)
    return normalize(x, lo, hi, method), x, run, names, lo, hi


def per_run_matrix(x_norm: np.ndarray, run: np.ndarray):
    """runs x channels matrix of per-run normalised means."""
    runs = sorted(set(run.tolist()))
    m = np.array([[x_norm[run == r, j].mean() for j in range(x_norm.shape[1])] for r in runs])
    return runs, m


def extremeness(runs: list[int], m: np.ndarray, names: list[str]):
    """Per run: mean |mean - other runs' mean| over channels, and the top channel."""
    rows = []
    for i, r in enumerate(runs):
        others = m[np.arange(len(runs)) != i].mean(axis=0)
        dist = np.abs(m[i] - others)
        rows.append({"run": r, "score": float(dist.mean()),
                     "top": names[int(dist.argmax())], "top_dist": float(dist.max())})
    return sorted(rows, key=lambda d: -d["score"])


def print_ranking(rows: list[dict], counts: dict) -> None:
    print(f"{'run':>4} {'frames':>7} {'extremeness':>12}   driving channel")
    for d in rows:
        print(f"{d['run']:>4} {counts[d['run']]:>7} {d['score']:>12.4f}   "
              f"{d['top']} ({d['top_dist']:.2f})")
    print(f"\nmost extreme run: {rows[0]['run']} (driven by {rows[0]['top']})")


def figure(runs, m, names, x_raw, run, rows, title: str):
    """Heatmap (fingerprints) + extremeness bar + wind scatter."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplot_mosaic([["heat", "bar"], ["heat", "wind"]],
                                 figsize=(15, max(5, 0.5 * len(runs) + 3)),
                                 width_ratios=[2.2, 1])

    im = ax["heat"].imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax["heat"].set_yticks(range(len(runs)), [f"run {r}" for r in runs])
    ax["heat"].set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax["heat"].set_title("per-run mean (normalised) -- outlier = off-colour row")
    fig.colorbar(im, ax=ax["heat"], fraction=0.025, pad=0.01)

    order = [d["run"] for d in rows][::-1]  # most extreme on top
    scores = [d["score"] for d in rows][::-1]
    ax["bar"].barh(range(len(order)), scores, color="#e67e22")
    ax["bar"].set_yticks(range(len(order)), [f"run {r}" for r in order])
    ax["bar"].set_xlabel("extremeness (dist. from other runs)")
    ax["bar"].set_title("who is far from the rest")

    if "wind_velocity_x" in names and "wind_velocity_y" in names:
        wx, wy = names.index("wind_velocity_x"), names.index("wind_velocity_y")
        for r in runs:
            mask = run == r
            px, py = x_raw[mask, wx].mean(), x_raw[mask, wy].mean()
            ax["wind"].scatter(px, py, s=40)
            ax["wind"].annotate(str(r), (px, py), fontsize=8,
                                xytext=(3, 3), textcoords="offset points")
        ax["wind"].axhline(0, c="gray", lw=0.5)
        ax["wind"].set_xlabel("mean wind_x (m/s)")
        ax["wind"].set_ylabel("mean wind_y (m/s)")
        ax["wind"].set_title("per-run wind (hidden variable)")
    else:
        ax["wind"].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    from boeing_landing.config import load_config
    from boeing_landing.train import DEFAULT_CONFIG, PROJECT_ROOT
    from utils.config import ensure_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the dataset: train_npz / val_npz")
    ap.add_argument("--save", action="store_true",
                    help="write the PNG into figures/dataset/ instead of showing")
    a = ap.parse_args()

    d = load_config(a.config)["dataset"]
    train_npz, val_npz = PROJECT_ROOT / d["train_npz"], PROJECT_ROOT / d["val_npz"]
    x_norm, x_raw, run, names, _, _ = _load_pooled(train_npz, val_npz)
    runs, m = per_run_matrix(x_norm, run)
    counts = {r: int((run == r).sum()) for r in runs}
    rows = extremeness(runs, m, names)
    print(f"dataset: {train_npz.parent.name}  ({len(runs)} runs, {len(names)} channels)")
    print_ranking(rows, counts)

    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        fig = figure(runs, m, names, x_raw, run, rows, f"run report -- {train_npz.parent.name}")
        out = ensure_dir(PROJECT_ROOT / "figures" / "dataset") / f"run_report_{train_npz.parent.name}.png"
        fig.savefig(out, dpi=130)
        print(f"saved -> {out}")
    else:
        try:
            import matplotlib.pyplot as plt
            figure(runs, m, names, x_raw, run, rows, f"run report -- {train_npz.parent.name}")
            plt.show()
        except Exception as e:  # headless machine: ranking already printed
            print(f"(no display for the figure: {e}; use --save to write a PNG)")


if __name__ == "__main__":
    main()
