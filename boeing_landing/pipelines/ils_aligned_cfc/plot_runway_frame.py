# -*- coding: utf-8 -*-
"""Plot the trajectories of an augmented landing CSV in the runway frame.

Three panels over the rows that have runway-frame coordinates (augment_ned):
top view (pos_along/pos_cross), vertical profile (pos_along/pos_up), and the
cross-check of pos_cross against the sim's own localizer_error_m -- the two
measure the same physical quantity through independent paths, so the cloud
must sit on y = x (both count positive LEFT of the centerline).

    python -m boeing_landing.pipelines.ils_aligned_cfc.plot_runway_frame datasets/ldg_dataset_images_ned.csv [--save]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito palette (colorblind-safe), one fixed color per runway designator
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def load_augmented(path: Path) -> pd.DataFrame:
    """The augmented rows only (runs whose runway is in the nav database)."""
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "pos_along" not in df:
        raise SystemExit(f"{path} has no runway-frame columns (run augment_ned first)")
    df["runway"] = df["runway"].str.strip()
    return df[df["pos_along"].notna()]


def _colors(runways: list[str]) -> dict[str, str]:
    return dict(zip(sorted(set(runways)), _PALETTE))


def _plot_runs(df: pd.DataFrame, ax, x: str, y: str, colors: dict, scale_x=1.0) -> None:
    """One line per run, one color and one legend entry per runway."""
    seen = set()
    for (qfu, _), run in df.groupby(["runway", "simulationindex"]):
        label = qfu if qfu not in seen else None
        seen.add(qfu)
        ax.plot(run[x] * scale_x, run[y], color=colors[qfu], lw=1.2, label=label)


def plot_top(df: pd.DataFrame, ax, colors: dict) -> None:
    _plot_runs(df, ax, "pos_along", "pos_cross", colors, scale_x=1e-3)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set(xlabel="pos_along (km)", ylabel="pos_cross (m, + = gauche)",
           title="Vue de dessus — repère piste")


def plot_profile(df: pd.DataFrame, ax, colors: dict) -> None:
    _plot_runs(df, ax, "pos_along", "pos_up", colors, scale_x=1e-3)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set(xlabel="pos_along (km)", ylabel="pos_up (m)", title="Profil vertical")


def plot_localizer_check(df: pd.DataFrame, ax, colors: dict) -> None:
    """pos_cross vs the sim's localizer error: expected on y = x."""
    for qfu, g in df.groupby("runway"):
        ax.scatter(g["localizer_error_m"], g["pos_cross"], s=3,
                   color=colors[qfu], alpha=0.4)
    lim = float(np.nanmax(np.abs(df["localizer_error_m"])))
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1, ls="--", label="y = x")
    ax.set(xlabel="localizer_error_m (sim, + = gauche)",
           ylabel="pos_cross (m, + = gauche)",
           title="Validation croisée (attendu : y = x)")


def figure(df: pd.DataFrame):
    colors = _colors(df["runway"].tolist())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_top(df, axes[0], colors)
    plot_profile(df, axes[1], colors)
    plot_localizer_check(df, axes[2], colors)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    airports = ", ".join(sorted(df["airport"].str.strip().unique()))
    fig.suptitle(f"{airports} — {df['simulationindex'].nunique()} runs dans le repère piste "
                 "(origine LTP/FTP, axes along/cross/up alignés ILS)")
    fig.tight_layout()
    return fig


def main() -> None:
    from boeing_landing.train import PROJECT_ROOT
    from utils.config import ensure_dir

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="augmented csv (output of augment_ned)")
    ap.add_argument("--save", action="store_true",
                    help="write the PNG into figures/dataset/ instead of showing")
    a = ap.parse_args()

    fig = figure(load_augmented(a.csv))
    if a.save:
        out = ensure_dir(PROJECT_ROOT / "figures" / "dataset") / f"{a.csv.stem}.png"
        fig.savefig(out, dpi=130)
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
