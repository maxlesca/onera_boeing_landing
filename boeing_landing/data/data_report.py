# -*- coding: utf-8 -*-
"""Per-feature data diagnostics for a built dataset.

Two questions this answers at a glance:
- which channels are **weak** (near-constant, low dynamic range) -- little to
  learn from, candidates to drop or investigate;
- how much of the [0,1] range each channel actually uses, and how far the
  validation run falls **outside** the train-fit bounds (distribution shift).

Metrics per input channel (train split, normalised with the npz's own bounds):
- raw [min,max] and raw std: the physical amplitude ("mesurande faible" = tiny);
- norm_std: spread in [0,1] -- the weakness signal (near 0 = flat, ~0.29 = fills
  the range uniformly). This is the reliable flag: with data-driven bounds the
  train span is ~1 by construction, so span alone says nothing, norm_std does;
- span: fraction of [0,1] the train data occupies (informative mainly for the
  fixed physical bounds, where it shows how much of the envelope is used);
- val_out%: share of validation frames normalised outside [0,1].

    python -m boeing_landing.data.data_report --config <pipeline yaml> [--save]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boeing_landing.data.normalization import normalize

CONSTANT_STD = 0.01   # normalised std below this -> effectively flat
WEAK_STD = 0.08       # normalised std below this -> low dynamic range
OVERFLOW_PCT = 5.0    # % of val frames outside [0,1] worth flagging


def _load(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    return (z["X"].astype(float), [str(n) for n in z["input_names"]],
            z["x_min"].astype(float), z["x_max"].astype(float))


def _flag(norm_std: float, val_out_pct: float) -> str:
    if norm_std < CONSTANT_STD:
        return "CONSTANT"
    if norm_std < WEAK_STD:
        return "weak"
    if val_out_pct > OVERFLOW_PCT:
        return "val-shift"
    return "ok"


def feature_stats(train_npz: Path, val_npz: Path) -> list[dict]:
    """One row of diagnostics per input channel."""
    x_tr, names, lo, hi = _load(train_npz)
    x_va, _, _, _ = _load(val_npz)
    n_tr, n_va = normalize(x_tr, lo, hi), normalize(x_va, lo, hi)
    rows = []
    for i, name in enumerate(names):
        col, ntr, nva = x_tr[:, i], n_tr[:, i], n_va[:, i]
        val_out = 100.0 * float(np.mean((nva < 0) | (nva > 1)))
        norm_std = float(ntr.std())
        rows.append({
            "name": name,
            "raw_min": float(col.min()), "raw_max": float(col.max()),
            "raw_std": float(col.std()),
            "norm_std": norm_std,
            "span": float(ntr.max() - ntr.min()),
            "val_out_pct": val_out,
            "flag": _flag(norm_std, val_out),
        })
    return rows


def print_report(rows: list[dict]) -> None:
    print(f"{'feature':22s} {'raw[min, max]':>26s} {'raw_std':>10s} "
          f"{'norm_std':>9s} {'span':>6s} {'val_out%':>9s}  flag")
    for r in rows:
        print(f"{r['name']:22s} [{r['raw_min']:11.4g},{r['raw_max']:11.4g}] "
              f"{r['raw_std']:10.4g} {r['norm_std']:9.3f} {r['span']:6.2f} "
              f"{r['val_out_pct']:8.1f}%  {r['flag']}")
    weak = [r["name"] for r in rows if r["flag"] in ("CONSTANT", "weak")]
    shift = [r["name"] for r in rows if r["flag"] == "val-shift"]
    if weak:
        print(f"\nweak / near-constant (norm_std < {WEAK_STD}): {weak}")
    if shift:
        print(f"val distribution shift (> {OVERFLOW_PCT:.0f}% out of [0,1]): {shift}")
    if not weak and not shift:
        print("\nno weak channel, no val shift flagged.")


_COLORS = {"CONSTANT": "#c0392b", "weak": "#e67e22", "val-shift": "#8e44ad", "ok": "#2e86c1"}


def figure(rows: list[dict], title: str):
    """Two panels sharing the channel axis: norm_std (weakness) and val_out%."""
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    y = np.arange(len(names))[::-1]  # first channel on top
    colors = [_COLORS[r["flag"]] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.42 * len(names) + 1.5), sharey=True)

    ax1.barh(y, [r["norm_std"] for r in rows], color=colors)
    ax1.axvline(CONSTANT_STD, ls=":", c="#c0392b", lw=1)
    ax1.axvline(WEAK_STD, ls="--", c="#e67e22", lw=1, label=f"weak < {WEAK_STD}")
    ax1.set_yticks(y, names)
    ax1.set_xlabel("norm_std (spread in [0,1] -- low = weak)")
    ax1.legend(loc="lower right", fontsize=8)

    ax2.barh(y, [r["val_out_pct"] for r in rows], color=colors)
    ax2.axvline(OVERFLOW_PCT, ls="--", c="#8e44ad", lw=1, label=f"shift > {OVERFLOW_PCT:.0f}%")
    ax2.set_xlabel("val frames outside [0,1]  (%)")
    ax2.legend(loc="lower right", fontsize=8)

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
    rows = feature_stats(train_npz, val_npz)
    print(f"dataset: {train_npz.parent.name}  ({len(rows)} input channels)")
    print_report(rows)

    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        fig = figure(rows, f"data report -- {train_npz.parent.name}")
        out = ensure_dir(PROJECT_ROOT / "figures" / "dataset") / f"data_report_{train_npz.parent.name}.png"
        fig.savefig(out, dpi=130)
        print(f"saved -> {out}")
    else:
        try:
            import matplotlib.pyplot as plt
            figure(rows, f"data report -- {train_npz.parent.name}")
            plt.show()
        except Exception as e:  # headless machine: table already printed
            print(f"(no display for the figure: {e}; use --save to write a PNG)")


if __name__ == "__main__":
    main()
