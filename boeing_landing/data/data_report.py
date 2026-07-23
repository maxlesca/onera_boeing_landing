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
    """Read the pieces of a split this report needs.

    Args:
        npz_path: a split written by build_dataset.
    Returns:
        (X as float, channel names, x_min, x_max, norm_method) -- the bounds
        come from the npz itself, so train and val are read on the same scale.
    """
    z = np.load(npz_path, allow_pickle=True)
    method = str(z["norm_method"]) if "norm_method" in z else "minmax"
    return (z["X"].astype(float), [str(n) for n in z["input_names"]],
            z["x_min"].astype(float), z["x_max"].astype(float), method)


def _flag(norm_std: float, val_out_pct: float) -> str:
    """Classify one channel.

    Args:
        norm_std: its spread in [0,1] on train.
        val_out_pct: share of validation frames it sends outside [0,1].
    Returns:
        'CONSTANT', 'weak', 'val-shift' or 'ok' -- weakness is checked first,
        a flat channel having no distribution to shift.
    """
    if norm_std < CONSTANT_STD:
        return "CONSTANT"
    if norm_std < WEAK_STD:
        return "weak"
    if val_out_pct > OVERFLOW_PCT:
        return "val-shift"
    return "ok"


def _feature_row(name: str, raw, norm_train, norm_val) -> dict:
    """Diagnostics of a single channel.

    Args:
        name: the channel name.
        raw: its raw train values.
        norm_train, norm_val: its normalised train and validation values.
    Returns:
        One report row: raw range and std, normalised std, span, the share of
        validation frames outside [0,1], and the resulting flag.
    """
    val_out = 100.0 * float(np.mean((norm_val < 0) | (norm_val > 1)))
    norm_std = float(norm_train.std())
    return {
        "name": name,
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "raw_std": float(raw.std()),
        "norm_std": norm_std,
        "span": float(norm_train.max() - norm_train.min()),
        "val_out_pct": val_out,
        "flag": _flag(norm_std, val_out),
    }


def feature_stats(train_npz: Path, val_npz: Path) -> list[dict]:
    """Diagnose every input channel of a built dataset.

    Args:
        train_npz: the training split, which also fixes the bounds used.
        val_npz: the validation split, normalised with those same bounds so
            val_out% really measures distribution shift.
    Returns:
        One _feature_row per channel, in npz order.
    """
    x_tr, names, lo, hi, method = _load(train_npz)
    x_va = _load(val_npz)[0]
    n_tr, n_va = normalize(x_tr, lo, hi, method), normalize(x_va, lo, hi, method)
    return [_feature_row(name, x_tr[:, i], n_tr[:, i], n_va[:, i])
            for i, name in enumerate(names)]


def print_report(rows: list[dict]) -> None:
    """Print the diagnostics table and the two summary lists.

    Args:
        rows: what feature_stats returned.
    Returns:
        Nothing.
    """
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
    """Draw the report.

    Args:
        rows: what feature_stats returned.
        title: figure title, usually the dataset name.
    Returns:
        The matplotlib figure: two panels sharing the channel axis, norm_std
        (weakness) and val_out% (shift), coloured by flag.
    """
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
    """CLI entrypoint: report on the dataset the --config points at.

    Returns:
        Nothing; prints the table, then shows or saves the figure.
    """
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
    method = _load(train_npz)[4]
    print(f"dataset: {train_npz.parent.name}  ({len(rows)} input channels, norm={method})")
    if method != "minmax":
        print("note: the [0,1]-based columns (span, val_out%, weak flag) assume min-max.")
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
