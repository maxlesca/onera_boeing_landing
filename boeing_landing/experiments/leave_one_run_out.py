# -*- coding: utf-8 -*-
"""Leave-one-run-out cross-validation over the dataset's runs.

Instead of trusting a single held-out run (e.g. run 7), hold out EVERY run in
turn: for K runs, train K times, each on the other K-1 runs, validating on the
one left out. The K scores estimate how a *recipe* (architecture + hyper-params +
normalisation) generalises to an unseen run -- a property of the training
procedure, not of any one trained model. The K models are read then discarded;
to deploy, retrain once on all runs with the winning recipe.

Each fold rebuilds its own npz so the normalisation bounds are fit on that fold's
training runs only -- the held-out run never leaks into the bounds. Same seed on
every fold (and across arms) so a difference is attributable to the held-out run
(or the recipe), not the init lottery.

`val_loss` is comparable across folds of one arm and across arms that share the
normalisation method (same label scale); it is NOT comparable min-max vs z-score.
The mean **R2** (scale-invariant, per-command, averaged) is comparable across all
arms -- use it when the arms differ in norm_method.

    # one arm (uses the config's build settings):
    python -m boeing_landing.experiments.leave_one_run_out --config <pipeline.yaml>
    # override the lever without a new yaml, and tag the arm:
    python -m boeing_landing.experiments.leave_one_run_out --config <cfg> \
        --physical-bounds all --tag ils_all
    # re-render a finished arm's figure from its results.json (no training):
    python -m boeing_landing.experiments.leave_one_run_out --compare \
        runs/loro/ils_core/results.json --save
    # compare two finished arms (also no training):
    python -m boeing_landing.experiments.leave_one_run_out --compare \
        runs/loro/ils_core/results.json runs/loro/ils_all/results.json --save
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

import numpy as np

from boeing_landing.config import load_pipeline_config
from boeing_landing.data import build_dataset
from boeing_landing.train import (DEFAULT_CONFIG, PROJECT_ROOT, train_config,
                                  val_loss_from_checkpoint)


def _parse_physical_bounds(value: str):
    """CLI string -> the value build_dataset expects (False/True/'core'/'all')."""
    low = value.strip().lower()
    if low in {"false", "0", "none", "off"}:
        return False
    if low in {"true", "1", "core", "on"}:
        return True
    return low  # 'all'


def _safe_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


def enumerate_runs(config: dict) -> list[int]:
    """The runs that survived cleaning, read from the dataset the config points
    at (train + val pooled). Enumerating from the built npz -- not the raw csv --
    means runs fully dropped at build (e.g. missing NavDB) are excluded, so every
    fold has a non-empty validation set."""
    d = config["dataset"]
    tr, va = PROJECT_ROOT / d["train_npz"], PROJECT_ROOT / d["val_npz"]
    if not tr.exists() or not va.exists():
        raise SystemExit(f"dataset not built: {tr} / {va}\n"
                         f"build it once first (make dataset CONFIG=...).")
    runs = np.concatenate([np.load(tr, allow_pickle=True)["run"],
                           np.load(va, allow_pickle=True)["run"]]).astype(int)
    return sorted(set(runs.tolist()))


def _fold_r2(config: dict, ckpt: Path) -> float:
    """Mean R2 over the command channels on the held-out run (scale-invariant, so
    comparable across normalisation methods). NaN channels (constant target, e.g.
    the dead 'directional') are skipped."""
    from boeing_landing.data.features import LABELS
    from boeing_landing.evaluate import _val_arrays
    from utils.evaluation import evaluate_arrays, regression_metrics
    inputs, outputs = _val_arrays(config)
    yhat, target, _ = evaluate_arrays(config, config["dataloader"], ckpt, inputs, outputs)
    reg = regression_metrics(yhat, target, LABELS)
    r2s = [m["r2"] for m in reg["per_channel"].values() if math.isfinite(m["r2"])]
    return float(statistics.mean(r2s)) if r2s else float("nan")


def _build_fold(source: Path, run: int, out_dir: Path, config: dict,
                physical_bounds, norm_method: str, reuse: bool) -> None:
    """Rebuild the npz with `run` held out, bounds fit on the other runs only."""
    if reuse and (out_dir / "landing_train.npz").exists() and (out_dir / "landing_val.npz").exists():
        print(f"  reuse existing {out_dir}")
        return
    b = config.get("build", {})
    build_dataset.build(source, {run}, out_dir,
                        extra_columns=b.get("extra_columns") or [],
                        input_set=b.get("input_set", "gps"),
                        physical_bounds=physical_bounds, norm_method=norm_method)


def sweep(config_path: Path, tag: str, physical_bounds, norm_method: str,
          seed: int, with_r2: bool, reuse_data: bool,
          runs: list[int] | None, max_epochs: int | None) -> dict:
    """Train one fold per run; return the arm's results dict."""
    base = load_pipeline_config(config_path)
    source = build_dataset._resolve_source(None, base)
    all_runs = runs or enumerate_runs(base)
    data_root = PROJECT_ROOT / "datasets" / "loro" / tag

    rows = []
    for run in all_runs:
        print(f"\n=== fold: hold out run {run} ({all_runs.index(run)+1}/{len(all_runs)}) ===")
        fold_dir = data_root / f"run{run}"
        _build_fold(source, run, fold_dir, base, physical_bounds, norm_method, reuse_data)

        cfg = deepcopy(base)
        cfg["training"]["seed"] = seed
        if max_epochs:
            cfg["training"]["max_epochs"] = max_epochs
        cfg["checkpoint_name"] = f"loro_{tag}"
        cfg["run_tag"] = f"run{run}"
        rel = fold_dir.relative_to(PROJECT_ROOT).as_posix()
        cfg["dataset"]["train_npz"] = f"{rel}/landing_train.npz"
        cfg["dataset"]["val_npz"] = f"{rel}/landing_val.npz"

        ckpt = train_config(cfg, PROJECT_ROOT)
        val_frames = int(np.load(fold_dir / "landing_val.npz", allow_pickle=True)["run"].size)
        row = {"run": run, "val_frames": val_frames,
               "val_loss": val_loss_from_checkpoint(ckpt),
               "mean_r2": float("nan"), "run_dir": str(ckpt.parent)}
        if with_r2:
            try:
                row["mean_r2"] = _fold_r2(cfg, ckpt)
            except Exception as e:   # never let R2 kill the sweep
                print(f"  (R2 skipped: {e})")
        rows.append(row)

    return {"tag": tag, "config": str(config_path), "seed": seed,
            "physical_bounds": physical_bounds, "norm_method": norm_method,
            "runs": rows, "summary": _summary(rows)}


def _summary(rows: list[dict]) -> dict:
    losses = [r["val_loss"] for r in rows if math.isfinite(r["val_loss"])]
    r2s = [r["mean_r2"] for r in rows if math.isfinite(r["mean_r2"])]
    def ms(xs):
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)
    lm, ls = ms(losses); rm, rs = ms(r2s)
    return {"val_loss_mean": lm, "val_loss_std": ls, "mean_r2_mean": rm, "mean_r2_std": rs}


def report(res: dict) -> None:
    print(f"\n=== leave-one-run-out :: {res['tag']} "
          f"(pb={res['physical_bounds']}, norm={res['norm_method']}, seed={res['seed']}) ===")
    print(f"{'run':>4} {'frames':>7} {'val_loss':>10} {'mean_r2':>9}")
    for r in sorted(res["runs"], key=lambda d: -d["val_loss"]):   # hardest fold first
        print(f"{r['run']:>4} {r['val_frames']:>7} {r['val_loss']:>10.6f} {r['mean_r2']:>9.4f}")
    s = res["summary"]
    print(f"\n  val_loss  mean={s['val_loss_mean']:.6f}  std={s['val_loss_std']:.6f}")
    print(f"  mean_r2   mean={s['mean_r2_mean']:.4f}  std={s['mean_r2_std']:.4f}")
    print("  (val_loss comparable only within one norm_method; use mean_r2 across methods)")


def save_results(res: dict) -> Path:
    out = PROJECT_ROOT / "runs" / "loro" / res["tag"] / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")
    return out


def figure_single(res: dict, save: bool) -> None:
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(res["runs"], key=lambda d: d["run"])
    runs = [str(r["run"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.bar(runs, [r["val_loss"] for r in rows], color="#2e86c1")
    ax1.axhline(res["summary"]["val_loss_mean"], ls="--", c="k", lw=1,
                label=f"mean {res['summary']['val_loss_mean']:.4f}")
    ax1.set_xlabel("held-out run"); ax1.set_ylabel("val_loss"); ax1.legend()
    ax1.set_title(f"LORO val_loss -- {res['tag']}")
    ax2.bar(runs, [r["mean_r2"] for r in rows], color="#27ae60")
    ax2.axhline(res["summary"]["mean_r2_mean"], ls="--", c="k", lw=1,
                label=f"mean {res['summary']['mean_r2_mean']:.3f}")
    ax2.set_xlabel("held-out run"); ax2.set_ylabel("mean R2"); ax2.legend()
    ax2.set_title("LORO mean R2 (scale-invariant)")
    fig.tight_layout()
    _show_or_save(fig, f"loro_{res['tag']}", save)


def compare(paths: list[Path], save: bool) -> None:
    """Side-by-side of finished arms, aligned by run: delta table + grouped bars."""
    arms = [json.loads(Path(p).read_text()) for p in paths]
    runs = sorted({r["run"] for a in arms for r in a["runs"]})
    def by_run(a):
        return {r["run"]: r for r in a["runs"]}
    maps = [by_run(a) for a in arms]

    print("\n=== LORO comparison (val_loss | mean_r2) ===")
    header = "run   " + "  ".join(f"{a['tag'][:16]:>16}" for a in arms)
    print(header)
    for run in runs:
        cells = []
        for m in maps:
            r = m.get(run)
            cells.append(f"{r['val_loss']:.4f}/{r['mean_r2']:+.3f}" if r else "        -       ")
        print(f"{run:>3}   " + "  ".join(f"{c:>16}" for c in cells))
    print("mean  " + "  ".join(f"{a['summary']['val_loss_mean']:.4f}/{a['summary']['mean_r2_mean']:+.3f}".rjust(16)
                                for a in arms))
    base = arms[0]
    for a in arms[1:]:
        dv = a["summary"]["val_loss_mean"] - base["summary"]["val_loss_mean"]
        dr = a["summary"]["mean_r2_mean"] - base["summary"]["mean_r2_mean"]
        print(f"  {a['tag']} vs {base['tag']}: dval_loss={dv:+.6f} (lower=better)  "
              f"dR2={dr:+.4f} (higher=better)")

    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(runs)); w = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(runs)), 4.5))
    for i, (a, m) in enumerate(zip(arms, maps)):
        ax.bar(x + i * w, [m.get(r, {"val_loss": np.nan})["val_loss"] for r in runs],
               w, label=a["tag"])
    ax.set_xticks(x + w * (len(arms) - 1) / 2, [str(r) for r in runs])
    ax.set_xlabel("held-out run"); ax.set_ylabel("val_loss"); ax.legend()
    ax.set_title("LORO val_loss by arm (comparable within a norm_method)")
    fig.tight_layout()
    _show_or_save(fig, "loro_compare_" + "_vs_".join(a["tag"] for a in arms), save)


def _show_or_save(fig, stem: str, save: bool) -> None:
    if save:
        from utils.config import ensure_dir
        out = ensure_dir(PROJECT_ROOT / "figures" / "loro") / f"{stem}.png"
        fig.savefig(out, dpi=130)
        print(f"saved -> {out}")
    else:
        try:   # headless training machine: the table + results.json already carry everything
            import matplotlib.pyplot as plt
            plt.show()
        except Exception as e:
            print(f"(no display for the figure: {e}; add SAVE=1 to write a PNG instead)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--physical-bounds", default=None,
                    help="override build.physical_bounds: false|true|core|all")
    ap.add_argument("--norm-method", default=None, help="override build.norm_method: minmax|zscore")
    ap.add_argument("--tag", default=None, help="arm name (folders, results.json); default derived")
    ap.add_argument("--seed", type=int, default=42, help="same seed on every fold (and arm)")
    ap.add_argument("--runs", default=None, help="comma list to restrict folds, e.g. 7,4,25 (default: all)")
    ap.add_argument("--max-epochs", type=int, default=None, help="override epochs (quick trial)")
    ap.add_argument("--no-r2", action="store_true", help="skip the R2 inference pass (faster)")
    ap.add_argument("--reuse-data", action="store_true", help="skip fold rebuild if the npz exist")
    ap.add_argument("--save", action="store_true", help="write PNG(s) instead of showing")
    ap.add_argument("--compare", nargs="+", type=Path, default=None,
                    help="re-render from saved results.json without training: one file "
                         "-> that arm's figure, several -> the side-by-side comparison")
    a = ap.parse_args()

    if a.compare:
        # re-render from saved results.json -- no training. One file -> the arm's
        # own two-panel figure; several -> the side-by-side comparison.
        if len(a.compare) == 1:
            res = json.loads(Path(a.compare[0]).read_text())
            report(res)
            figure_single(res, a.save)
        else:
            compare(a.compare, a.save)
        return

    base = load_pipeline_config(a.config)
    b = base.get("build", {})
    physical_bounds = (_parse_physical_bounds(a.physical_bounds) if a.physical_bounds is not None
                       else b.get("physical_bounds", False))
    norm_method = a.norm_method or b.get("norm_method", "minmax")
    tag = _safe_tag(a.tag or f"{Path(a.config).stem}_pb-{physical_bounds}_{norm_method}")
    runs = [int(r) for r in a.runs.split(",")] if a.runs else None

    res = sweep(a.config, tag, physical_bounds, norm_method, a.seed,
                not a.no_r2, a.reuse_data, runs, a.max_epochs)
    report(res)
    save_results(res)
    figure_single(res, a.save)


if __name__ == "__main__":
    main()
