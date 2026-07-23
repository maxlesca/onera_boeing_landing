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

K is the population the config declares (build.train_runs + build.val_runs, e.g.
30 + 3 for ned_wind_cfc phase 1), so a fold trains on the same number of runs as
the pipeline it scores -- not on every run of the csv.

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
    """Translate the CLI's string into what build_dataset expects.

    Args:
        value: the --physical-bounds argument.
    Returns:
        False for the off spellings, True for the core ones, else the string
        itself ('all' being the only other tier).
    """
    low = value.strip().lower()
    if low in {"false", "0", "none", "off"}:
        return False
    if low in {"true", "1", "core", "on"}:
        return True
    return low  # 'all'


def _safe_tag(tag: str) -> str:
    """Make an arm name usable as a directory name.

    Args:
        tag: the arm name, possibly holding '=' or '/' from a CLI override.
    Returns:
        It with everything but letters, digits, '-', '_' and '.' replaced by
        '_'.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)


def enumerate_runs(config: dict) -> list[int]:
    """The runs to hold out one by one.

    Args:
        config: the pipeline config, read for the dataset it points at.
    Returns:
        The sorted runs that survived cleaning, train and val pooled. Reading
        the built npz rather than the raw csv is what excludes runs dropped at
        build (e.g. missing from the NavDB), so no fold gets an empty
        validation set.
    Raises:
        SystemExit: that dataset has not been built yet.
    """
    d = config["dataset"]
    tr, va = PROJECT_ROOT / d["train_npz"], PROJECT_ROOT / d["val_npz"]
    if not tr.exists() or not va.exists():
        raise SystemExit(f"dataset not built: {tr} / {va}\n"
                         f"build it once first (make dataset CONFIG=...).")
    runs = np.concatenate([np.load(tr, allow_pickle=True)["run"],
                           np.load(va, allow_pickle=True)["run"]]).astype(int)
    return sorted(set(runs.tolist()))


def _fold_r2(ckpt: Path) -> float:
    """Score one fold on a scale-invariant metric, so arms that differ in
    norm_method stay comparable.

    The config is re-read from the run directory, not taken from the caller:
    only the archived one carries the I/O dimensions and the channel names the
    model was built with, which is exactly what rebuilding it needs.

    Args:
        ckpt: the fold's trained checkpoint; its directory holds config.yaml.
    Returns:
        The mean R2 over the command channels on the held-out run; channels
        with a constant target (e.g. the dead 'directional') are skipped.
    """
    from boeing_landing.evaluate import _labels, _val_arrays, mean_r2
    from utils.config import load_yaml
    from utils.evaluation import evaluate_arrays, regression_metrics
    config = load_yaml(ckpt.parent / "config.yaml")
    inputs, outputs = _val_arrays(config)
    yhat, target, _ = evaluate_arrays(config, config["dataloader"], ckpt, inputs, outputs)
    return mean_r2(regression_metrics(yhat, target, _labels(config)))


def _fold_train_runs(config: dict, held_out: int) -> set[int] | None:
    """The runs one fold trains on -- the population the config declares, so a
    fold scores the configured recipe and not a bigger one.

    Args:
        config: the pipeline config (build.train_runs + build.val_runs).
        held_out: the run this fold validates on.
    Returns:
        That population minus the held-out run, or None when the config
        declares none -- the fold then takes every run of the csv.
    """
    b = config.get("build", {})
    declared = {int(r) for r in b.get("train_runs") or []}
    if not declared:
        return None
    return (declared | {int(r) for r in b.get("val_runs") or []}) - {held_out}


def _build_fold(source: Path, run: int, out_dir: Path, config: dict,
                physical_bounds, norm_method: str, reuse: bool) -> None:
    """Rebuild the npz with one run held out, its bounds fit on the other runs
    only -- the held-out run never leaks into the normalisation.

    Args:
        source: the csv every fold is cut from.
        run: the run to hold out.
        out_dir: this fold's dataset directory.
        config: the pipeline config; every build knob comes from it, so a fold
            is the pipeline's own recipe with a different split, never a partly
            defaulted one.
        physical_bounds, norm_method: the arm's normalisation levers, which the
            CLI may override.
        reuse: skip the rebuild when both npz are already there.
    Returns:
        Nothing; writes the fold's npz.
    """
    if reuse and (out_dir / "landing_train.npz").exists() and (out_dir / "landing_val.npz").exists():
        print(f"  reuse existing {out_dir}")
        return
    b = config.get("build", {})
    build_dataset.build(source, {run}, out_dir,
                        extra_columns=b.get("extra_columns") or [],
                        input_set=b.get("input_set", "gps"),
                        physical_bounds=physical_bounds, norm_method=norm_method,
                        label_set=b.get("label_set", "commands"),
                        train_runs=_fold_train_runs(config, run))


def _fold_config(base: dict, tag: str, run: int, seed: int, fold_dir: Path,
                 max_epochs: int | None) -> dict:
    """The config one fold trains with.

    Args:
        base: the pipeline config, left untouched.
        tag: the arm name, which names the run folder.
        run: the held-out run.
        seed: the same seed on every fold and every arm, so a difference is
            attributable to the held-out run or the recipe, not to the init.
        fold_dir: this fold's dataset directory.
        max_epochs: epoch override for a quick trial, None to keep the config's.
    Returns:
        A deep copy pointed at the fold's npz and tagged runs/loro_<tag>/run<n>/.
    """
    cfg = deepcopy(base)
    cfg["training"]["seed"] = seed
    if max_epochs:
        cfg["training"]["max_epochs"] = max_epochs
    cfg["checkpoint_name"] = f"loro_{tag}"
    cfg["run_tag"] = f"run{run}"
    rel = fold_dir.relative_to(PROJECT_ROOT).as_posix()
    cfg["dataset"]["train_npz"] = f"{rel}/landing_train.npz"
    cfg["dataset"]["val_npz"] = f"{rel}/landing_val.npz"
    return cfg


def _fold(base: dict, source: Path, run: int, position: str, tag: str, seed: int,
          physical_bounds, norm_method: str, with_r2: bool, reuse_data: bool,
          max_epochs: int | None) -> dict:
    """Run one fold end to end: build, train, score.

    Args:
        base: the pipeline config.
        source: the csv every fold is cut from.
        run: the run held out here.
        position: progress marker printed in the header, e.g. '3/31'.
        tag: the arm name.
        seed: the seed shared by every fold.
        physical_bounds, norm_method: the arm's normalisation levers.
        with_r2: also run the inference pass that yields the mean R2.
        reuse_data: reuse an existing fold dataset instead of rebuilding it.
        max_epochs: epoch override for a quick trial.
    Returns:
        That fold's result row: run, validation frames, val_loss, mean R2 (NaN
        when skipped) and run dir.
    """
    print(f"\n=== fold: hold out run {run} ({position}) ===")
    fold_dir = PROJECT_ROOT / "datasets" / "loro" / tag / f"run{run}"
    _build_fold(source, run, fold_dir, base, physical_bounds, norm_method, reuse_data)

    cfg = _fold_config(base, tag, run, seed, fold_dir, max_epochs)
    ckpt = train_config(cfg, PROJECT_ROOT)
    row = {"run": run,
           "val_frames": int(np.load(fold_dir / "landing_val.npz", allow_pickle=True)["run"].size),
           "val_loss": val_loss_from_checkpoint(ckpt),
           "mean_r2": float("nan"), "run_dir": str(ckpt.parent)}
    if with_r2:
        try:
            row["mean_r2"] = _fold_r2(ckpt)
        except Exception as e:   # never let R2 kill the sweep
            print(f"  (R2 skipped: {type(e).__name__}: {e})")
    return row


def sweep(config_path: Path, tag: str, physical_bounds, norm_method: str,
          seed: int, with_r2: bool, reuse_data: bool,
          runs: list[int] | None, max_epochs: int | None) -> dict:
    """Train one fold per run: the whole arm.

    Args:
        config_path: the pipeline config being scored.
        tag: the arm name (dataset folders, run folders, results.json).
        physical_bounds, norm_method: the arm's normalisation levers.
        seed: the seed shared by every fold and every arm.
        with_r2: also compute the mean R2 of each fold.
        reuse_data: reuse existing fold datasets.
        runs: restrict the folds to these runs; None holds out every run of the
            built dataset.
        max_epochs: epoch override for a quick trial.
    Returns:
        The arm's results dict: its levers, one row per fold, and the summary.
    """
    base = load_pipeline_config(config_path)
    source = build_dataset._resolve_source(None, base)
    all_runs = runs or enumerate_runs(base)
    rows = [_fold(base, source, run, f"{i}/{len(all_runs)}", tag, seed,
                  physical_bounds, norm_method, with_r2, reuse_data, max_epochs)
            for i, run in enumerate(all_runs, 1)]
    return {"tag": tag, "config": str(config_path), "seed": seed,
            "physical_bounds": physical_bounds, "norm_method": norm_method,
            "runs": rows, "summary": _summary(rows)}


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and spread of the finite scores.

    Args:
        values: the scores kept.
    Returns:
        (mean, standard deviation); (NaN, 0) on an empty list and a 0 spread on
        a single value.
    """
    if not values:
        return float("nan"), 0.0
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _summary(rows: list[dict]) -> dict:
    """Aggregate the folds into the arm's score.

    Args:
        rows: the fold rows sweep collected; NaN scores are dropped, so a fold
            whose R2 was skipped does not poison the mean.
    Returns:
        Mean and std of val_loss and of mean_r2.
    """
    loss_mean, loss_std = _mean_std([r["val_loss"] for r in rows if math.isfinite(r["val_loss"])])
    r2_mean, r2_std = _mean_std([r["mean_r2"] for r in rows if math.isfinite(r["mean_r2"])])
    return {"val_loss_mean": loss_mean, "val_loss_std": loss_std,
            "mean_r2_mean": r2_mean, "mean_r2_std": r2_std}


def report(res: dict) -> None:
    """Print an arm's folds, hardest first, and its summary.

    Args:
        res: what sweep returned (or a results.json read back).
    Returns:
        Nothing.
    """
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
    """Archive an arm so it can be re-read and compared without retraining.

    Args:
        res: what sweep returned.
    Returns:
        Path of the written runs/loro/<tag>/results.json.
    """
    out = PROJECT_ROOT / "runs" / "loro" / res["tag"] / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")
    return out


def figure_single(res: dict, save: bool) -> None:
    """Draw one arm: val_loss and mean R2 per held-out run.

    Args:
        res: what sweep returned (or a results.json read back).
        save: write a PNG instead of showing the window.
    Returns:
        Nothing.
    """
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


def _by_run(arm: dict) -> dict:
    """Index one arm's folds by held-out run.

    Args:
        arm: a results dict.
    Returns:
        {run: its fold row}, so arms can be aligned run by run even when they
        did not cover exactly the same runs.
    """
    return {r["run"]: r for r in arm["runs"]}


def compare(paths: list[Path], save: bool) -> None:
    """Put finished arms side by side, aligned by run: delta table and grouped
    bars. No training -- everything comes from the saved results.

    Args:
        paths: the results.json of each arm; the first is the reference the
            deltas are taken against.
        save: write a PNG instead of showing the window.
    Returns:
        Nothing.
    """
    arms = [json.loads(Path(p).read_text()) for p in paths]
    runs = sorted({r["run"] for a in arms for r in a["runs"]})
    maps = [_by_run(a) for a in arms]

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
    """Deliver a figure the way the machine allows.

    Args:
        fig: the matplotlib figure.
        stem: file name without extension, used when saving.
        save: True writes figures/loro/<stem>.png, False opens a window.
    Returns:
        Nothing; on a headless machine without --save it says so instead of
        failing -- the table and results.json already carry everything.
    """
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
    """CLI entrypoint: run an arm, or re-render finished ones with --compare.

    Returns:
        Nothing; an arm prints its table, writes results.json and its figure.
    """
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
