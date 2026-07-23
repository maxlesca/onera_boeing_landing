# -*- coding: utf-8 -*-
"""Build the training dataset: raw CSV -> npz, per-run train/val split.

Inputs = inertial state + raw GPS (ILS dropped). Labels = flight commands.
Columns are stored in the canonical order (features.CANONICAL_INPUTS); the
loader permutes them for the conv-order study. Normalisation bounds are computed
on the train runs only and embedded in each npz, so they travel with the data.

Split is per run, never per frame: consecutive frames are near-identical, so a
random split would leak the validation set into training.

Val runs and output dir come from the pipeline config (`build:` section).

Usage:
    python -m boeing_landing.data.build_dataset [SOURCE] [--config path.yaml]
    SOURCE = the dataset .zip or the extracted ';'-separated .csv. Optional: when
    omitted, the pipeline's upstream step (`prepare:` for a raw delivery,
    `augment:` for the local-frame coordinates) is run to produce it, so one
    command builds the whole chain. --force re-runs that step.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from boeing_landing.data.features import (INPUT_SETS, LABEL_SETS, LABELS,
                                          OPTIONAL_COLUMNS)
from boeing_landing.data.normalization import add_angle_encodings, resolve_norm


def load_csv(source: Path) -> pd.DataFrame:
    """Read the dataset CSV, from a .zip or a plain .csv (';'-separated).
    low_memory=False: the runway designator column mixes '16R' and '2', which
    otherwise triggers a mixed-dtype warning."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            print(f"reading {name} from {source.name}")
            return pd.read_csv(io.BytesIO(z.read(name)), sep=";", low_memory=False)
    return pd.read_csv(source, sep=";", low_memory=False)


def clean(df: pd.DataFrame, inputs: list[str], labels: list[str] = LABELS) -> pd.DataFrame:
    """Drop rows with missing fields, add an int `run`, sort by (run, time)."""
    n_total = len(df)
    needed = ["simulationindex", "time"] + list(inputs) + list(labels)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV lacks the columns {missing}. The local-frame pipelines need the "
            f"AUGMENTED csv, which `make dataset` produces on its own from the "
            f"pipeline's prepare:/augment: block -- pass CSV=... only to override it.")
    # dropna on the REQUIRED columns only: an optional column (image_filename) is
    # neither an input nor a label, so a hole in it must not cost a training frame.
    df = df.dropna(subset=needed).copy()
    print(f"{n_total} rows read, {n_total - len(df)} dropped (NaN), {len(df)} kept")
    for col in (c for c in OPTIONAL_COLUMNS if c in df.columns):
        if n_missing := int(df[col].isna().sum()):
            print(f"  note: {col} empty on {n_missing} kept rows")
            df[col] = df[col].fillna("")
    df["run"] = df["simulationindex"].astype(int)
    return df.sort_values(["run", "time"]).reset_index(drop=True)


def split_runs(df: pd.DataFrame, val_runs: set[int],
               train_runs: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, val) by run id. `train_runs` restricts training to an
    explicit list -- every other run is discarded, which is what lets a phase-1
    study train on 30 of the 85 available runs. Left empty, training takes every
    run that is not held out (the historical behaviour)."""
    present = set(df["run"].unique())
    # strict, like the train check below: val runs are picked one by one for what
    # each tests, so validating on the survivors would drop a test case in silence
    if missing := val_runs - present:
        raise SystemExit(f"val runs absent from the csv (or dropped as NaN by clean): "
                         f"{sorted(missing)} -- runs available: {sorted(present)}")
    val = df[df["run"].isin(val_runs)]
    if not train_runs:
        return df[~df["run"].isin(val_runs)], val
    if missing := train_runs - present:
        raise SystemExit(f"train runs absent from the csv: {sorted(missing)}")
    if overlap := train_runs & val_runs:
        raise SystemExit(f"runs listed in BOTH train and val: {sorted(overlap)}")
    return df[df["run"].isin(train_runs)], val


def compute_bounds(train: pd.DataFrame, inputs: list[str], physical=False,
                   method: str = "minmax", labels: list[str] = LABELS) -> dict:
    """Normalisation params (see data.normalization). `method`: 'minmax' (default)
    or 'zscore'. For minmax, `physical` (True/"core" -> position+attitude, "all" ->
    also wind/velocities/rates) uses the fixed bound for channels that have one;
    every other channel (and all labels) uses the train-split stat. x_min/x_max
    hold (min,max) for minmax and (mean,std) for zscore.
    Computed on the TRAIN split only, then embedded in both npz -- validation is
    normalised with the training bounds, never with its own."""
    x_min, x_max = resolve_norm(train, inputs, method, physical)
    y_min, y_max = resolve_norm(train, labels, method, physical)
    return {"inputs": inputs, "labels": list(labels), "norm_method": method,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def save_split(name: str, part: pd.DataFrame, bounds: dict, out_dir: Path) -> None:
    """Write one split to `landing_<name>.npz` (raw values + embedded bounds)."""
    inputs, labels = bounds["inputs"], bounds["labels"]
    optional = {c: part[c].to_numpy() for c in OPTIONAL_COLUMNS if c in part.columns}
    np.savez_compressed(
        out_dir / f"landing_{name}.npz",
        X=part[inputs].to_numpy(np.float32),
        Y=part[labels].to_numpy(np.float32),
        run=part["run"].to_numpy(np.int32),
        t=part["time"].to_numpy(np.float32),
        input_names=np.array(inputs),
        label_names=np.array(labels),
        norm_method=np.array(bounds.get("norm_method", "minmax")),
        x_min=np.array(bounds["x_min"], np.float32),
        x_max=np.array(bounds["x_max"], np.float32),
        y_min=np.array(bounds["y_min"], np.float32),
        y_max=np.array(bounds["y_max"], np.float32),
        **{"image" if c == "image_filename" else c: v for c, v in optional.items()},
    )
    print(f"  {name}: {len(part):6d} frames, runs {sorted(part['run'].unique())} "
          f"-> {out_dir / f'landing_{name}.npz'}")


def build(source: Path, val_runs: set[int], out_dir: Path,
          extra_columns: list[str] = (), input_set: str = "gps",
          physical_bounds=False, norm_method: str = "minmax",
          label_set: str = "commands", train_runs: set[int] | None = None) -> None:
    """input_set: the base input columns (features.INPUT_SETS -- 'gps' keeps the
    GPS position as absolute lat/lon/alt, 'runway'/'magnetic' convert that same
    position into a local frame). ILS is in none of them.
    label_set: the command channels to predict (features.LABEL_SETS).
    extra_columns: additional CSV columns appended as inputs.
    train_runs: explicit training runs (default: everything not held out).
    physical_bounds: normalise with the fixed physical bounds (data.normalization)
    where a channel has one -- airport-independent; off by default (gps_cfc)."""
    if input_set not in INPUT_SETS:
        raise SystemExit(f"unknown input_set {input_set!r}; choose from {sorted(INPUT_SETS)}")
    if label_set not in LABEL_SETS:
        raise SystemExit(f"unknown label_set {label_set!r}; choose from {sorted(LABEL_SETS)}")
    inputs, labels = INPUT_SETS[input_set] + list(extra_columns), LABEL_SETS[label_set]
    df = add_angle_encodings(load_csv(source), inputs)
    df = clean(df, inputs, labels)
    train, val = split_runs(df, val_runs, train_runs)
    bounds = compute_bounds(train, inputs, physical_bounds, norm_method, labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_split("train", train, bounds, out_dir)
    save_split("val", val, bounds, out_dir)
    (out_dir / "normalization_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print(f"inputs={len(inputs)} (set={input_set}, no ILS), "
          f"labels={len(labels)} (set={label_set}), "
          f"norm={norm_method}, physical_bounds={physical_bounds}")


def _resolve_source(source: Path | None, config: dict, force: bool = False) -> Path:
    """The source csv to build from, produced on demand.

    A pipeline declares at most one upstream step -- `prepare:` (rename a raw
    delivery) or `augment:` (add the local-frame coordinates) -- and this runs it
    when its output csv is not there yet, so `make dataset` builds the whole
    chain in one command instead of asking for the steps in the right order.
    An existing csv is reused as is; FORCE=1 rebuilds it.
    gps_cfc declares neither and must be given a source explicitly.

    Declaring both is rejected rather than half-executed: only the first would
    run, and build() would then fail on the columns the second was to produce.
    The day a delivery needs both, make this an ordered list of steps.
    """
    if source is not None:
        return source
    if all(config.get(s) for s in ("prepare", "augment")):
        raise SystemExit("this config declares both a prepare: and an augment: step; "
                         "only one upstream step is supported (see _resolve_source)")
    for section, run_step in (("prepare", _run_prepare), ("augment", _run_augment)):
        cfg = config.get(section)
        if not cfg:
            continue
        out = Path(cfg["out_csv"])
        if out.exists() and not force:
            print(f"reusing {out} (FORCE=1 to rebuild it)")
            return out
        return run_step(config)
    raise SystemExit("no source csv: pass it as an argument (CSV=...); this config "
                     "declares neither a prepare: nor an augment: step (e.g. gps_cfc "
                     "builds straight from the raw csv)")


def _run_prepare(config: dict) -> Path:
    from boeing_landing.data.prepare import run_prepare
    return run_prepare(config, None, None, None)


def _run_augment(config: dict) -> Path:
    from boeing_landing.data.augment import run_augment
    return run_augment(config)


def main() -> None:
    from boeing_landing.config import load_config
    from boeing_landing.train import DEFAULT_CONFIG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, nargs="?", default=None,
                    help="dataset .zip or .csv (default: the config's augment.out_csv)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the build: section")
    ap.add_argument("--force", action="store_true",
                    help="re-run the upstream prepare/augment step even if its csv exists")
    a = ap.parse_args()

    config = load_config(a.config)
    build_cfg = config.get("build", {})
    val_runs = {int(r) for r in build_cfg.get("val_runs", [8])}
    train_runs = {int(r) for r in build_cfg.get("train_runs") or []}
    build(_resolve_source(a.source, config, a.force), val_runs,
          Path(build_cfg.get("out_dir", "datasets/gps_no_ils")),
          extra_columns=build_cfg.get("extra_columns") or [],
          input_set=build_cfg.get("input_set", "gps"),
          physical_bounds=build_cfg.get("physical_bounds", False),
          norm_method=build_cfg.get("norm_method", "minmax"),
          label_set=build_cfg.get("label_set", "commands"),
          train_runs=train_runs)


if __name__ == "__main__":
    main()
