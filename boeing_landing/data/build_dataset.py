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
    """Read the dataset csv.

    Args:
        source: a ';'-separated .csv, or a .zip holding one.
    Returns:
        The frame as delivered, nothing renamed or dropped. low_memory=False:
        the runway designator column mixes '16R' and '2', which otherwise
        triggers a mixed-dtype warning.
    """
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            print(f"reading {name} from {source.name}")
            return pd.read_csv(io.BytesIO(z.read(name)), sep=";", low_memory=False)
    return pd.read_csv(source, sep=";", low_memory=False)


def _require_columns(df: pd.DataFrame, needed: list[str]) -> None:
    """Fail before any work when the csv cannot answer the config.

    Args:
        df: the source frame.
        needed: the columns the build is about to read.
    Returns:
        Nothing.
    Raises:
        SystemExit: at least one is absent, listing them all at once.
    """
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV lacks the columns {missing}. The local-frame pipelines need the "
            f"AUGMENTED csv, which `make dataset` produces on its own from the "
            f"pipeline's prepare:/augment: block -- pass CSV=... only to override it.")


def _fill_optional(df: pd.DataFrame) -> pd.DataFrame:
    """Close the holes in the optional side columns and report them.

    Args:
        df: the cleaned frame.
    Returns:
        A frame whose OPTIONAL_COLUMNS carry '' instead of NaN -- they are
        neither inputs nor labels, so a hole is worth a note, not a lost frame.
    """
    holes = {c: int(df[c].isna().sum()) for c in OPTIONAL_COLUMNS if c in df.columns}
    for col, n_missing in holes.items():
        if n_missing:
            print(f"  note: {col} empty on {n_missing} kept rows")
    return df.assign(**{c: df[c].fillna("") for c in holes})


def clean(df: pd.DataFrame, inputs: list[str], labels: list[str] = LABELS) -> pd.DataFrame:
    """Reduce a source csv to the rows and the ordering the build can use.

    Args:
        df: the frame read from the csv.
        inputs: input columns the pipeline selected.
        labels: label columns the pipeline selected.
    Returns:
        A new frame holding only rows complete on those columns, with an int
        `run` column added and sorted by (run, time). Rows are dropped on the
        REQUIRED columns only -- a run whose geodesy failed (NaN position) is
        what silently disappears here, hence the printed count.
    Raises:
        SystemExit: a required column is missing from the csv.
    """
    n_total = len(df)
    needed = ["simulationindex", "time"] + list(inputs) + list(labels)
    _require_columns(df, needed)
    kept = df.dropna(subset=needed)
    print(f"{n_total} rows read, {n_total - len(kept)} dropped (NaN), {len(kept)} kept")
    return (_fill_optional(kept)
            .assign(run=lambda d: d["simulationindex"].astype(int))
            .sort_values(["run", "time"])
            .reset_index(drop=True))


def split_runs(df: pd.DataFrame, val_runs: set[int],
               train_runs: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by run id -- never by frame: consecutive frames are near-identical,
    so a random split would leak validation into training.

    Args:
        df: the cleaned frame.
        val_runs: runs held out for validation.
        train_runs: explicit training runs, every other run being discarded --
            what lets a phase-1 study train on 30 of the 85 available runs.
            Empty/None trains on every run that is not held out.
    Returns:
        (train, val), two views of `df`.
    Raises:
        SystemExit: a declared run is absent from the csv, or a run is listed on
            both sides.
    """
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
    """Fit the normalisation on the TRAIN split only, then let it travel with the
    data: both npz embed it, so validation is normalised with the training
    bounds and never with its own.

    Args:
        train: the training split.
        inputs: input columns, in npz order.
        physical: fixed-bounds selector -- True/'core' for position+attitude,
            'all' to add wind/velocities/rates (see data.normalization). Any
            channel without a fixed bound, and every label, uses the train stat.
        method: 'minmax' or 'zscore'.
        labels: label columns.
    Returns:
        The bounds dict saved into each npz: the column lists, the method, and
        x_min/x_max, y_min/y_max holding (min, max) for minmax and (mean, std)
        for zscore.
    """
    x_min, x_max = resolve_norm(train, inputs, method, physical)
    y_min, y_max = resolve_norm(train, labels, method, physical)
    return {"inputs": inputs, "labels": list(labels), "norm_method": method,
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def save_split(name: str, part: pd.DataFrame, bounds: dict, out_dir: Path) -> None:
    """Write one split to `landing_<name>.npz`.

    Args:
        name: 'train' or 'val', which names the file.
        part: that split's rows.
        bounds: the compute_bounds dict -- it also fixes the column order.
        out_dir: destination directory, expected to exist.
    Returns:
        Nothing; writes the npz (raw values, run/time columns, any optional
        side column, and the embedded bounds) and prints what it wrote.
    """
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


def dataset_columns(input_set: str, label_set: str,
                    extra_columns: list[str] = ()) -> tuple[list[str], list[str]]:
    """Resolve the named sets a config declares into actual column lists.

    Args:
        input_set: key of features.INPUT_SETS.
        label_set: key of features.LABEL_SETS.
        extra_columns: additional csv columns appended as inputs.
    Returns:
        (inputs, labels), in the canonical order the npz stores.
    Raises:
        SystemExit: either name is unknown -- a typo must not fall back to a
            default set and silently train the wrong recipe.
    """
    if input_set not in INPUT_SETS:
        raise SystemExit(f"unknown input_set {input_set!r}; choose from {sorted(INPUT_SETS)}")
    if label_set not in LABEL_SETS:
        raise SystemExit(f"unknown label_set {label_set!r}; choose from {sorted(LABEL_SETS)}")
    return INPUT_SETS[input_set] + list(extra_columns), LABEL_SETS[label_set]


def build(source: Path, val_runs: set[int], out_dir: Path,
          extra_columns: list[str] = (), input_set: str = "gps",
          physical_bounds=False, norm_method: str = "minmax",
          label_set: str = "commands", train_runs: set[int] | None = None) -> None:
    """Whole build: csv -> two npz plus their bounds file.

    Args:
        source: the csv (or zip) to read.
        val_runs: runs held out for validation.
        out_dir: destination directory, created if needed.
        extra_columns: extra csv columns appended as inputs.
        input_set: the base input columns (features.INPUT_SETS -- 'gps' keeps
            the GPS position as absolute lat/lon/alt, the local-frame sets
            convert that same position). ILS is in none of them.
        physical_bounds: normalise with the fixed airport-independent bounds
            where a channel has one; off by default (gps_cfc).
        norm_method: 'minmax' or 'zscore'.
        label_set: the command channels to predict (features.LABEL_SETS).
        train_runs: explicit training runs (default: everything not held out).
    Returns:
        Nothing; writes landing_train.npz, landing_val.npz and
        normalization_bounds.json into out_dir, and prints the recipe used.
    """
    inputs, labels = dataset_columns(input_set, label_set, extra_columns)
    df = clean(add_angle_encodings(load_csv(source), inputs), inputs, labels)
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

    Declaring both is rejected rather than half-executed: only the first would
    run, and build() would then fail on the columns the second was to produce.
    The day a delivery needs both, make this an ordered list of steps.

    Args:
        source: an explicit csv, which short-circuits everything below.
        config: the resolved pipeline config, read for its prepare:/augment:
            block.
        force: re-run the upstream step even when its csv already exists;
            otherwise an existing csv is reused as is.
    Returns:
        Path of the csv to build from.
    Raises:
        SystemExit: both steps are declared, or none is and no source was given
            (gps_cfc, which builds straight from the raw csv).
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
    """Run the config's prepare: step (imported late, so a pipeline without one
    never pulls pandas machinery it does not use).

    Args:
        config: the resolved pipeline config.
    Returns:
        Path of the canonical csv it wrote.
    """
    from boeing_landing.data.prepare import run_prepare
    return run_prepare(config, None, None, None)


def _run_augment(config: dict) -> Path:
    """Run the config's augment: step.

    Args:
        config: the resolved pipeline config.
    Returns:
        Path of the augmented csv it wrote.
    """
    from boeing_landing.data.augment import run_augment
    return run_augment(config)


def main() -> None:
    """CLI entrypoint: read the config's build: section and build the npz.

    Returns:
        Nothing; see build() for what lands on disk.
    """
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
