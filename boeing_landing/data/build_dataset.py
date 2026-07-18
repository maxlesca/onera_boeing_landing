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
    python -m boeing_landing.data.build_dataset SOURCE [--config path.yaml]
    SOURCE = the dataset .zip or the extracted ';'-separated .csv.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from boeing_landing.data.features import CANONICAL_INPUTS, LABELS


def load_csv(source: Path) -> pd.DataFrame:
    """Read the dataset CSV, from a .zip or a plain .csv (';'-separated)."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            print(f"reading {name} from {source.name}")
            return pd.read_csv(io.BytesIO(z.read(name)), sep=";")
    return pd.read_csv(source, sep=";")


def clean(df: pd.DataFrame, extra: list[str] = ()) -> pd.DataFrame:
    """Drop rows with missing fields, add an int `run`, sort by (run, time)."""
    n_total = len(df)
    needed = ["simulationindex", "time", "image_filename"] + CANONICAL_INPUTS + LABELS + list(extra)
    df = df.dropna(subset=needed).copy()
    print(f"{n_total} rows read, {n_total - len(df)} dropped (NaN), {len(df)} kept")
    df["run"] = df["simulationindex"].astype(int)
    return df.sort_values(["run", "time"]).reset_index(drop=True)


def split_runs(df: pd.DataFrame, val_runs: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, val) by run id."""
    present = set(df["run"].unique())
    if not val_runs & present:
        raise SystemExit(f"none of the val runs {val_runs} exist (runs: {sorted(present)})")
    val_mask = df["run"].isin(val_runs)
    return df[~val_mask], df[val_mask]


def compute_bounds(train: pd.DataFrame, inputs: list[str]) -> dict:
    """Min/max normalisation bounds from the train split only."""
    return {
        "inputs": inputs,
        "labels": LABELS,
        "x_min": train[inputs].min().tolist(),
        "x_max": train[inputs].max().tolist(),
        "y_min": train[LABELS].min().tolist(),
        "y_max": train[LABELS].max().tolist(),
    }


def save_split(name: str, part: pd.DataFrame, bounds: dict, out_dir: Path) -> None:
    """Write one split to `landing_<name>.npz` (raw values + embedded bounds)."""
    inputs = bounds["inputs"]
    np.savez_compressed(
        out_dir / f"landing_{name}.npz",
        X=part[inputs].to_numpy(np.float32),
        Y=part[LABELS].to_numpy(np.float32),
        run=part["run"].to_numpy(np.int32),
        t=part["time"].to_numpy(np.float32),
        image=part["image_filename"].to_numpy(),
        input_names=np.array(inputs),
        label_names=np.array(LABELS),
        x_min=np.array(bounds["x_min"], np.float32),
        x_max=np.array(bounds["x_max"], np.float32),
        y_min=np.array(bounds["y_min"], np.float32),
        y_max=np.array(bounds["y_max"], np.float32),
    )
    print(f"  {name}: {len(part):6d} frames, runs {sorted(part['run'].unique())} "
          f"-> {out_dir / f'landing_{name}.npz'}")


def build(source: Path, val_runs: set[int], out_dir: Path,
          extra_columns: list[str] = ()) -> None:
    """extra_columns: additional CSV columns appended as inputs."""
    df = clean(load_csv(source), list(extra_columns))
    inputs = CANONICAL_INPUTS + list(extra_columns)
    train, val = split_runs(df, val_runs)
    bounds = compute_bounds(train, inputs)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_split("train", train, bounds, out_dir)
    save_split("val", val, bounds, out_dir)
    (out_dir / "normalization_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print(f"inputs={len(inputs)} (incl. GPS, no ILS), labels={len(LABELS)}")


def main() -> None:
    from boeing_landing.train import DEFAULT_CONFIG
    from utils.config import load_yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="dataset .zip or extracted .csv")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the build: section")
    a = ap.parse_args()

    build_cfg = load_yaml(a.config).get("build", {})
    val_runs = {int(r) for r in build_cfg.get("val_runs", [8])}
    build(a.source, val_runs, Path(build_cfg.get("out_dir", "datasets/gps_no_ils")),
          extra_columns=build_cfg.get("extra_columns") or [])


if __name__ == "__main__":
    main()
