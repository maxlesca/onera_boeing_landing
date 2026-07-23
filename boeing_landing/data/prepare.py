# -*- coding: utf-8 -*-
"""Run the preparation a pipeline config declares, on a raw delivery.

A delivery arrives with its own column names and its own side tables; the build
step expects one canonical ';'-separated csv. Each pipeline names, in its
`prepare:` config block, the module that performs that translation and the file
it writes. Every preparation exposes the same contract

    prepare(sims: DataFrame, side: DataFrame) -> (DataFrame, list_of_warnings)

so this dispatches by config. It is not a target of its own: `make dataset` runs
it when the canonical csv is missing, so one command builds the whole chain.
Mirrors data/augment.py, which does the same for the geodesy step.

Pipelines fed a csv that is already canonical simply have no `prepare:` section.

    make dataset CONFIG=ned_wind_cfc      # runs this step when needed
    python -m boeing_landing.data.prepare --config path.yaml   # or on its own
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd


def load_source(path: Path) -> pd.DataFrame:
    """Read a delivery csv, accepting either separator: deliveries have arrived
    both ',' and ';' separated, and guessing wrong yields a single-column frame.
    The separator is sniffed on the header alone (the python engine cannot take
    low_memory, and these files are large enough for that to matter)."""
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if df.shape[1] < 2:
        raise SystemExit(f"{path} parsed into {df.shape[1]} column(s) -- wrong separator?")
    return df


def run_prepare(config: dict, sims: Path | None, side: Path | None,
                output: Path | None) -> Path:
    """Dispatch to the pipeline's preparation module and write its output csv."""
    cfg = config.get("prepare")
    if not cfg:
        raise SystemExit("this config has no `prepare:` section -- its pipeline reads "
                         "an already canonical csv (nothing to prepare)")
    sims = sims or Path(cfg["sims_csv"])
    side = side or Path(cfg["side_csv"])
    out = output or Path(cfg["out_csv"])
    if out.resolve() in (sims.resolve(), side.resolve()):
        raise SystemExit("refusing to overwrite an input file")

    module = importlib.import_module(cfg["module"])
    prepared, warnings = module.prepare(load_source(sims), load_source(side))
    out.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out, sep=";", index=False)

    runs = prepared["simulationindex"].nunique()
    print(f"{len(prepared)} rows, {runs} runs, {prepared.shape[1]} columns "
          f"({cfg['module']}) -> {out}")
    for warning in warnings:
        print(f"  WARNING: {warning} is flown but absent from the airport table")
    return out


def main() -> None:
    from boeing_landing.config import load_config
    from boeing_landing.train import DEFAULT_CONFIG

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the prepare: section")
    ap.add_argument("--sims", type=Path, default=None,
                    help="simulation csv (default: the config's prepare.sims_csv)")
    ap.add_argument("--side", type=Path, default=None,
                    help="side table, e.g. the airport info (default: prepare.side_csv)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output csv (default: the config's prepare.out_csv)")
    a = ap.parse_args()
    run_prepare(load_config(a.config), a.sims, a.side, a.output)


if __name__ == "__main__":
    main()
