# -*- coding: utf-8 -*-
"""Run the augmentation a pipeline config declares, on the raw CSV.

Each frame pipeline names, in its `augment:` config block, the augmentation
module to run and the output csv. Every augmentation exposes the same contract
(`augment(df, navdb) -> (df, missing)` and a `POS_COLUMNS` list), so this
dispatches by config. It is not a target of its own: `make dataset` runs it when
the augmented csv is missing, so one command builds the whole chain.

Pipelines that need no augmentation (e.g. gps_cfc uses the raw GPS directly)
simply have no `augment:` section.

    python -m boeing_landing.data.augment SOURCE NAVDB --config path.yaml
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

from boeing_landing.data.geodesy import load_navdb


def run_augment(config: dict, source: Path | None = None, navdb: Path | None = None,
                output: Path | None = None) -> Path:
    """Dispatch to the pipeline's augmentation module and write its output csv.

    Args:
        config: the resolved pipeline config, read for its `augment:` block
            (module, raw_csv, navdb, out_csv).
        source: raw csv to read; defaults to the block's raw_csv, so the build
            step can call this with the config alone.
        navdb: nav database json; defaults to the block's navdb.
        output: csv to write; defaults to the block's out_csv.
    Returns:
        Path of the written csv. Runs missing from the database keep NaN and
        are reported, never dropped.
    Raises:
        SystemExit: the config declares no augmentation, or the output would
            overwrite one of the inputs.
    """
    cfg = config.get("augment")
    if not cfg:
        raise SystemExit("this config has no `augment:` section -- its pipeline "
                         "uses the raw csv directly (e.g. gps_cfc), nothing to augment")
    source = source or Path(cfg["raw_csv"])
    navdb = navdb or Path(cfg["navdb"])
    out = output or Path(cfg["out_csv"])
    if out.resolve() in (source.resolve(), navdb.resolve()):
        raise SystemExit("refusing to overwrite an input file")

    module = importlib.import_module(cfg["module"])
    df = pd.read_csv(source, sep=";", low_memory=False)
    augmented, missing = module.augment(df, load_navdb(navdb))
    out.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(out, sep=";", index=False)

    done = augmented[module.POS_COLUMNS[0]].notna()
    print(f"{done.sum()}/{len(augmented)} rows augmented ({cfg['module']}) -> {out}")
    for airport, runway, n in missing:
        print(f"  WARNING: {airport} {runway} not in the nav database ({n} rows left NaN)")
    return out


def main() -> None:
    """CLI entrypoint: run the augmentation on its own.

    Returns:
        Nothing; see run_augment for what is written.
    """
    from boeing_landing.config import load_config
    from boeing_landing.train import DEFAULT_CONFIG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, nargs="?", default=None,
                    help="raw ldg_*.csv (read-only; default: the config's augment.raw_csv)")
    ap.add_argument("navdb", type=Path, nargs="?", default=None,
                    help="NavDB json (read-only; default: the config's augment.navdb)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the augment: section")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output csv (default: the config's augment.out_csv)")
    a = ap.parse_args()
    run_augment(load_config(a.config), a.source, a.navdb, a.output)


if __name__ == "__main__":
    main()
