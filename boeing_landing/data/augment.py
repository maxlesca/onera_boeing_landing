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
from pathlib import Path

import pandas as pd

from boeing_landing.data.geodesy import load_navdb
from boeing_landing.data.step import (load_module, report_warnings,
                                      resolve_path, step_config, write_csv)


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
    cfg = step_config(config, "augment", "its pipeline uses the raw csv "
                      "directly (e.g. gps_cfc), nothing to augment")
    source = resolve_path(cfg, "raw_csv", source)
    navdb = resolve_path(cfg, "navdb", navdb)
    out = resolve_path(cfg, "out_csv", output)

    module = load_module(cfg)
    df = pd.read_csv(source, sep=";", low_memory=False)
    augmented, missing = module.augment(df, load_navdb(navdb))
    write_csv(augmented, out, [source, navdb])

    done = augmented[module.POS_COLUMNS[0]].notna()
    print(f"{done.sum()}/{len(augmented)} rows augmented ({cfg['module']}) -> {out}")
    report_warnings(missing)
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
