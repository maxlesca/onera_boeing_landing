# -*- coding: utf-8 -*-
"""Run the augmentation a pipeline config declares, on the raw CSV.

Each frame pipeline names, in its `augment:` config block, the augmentation
module to run and the output csv. Every augmentation exposes the same contract
(`augment(df, navdb) -> (df, missing)` and a `POS_COLUMNS` list), so this
dispatches by config -- `make augment CONFIG=<pipeline>` augments the raw data
the way that pipeline expects, instead of one make target per frame.

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


def main() -> None:
    from boeing_landing.config import load_config
    from boeing_landing.train import DEFAULT_CONFIG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="raw ldg_*.csv (read-only)")
    ap.add_argument("navdb", type=Path, help="NavDB json (read-only)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="pipeline config holding the augment: section")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output csv (default: the config's augment.out_csv)")
    a = ap.parse_args()

    cfg = load_config(a.config).get("augment")
    if not cfg:
        raise SystemExit(f"{a.config} has no `augment:` section -- this pipeline "
                         "uses the raw csv directly (e.g. gps_cfc), nothing to augment")
    out = a.output or Path(cfg["out_csv"])
    if out.resolve() in (a.source.resolve(), a.navdb.resolve()):
        raise SystemExit("refusing to overwrite an input file")

    module = importlib.import_module(cfg["module"])
    df = pd.read_csv(a.source, sep=";", low_memory=False)
    augmented, missing = module.augment(df, load_navdb(a.navdb))
    augmented.to_csv(out, sep=";", index=False)

    done = augmented[module.POS_COLUMNS[0]].notna()
    print(f"{done.sum()}/{len(augmented)} rows augmented ({cfg['module']}) -> {out}")
    for airport, runway, n in missing:
        print(f"  WARNING: {airport} {runway} not in the nav database ({n} rows left NaN)")


if __name__ == "__main__":
    main()
