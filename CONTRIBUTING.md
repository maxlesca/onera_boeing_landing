# Contributing

## Where things live

| Piece | Responsibility |
|---|---|
| `boeing_landing/configs/*.yaml` | everything tunable — one file = one pipeline |
| `boeing_landing/data/build_dataset.py` | raw CSV -> npz (columns, split, normalisation) |
| `boeing_landing/data/features.py` | which columns are inputs/labels + conv channel orders |
| `boeing_landing/data/loader.py` | npz -> fixed-length portions -> training tensors |
| `boeing_landing/train.py` | generic engine: `assemble` (config -> model+data) + `fit_and_save` |
| `boeing_landing/evaluate.py`, `report.py` | post-training analysis of any run dir |
| `boeing_landing/experiments/` | sweeps built on top of `train` (channel orders, seeds) |
| `utils/` | shared library — put code here when two or more folders need it |
| `quadrotor_baseline/` | reference implementation; do not modify |
| `Yolo_models_LARD_V2/` | third-party submodule; do not modify |

## Adding a new pipeline (same data)

Different model, channel order, window length… no code needed:

1. copy `boeing_landing/configs/gps_cfc.yaml` to `configs/<name>.yaml`;
2. edit the knobs (`model.type`, `dataset.*`, `checkpoint_name`);
3. `make train CONFIG=<name>`.

Training, evaluation (`make evaluate`), plots (`make plots`) and experiments
are config-driven and work on any run directory unchanged.

## Adding new inputs or outputs

1. extend `data/features.py` (input/label lists, channel orders);
2. adapt `data/build_dataset.py` if the CSV columns change;
3. rebuild with `make dataset`, then create the pipeline config.

## Adding a new model family

Add it to `utils/model_builder.py` — the single factory used by every
entrypoint — and select it with `model.type` in a config.

## Adding a new experiment

A small script in `boeing_landing/experiments/` that imports
`boeing_landing.train.train` (or `train_config` for in-memory config edits),
reads its settings from the config's `experiments:` section, and gets a
Makefile target.

## Conventions

- Commit messages: `fix:` / `feat:` / `refactor:` / `chore:` prefixes.
- Comments and docstrings in English, short and natural.
- Small single-purpose functions; split rather than grow.
- No code duplication: shared logic moves to `utils/`.
- Parameters belong in configs, never hardcoded in `.py`.
- Every training run writes to its own `runs/<pipeline>/<variant>/<timestamp>/`
  directory; saved plots go to the flat `figures/` folder, never into the runs.
