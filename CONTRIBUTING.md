# Contributing

## Where things live

| Piece | Responsibility |
|---|---|
| `boeing_landing/pipelines/<name>/` | one folder = one pipeline: `base.yaml` + training variants (`extends: base.yaml`) + the code ONLY that pipeline uses (e.g. `ils_aligned_cfc/augment_ned.py`) + a short README |
| `boeing_landing/data/build_dataset.py` | raw CSV -> npz (columns, split, normalisation) |
| `boeing_landing/data/features.py` | which columns are inputs/labels + conv channel orders |
| `boeing_landing/data/loader.py` | npz -> fixed-length portions -> training tensors |
| `boeing_landing/train.py` | generic engine: `assemble` (config -> model+data) + `fit_and_save` |
| `boeing_landing/evaluate.py`, `report.py` | post-training analysis of any run dir |
| `boeing_landing/experiments/` | sweeps built on top of `train` (channel orders, seeds) |
| `boeing_landing/config.py` | pipeline config loading (`extends` inheritance) — boeing-only, hence not in `utils/` |
| `utils/` | shared library — code carrying **no boeing-specific knowledge** (model factory, Lightning wrapper, schedules). `quadrotor_baseline/` is frozen, so "both sides import it" cannot be the test for new code; the test is whether the module would need rewriting for another aircraft. |
| `quadrotor_baseline/` | reference implementation; do not modify |
| `Yolo_models_LARD_V2/` | third-party submodule; do not modify |

## Adding a new pipeline (same data)

Different model, channel order, window length… no code needed:

1. create `boeing_landing/pipelines/<name>/` and copy an existing `base.yaml`;
2. edit the knobs (`model.type`, `dataset.*`, `checkpoint_name`);
3. `make train CONFIG=<name>`.

Training, evaluation (`make evaluate`), plots (`make plots`) and experiments
are config-driven and work on any run directory unchanged.

**Where pipeline code goes.** To read one pipeline, open its folder: it holds
its configs AND its specific code (e.g. the image pipeline's YOLO/embedding
step), with a short README pointing at the engine pieces it reuses. The rule
mirrors the utils/ one, one level down:

- used by ONE pipeline  -> `pipelines/<name>/*.py`
- used by ≥ 2 pipelines -> the engine (`data/`, `train.py`, …); promote it
  the day a second pipeline needs it — never copy it
- knows nothing about the boeing data -> `utils/` (`utils/scheduler.py` is the
  example: it wraps `utils/lightning.py`, so it belongs beside it; `config.py`
  is the counter-example — it knows the pipeline layout, so it stays in
  `boeing_landing/`)

## Adding a training variant of an existing pipeline

One yaml next to the pipeline's `base.yaml`, holding ONLY the overrides:

```yaml
# pipelines/gps_cfc/quick.yaml
extends: base.yaml
training:
  max_epochs: 3
```

`make train CONFIG=gps_cfc/quick` — the variant name tags the runs
(`runs/gps_cfc/quick_<order>/`). Never copy a full config: variants must
stay diffs against their base.

## Adding new inputs or outputs

1. extend `data/features.py` (input/label lists, channel orders);
2. adapt `data/build_dataset.py` if the CSV columns change;
3. rebuild with `make dataset`, then create the pipeline config.

Fixed normalisation bounds are data, not code: they live in
`data/physical_bounds.yaml`, never in a Python dict.

## Adding a new data delivery

A delivery arrives with its own column names and its own side tables; the build
step expects one canonical `;`-separated csv. Write that translation as a module
exposing

```python
prepare(sims: DataFrame, side: DataFrame) -> (DataFrame, list_of_warnings)
```

and declare it in the pipeline config:

```yaml
prepare:
  module: boeing_landing.pipelines.<pipeline>.prepare
  sims_csv: datasets/<delivery>.csv
  side_csv: datasets/<side table>.csv
  out_csv: datasets/<canonical>.csv
```

`make dataset` runs it when `out_csv` is missing, so a delivery still takes one
command (`FORCE=1` re-runs it). Rename only the columns the repo already knows
under another name, drop what no input set uses, and never recompute a value —
a preparation step that transforms data is a source of silent divergence between
what was delivered and what was trained on. `augment:` is the same mechanism for
a step that must *derive* columns (the local-frame geodesy).

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
  directory; saved plots go to `figures/<pipeline>/`, never into the runs.
