# Boeing 737 landing — behavioural cloning

Train a neural-network flight controller for Boeing 737 landing by cloning an
expert autopilot from simulation data. The controller maps inertial state,
GPS and wind to control commands through a 1D convolution and a closed-form
continuous-time recurrent network (CfC). An image branch (CNN pretrained on
LARD) is a planned extension.

## Layout

```
Makefile                  command shortcuts
requirements.txt          dependencies (venv or conda)
environment.yml           conda env (reads requirements.txt)
boeing_landing/           project code
  data/build_dataset.py     CSV -> npz (GPS in, ILS out, per-run split)
  data/features.py          input/label lists + channel orders for the conv
  data/loader.py            npz -> fixed-length portions -> training tensors
  config.py                 pipeline config loading (extends inheritance)
  pipelines/                one folder per pipeline: configs + pipeline-specific code
    gps_cfc/                  base.yaml + quick/long variants (extends), README
    ils_aligned_cfc/          GPS -> runway/ILS-aligned NED: augment_ned.py, plot, base.yaml
    magnetic_north_cfc/       GPS -> magnetic-north NED: augment_magnetic.py, base.yaml
  experiments/feature_order.py   sweep conv channel orders
  experiments/convergence.py     seed-stability study
  train.py                  assemble + fit_and_save (generic, config-driven)
  evaluate.py               metrics + feature-group ablations of a run
  report.py                 training curves for one run, or several compared
utils/                    shared library (imported across the repo)
quadrotor_baseline/       reference quadrotor controller and pretrained models
dataset_preparation/      dataset tools (image crop, rosbag extract)
Yolo_models_LARD_V2/      submodule (YOLO models)
datasets/                 built npz files (gitignored, created by make dataset)
runs/                     training outputs (gitignored), one subfolder per pipeline
figures/                  saved plots (gitignored), one subfolder per pipeline (make plots SAVE=1)
```

## Install

Prerequisites: Python 3.12, GNU make, and **git-lfs** (the submodule stores its
model weights with LFS — run `git lfs install` once *before* cloning, otherwise
`Yolo_models_LARD_V2/` contains 132-byte pointers instead of models; recover
with `git lfs pull` inside the submodule).

```bash
git clone --recurse-submodules <repo-url>
cd onera_boeing_landing

make install     # venv + dependencies
# or conda:
conda env create -f environment.yml && conda activate boeing_landing
```

`make` auto-detects the interpreter (Linux venv, Windows venv — including from
WSL — or the active conda env); no variable to pass.

`torch` is the CPU build by default; swap for a CUDA build on a GPU machine.

## Data

The dataset is a semicolon-separated CSV of 25 Hz landing frames (GPS as
lat/lon in radians and altitude, attitude, body/NED velocities, commands, ILS,
runway id). It is not shipped with the repo. Point the build step at it:

```bash
make dataset CSV=path/to/dataset.csv
```

This writes `landing_{train,val}.npz` into the config's `build.out_dir`
(18 inputs = inertial + GPS + wind, ILS excluded), split per run with
normalisation bounds computed on the train split only.

The `build:` section can extend the inputs: `extra_columns: [...]` appends
other CSV columns as-is. Each pipeline config owns its dataset directory
(`build.out_dir`).

## Usage

`make` alone lists the targets. Each target below is shown with every option it
accepts (`OPT=value`); options in brackets are optional and show their default.

```bash
make dataset CONFIG=<pipeline> [CSV=path/to.csv]
    # build the npz. CONFIG: pipeline whose build: section decides input set /
    # val runs / out dir (gps_cfc, ils_aligned_cfc, magnetic_north_cfc). CSV:
    # source file; optional for the frame pipelines (falls back to their
    # augment.out_csv), required for gps_cfc.

make augment CONFIG=<pipeline> [RAW_CSV=datasets/ldg_dataset_images.csv] [NAVDB=datasets/NavDB_MFS.json]
    # augment RAW_CSV the way CONFIG's augment: block says (which augmentation
    # module, which out_csv). CONFIG = ils_aligned_cfc or magnetic_north_cfc; it
    # is required (gps_cfc has no augment: block). Sources are read-only; runways
    # missing from the NavDB keep NaN.

make trajectories [NED_CSV=datasets/ldg_dataset_images_ned.csv] [SAVE=1]
    # plot the approaches of an augmented csv: top view, vertical profile, and
    # pos_cross vs the sim localizer (expected y = x). SAVE=1 -> figures/dataset/.

make train [CONFIG=gps_cfc] [ORDER=grouped] [EPOCHS=n]
    # CONFIG: pipeline name (gps_cfc -> its base.yaml), pipeline/variant
    #         (gps_cfc/quick, gps_cfc/long) or path to a yaml.
    # ORDER: conv channel order — grouped, gps_first, gps_last, pos_vel, by_axis,
    #        reversed, random_1..3 (see data/features.py).
    # EPOCHS: override training.max_epochs for a quick trial (e.g. EPOCHS=3).

make evaluate RUN=runs/<pipeline>/<variant>/<timestamp>
    # metrics + feature-group ablations; writes evaluation.json into the run dir.

make plots RUNS="runs/.../<ts> [more...]" [SAVE=1] [BARS=1] [NOISE=0.005]
    # one run: curves + per-command MSE + ablation panels; several: comparison.
    # SAVE=1: write the PNG into figures/<pipeline>/ (named after the runs) instead of a window.
    # BARS=1: several runs as best-val_loss bars instead of overlaid curves.
    # NOISE: seed-noise threshold line on the bar charts (0 = none).

make plots-orders [CONFIG=gps_cfc] [STAMP=prefix] [SAVE=1] [NOISE=0.005]
    # bars of the channel-order sweep of CONFIG (latest run per order).
    # STAMP: timestamp prefix selecting one sweep session (e.g. STAMP=20260716).

make experiment-order [CONFIG=gps_cfc]        # train one run per channel order, rank by val_loss
make experiment-convergence [CONFIG=gps_cfc]  # same config under experiments.seeds (stability)

make quadrotor-train     # train the quadrotor baseline (its own train_config.yaml)
make quadrotor-test [PLOT=1]   # evaluate it (model picked in its test_config.yaml)
make clean               # remove runs/, logs, caches (figures/ is kept)
```

The quadrotor baseline is a reference implementation and keeps its own
config/checkpoint layout: the boeing-side options (`CONFIG=`, `RUN=`,
`RUNS=`) do not apply to it.

Every target also accepts `PYTHON=...` to override the auto-detected
interpreter (e.g. `make train PYTHON=python`).

## Batch / cluster usage (HPC)

Every make target wraps a plain `python -m` command, so jobs don't need make:

| make | raw command |
|---|---|
| `make dataset CSV=...` | `python -m boeing_landing.data.build_dataset <csv> --config boeing_landing/pipelines/gps_cfc/base.yaml` |
| `make train` | `python -m boeing_landing.train --config ... [--input-order X] [--max-epochs N]` |
| `make evaluate RUN=...` | `python -m boeing_landing.evaluate --run <run_dir>` |
| `make experiment-*` | `python -m boeing_landing.experiments.<name> --config ...` |

SLURM example:

```bash
#!/bin/bash
#SBATCH --job-name=gps_cfc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
conda activate boeing_landing
cd "$SLURM_SUBMIT_DIR"
python -m boeing_landing.train --config boeing_landing/pipelines/gps_cfc/base.yaml
```

Notes for GPU nodes: install the CUDA torch build, then set `accelerator: gpu`,
`precision: 16-mixed` and raise `dataloader.num_workers` in the config. Compute
nodes are headless: use `make plots ... SAVE=1` / `--save` (never bare `--plot`).
Runs land in `runs/<pipeline>/<variant>/<timestamp>/` — safe on a shared
filesystem, no two jobs write to the same folder.

## Config

Model hyperparameters live in
[boeing_landing/pipelines/gps_cfc/base.yaml](boeing_landing/pipelines/gps_cfc/base.yaml).
The config is the single control panel: model, data, build, evaluation, and
experiment settings all live there. Each pipeline is a folder under
`boeing_landing/pipelines/`; training variants are extra yamls in the same
folder that declare `extends: base.yaml` and override only the knobs they
change (no duplication). A variant's runs are tagged with its name:
`runs/<pipeline>/<variant>_<order>/`.
Data knobs:

| Key | Meaning |
|---|---|
| `dataset.portion_len` | portion length in frames (125 = 5 s at 25 Hz) |
| `dataset.stride` | step between portions (overlap) |
| `dataset.input_order` | conv channel order: `grouped`, `gps_last`, `pos_vel`, `by_axis`, `reversed`, `random_1..3` (dataset-only channels, e.g. extra_columns, are appended at the end) |
| `dataset.use_dt` | append the per-frame time step as CfC timespans (baseline recipe) |
| `sequencing.seq_len` | 1 — the conv sees one frame at a time, over the feature axis |

Changing `portion_len` / `stride` / `input_order` needs no npz rebuild.

## Training options

The `training:` block controls the run. Defaults reproduce the baseline; enable
the rest when needed.

| Key | Meaning |
|---|---|
| `accelerator` | `auto` / `cpu` / `gpu` — where to run (auto picks GPU if usable) |
| `devices` | number of accelerators (1, or more for multi-GPU) |
| `precision` | `32` or `16-mixed` (half precision, faster/lighter — GPU only) |
| `gradient_clip_val` | cap gradient magnitude to avoid blow-ups (RNN stability); 0 = off |
| `early_stopping_patience` | stop after N epochs without val_loss improvement; 0 = off |

**CPU vs CUDA torch.** `accelerator: gpu` only works with a CUDA build of torch.
The default install is CPU-only, so `torch.cuda.is_available()` is False and any
GPU is ignored. Install the matching CUDA build (pytorch.org) first; the param
is already in place, so no code change is needed on a GPU machine.

## Outputs

One folder per run, grouped by pipeline (the yaml), never shared or overwritten.
`<variant>` is the channel order, prefixed by the seed for convergence runs
(e.g. `grouped`, `gps_last`, `seed43_grouped`):

```
runs/<pipeline>/<variant>/<timestamp>/
  epoch=NN_val_loss=0.xxxxxx.ckpt   best checkpoint
  config.yaml                        exact resolved config
  summary.json                       best epoch/val_loss, wall time, n parameters
  evaluation.json                    regression metrics + ablations (make evaluate)
  lightning_logs/                    per-step metrics: losses, grad_norm, epoch time
```

Saved plots never go into the run folders: `make plots ... SAVE=1` writes them
to `figures/<pipeline>/`, named `<variant>_<timestamp>.png` (single run) or
`<pipelines>_<comparison|bars>_<date>.png`; plots mixing several pipelines go
to `figures/comparisons/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): where things live, how to add a
pipeline / inputs / model / experiment, and the project conventions.

## Method

- Runs are cut into fixed-length portions; the CfC carries state within a
  portion and resets between portions, which are shuffled.
- Split is per run, not per frame: neighbouring frames are near-identical, so a
  random split would leak validation data into training.
- The training engine (`train.assemble` + `train.fit_and_save`) is generic and
  config-driven, so new pipelines are added as configs, not new training code.
