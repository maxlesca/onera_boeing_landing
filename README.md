# Boeing 747 landing — behavioural cloning

Train a neural-network flight controller for Boeing 747 landing by cloning an
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
    ned_wind_cfc/             85 runs / 8 airports, wind as an input: prepare.py, encoder arms
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

### GPU or CPU

`torch` is the CPU build by default, so training runs anywhere. On a GPU machine,
install a `torch` build matching your **driver's** CUDA version — e.g. a driver
capped at CUDA 12.2 needs the cu121 wheel:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu121
```

A build newer than the driver crashes at the first CUDA call (`NVIDIA driver ...
is too old`). To train on CPU regardless — broken/mismatched GPU, or simply to
avoid it — hide the device from the process:

```bash
CUDA_VISIBLE_DEVICES="" make train CONFIG=ils_aligned_cfc
```

Setting `accelerator: cpu` in the yaml is **not** enough on its own: Lightning
still snapshots the CUDA RNG while the GPU is visible, which re-triggers the
driver error. Hiding the device (above) is the robust switch.

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
make dataset CONFIG=<pipeline> [CSV=path/to.csv] [FORCE=1]
    # raw delivery -> csv -> npz, in one command. CONFIG: pipeline whose build:
    # section decides input set / label set / train and val runs / out dir. The
    # csv the build needs is produced on the way when it is missing, from the
    # pipeline's own upstream block -- `prepare:` renames a raw delivery
    # (ned_wind_cfc), `augment:` adds the local-frame coordinates
    # (ils_aligned_cfc, magnetic_north_cfc). Sources are read-only; runways
    # missing from the NavDB keep NaN. CSV: override the source (required for
    # gps_cfc, which declares no upstream step). FORCE=1 re-runs that step even
    # if its csv already exists.

make trajectories [NED_CSV=datasets/ldg_dataset_images_ned.csv] [SAVE=1]
    # plot the approaches of an augmented csv: top view, vertical profile, and
    # pos_cross vs the sim localizer (expected y = x). SAVE=1 -> figures/dataset/.

make data-report CONFIG=<pipeline> [SAVE=1]
    # per-channel diagnostics of a built npz: raw range/std, spread in [0,1]
    # (weak/near-constant channels), and % of val frames outside the train
    # bounds (distribution shift). SAVE=1 -> figures/dataset/.

make run-report CONFIG=<pipeline> [SAVE=1]
    # per-run distribution: ranks which run is an outlier and on which channel
    # (fingerprint heatmap + extremeness bar + wind scatter). Helps pick a
    # "far" validation run. SAVE=1 -> figures/dataset/.

make train CONFIG=<pipeline> [ORDER=...] [EPOCHS=n]
    # CONFIG: pipeline name (gps_cfc -> its base.yaml), pipeline/variant
    #         (gps_cfc/quick, gps_cfc/long) or path to a yaml.
    # ORDER: optional -- overrides the conv channel order set by the yaml's
    #        dataset.input_order. Only for the ordering study: grouped, gps_last,
    #        pos_vel, by_axis, reversed, random_1..3 (see data/features.py --
    #        an unknown name is rejected, never silently replaced by the default).
    #        Each pipeline defaults to its own order, so it is not needed normally.
    # EPOCHS: override training.max_epochs for a quick trial (e.g. EPOCHS=3).

make evaluate RUN=runs/<pipeline>/<variant>/<timestamp>
    # metrics + feature-group ablations; writes evaluation.json into the run dir.
    # With several validation runs, also prints one line per held-out run: a
    # single averaged score cannot tell "does not extrapolate" from "has not
    # converged", per-run scores can.

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
make loro CONFIG=<pipeline> [PB=all] [NORM=zscore] [TAG=..] [SAVE=1]
    # leave-one-run-out CV: hold out every run in turn, train K models, report
    # the recipe's mean generalisation (val_loss + scale-invariant mean R2).
    # PB/NORM override build.physical_bounds / norm_method to test a lever without
    # a new yaml. Writes runs/loro/<tag>/results.json (always) + a PNG (SAVE=1).
make loro-plot RESULTS="runs/loro/a/results.json [runs/loro/b/results.json]" [SAVE=1]
    # re-render figure(s) from saved results.json, NO training: one file -> that
    # arm's val_loss+R2 panels; several -> the side-by-side comparison of arms.

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
| `dataset.with_noise` / `noise_std` | gaussian noise (sigma `noise_std`) on the normalised inputs of the TRAINING split only — behavioural cloning never sees off-trajectory states; validation stays clean so the score does not move with the seed |
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
