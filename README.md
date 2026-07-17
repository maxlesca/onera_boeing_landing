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
  configs/gps_cfc.yaml      one pipeline = one config (single control panel)
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
runs/                     training outputs (gitignored)
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

## Usage

```bash
make                     # list targets
make dataset CSV=...     # build the npz
make train               # train the default pipeline (gps_cfc)
make train ORDER=gps_last            # pick a conv channel order
make train CONFIG=gps_cfc            # pick a pipeline config
make evaluate RUN=runs/<n>/<ts>      # metrics + feature-group ablations
make plots RUNS="runs/<n>/<ts>"      # training curves (several RUNS = comparison)
make plots RUNS="..." SAVE=1         # save PNG instead of showing
make experiment-order          # sweep channel orders, rank by val_loss
make experiment-convergence    # same config, several seeds (stability)
make quadrotor-train     # train the quadrotor baseline
make clean               # remove runs/, logs, caches
```

Override the interpreter elsewhere: `make train PYTHON=python`.

## Batch / cluster usage (HPC)

Every make target wraps a plain `python -m` command, so jobs don't need make:

| make | raw command |
|---|---|
| `make dataset CSV=...` | `python -m boeing_landing.data.build_dataset <csv> --config boeing_landing/configs/gps_cfc.yaml` |
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
python -m boeing_landing.train --config boeing_landing/configs/gps_cfc.yaml
```

Notes for GPU nodes: install the CUDA torch build, then set `accelerator: gpu`,
`precision: 16-mixed` and raise `dataloader.num_workers` in the config. Compute
nodes are headless: use `make plots ... SAVE=1` / `--save` (never bare `--plot`).
Runs land in `runs/<name>/<timestamp>/` — safe on a shared filesystem, no two
jobs write to the same folder.

## Config

Model hyperparameters live in
[boeing_landing/configs/gps_cfc.yaml](boeing_landing/configs/gps_cfc.yaml).
The config is the single control panel: model, data, build, evaluation, and
experiment settings all live there.
Data knobs:

| Key | Meaning |
|---|---|
| `dataset.portion_len` | portion length in frames (125 = 5 s at 25 Hz) |
| `dataset.stride` | step between portions (overlap) |
| `dataset.input_order` | conv channel order: `grouped`, `gps_last`, `pos_vel`, `by_axis`, `reversed`, `random_1..3` |
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

One folder per run, never shared or overwritten:

```
runs/<pipeline>_<order>/<timestamp>/
  epoch=NN_val_loss=0.xxxxxx.ckpt   best checkpoint
  config.yaml                        exact resolved config
  summary.json                       best epoch/val_loss, wall time, n parameters
  evaluation.json                    regression metrics + ablations (make evaluate)
  lightning_logs/                    per-step metrics: losses, grad_norm, epoch time
```

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
