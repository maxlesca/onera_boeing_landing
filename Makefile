# Boeing landing -- command shortcuts. Run from the repo root.
# The interpreter is auto-detected: Linux venv, then Windows venv (also works
# from WSL via interop), then the active environment's `python` (e.g. conda).
# Still overridable:  make train PYTHON=/path/to/python
PYTHON  ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,python))
CSV     ?=
CONFIG  ?= gps_cfc
# ORDER overrides the conv channel order for the ordering study; empty by default
# so each pipeline uses its own dataset.input_order from the yaml.
ORDER   ?=
# CONFIG accepts: a pipeline name (gps_cfc -> pipelines/gps_cfc/base.yaml),
# a pipeline/variant pair (gps_cfc/quick -> pipelines/gps_cfc/quick.yaml),
# or a full path to a yaml (used verbatim).
CFGPATH  = $(if $(findstring .yaml,$(CONFIG)),$(if $(findstring /,$(CONFIG)),$(CONFIG),boeing_landing/pipelines/$(basename $(CONFIG))/base.yaml),$(if $(findstring /,$(CONFIG)),boeing_landing/pipelines/$(CONFIG).yaml,boeing_landing/pipelines/$(CONFIG)/base.yaml))

.DEFAULT_GOAL := help
.PHONY: help install deps dataset augment trajectories data-report train evaluate plots plots-orders \
        experiment-order experiment-convergence quadrotor-train quadrotor-test clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-22s %s\n", $$1, $$2}'

install:  ## create .venv and install dependencies
	python -m venv .venv
	$(MAKE) deps

deps:  ## install dependencies into the detected interpreter
	$(PYTHON) -m pip install -r requirements.txt

dataset:  ## build the npz: make dataset CONFIG=ils_aligned_cfc (frame pipelines read augment.out_csv; add CSV=... to override, required for gps_cfc)
	$(PYTHON) -m boeing_landing.data.build_dataset $(CSV) --config $(CFGPATH)

# csv augmentation (sources are read-only). The augmentation to run and where it
# writes come from the pipeline's `augment:` config, so one target serves every
# frame: make augment CONFIG=ils_aligned_cfc  /  CONFIG=magnetic_north_cfc.
RAW_CSV ?= datasets/ldg_dataset_images.csv
NAVDB   ?= datasets/NavDB_MFS.json
NED_CSV ?= datasets/ldg_dataset_images_ned.csv

augment:  ## augment the raw csv the way CONFIG says: make augment CONFIG=ils_aligned_cfc [RAW_CSV=...]
	$(PYTHON) -m boeing_landing.data.augment $(RAW_CSV) $(NAVDB) --config $(CFGPATH)

trajectories:  ## plot the approaches of the augmented csv: make trajectories [NED_CSV=...] [SAVE=1]
	$(PYTHON) -m boeing_landing.pipelines.ils_aligned_cfc.plot_runway_frame $(NED_CSV) $(if $(SAVE),--save)

data-report:  ## per-channel diagnostics (weak/near-constant channels, [0,1] use, val shift): make data-report CONFIG=ils_aligned_cfc [SAVE=1]
	$(PYTHON) -m boeing_landing.data.data_report --config $(CFGPATH) $(if $(SAVE),--save)

train:  ## train a pipeline: make train CONFIG=ils_aligned_cfc (ORDER=... overrides the yaml's input_order; EPOCHS=3 for a quick trial)
	$(PYTHON) -m boeing_landing.train --config $(CFGPATH) $(if $(ORDER),--input-order $(ORDER)) $(if $(EPOCHS),--max-epochs $(EPOCHS))

evaluate:  ## metrics + ablations of a run: make evaluate RUN=runs/<pipeline>/<variant>/<timestamp>
	$(PYTHON) -m boeing_landing.evaluate --run $(RUN)

plots:  ## curves/ablation/sweeps: make plots RUNS="..." (SAVE=1 -> PNG in figures/; BARS=1 -> best-val_loss bars; NOISE=0.005)
	$(PYTHON) -m boeing_landing.report --runs $(RUNS) $(if $(SAVE),--save) $(if $(BARS),--bars) $(if $(NOISE),--noise $(NOISE))

plots-orders:  ## conv-order sweep bars: CONFIG=<pipeline> STAMP=<session prefix> SAVE=1
	$(PYTHON) -m boeing_landing.report --orders --config $(CFGPATH) $(if $(STAMP),--stamp $(STAMP)) $(if $(SAVE),--save) $(if $(NOISE),--noise $(NOISE))

experiment-order:  ## sweep the conv channel orders and compare val loss
	$(PYTHON) -m boeing_landing.experiments.feature_order --config $(CFGPATH)

experiment-convergence:  ## train the config under several seeds (experiments.seeds)
	$(PYTHON) -m boeing_landing.experiments.convergence --config $(CFGPATH)

quadrotor-train:  ## train the quadrotor baseline (settings: quadrotor_baseline/train_config.yaml)
	$(PYTHON) -m quadrotor_baseline.train

quadrotor-test:  ## evaluate the quadrotor baseline (model picked in test_config.yaml; PLOT=1)
	$(PYTHON) -m quadrotor_baseline.test $(if $(PLOT),--plot)

clean:  ## remove generated runs, logs and __pycache__ caches
	rm -rf lightning_logs runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
