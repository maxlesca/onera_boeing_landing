# Boeing landing -- command shortcuts. Run from the repo root.
# The interpreter is auto-detected: Linux venv, then Windows venv (also works
# from WSL via interop), then the active environment's `python` (e.g. conda).
# Still overridable:  make train PYTHON=/path/to/python
PYTHON  ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,python))
CSV     ?= ../datasets/dataset_sans_barres/ldg_dataset_images_Maxime.csv
CONFIG  ?= gps_cfc
ORDER   ?= grouped
CFGPATH  = boeing_landing/configs/$(CONFIG).yaml

.DEFAULT_GOAL := help
.PHONY: help install deps dataset train evaluate plots experiment-order experiment-convergence quadrotor-train clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-22s %s\n", $$1, $$2}'

install:  ## create .venv and install dependencies
	python -m venv .venv
	$(MAKE) deps

deps:  ## install dependencies into the detected interpreter
	$(PYTHON) -m pip install -r requirements.txt

dataset:  ## build the npz from the CSV (val runs / out dir come from the config)
	$(PYTHON) -m boeing_landing.data.build_dataset $(CSV) --config $(CFGPATH)

train:  ## train a pipeline: make train CONFIG=gps_cfc ORDER=grouped (EPOCHS=3 for a quick trial)
	$(PYTHON) -m boeing_landing.train --config $(CFGPATH) --input-order $(ORDER) $(if $(EPOCHS),--max-epochs $(EPOCHS))

evaluate:  ## metrics + ablations of a run: make evaluate RUN=runs/<name>/<timestamp>
	$(PYTHON) -m boeing_landing.evaluate --run $(RUN)

plots:  ## curves/ablation/sweeps: make plots RUNS="..." (SAVE=1 -> PNG; BARS=1 -> best-val_loss bars; NOISE=0.005)
	$(PYTHON) -m boeing_landing.report --runs $(RUNS) $(if $(SAVE),--save) $(if $(BARS),--bars) $(if $(NOISE),--noise $(NOISE))

plots-orders:  ## bar chart of the conv-order sweep (auto-discovers runs; SAVE=1 -> PNG)
	$(PYTHON) -m boeing_landing.report --orders $(if $(SAVE),--save) $(if $(NOISE),--noise $(NOISE))

experiment-order:  ## sweep the conv channel orders and compare val loss
	$(PYTHON) -m boeing_landing.experiments.feature_order --config $(CFGPATH)

experiment-convergence:  ## train the config under several seeds (experiments.seeds)
	$(PYTHON) -m boeing_landing.experiments.convergence --config $(CFGPATH)

quadrotor-train:  ## train the quadrotor baseline
	$(PYTHON) -m quadrotor_baseline.train

clean:  ## remove generated runs, logs and __pycache__ caches
	rm -rf lightning_logs runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
