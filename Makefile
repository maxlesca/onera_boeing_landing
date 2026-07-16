# Boeing landing -- command shortcuts. Run from the repo root.
# Override the interpreter on another machine:  make train PYTHON=python
PYTHON ?= .venv/Scripts/python.exe
CSV    ?= ../datasets/dataset_sans_barres/ldg_dataset_images_Maxime.csv
CONFIG ?= step1_cfc
ORDER  ?= grouped

.DEFAULT_GOAL := help
.PHONY: help install dataset train evaluate plots experiment-order quadrotor-train clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n", $$1, $$2}'

install:  ## create .venv and install dependencies
	python -m venv .venv && $(PYTHON) -m pip install -r requirements.txt

dataset:  ## build the step-1 npz from the CSV (GPS in, ILS out)
	$(PYTHON) -m boeing_landing.data.build_dataset $(CSV)

train:  ## train a pipeline: make train CONFIG=step1_cfc ORDER=grouped
	$(PYTHON) -m boeing_landing.train --config boeing_landing/configs/$(CONFIG).yaml --input-order $(ORDER)

evaluate:  ## evaluate a run + feature ablations: make evaluate RUN=runs/<name>/<timestamp>
	$(PYTHON) -m boeing_landing.evaluate --run $(RUN) --ablation

plots:  ## training curves: make plots RUNS="runs/<n>/<ts>" (several = comparison; SAVE=1 -> PNG)
	$(PYTHON) -m boeing_landing.report --runs $(RUNS) $(if $(SAVE),--save)

experiment-order:  ## sweep the conv channel orders and compare val loss
	$(PYTHON) -m boeing_landing.experiments.feature_order

quadrotor-train:  ## train Tudor's original quadrotor baseline
	$(PYTHON) -m quadrotor_baseline.train

clean:  ## remove generated runs, logs and __pycache__ caches
	rm -rf lightning_logs runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
