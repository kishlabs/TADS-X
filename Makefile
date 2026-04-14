# Makefile for TADS-X — Team ChipSmiths
# Usage: make <target>

PYTHON     := python
PIP        := pip
VENV       := .venv
VENV_BIN   := $(VENV)/bin
ACTIVATE   := source $(VENV_BIN)/activate

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  TADS-X — Task-Aware Dual-Stream Detection"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  Setup"
	@echo "    venv          Create a virtual environment in .venv/"
	@echo "    install       Install runtime dependencies"
	@echo "    install-dev   Install runtime + dev dependencies"
	@echo ""
	@echo "  Quality"
	@echo "    lint          Run flake8 linter"
	@echo "    format        Auto-format code with autopep8"
	@echo ""
	@echo "  Run"
	@echo "    cache         Build the ROI feature cache (requires COCO)"
	@echo "    train         Train the model (requires COCO + cache)"
	@echo "    eval          Run full evaluation (requires COCO)"
	@echo "    eval-fast     Run subset evaluation (500 images/task)"
	@echo ""
	@echo "  Clean"
	@echo "    clean         Remove Python cache files"
	@echo "    clean-all     Remove cache files + checkpoints"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV_BIN)/activate"

.PHONY: install
install:
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# ── Quality ───────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(PYTHON) -m flake8 application/ --max-line-length=120 --exclude=application/tinybert_local

.PHONY: format
format:
	$(PYTHON) -m autopep8 --in-place --recursive --max-line-length=120 application/

# ── Run ───────────────────────────────────────────────────────────────────────
COCO_DIR   ?= ./COCO
TASKS_DIR  ?= ./COCO/dataset-master/coco-tasks/annotations
CACHE_DIR  ?= application/data/roi_cache
CONFIG     ?= application/configs/train_config.yaml

.PHONY: cache
cache:
	$(PYTHON) application/train.py --build-cache \
	    --coco-dir  $(COCO_DIR) \
	    --tasks-dir $(TASKS_DIR) \
	    --cache-dir $(CACHE_DIR)

.PHONY: train
train:
	$(PYTHON) application/train.py --config $(CONFIG)

.PHONY: eval
eval:
	$(PYTHON) application/evaluate.py \
	    --coco-dir  $(COCO_DIR) \
	    --tasks-dir $(TASKS_DIR)

.PHONY: eval-fast
eval-fast:
	$(PYTHON) application/evaluate.py \
	    --coco-dir  $(COCO_DIR) \
	    --tasks-dir $(TASKS_DIR) \
	    --subset 500

# ── Clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete

.PHONY: clean-all
clean-all: clean
	rm -rf checkpoints/ checkpoints_test/
	rm -rf application/data/roi_cache*/
