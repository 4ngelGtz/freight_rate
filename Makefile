# GBM training pipeline — mirrors scripts/run_pipeline.py
#
#   make help              list targets (default)
#   make pipeline          download → HPO → walk-forward → manifest
#   make pipeline-smoke    fast sanity check (2 HPO configs, 3 folds)

PYTHON ?= .venv/bin/python
ifeq ($(wildcard $(PYTHON)),)
  PYTHON := python3
endif

HPO_PARAMS := models/nested_hpo/best_params.json
MODELS_DIR := models

.PHONY: help setup download download-rates download-diesel \
        hpo hpo-smoke walkforward walkforward-smoke manifest \
        pipeline pipeline-smoke pipeline-no-download \
        score score-no-download \
        test lint

help:
	@echo "Freight rate pipeline targets:"
	@echo ""
	@echo "  setup                 pip install -e ."
	@echo "  download              rates + diesel snapshots → data/raw/"
	@echo "  download-rates        USDA refrigerated truck rates only"
	@echo "  download-diesel       EIA diesel weekly only"
	@echo "  hpo                   tune θ* on val → models/nested_hpo/"
	@echo "  walkforward           full eval with θ* → models/walkforward_gbm/"
	@echo "  manifest              write models/run_manifest.json"
	@echo "  pipeline              download + hpo + walkforward + manifest"
	@echo "  pipeline-no-download  hpo + walkforward + manifest (data already local)"
	@echo "  pipeline-smoke        pipeline with --max-configs 2 --max-folds 3"
	@echo "  score                 download + score next Tuesday (forward forecast, θ*)"
	@echo "  score-no-download     score next Tuesday (data already local)"
	@echo "  hpo-smoke / walkforward-smoke   individual smoke steps"
	@echo "  test / lint           pytest / ruff"

setup:
	$(PYTHON) -m pip install -e .

download: download-rates download-diesel

download-rates:
	$(PYTHON) scripts/download_usda_data.py

download-diesel:
	$(PYTHON) scripts/download_diesel_data.py

hpo:
	$(PYTHON) scripts/run_nested_hpo_gbm.py

# Hyperparameter optimization
hpo-smoke:
	$(PYTHON) scripts/run_nested_hpo_gbm.py --max-configs 2 --max-folds 3

walkforward:
	$(PYTHON) scripts/train_walkforward_gbm.py --hpo-params $(HPO_PARAMS)

walkforward-smoke:
	$(PYTHON) scripts/train_walkforward_gbm.py --hpo-params $(HPO_PARAMS) --max-folds 3

manifest:
	$(PYTHON) -c "from pathlib import Path; from freight_rates.artifacts import write_run_manifest; print(write_run_manifest(models_dir=Path('$(MODELS_DIR)')))"

pipeline: download hpo walkforward manifest

pipeline-no-download: hpo walkforward manifest

pipeline-smoke: download hpo-smoke walkforward-smoke manifest

score: download score-no-download

score-no-download:
	$(PYTHON) scripts/score_week.py --hpo-params $(HPO_PARAMS)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts
