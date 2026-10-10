PYTHON ?= .venv/bin/python
PAPER_PYTHON ?= .runtime/esol-paper/bin/python
MODEL_PATH ?= models/trained/scaffold-tuned/model.pth
export MPLBACKEND := Agg
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export PYTHONHASHSEED := 42

.PHONY: test train tune gat megan-no-aux paper-esol mask-chemistry api verify-results

test:
	$(PYTHON) -m pytest -q backend/tests/research

train:
	cd backend && $(abspath $(PYTHON)) scripts/train_model.py \
		--config-file configs/scaffold_esol.yaml --n-folds 5 --epochs 30 --seed 42 \
		--device cpu --save-dir models/trained/scaffold-reproduction \
		--results-dir results/scaffold-reproduction --refit --disable-mlflow

tune:
	cd backend && $(abspath $(PYTHON)) -u scripts/tune_model.py \
		--search-config configs/scaffold_search.yaml \
		--splits ../results/scaffold-tuning/splits.json \
		--save-dir models/trained/scaffold-tuning-reproduction \
		--results-dir results/scaffold-tuning-reproduction --workers 3 --stage all --refit

gat:
	cd backend && $(abspath $(PYTHON)) -u scripts/train_gat.py \
		--config-file configs/scaffold_gat.yaml \
		--splits ../results/scaffold-tuning/splits.json \
		--save-dir models/trained/scaffold-gat-reproduction \
		--results-dir results/scaffold-gat-reproduction

megan-no-aux:
	cd backend && $(abspath $(PYTHON)) -u scripts/train_model.py \
		--config-file configs/scaffold_megan_no_aux.yaml \
		--splits ../results/scaffold-tuning/splits.json --n-folds 5 --seed 42 \
		--device cpu --save-dir models/trained/scaffold-megan-no-aux-reproduction \
		--results-dir results/scaffold-megan-no-aux-reproduction \
		--skip-random-forest --disable-mlflow

paper-esol:
	$(PAPER_PYTHON) -u backend/scripts/reproduce_paper_esol.py \
		--results-dir backend/results/paper-esol-reproduction \
		--save-dir backend/models/trained/paper-esol-reproduction --workers 2

mask-chemistry:
	$(PYTHON) -u backend/scripts/evaluate_mask_chemistry.py --cohort scaffold_prediction_only
	$(PYTHON) -u backend/scripts/evaluate_mask_chemistry.py --cohort scaffold_joint_auxiliary
	$(PAPER_PYTHON) -u backend/scripts/evaluate_paper_masks.py
	$(PYTHON) backend/scripts/summarize_mask_chemistry.py
	$(PYTHON) backend/scripts/plot_mask_chemistry.py

api:
	cd backend && MEGAN_MODEL_PATH=$(MODEL_PATH) $(abspath $(PYTHON)) -m uvicorn main:app --reload

verify-results:
	$(PYTHON) results/scaffold-tuning/verify_results.py
