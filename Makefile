PYTHON ?= .venv/bin/python
MODEL_PATH ?= models/trained/scaffold-tuned/model.pth
export MPLBACKEND := Agg
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export PYTHONHASHSEED := 42

.PHONY: test train tune api verify-results

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

api:
	cd backend && MEGAN_MODEL_PATH=$(MODEL_PATH) $(abspath $(PYTHON)) -m uvicorn main:app --reload

verify-results:
	$(PYTHON) results/scaffold-tuning/verify_results.py
