# Drugzello ML Backend

This is the machine learning backend for the Drugzello solubility prediction project.

## Project Structure

- `configs/`: Configuration files for different environments and components
- `data/`: Data storage and organization
  - `raw/`: Original, immutable datasets
  - `processed/`: Cleaned and transformed data
  - `features/`: Feature engineering outputs
  - `external/`: External reference data
- `models/`: Model artifacts and registry
  - `trained/`: Saved model files
  - `experiments/`: Experiment tracking
  - `registry/`: Production model registry
- `src/`: Source code modules
  - `data/`: Data processing utilities
  - `models/`: Model implementations
  - `features/`: Feature engineering
  - `utils/`: Common utilities
  - `monitoring/`: Model monitoring and drift detection
- `scripts/`: Executable scripts for training, evaluation, etc.
- `tests/`: Test suite
- `requirements/`: Dependency management
- `docs/`: Documentation

## Getting Started

1. Install dependencies: `pip install -r requirements/base.txt`
2. Configure environment: Copy `.env.example` to `.env` and update settings
3. Run tests: `pytest tests/`
4. Train model: `python scripts/train_model.py`
