"""Load schema-checked trained MEGAN checkpoints. Errors never fabricate outputs."""

import logging
import math
from pathlib import Path
from threading import RLock

import torch

from core.config import settings
from src.data.graphs import FEATURE_SCHEMA
from src.models.experiment import CHECKPOINT_FORMAT
from src.models.megan_architecture import SUPPORTED_ARCHITECTURES, MEGANCore
from src.models.megan_inference import MEGANInference

logger = logging.getLogger(__name__)


class MEGANModelWrapper:
    def __init__(self, model, model_config, normalization, device="cpu"):
        self.model = model
        self.model_version = model.architecture_version
        self.inference_module = MEGANInference(model, model_config, device, normalization)

    def predict(self, smiles, include_uncertainty=False, include_explanations=False):
        return self.inference_module.predict_with_uncertainty(
            smiles, n_samples=20 if include_uncertainty else 1,
            include_explanations=include_explanations
        )

    def predict_with_uncertainty(self, smiles, n_samples=20, include_explanations=True):
        return self.inference_module.predict_with_uncertainty(smiles, n_samples, include_explanations)


class ModelLoader:
    def __init__(self):
        self._model = self._model_version = None
        self.last_error = None
        self._lock = RLock()

    def load_model(self, model_path=None):
        with self._lock:
            self.unload_model()
            path = Path(model_path or settings.model_path)
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Trained checkpoint is missing: {path}")
                data = torch.load(path, map_location="cpu", weights_only=True)
                if data.get("format") != CHECKPOINT_FORMAT or data.get("architecture_version") not in SUPPORTED_ARCHITECTURES:
                    raise ValueError("Incompatible legacy checkpoint; retrain with the maintained scaffold CLI")
                if data.get("feature_schema") != FEATURE_SCHEMA:
                    raise ValueError("Checkpoint feature schema does not match molecular preprocessing")
                norm = data["normalization"]
                if (norm.get("method") != "standard" or norm.get("std", 0) <= 0
                        or not all(math.isfinite(norm[k]) for k in ("mean", "std"))):
                    raise ValueError("Invalid checkpoint target normalization")
                model = MEGANCore(**data["model_kwargs"])
                if model.architecture_version != data["architecture_version"]:
                    raise ValueError("Checkpoint architecture version disagrees with its encoding")
                model.load_state_dict(data["model_state_dict"], strict=True)
                self._model = MEGANModelWrapper(model.eval(), data["metadata"].get("config", {}), norm)
                self._model_version = f"{model.architecture_version}:{path.stem}"
                self.last_error = None
                return True
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("Model unavailable: %s", exc)
                return False

    def get_model(self):
        with self._lock:
            if self._model is None and not self.load_model():
                raise RuntimeError(self.last_error or "Trained MEGAN checkpoint unavailable")
            return self._model

    def is_loaded(self):
        return self._model is not None

    def get_version(self):
        return self._model_version

    def unload_model(self):
        self._model = self._model_version = None


model_loader = ModelLoader()
