import pickle
import torch
import logging
import sys
import os
from typing import Optional, Any
from pathlib import Path

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.config import settings

# Import MEGAN model classes
try:
    from src.models.megan_architecture import MEGANCore
    from src.models.megan_inference import MEGANInference
    from torch_geometric.data import Data
    MEGAN_AVAILABLE = True
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import MEGAN classes: {e}")
    MEGAN_AVAILABLE = False

logger = logging.getLogger(__name__)

class MEGANModelWrapper:
    """Wrapper for the MEGAN PyTorch model to provide a consistent interface."""
    
    def __init__(self, model: MEGANCore, model_config: dict, device: str = 'cpu'):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.model_version = "1.0.0-pytorch"
        self.is_loaded = True
        
        # Initialize the MEGAN inference module
        self.inference_module = MEGANInference(model, device=device)
        logger.info(f"Initialized MEGAN model with inference module on device: {device}")
    
    def predict(self, smiles: str, include_uncertainty: bool = False, include_explanations: bool = False) -> dict:
        """Prediction method for MEGAN model with comprehensive analysis."""
        try:
            if include_uncertainty or include_explanations:
                # Use full analysis with uncertainty quantification and interpretability
                return self.inference_module.predict_with_uncertainty(
                    smiles, 
                    uncertainty_samples=50,
                    include_explanations=include_explanations
                )
            else:
                # Use simple prediction for compatibility
                return self.inference_module.predict(smiles)
                
        except Exception as e:
            logger.error(f"Error in MEGAN prediction: {e}")
            # Fallback to mock prediction
            mock_solubility = len(smiles) * -0.08
            mock_confidence = max(0.6, min(0.98, 1.0 - len(smiles) * 0.015))
            
            return {
                "solubility": mock_solubility,
                "confidence": mock_confidence,
                "model_type": "megan_pytorch_fallback",
                "error": str(e)
            }

class MockMEGANModel:
    """Mock MEGAN model for testing purposes."""
    
    def __init__(self):
        self.model_version = "1.0.0-mock"
        self.is_loaded = True
        logger.info("Initialized mock MEGAN model")
    
    def predict(self, smiles: str, include_uncertainty: bool = False, include_explanations: bool = False) -> dict:
        """Mock prediction method with optional uncertainty and explanations."""
        # Simple mock logic based on molecule size
        mock_solubility = len(smiles) * -0.1  # Longer molecules = less soluble
        mock_confidence = max(0.5, min(0.95, 1.0 - len(smiles) * 0.02))
        
        result = {
            "solubility": mock_solubility,
            "confidence": mock_confidence,
            "model_type": "mock"
        }
        
        if include_uncertainty:
            result.update({
                "uncertainty": {
                    "prediction_std": 0.1,
                    "uaa_score": 0.15,
                    "aau_scores": [0.05] * min(len(smiles), 10)  # Mock per-atom uncertainty
                }
            })
        
        if include_explanations:
            result.update({
                "explanations": {
                    "node_importances": [0.1] * min(len(smiles), 10),  # Mock atom importances
                    "edge_importances": [0.05] * max(0, min(len(smiles), 10) - 1),  # Mock bond importances
                    "interpretation": f"Mock interpretation: This molecule with {len(smiles)} characters shows moderate solubility.",
                    "rdkit_comparison": {
                        "crippen_logp": mock_solubility * -0.5,
                        "agreement_score": 0.7
                    }
                }
            })
        
        return result

class ModelLoader:
    """Handles loading and managing the MEGAN model."""
    def __init__(self):
        self._model: Optional[Any] = None
        self._model_version: Optional[str] = None
        self._is_loaded = False
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the MEGAN model from file or create mock model."""
        try:
            model_path = model_path or settings.model_path
            
            if model_path and Path(model_path).exists() and MEGAN_AVAILABLE:
                logger.info(f"Loading MEGAN PyTorch model from {model_path}")
                
                # Create MEGANConfig in __main__ module for pickle compatibility
                import __main__
                if not hasattr(__main__, 'MEGANConfig'):
                    class MEGANConfig:
                        """Stub config class for backward compatibility with saved models."""
                        def __init__(self, **kwargs):
                            for key, value in kwargs.items():
                                setattr(self, key, value)
                    __main__.MEGANConfig = MEGANConfig
                
                # Load the PyTorch model data
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load with explicit weights_only=False and suppress warning
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
                    model_data = torch.load(model_path, map_location=device, weights_only=False)
                
                # Extract model configuration and kwargs
                model_kwargs = model_data.get('model_kwargs', {})
                config = model_data.get('config', {})
                
                logger.info(f"Model config: {config}")
                logger.info(f"Model kwargs: {model_kwargs}")
                
                # Create the MEGAN model with the saved configuration
                megan_model = MEGANCore(**model_kwargs)
                
                # Load the trained weights
                megan_model.load_state_dict(model_data['model_state_dict'])
                megan_model.to(device)
                megan_model.eval()  # Set to evaluation mode
                
                # Wrap the model in our interface
                self._model = MEGANModelWrapper(megan_model, config, str(device))
                self._model_version = f"1.0.0-pytorch-epoch-{model_data.get('epoch', 'unknown')}"
                
                logger.info(f"Successfully loaded MEGAN PyTorch model version {self._model_version}")
                
            elif model_path and Path(model_path).exists() and not MEGAN_AVAILABLE:
                logger.warning("MEGAN classes not available, but model file exists. Using mock model.")
                self._model = MockMEGANModel()
                self._model_version = self._model.model_version
                
            else:
                logger.warning(f"Model file not found at {model_path}, using mock model")
                self._model = MockMEGANModel()
                self._model_version = self._model.model_version
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fall back to mock model
            logger.info("Falling back to mock model")
            self._model = MockMEGANModel()
            self._model_version = self._model.model_version
            self._is_loaded = True
            return True
    
    def get_model(self) -> Optional[Any]:
        """Get the loaded model."""
        if not self._is_loaded:
            self.load_model()
        return self._model
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self._model is not None
    
    def get_version(self) -> Optional[str]:
        """Get model version."""
        return self._model_version
    
    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None
        self._model_version = None
        self._is_loaded = False
        logger.info("Model unloaded")

# Global model loader instance
model_loader = ModelLoader()
