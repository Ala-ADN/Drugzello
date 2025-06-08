"""
Comprehensive inference tests for MEGAN molecular solubility prediction models.
Tests model loading, prediction functionality, and end-to-end inference pipelines.
"""

import pytest
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Union
import json

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Module-level fixture for sample molecular data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }

from src.models.megan_architecture import MEGANCore
from src.models.trainer import MEGANTrainer
from src.data.data_loader import load_molecular_data
from src.utils.config import MEGANConfig
from src.utils.config import SearchSpace
from src.utils.evaluation import ModelEvaluator

# Module-level fixture for sample molecular data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }

# Module-level fixture for sample molecular data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }
# Module-level fixture for sample molecular data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }

# Module-level fixture for sample molecular data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }

# Module-level fixture for sample data
@pytest.fixture
def sample_molecular_data():
    """Create sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'C1=CC=C(C=C1)O',  # Phenol
            'CCCCC',  # Pentane
        ],
        'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
    }


class MockMEGANModel:
    """Mock MEGAN model for testing inference pipeline."""
    
    def __init__(self, config=None):
        self.config = config or {"hidden_dim": 64, "num_layers": 3}
        self.model_type = "MEGAN_Solubility_Predictor"
        self.version = "1.0.0"
        self.device = torch.device("cpu")
        self.is_trained = True
        self.input_features = ["node_features", "edge_features", "edge_indices"]
        
    def forward(self, batch):
        """Mock forward pass."""
        batch_size = batch.x.shape[0] if hasattr(batch, 'x') else 32
        return torch.randn(batch_size, 1)
    
    def predict(self, data):
        """Mock prediction method."""
        # Handle None or empty inputs
        if data is None:
            raise ValueError("No data provided for prediction.")
        # Determine batch size
        if isinstance(data, dict):
            smiles = data.get('smiles', [])
            if not smiles:
                raise ValueError("Empty SMILES in input dict.")
            batch_size = len(smiles)
        elif hasattr(data, '__len__'):
            if len(data) == 0:
                raise ValueError("Empty input sequence.")
            batch_size = len(data)
        else:
            batch_size = 1
        # Return realistic solubility predictions (log mol/L)
        torch.manual_seed(42)
        predictions = torch.randn(batch_size) * 2.0 - 3.0  # Mean around -3, std 2
        return predictions.numpy()
    
    def predict_uncertainty(self, data):
        """Mock uncertainty prediction."""
        predictions = self.predict(data)
        uncertainties = np.abs(np.random.normal(0, 0.5, len(predictions)))
        return predictions, uncertainties
    
    def eval(self):
        """Set model to evaluation mode."""
        pass
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        return self


class TestModelLoading:
    """Test suite for model loading functionality."""
    
    @pytest.fixture
    def mock_pytorch_model_file(self, tmp_path):
        """Create a temporary PyTorch model file."""
        model = MockMEGANModel()
        model_path = tmp_path / "mock_megan_model.pth"
        
        # Save as PyTorch state dict
        torch.save({
            'model_state_dict': {'dummy': torch.tensor([1.0])},
            'config': model.config,
            'model_type': model.model_type,
            'version': model.version,
            'metadata': {
                'training_samples': 1000,
                'validation_rmse': 0.85,
                'feature_importance': {'molecular_weight': 0.3, 'logp': 0.25}
            }
        }, model_path)
        
        return model_path
    
    @pytest.fixture
    def mock_pickle_model_file(self, tmp_path):
        """Create a temporary pickled model file."""
        model = MockMEGANModel()
        model_path = tmp_path / "mock_megan_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model_path
    
    @pytest.fixture
    def sample_molecular_data(self):
        """Create sample molecular data for testing."""
        return {
            'smiles': [
                'CCO',  # Ethanol
                'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
                'C1=CC=C(C=C1)O',  # Phenol
                'CCCCC',  # Pentane
            ],
            'true_solubility': [-0.77, -2.23, -0.55, -4.25, -0.04, -4.8]
        }
    
    def test_load_pytorch_model(self, mock_pytorch_model_file):
        """Test loading PyTorch model checkpoint."""
        checkpoint = torch.load(mock_pytorch_model_file, map_location='cpu')
        
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'model_type' in checkpoint
        assert checkpoint['model_type'] == "MEGAN_Solubility_Predictor"
        assert checkpoint['version'] == "1.0.0"
    
    def test_load_pickle_model(self, mock_pickle_model_file):
        """Test loading pickled model."""
        with open(mock_pickle_model_file, 'rb') as f:
            model = pickle.load(f)
        
        assert model.model_type == "MEGAN_Solubility_Predictor"
        assert model.version == "1.0.0"
        assert model.is_trained is True
        assert hasattr(model, 'predict')
    
    def test_model_metadata_extraction(self, mock_pytorch_model_file):
        """Test extraction of model metadata."""
        checkpoint = torch.load(mock_pytorch_model_file, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        
        assert 'training_samples' in metadata
        assert 'validation_rmse' in metadata
        assert 'feature_importance' in metadata
        assert metadata['training_samples'] == 1000


class TestInferenceFunctionality:
    """Test suite for model inference functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        return MockMEGANModel()
    
    def test_single_molecule_prediction(self, mock_model, sample_molecular_data):
        """Test prediction on a single molecule."""
        single_smiles = sample_molecular_data['smiles'][0]
        
        prediction = mock_model.predict([single_smiles])
        
        assert len(prediction) == 1
        assert isinstance(prediction[0], (float, np.floating))
        assert -10 < prediction[0] < 5  # Reasonable solubility range
    
    def test_batch_prediction(self, mock_model, sample_molecular_data):
        """Test batch prediction on multiple molecules."""
        smiles_list = sample_molecular_data['smiles']
        
        predictions = mock_model.predict(smiles_list)
        
        assert len(predictions) == len(smiles_list)
        assert all(isinstance(pred, (float, np.floating)) for pred in predictions)
        assert all(-10 < pred < 5 for pred in predictions)
    
    def test_prediction_consistency(self, mock_model):
        """Test that predictions are consistent across multiple runs."""
        test_smiles = ['CCO', 'CC(=O)O']
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        predictions1 = mock_model.predict(test_smiles)
        
        torch.manual_seed(42)
        np.random.seed(42)
        predictions2 = mock_model.predict(test_smiles)
        
        np.testing.assert_array_almost_equal(predictions1, predictions2, decimal=5)
    
    def test_uncertainty_prediction(self, mock_model, sample_molecular_data):
        """Test uncertainty estimation in predictions."""
        smiles_list = sample_molecular_data['smiles']
        
        predictions, uncertainties = mock_model.predict_uncertainty(smiles_list)
        
        assert len(predictions) == len(smiles_list)
        assert len(uncertainties) == len(smiles_list)
        assert all(unc >= 0 for unc in uncertainties)  # Uncertainties should be non-negative
    
    def test_empty_input_handling(self, mock_model):
        """Test handling of empty input."""
        with pytest.raises((ValueError, IndexError)):
            mock_model.predict([])
    
    def test_invalid_smiles_handling(self, mock_model):
        """Test handling of invalid SMILES strings."""
        invalid_smiles = ['INVALID_SMILES', 'C[C@H]([C@H]']
        
        # Model should either handle gracefully or raise appropriate error
        try:
            predictions = mock_model.predict(invalid_smiles)
            assert len(predictions) == len(invalid_smiles)
        except (ValueError, RuntimeError):
            # Expected behavior for invalid SMILES
            pass


class TestPerformanceMetrics:
    """Test suite for inference performance evaluation."""
    
    @pytest.fixture
    def prediction_results(self, sample_molecular_data):
        """Generate mock prediction results."""
        true_values = sample_molecular_data['true_solubility']
        # Add some noise to true values to simulate predictions
        predicted_values = [val + np.random.normal(0, 0.5) for val in true_values]
        
        return {
            'smiles': sample_molecular_data['smiles'],
            'true_solubility': true_values,
            'predicted_solubility': predicted_values
        }
    
    def test_mae_calculation(self, prediction_results):
        """Test Mean Absolute Error calculation."""
        true_vals = np.array(prediction_results['true_solubility'])
        pred_vals = np.array(prediction_results['predicted_solubility'])
        
        mae = np.mean(np.abs(true_vals - pred_vals))
        
        assert mae >= 0
        assert isinstance(mae, (float, np.floating))
    
    def test_rmse_calculation(self, prediction_results):
        """Test Root Mean Square Error calculation."""
        true_vals = np.array(prediction_results['true_solubility'])
        pred_vals = np.array(prediction_results['predicted_solubility'])
        
        rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
        
        assert rmse >= 0
        assert isinstance(rmse, (float, np.floating))
    
    def test_r2_calculation(self, prediction_results):
        """Test R-squared calculation."""
        true_vals = np.array(prediction_results['true_solubility'])
        pred_vals = np.array(prediction_results['predicted_solubility'])
        
        # Calculate R-squared
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        assert -np.inf < r2 <= 1  # R-squared can be negative but not greater than 1
        assert isinstance(r2, (float, np.floating))


class TestInferenceSpeed:
    """Test suite for inference speed and performance."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for performance testing."""
        return MockMEGANModel()
    
    def test_single_prediction_speed(self, mock_model):
        """Test speed of single molecule prediction."""
        import time
        
        test_smiles = 'CCO'
        
        start_time = time.time()
        prediction = mock_model.predict([test_smiles])
        inference_time = time.time() - start_time
        
        assert len(prediction) == 1
        assert inference_time < 1.0  # Should be fast for single prediction
    
    def test_batch_prediction_speed(self, mock_model):
        """Test speed of batch prediction."""
        import time
        
        # Create larger batch
        test_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1'] * 100  # 300 molecules
        
        start_time = time.time()
        predictions = mock_model.predict(test_smiles)
        inference_time = time.time() - start_time
        
        assert len(predictions) == len(test_smiles)
        assert inference_time < 5.0  # Should handle batch efficiently
        
        # Calculate throughput
        throughput = len(test_smiles) / inference_time
        assert throughput > 10  # At least 10 molecules per second
    
    def test_memory_usage(self, mock_model):
        """Test memory usage during inference."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference on larger batch
        large_batch = ['CCO'] * 1000
        predictions = mock_model.predict(large_batch)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        assert len(predictions) == 1000
        assert memory_increase < 100  # Should not use excessive memory (< 100MB)


class TestEndToEndInference:
    """End-to-end inference testing."""
    
    @pytest.fixture
    def inference_pipeline(self):
        """Create complete inference pipeline."""
        class InferencePipeline:
            def __init__(self):
                self.model = MockMEGANModel()
                self.preprocessor = self._create_mock_preprocessor()
                self.postprocessor = self._create_mock_postprocessor()
            
            def _create_mock_preprocessor(self):
                """Create mock preprocessor."""
                def preprocess(smiles_list):
                    # Mock preprocessing that converts SMILES to model inputs
                    return {'processed_smiles': smiles_list}
                return preprocess
            
            def _create_mock_postprocessor(self):
                """Create mock postprocessor."""
                def postprocess(predictions, smiles_list):
                    # Mock postprocessing
                    results = []
                    for smiles, pred in zip(smiles_list, predictions):
                        results.append({
                            'smiles': smiles,
                            'predicted_solubility': float(pred),
                            'solubility_class': 'soluble' if pred > -3 else 'insoluble',
                            'confidence': min(abs(pred) / 5.0, 1.0)
                        })
                    return results
                return postprocess
            
            def predict(self, smiles_list):
                """Complete inference pipeline."""
                # Preprocess
                processed_data = self.preprocessor(smiles_list)
                
                # Predict
                predictions = self.model.predict(processed_data['processed_smiles'])
                
                # Postprocess
                results = self.postprocessor(predictions, smiles_list)
                
                return results
        
        return InferencePipeline()
    
    def test_complete_pipeline(self, inference_pipeline, sample_molecular_data):
        """Test complete inference pipeline."""
        smiles_list = sample_molecular_data['smiles']
        
        results = inference_pipeline.predict(smiles_list)
        
        assert len(results) == len(smiles_list)
        
        for result in results:
            assert 'smiles' in result
            assert 'predicted_solubility' in result
            assert 'solubility_class' in result
            assert 'confidence' in result
            assert result['solubility_class'] in ['soluble', 'insoluble']
            assert 0 <= result['confidence'] <= 1
    
    def test_pipeline_error_handling(self, inference_pipeline):
        """Test pipeline error handling."""
        # Test with empty input
        with pytest.raises((ValueError, IndexError)):
            inference_pipeline.predict([])
        
        # Test with None input
        with pytest.raises((TypeError, ValueError)):
            inference_pipeline.predict(None)


class TestRealModelIntegration:
    """Test integration with real trained models if available."""
    
    @pytest.mark.skipif(
        not Path("models/trained").exists() or not list(Path("models/trained").glob("*.pth")),
        reason="No trained PyTorch models found"
    )
    def test_load_real_pytorch_model(self):
        """Test loading real PyTorch models if they exist."""
        models_dir = Path("models/trained")
        model_files = list(models_dir.glob("*.pth"))
        
        if model_files:
            model_file = model_files[0]
            print(f"Testing with real model: {model_file}")
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Basic validation
            assert isinstance(checkpoint, dict)
            
            if 'model_state_dict' in checkpoint:
                assert isinstance(checkpoint['model_state_dict'], dict)
                print(f"Model has {len(checkpoint['model_state_dict'])} parameters")
    
    @pytest.mark.skipif(
        not Path("models/trained").exists() or not list(Path("models/trained").glob("*.pkl")),
        reason="No trained pickle models found"
    )
    def test_load_real_pickle_model(self):
        """Test loading real pickled models if they exist."""
        models_dir = Path("models/trained")
        model_files = list(models_dir.glob("*.pkl"))

        if model_files:
            model_file = model_files[0]
            print(f"Testing with real model: {model_file}")

            # Prepare for unpickling SearchSpace if referenced under __main__
            import sys
            from src.utils.config import SearchSpace
            sys.modules['__main__'].SearchSpace = SearchSpace

            # Load model data
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            print(f"Loaded model type: {type(model_data)}")

            # If it's a results dictionary, check structure
            if isinstance(model_data, dict):
                print(f"Model data keys: {list(model_data.keys())}")
    
    def test_inference_with_real_data(self):
        """Test inference with real molecular data if available."""
        data_file = Path("data/raw/delaney-processed.csv")
        
        if data_file.exists():
            # Load real data
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} real molecules")
            
            # Take a small sample for testing
            sample_size = min(10, len(df))
            sample_data = df.sample(sample_size, random_state=42)
            
            # Test with mock model
            mock_model = MockMEGANModel()
            
            if 'smiles' in df.columns:
                smiles_list = sample_data['smiles'].tolist()
                predictions = mock_model.predict(smiles_list)
                
                assert len(predictions) == sample_size
                print(f"Generated predictions for {sample_size} real molecules")
                
                # Display sample results
                for i, (smiles, pred) in enumerate(zip(smiles_list[:3], predictions[:3])):
                    print(f"  {smiles}: {pred:.3f}")


class TestInferenceUtilities:
    """Test utility functions for inference."""
    
    def test_prediction_validation(self):
        """Test prediction validation utilities."""
        def validate_solubility_prediction(prediction):
            """Validate solubility prediction range."""
            return -15 <= prediction <= 5  # Reasonable range for log(mol/L)
        
        # Test valid predictions
        valid_preds = [-3.5, -1.2, 0.8, -7.1]
        assert all(validate_solubility_prediction(pred) for pred in valid_preds)
        
        # Test invalid predictions
        invalid_preds = [15.0, -20.0]
        assert not all(validate_solubility_prediction(pred) for pred in invalid_preds)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        def calculate_confidence(prediction, uncertainty):
            """Calculate confidence based on prediction and uncertainty."""
            # Higher uncertainty -> lower confidence
            confidence = 1.0 / (1.0 + uncertainty)
            return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Test with different uncertainty values
        pred = -3.0
        
        low_uncertainty = 0.1
        high_uncertainty = 2.0
        
        conf_low = calculate_confidence(pred, low_uncertainty)
        conf_high = calculate_confidence(pred, high_uncertainty)
        
        assert 0 <= conf_low <= 1
        assert 0 <= conf_high <= 1
        assert conf_low > conf_high  # Lower uncertainty should give higher confidence
    
    def test_result_formatting(self):
        """Test formatting of inference results."""
        def format_results(smiles_list, predictions, uncertainties=None):
            """Format inference results for output."""
            results = []
            for i, (smiles, pred) in enumerate(zip(smiles_list, predictions)):
                result = {
                    'smiles': smiles,
                    'predicted_log_solubility': round(float(pred), 3),
                    'predicted_solubility_mgL': round(10**(float(pred)) * 1000, 2),
                    'solubility_category': 'highly_soluble' if pred > -1 else 
                                         'moderately_soluble' if pred > -3 else 'poorly_soluble'
                }
                
                if uncertainties is not None:
                    result['uncertainty'] = round(float(uncertainties[i]), 3)
                    result['confidence'] = round(1.0 / (1.0 + uncertainties[i]), 3)
                
                results.append(result)
            
            return results
        
        # Test formatting
        smiles = ['CCO', 'c1ccccc1']
        predictions = [-0.5, -4.2]
        uncertainties = [0.2, 0.8]
        
        results = format_results(smiles, predictions, uncertainties)
        
        assert len(results) == 2
        assert all('smiles' in result for result in results)
        assert all('predicted_log_solubility' in result for result in results)
        assert all('solubility_category' in result for result in results)
        assert results[0]['solubility_category'] == 'highly_soluble'
        assert results[1]['solubility_category'] == 'poorly_soluble'


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
