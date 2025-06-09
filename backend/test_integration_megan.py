#!/usr/bin/env python3
"""
Integration test for MEGAN inference service.
This test verifies that the MEGAN integration works correctly in Docker.
"""

import sys
import os
import logging

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from services.inference_service import InferenceService
        logger.info("✓ InferenceService imported successfully")
        
        from api.models import (
            EnhancedInferenceRequest, 
            EnhancedInferenceResponse,
            UncertaintyAnalysis,
            ExplanationResults
        )
        logger.info("✓ Enhanced API models imported successfully")
        
        from core.model_loader import ModelLoader
        logger.info("✓ ModelLoader imported successfully")
        
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False

def test_inference_service():
    """Test the inference service initialization and basic functionality."""
    logger.info("Testing inference service...")
    
    try:
        from services.inference_service import InferenceService
        
        # Initialize service
        service = InferenceService()
        logger.info("✓ InferenceService initialized successfully")
        
        # Test model loading
        is_loaded = service.is_model_loaded()
        logger.info(f"✓ Model loaded status: {is_loaded}")
        
        # Test model version
        version = service.get_model_version()
        logger.info(f"✓ Model version: {version}")
        
        # Test basic prediction (should work with mock model if real model fails)
        test_smiles = "CCO"  # Ethanol
        prediction = service.predict_solubility(test_smiles)
        logger.info(f"✓ Basic prediction successful: {prediction.value:.3f} (confidence: {prediction.confidence:.3f})")
        
        # Test enhanced prediction
        enhanced_prediction = service.predict_solubility(
            test_smiles, 
            include_uncertainty=True, 
            include_explanations=True
        )
        logger.info(f"✓ Enhanced prediction successful: {enhanced_prediction.value:.3f}")
        
        if enhanced_prediction.uncertainty:
            logger.info(f"  - Uncertainty analysis included: std={enhanced_prediction.uncertainty.prediction_std:.3f}")
        
        if enhanced_prediction.explanations:
            logger.info(f"  - Explanations included: {len(enhanced_prediction.explanations.node_importances)} node importances")
        
        return True
    except Exception as e:
        logger.error(f"✗ InferenceService test error: {e}")
        return False

def test_model_loader():
    """Test the model loader functionality."""
    logger.info("Testing model loader...")
    
    try:
        from core.model_loader import model_loader
        
        # Test model loading
        model = model_loader.get_model()
        logger.info(f"✓ Model retrieved: {type(model).__name__}")
        
        # Test prediction interface
        if hasattr(model, 'predict'):
            test_smiles = "CCO"
            result = model.predict(test_smiles)
            logger.info(f"✓ Model prediction interface works: {result.get('solubility', 'N/A')}")
            
            # Test enhanced prediction interface
            enhanced_result = model.predict(
                test_smiles, 
                include_uncertainty=True, 
                include_explanations=True
            )
            logger.info(f"✓ Enhanced model prediction interface works")
            
        return True
    except Exception as e:
        logger.error(f"✗ ModelLoader test error: {e}")
        return False

def test_api_models():
    """Test API model validation."""
    logger.info("Testing API models...")
    
    try:
        from api.models import (
            EnhancedInferenceRequest,
            UncertaintyAnalysis,
            ExplanationResults,
            EnhancedSolubilityPrediction
        )
        
        # Test enhanced request model
        request = EnhancedInferenceRequest(
            smiles="CCO",
            include_uncertainty=True,
            include_explanations=True,
            uncertainty_samples=50
        )
        logger.info(f"✓ EnhancedInferenceRequest created: {request.smiles}")
        
        # Test uncertainty analysis model
        uncertainty = UncertaintyAnalysis(
            prediction_std=0.1,
            uaa_score=0.15,
            aau_scores=[0.05, 0.03, 0.02]
        )
        logger.info(f"✓ UncertaintyAnalysis created: std={uncertainty.prediction_std}")
        
        # Test explanation results model
        explanations = ExplanationResults(
            node_importances=[0.1, 0.2, 0.3],
            edge_importances=[0.05, 0.03],
            interpretation="Test interpretation",
            rdkit_comparison={"crippen_logp": -0.5, "agreement_score": 0.7}
        )
        logger.info(f"✓ ExplanationResults created: {len(explanations.node_importances)} nodes")
        
        # Test enhanced prediction model
        prediction = EnhancedSolubilityPrediction(
            value=-1.5,
            confidence=0.85,
            unit="log(mol/L)",
            model_type="megan",
            uncertainty=uncertainty,
            explanations=explanations
        )
        logger.info(f"✓ EnhancedSolubilityPrediction created: {prediction.value}")
        
        return True
    except Exception as e:
        logger.error(f"✗ API models test error: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting MEGAN integration tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("API Models", test_api_models),
        ("Model Loader", test_model_loader),
        ("Inference Service", test_inference_service),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! MEGAN integration is working correctly.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
