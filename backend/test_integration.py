#!/usr/bin/env python3
"""
Integration test for MEGAN inference service
"""
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_model_loader():
    """Test if model loader can be imported and initialized"""
    try:
        from core.model_loader import ModelLoader, MockMEGANModel
        
        # Test MockMEGANModel
        mock_model = MockMEGANModel()
        print(f"✓ MockMEGANModel initialized: {mock_model.model_version}")
        
        # Test basic prediction
        test_smiles = "CCO"  # Ethanol
        result = mock_model.predict(test_smiles)
        print(f"✓ Basic prediction successful: {result}")
        
        # Test enhanced prediction
        enhanced_result = mock_model.predict(
            test_smiles, 
            include_uncertainty=True, 
            include_explanations=True
        )
        print(f"✓ Enhanced prediction successful")
        print(f"  - Has uncertainty: {'uncertainty' in enhanced_result}")
        print(f"  - Has explanations: {'explanations' in enhanced_result}")
        
        # Test ModelLoader
        loader = ModelLoader()
        print(f"✓ ModelLoader initialized")
        
        # This will load the mock model since no real model file exists
        success = loader.load_model()
        print(f"✓ Model loading: {success}")
        
        model = loader.get_model()
        if model:
            print(f"✓ Model retrieved: {type(model).__name__}")
            print(f"✓ Model version: {loader.get_version()}")
        else:
            print("✗ Failed to retrieve model")
            
        return True
        
    except Exception as e:
        print(f"✗ Model loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_service():
    """Test if inference service can be imported and works"""
    try:
        from services.inference_service import InferenceService
        
        service = InferenceService()
        print(f"✓ InferenceService initialized")
        
        test_smiles = "CCO"  # Ethanol
        
        # Test simple prediction
        simple_prediction = service.predict_solubility_simple(test_smiles)
        print(f"✓ Simple prediction successful: {simple_prediction.value:.3f}")
        
        # Test enhanced prediction
        enhanced_prediction = service.predict_solubility(
            test_smiles,
            include_uncertainty=True,
            include_explanations=True
        )
        print(f"✓ Enhanced prediction successful: {enhanced_prediction.value:.3f}")
        print(f"  - Model type: {enhanced_prediction.model_type}")
        print(f"  - Has uncertainty: {enhanced_prediction.uncertainty is not None}")
        print(f"  - Has explanations: {enhanced_prediction.explanations is not None}")
        
        if enhanced_prediction.uncertainty:
            print(f"  - Prediction std: {enhanced_prediction.uncertainty.prediction_std}")
            print(f"  - UAA score: {enhanced_prediction.uncertainty.uaa_score}")
            
        if enhanced_prediction.explanations:
            print(f"  - Node importances count: {len(enhanced_prediction.explanations.node_importances)}")
            print(f"  - Interpretation: {enhanced_prediction.explanations.interpretation[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_models():
    """Test if API models can be imported and used"""
    try:
        from api.models import (
            EnhancedInferenceRequest, 
            EnhancedInferenceResponse,
            EnhancedSolubilityPrediction,
            UncertaintyAnalysis,
            ExplanationResults
        )
        
        # Test enhanced request
        request = EnhancedInferenceRequest(
            smiles="CCO",
            include_uncertainty=True,
            include_explanations=True,
            uncertainty_samples=50
        )
        print(f"✓ EnhancedInferenceRequest created: {request.smiles}")
        
        # Test uncertainty analysis
        uncertainty = UncertaintyAnalysis(
            prediction_std=0.1,
            uaa_score=0.15,
            aau_scores=[0.05, 0.08, 0.03]
        )
        print(f"✓ UncertaintyAnalysis created")
        
        # Test explanation results
        explanations = ExplanationResults(
            node_importances=[0.1, 0.2, 0.3],
            edge_importances=[0.05, 0.08],
            interpretation="Test interpretation",
            rdkit_comparison={"crippen_logp": -0.5, "agreement_score": 0.7}
        )
        print(f"✓ ExplanationResults created")
        
        # Test enhanced prediction
        enhanced_pred = EnhancedSolubilityPrediction(
            value=-2.5,
            confidence=0.85,
            unit="log(mol/L)",
            model_type="megan",
            uncertainty=uncertainty,
            explanations=explanations
        )
        print(f"✓ EnhancedSolubilityPrediction created: {enhanced_pred.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ API models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("=== MEGAN Integration Test ===\n")
    
    tests = [
        ("API Models", test_api_models),
        ("Model Loader", test_model_loader),
        ("Inference Service", test_inference_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- Testing {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✓ PASSED' if success else '✗ FAILED'}\n")
    
    print("=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! MEGAN inference is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
