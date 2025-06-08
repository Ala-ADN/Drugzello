#!/usr/bin/env python3
"""
Test script to verify the newly trained MEGAN model loading functionality.
"""
import sys
import os
import logging

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_new_model_loading():
    """Test loading the newly trained model."""
    try:
        logger.info("Testing newly trained model loading...")
        
        # Import the model loader
        from core.model_loader import model_loader
        
        # Test loading with different model files
        model_files = [
            "models/trained/best_model.pth",
            "models/trained/fold_0_model.pth", 
            "models/trained/fold_1_model.pth",
            "models/trained/megan_pytorch_model.pth"
        ]
        
        for model_file in model_files:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing model: {model_file}")
            logger.info(f"{'='*50}")
            
            full_path = os.path.join(backend_dir, model_file)
            if os.path.exists(full_path):
                # Create a fresh model loader for each test
                from core.model_loader import ModelLoader
                loader = ModelLoader()
                
                # Try loading the model
                success = loader.load_model(full_path)
                logger.info(f"Model loaded successfully: {success}")
                
                if success:
                    logger.info(f"Model version: {loader.get_version()}")
                    logger.info(f"Model is loaded: {loader.is_loaded()}")
                    
                    # Get the model object
                    model = loader.get_model()
                    logger.info(f"Model type: {type(model)}")
                    
                    # Test prediction
                    test_smiles = "CCO"  # Ethanol
                    logger.info(f"Testing prediction with SMILES: {test_smiles}")
                    result = model.predict(test_smiles)
                    logger.info(f"Prediction result: {result}")
                    
                    # Test with a more complex molecule
                    test_smiles2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
                    logger.info(f"Testing prediction with SMILES: {test_smiles2}")
                    result2 = model.predict(test_smiles2)
                    logger.info(f"Prediction result: {result2}")
                    
                    logger.info("✅ Model loading and prediction successful!")
                    return True
                else:
                    logger.warning(f"Failed to load model: {model_file}")
            else:
                logger.warning(f"Model file not found: {full_path}")
        
        logger.error("❌ No models could be loaded successfully")
        return False
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_model_loading()
    sys.exit(0 if success else 1)
