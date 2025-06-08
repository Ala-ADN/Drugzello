#!/usr/bin/env python3
"""
Test script to verify the MEGAN model loading functionality.
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

def test_model_loading():
    """Test the model loading functionality."""
    try:
        logger.info("Testing model loader import...")
        from core.model_loader import model_loader, MEGAN_AVAILABLE
        
        logger.info(f"MEGAN classes available: {MEGAN_AVAILABLE}")
        
        logger.info("Testing model loading...")
        result = model_loader.load_model()
        
        logger.info(f"Model loaded successfully: {result}")
        logger.info(f"Model version: {model_loader.get_version()}")
        logger.info(f"Model is loaded: {model_loader.is_loaded()}")
        
        # Test prediction
        if model_loader.is_loaded():
            model = model_loader.get_model()
            test_smiles = "CCO"  # Ethanol
            logger.info(f"Testing prediction with SMILES: {test_smiles}")
            prediction = model.predict(test_smiles)
            logger.info(f"Prediction result: {prediction}")
        
        logger.info("All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
