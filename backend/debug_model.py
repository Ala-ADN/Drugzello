#!/usr/bin/env python3
"""
Debug script to examine the model file and understand the MEGANConfig issue.
"""
import sys
import os
import torch
import pickle
import logging

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_model_file():
    """Debug the model file to understand what's causing the MEGANConfig issue."""
    model_path = r"c:\code\Drugzello\backend\models\trained\megan_pytorch_model.pth"
    
    try:
        logger.info(f"Examining model file: {model_path}")
        
        # Try to load with weights_only=True to avoid pickle issues
        try:
            model_data = torch.load(model_path, map_location='cpu', weights_only=True)
            logger.info("Successfully loaded with weights_only=True")
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}")
            logger.info("Trying with weights_only=False...")
            
            # Create a minimal MEGANConfig class to satisfy pickle
            class MEGANConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            # Make it available in the global namespace for pickle
            globals()['MEGANConfig'] = MEGANConfig
            
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info("Successfully loaded with weights_only=False and MEGANConfig stub")
        
        # Examine the contents
        logger.info(f"Model data keys: {list(model_data.keys())}")
        
        for key, value in model_data.items():
            if key == 'model_state_dict':
                logger.info(f"{key}: {len(value)} parameters")
                # Show a few parameter names
                param_names = list(value.keys())[:5]
                logger.info(f"  Sample parameters: {param_names}")
            elif hasattr(value, '__len__'):
                logger.info(f"{key}: {type(value).__name__} with length {len(value)}")
            else:
                logger.info(f"{key}: {type(value).__name__} = {value}")
        
        # Extract model_kwargs if available
        if 'model_kwargs' in model_data:
            logger.info(f"Model kwargs: {model_data['model_kwargs']}")
        
        # Extract config if available
        if 'config' in model_data:
            logger.info(f"Config: {model_data['config']}")
            logger.info(f"Config type: {type(model_data['config'])}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Failed to debug model file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_model_file()
