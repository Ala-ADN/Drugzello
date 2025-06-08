#!/usr/bin/env python3
"""
Examine the saved model structure to understand the original architecture.
"""
import sys
import os
import torch

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def examine_saved_model():
    """Examine the saved model to understand its structure."""
    model_path = r"c:\code\Drugzello\backend\models\trained\megan_pytorch_model.pth"
    
    # Create MEGANConfig in __main__ for loading
    import __main__
    if not hasattr(__main__, 'MEGANConfig'):
        class MEGANConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        __main__.MEGANConfig = MEGANConfig
    
    print("Loading saved model data...")
    device = torch.device('cpu')
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model_data = torch.load(model_path, map_location=device, weights_only=False)
    
    print("Model data keys:", list(model_data.keys()))
    
    # Examine the state dict structure
    state_dict = model_data['model_state_dict']
    print(f"\nModel state_dict has {len(state_dict)} parameters:")
    
    # Group parameters by layer type
    layer_groups = {}
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            layer_type = '.'.join(parts[:2])  # First two parts
        else:
            layer_type = parts[0]
        
        if layer_type not in layer_groups:
            layer_groups[layer_type] = []
        layer_groups[layer_type].append(key)
    
    print("\nParameters grouped by layer:")
    for layer_type, params in sorted(layer_groups.items()):
        print(f"\n{layer_type}:")
        for param in params:
            shape = state_dict[param].shape
            print(f"  {param}: {shape}")
    
    # Examine model kwargs and config
    print(f"\nModel kwargs: {model_data.get('model_kwargs', {})}")
    print(f"Config type: {type(model_data.get('config', {}))}")
    
    # Try to understand the attention layer structure
    print(f"\nAttention layer analysis:")
    attn_params = [k for k in state_dict.keys() if 'attn' in k or 'att' in k]
    for param in attn_params:
        shape = state_dict[param].shape
        print(f"  {param}: {shape}")

if __name__ == "__main__":
    examine_saved_model()
