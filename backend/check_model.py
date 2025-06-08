import torch
import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import the required classes
sys.path.append(str(Path(__file__).parent / 'src'))

# Import the required classes that were saved with the model
from src.utils.config import MEGANConfig
from src.models.megan_architecture import MEGANCore

# Load the model file
model_data = torch.load('models/trained/megan_pytorch_model.pth', map_location='cpu')

print("Model data keys:", list(model_data.keys()))

# Check if it has config and model_kwargs
if 'config' in model_data:
    print("\nConfig:", model_data['config'])

if 'model_kwargs' in model_data:
    print("\nModel kwargs:", model_data['model_kwargs'])

if 'model_state_dict' in model_data:
    print(f"\nState dict has {len(model_data['model_state_dict'])} parameters")
    # Show first few parameter names to understand the model structure
    param_names = list(model_data['model_state_dict'].keys())[:10]
    for name in param_names:
        shape = model_data['model_state_dict'][name].shape
        print(f"  {name}: {shape}")
