#!/usr/bin/env python3
"""
Script to examine the saved MEGAN model structure
"""

import torch
import pickle
import sys
from pathlib import Path

def examine_model():
    """Examine the saved MEGAN model and training artifacts"""
    
    print("=" * 60)
    print("EXAMINING SAVED MEGAN MODEL")
    print("=" * 60)
    
    # Check PyTorch model file
    model_path = Path("models/trained/megan_pytorch_model.pth")
    if model_path.exists():
        print(f"\n1. PyTorch Model File: {model_path}")
        print("-" * 40)
        try:
            model_data = torch.load(model_path, map_location='cpu')
            print(f"Model data type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print("Model data keys:")
                for key in model_data.keys():
                    print(f"  - {key}: {type(model_data[key])}")
                    
                    # Show additional details for specific keys
                    if key == 'config' and isinstance(model_data[key], dict):
                        print("    Config details:")
                        for config_key, config_value in model_data[key].items():
                            print(f"      {config_key}: {config_value}")
                    
                    elif key == 'model_kwargs' and isinstance(model_data[key], dict):
                        print("    Model kwargs:")
                        for kwargs_key, kwargs_value in model_data[key].items():
                            print(f"      {kwargs_key}: {kwargs_value}")
                            
                    elif key == 'model_state_dict' and isinstance(model_data[key], dict):
                        print(f"    State dict has {len(model_data[key])} parameters")
                        # Show first few parameter names
                        param_names = list(model_data[key].keys())[:5]
                        for param_name in param_names:
                            print(f"      {param_name}: {model_data[key][param_name].shape}")
                        if len(model_data[key]) > 5:
                            print(f"      ... and {len(model_data[key]) - 5} more parameters")
            else:
                print(f"Model data is not a dict: {model_data}")
                
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
    else:
        print(f"PyTorch model file not found: {model_path}")
    
    # Check training results
    training_results_path = Path("models/trained/training_results.pkl")
    if training_results_path.exists():
        print(f"\n2. Training Results: {training_results_path}")
        print("-" * 40)
        try:
            # Try to load without unpickling the complex objects
            import pickle
            
            # Read raw bytes first to see the structure
            with open(training_results_path, 'rb') as f:
                try:
                    # Add the current directory to path to help with imports
                    sys.path.insert(0, str(Path.cwd()))
                    training_results = pickle.load(f)
                    print(f"Training results type: {type(training_results)}")
                    
                    if isinstance(training_results, (list, tuple)):
                        print(f"Training results length: {len(training_results)}")
                        if len(training_results) > 0:
                            print(f"First item type: {type(training_results[0])}")
                            if isinstance(training_results[0], dict):
                                print("First item keys:")
                                for key in training_results[0].keys():
                                    print(f"  - {key}")
                    
                    elif isinstance(training_results, dict):
                        print("Training results keys:")
                        for key in training_results.keys():
                            print(f"  - {key}: {type(training_results[key])}")
                            
                except Exception as e:
                    print(f"Could not unpickle training results: {e}")
                    
        except Exception as e:
            print(f"Error reading training results: {e}")
    else:
        print(f"Training results file not found: {training_results_path}")
    
    # Check hyperparameter results
    hyperparam_path = Path("models/trained/hyperparam_results.pkl")
    if hyperparam_path.exists():
        print(f"\n3. Hyperparameter Results: {hyperparam_path}")
        print("-" * 40)
        try:
            with open(hyperparam_path, 'rb') as f:
                hyperparam_results = pickle.load(f)
                print(f"Hyperparam results type: {type(hyperparam_results)}")
                
                if isinstance(hyperparam_results, dict):
                    print("Hyperparam results keys:")
                    for key in hyperparam_results.keys():
                        print(f"  - {key}: {type(hyperparam_results[key])}")
                        
        except Exception as e:
            print(f"Error loading hyperparam results: {e}")
    else:
        print(f"Hyperparameter results file not found: {hyperparam_path}")

if __name__ == "__main__":
    examine_model()
