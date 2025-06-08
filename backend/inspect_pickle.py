#!/usr/bin/env python3
"""
Deep inspection of the saved model to understand the MEGANConfig issue.
"""
import sys
import os
import pickle
import torch

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def inspect_model_pickle():
    """Inspect the model file to understand pickle dependencies."""
    model_path = r"c:\code\Drugzello\backend\models\trained\megan_pytorch_model.pth"
    
    print("Attempting to load model with pickle inspection...")
    
    class PickleInspector:
        def __init__(self):
            self.classes_needed = set()
        
        def find_class(self, module, name):
            self.classes_needed.add(f"{module}.{name}")
            print(f"Pickle needs: {module}.{name}")
            
            # Handle MEGANConfig specifically
            if name == "MEGANConfig":
                print(f"Found MEGANConfig in module: {module}")
                # Create a dummy class
                class MEGANConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                return MEGANConfig
            
            # For other classes, try to import normally
            try:
                module_obj = __import__(module, fromlist=[name])
                return getattr(module_obj, name)
            except (ImportError, AttributeError):
                print(f"Could not import {module}.{name}")
                # Return a dummy class
                class DummyClass:
                    def __init__(self, *args, **kwargs):
                        pass
                return DummyClass
      # Try to load with torch.load first to see the error
    print("Trying torch.load to see the specific error...")
    try:
        device = torch.device('cpu')
        data = torch.load(model_path, map_location=device, weights_only=False)
        print("Successfully loaded with torch.load!")
        print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return data
    except Exception as e:
        print(f"torch.load failed with: {e}")
        print("This confirms the MEGANConfig issue")
    
    # Create a custom unpickler class
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            inspector.classes_needed.add(f"{module}.{name}")
            print(f"Pickle needs: {module}.{name}")
            
            # Handle MEGANConfig specifically
            if name == "MEGANConfig":
                print(f"Found MEGANConfig in module: {module}")
                # Create a dummy class
                class MEGANConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                return MEGANConfig
            
            # For other classes, try to import normally
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                print(f"Could not import {module}.{name}")
                # Return a dummy class
                class DummyClass:
                    def __init__(self, *args, **kwargs):
                        pass
                return DummyClass
    
    # Try with custom unpickler
    print("Trying with custom unpickler...")
    with open(model_path, 'rb') as f:
        inspector = PickleInspector()
        unpickler = CustomUnpickler(f)
        
        try:
            data = unpickler.load()
            print("Successfully loaded with custom unpickler!")
            print(f"Classes needed: {inspector.classes_needed}")
            print(f"Data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return data
        except Exception as e:
            print(f"Failed even with custom unpickler: {e}")
            print(f"Classes that were needed: {inspector.classes_needed}")
            return None

if __name__ == "__main__":
    inspect_model_pickle()
