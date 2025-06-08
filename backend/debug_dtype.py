#!/usr/bin/env python3
"""
Debug script to check data types in the dataset and model.
"""
import sys
from pathlib import Path
import torch

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import load_molecular_data

def debug_data_types():
    """Debug data types to find the dtype mismatch."""
    print("Debugging data types in ESOL dataset...")
    
    try:
        # Load the data the same way as training
        data_info = load_molecular_data(
            dataset_name='esol',
            data_root='data',
            split_type='kfold',
            n_folds=2,
            normalize_targets=True,
            random_state=42
        )
        
        dataset = data_info['dataset']
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset length: {len(dataset)}")
        
        # Check the first few samples for data types
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  x.dtype: {sample.x.dtype}, shape: {sample.x.shape}")
            print(f"  edge_index.dtype: {sample.edge_index.dtype}, shape: {sample.edge_index.shape}")
            if sample.edge_attr is not None:
                print(f"  edge_attr.dtype: {sample.edge_attr.dtype}, shape: {sample.edge_attr.shape}")
            else:
                print(f"  edge_attr: None")
            print(f"  y.dtype: {sample.y.dtype}, shape: {sample.y.shape}")
            
            # Check for any unusual values
            if torch.isnan(sample.x).any():
                print(f"  WARNING: NaN values in x")
            if torch.isnan(sample.y).any():
                print(f"  WARNING: NaN values in y")
                
        # Check if edge_index has the right type (should be long/int64)
        sample = dataset[0]
        print(f"\nEdge index analysis:")
        print(f"  edge_index.dtype: {sample.edge_index.dtype}")
        print(f"  edge_index min/max: {sample.edge_index.min()}/{sample.edge_index.max()}")
        
        return dataset
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_data_types()
