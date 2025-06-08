#!/usr/bin/env python3
"""
Debug script to check what MoleculeNet returns for ESOL dataset.
"""
import sys
from pathlib import Path
import torch

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from torch_geometric.datasets import MoleculeNet
from torch_geometric.transforms import ToUndirected

def debug_dataset():
    """Debug the dataset loading to understand the structure."""
    print("Debugging ESOL dataset loading...")
    
    try:
        print("Loading dataset with MoleculeNet...")
        dataset = MoleculeNet(
            root="data/ESOL",
            name="ESOL",
            transform=ToUndirected()
        )
        
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset length: {len(dataset)}")
        
        if hasattr(dataset, 'num_node_features'):
            print(f"Dataset num_node_features: {dataset.num_node_features}")
        else:
            print("Dataset does not have num_node_features attribute")
        
        if hasattr(dataset, 'num_edge_features'):
            print(f"Dataset num_edge_features: {dataset.num_edge_features}")
        else:
            print("Dataset does not have num_edge_features attribute")
        
        # Check the first sample
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"First sample type: {type(first_sample)}")
            print(f"First sample: {first_sample}")
            
            if hasattr(first_sample, 'x'):
                print(f"Node features shape: {first_sample.x.shape}")
            if hasattr(first_sample, 'edge_attr'):
                print(f"Edge features shape: {first_sample.edge_attr.shape if first_sample.edge_attr is not None else None}")
            if hasattr(first_sample, 'y'):
                print(f"Target shape: {first_sample.y.shape if first_sample.y is not None else None}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    dataset = debug_dataset()
