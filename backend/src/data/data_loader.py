"""
Data loading and preprocessing utilities for molecular solubility prediction.
"""

import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle


class MolecularDataLoader:
    """
    Data loader for molecular datasets with preprocessing and splitting capabilities.
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def load_esol_dataset(self, transform=None) -> Dataset:
        """
        Load ESOL (solubility) dataset from MoleculeNet.
        
        Args:
            transform: Optional transform to apply to the data
        
        Returns:
            PyTorch Geometric dataset
        """
        if transform is None:
            transform = ToUndirected()
        
        dataset = MoleculeNet(
            root=str(self.data_root / "ESOL"),
            name="ESOL",
            transform=transform
        )
        
        return dataset
    
    def load_custom_dataset(self, csv_path: str, smiles_col: str = 'smiles', 
                           target_col: str = 'solubility') -> Dataset:
        """
        Load custom molecular dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file with SMILES and target values
            smiles_col: Name of column containing SMILES strings
            target_col: Name of column containing target values
        
        Returns:
            PyTorch Geometric dataset
        """
        # This would require implementing SMILES to graph conversion
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError(
            "Custom dataset loading requires SMILES to graph conversion. "
            "This will be implemented in a future version."
        )
    
    def create_splits(self, dataset: Dataset, split_type: str = "random", 
                     test_size: float = 0.2, val_size: float = 0.1, 
                     n_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Create data splits for training, validation, and testing.
        
        Args:
            dataset: Dataset to split
            split_type: Type of split ('random', 'kfold', 'scaffold')
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            n_folds: Number of folds for k-fold cross-validation
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary containing split indices or fold information
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        if split_type == "random":
            return self._random_split(indices, test_size, val_size, random_state)
        elif split_type == "kfold":
            return self._kfold_split(indices, n_folds, random_state)
        elif split_type == "scaffold":
            # Scaffold splitting would require molecular scaffold computation
            raise NotImplementedError(
                "Scaffold splitting is not yet implemented. Use 'random' or 'kfold'."
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")
    
    def _random_split(self, indices: List[int], test_size: float, 
                     val_size: float, random_state: int) -> Dict[str, List[int]]:
        """Create random train/val/test split."""
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # Second split: separate validation from training
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_size_adjusted, 
                random_state=random_state + 1
            )
        else:
            train_idx = train_val_idx
            val_idx = []
        
        return {
            'split_type': 'random',
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx)
        }
    
    def _kfold_split(self, indices: List[int], n_folds: int, 
                    random_state: int) -> Dict[str, Any]:
        """Create k-fold cross-validation splits."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_splits = list(kf.split(indices))
        
        return {
            'split_type': 'kfold',
            'n_folds': n_folds,
            'fold_splits': fold_splits,
            'total_size': len(indices)
        }
    
    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Compute statistics for the dataset.
        
        Args:
            dataset: Dataset to analyze
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(dataset),
            'num_node_features': dataset.num_node_features,
            'num_edge_features': getattr(dataset, 'num_edge_features', 0),
        }
        
        # Collect target statistics
        targets = []
        node_counts = []
        edge_counts = []
        
        for i in range(min(len(dataset), 1000)):  # Sample first 1000 for efficiency
            data = dataset[i]
            targets.append(data.y.item() if data.y.dim() == 0 else data.y[0].item())
            node_counts.append(data.x.size(0))
            edge_counts.append(data.edge_index.size(1))
        
        targets = np.array(targets)
        node_counts = np.array(node_counts)
        edge_counts = np.array(edge_counts)
        
        stats.update({
            'target_mean': float(np.mean(targets)),
            'target_std': float(np.std(targets)),
            'target_min': float(np.min(targets)),
            'target_max': float(np.max(targets)),
            'avg_nodes_per_graph': float(np.mean(node_counts)),
            'avg_edges_per_graph': float(np.mean(edge_counts)),
            'max_nodes_per_graph': int(np.max(node_counts)),
            'max_edges_per_graph': int(np.max(edge_counts))
        })
        
        return stats
    
    def save_splits(self, splits: Dict[str, Any], save_path: str):
        """Save data splits to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(splits, f)
    
    def load_splits(self, save_path: str) -> Dict[str, Any]:
        """Load data splits from file."""
        with open(save_path, 'rb') as f:
            splits = pickle.load(f)
        return splits


class DataPreprocessor:
    """
    Preprocessing utilities for molecular graph data.
    """
    
    @staticmethod
    def normalize_targets(dataset: Dataset, method: str = 'standard') -> Tuple[Dataset, Dict]:
        """
        Normalize target values in the dataset.
        
        Args:
            dataset: Dataset with targets to normalize
            method: Normalization method ('standard', 'minmax', 'robust')
        
        Returns:
            Tuple of (normalized_dataset, normalization_params)
        """
        # Collect all targets
        targets = []
        for data in dataset:
            if data.y.dim() == 0:
                targets.append(data.y.item())
            else:
                targets.append(data.y[0].item())
        
        targets = np.array(targets)
        
        # Compute normalization parameters
        if method == 'standard':
            mean = np.mean(targets)
            std = np.std(targets)
            norm_params = {'method': 'standard', 'mean': mean, 'std': std}
            normalized_targets = (targets - mean) / std
        elif method == 'minmax':
            min_val = np.min(targets)
            max_val = np.max(targets)
            norm_params = {'method': 'minmax', 'min': min_val, 'max': max_val}
            normalized_targets = (targets - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = np.median(targets)
            mad = np.median(np.abs(targets - median))
            norm_params = {'method': 'robust', 'median': median, 'mad': mad}
            normalized_targets = (targets - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new dataset with normalized targets
        normalized_dataset = []
        for i, data in enumerate(dataset):
            new_data = data.clone()
            new_data.y = torch.tensor([normalized_targets[i]], dtype=torch.float)
            normalized_dataset.append(new_data)
        
        return normalized_dataset, norm_params
    
    @staticmethod
    def denormalize_predictions(predictions: np.ndarray, 
                              norm_params: Dict) -> np.ndarray:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions
            norm_params: Normalization parameters from normalize_targets
        
        Returns:
            Denormalized predictions
        """
        method = norm_params['method']
        
        if method == 'standard':
            return predictions * norm_params['std'] + norm_params['mean']
        elif method == 'minmax':
            return predictions * (norm_params['max'] - norm_params['min']) + norm_params['min']
        elif method == 'robust':
            return predictions * norm_params['mad'] + norm_params['median']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def add_node_features(dataset: Dataset, feature_type: str = 'degree') -> Dataset:
        """
        Add additional node features to the dataset.
        
        Args:
            dataset: Input dataset
            feature_type: Type of features to add ('degree', 'centrality')
        
        Returns:
            Dataset with additional node features
        """
        if feature_type == 'degree':
            return DataPreprocessor._add_degree_features(dataset)
        else:
            raise NotImplementedError(f"Feature type '{feature_type}' not implemented")
    
    @staticmethod
    def _add_degree_features(dataset: Dataset) -> Dataset:
        """Add node degree as an additional feature."""
        from torch_geometric.utils import degree
        
        augmented_dataset = []
        for data in dataset:
            # Compute node degrees
            edge_index = data.edge_index
            num_nodes = data.x.size(0)
            node_degrees = degree(edge_index[1], num_nodes, dtype=torch.float)
            
            # Concatenate degree to existing features
            new_x = torch.cat([data.x, node_degrees.unsqueeze(1)], dim=1)
            
            # Create new data object
            new_data = Data(
                x=new_x,
                edge_index=data.edge_index,
                edge_attr=getattr(data, 'edge_attr', None),
                y=data.y,
                **{k: v for k, v in data.items() if k not in ['x', 'edge_index', 'edge_attr', 'y']}
            )
            
            augmented_dataset.append(new_data)
        
        return augmented_dataset


# Convenience function for common data loading workflow
def load_molecular_data(dataset_name: str = "ESOL", data_root: str = "data", 
                       split_type: str = "kfold", n_folds: int = 5, 
                       normalize_targets: bool = True, 
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Complete data loading workflow for molecular datasets.
    
    Args:
        dataset_name: Name of dataset to load
        data_root: Root directory for data storage
        split_type: Type of data splitting
        n_folds: Number of folds for cross-validation
        normalize_targets: Whether to normalize target values
        random_state: Random seed
    
    Returns:
        Dictionary containing dataset, splits, and metadata
    """
    loader = MolecularDataLoader(data_root)
    
    # Load dataset
    if dataset_name.upper() == "ESOL":
        dataset = loader.load_esol_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get dataset statistics
    stats = loader.get_dataset_statistics(dataset)
    
    # Normalize targets if requested
    norm_params = None
    if normalize_targets:
        dataset, norm_params = DataPreprocessor.normalize_targets(dataset)
    
    # Create splits
    splits = loader.create_splits(
        dataset, split_type=split_type, n_folds=n_folds, random_state=random_state
    )
    
    return {
        'dataset': dataset,
        'splits': splits,
        'stats': stats,
        'norm_params': norm_params,
        'dataset_name': dataset_name,
        'random_state': random_state
    }
