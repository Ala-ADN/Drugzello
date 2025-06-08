"""
Enhanced data loading and preprocessing utilities for molecular solubility prediction.
Includes multi-solvent support and duplicate handling.
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
import hashlib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class MultiSolventDataset(Dataset):
    """Custom dataset class for multi-solvent molecular data."""
    
    def __init__(self, data_list: List[Data], dataset_info: List[Dict], solvent_vocab: Dict[str, int]):
        super().__init__()
        self.data_list = data_list
        self.dataset_info = dataset_info
        self.solvent_vocab = solvent_vocab
        
        # Calculate features from first sample
        if data_list:
            self.num_node_features = data_list[0].x.size(1)
            self.num_edge_features = data_list[0].edge_attr.size(1) if data_list[0].edge_attr is not None else 0
        
        # Store reverse vocab for lookup
        self.id_to_solvent = {v: k for k, v in solvent_vocab.items()}
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def __iter__(self):
        return iter(self.data_list)
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics broken down by solvent and source dataset."""
        stats = {'by_solvent': {}, 'by_dataset': {}, 'overall': {}}
        
        # Group by solvent
        solvent_groups = {}
        dataset_groups = {}
        
        for data in self.data_list:
            solvent = data.solvent_name
            dataset = data.source_dataset
            target = data.y.item() if data.y.dim() == 0 else data.y[0].item()
            
            if solvent not in solvent_groups:
                solvent_groups[solvent] = []
            solvent_groups[solvent].append(target)
            
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(target)
        
        # Calculate statistics
        for solvent, targets in solvent_groups.items():
            targets = np.array(targets)
            stats['by_solvent'][solvent] = {
                'count': len(targets),
                'mean': float(np.mean(targets)),
                'std': float(np.std(targets)),
                'min': float(np.min(targets)),
                'max': float(np.max(targets))
            }
        
        for dataset, targets in dataset_groups.items():
            targets = np.array(targets)
            stats['by_dataset'][dataset] = {
                'count': len(targets),
                'mean': float(np.mean(targets)),
                'std': float(np.std(targets)),
                'min': float(np.min(targets)),
                'max': float(np.max(targets))
            }
        
        # Overall stats
        all_targets = []
        for data in self.data_list:
            target = data.y.item() if data.y.dim() == 0 else data.y[0].item()
            all_targets.append(target)
        
        all_targets = np.array(all_targets)
        stats['overall'] = {
            'total_count': len(all_targets),
            'num_solvents': len(self.solvent_vocab),
            'num_datasets': len(set(data.source_dataset for data in self.data_list)),
            'mean': float(np.mean(all_targets)),
            'std': float(np.std(all_targets))
        }
        
        return stats


class MolecularDataLoader:
    """
    Enhanced data loader for molecular datasets with preprocessing, multi-solvent support, 
    and duplicate handling capabilities.
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
    
    def load_moleculenet_datasets_by_solvent(self, 
                                            dataset_configs: List[Dict[str, str]],
                                            handle_duplicates: str = 'remove') -> Tuple[Dataset, Dict[str, int], Dict[str, Any]]:
        """
        Load and concatenate multiple MoleculeNet datasets with duplicate handling.
        
        Args:
            dataset_configs: List of dictionaries with 'name' and 'solvent' keys
                Example: [
                    {'name': 'ESOL', 'solvent': 'water'},
                    {'name': 'FreeSolv', 'solvent': 'water'},
                    {'name': 'Lipo', 'solvent': 'octanol'}
                ]
            handle_duplicates: How to handle duplicates ('remove', 'keep_first', 'keep_last', 'average')
        
        Returns:
            Tuple of (concatenated_dataset, solvent_vocabulary, duplicate_info)
        """
        all_data = []
        solvent_vocab = {}
        dataset_info = []
        duplicate_tracker = {}  # canonical_smiles -> list of (data, metadata)
        duplicate_info = {'method': handle_duplicates, 'duplicates_found': 0, 'duplicates_removed': 0}
        
        for config in dataset_configs:
            dataset_name = config['name']
            solvent_name = config['solvent']
            
            # Add to solvent vocabulary
            if solvent_name not in solvent_vocab:
                solvent_vocab[solvent_name] = len(solvent_vocab)
            
            # Load MoleculeNet dataset
            dataset = MoleculeNet(
                root=str(self.data_root / dataset_name),
                name=dataset_name,
                transform=ToUndirected()
            )
            
            # Process each data point and check for duplicates
            for data in dataset:
                # Get canonical SMILES for duplicate detection
                canonical_smiles = self._get_canonical_smiles(data)
                
                if canonical_smiles is None:
                    continue  # Skip invalid molecules
                
                # Create enhanced data object
                new_data = data.clone()
                new_data.solvent_id = torch.tensor([solvent_vocab[solvent_name]], dtype=torch.long)
                new_data.solvent_name = solvent_name
                new_data.source_dataset = dataset_name
                new_data.canonical_smiles = canonical_smiles
                
                metadata = {
                    'dataset': dataset_name,
                    'solvent': solvent_name,
                    'original_index': len(all_data)
                }
                
                # Track duplicates
                if canonical_smiles in duplicate_tracker:
                    duplicate_tracker[canonical_smiles].append((new_data, metadata))
                    duplicate_info['duplicates_found'] += 1
                else:
                    duplicate_tracker[canonical_smiles] = [(new_data, metadata)]
        
        # Handle duplicates according to strategy
        final_data = self._resolve_duplicates(duplicate_tracker, handle_duplicates, duplicate_info)
        
        # Create dataset info
        for config in dataset_configs:
            count = sum(1 for data in final_data if data.source_dataset == config['name'])
            dataset_info.append({
                'dataset': config['name'],
                'solvent': config['solvent'],
                'num_samples': count,
                'solvent_id': solvent_vocab[config['solvent']]
            })
        
        concatenated_dataset = MultiSolventDataset(final_data, dataset_info, solvent_vocab)
        
        return concatenated_dataset, solvent_vocab, duplicate_info
    
    def load_compatible_moleculenet_datasets(self) -> Dict[str, List[str]]:
        """
        Return compatible MoleculeNet datasets grouped by typical solvent systems.
        
        Returns:
            Dictionary mapping solvents to compatible datasets
        """
        compatible_datasets = {
            'water': [
                'ESOL',      # Aqueous solubility
                'FreeSolv',  # Hydration free energy (water-based)
                'BBBP',      # Blood-brain barrier (aqueous)
            ],
            'octanol': [
                'Lipo',      # Lipophilicity (octanol/water partition)
            ],
            'mixed_aqueous': [
                'HIV',       # Biological assays (aqueous conditions)
                'BACE',      # Enzyme inhibition (aqueous)
                'ToxCast',   # Toxicity (biological/aqueous)
            ]
        }
        return compatible_datasets
    
    def _get_canonical_smiles(self, data: Data) -> Optional[str]:
        """
        Extract canonical SMILES from molecular data.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            # Try to get SMILES from data attributes
            if hasattr(data, 'smiles'):
                mol = Chem.MolFromSmiles(data.smiles)
            elif hasattr(data, 'mol'):
                mol = data.mol
            else:
                # Reconstruct molecule from graph (more complex, not always accurate)
                return None
            
            if mol is None:
                return None
                
            # Generate canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            return canonical_smiles
            
        except Exception:
            return None
    
    def _resolve_duplicates(self, duplicate_tracker: Dict[str, List], 
                          strategy: str, duplicate_info: Dict) -> List[Data]:
        """
        Resolve duplicates according to specified strategy.
        
        Args:
            duplicate_tracker: Dictionary mapping canonical SMILES to list of (data, metadata)
            strategy: How to handle duplicates
            duplicate_info: Dictionary to update with duplicate statistics
            
        Returns:
            List of data objects with duplicates resolved
        """
        final_data = []
        
        for canonical_smiles, data_list in duplicate_tracker.items():
            if len(data_list) == 1:
                # No duplicates
                final_data.append(data_list[0][0])
            else:
                # Handle duplicates
                if strategy == 'remove':
                    # Skip all duplicates
                    duplicate_info['duplicates_removed'] += len(data_list)
                    continue
                    
                elif strategy == 'keep_first':
                    final_data.append(data_list[0][0])
                    duplicate_info['duplicates_removed'] += len(data_list) - 1
                    
                elif strategy == 'keep_last':
                    final_data.append(data_list[-1][0])
                    duplicate_info['duplicates_removed'] += len(data_list) - 1
                    
                elif strategy == 'average':
                    # Average target values for same molecule in different solvents
                    averaged_data = self._average_duplicate_targets(data_list)
                    if averaged_data:
                        final_data.append(averaged_data)
                    duplicate_info['duplicates_removed'] += len(data_list) - 1
                    
                else:
                    raise ValueError(f"Unknown duplicate handling strategy: {strategy}")
        
        return final_data
    
    def _average_duplicate_targets(self, data_list: List[Tuple[Data, Dict]]) -> Optional[Data]:
        """
        Average target values for duplicate molecules.
        
        Args:
            data_list: List of (data, metadata) tuples for the same molecule
            
        Returns:
            Data object with averaged targets, or None if incompatible
        """
        # Check if all duplicates are for the same solvent
        solvents = set(data.solvent_name for data, _ in data_list)
        
        if len(solvents) == 1:
            # Same molecule, same solvent - average the targets
            targets = []
            for data, _ in data_list:
                target = data.y.item() if data.y.dim() == 0 else data.y[0].item()
                targets.append(target)
            
            avg_target = np.mean(targets)
            
            # Use first data object as template
            base_data, _ = data_list[0]
            new_data = base_data.clone()
            new_data.y = torch.tensor([avg_target], dtype=torch.float)
            new_data.num_duplicates = len(data_list)
            
            return new_data
        else:
            # Different solvents - keep all (this shouldn't happen with proper grouping)
            return data_list[0][0]
    
    def load_custom_dataset(self, csv_path: str, smiles_col: str = 'smiles', 
                           target_col: str = 'solubility', solvent_col: str = None) -> Dataset:
        """
        Load custom molecular dataset from CSV file with optional solvent information.
        
        Args:
            csv_path: Path to CSV file with SMILES and target values
            smiles_col: Name of column containing SMILES strings
            target_col: Name of column containing target values
            solvent_col: Name of column containing solvent information (optional)
        
        Returns:
            PyTorch Geometric dataset
        """
        df = pd.read_csv(csv_path)
        data_list = []
        
        for _, row in df.iterrows():
            # Convert SMILES to molecular graph
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                continue
                
            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic())
                ]
                atom_features.append(features)
            
            # Extract bond information
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bond_type = bond.GetBondType()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                bond_features = [int(bond_type), int(bond.GetIsAromatic())]
                edge_features.extend([bond_features, bond_features])
            
            # Create PyG Data object
            data = Data(
                x=torch.tensor(atom_features, dtype=torch.float),
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float) if edge_features else None,
                y=torch.tensor([row[target_col]], dtype=torch.float),
                smiles=row[smiles_col]
            )
            
            # Add solvent information if available
            if solvent_col and solvent_col in row:
                data.solvent = row[solvent_col]
                
            data_list.append(data)
        
        return CustomDataset(data_list)
    
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
            'num_node_features': getattr(dataset, 'num_node_features', 0),
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


class CustomDataset(Dataset):
    """Dataset wrapper for custom molecular data."""
    
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
        # Calculate features from first sample
        if data_list:
            self.num_node_features = data_list[0].x.size(1)
            self.num_edge_features = data_list[0].edge_attr.size(1) if data_list[0].edge_attr is not None else 0
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


class DataPreprocessor:
    """
    Enhanced preprocessing utilities for molecular graph data.
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
        normalized_data_list = []
        for i, data in enumerate(dataset):
            new_data = data.clone()
            new_data.y = torch.tensor([normalized_targets[i]], dtype=torch.float)
            normalized_data_list.append(new_data)
        
        # Create a new dataset object that maintains the same attributes
        class NormalizedDataset:
            def __init__(self, data_list, original_dataset):
                self.data_list = data_list
                self.num_node_features = getattr(original_dataset, 'num_node_features', 0)
                self.num_edge_features = getattr(original_dataset, 'num_edge_features', 0)
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
            
            def __iter__(self):
                return iter(self.data_list)
        
        normalized_dataset = NormalizedDataset(normalized_data_list, dataset)
        
        return normalized_dataset, norm_params
    
    @staticmethod
    def normalize_targets_by_solvent(dataset: MultiSolventDataset, 
                                   method: str = 'standard') -> Tuple[Dataset, Dict]:
        """
        Normalize targets separately for each solvent to account for different scales.
        
        Args:
            dataset: Multi-solvent dataset
            method: Normalization method
        
        Returns:
            Tuple of (normalized_dataset, normalization_params)
        """
        # Group data by solvent
        solvent_groups = {}
        for i, data in enumerate(dataset):
            solvent = data.solvent_name
            if solvent not in solvent_groups:
                solvent_groups[solvent] = []
            solvent_groups[solvent].append((i, data))
        
        # Calculate normalization parameters for each solvent
        norm_params = {'method': method, 'by_solvent': {}}
        normalized_data_list = [None] * len(dataset)
        
        for solvent, data_list in solvent_groups.items():
            targets = []
            indices = []
            
            for idx, data in data_list:
                target = data.y.item() if data.y.dim() == 0 else data.y[0].item()
                targets.append(target)
                indices.append(idx)
            
            targets = np.array(targets)
            
            # Calculate normalization for this solvent
            if method == 'standard':
                mean = np.mean(targets)
                std = np.std(targets)
                norm_params['by_solvent'][solvent] = {'mean': mean, 'std': std}
                normalized_targets = (targets - mean) / std
            elif method == 'minmax':
                min_val = np.min(targets)
                max_val = np.max(targets)
                norm_params['by_solvent'][solvent] = {'min': min_val, 'max': max_val}
                normalized_targets = (targets - min_val) / (max_val - min_val)
            
            # Apply normalization to data
            for i, (idx, data) in enumerate(data_list):
                new_data = data.clone()
                new_data.y = torch.tensor([normalized_targets[i]], dtype=torch.float)
                normalized_data_list[idx] = new_data
        
        # Create normalized dataset
        normalized_dataset = MultiSolventDataset(
            normalized_data_list, 
            dataset.dataset_info, 
            dataset.solvent_vocab
        )
        
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
    def detect_duplicates_in_dataset(dataset: Dataset, 
                                   similarity_threshold: float = 1.0) -> Dict[str, Any]:
        """
        Detect potential duplicates in a dataset.
        
        Args:
            dataset: Dataset to analyze
            similarity_threshold: Threshold for considering molecules as duplicates (1.0 = exact match)
            
        Returns:
            Dictionary with duplicate analysis results
        """
        fingerprints = {}
        smiles_groups = {}
        duplicates = []
        
        for i, data in enumerate(dataset):
            if hasattr(data, 'canonical_smiles'):
                smiles = data.canonical_smiles
                
                if smiles in smiles_groups:
                    smiles_groups[smiles].append(i)
                else:
                    smiles_groups[smiles] = [i]
        
        # Find exact duplicates
        exact_duplicates = {smiles: indices for smiles, indices in smiles_groups.items() 
                           if len(indices) > 1}
        
        duplicate_stats = {
            'total_molecules': len(dataset),
            'unique_molecules': len(smiles_groups),
            'exact_duplicates': len(exact_duplicates),
            'duplicate_groups': exact_duplicates,
            'duplicate_molecules': sum(len(indices) - 1 for indices in exact_duplicates.values())
        }
        
        return duplicate_stats
    
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
    
    @staticmethod
    def convert_to_float(dataset: Dataset) -> Dataset:
        """
        Convert integer node and edge features to float32 for PyTorch compatibility.
        
        Args:
            dataset: Dataset with potentially integer features
            
        Returns:
            Dataset with float32 features
        """
        from torch_geometric.data import Data
        
        converted_data_list = []
        for data in dataset:
            # Convert node features to float32
            x = data.x.float() if data.x.dtype != torch.float32 else data.x
            
            # Convert edge features to float32 if they exist
            edge_attr = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr.float() if data.edge_attr.dtype != torch.float32 else data.edge_attr
            
            # Keep edge_index as int64 (this is correct for indices)
            edge_index = data.edge_index
            
            # Keep targets as they are (should already be float)
            y = data.y
            
            # Create new data object with converted features
            converted_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )
            
            # Copy any additional attributes
            for key, value in data.__dict__.items():
                if key not in ['x', 'edge_index', 'edge_attr', 'y']:
                    setattr(converted_data, key, value)
            
            converted_data_list.append(converted_data)
        
        # Create a new dataset-like object that preserves the original dataset's attributes
        class ConvertedDataset:
            def __init__(self, data_list, original_dataset):
                self.data_list = data_list
                # Copy important attributes from original dataset
                if hasattr(original_dataset, 'num_node_features'):
                    self.num_node_features = original_dataset.num_node_features
                if hasattr(original_dataset, 'num_edge_features'):
                    self.num_edge_features = original_dataset.num_edge_features
                if hasattr(original_dataset, 'num_features'):
                    self.num_features = original_dataset.num_features
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
            
            def __iter__(self):
                return iter(self.data_list)
        
        converted_dataset = ConvertedDataset(converted_data_list, dataset)
        return converted_dataset


# Convenience functions
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
    
    # Convert integer features to float32 for PyTorch compatibility
    dataset = DataPreprocessor.convert_to_float(dataset)
    
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


def load_multi_solvent_molecular_data(dataset_configs: List[Dict[str, str]], 
                                     data_root: str = "data",
                                     handle_duplicates: str = 'remove',
                                     split_type: str = "kfold", 
                                     n_folds: int = 5,
                                     normalize_by_solvent: bool = True,
                                     random_state: int = 42) -> Dict[str, Any]:
    """
    Complete data loading workflow for multi-solvent molecular datasets.
    
    Args:
        dataset_configs: List of dataset configurations
        data_root: Root directory for data storage
        handle_duplicates: Strategy for duplicate handling
        split_type: Type of data splitting
        n_folds: Number of folds for cross-validation
        normalize_by_solvent: Whether to normalize targets separately by solvent
        random_state: Random seed
    
    Returns:
        Dictionary containing dataset, splits, and metadata
    """
    loader = MolecularDataLoader(data_root)
    
    # Load and concatenate datasets
    dataset, solvent_vocab, duplicate_info = loader.load_moleculenet_datasets_by_solvent(
        dataset_configs, handle_duplicates
    )
    
    # Get dataset statistics
    stats = dataset.get_dataset_statistics()
    
    # Convert integer features to float32 for PyTorch compatibility
    dataset = DataPreprocessor.convert_to_float(dataset)
    
    # Normalize targets
    norm_params = None
    if normalize_by_solvent:
        dataset, norm_params = DataPreprocessor.normalize_targets_by_solvent(dataset)
    else:
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
        'solvent_vocab': solvent_vocab,
        'duplicate_info': duplicate_info,
        'dataset_configs': dataset_configs,
        'random_state': random_state
    }
