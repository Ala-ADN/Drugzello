"""
Comprehensive test file for the enhanced molecular data loader.
Tests multi-solvent support, duplicate handling, and all major functionality.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import (
    MolecularDataLoader, 
    MultiSolventDataset, 
    DataPreprocessor,
    load_molecular_data,
    load_multi_solvent_molecular_data
)

def test_basic_esol_loading():
    """Test basic ESOL dataset loading."""
    print("🧪 Testing basic ESOL dataset loading...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = MolecularDataLoader(data_root=temp_dir)
            dataset = loader.load_esol_dataset()
            
            print(f"✅ ESOL dataset loaded successfully")
            print(f"   - Dataset size: {len(dataset)}")
            print(f"   - Node features: {dataset.num_node_features}")
            print(f"   - Edge features: {dataset.num_edge_features}")
            
            # Test first sample
            first_sample = dataset[0]
            print(f"   - First sample shape: {first_sample.x.shape}")
            print(f"   - Target value: {first_sample.y.item():.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ ESOL loading failed: {e}")
        return False

def test_multi_solvent_loading():
    """Test multi-solvent dataset loading with duplicate handling."""
    print("\n🧪 Testing multi-solvent dataset loading...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = MolecularDataLoader(data_root=temp_dir)
            
            # Test configuration
            dataset_configs = [
                {'name': 'ESOL', 'solvent': 'water'},
                {'name': 'FreeSolv', 'solvent': 'water'}
            ]
            
            # Test different duplicate handling strategies
            for strategy in ['remove', 'keep_first', 'keep_last', 'average']:
                print(f"   Testing duplicate strategy: {strategy}")
                
                dataset, solvent_vocab, duplicate_info = loader.load_moleculenet_datasets_by_solvent(
                    dataset_configs, handle_duplicates=strategy
                )
                
                print(f"   ✅ Strategy '{strategy}' completed")
                print(f"      - Total samples: {len(dataset)}")
                print(f"      - Solvent vocab: {solvent_vocab}")
                print(f"      - Duplicates found: {duplicate_info['duplicates_found']}")
                print(f"      - Duplicates removed: {duplicate_info['duplicates_removed']}")
                
                # Test dataset statistics
                stats = dataset.get_dataset_statistics()
                print(f"      - Solvents: {list(stats['by_solvent'].keys())}")
                print(f"      - Datasets: {list(stats['by_dataset'].keys())}")
                
        return True
        
    except Exception as e:
        print(f"❌ Multi-solvent loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_dataset_loading():
    """Test custom CSV dataset loading."""
    print("\n🧪 Testing custom CSV dataset loading...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a sample CSV file
            csv_path = Path(temp_dir) / "test_molecules.csv"
            
            sample_data = {
                'smiles': [
                    'CCO',  # Ethanol
                    'CC(C)O',  # Isopropanol
                    'CCCCO',  # Butanol
                    'c1ccccc1O',  # Phenol
                    'CC(=O)O'  # Acetic acid
                ],
                'solubility': [-0.24, -0.77, -0.88, -0.04, 1.38],
                'solvent': ['water', 'water', 'water', 'water', 'water']
            }
            
            df = pd.DataFrame(sample_data)
            df.to_csv(csv_path, index=False)
            
            print(f"   Created test CSV with {len(df)} molecules")
            
            # Load custom dataset
            loader = MolecularDataLoader(data_root=temp_dir)
            dataset = loader.load_custom_dataset(
                csv_path=str(csv_path),
                smiles_col='smiles',
                target_col='solubility',
                solvent_col='solvent'
            )
            
            print(f"✅ Custom dataset loaded successfully")
            print(f"   - Dataset size: {len(dataset)}")
            print(f"   - Node features: {dataset.num_node_features}")
            print(f"   - Edge features: {dataset.num_edge_features}")
            
            # Test sample
            sample = dataset[0]
            print(f"   - Sample SMILES: {sample.smiles}")
            print(f"   - Sample target: {sample.y.item():.4f}")
            print(f"   - Sample solvent: {sample.solvent}")
            
            return True
            
    except Exception as e:
        print(f"❌ Custom dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_splits():
    """Test different data splitting strategies."""
    print("\n🧪 Testing data splitting strategies...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = MolecularDataLoader(data_root=temp_dir)
            dataset = loader.load_esol_dataset()
            
            # Test random split
            print("   Testing random split...")
            random_splits = loader.create_splits(
                dataset, split_type="random", test_size=0.2, val_size=0.1
            )
            
            print(f"   ✅ Random split created")
            print(f"      - Train size: {random_splits['train_size']}")
            print(f"      - Val size: {random_splits['val_size']}")
            print(f"      - Test size: {random_splits['test_size']}")
            
            # Test k-fold split
            print("   Testing k-fold split...")
            kfold_splits = loader.create_splits(
                dataset, split_type="kfold", n_folds=5
            )
            
            print(f"   ✅ K-fold split created")
            print(f"      - Number of folds: {kfold_splits['n_folds']}")
            print(f"      - Total size: {kfold_splits['total_size']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing utilities."""
    print("\n🧪 Testing data preprocessing...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = MolecularDataLoader(data_root=temp_dir)
            dataset = loader.load_esol_dataset()
            
            # Take a small subset for testing
            subset_data = [dataset[i] for i in range(min(100, len(dataset)))]
            
            class TestDataset:
                def __init__(self, data_list):
                    self.data_list = data_list
                    self.num_node_features = data_list[0].x.size(1)
                    self.num_edge_features = data_list[0].edge_attr.size(1) if data_list[0].edge_attr is not None else 0
                
                def __len__(self):
                    return len(self.data_list)
                
                def __getitem__(self, idx):
                    return self.data_list[idx]
                
                def __iter__(self):
                    return iter(self.data_list)
            
            test_dataset = TestDataset(subset_data)
            
            # Test normalization methods
            for method in ['standard', 'minmax', 'robust']:
                print(f"   Testing {method} normalization...")
                
                normalized_dataset, norm_params = DataPreprocessor.normalize_targets(
                    test_dataset, method=method
                )
                
                print(f"   ✅ {method} normalization completed")
                print(f"      - Method: {norm_params['method']}")
                print(f"      - Params: {list(norm_params.keys())}")
                
                # Test denormalization
                sample_pred = np.array([0.0, 1.0, -1.0])
                denorm_pred = DataPreprocessor.denormalize_predictions(sample_pred, norm_params)
                print(f"      - Denormalization test: {denorm_pred}")
            
            # Test duplicate detection
            print("   Testing duplicate detection...")
            duplicate_stats = DataPreprocessor.detect_duplicates_in_dataset(test_dataset)
            print(f"   ✅ Duplicate detection completed")
            print(f"      - Total molecules: {duplicate_stats['total_molecules']}")
            print(f"      - Unique molecules: {duplicate_stats['unique_molecules']}")
            print(f"      - Exact duplicates: {duplicate_stats['exact_duplicates']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    """Test convenience functions for complete workflows."""
    print("\n🧪 Testing convenience functions...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test single dataset workflow
            print("   Testing load_molecular_data...")
            result = load_molecular_data(
                dataset_name="ESOL",
                data_root=temp_dir,
                split_type="kfold",
                n_folds=3,
                normalize_targets=True
            )
            
            print(f"   ✅ Single dataset workflow completed")
            print(f"      - Dataset size: {len(result['dataset'])}")
            print(f"      - Split type: {result['splits']['split_type']}")
            print(f"      - Normalization: {result['norm_params'] is not None}")
            print(f"      - Stats keys: {list(result['stats'].keys())}")
            
            # Test multi-solvent workflow
            print("   Testing load_multi_solvent_molecular_data...")
            dataset_configs = [
                {'name': 'ESOL', 'solvent': 'water'}
            ]
            
            multi_result = load_multi_solvent_molecular_data(
                dataset_configs=dataset_configs,
                data_root=temp_dir,
                handle_duplicates='remove',
                split_type="kfold",
                n_folds=3,
                normalize_by_solvent=True
            )
            
            print(f"   ✅ Multi-solvent workflow completed")
            print(f"      - Dataset size: {len(multi_result['dataset'])}")
            print(f"      - Solvent vocab: {multi_result['solvent_vocab']}")
            print(f"      - Duplicate info: {multi_result['duplicate_info']}")
            print(f"      - Stats by solvent: {list(multi_result['stats']['by_solvent'].keys())}")
            
            return True
            
    except Exception as e:
        print(f"❌ Convenience functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_statistics():
    """Test dataset statistics computation."""
    print("\n🧪 Testing dataset statistics...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = MolecularDataLoader(data_root=temp_dir)
            dataset = loader.load_esol_dataset()
            
            # Test general statistics
            stats = loader.get_dataset_statistics(dataset)
            
            print(f"✅ Dataset statistics computed")
            print(f"   - Number of samples: {stats['num_samples']}")
            print(f"   - Node features: {stats['num_node_features']}")
            print(f"   - Edge features: {stats['num_edge_features']}")
            print(f"   - Target mean: {stats['target_mean']:.4f}")
            print(f"   - Target std: {stats['target_std']:.4f}")
            print(f"   - Avg nodes per graph: {stats['avg_nodes_per_graph']:.2f}")
            print(f"   - Avg edges per graph: {stats['avg_edges_per_graph']:.2f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Dataset statistics failed: {e}")
        return False

def test_compatible_datasets():
    """Test compatible dataset groupings."""
    print("\n🧪 Testing compatible dataset groupings...")
    
    try:
        loader = MolecularDataLoader()
        compatible = loader.load_compatible_moleculenet_datasets()
        
        print(f"✅ Compatible datasets retrieved")
        for solvent, datasets in compatible.items():
            print(f"   - {solvent}: {datasets}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatible datasets failed: {e}")
        return False

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n🧪 Testing error handling...")
    
    try:
        loader = MolecularDataLoader()
        
        # Test invalid split type
        try:
            dataset = loader.load_esol_dataset()
            loader.create_splits(dataset, split_type="invalid")
            print("❌ Should have raised error for invalid split type")
            return False
        except ValueError:
            print("   ✅ Correctly handled invalid split type")
        
        # Test invalid duplicate strategy
        try:
            loader.load_moleculenet_datasets_by_solvent(
                [{'name': 'ESOL', 'solvent': 'water'}],
                handle_duplicates='invalid'
            )
            print("❌ Should have raised error for invalid duplicate strategy")
            return False
        except ValueError:
            print("   ✅ Correctly handled invalid duplicate strategy")
        
        # Test invalid normalization method
        try:
            dataset = loader.load_esol_dataset()
            DataPreprocessor.normalize_targets(dataset, method='invalid')
            print("❌ Should have raised error for invalid normalization")
            return False
        except ValueError:
            print("   ✅ Correctly handled invalid normalization method")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all data loader tests."""
    print("🚀 Running comprehensive data loader tests...\n")
    
    tests = [
        test_basic_esol_loading,
        test_multi_solvent_loading,
        test_custom_dataset_loading,
        test_data_splits,
        test_data_preprocessing,
        test_convenience_functions,
        test_dataset_statistics,
        test_compatible_datasets,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Data loader is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
