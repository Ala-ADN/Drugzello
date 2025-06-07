"""
Hyperparameter optimization script for MEGAN model.
Supports grid search and random search strategies with MLflow tracking.
"""

import argparse
import torch
import sys
import time
import pickle
import json
from pathlib import Path
from itertools import product
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import load_molecular_data
from src.models.trainer import train_megan
from src.utils.config import MEGANConfig, SearchSpace, SEARCH_SPACES
from src.utils.evaluation import ModelEvaluator
from src.utils.mlflow_integration import MLflowExperimentTracker


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for MEGAN model.
    """
    def __init__(self, dataset, device: torch.device, data_splits: Dict,
                 base_config: MEGANConfig, verbose: bool = True, 
                 use_mlflow: bool = True, experiment_name: str = None):
        self.dataset = dataset
        self.device = device
        self.data_splits = data_splits
        self.base_config = base_config
        self.verbose = verbose
        self.results = []
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = None
        
        # Setup MLflow if enabled
        if use_mlflow:
            try:
                self.mlflow_tracker = MLflowExperimentTracker(experiment_name)
                if verbose:
                    print("MLflow tracking enabled for hyperparameter optimization")
            except Exception as e:
                print(f"Warning: MLflow setup failed: {e}")
                self.use_mlflow = False
    
    def grid_search(self, search_space: SearchSpace, max_trials: Optional[int] = None) -> List[Dict]:
        """
        Perform grid search over hyperparameter space.
        
        Args:
            search_space: SearchSpace object defining parameter ranges
            max_trials: Maximum number of trials (None for exhaustive search)
        
        Returns:
            List of trial results
        """
        param_combinations = search_space.get_param_combinations()
        
        if max_trials and len(param_combinations) > max_trials:
            # Randomly sample from combinations
            import random
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_trials]
        
        if self.verbose:
            print(f"Starting grid search with {len(param_combinations)} combinations")
        
        return self._run_trials(param_combinations)
    
    def random_search(self, search_space: SearchSpace, n_trials: int = 50) -> List[Dict]:
        """
        Perform random search over hyperparameter space.
        
        Args:
            search_space: SearchSpace object defining parameter ranges
            n_trials: Number of random trials to perform
        
        Returns:
            List of trial results
        """
        param_combinations = []
        for i in range(n_trials):
            param_combinations.append(search_space.sample_random_config(random_seed=42 + i))
        
        if self.verbose:
            print(f"Starting random search with {n_trials} trials")
        
        return self._run_trials(param_combinations)
    
    def _run_trials(self, param_combinations: List[Dict]) -> List[Dict]:
        """Run hyperparameter optimization trials."""
        start_time = time.time()
        
        for trial_idx, params in enumerate(param_combinations):
            trial_start = time.time()
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Trial {trial_idx + 1}/{len(param_combinations)}")
                print(f"{'='*60}")
                print("Parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            
            # Create configuration for this trial
            trial_config = MEGANConfig()
            # Start with base config
            for attr in dir(self.base_config):
                if not attr.startswith('_') and hasattr(trial_config, attr):
                    setattr(trial_config, attr, getattr(self.base_config, attr))
            
            # Apply trial parameters
            for key, value in params.items():
                setattr(trial_config, key, value)
            
            try:                # Train model
                fold_results = train_megan(
                    dataset=self.dataset,
                    config=trial_config,
                    fold_splits=self.data_splits['fold_splits'],
                    device=self.device,
                    save_models=False,  # Don't save during hyperparameter search
                    verbose=False,  # Reduce verbosity during search
                    use_mlflow=False  # Disable MLflow for individual trials (we'll log at optimization level)
                )
                
                # Evaluate results
                eval_metrics = ModelEvaluator.evaluate_fold_results(fold_results)
                
                # Store trial result
                trial_result = {
                    'trial_idx': trial_idx,
                    'parameters': params.copy(),
                    'config': trial_config.to_dict(),
                    'val_loss_mean': eval_metrics['val_loss_mean'],
                    'val_loss_std': eval_metrics['val_loss_std'],
                    'val_mae_mean': eval_metrics['val_mae_mean'],
                    'val_mae_std': eval_metrics['val_mae_std'],
                    'best_fold': eval_metrics['best_fold'],
                    'cv_stability': eval_metrics['cv_stability'],
                    'fold_results': fold_results,
                    'trial_duration': time.time() - trial_start
                }
                
                self.results.append(trial_result)
                
                if self.verbose:
                    print(f"Trial {trial_idx + 1} completed:")
                    print(f"  Val Loss: {eval_metrics['val_loss_mean']:.4f} ± {eval_metrics['val_loss_std']:.4f}")
                    print(f"  Val MAE:  {eval_metrics['val_mae_mean']:.4f} ± {eval_metrics['val_mae_std']:.4f}")
                    print(f"  Duration: {trial_result['trial_duration']:.1f}s")
                
            except Exception as e:
                if self.verbose:
                    print(f"Trial {trial_idx + 1} failed: {str(e)}")
                
                # Store failed trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'parameters': params.copy(),
                    'config': trial_config.to_dict(),
                    'error': str(e),
                    'trial_duration': time.time() - trial_start
                }
                self.results.append(trial_result)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            self._print_search_summary(total_time)
        
        return self.results
    
    def _print_search_summary(self, total_time: float):
        """Print summary of hyperparameter search."""
        successful_trials = [r for r in self.results if 'error' not in r]
        failed_trials = [r for r in self.results if 'error' in r]
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER SEARCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total trials: {len(self.results)}")
        print(f"Successful: {len(successful_trials)}")
        print(f"Failed: {len(failed_trials)}")
        print(f"Total time: {total_time/60:.1f} minutes")
        
        if successful_trials:
            # Find best trial
            best_trial = min(successful_trials, key=lambda x: x['val_loss_mean'])
            
            print(f"\nBest trial (#{best_trial['trial_idx'] + 1}):")
            print(f"  Val Loss: {best_trial['val_loss_mean']:.4f} ± {best_trial['val_loss_std']:.4f}")
            print(f"  Val MAE:  {best_trial['val_mae_mean']:.4f} ± {best_trial['val_mae_std']:.4f}")
            print(f"  Parameters:")
            for key, value in best_trial['parameters'].items():
                print(f"    {key}: {value}")
    
    def get_best_trial(self) -> Optional[Dict]:
        """Get the best trial result."""
        successful_trials = [r for r in self.results if 'error' not in r]
        if not successful_trials:
            return None
        
        return min(successful_trials, key=lambda x: x['val_loss_mean'])
    
    def save_results(self, save_path: str):
        """Save optimization results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.results, f)
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        rows = []
        for result in self.results:
            if 'error' not in result:
                row = result['parameters'].copy()
                row.update({
                    'trial_idx': result['trial_idx'],
                    'val_loss_mean': result['val_loss_mean'],
                    'val_loss_std': result['val_loss_std'],
                    'val_mae_mean': result['val_mae_mean'],
                    'val_mae_std': result['val_mae_std'],
                    'cv_stability': result['cv_stability'],
                    'trial_duration': result['trial_duration']
                })
                rows.append(row)
        
        return pd.DataFrame(rows)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for MEGAN model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ESOL',
                       help='Dataset name (default: ESOL)')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for data storage')
    
    # Search configuration
    parser.add_argument('--search-type', type=str, choices=['grid', 'random'], default='random',
                       help='Type of hyperparameter search (default: random)')
    parser.add_argument('--search-space', type=str, default='quick',
                       choices=['quick', 'comprehensive', 'architecture_focus'],
                       help='Predefined search space (default: quick)')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials for random search (default: 50)')
    parser.add_argument('--max-trials', type=int, default=None,
                       help='Maximum trials for grid search (default: None)')
    
    # Base configuration
    parser.add_argument('--base-config', type=str, default='default',
                       help='Base configuration to start from')
    
    # Cross-validation
    parser.add_argument('--n-folds', type=int, default=3,
                       help='Number of CV folds (reduced for faster search)')
      # Output
    parser.add_argument('--results-dir', type=str, default='hyperparameter_results',
                       help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    
    # MLflow settings
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='Disable MLflow tracking')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main hyperparameter optimization function."""
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("MEGAN Hyperparameter Optimization")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Search type: {args.search_type}")
    print(f"Search space: {args.search_space}")
    if args.search_type == 'random':
        print(f"Number of trials: {args.n_trials}")
    print(f"Cross-validation folds: {args.n_folds}")
    print(f"Device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load data
    print("\nLoading dataset...")
    data_info = load_molecular_data(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split_type='kfold',
        n_folds=args.n_folds,
        normalize_targets=True,
        random_state=args.seed
    )
    
    dataset = data_info['dataset']
    splits = data_info['splits']
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Setup base configuration
    base_config = MEGANConfig(args.base_config)
    # Reduce epochs for faster hyperparameter search
    base_config.num_epochs = 50
    base_config.patience = 10
    
    # Get search space
    if args.search_space in SEARCH_SPACES:
        search_space = SEARCH_SPACES[args.search_space]
    else:
        search_space = SearchSpace()  # Default search space
      # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        dataset=dataset,
        device=device,
        data_splits=splits,
        base_config=base_config,
        verbose=args.verbose,
        use_mlflow=not args.disable_mlflow,
        experiment_name=args.experiment_name
    )
    
    # Run optimization
    if args.search_type == 'grid':
        results = optimizer.grid_search(search_space, max_trials=args.max_trials)
    else:  # random
        results = optimizer.random_search(search_space, n_trials=args.n_trials)
    
    # Setup results directory
    results_dir = Path(args.results_dir)
    if args.experiment_name:
        results_dir = results_dir / args.experiment_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = results_dir / f"{args.search_type}_search_{timestamp}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    optimizer.save_results(results_dir / 'optimization_results.pkl')
    
    # Save results as DataFrame
    df = optimizer.results_to_dataframe()
    if not df.empty:
        df.to_csv(results_dir / 'results_summary.csv', index=False)
        
        # Save best configuration
        best_trial = optimizer.get_best_trial()
        if best_trial:
            best_config = MEGANConfig()
            for key, value in best_trial['parameters'].items():
                setattr(best_config, key, value)
            best_config.save(str(results_dir / 'best_config.yaml'))
            
            # Save best trial info
            with open(results_dir / 'best_trial.json', 'w') as f:
                trial_info = {
                    'trial_idx': best_trial['trial_idx'],
                    'parameters': best_trial['parameters'],
                    'val_loss_mean': best_trial['val_loss_mean'],
                    'val_loss_std': best_trial['val_loss_std'],
                    'val_mae_mean': best_trial['val_mae_mean'],
                    'val_mae_std': best_trial['val_mae_std'],
                    'cv_stability': best_trial['cv_stability']
                }
                json.dump(trial_info, f, indent=2)
      # Save experiment metadata
    with open(results_dir / 'experiment_info.json', 'w') as f:
        experiment_info = {
            'dataset': args.dataset,
            'search_type': args.search_type,
            'search_space': args.search_space,
            'n_trials': args.n_trials if args.search_type == 'random' else len(results),
            'n_folds': args.n_folds,
            'base_config': args.base_config,
            'device': str(device),
            'seed': args.seed,
            'successful_trials': len([r for r in results if 'error' not in r]),
            'failed_trials': len([r for r in results if 'error' in r])
        }
        json.dump(experiment_info, f, indent=2)
    
    # Track hyperparameter optimization with MLflow
    if not args.disable_mlflow and optimizer.use_mlflow and optimizer.mlflow_tracker:
        try:
            print("\nLogging hyperparameter optimization to MLflow...")
            
            # Get search space for logging
            if args.search_space in SEARCH_SPACES:
                search_space_dict = SEARCH_SPACES[args.search_space].to_dict()
            else:
                search_space_dict = SearchSpace().to_dict()
            
            parent_run_id = optimizer.mlflow_tracker.track_hyperparameter_optimization(
                base_config=base_config,
                search_space=search_space_dict,
                optimization_results=results
            )
            
            print(f"MLflow hyperparameter optimization logged with parent run ID: {parent_run_id}")
            
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
    
    print(f"\nResults saved to: {results_dir}")
    
    return results, optimizer


if __name__ == '__main__':
    try:
        results, optimizer = main()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        sys.exit(1)
