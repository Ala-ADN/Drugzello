"""
Main training script for MEGAN model.
Provides a command-line interface for training with various configurations and MLflow tracking.
"""

import argparse
import torch
import sys
from pathlib import Path
import yaml
import json

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import load_molecular_data
from src.models.trainer import train_megan
from src.utils.config import MEGANConfig, SEARCH_SPACES
from src.utils.evaluation import VisualizationUtils, ModelEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MEGAN model for molecular solubility prediction')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ESOL',
                       help='Dataset name (default: ESOL)')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for data storage (default: data)')
    
    # Model configuration
    parser.add_argument('--config', type=str, default='default',
                       help='Predefined configuration name (default, small, large, edge_focused, regularized)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration YAML file')
    
    # Training arguments
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    # Regularization arguments
    parser.add_argument('--gamma-exp', type=float, default=None,
                       help='Explanation loss weight')
    parser.add_argument('--beta-sparsity', type=float, default=None,
                       help='Sparsity regularization weight')
    parser.add_argument('--delta-decor', type=float, default=None,
                       help='Decorrelation regularization weight')
      # Output arguments
    parser.add_argument('--save-dir', type=str, default='models/trained',
                       help='Directory to save models and results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    # MLflow arguments
    parser.add_argument('--disable-mlflow', action='store_true',
                       help='Disable MLflow experiment tracking')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode (minimal output)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    return device


def load_config(args) -> MEGANConfig:
    """Load and setup model configuration."""
    if args.config_file:
        config = MEGANConfig.load(args.config_file)
    else:
        config = MEGANConfig(args.config)
    
    # Apply command line overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gamma_exp is not None:
        config.gamma_exp = args.gamma_exp
    if args.beta_sparsity is not None:
        config.beta_sparsity = args.beta_sparsity
    if args.delta_decor is not None:
        config.delta_decor = args.delta_decor
    
    # Update random seed
    config.random_seed = args.seed
    
    return config


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging level
    if args.quiet:
        verbose = False
    else:
        verbose = True if args.verbose else True  # Default to verbose
    
    if verbose:
        print("MEGAN Model Training")
        print("=" * 50)
        print(f"Dataset: {args.dataset}")
        print(f"Configuration: {args.config}")
        print(f"Cross-validation folds: {args.n_folds}")
        print(f"Random seed: {args.seed}")
    
    # Setup device
    device = setup_device(args.device)
    if verbose:
        print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args)
    if verbose:
        print(f"Model configuration loaded: {config.__class__.__name__}")
    
    # Load data
    if verbose:
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
    
    if verbose:
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Node features: {dataset.num_node_features}")
        print(f"Edge features: {getattr(dataset, 'num_edge_features', 0)}")
        if data_info['norm_params']:
            print(f"Target normalization: {data_info['norm_params']['method']}")
    
    # Create output directories
    save_dir = Path(args.save_dir)
    results_dir = Path(args.results_dir)
    
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    if not args.no_save:
        config_save_path = save_dir / 'config.yaml'
        config.save(str(config_save_path))
        
        # Save data info
        data_info_path = save_dir / 'data_info.json'
        with open(data_info_path, 'w') as f:
            # Convert data_info to JSON-serializable format
            json_data = {
                'dataset_name': data_info['dataset_name'],
                'stats': data_info['stats'],
                'norm_params': data_info['norm_params'],
                'random_state': data_info['random_state'],
                'splits': {
                    'split_type': splits['split_type'],
                    'n_folds': splits['n_folds'],
                    'total_size': splits['total_size']
                }
            }
            json.dump(json_data, f, indent=2)
      # Train model
    if verbose:
        print(f"\nStarting training with {args.n_folds}-fold cross-validation...")
    
    fold_results = train_megan(
        dataset=dataset,
        config=config,
        fold_splits=splits['fold_splits'],
        device=device,
        save_models=not args.no_save,
        save_dir=str(save_dir) if not args.no_save else None,
        verbose=verbose,
        use_mlflow=not args.disable_mlflow,
        experiment_name=args.experiment_name
    )
    
    # Evaluate results
    if verbose:
        print("\nGenerating evaluation report...")
    
    evaluation_metrics = ModelEvaluator.evaluate_fold_results(fold_results)
    
    # Create visualizations and comprehensive report
    if not args.no_save:
        VisualizationUtils.create_comprehensive_report(
            fold_results=fold_results,
            dataset_info=data_info['stats'],
            config_info=config.to_dict(),
            save_dir=str(results_dir)
        )
    
    # Print final summary
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Cross-Validation Results:")
        print(f"  Validation Loss: {evaluation_metrics['val_loss_mean']:.4f} ± {evaluation_metrics['val_loss_std']:.4f}")
        print(f"  Validation MAE:  {evaluation_metrics['val_mae_mean']:.4f} ± {evaluation_metrics['val_mae_std']:.4f}")
        print(f"  Best Fold:       {evaluation_metrics['best_fold'] + 1} (Loss: {evaluation_metrics['best_fold_loss']:.4f})")
        print(f"  CV Stability:    {evaluation_metrics['cv_stability']:.3f}")
        
        if not args.no_save:
            print(f"\nResults saved to:")
            print(f"  Models: {save_dir}")
            print(f"  Evaluation: {results_dir}")
    
    return fold_results, evaluation_metrics


if __name__ == '__main__':
    try:
        fold_results, evaluation_metrics = main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)
