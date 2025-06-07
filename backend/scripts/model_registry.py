"""
MLflow Model Registry Management Script.
Provides utilities for managing models in the MLflow Model Registry including
promotion between stages, model comparison, and deployment preparation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any
import yaml
import json

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.mlflow_integration import MLflowManager
from src.utils.config import MEGANConfig
from src.utils.evaluation import ModelEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MLflow Model Registry Management')
    
    # Action subcommands
    subparsers = parser.add_subparsers(dest='action', help='Available actions')
    
    # List models
    list_parser = subparsers.add_parser('list', help='List registered models')
    list_parser.add_argument('--name', type=str, help='Filter by model name')
    list_parser.add_argument('--stage', type=str, choices=['None', 'Staging', 'Production', 'Archived'],
                            help='Filter by stage')
    
    # Get model info
    info_parser = subparsers.add_parser('info', help='Get detailed model information')
    info_parser.add_argument('model_name', type=str, help='Model name')
    info_parser.add_argument('--version', type=str, help='Model version (default: latest)')
    
    # Compare models
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('model_name', type=str, help='Model name')
    compare_parser.add_argument('--versions', type=str, nargs='+', help='Model versions to compare')
    compare_parser.add_argument('--stages', type=str, nargs='+', 
                               choices=['None', 'Staging', 'Production', 'Archived'],
                               help='Compare models in specified stages')
    
    # Promote model
    promote_parser = subparsers.add_parser('promote', help='Promote model to a stage')
    promote_parser.add_argument('model_name', type=str, help='Model name')
    promote_parser.add_argument('version', type=str, help='Model version')
    promote_parser.add_argument('stage', type=str, choices=['Staging', 'Production', 'Archived'],
                               help='Target stage')
    promote_parser.add_argument('--archive-existing', action='store_true',
                               help='Archive existing models in target stage')
    promote_parser.add_argument('--description', type=str, help='Update description')
    
    # Register new model
    register_parser = subparsers.add_parser('register', help='Register a model from run')
    register_parser.add_argument('run_id', type=str, help='MLflow run ID')
    register_parser.add_argument('model_name', type=str, help='Model name')
    register_parser.add_argument('--stage', type=str, default='None',
                                choices=['None', 'Staging', 'Production'],
                                help='Initial stage (default: None)')
    register_parser.add_argument('--description', type=str, help='Model description')
    
    # Delete model
    delete_parser = subparsers.add_parser('delete', help='Delete a model version')
    delete_parser.add_argument('model_name', type=str, help='Model name')
    delete_parser.add_argument('version', type=str, help='Model version')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    # Search runs
    search_parser = subparsers.add_parser('search', help='Search for runs')
    search_parser.add_argument('--experiment-name', type=str, help='Experiment name')
    search_parser.add_argument('--filter', type=str, help='Filter string')
    search_parser.add_argument('--order-by', type=str, nargs='+', help='Order by columns')
    search_parser.add_argument('--max-results', type=int, default=100, help='Maximum results')
    
    # Export model
    export_parser = subparsers.add_parser('export', help='Export model for deployment')
    export_parser.add_argument('model_name', type=str, help='Model name')
    export_parser.add_argument('--version', type=str, help='Model version (default: latest)')
    export_parser.add_argument('--stage', type=str, help='Model stage')
    export_parser.add_argument('--output-dir', type=str, default='exported_models',
                              help='Output directory')
    export_parser.add_argument('--format', type=str, choices=['pytorch', 'onnx', 'torchscript'],
                              default='pytorch', help='Export format')
    
    return parser.parse_args()


class ModelRegistryManager:
    """Manages MLflow Model Registry operations."""
    
    def __init__(self):
        self.mlflow_manager = MLflowManager()
    
    def list_models(self, name_filter: Optional[str] = None, 
                   stage_filter: Optional[str] = None) -> pd.DataFrame:
        """List registered models with optional filters."""
        try:
            models = self.mlflow_manager.client.search_registered_models()
            
            if name_filter:
                models = [m for m in models if name_filter.lower() in m.name.lower()]
            
            model_data = []
            for model in models:
                latest_versions = model.latest_versions
                
                if stage_filter:
                    latest_versions = [v for v in latest_versions if v.current_stage == stage_filter]
                
                for version in latest_versions:
                    try:
                        run = self.mlflow_manager.client.get_run(version.run_id)
                        metrics = run.data.metrics
                        
                        model_data.append({
                            'name': model.name,
                            'version': version.version,
                            'stage': version.current_stage,
                            'run_id': version.run_id,
                            'val_loss': metrics.get('val_loss_mean', 'N/A'),
                            'val_mae': metrics.get('val_mae_mean', 'N/A'),
                            'creation_time': version.creation_timestamp,
                            'description': version.description or 'No description'
                        })
                    except:
                        model_data.append({
                            'name': model.name,
                            'version': version.version,
                            'stage': version.current_stage,
                            'run_id': version.run_id,
                            'val_loss': 'N/A',
                            'val_mae': 'N/A',
                            'creation_time': version.creation_timestamp,
                            'description': version.description or 'No description'
                        })
            
            return pd.DataFrame(model_data)
        
        except Exception as e:
            print(f"Error listing models: {e}")
            return pd.DataFrame()
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model."""
        try:
            if version:
                model_version = self.mlflow_manager.client.get_model_version(model_name, version)
            else:
                # Get latest version
                model = self.mlflow_manager.client.get_registered_model(model_name)
                if not model.latest_versions:
                    return {"error": "No versions found for model"}
                model_version = model.latest_versions[-1]
            
            # Get run information
            run = self.mlflow_manager.client.get_run(model_version.run_id)
            
            return {
                'model_name': model_name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'run_name': run.info.run_name,
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags),
                'description': model_version.description,
                'creation_time': model_version.creation_timestamp,
                'model_uri': f"models:/{model_name}/{model_version.version}"
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def compare_models(self, model_name: str, versions: Optional[List[str]] = None,
                      stages: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare different versions or stages of a model."""
        try:
            model = self.mlflow_manager.client.get_registered_model(model_name)
            model_versions = model.latest_versions
            
            if versions:
                model_versions = [v for v in model_versions if v.version in versions]
            elif stages:
                model_versions = [v for v in model_versions if v.current_stage in stages]
            
            comparison_data = []
            for version in model_versions:
                try:
                    run = self.mlflow_manager.client.get_run(version.run_id)
                    metrics = run.data.metrics
                    params = run.data.params
                    
                    row = {
                        'version': version.version,
                        'stage': version.current_stage,
                        'run_id': version.run_id[:8] + '...',  # Truncate for display
                        'val_loss_mean': metrics.get('val_loss_mean', 'N/A'),
                        'val_mae_mean': metrics.get('val_mae_mean', 'N/A'),
                        'cv_stability': metrics.get('cv_stability', 'N/A'),
                        'learning_rate': params.get('config.learning_rate', 'N/A'),
                        'hidden_channels': params.get('config.hidden_channels', 'N/A'),
                        'num_layers': params.get('config.num_layers', 'N/A'),
                        'creation_time': version.creation_timestamp
                    }
                    comparison_data.append(row)
                except:
                    continue
            
            return pd.DataFrame(comparison_data)
        
        except Exception as e:
            print(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def promote_model(self, model_name: str, version: str, stage: str,
                     archive_existing: bool = False, description: Optional[str] = None):
        """Promote a model version to a new stage."""
        try:
            self.mlflow_manager.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            if description:
                self.mlflow_manager.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
            
            print(f"Successfully promoted {model_name} v{version} to {stage}")
            
        except Exception as e:
            print(f"Error promoting model: {e}")
    
    def register_model_from_run(self, run_id: str, model_name: str, 
                               stage: str = "None", description: Optional[str] = None):
        """Register a model from an MLflow run."""
        try:
            # Construct model URI from run
            model_uri = f"runs:/{run_id}/megan_model"
            
            # Register the model
            version = self.mlflow_manager.register_model(
                model_uri=model_uri,
                model_name=model_name,
                stage=stage,
                description=description
            )
            
            print(f"Successfully registered {model_name} v{version} from run {run_id[:8]}...")
            
        except Exception as e:
            print(f"Error registering model: {e}")
    
    def delete_model_version(self, model_name: str, version: str, force: bool = False):
        """Delete a model version."""
        if not force:
            response = input(f"Are you sure you want to delete {model_name} v{version}? (y/N): ")
            if response.lower() != 'y':
                print("Deletion cancelled")
                return
        
        try:
            self.mlflow_manager.client.delete_model_version(model_name, version)
            print(f"Successfully deleted {model_name} v{version}")
        except Exception as e:
            print(f"Error deleting model: {e}")
    
    def search_runs(self, experiment_name: Optional[str] = None, 
                   filter_string: str = "", order_by: Optional[List[str]] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """Search for runs in experiments."""
        try:
            experiment_ids = None
            if experiment_name:
                experiment = self.mlflow_manager.client.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_ids = [experiment.experiment_id]
                else:
                    print(f"Experiment '{experiment_name}' not found")
                    return pd.DataFrame()
            
            runs = self.mlflow_manager.search_runs(
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results
            )
            
            return runs
            
        except Exception as e:
            print(f"Error searching runs: {e}")
            return pd.DataFrame()
    
    def export_model(self, model_name: str, version: Optional[str] = None,
                    stage: Optional[str] = None, output_dir: str = "exported_models",
                    export_format: str = "pytorch"):
        """Export a model for deployment."""
        try:
            # Determine model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if export_format == "pytorch":
                model = self.mlflow_manager.load_model(model_uri)
                export_path = output_path / f"{model_name}_v{version or 'latest'}.pth"
                torch.save(model.state_dict(), export_path)
                
                # Also save model info
                info = self.get_model_info(model_name, version)
                info_path = output_path / f"{model_name}_v{version or 'latest'}_info.json"
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2, default=str)
                
                print(f"Model exported to {export_path}")
                print(f"Model info saved to {info_path}")
            
            else:
                print(f"Export format '{export_format}' not yet implemented")
                
        except Exception as e:
            print(f"Error exporting model: {e}")


def main():
    """Main function for model registry management."""
    args = parse_arguments()
    
    if not args.action:
        print("Please specify an action. Use --help for available actions.")
        return
    
    manager = ModelRegistryManager()
    
    if args.action == 'list':
        df = manager.list_models(args.name, args.stage)
        if not df.empty:
            print("\nRegistered Models:")
            print("=" * 80)
            print(df.to_string(index=False))
        else:
            print("No models found matching criteria")
    
    elif args.action == 'info':
        info = manager.get_model_info(args.model_name, args.version)
        if 'error' in info:
            print(f"Error: {info['error']}")
        else:
            print(f"\nModel Information for {args.model_name}:")
            print("=" * 50)
            for key, value in info.items():
                if key in ['metrics', 'params', 'tags']:
                    print(f"{key.title()}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
    
    elif args.action == 'compare':
        df = manager.compare_models(args.model_name, args.versions, args.stages)
        if not df.empty:
            print(f"\nModel Comparison for {args.model_name}:")
            print("=" * 80)
            print(df.to_string(index=False))
        else:
            print("No models found for comparison")
    
    elif args.action == 'promote':
        manager.promote_model(args.model_name, args.version, args.stage, 
                            args.archive_existing, args.description)
    
    elif args.action == 'register':
        manager.register_model_from_run(args.run_id, args.model_name, 
                                      args.stage, args.description)
    
    elif args.action == 'delete':
        manager.delete_model_version(args.model_name, args.version, args.force)
    
    elif args.action == 'search':
        df = manager.search_runs(args.experiment_name, args.filter or "", 
                               args.order_by, args.max_results)
        if not df.empty:
            print("\nSearch Results:")
            print("=" * 80)
            # Display only key columns for readability
            display_cols = ['run_id', 'experiment_id', 'status', 'start_time', 
                          'metrics.val_loss_mean', 'metrics.val_mae_mean']
            available_cols = [col for col in display_cols if col in df.columns]
            print(df[available_cols].to_string(index=False))
        else:
            print("No runs found matching criteria")
    
    elif args.action == 'export':
        manager.export_model(args.model_name, args.version, args.stage, 
                           args.output_dir, args.format)


if __name__ == '__main__':
    main()
