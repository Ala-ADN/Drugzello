"""
MLflow integration for experiment tracking and model registry.
Provides utilities for logging experiments, metrics, artifacts, and managing model lifecycle.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature, infer_signature
import torch
import numpy as np
import pandas as pd
import yaml
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import tempfile
import shutil
from datetime import datetime
import logging

from ..utils.config import MEGANConfig


class MLflowManager:
    """
    Manages MLflow experiment tracking and model registry operations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MLflow manager with configuration.
        
        Args:
            config_path: Path to MLflow configuration file
        """
        self.config = self._load_config(config_path)
        self.client = None
        self.current_run = None
        self.experiment_id = None
        
        self._setup_mlflow()
        self._setup_experiment()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load MLflow configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "mlflow_config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            # Return default configuration
            return {
                'mlflow': {
                    'tracking_uri': './mlruns',
                    'experiment_name': 'MEGAN_Solubility_Prediction',
                    'default_tags': {'project': 'drugzello'},
                    'logging': {'log_models': True, 'log_artifacts': True}
                }
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_mlflow(self):
        """Configure MLflow tracking and registry URIs."""
        mlflow_config = self.config['mlflow']
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set registry URI if different
        registry_uri = mlflow_config.get('registry_uri')
        if registry_uri and registry_uri != tracking_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Initialize client
        self.client = MlflowClient()
        
        # Setup auto-logging if enabled
        autolog_config = mlflow_config.get('auto_logging', mlflow_config.get('autolog', {}))
        if autolog_config.get('pytorch', False):
            mlflow.pytorch.autolog(
                log_models=autolog_config.get('log_models', True),
                disable=autolog_config.get('disable', False)
            )
    
    def _setup_experiment(self):
        """Create or get experiment."""
        mlflow_config = self.config['mlflow']
        experiment_name = mlflow_config.get('experiment_name', 'MEGAN_Solubility_Prediction')
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                artifact_location = mlflow_config.get('default_artifact_location')
                tags = mlflow_config.get('default_tags', {})
                
                self.experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location,
                    tags=tags
                )
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logging.warning(f"Could not setup experiment: {e}")
            self.experiment_id = None
    
    def start_run(self, run_name: Optional[str] = None, 
                  nested: bool = False, tags: Optional[Dict[str, Any]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            nested: Whether this is a nested run
            tags: Additional tags for the run
        
        Returns:
            Active MLflow run
        """
        # Merge default tags with provided tags
        default_tags = self.config['mlflow'].get('default_tags', {})
        run_tags = {**default_tags}
        if tags:
            run_tags.update(tags)
        
        # Add timestamp to run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"megan_run_{timestamp}"
        
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            tags=run_tags
        )
        
        return self.current_run
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run(status=status)
            self.current_run = None
    
    def log_config(self, config: MEGANConfig, prefix: str = "config"):
        """
        Log model configuration as parameters.
        
        Args:
            config: MEGAN configuration object
            prefix: Prefix for parameter names
        """
        config_dict = config.to_dict()
        
        # Flatten nested dictionaries
        flattened = self._flatten_dict(config_dict, prefix)
        
        # Log parameters in bulk
        mlflow.log_params({key: value for key, value in flattened.items() if len(str(key)) <= 250})
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information."""
        mlflow.log_param("dataset_name", dataset_info.get("name", "unknown"))
        mlflow.log_param("dataset_size", dataset_info.get("size", 0))
        mlflow.log_param("n_features", dataset_info.get("n_features", 0))
        
        if "splits" in dataset_info:
            splits = dataset_info["splits"]
            mlflow.log_param("train_size", len(splits.get("train_idx", [])))
            mlflow.log_param("val_size", len(splits.get("val_idx", [])))
            mlflow.log_param("test_size", len(splits.get("test_idx", [])))
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number for time series metrics
        """
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=step)
    
    def log_fold_results(self, fold_results: List[Dict[str, Any]]):
        """
        Log cross-validation fold results.
        
        Args:
            fold_results: List of fold results from training
        """
        # Log individual fold metrics
        for fold_idx, fold_result in enumerate(fold_results):
            metrics = fold_result.get('metrics', {})
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"fold_{fold_idx}_{metric_name}", value)
        
        # Log aggregated metrics
        all_metrics = {}
        for fold_result in fold_results:
            metrics = fold_result.get('metrics', {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Log mean and std across folds
        for metric_name, values in all_metrics.items():
            mlflow.log_metric(f"{metric_name}_mean", np.mean(values))
            mlflow.log_metric(f"{metric_name}_std", np.std(values))
            mlflow.log_metric(f"{metric_name}_min", np.min(values))
            mlflow.log_metric(f"{metric_name}_max", np.max(values))
    
    def log_model(self, model: torch.nn.Module, model_name: str = "megan_model",
                  signature: Optional[ModelSignature] = None,
                  input_example: Optional[np.ndarray] = None,
                  conda_env: Optional[str] = None,
                  pip_requirements: Optional[List[str]] = None) -> str:
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            model_name: Name for the model
            signature: Model signature
            input_example: Example input for the model
            conda_env: Conda environment file path
            pip_requirements: List of pip requirements
        
        Returns:
            Model URI
        """
        # Default pip requirements if not provided
        if pip_requirements is None:
            pip_requirements = [
                f"torch=={torch.__version__}",
                "numpy",
                "pandas",
                "scikit-learn",
                "rdkit-pypi"
            ]
        
        # Log the model
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            conda_env=conda_env
        )
        
        return model_info.model_uri
    
    def log_artifacts(self, artifacts: Dict[str, Any], artifact_dir: str = "artifacts"):
        """
        Log various artifacts to MLflow.
        
        Args:
            artifacts: Dictionary of artifact names and values/paths
            artifact_dir: Directory name for artifacts
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for artifact_name, artifact_value in artifacts.items():
                artifact_path = temp_path / f"{artifact_name}"
                
                if isinstance(artifact_value, (dict, list)):
                    # Save as JSON
                    with open(artifact_path.with_suffix('.json'), 'w') as f:
                        json.dump(artifact_value, f, indent=2, default=str)
                elif isinstance(artifact_value, pd.DataFrame):
                    # Save as CSV
                    artifact_value.to_csv(artifact_path.with_suffix('.csv'), index=False)
                elif isinstance(artifact_value, np.ndarray):
                    # Save as numpy array
                    np.save(artifact_path.with_suffix('.npy'), artifact_value)
                elif isinstance(artifact_value, (str, Path)) and Path(artifact_value).exists():
                    # Copy existing file
                    shutil.copy2(artifact_value, artifact_path)
                else:
                    # Try to pickle
                    with open(artifact_path.with_suffix('.pkl'), 'wb') as f:
                        pickle.dump(artifact_value, f)
            
            # Log all artifacts
            mlflow.log_artifacts(temp_dir, artifact_path=artifact_dir)
    
    def register_model(self, model_uri: str, model_name: str, 
                      stage: str = "None", description: Optional[str] = None,
                      tags: Optional[Dict[str, Any]] = None) -> str:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model
            stage: Stage to transition to ("None", "Staging", "Production", "Archived")
            description: Model description
            tags: Model tags
        
        Returns:
            Model version
        """
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        
        # Update description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        # Transition to stage if specified
        if stage != "None":
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage,
                archive_existing_versions=False
            )
        
        return model_version.version
    
    def get_best_model(self, model_name: str, metric_name: str = "val_loss_mean",
                      stage: Optional[str] = None) -> Optional[str]:
        """
        Get the best model version based on a metric.
        
        Args:
            model_name: Name of the registered model
            metric_name: Metric to optimize (lower is better)
            stage: Filter by stage (None for all stages)
        
        Returns:
            Model URI of the best model
        """
        try:
            # Get all versions of the model
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if stage:
                versions = [v for v in versions if v.current_stage == stage]
            
            if not versions:
                return None
            
            # Find version with best metric
            best_version = None
            best_metric = float('inf')
            
            for version in versions:
                try:
                    run = self.client.get_run(version.run_id)
                    metric_value = float(run.data.metrics.get(metric_name, float('inf')))
                    
                    if metric_value < best_metric:
                        best_metric = metric_value
                        best_version = version
                except:
                    continue
            
            if best_version:
                return f"models:/{model_name}/{best_version.version}"
            
        except Exception as e:
            logging.error(f"Error getting best model: {e}")
        
        return None
    
    def load_model(self, model_uri: str) -> torch.nn.Module:
        """
        Load a PyTorch model from MLflow.
        
        Args:
            model_uri: URI of the model to load
        
        Returns:
            Loaded PyTorch model
        """
        return mlflow.pytorch.load_model(model_uri)
    
    def search_runs(self, filter_string: str = "", 
                   order_by: Optional[List[str]] = None,
                   max_results: int = 100) -> pd.DataFrame:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter string for the search
            order_by: List of columns to order by
            max_results: Maximum number of results
        
        Returns:
            DataFrame with run information
        """
        if self.experiment_id is None:
            return pd.DataFrame()
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results
        )
        
        return runs
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs based on specified metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
        
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                row = {'run_id': run_id, 'run_name': run.info.run_name}
                
                for metric in metrics:
                    row[metric] = run.data.metrics.get(metric, None)
                
                comparison_data.append(row)
            except:
                continue
        
        return pd.DataFrame(comparison_data)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class MLflowExperimentTracker:
    """
    High-level interface for tracking MEGAN experiments with MLflow.
    """
    
    def __init__(self, experiment_name: Optional[str] = None):
        """Initialize experiment tracker."""
        self.mlflow_manager = MLflowManager()
        if experiment_name:
            self.mlflow_manager.config['mlflow']['experiment_name'] = experiment_name
            self.mlflow_manager._setup_experiment()
    
    def track_training_experiment(self, config: MEGANConfig, dataset_info: Dict[str, Any],
                                fold_results: List[Dict[str, Any]], model: torch.nn.Module,
                                additional_artifacts: Optional[Dict[str, Any]] = None,
                                model_signature: Optional[ModelSignature] = None) -> Tuple[str, str]:
        """
        Track a complete training experiment.
        
        Args:
            config: MEGAN configuration
            dataset_info: Information about the dataset
            fold_results: Results from cross-validation folds
            model: Trained model
            additional_artifacts: Additional artifacts to log
            model_signature: Model signature for input/output schema
        
        Returns:
            Tuple of (run_id, model_uri)
        """
        # Start run
        run_name = f"megan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = self.mlflow_manager.start_run(
            run_name=run_name,
            tags={
                "type": "training",
                "model": "megan",
                "dataset": dataset_info.get("name", "unknown")
            }
        )
        
        try:
            # Log configuration
            self.mlflow_manager.log_config(config)
            
            # Log dataset info
            self.mlflow_manager.log_dataset_info(dataset_info)
            
            # Log fold results
            self.mlflow_manager.log_fold_results(fold_results)
            
            # Log model
            model_uri = self.mlflow_manager.log_model(
                model=model,
                model_name="megan_model",
                signature=model_signature
            )
            
            # Log additional artifacts
            if additional_artifacts:
                self.mlflow_manager.log_artifacts(additional_artifacts)
            
            # Log system info
            self._log_system_info()
            
            run_id = run.info.run_id
            
        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        else:
            self.mlflow_manager.end_run(status="FINISHED")
        
        return run_id, model_uri
    
    def track_hyperparameter_optimization(self, base_config: MEGANConfig,
                                        search_space: Dict[str, Any],
                                        optimization_results: List[Dict[str, Any]]) -> str:
        """
        Track hyperparameter optimization experiment.
        
        Args:
            base_config: Base configuration
            search_space: Search space definition
            optimization_results: Results from optimization
        
        Returns:
            Parent run ID
        """
        # Start parent run for hyperparameter optimization
        parent_run_name = f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = self.mlflow_manager.start_run(
            run_name=parent_run_name,
            tags={
                "type": "hyperparameter_optimization",
                "model": "megan"
            }
        )
        
        try:
            # Log base configuration and search space
            self.mlflow_manager.log_config(base_config, prefix="base_config")
            self.mlflow_manager.log_artifacts({"search_space": search_space})
            
            # Log optimization summary
            successful_trials = [r for r in optimization_results if 'error' not in r]
            self.mlflow_manager.log_metrics({
                "total_trials": len(optimization_results),
                "successful_trials": len(successful_trials),
                "failed_trials": len(optimization_results) - len(successful_trials)
            })
            
            # Log individual trials as nested runs
            for trial_result in successful_trials:
                trial_run_name = f"trial_{trial_result['trial_idx']}"
                trial_run = self.mlflow_manager.start_run(
                    run_name=trial_run_name,
                    nested=True,
                    tags={
                        "type": "hyperparameter_trial",
                        "trial_idx": str(trial_result['trial_idx'])
                    }
                )
                
                # Log trial parameters
                for param_name, param_value in trial_result['parameters'].items():
                    mlflow.log_param(param_name, param_value)
                
                # Log trial metrics
                trial_metrics = {
                    k: v for k, v in trial_result.items()
                    if k not in ['trial_idx', 'parameters', 'config', 'fold_results']
                    and isinstance(v, (int, float))
                }
                self.mlflow_manager.log_metrics(trial_metrics)
                
                self.mlflow_manager.end_run()
            
            parent_run_id = parent_run.info.run_id
            
        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        else:
            self.mlflow_manager.end_run(status="FINISHED")
        
        return parent_run_id
    
    def _log_system_info(self):
        """Log system information."""
        import platform
        import psutil
        
        # System info
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("cpu_count", psutil.cpu_count())
        mlflow.log_param("memory_gb", round(psutil.virtual_memory().total / (1024**3), 2))
        
        # PyTorch info
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("cuda_version", torch.version.cuda)
            mlflow.log_param("gpu_count", torch.cuda.device_count())
