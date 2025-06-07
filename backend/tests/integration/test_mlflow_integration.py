"""
Integration tests for MLflow with the complete MEGAN pipeline.
Tests the integration between training, tracking, registry, and deployment.
"""

import pytest
import tempfile
import shutil
import subprocess
import time
import requests
import json
import yaml
from pathlib import Path
import torch
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.mlflow_integration import MLflowManager, MLflowExperimentTracker
from utils.config import MEGANConfig
from models.megan_architecture import MEGANCore


class TestMLflowTrainingIntegration:
    """Test MLflow integration with training pipeline."""
    
    @pytest.fixture
    def temp_mlflow_dir(self):
        """Create temporary MLflow directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mlflow_config(self, temp_mlflow_dir):
        """Create MLflow configuration for testing."""
        config = {
            'tracking_uri': f'file://{temp_mlflow_dir}/mlruns',
            'artifact_location': f'{temp_mlflow_dir}/artifacts',
            'experiment': {
                'name': 'integration_test_experiment',
                'tags': {'test': 'integration'}
            },
            'auto_logging': {
                'pytorch': True,
                'sklearn': False
            },
            'model_registry': {
                'default_stage': 'None',
                'auto_register': True
            }
        }
        
        config_path = Path(temp_mlflow_dir) / "mlflow_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def test_training_with_mlflow_tracking(self, mlflow_config, temp_mlflow_dir):
        """Test training a model with MLflow tracking enabled."""
        # Set up MLflow
        mlflow.set_tracking_uri(f'file://{temp_mlflow_dir}/mlruns')
        
        # Create experiment tracker
        tracker = MLflowExperimentTracker(
            experiment_name='integration_test_experiment'
        )
        
        # Create a simple config
        config = MEGANConfig(
            num_layers=2,
            hidden_channels=64,
            dropout=0.1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=2
        )
        
        # Create a simple model
        model = MEGANCore(
            in_channels=9,
            hidden_channels=64,
            out_channels=1,
            edge_dim=3,
            num_layers=2
        )
        
        # Simulate training metrics
        training_metrics = {
            'train_loss': 0.5,
            'val_loss': 0.6,
            'train_mse': 0.3,
            'val_mse': 0.4,
            'train_r2': 0.7,
            'val_r2': 0.6
        }
        
        # Log training run
        run_info = tracker.log_training_run(
            config=config,
            model=model,
            metrics=training_metrics,
            run_name='integration_test_run'
        )
        
        # Verify run was logged
        assert 'run_id' in run_info
        assert run_info['run_id'] is not None
        
        # Verify experiment exists
        client = MlflowClient(f'file://{temp_mlflow_dir}/mlruns')
        experiment = client.get_experiment_by_name('integration_test_experiment')
        assert experiment is not None
        
        # Verify run exists
        run = client.get_run(run_info['run_id'])
        assert run is not None
        assert run.data.metrics['train_loss'] == 0.5
        assert 'config.num_layers' in run.data.params
    
    def test_hyperparameter_search_integration(self, mlflow_config, temp_mlflow_dir):
        """Test hyperparameter search with MLflow tracking."""
        mlflow.set_tracking_uri(f'file://{temp_mlflow_dir}/mlruns')
        
        tracker = MLflowExperimentTracker(
            experiment_name='hp_search_test'
        )
        
        # Simulate hyperparameter search
        search_params = {
            'param_grid': {
                'learning_rate': [0.001, 0.01],
                'hidden_channels': [64, 128],
                'dropout': [0.1, 0.2]
            },
            'search_type': 'grid_search',
            'cv_folds': 3
        }
        
        # Simulate trial results
        trial_results = []
        for lr in [0.001, 0.01]:
            for hidden in [64, 128]:
                for dropout in [0.1, 0.2]:
                    model = MEGANCore(
                        in_channels=9,
                        hidden_channels=hidden,
                        out_channels=1,
                        edge_dim=3,
                        num_layers=2
                    )
                    
                    # Simulate metrics (better with lower learning rate)
                    val_loss = 0.5 + lr * 10 + (dropout - 0.15) * 2
                    
                    trial_results.append({
                        'params': {
                            'learning_rate': lr,
                            'hidden_channels': hidden,
                            'dropout': dropout
                        },
                        'metrics': {
                            'val_loss': val_loss,
                            'val_r2': 1 - val_loss
                        },
                        'model': model
                    })
        
        # Log hyperparameter search
        search_info = tracker.log_hyperparameter_search(
            search_params=search_params,
            trial_results=trial_results,
            run_name='hp_search_integration'
        )
        
        # Verify search was logged
        assert 'parent_run_id' in search_info
        assert len(search_info['trial_runs']) == 8  # 2 * 2 * 2
        
        # Verify best trial information
        assert 'best_trial' in search_info
        assert search_info['best_trial']['params']['learning_rate'] == 0.001


class TestModelRegistryIntegration:
    """Test model registry integration."""
    
    @pytest.fixture
    def setup_registry_test(self):
        """Set up model registry test environment."""
        temp_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f'file://{temp_dir}/mlruns')
        
        # Create experiment and log a model
        experiment_id = mlflow.create_experiment('registry_test')
        
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Create and log a simple model
            model = torch.nn.Linear(10, 1)
            mlflow.pytorch.log_model(
                model,
                artifact_path='model',
                registered_model_name='test_megan_model'
            )
            run_id = run.info.run_id
        
        yield {
            'temp_dir': temp_dir,
            'run_id': run_id,
            'model_name': 'test_megan_model'
        }
        
        shutil.rmtree(temp_dir)
    
    def test_model_registration_and_staging(self, setup_registry_test):
        """Test model registration and stage transitions."""
        setup = setup_registry_test
        client = MlflowClient()
        
        # Get the registered model
        model = client.get_registered_model(setup['model_name'])
        assert model.name == setup['model_name']
        
        # Get the latest version
        latest_version = model.latest_versions[0]
        assert latest_version.current_stage == 'None'
        
        # Transition to Staging
        client.transition_model_version_stage(
            name=setup['model_name'],
            version=latest_version.version,
            stage='Staging'
        )
        
        # Verify stage transition
        updated_version = client.get_model_version(
            setup['model_name'],
            latest_version.version
        )
        assert updated_version.current_stage == 'Staging'
        
        # Transition to Production
        client.transition_model_version_stage(
            name=setup['model_name'],
            version=latest_version.version,
            stage='Production'
        )
        
        # Verify production stage
        prod_version = client.get_model_version(
            setup['model_name'],
            latest_version.version
        )
        assert prod_version.current_stage == 'Production'


class TestDeploymentIntegration:
    """Test deployment preparation integration."""
    
    @pytest.fixture
    def setup_deployment_test(self):
        """Set up deployment test environment."""
        temp_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f'file://{temp_dir}/mlruns')
        
        # Create and register a model
        with mlflow.start_run() as run:
            model = MEGANCore(
                in_channels=9,
                hidden_channels=64,
                out_channels=1,
                edge_dim=3,
                num_layers=2
            )
            
            # Log some parameters
            mlflow.log_params({
                'config.num_node_features': 9,
                'config.hidden_channels': 64,
                'config.num_edge_features': 3,
                'config.num_layers': 2
            })
            
            # Log model
            mlflow.pytorch.log_model(
                model,
                artifact_path='model',
                registered_model_name='deployment_test_model'
            )
            
            run_id = run.info.run_id
        
        yield {
            'temp_dir': temp_dir,
            'run_id': run_id,
            'model_name': 'deployment_test_model'
        }
        
        shutil.rmtree(temp_dir)
    
    def test_local_deployment_preparation(self, setup_deployment_test):
        """Test local deployment preparation."""
        setup = setup_deployment_test
        
        # Import deployment script functionality
        sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
        from deploy_model import ModelDeploymentPreparer
        
        # Create deployment preparer
        preparer = ModelDeploymentPreparer(
            model_name=setup['model_name'],
            version='1'
        )
        
        # Prepare local deployment
        output_dir = Path(setup['temp_dir']) / 'deployment_output'
        preparer.prepare_local_deployment(str(output_dir), 'test_service')
        
        # Verify deployment artifacts
        local_dir = output_dir / 'local'
        assert local_dir.exists()
        assert (local_dir / 'model.pth').exists()
        assert (local_dir / 'model_info.json').exists()
        assert (local_dir / 'inference.py').exists()
        assert (local_dir / 'requirements.txt').exists()
        
        # Verify model info
        with open(local_dir / 'model_info.json', 'r') as f:
            model_info = json.load(f)
        
        assert model_info['model_name'] == setup['model_name']
        assert model_info['version'] == '1'
    
    def test_fastapi_deployment_preparation(self, setup_deployment_test):
        """Test FastAPI deployment preparation."""
        setup = setup_deployment_test
        
        sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
        from deploy_model import ModelDeploymentPreparer
        
        preparer = ModelDeploymentPreparer(
            model_name=setup['model_name'],
            version='1'
        )
        
        output_dir = Path(setup['temp_dir']) / 'deployment_output'
        preparer.prepare_fastapi_deployment(str(output_dir), 'test_api', api_port=8001)
        
        # Verify FastAPI deployment artifacts
        fastapi_dir = output_dir / 'fastapi'
        assert fastapi_dir.exists()
        assert (fastapi_dir / 'app.py').exists() or (fastapi_dir / 'main.py').exists()
        assert (fastapi_dir / 'requirements.txt').exists()


class TestEndToEndWorkflow:
    """Test complete end-to-end MLflow workflow."""
    
    def test_complete_ml_pipeline_with_mlflow(self):
        """Test complete ML pipeline from training to deployment."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Set up MLflow
            mlflow.set_tracking_uri(f'file://{temp_dir}/mlruns')
            
            # 2. Train model with tracking
            tracker = MLflowExperimentTracker('e2e_test')
            
            config = MEGANConfig(
                num_layers=2,
                hidden_channels=32,
                learning_rate=0.01,
                batch_size=16,
                num_epochs=1
            )
            
            model = MEGANCore(
                in_channels=9,
                hidden_channels=32,
                out_channels=1,
                edge_dim=3,
                num_layers=2
            )
            
            # Simulate training metrics
            metrics = {
                'final_train_loss': 0.4,
                'final_val_loss': 0.5,
                'final_r2': 0.75
            }
            
            # 3. Log training run
            run_info = tracker.log_training_run(
                config=config,
                model=model,
                metrics=metrics,
                run_name='e2e_training'
            )
            
            # 4. Register model
            manager = MLflowManager()
            manager.register_model(
                run_info['run_id'],
                'e2e_test_model',
                'model'
            )
            
            # 5. Promote to staging
            client = MlflowClient()
            client.transition_model_version_stage(
                name='e2e_test_model',
                version='1',
                stage='Staging'
            )
            
            # 6. Prepare deployment
            sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
            from deploy_model import ModelDeploymentPreparer
            
            preparer = ModelDeploymentPreparer(
                model_name='e2e_test_model',
                stage='Staging'
            )
            
            output_dir = Path(temp_dir) / 'deployment'
            preparer.prepare_local_deployment(str(output_dir), 'e2e_service')
            
            # 7. Verify complete workflow
            assert (output_dir / 'local' / 'model.pth').exists()
            assert (output_dir / 'local' / 'inference.py').exists()
            
            # Verify model info contains correct metadata
            with open(output_dir / 'local' / 'model_info.json', 'r') as f:
                model_info = json.load(f)
            
            assert model_info['model_name'] == 'e2e_test_model'
            assert model_info['stage'] == 'Staging'
            assert 'final_r2' in model_info['metrics']
            
        finally:
            shutil.rmtree(temp_dir)


class TestMLflowScriptIntegration:
    """Test integration with MLflow CLI scripts."""
    
    def test_training_script_with_mlflow(self):
        """Test training script with MLflow integration."""
        # This would test the actual training script
        # For now, we'll create a placeholder
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Would run: python scripts/train_model.py --experiment-name test --disable-mlflow false
            # But we'll simulate the key parts
            
            script_path = Path(__file__).parent.parent.parent / "scripts" / "train_model.py"
            if script_path.exists():
                # Could test actual script execution here
                pass
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_model_registry_script(self):
        """Test model registry management script."""
        # This would test the model registry script
        script_path = Path(__file__).parent.parent.parent / "scripts" / "model_registry.py"
        if script_path.exists():
            # Could test actual script execution here
            pass


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
