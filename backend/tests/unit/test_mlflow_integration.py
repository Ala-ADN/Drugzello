"""
Unit tests for MLflow integration components.
Tests the core MLflow functionality including experiment tracking,
model registry, and deployment preparation.
"""

import pytest
import tempfile
import shutil
import json
import yaml
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.mlflow_integration import MLflowManager, MLflowExperimentTracker
from utils.config import MEGANConfig


class TestMLflowManager:
    """Test cases for MLflowManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock MLflow configuration."""
        config = {
            'tracking_uri': f'file://{temp_dir}/mlruns',
            'artifact_location': f'{temp_dir}/artifacts',
            'experiment': {
                'name': 'test_experiment',
                'tags': {'test': 'true'}
            },
            'auto_logging': {
                'pytorch': True,
                'sklearn': False
            }
        }
        return config
    
    @pytest.fixture
    def mlflow_manager(self, mock_config, temp_dir):
        """Create MLflowManager instance for testing."""
        config_path = Path(temp_dir) / "mlflow_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_config, f)
        
        with patch('src.utils.mlflow_integration.load_config') as mock_load:
            mock_load.return_value = mock_config
            manager = MLflowManager(config_path=str(config_path))
            return manager
    
    def test_mlflow_manager_initialization(self, mlflow_manager, mock_config):
        """Test MLflowManager initialization."""
        assert mlflow_manager.config == mock_config
        assert mlflow_manager.client is not None
        assert mlflow_manager.experiment_name == 'test_experiment'
    
    def test_setup_experiment(self, mlflow_manager):
        """Test experiment setup."""
        with patch.object(mlflow_manager.client, 'create_experiment') as mock_create:
            with patch.object(mlflow_manager.client, 'get_experiment_by_name') as mock_get:
                mock_get.return_value = None
                mock_create.return_value = '123'
                
                exp_id = mlflow_manager.setup_experiment('new_experiment', {'env': 'test'})
                
                mock_create.assert_called_once_with(
                    'new_experiment',
                    artifact_location=mlflow_manager.config.get('artifact_location'),
                    tags={'env': 'test'}
                )
                assert exp_id == '123'
    
    def test_start_run(self, mlflow_manager):
        """Test starting MLflow run."""
        with patch('mlflow.start_run') as mock_start:
            mock_run = Mock()
            mock_run.info.run_id = 'test_run_id'
            mock_start.return_value = mock_run
            
            run = mlflow_manager.start_run(
                run_name='test_run',
                tags={'test': 'true'},
                nested=False
            )
            
            mock_start.assert_called_once_with(
                experiment_id=mlflow_manager.experiment_id,
                run_name='test_run',
                tags={'test': 'true'},
                nested=False
            )
            assert run.info.run_id == 'test_run_id'
    
    def test_log_config(self, mlflow_manager):
        """Test logging configuration."""
        config = MEGANConfig(
            num_layers=3,
            hidden_channels=256,
            dropout=0.2
        )
        
        with patch('mlflow.log_params') as mock_log_params:
            mlflow_manager.log_config(config)
            
            # Verify that parameters were logged
            mock_log_params.assert_called()
            
            # Check the logged parameters
            call_args = mock_log_params.call_args[0][0]
            assert 'config.num_layers' in call_args
            assert call_args['config.num_layers'] == 3
    
    def test_log_metrics(self, mlflow_manager):
        """Test logging metrics."""
        metrics = {
            'loss': 0.5,
            'mse': 0.3,
            'r2': 0.8
        }
        
        with patch('mlflow.log_metrics') as mock_log_metrics:
            mlflow_manager.log_metrics(metrics, step=10)
            
            mock_log_metrics.assert_called_with(metrics, step=10)
    
    def test_log_model(self, mlflow_manager, temp_dir):
        """Test model logging."""
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        model_path = Path(temp_dir) / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        with patch('mlflow.pytorch.log_model') as mock_log_model:
            mlflow_manager.log_model(
                model,
                artifact_path='model',
                model_name='test_model'
            )
            
            mock_log_model.assert_called_once()
    
    def test_load_model(self, mlflow_manager):
        """Test model loading."""
        with patch('mlflow.pytorch.load_model') as mock_load_model:
            mock_model = Mock()
            mock_load_model.return_value = mock_model
            
            model = mlflow_manager.load_model('models:/test_model/1')
            
            mock_load_model.assert_called_once_with('models:/test_model/1')
            assert model == mock_model
    
    def test_register_model(self, mlflow_manager):
        """Test model registration."""
        with patch.object(mlflow_manager.client, 'create_registered_model') as mock_create:
            with patch.object(mlflow_manager.client, 'get_registered_model') as mock_get:
                mock_get.side_effect = Exception("Model not found")
                
                mlflow_manager.register_model('test_run_id', 'test_model', 'model')
                
                mock_create.assert_called_once_with(
                    'test_model',
                    tags=None,
                    description=None
                )


class TestMLflowExperimentTracker:
    """Test cases for MLflowExperimentTracker class."""
    
    @pytest.fixture
    def mock_mlflow_manager(self):
        """Create mock MLflowManager."""
        manager = Mock(spec=MLflowManager)
        manager.experiment_id = 'test_exp_id'
        return manager
    
    @pytest.fixture
    def experiment_tracker(self, mock_mlflow_manager):
        """Create MLflowExperimentTracker instance for testing."""
        tracker = MLflowExperimentTracker(
            experiment_name='test_experiment',
            mlflow_manager=mock_mlflow_manager
        )
        return tracker
    
    def test_experiment_tracker_initialization(self, experiment_tracker, mock_mlflow_manager):
        """Test ExperimentTracker initialization."""
        assert experiment_tracker.experiment_name == 'test_experiment'
        assert experiment_tracker.mlflow_manager == mock_mlflow_manager
        assert experiment_tracker.active_run is None
    
    def test_start_experiment_run(self, experiment_tracker):
        """Test starting experiment run."""
        mock_run = Mock()
        mock_run.info.run_id = 'test_run_id'
        
        experiment_tracker.mlflow_manager.start_run.return_value = mock_run
        
        run = experiment_tracker.start_run(
            run_name='test_run',
            tags={'test': 'true'}
        )
        
        experiment_tracker.mlflow_manager.start_run.assert_called_once_with(
            run_name='test_run',
            tags={'test': 'true'},
            nested=False
        )
        assert experiment_tracker.active_run == mock_run
        assert run == mock_run
    
    def test_log_training_run(self, experiment_tracker):
        """Test logging complete training run."""
        config = MEGANConfig(num_layers=3, hidden_channels=256)
        model = torch.nn.Linear(10, 1)
        metrics = {'loss': 0.5, 'mse': 0.3}
        artifacts = {'plot.png': '/path/to/plot.png'}
        
        mock_run = Mock()
        mock_run.info.run_id = 'test_run_id'
        experiment_tracker.mlflow_manager.start_run.return_value = mock_run
        
        with patch('mlflow.end_run') as mock_end_run:
            run_info = experiment_tracker.log_training_run(
                config=config,
                model=model,
                metrics=metrics,
                artifacts=artifacts,
                run_name='training_run'
            )
            
            # Verify all logging methods were called
            experiment_tracker.mlflow_manager.log_config.assert_called_with(config)
            experiment_tracker.mlflow_manager.log_metrics.assert_called_with(metrics)
            experiment_tracker.mlflow_manager.log_model.assert_called()
            experiment_tracker.mlflow_manager.log_artifacts.assert_called_with(artifacts)
            mock_end_run.assert_called_once()
            
            assert run_info['run_id'] == 'test_run_id'
    
    def test_log_hyperparameter_search(self, experiment_tracker):
        """Test logging hyperparameter search."""
        search_params = {
            'param_grid': {'lr': [0.001, 0.01], 'batch_size': [32, 64]},
            'search_type': 'grid_search'
        }
        
        trial_results = [
            {
                'params': {'lr': 0.001, 'batch_size': 32},
                'metrics': {'loss': 0.5, 'r2': 0.8},
                'model': torch.nn.Linear(10, 1)
            },
            {
                'params': {'lr': 0.01, 'batch_size': 64},
                'metrics': {'loss': 0.3, 'r2': 0.9},
                'model': torch.nn.Linear(10, 1)
            }
        ]
        
        mock_parent_run = Mock()
        mock_parent_run.info.run_id = 'parent_run_id'
        mock_child_run = Mock()
        mock_child_run.info.run_id = 'child_run_id'
        
        experiment_tracker.mlflow_manager.start_run.side_effect = [mock_parent_run, mock_child_run, mock_child_run]
        
        with patch('mlflow.end_run') as mock_end_run:
            search_info = experiment_tracker.log_hyperparameter_search(
                search_params=search_params,
                trial_results=trial_results,
                run_name='hp_search'
            )
            
            # Verify parent run was created
            assert experiment_tracker.mlflow_manager.start_run.call_count == 3  # 1 parent + 2 trials
            
            # Verify metrics were logged
            assert experiment_tracker.mlflow_manager.log_metrics.call_count >= 2
            
            assert search_info['parent_run_id'] == 'parent_run_id'
            assert len(search_info['trial_runs']) == 2


class TestModelRegistryIntegration:
    """Test cases for model registry functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock MLflow client."""
        client = Mock(spec=MlflowClient)
        return client
    
    def test_model_registration_workflow(self, mock_client):
        """Test complete model registration workflow."""
        # Mock registered model
        mock_model = Mock()
        mock_model.name = 'test_model'
        mock_model.latest_versions = []
        
        # Mock model version
        mock_version = Mock()
        mock_version.version = '1'
        mock_version.current_stage = 'None'
        mock_version.run_id = 'test_run_id'
        
        mock_client.get_registered_model.return_value = mock_model
        mock_client.create_model_version.return_value = mock_version
        
        # Test model registration
        with patch('src.utils.mlflow_integration.MLflowManager') as MockManager:
            manager = MockManager.return_value
            manager.client = mock_client
            
            # Register model
            manager.register_model('test_run_id', 'test_model', 'model')
            
            # Verify calls
            mock_client.get_registered_model.assert_called_with('test_model')


@pytest.fixture
def mock_mlflow_environment():
    """Set up mock MLflow environment for tests."""
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.start_run'), \
         patch('mlflow.end_run'), \
         patch('mlflow.log_params'), \
         patch('mlflow.log_metrics'), \
         patch('mlflow.log_artifacts'), \
         patch('mlflow.pytorch.log_model'):
        yield


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_end_to_end_training_workflow(self, mock_mlflow_environment):
        """Test complete training workflow with MLflow."""
        # This would test the integration between trainer.py and MLflow
        pass
    
    def test_deployment_preparation_workflow(self, mock_mlflow_environment):
        """Test deployment preparation workflow."""
        # This would test the deployment script functionality
        pass


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__])
