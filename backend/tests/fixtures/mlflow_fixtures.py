"""
Test fixtures and utilities for MLflow integration tests.
Provides common test data, mock objects, and helper functions.
"""

import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock

import torch
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.config import MEGANConfig
from models.megan_architecture import MEGANCore


@pytest.fixture(scope="session")
def temp_directory():
    """Create a temporary directory for the test session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_megan_config():
    """Provide a sample MEGAN configuration for testing."""
    return MEGANConfig(
        num_layers=3,
        hidden_channels=128,
        num_node_features=9,
        num_edge_features=3,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=10,
        early_stopping_patience=5,
        K=2,
        heads_gat=8,
        use_edge_features=True,
        layer_norm=True,
        residual=True
    )


@pytest.fixture
def sample_megan_model():
    """Provide a sample MEGAN model for testing."""
    return MEGANCore(
        in_channels=9,
        hidden_channels=128,
        out_channels=1,
        edge_dim=3,
        num_layers=3,
        K=2,
        heads_gat=8,
        use_edge_features=True,
        dropout=0.2,
        layer_norm=True,
        residual=True
    )


@pytest.fixture
def sample_training_metrics():
    """Provide sample training metrics for testing."""
    return {
        'train_loss': 0.45,
        'val_loss': 0.52,
        'train_mse': 0.32,
        'val_mse': 0.38,
        'train_r2': 0.78,
        'val_r2': 0.72,
        'train_mae': 0.41,
        'val_mae': 0.46,
        'epoch': 10,
        'best_epoch': 8,
        'learning_rate': 0.001
    }


@pytest.fixture
def sample_hyperparameter_grid():
    """Provide sample hyperparameter grid for testing."""
    return {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_channels': [64, 128, 256],
        'dropout': [0.1, 0.2, 0.3],
        'num_layers': [2, 3, 4],
        'batch_size': [16, 32, 64]
    }


@pytest.fixture
def mlflow_test_config(temp_directory):
    """Create MLflow configuration for testing."""
    config = {
        'tracking_uri': f'file://{temp_directory}/mlruns',
        'artifact_location': f'{temp_directory}/artifacts',
        'experiment': {
            'name': 'test_experiment',
            'tags': {
                'environment': 'test',
                'project': 'megan',
                'version': '1.0.0'
            }
        },
        'auto_logging': {
            'pytorch': True,
            'sklearn': False,
            'log_models': True,
            'log_input_examples': False,
            'log_model_signatures': True
        },
        'model_registry': {
            'default_stage': 'None',
            'auto_register': True,
            'enable_automatic_promotion': False
        },
        'logging': {
            'level': 'INFO',
            'log_system_metrics': True,
            'log_artifacts': True
        }
    }
    
    config_path = Path(temp_directory) / "mlflow_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


@pytest.fixture
def mock_mlflow_client():
    """Create a mock MLflow client for testing."""
    client = Mock(spec=MlflowClient)
    
    # Mock experiment
    mock_experiment = Mock()
    mock_experiment.experiment_id = 'test_exp_id'
    mock_experiment.name = 'test_experiment'
    client.get_experiment_by_name.return_value = mock_experiment
    client.create_experiment.return_value = 'test_exp_id'
    
    # Mock run
    mock_run = Mock()
    mock_run.info.run_id = 'test_run_id'
    mock_run.info.experiment_id = 'test_exp_id'
    mock_run.info.status = 'FINISHED'
    mock_run.data.metrics = {'loss': 0.5, 'r2': 0.8}
    mock_run.data.params = {'lr': '0.001', 'batch_size': '32'}
    mock_run.data.tags = {'test': 'true'}
    client.get_run.return_value = mock_run
    
    # Mock registered model
    mock_model = Mock()
    mock_model.name = 'test_model'
    mock_model.latest_versions = []
    client.get_registered_model.return_value = mock_model
    client.create_registered_model.return_value = mock_model
    
    # Mock model version
    mock_version = Mock()
    mock_version.version = '1'
    mock_version.current_stage = 'None'
    mock_version.run_id = 'test_run_id'
    mock_version.name = 'test_model'
    client.get_model_version.return_value = mock_version
    client.create_model_version.return_value = mock_version
    
    return client


@pytest.fixture
def sample_molecule_data():
    """Provide sample molecular data for testing."""
    return {
        'smiles': [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)O'  # Isopropanol
        ],
        'solubility': [0.0, -0.5, -2.1, -1.2, 0.1],  # Log solubility values
        'molecular_weight': [46.07, 60.05, 78.11, 101.19, 60.10]
    }


@pytest.fixture
def cloud_deployment_config(temp_directory):
    """Create cloud deployment configuration for testing."""
    config = {
        'aws': {
            'deployment_type': 'lambda',
            'region': 'us-east-1',
            'lambda': {
                'timeout': 30,
                'memory_size': 1024,
                'runtime': 'python3.9'
            },
            'ecs': {
                'cluster_name': 'test-cluster',
                'service_name': 'test-service',
                'task_definition_family': 'test-task',
                'container_port': 8001,
                'desired_count': 1,
                'cpu': 256,
                'memory': 512
            }
        },
        'azure': {
            'deployment_type': 'container_instances',
            'location': 'East US',
            'resource_group': 'test-rg',
            'container_instances': {
                'container_name': 'test-container',
                'cpu': 1.0,
                'memory': 1.5,
                'port': 8001
            }
        }
    }
    
    config_path = Path(temp_directory) / "cloud_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


class MockMLflowRun:
    """Mock MLflow run for testing."""
    
    def __init__(self, run_id='test_run_id', experiment_id='test_exp_id'):
        self.info = Mock()
        self.info.run_id = run_id
        self.info.experiment_id = experiment_id
        self.info.status = 'FINISHED'
        self.info.start_time = 1234567890
        self.info.end_time = 1234567950
        
        self.data = Mock()
        self.data.metrics = {}
        self.data.params = {}
        self.data.tags = {}
    
    def add_metric(self, key, value):
        """Add a metric to the mock run."""
        self.data.metrics[key] = value
    
    def add_param(self, key, value):
        """Add a parameter to the mock run."""
        self.data.params[key] = str(value)
    
    def add_tag(self, key, value):
        """Add a tag to the mock run."""
        self.data.tags[key] = value


class MockMLflowExperiment:
    """Mock MLflow experiment for testing."""
    
    def __init__(self, experiment_id='test_exp_id', name='test_experiment'):
        self.experiment_id = experiment_id
        self.name = name
        self.artifact_location = '/tmp/artifacts'
        self.lifecycle_stage = 'active'
        self.tags = {'test': 'true'}


class MockModelVersion:
    """Mock MLflow model version for testing."""
    
    def __init__(self, name='test_model', version='1', stage='None'):
        self.name = name
        self.version = version
        self.current_stage = stage
        self.run_id = 'test_run_id'
        self.status = 'READY'
        self.creation_timestamp = 1234567890
        self.last_updated_timestamp = 1234567890
        self.tags = {}
        self.description = None


def create_test_model_artifacts(temp_dir: str, model_name: str = 'test_model'):
    """Create test model artifacts for deployment testing."""
    artifacts_dir = Path(temp_dir) / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create model file
    model = MEGANCore(in_channels=9, hidden_channels=64, out_channels=1, edge_dim=3, num_layers=2)
    model_path = artifacts_dir / f'{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    
    # Create model info
    model_info = {
        'model_name': model_name,
        'version': '1',
        'stage': 'None',
        'run_id': 'test_run_id',
        'metrics': {'r2': 0.85, 'mse': 0.25},
        'params': {
            'config.num_node_features': '9',
            'config.hidden_channels': '64',
            'config.num_edge_features': '3',
            'config.num_layers': '2'
        },
        'tags': {'test': 'true'}
    }
    
    info_path = artifacts_dir / f'{model_name}_info.json'
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return {
        'model_path': str(model_path),
        'info_path': str(info_path),
        'artifacts_dir': str(artifacts_dir)
    }


def generate_hyperparameter_trials(param_grid: dict, num_trials: int = 10):
    """Generate sample hyperparameter trial results."""
    import random
    
    trials = []
    
    for i in range(num_trials):
        # Sample parameters
        params = {}
        for key, values in param_grid.items():
            params[key] = random.choice(values)
        
        # Generate realistic metrics (lower lr generally better)
        base_loss = 0.5
        lr_penalty = (params.get('learning_rate', 0.001) - 0.001) * 10
        dropout_penalty = abs(params.get('dropout', 0.2) - 0.2) * 2
        
        val_loss = max(0.1, base_loss + lr_penalty + dropout_penalty + random.normal(0, 0.1))
        val_r2 = max(0.1, min(0.95, 1 - val_loss + random.normal(0, 0.05)))
        
        # Create model
        model = MEGANCore(
            in_channels=9,
            hidden_channels=params.get('hidden_channels', 128),
            out_channels=1,
            edge_dim=3,
            num_layers=params.get('num_layers', 3)
        )
        
        trials.append({
            'trial_id': i,
            'params': params,
            'metrics': {
                'val_loss': val_loss,
                'val_r2': val_r2,
                'val_mse': val_loss * 0.8,
                'train_loss': val_loss * 0.9
            },
            'model': model,
            'status': 'COMPLETED'
        })
    
    return trials


# Test data constants
SAMPLE_SMILES = [
    'CCO',
    'CC(=O)O', 
    'c1ccccc1',
    'CCN(CC)CC',
    'CC(C)O',
    'CC(C)(C)O',
    'CCCCO',
    'c1ccc(O)cc1'
]

SAMPLE_SOLUBILITY_VALUES = [
    0.0,   # Ethanol
    -0.5,  # Acetic acid
    -2.1,  # Benzene
    -1.2,  # Triethylamine
    0.1,   # Isopropanol
    -0.3,  # tert-Butanol
    -0.8,  # Butanol
    -1.5   # Phenol
]
