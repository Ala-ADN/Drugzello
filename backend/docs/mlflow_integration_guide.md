# MLflow Integration Guide for MEGAN Models

This guide provides comprehensive documentation for using MLflow with the MEGAN molecular solubility prediction project. MLflow provides experiment tracking, model registry, and deployment capabilities for machine learning workflows.

## Table of Contents

1. [Overview](#overview)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
4. [Experiment Tracking](#experiment-tracking)
5. [Model Registry](#model-registry)
6. [Deployment](#deployment)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The MLflow integration provides:

- **Experiment Tracking**: Track training runs, hyperparameters, metrics, and artifacts
- **Model Registry**: Version control and lifecycle management for trained models
- **Deployment**: Automated deployment preparation for multiple targets (local, cloud, containers)
- **Reproducibility**: Complete tracking of model lineage and experimental conditions

### Key Components

- `MLflowManager`: Core MLflow operations and client management
- `MLflowExperimentTracker`: High-level experiment tracking workflows
- Training integration: Enhanced training scripts with MLflow tracking
- Model registry management: CLI tools for model lifecycle operations
- Deployment preparation: Automated generation of deployment artifacts

## Installation and Setup

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install mlflow==2.8.1
pip install boto3  # For AWS artifact storage
pip install azure-storage-blob  # For Azure artifact storage
```

These are already included in `requirements/base.txt`.

### MLflow Server Setup

#### Local Setup (Development)

For local development, MLflow uses a file-based backend:

```bash
# Start MLflow UI server
mlflow ui --backend-store-uri file:///path/to/mlruns --default-artifact-root file:///path/to/artifacts
```

#### Remote Setup (Production)

For production, set up a remote MLflow server with database backend:

```bash
# Example with PostgreSQL backend
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://your-mlflow-bucket/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

## Configuration

MLflow configuration is managed through `configs/mlflow_config.yaml`:

```yaml
# MLflow tracking configuration
tracking_uri: "http://localhost:5000" # or file:///path/to/mlruns for local
artifact_location: "s3://your-bucket/artifacts" # or local path

# Experiment configuration
experiment:
  name: "megan_solubility_prediction"
  tags:
    project: "drugzello"
    model_type: "megan"
    task: "molecular_solubility"

# Auto-logging configuration
auto_logging:
  pytorch: true
  sklearn: false
  log_models: true
  log_input_examples: false
  log_model_signatures: true

# Model registry configuration
model_registry:
  default_stage: "None"
  auto_register: true
  enable_automatic_promotion: false

# Logging settings
logging:
  level: "INFO"
  log_system_metrics: true
  log_artifacts: true
```

### Environment Variables

You can also configure MLflow using environment variables:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="https://s3.amazonaws.com"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

## Experiment Tracking

### Basic Training with MLflow

The simplest way to use MLflow is through the enhanced training script:

```bash
# Train with MLflow tracking
python scripts/train_model.py \
    --config configs/megan_config.yaml \
    --experiment-name "megan_baseline" \
    --run-name "baseline_v1"
```

### Manual Experiment Tracking

For custom tracking in your code:

```python
from src.utils.mlflow_integration import MLflowExperimentTracker
from src.utils.config import MEGANConfig

# Create experiment tracker
tracker = MLflowExperimentTracker(
    experiment_name="custom_experiment"
)

# Prepare your config and model
config = MEGANConfig(num_layers=3, hidden_channels=256)
model = train_your_model(config)

# Log training run
run_info = tracker.log_training_run(
    config=config,
    model=model,
    metrics={
        'final_train_loss': 0.45,
        'final_val_loss': 0.52,
        'final_r2': 0.78
    },
    artifacts={
        'training_plot.png': '/path/to/plot.png',
        'confusion_matrix.png': '/path/to/matrix.png'
    },
    run_name='custom_training'
)
```

### Hyperparameter Search with MLflow

Run hyperparameter optimization with full tracking:

```bash
python scripts/hyperparameter_search.py \
    --config configs/megan_config.yaml \
    --search-config configs/hyperparameter_search.yaml \
    --experiment-name "megan_hyperparameter_search" \
    --n-trials 50
```

This creates a parent run for the search and child runs for each trial, making it easy to compare results.

### Viewing Results

Access the MLflow UI to view your experiments:

```bash
# Open browser to MLflow UI
mlflow ui
# Navigate to http://localhost:5000
```

## Model Registry

### Registering Models

Models can be registered automatically during training or manually:

```bash
# Register a model from a specific run
python scripts/model_registry.py register \
    --run-id "abc123def456" \
    --model-name "megan_solubility_v1" \
    --artifact-path "model"
```

### Model Lifecycle Management

```bash
# List all registered models
python scripts/model_registry.py list

# Compare model versions
python scripts/model_registry.py compare \
    --model-name "megan_solubility_v1" \
    --version1 "1" \
    --version2 "2"

# Promote model to staging
python scripts/model_registry.py promote \
    --model-name "megan_solubility_v1" \
    --version "2" \
    --stage "Staging"

# Promote to production
python scripts/model_registry.py promote \
    --model-name "megan_solubility_v1" \
    --version "2" \
    --stage "Production"
```

### Model Versioning Strategy

Recommended versioning strategy:

1. **None**: Newly registered models
2. **Staging**: Models being tested/validated
3. **Production**: Models serving live predictions
4. **Archived**: Deprecated models

## Deployment

### Local Deployment

Prepare a model for local inference:

```bash
python scripts/deploy_model.py megan_solubility_v1 \
    --deployment-type local \
    --version "1" \
    --output-dir deployment/local
```

This creates:

- `model.pth`: Serialized model weights
- `model_info.json`: Model metadata
- `inference.py`: Standalone inference script
- `requirements.txt`: Dependencies

### FastAPI Deployment

Create a FastAPI web service:

```bash
python scripts/deploy_model.py megan_solubility_v1 \
    --deployment-type fastapi \
    --stage "Production" \
    --output-dir deployment/api \
    --api-port 8001
```

This creates a complete FastAPI application with:

- REST API endpoints
- Health checks
- CORS configuration
- Input validation
- Error handling

### Docker Deployment

Prepare Docker containerized deployment:

```bash
python scripts/deploy_model.py megan_solubility_v1 \
    --deployment-type docker \
    --stage "Production" \
    --output-dir deployment/docker \
    --base-image "python:3.9-slim"
```

Creates:

- `Dockerfile`
- `docker-compose.yml`
- Build and run scripts
- FastAPI application

### Cloud Deployment

#### AWS Lambda

```bash
python scripts/deploy_model.py megan_solubility_v1 \
    --deployment-type aws \
    --stage "Production" \
    --cloud-config configs/cloud_deployment_config.yaml \
    --output-dir deployment/aws
```

#### Azure Container Instances

```bash
python scripts/deploy_model.py megan_solubility_v1 \
    --deployment-type azure \
    --stage "Production" \
    --cloud-config configs/cloud_deployment_config.yaml \
    --output-dir deployment/azure
```

### Cloud Configuration

Configure cloud deployments in `configs/cloud_deployment_config.yaml`:

```yaml
aws:
  deployment_type: lambda # or ecs
  region: us-east-1
  lambda:
    timeout: 30
    memory_size: 1024

azure:
  deployment_type: container_instances # or functions
  location: "East US"
  resource_group: "megan-models-rg"
```

## Best Practices

### Experiment Organization

1. **Use meaningful experiment names**: Group related runs together

   ```python
   # Good
   experiment_name = "megan_solubility_baseline"
   experiment_name = "megan_hyperparameter_optimization"

   # Bad
   experiment_name = "test1"
   ```

2. **Tag experiments appropriately**:

   ```python
   tags = {
       'model_type': 'megan',
       'dataset_version': 'v2.1',
       'environment': 'production',
       'researcher': 'john_doe'
   }
   ```

3. **Use descriptive run names**:
   ```python
   run_name = "baseline_3layer_256hidden_lr0.001"
   ```

### Metric Logging

1. **Log comprehensive metrics**:

   ```python
   metrics = {
       'train_loss': train_loss,
       'val_loss': val_loss,
       'test_loss': test_loss,
       'train_r2': train_r2,
       'val_r2': val_r2,
       'test_r2': test_r2,
       'best_epoch': best_epoch,
       'total_params': total_params
   }
   ```

2. **Log metrics at multiple steps** for training curves:
   ```python
   for epoch in range(num_epochs):
       # Training step
       train_loss = train_epoch(model, train_loader)
       val_loss = validate_epoch(model, val_loader)

       # Log metrics with step
       mlflow.log_metrics({
           'train_loss': train_loss,
           'val_loss': val_loss
       }, step=epoch)
   ```

### Artifact Management

1. **Log important artifacts**:

   ```python
   artifacts = {
       'training_curves.png': plot_path,
       'model_architecture.txt': arch_path,
       'feature_importance.json': features_path,
       'predictions.csv': predictions_path
   }
   ```

2. **Organize artifacts logically**:
   ```
   artifacts/
   ├── plots/
   │   ├── training_curves.png
   │   └── feature_importance.png
   ├── data/
   │   ├── predictions.csv
   │   └── metrics.json
   └── models/
       └── best_model.pth
   ```

### Model Registry

1. **Use semantic versioning for model names**:

   ```
   megan_solubility_v1.0.0
   megan_solubility_v1.1.0
   megan_solubility_v2.0.0
   ```

2. **Add descriptive model descriptions**:

   ```python
   description = """
   MEGAN model for molecular solubility prediction.

   Features:
   - 3-layer Graph Attention Network
   - 256 hidden channels
   - Edge features enabled
   - Trained on ChEMBL dataset v29

   Performance:
   - Test R²: 0.82
   - Test RMSE: 0.45
   """
   ```

3. **Implement staged promotion workflow**:
   ```
   None → Staging → Production → Archived
   ```

### Deployment

1. **Test deployments thoroughly**:

   ```bash
   # Test local deployment
   python deployment/local/inference.py "CCO"

   # Test API deployment
   curl -X POST "http://localhost:8001/predict" \
        -H "Content-Type: application/json" \
        -d '{"smiles": "CCO"}'
   ```

2. **Monitor deployed models**:

   - Log prediction requests and responses
   - Track model performance metrics
   - Set up alerting for errors or performance degradation

3. **Version deployment artifacts**:
   - Tag Docker images with model versions
   - Include model metadata in deployment packages
   - Maintain deployment history

## Troubleshooting

### Common Issues

#### MLflow UI Not Starting

```bash
# Check if port is available
lsof -i :5000

# Start with different port
mlflow ui --port 5001
```

#### Database Connection Issues

```bash
# Check database connectivity
python -c "import sqlalchemy; sqlalchemy.create_engine('your_db_uri').connect()"

# Reset MLflow database (caution: deletes all data)
mlflow db upgrade your_db_uri
```

#### S3 Artifact Storage Issues

```bash
# Check AWS credentials
aws sts get-caller-identity

# Test S3 bucket access
aws s3 ls s3://your-mlflow-bucket/
```

#### Model Loading Issues

```python
# Debug model loading
import mlflow.pytorch

try:
    model = mlflow.pytorch.load_model("models:/model_name/1")
except Exception as e:
    print(f"Model loading error: {e}")
    # Check model artifacts
    artifacts = mlflow.artifacts.list_artifacts("models:/model_name/1")
    print(f"Available artifacts: {artifacts}")
```

### Performance Optimization

#### Large Model Handling

```python
# For large models, use model checkpointing
mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    pickle_module=pickle,  # Use optimized serialization
)
```

#### Batch Logging

```python
# Log metrics in batches for better performance
metrics_batch = {}
for epoch in range(num_epochs):
    metrics_batch[f'train_loss'] = train_losses[epoch]
    metrics_batch[f'val_loss'] = val_losses[epoch]

# Log all at once
mlflow.log_metrics(metrics_batch)
```

### Debugging

#### Enable Debug Logging

```python
import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)
```

#### Check MLflow Configuration

```python
import mlflow
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Artifact URI: {mlflow.get_artifact_uri()}")
print(f"Active experiment: {mlflow.active_run()}")
```

## Advanced Features

### Custom Metrics

```python
# Define custom evaluation metrics
def custom_solubility_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Custom domain-specific metrics
    within_1_log = np.mean(np.abs(y_true - y_pred) < 1.0)

    return {
        'r2_score': r2,
        'rmse': rmse,
        'predictions_within_1_log': within_1_log
    }
```

### Model Signatures

```python
# Log model with input/output signatures
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Create sample input/output
sample_input = torch.randn(1, 10)
sample_output = model(sample_input)

signature = infer_signature(sample_input.numpy(), sample_output.numpy())

mlflow.pytorch.log_model(
    pytorch_model=model,
    artifact_path="model",
    signature=signature
)
```

### A/B Testing Setup

```python
# Register multiple model versions for A/B testing
mlflow_client.transition_model_version_stage(
    name="megan_solubility",
    version="1",
    stage="Production"  # Model A
)

mlflow_client.transition_model_version_stage(
    name="megan_solubility",
    version="2",
    stage="Staging"  # Model B for testing
)
```

This comprehensive guide should help you effectively use MLflow with your MEGAN molecular solubility prediction project. For additional help, refer to the [MLflow documentation](https://mlflow.org/docs/latest/index.html) or check the troubleshooting section above.
