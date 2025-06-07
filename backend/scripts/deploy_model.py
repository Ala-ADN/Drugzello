"""
MLflow Model Deployment Preparation Script.
Prepares trained MEGAN models for deployment by creating inference endpoints,
Docker containers, and deployment configurations.
"""

import argparse
import sys
import os
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.mlflow_integration import MLflowManager
from src.utils.config import MEGANConfig
from src.models.megan_architecture import MEGANCore


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare MEGAN models for deployment')
    
    # Model selection
    parser.add_argument('model_name', type=str, help='MLflow registered model name')
    parser.add_argument('--version', type=str, help='Model version (default: latest)')
    parser.add_argument('--stage', type=str, choices=['Staging', 'Production'],
                       help='Model stage to deploy')
    
    # Deployment type
    parser.add_argument('--deployment-type', type=str, 
                       choices=['fastapi', 'docker', 'azure', 'aws', 'local'],
                       default='local', help='Deployment target type')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='deployment',
                       help='Output directory for deployment artifacts')
    parser.add_argument('--service-name', type=str, help='Service name for deployment')
    
    # API configuration
    parser.add_argument('--api-port', type=int, default=8001,
                       help='API port for FastAPI deployment')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    # Docker configuration
    parser.add_argument('--base-image', type=str, default='python:3.9-slim',
                       help='Base Docker image')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU-enabled Docker image')
    
    # Cloud configuration
    parser.add_argument('--cloud-config', type=str,
                       help='Cloud deployment configuration file')
    
    return parser.parse_args()


class ModelDeploymentPreparer:
    """Prepares models for various deployment targets."""
    
    def __init__(self, model_name: str, version: Optional[str] = None, 
                 stage: Optional[str] = None):
        self.model_name = model_name
        self.version = version
        self.stage = stage
        self.mlflow_manager = MLflowManager()
        
        # Determine model URI
        if version:
            self.model_uri = f"models:/{model_name}/{version}"
        elif stage:
            self.model_uri = f"models:/{model_name}/{stage}"
        else:
            self.model_uri = f"models:/{model_name}/latest"
        
        # Load model information
        self.model_info = self._load_model_info()
    
    def _load_model_info(self) -> Dict[str, Any]:
        """Load model information from MLflow."""
        try:
            if self.version:
                model_version = self.mlflow_manager.client.get_model_version(
                    self.model_name, self.version)
            else:
                model = self.mlflow_manager.client.get_registered_model(self.model_name)
                if self.stage:
                    model_versions = [v for v in model.latest_versions 
                                    if v.current_stage == self.stage]
                    if not model_versions:
                        raise ValueError(f"No model found in stage {self.stage}")
                    model_version = model_versions[0]
                else:
                    model_version = model.latest_versions[-1]
            
            # Get run information
            run = self.mlflow_manager.client.get_run(model_version.run_id)
            
            return {
                'model_name': self.model_name,
                'version': model_version.version,
                'stage': model_version.current_stage,
                'run_id': model_version.run_id,
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags),
                'model_uri': self.model_uri
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model info: {e}")
    
    def prepare_local_deployment(self, output_dir: str, service_name: str):
        """Prepare for local deployment."""
        deployment_dir = Path(output_dir) / "local"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model = self.mlflow_manager.load_model(self.model_uri)
        model_path = deployment_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save model info
        info_path = deployment_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2, default=str)
        
        # Create inference script
        self._create_inference_script(deployment_dir)
        
        # Create requirements file
        self._create_requirements_file(deployment_dir)
        
        print(f"Local deployment prepared in {deployment_dir}")
        print(f"To run: python {deployment_dir}/inference.py")
    
    def prepare_fastapi_deployment(self, output_dir: str, service_name: str, 
                                 api_port: int = 8001, workers: int = 1):
        """Prepare FastAPI deployment."""
        deployment_dir = Path(output_dir) / "fastapi"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model = self.mlflow_manager.load_model(self.model_uri)
        model_path = deployment_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create FastAPI app
        self._create_fastapi_app(deployment_dir, service_name)
        
        # Create deployment configuration
        self._create_fastapi_config(deployment_dir, api_port, workers)
        
        # Create requirements file
        self._create_requirements_file(deployment_dir, include_fastapi=True)
        
        # Create startup script
        self._create_startup_script(deployment_dir, api_port, workers)
        
        print(f"FastAPI deployment prepared in {deployment_dir}")
        print(f"To run: cd {deployment_dir} && ./start.sh")
    
    def prepare_docker_deployment(self, output_dir: str, service_name: str,
                                base_image: str = "python:3.9-slim", gpu: bool = False):
        """Prepare Docker deployment."""
        deployment_dir = Path(output_dir) / "docker"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare FastAPI app first
        self.prepare_fastapi_deployment(str(deployment_dir), service_name)
        
        # Create Dockerfile
        self._create_dockerfile(deployment_dir, base_image, gpu)
        
        # Create docker-compose file
        self._create_docker_compose(deployment_dir, service_name)
        
        # Create build script
        self._create_docker_build_script(deployment_dir, service_name)
        
        print(f"Docker deployment prepared in {deployment_dir}")
        print(f"To build: cd {deployment_dir} && ./build.sh")
        print(f"To run: cd {deployment_dir} && docker-compose up")
    
    def _create_inference_script(self, deployment_dir: Path):
        """Create a simple inference script."""
        script_content = f'''"""
Simple inference script for MEGAN model.
"""

import torch
import torch.nn.functional as F
import json
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle

# Add the backend src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.megan_architecture import MEGANCore
from data.data_loader import molecule_to_graph


class MEGANPredictor:
    """MEGAN model predictor for deployment."""
    
    def __init__(self, model_path: str, model_info_path: str):
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Extract model parameters from the saved info
        params = self.model_info['params']
        
        # Initialize model architecture
        self.model = MEGANCore(
            in_channels=int(params.get('config.num_node_features', 9)),
            hidden_channels=int(params.get('config.hidden_channels', 256)),
            out_channels=1,
            edge_dim=int(params.get('config.num_edge_features', 3)),
            num_layers=int(params.get('config.num_layers', 3)),
            K=int(params.get('config.K', 2)),
            heads_gat=int(params.get('config.heads_gat', 8)),
            use_edge_features=params.get('config.use_edge_features', 'True') == 'True',
            dropout=float(params.get('config.dropout', 0.2)),
            layer_norm=params.get('config.layer_norm', 'True') == 'True',
            residual=params.get('config.residual', 'True') == 'True'
        )
        
        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, smiles: str) -> dict:
        """
        Predict solubility for a SMILES string.
        
        Args:
            smiles: SMILES representation of the molecule
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            # Convert SMILES to molecular graph
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {{"error": "Invalid SMILES string"}}
            
            # Create graph data
            graph_data = molecule_to_graph(mol)
            
            # Add batch dimension
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            graph_data = graph_data.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(
                    graph_data.x, 
                    graph_data.edge_index, 
                    graph_data.edge_attr, 
                    graph_data.batch
                )
                
                # Convert to log solubility
                log_solubility = prediction.item()
                
                # Convert to mol/L (assuming target was log-scaled)
                solubility_mol_per_l = 10 ** log_solubility
                
                return {{
                    "smiles": smiles,
                    "log_solubility": log_solubility,
                    "solubility_mol_per_l": solubility_mol_per_l,
                    "model_version": self.model_info['version'],
                    "model_stage": self.model_info['stage']
                }}
                
        except Exception as e:
            return {{"error": str(e)}}


def main():
    """Simple command-line interface."""
    if len(sys.argv) != 2:
        print("Usage: python inference.py <SMILES>")
        return
    
    smiles = sys.argv[1]
    
    # Initialize predictor
    predictor = MEGANPredictor("model.pth", "model_info.json")
    
    # Make prediction
    result = predictor.predict(smiles)
    
    # Print result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''
        
        script_path = deployment_dir / "inference.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _create_fastapi_app(self, deployment_dir: Path, service_name: str):
        """Create FastAPI application."""
        app_content = f'''"""
FastAPI application for MEGAN model serving.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import json
import sys
from pathlib import Path
import logging
from typing import Optional

# Add the backend src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.megan_architecture import MEGANCore
from data.data_loader import molecule_to_graph
from rdkit import Chem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="{service_name}",
    description="MEGAN model API for molecular solubility prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    smiles: str
    return_attention: bool = False

class PredictionResponse(BaseModel):
    smiles: str
    log_solubility: float
    solubility_mol_per_l: float
    model_version: str
    model_stage: str
    attention_weights: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: dict

# Global model instance
predictor = None

class MEGANPredictor:
    """MEGAN model predictor for FastAPI."""
    
    def __init__(self, model_path: str, model_info_path: str):
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Extract model parameters
        params = self.model_info['params']
        
        # Initialize model
        self.model = MEGANCore(
            in_channels=int(params.get('config.num_node_features', 9)),
            hidden_channels=int(params.get('config.hidden_channels', 256)),
            out_channels=1,
            edge_dim=int(params.get('config.num_edge_features', 3)),
            num_layers=int(params.get('config.num_layers', 3)),
            K=int(params.get('config.K', 2)),
            heads_gat=int(params.get('config.heads_gat', 8)),
            use_edge_features=params.get('config.use_edge_features', 'True') == 'True',
            dropout=float(params.get('config.dropout', 0.2)),
            layer_norm=params.get('config.layer_norm', 'True') == 'True',
            residual=params.get('config.residual', 'True') == 'True'
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Model loaded on {{self.device}}")
    
    def predict(self, smiles: str, return_attention: bool = False) -> dict:
        """Make prediction for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string")
            
            graph_data = molecule_to_graph(mol)
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
            graph_data = graph_data.to(self.device)
            
            with torch.no_grad():
                prediction = self.model(
                    graph_data.x, 
                    graph_data.edge_index, 
                    graph_data.edge_attr, 
                    graph_data.batch
                )
                
                log_solubility = prediction.item()
                solubility_mol_per_l = 10 ** log_solubility
                
                result = {{
                    "smiles": smiles,
                    "log_solubility": log_solubility,
                    "solubility_mol_per_l": solubility_mol_per_l,
                    "model_version": self.model_info['version'],
                    "model_stage": self.model_info['stage']
                }}
                
                if return_attention:
                    # Extract attention weights (simplified)
                    # This would need to be implemented based on the specific model architecture
                    result["attention_weights"] = {{"note": "Attention extraction not implemented"}}
                
                return result
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        predictor = MEGANPredictor("model.pth", "model_info.json")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        model_info=predictor.model_info if predictor else {{}}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_solubility(request: PredictionRequest):
    """Predict molecular solubility."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = predictor.predict(request.smiles, request.return_attention)
    return PredictionResponse(**result)

@app.get("/")
async def root():
    """Root endpoint."""
    return {{
        "message": "MEGAN Solubility Prediction API",
        "model_info": predictor.model_info if predictor else None,
        "endpoints": {{
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }}
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''
        
        app_path = deployment_dir / "app.py"
        with open(app_path, 'w') as f:
            f.write(app_content)
    
    def _create_requirements_file(self, deployment_dir: Path, include_fastapi: bool = False):
        """Create requirements file for deployment."""
        requirements = [
            "torch>=1.12.0",
            "torch-geometric>=2.0.0",
            "rdkit-pypi>=2022.9.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "pydantic>=1.8.0"
        ]
        
        if include_fastapi:
            requirements.extend([
                "fastapi>=0.100.0",
                "uvicorn>=0.20.0",
                "python-multipart>=0.0.5"
            ])
        
        req_path = deployment_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_fastapi_config(self, deployment_dir: Path, port: int, workers: int):
        """Create FastAPI configuration."""
        config = {
            "host": "0.0.0.0",
            "port": port,
            "workers": workers,
            "log_level": "info",
            "access_log": True,
            "loop": "auto",
            "http": "auto"
        }
        
        config_path = deployment_dir / "uvicorn_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _create_startup_script(self, deployment_dir: Path, port: int, workers: int):
        """Create startup script for FastAPI."""
        script_content = f'''#!/bin/bash
# Startup script for MEGAN FastAPI service

echo "Starting MEGAN Solubility Prediction API..."

# Install requirements
pip install -r requirements.txt

# Start the service
uvicorn app:app --host 0.0.0.0 --port {port} --workers {workers}
'''
        
        script_path = deployment_dir / "start.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        try:
            os.chmod(script_path, 0o755)
        except:
            pass
    
    def _create_dockerfile(self, deployment_dir: Path, base_image: str, gpu: bool):
        """Create Dockerfile."""
        if gpu:
            base_image = "pytorch/pytorch:latest"
        
        dockerfile_content = f'''FROM {base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
'''
        
        dockerfile_path = deployment_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
    
    def _create_docker_compose(self, deployment_dir: Path, service_name: str):
        """Create docker-compose file."""
        compose_content = f'''version: '3.8'

services:
  {service_name.lower().replace('_', '-')}:
    build: .
    ports:
      - "8001:8001"
    environment:
      - PYTHONPATH=/app
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add nginx reverse proxy
  # nginx:
  #   image: nginx:alpine
  #   ports:
  #     - "80:80"
  #   volumes:
  #     - ./nginx.conf:/etc/nginx/nginx.conf
  #   depends_on:
  #     - {service_name.lower().replace('_', '-')}
'''
        
        compose_path = deployment_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
    
    def _create_docker_build_script(self, deployment_dir: Path, service_name: str):
        """Create Docker build script."""
        script_content = f'''#!/bin/bash
# Docker build script for {service_name}

echo "Building Docker image for {service_name}..."

# Build the image
docker build -t {service_name.lower()}:latest .

echo "Build complete!"
echo "To run: docker-compose up"
echo "To run detached: docker-compose up -d"
'''
        
        script_path = deployment_dir / "build.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
          # Make executable
        try:
            os.chmod(script_path, 0o755)
        except:
            pass

    def prepare_aws_deployment(self, output_dir: str, service_name: str, cloud_config: str):
        """Prepare AWS deployment using AWS Lambda or ECS."""
        deployment_dir = Path(output_dir) / "aws"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cloud configuration
        with open(cloud_config, 'r') as f:
            config = yaml.safe_load(f)
        
        aws_config = config.get('aws', {})
        deployment_type = aws_config.get('deployment_type', 'lambda')
        
        if deployment_type == 'lambda':
            self._prepare_aws_lambda_deployment(deployment_dir, service_name, aws_config)
        elif deployment_type == 'ecs':
            self._prepare_aws_ecs_deployment(deployment_dir, service_name, aws_config)
        else:
            raise ValueError(f"Unsupported AWS deployment type: {deployment_type}")
        
        print(f"AWS deployment prepared in {deployment_dir}")
    
    def prepare_azure_deployment(self, output_dir: str, service_name: str, cloud_config: str):
        """Prepare Azure deployment using Container Instances or Functions."""
        deployment_dir = Path(output_dir) / "azure"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cloud configuration
        with open(cloud_config, 'r') as f:
            config = yaml.safe_load(f)
        
        azure_config = config.get('azure', {})
        deployment_type = azure_config.get('deployment_type', 'container_instances')
        
        if deployment_type == 'container_instances':
            self._prepare_azure_container_deployment(deployment_dir, service_name, azure_config)
        elif deployment_type == 'functions':
            self._prepare_azure_functions_deployment(deployment_dir, service_name, azure_config)
        else:
            raise ValueError(f"Unsupported Azure deployment type: {deployment_type}")
        
        print(f"Azure deployment prepared in {deployment_dir}")
    
    def _prepare_aws_lambda_deployment(self, deployment_dir: Path, service_name: str, aws_config: dict):
        """Prepare AWS Lambda deployment."""
        # Create Lambda handler
        handler_content = f'''"""
AWS Lambda handler for MEGAN model.
"""

import json
import base64
import torch
import sys
from pathlib import Path

# Add local modules to path
sys.path.append(str(Path(__file__).parent))

from inference import MEGANPredictor

# Global predictor instance
predictor = None

def lambda_handler(event, context):
    """Lambda function handler."""
    global predictor
    
    # Initialize predictor on cold start
    if predictor is None:
        try:
            predictor = MEGANPredictor("model.pth", "model_info.json")
        except Exception as e:
            return {{
                'statusCode': 500,
                'body': json.dumps({{'error': f'Failed to load model: {{str(e)}}'}}),
                'headers': {{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }}
            }}
    
    try:
        # Parse request
        if 'body' in event:
            if event.get('isBase64Encoded'):
                body = base64.b64decode(event['body']).decode('utf-8')
            else:
                body = event['body']
            
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = event
        
        smiles = body.get('smiles')
        if not smiles:
            return {{
                'statusCode': 400,
                'body': json.dumps({{'error': 'SMILES string required'}}),
                'headers': {{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }}
            }}
        
        # Make prediction
        result = predictor.predict(smiles)
        
        return {{
            'statusCode': 200,
            'body': json.dumps(result),
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }}
        }}
        
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps({{'error': str(e)}}),
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }}
        }}
'''
        
        handler_path = deployment_dir / "lambda_function.py"
        with open(handler_path, 'w') as f:
            f.write(handler_content)
        
        # Copy model and create inference script
        model = self.mlflow_manager.load_model(self.model_uri)
        model_path = deployment_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create simplified inference script for Lambda
        self._create_lambda_inference_script(deployment_dir)
        
        # Create model info
        info_path = deployment_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2, default=str)
        
        # Create requirements for Lambda
        self._create_lambda_requirements(deployment_dir)
        
        # Create SAM template
        self._create_sam_template(deployment_dir, service_name, aws_config)
        
        # Create deployment script
        self._create_aws_deploy_script(deployment_dir, service_name)
    
    def _prepare_aws_ecs_deployment(self, deployment_dir: Path, service_name: str, aws_config: dict):
        """Prepare AWS ECS deployment."""
        # First prepare Docker deployment
        self.prepare_docker_deployment(str(deployment_dir), service_name)
        
        # Create ECS task definition
        self._create_ecs_task_definition(deployment_dir, service_name, aws_config)
        
        # Create ECS service definition
        self._create_ecs_service_definition(deployment_dir, service_name, aws_config)
        
        # Create CloudFormation template
        self._create_ecs_cloudformation_template(deployment_dir, service_name, aws_config)
    
    def _prepare_azure_container_deployment(self, deployment_dir: Path, service_name: str, azure_config: dict):
        """Prepare Azure Container Instances deployment."""
        # First prepare Docker deployment
        self.prepare_docker_deployment(str(deployment_dir), service_name)
        
        # Create Azure Resource Manager template
        self._create_azure_arm_template(deployment_dir, service_name, azure_config)
        
        # Create deployment script
        self._create_azure_deploy_script(deployment_dir, service_name, azure_config)
    
    def _create_lambda_inference_script(self, deployment_dir: Path):
        """Create simplified inference script for Lambda."""
        script_content = '''"""
Simplified inference script for AWS Lambda.
"""

import torch
import json
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Minimal molecular graph conversion for Lambda
def simple_molecule_to_features(mol):
    """Simple molecular feature extraction for Lambda deployment."""
    # This is a simplified version - you may need to adapt based on your actual data_loader
    atoms = mol.GetAtoms()
    
    # Node features (simplified)
    node_features = []
    for atom in atoms:
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass() / 100.0,  # Normalized mass
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
        ]
        node_features.append(features)
    
    # Edge indices
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions
        edge_indices.extend([[i, j], [j, i]])
        
        # Bond features
        bond_features = [
            int(bond.GetBondType()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated())
        ]
        edge_features.extend([bond_features, bond_features])
    
    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long),
        torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 3), dtype=torch.float32)
    )


class MEGANCore(torch.nn.Module):
    """Simplified MEGAN model for Lambda deployment."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, **kwargs):
        super().__init__()
        # This is a placeholder - you'll need to implement the actual model architecture
        # or copy the relevant parts from your MEGAN implementation
        self.linear = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Simplified forward pass - replace with actual MEGAN implementation
        x = self.linear(x)
        return torch.mean(x, dim=0, keepdim=True)


class MEGANPredictor:
    """MEGAN predictor for Lambda deployment."""
    
    def __init__(self, model_path: str, model_info_path: str):
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        params = self.model_info['params']
        
        # Initialize simplified model
        self.model = MEGANCore(
            in_channels=int(params.get('config.num_node_features', 9)),
            hidden_channels=int(params.get('config.hidden_channels', 256)),
            out_channels=1,
            edge_dim=int(params.get('config.num_edge_features', 3))
        )
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
        except:
            # Handle potential state dict key mismatches
            pass
        
        self.model.eval()
    
    def predict(self, smiles: str) -> dict:
        """Predict solubility."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES string"}
            
            x, edge_index, edge_attr = simple_molecule_to_features(mol)
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            with torch.no_grad():
                prediction = self.model(x, edge_index, edge_attr, batch)
                log_solubility = prediction.item()
                
                return {
                    "smiles": smiles,
                    "log_solubility": log_solubility,
                    "solubility_mol_per_l": 10 ** log_solubility,
                    "model_version": self.model_info['version'],
                    "model_stage": self.model_info['stage']
                }
        
        except Exception as e:
            return {"error": str(e)}
'''
        
        script_path = deployment_dir / "inference.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def _create_lambda_requirements(self, deployment_dir: Path):
        """Create requirements.txt for Lambda."""
        requirements = [
            "torch==2.0.1",
            "rdkit-pypi",
            "numpy",
            "requests"
        ]
        
        req_path = deployment_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_sam_template(self, deployment_dir: Path, service_name: str, aws_config: dict):
        """Create SAM template for Lambda deployment."""
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Transform": "AWS::Serverless-2016-10-31",
            "Description": f"MEGAN model API - {service_name}",
            "Resources": {
                f"{service_name}Function": {
                    "Type": "AWS::Serverless::Function",
                    "Properties": {
                        "CodeUri": ".",
                        "Handler": "lambda_function.lambda_handler",
                        "Runtime": "python3.9",
                        "Timeout": aws_config.get('timeout', 30),
                        "MemorySize": aws_config.get('memory_size', 1024),
                        "Environment": {
                            "Variables": {
                                "MODEL_NAME": self.model_name
                            }
                        },
                        "Events": {
                            "ApiGateway": {
                                "Type": "Api",
                                "Properties": {
                                    "Path": "/predict",
                                    "Method": "POST"
                                }
                            }
                        }
                    }
                }
            },
            "Outputs": {
                "ApiGatewayUrl": {
                    "Description": "API Gateway endpoint URL",
                    "Value": {
                        "Fn::Sub": "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/"
                    }
                }
            }
        }
        
        template_path = deployment_dir / "template.yaml"
        with open(template_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
    
    def _create_aws_deploy_script(self, deployment_dir: Path, service_name: str):
        """Create AWS deployment script."""
        script_content = f'''#!/bin/bash
# AWS deployment script for {service_name}

echo "Deploying {service_name} to AWS Lambda..."

# Check if SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "SAM CLI is not installed. Please install it first:"
    echo "pip install aws-sam-cli"
    exit 1
fi

# Build and deploy
sam build
sam deploy --guided

echo "Deployment complete!"
'''
        
        script_path = deployment_dir / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        try:
            os.chmod(script_path, 0o755)
        except:
            pass


def main():
    """Main deployment preparation function."""
    args = parse_arguments()
    
    # Determine service name
    service_name = args.service_name or f"{args.model_name}_api"
    
    print(f"Preparing deployment for model: {args.model_name}")
    print(f"Deployment type: {args.deployment_type}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize preparer
    try:
        preparer = ModelDeploymentPreparer(
            args.model_name, 
            args.version, 
            args.stage
        )
        
        print(f"Model info loaded:")
        print(f"  Version: {preparer.model_info['version']}")
        print(f"  Stage: {preparer.model_info['stage']}")
        print(f"  Run ID: {preparer.model_info['run_id'][:8]}...")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare deployment based on type
    if args.deployment_type == 'local':
        preparer.prepare_local_deployment(args.output_dir, service_name)
    
    elif args.deployment_type == 'fastapi':
        preparer.prepare_fastapi_deployment(
            args.output_dir, service_name, args.api_port, args.workers
        )
    elif args.deployment_type == 'docker':
        preparer.prepare_docker_deployment(
            args.output_dir, service_name, args.base_image, args.gpu
        )
    
    elif args.deployment_type == 'aws':
        if not args.cloud_config:
            print("AWS deployment requires --cloud-config file")
            return
        preparer.prepare_aws_deployment(args.output_dir, service_name, args.cloud_config)
    
    elif args.deployment_type == 'azure':
        if not args.cloud_config:
            print("Azure deployment requires --cloud-config file")
            return
        preparer.prepare_azure_deployment(args.output_dir, service_name, args.cloud_config)
    
    else:
        print(f"Deployment type '{args.deployment_type}' not yet implemented")


if __name__ == '__main__':
    main()
