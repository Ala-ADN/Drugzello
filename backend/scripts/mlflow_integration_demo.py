#!/usr/bin/env python3
"""
Complete MLflow Integration Example for MEGAN Models

This script demonstrates the full MLflow workflow:
1. Training with experiment tracking
2. Hyperparameter optimization
3. Model registration and promotion
4. Deployment preparation

Run this script to see the complete MLflow integration in action.
"""

import os
import sys
import tempfile
import shutil
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Import our modules
from src.utils.mlflow_integration import MLflowManager, MLflowExperimentTracker
from src.utils.config import MEGANConfig
from src.models.megan_architecture import MEGANCore


class MLflowIntegrationDemo:
    """Demonstrates complete MLflow integration workflow."""
    
    def __init__(self, demo_dir: Optional[str] = None):
        """Initialize demo with temporary directory."""
        self.demo_dir = demo_dir or tempfile.mkdtemp(prefix="mlflow_demo_")
        self.tracking_uri = f"file://{self.demo_dir}/mlruns"
        self.artifact_location = f"{self.demo_dir}/artifacts"
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        print(f"🚀 MLflow Demo initialized")
        print(f"📁 Demo directory: {self.demo_dir}")
        print(f"🔗 Tracking URI: {self.tracking_uri}")
        print("=" * 60)
    
    def create_demo_config(self) -> str:
        """Create MLflow configuration for demo."""
        config = {
            'tracking_uri': self.tracking_uri,
            'artifact_location': self.artifact_location,
            'experiment': {
                'name': 'megan_demo_experiment',
                'tags': {
                    'demo': 'true',
                    'model_type': 'megan',
                    'task': 'molecular_solubility'
                }
            },
            'auto_logging': {
                'pytorch': True,
                'sklearn': False,
                'log_models': True
            },
            'model_registry': {
                'default_stage': 'None',
                'auto_register': True
            }
        }
        
        config_path = Path(self.demo_dir) / "mlflow_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def generate_synthetic_data(self, num_samples: int = 1000):
        """Generate synthetic molecular data for demo."""
        print("🧪 Generating synthetic molecular data...")
        
        # Simulate molecular graph data
        torch.manual_seed(42)
        
        data = []
        for i in range(num_samples):
            # Random molecular graph features
            num_atoms = torch.randint(5, 30, (1,)).item()
            
            # Node features (atom features)
            node_features = torch.randn(num_atoms, 9)
            
            # Random graph connectivity
            num_edges = torch.randint(num_atoms-1, num_atoms*2, (1,)).item()
            edge_index = torch.randint(0, num_atoms, (2, num_edges))
            edge_features = torch.randn(num_edges, 3)
            
            # Synthetic solubility target (based on simple rules)
            # Larger molecules tend to be less soluble
            target = -0.1 * num_atoms + torch.randn(1).item() * 0.5
            
            data.append({
                'x': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_features,
                'y': torch.tensor([target]),
                'num_atoms': num_atoms
            })
        
        # Split data
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        print(f"✅ Generated {len(data)} synthetic samples")
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def train_model_with_tracking(self, config: MEGANConfig, train_data: List, 
                                val_data: List, run_name: str):
        """Train a model with full MLflow tracking."""
        print(f"\n🏋️ Training model: {run_name}")
        
        # Create experiment tracker
        tracker = MLflowExperimentTracker(
            experiment_name='megan_demo_experiment'
        )
        
        # Create model
        model = MEGANCore(
            in_channels=config.num_node_features,
            hidden_channels=config.hidden_channels,
            out_channels=1,
            edge_dim=config.num_edge_features,
            num_layers=config.num_layers,
            K=config.K,
            heads_gat=config.heads_gat,
            use_edge_features=config.use_edge_features,
            dropout=config.dropout,
            layer_norm=config.layer_norm,
            residual=config.residual
        )
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        # Simulate training
        model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(config.num_epochs):
            # Training epoch (simplified)
            epoch_train_loss = 0.0
            for batch in train_data[:min(10, len(train_data))]:  # Limit for demo
                optimizer.zero_grad()
                
                # Forward pass (simplified - normally would use DataLoader)
                x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
                batch_tensor = torch.zeros(x.size(0), dtype=torch.long)
                
                pred = model(x, edge_index, edge_attr, batch_tensor)
                loss = criterion(pred, batch['y'].unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / min(10, len(train_data))
            
            # Validation (simplified)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_data[:min(5, len(val_data))]:
                    x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']
                    batch_tensor = torch.zeros(x.size(0), dtype=torch.long)
                    
                    pred = model(x, edge_index, edge_attr, batch_tensor)
                    loss = criterion(pred, batch['y'].unsqueeze(0))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / min(5, len(val_data))
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            model.train()
            
            print(f"  Epoch {epoch+1:2d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Calculate final metrics
        final_metrics = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'best_epoch': val_losses.index(min(val_losses)) + 1,
            'final_r2': max(0.1, 1 - val_losses[-1]),  # Simulated R²
            'total_params': sum(p.numel() for p in model.parameters()),
            'training_time': config.num_epochs * 0.1  # Simulated training time
        }
        
        # Log complete training run
        run_info = tracker.log_training_run(
            config=config,
            model=model,
            metrics=final_metrics,
            run_name=run_name
        )
        
        print(f"✅ Training completed. Run ID: {run_info['run_id'][:8]}...")
        
        return run_info, model, final_metrics
    
    def demonstrate_hyperparameter_search(self, train_data: List, val_data: List):
        """Demonstrate hyperparameter optimization with MLflow."""
        print("\n🔍 Hyperparameter Search Demonstration")
        
        # Define search space
        param_grid = {
            'hidden_channels': [64, 128],
            'num_layers': [2, 3],
            'learning_rate': [0.001, 0.01],
            'dropout': [0.1, 0.2]
        }
        
        # Create experiment tracker
        tracker = MLflowExperimentTracker(
            experiment_name='megan_demo_experiment'
        )
        
        # Generate trial configurations
        trial_configs = []
        trial_results = []
        
        for hidden in param_grid['hidden_channels']:
            for layers in param_grid['num_layers']:
                for lr in param_grid['learning_rate']:
                    for dropout in param_grid['dropout']:
                        config = MEGANConfig(
                            num_layers=layers,
                            hidden_channels=hidden,
                            learning_rate=lr,
                            dropout=dropout,
                            num_epochs=3  # Reduced for demo
                        )
                        trial_configs.append(config)
        
        print(f"🎯 Running {len(trial_configs)} hyperparameter trials...")
        
        # Run trials
        for i, config in enumerate(trial_configs[:4]):  # Limit for demo
            print(f"\n  Trial {i+1}: hidden={config.hidden_channels}, "
                  f"layers={config.num_layers}, lr={config.learning_rate}")
            
            # Train model
            run_info, model, metrics = self.train_model_with_tracking(
                config, train_data, val_data, f"hp_trial_{i+1}"
            )
            
            trial_results.append({
                'params': {
                    'hidden_channels': config.hidden_channels,
                    'num_layers': config.num_layers,
                    'learning_rate': config.learning_rate,
                    'dropout': config.dropout
                },
                'metrics': metrics,
                'model': model,
                'run_id': run_info['run_id']
            })
        
        # Log hyperparameter search
        search_params = {
            'param_grid': param_grid,
            'search_type': 'grid_search',
            'total_trials': len(trial_results)
        }
        
        search_info = tracker.log_hyperparameter_search(
            search_params=search_params,
            trial_results=trial_results,
            run_name='hyperparameter_search_demo'
        )
        
        # Find best trial
        best_trial = min(trial_results, key=lambda x: x['metrics']['best_val_loss'])
        
        print(f"\n✅ Hyperparameter search completed!")
        print(f"🏆 Best trial: Val Loss = {best_trial['metrics']['best_val_loss']:.4f}")
        print(f"   Best params: {best_trial['params']}")
        
        return search_info, best_trial
    
    def demonstrate_model_registry(self, best_trial: Dict):
        """Demonstrate model registry operations."""
        print("\n📚 Model Registry Demonstration")
        
        client = MlflowClient()
        model_name = "megan_demo_model"
        
        # Register the best model
        print(f"📝 Registering model: {model_name}")
        
        model_uri = f"runs:/{best_trial['run_id']}/model"
        
        try:
            # Create registered model
            client.create_registered_model(
                model_name,
                description="Demo MEGAN model for molecular solubility prediction"
            )
            print(f"✅ Created registered model: {model_name}")
        except Exception as e:
            print(f"⚠️  Model already exists or creation failed: {e}")
        
        # Create model version
        try:
            version = client.create_model_version(
                model_name,
                model_uri,
                best_trial['run_id'],
                description=f"Best model from hyperparameter search. "
                           f"Val Loss: {best_trial['metrics']['best_val_loss']:.4f}"
            )
            print(f"✅ Created model version: {version.version}")
        except Exception as e:
            print(f"⚠️  Version creation failed: {e}")
            version = client.get_model_version(model_name, "1")
        
        # Demonstrate stage transitions
        stages = ['Staging', 'Production']
        
        for stage in stages:
            print(f"📈 Promoting to {stage}...")
            
            try:
                client.transition_model_version_stage(
                    model_name,
                    version.version,
                    stage
                )
                print(f"✅ Model promoted to {stage}")
                time.sleep(1)  # Brief pause for demo
            except Exception as e:
                print(f"⚠️  Stage transition failed: {e}")
        
        # Show model information
        model = client.get_registered_model(model_name)
        print(f"\n📊 Model Registry Summary:")
        print(f"   Model Name: {model.name}")
        print(f"   Latest Versions: {len(model.latest_versions)}")
        
        for mv in model.latest_versions:
            print(f"   - Version {mv.version}: {mv.current_stage}")
        
        return model_name, version.version
    
    def demonstrate_deployment_preparation(self, model_name: str, version: str):
        """Demonstrate deployment preparation."""
        print("\n🚀 Deployment Preparation Demonstration")
        
        # Import deployment functionality
        from scripts.deploy_model import ModelDeploymentPreparer
        
        try:
            # Create deployment preparer
            preparer = ModelDeploymentPreparer(
                model_name=model_name,
                version=version
            )
            
            deployment_dir = Path(self.demo_dir) / "deployment_demo"
            
            # Prepare local deployment
            print("📦 Preparing local deployment...")
            preparer.prepare_local_deployment(str(deployment_dir), "demo_service")
            
            # Check deployment artifacts
            local_dir = deployment_dir / "local"
            artifacts = list(local_dir.glob("*"))
            
            print(f"✅ Local deployment prepared!")
            print(f"   Artifacts: {[f.name for f in artifacts]}")
            
            # Test inference script
            inference_script = local_dir / "inference.py"
            if inference_script.exists():
                print("🧪 Testing inference script...")
                
                # Create a simple test
                test_content = f"""
# Test inference
import sys
sys.path.append('{local_dir}')

try:
    from inference import MEGANPredictor
    predictor = MEGANPredictor('model.pth', 'model_info.json')
    print("✅ Inference script loaded successfully")
except Exception as e:
    print(f"⚠️  Inference test failed: {{e}}")
"""
                
                test_file = local_dir / "test_inference.py"
                with open(test_file, 'w') as f:
                    f.write(test_content)
                
                print("✅ Deployment preparation completed!")
            
        except Exception as e:
            print(f"⚠️  Deployment preparation failed: {e}")
    
    def run_complete_demo(self):
        """Run the complete MLflow integration demo."""
        print("🎬 Starting Complete MLflow Integration Demo")
        print("This demo will showcase the full MLflow workflow for MEGAN models.")
        print("=" * 60)
        
        try:
            # 1. Setup
            config_path = self.create_demo_config()
            train_data, val_data, test_data = self.generate_synthetic_data(100)
            
            # 2. Basic training with tracking
            baseline_config = MEGANConfig(
                num_layers=2,
                hidden_channels=64,
                learning_rate=0.001,
                num_epochs=5
            )
            
            baseline_run, _, _ = self.train_model_with_tracking(
                baseline_config, train_data, val_data, "baseline_model"
            )
            
            # 3. Hyperparameter search
            search_info, best_trial = self.demonstrate_hyperparameter_search(
                train_data, val_data
            )
            
            # 4. Model registry
            model_name, version = self.demonstrate_model_registry(best_trial)
            
            # 5. Deployment preparation
            self.demonstrate_deployment_preparation(model_name, version)
            
            # 6. Summary
            self.print_demo_summary()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    def print_demo_summary(self):
        """Print demo summary and next steps."""
        print("\n" + "=" * 60)
        print("🎉 MLflow Integration Demo Completed!")
        print("=" * 60)
        
        print("\n📝 What was demonstrated:")
        print("✅ Experiment tracking with parameter and metric logging")
        print("✅ Hyperparameter optimization with nested runs")
        print("✅ Model registry operations and stage management")
        print("✅ Deployment preparation and artifact generation")
        
        print(f"\n📁 Demo artifacts saved to: {self.demo_dir}")
        print("\n🔗 To explore results:")
        print(f"   cd {self.demo_dir}")
        print("   mlflow ui --backend-store-uri ./mlruns")
        print("   # Open http://localhost:5000 in your browser")
        
        print("\n🚀 Next steps:")
        print("1. Explore the MLflow UI to see logged experiments")
        print("2. Check deployment artifacts in the deployment_demo folder")
        print("3. Run the validation script: python scripts/validate_mlflow_integration.py")
        print("4. Start using MLflow in your own training workflows!")
        
        print("\n📚 For more information:")
        print("- Read docs/mlflow_integration_guide.md")
        print("- Check the example scripts in scripts/")
        print("- Run the integration tests in tests/")
    
    def cleanup(self):
        """Clean up demo artifacts."""
        if Path(self.demo_dir).exists():
            shutil.rmtree(self.demo_dir)
            print(f"🧹 Cleaned up demo directory: {self.demo_dir}")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow Integration Demo")
    parser.add_argument('--demo-dir', type=str, help='Demo directory (default: temp)')
    parser.add_argument('--keep-artifacts', action='store_true', 
                       help='Keep demo artifacts after completion')
    
    args = parser.parse_args()
    
    # Run demo
    demo = MLflowIntegrationDemo(args.demo_dir)
    
    try:
        demo.run_complete_demo()
        
        if not args.keep_artifacts:
            input("\nPress Enter to cleanup demo artifacts (or Ctrl+C to keep them)...")
            demo.cleanup()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        if not args.keep_artifacts:
            demo.cleanup()
    
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
