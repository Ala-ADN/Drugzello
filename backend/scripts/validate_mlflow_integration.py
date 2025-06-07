#!/usr/bin/env python3
"""
MLflow Integration Test Runner and Validator

This script runs comprehensive tests for the MLflow integration,
validates the setup, and provides diagnostic information.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mlflow
    import torch
    import pytest
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)


class MLflowIntegrationValidator:
    """Validates MLflow integration setup and functionality."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = None
        
    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation checks."""
        print("🔍 Starting MLflow Integration Validation...")
        print("=" * 60)
        
        # Check dependencies
        self.validate_dependencies()
        
        # Check configuration
        self.validate_configuration()
        
        # Check MLflow functionality
        self.validate_mlflow_functionality()
        
        # Run unit tests
        self.run_unit_tests()
        
        # Run integration tests
        self.run_integration_tests()
        
        # Check deployment preparation
        self.validate_deployment_preparation()
        
        # Display results
        self.display_results()
        
        return self.results
    
    def validate_dependencies(self):
        """Check if all required dependencies are installed."""
        print("\n📦 Checking Dependencies...")
        
        if not DEPENDENCIES_AVAILABLE:
            print(f"❌ Missing dependencies: {MISSING_DEPS}")
            self.results['dependencies'] = False
            return
        
        # Check specific versions
        required_packages = {
            'mlflow': '2.8.1',
            'torch': None,  # Any version
            'boto3': None,
            'pytest': None
        }
        
        all_present = True
        for package, required_version in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package} is installed")
            except ImportError:
                print(f"❌ {package} is missing")
                all_present = False
        
        self.results['dependencies'] = all_present
    
    def validate_configuration(self):
        """Validate MLflow configuration files."""
        print("\n⚙️  Checking Configuration...")
        
        config_files = [
            'configs/mlflow_config.yaml',
            'configs/cloud_deployment_config.yaml'
        ]
        
        all_configs_valid = True
        
        for config_file in config_files:
            config_path = Path(__file__).parent.parent / config_file
            
            if not config_path.exists():
                print(f"❌ Missing config file: {config_file}")
                all_configs_valid = False
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Basic validation
                if config_file.endswith('mlflow_config.yaml'):
                    required_keys = ['tracking_uri', 'experiment', 'auto_logging']
                    if all(key in config for key in required_keys):
                        print(f"✅ {config_file} is valid")
                    else:
                        print(f"❌ {config_file} missing required keys")
                        all_configs_valid = False
                else:
                    print(f"✅ {config_file} exists and is readable")
                    
            except Exception as e:
                print(f"❌ Error reading {config_file}: {e}")
                all_configs_valid = False
        
        self.results['configuration'] = all_configs_valid
    
    def validate_mlflow_functionality(self):
        """Test basic MLflow functionality."""
        print("\n🧪 Testing MLflow Functionality...")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Skipping MLflow tests - dependencies not available")
            self.results['mlflow_functionality'] = False
            return
        
        try:
            # Create temporary MLflow setup
            self.temp_dir = tempfile.mkdtemp()
            tracking_uri = f"file://{self.temp_dir}/mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            
            # Test experiment creation
            experiment_id = mlflow.create_experiment("test_validation_experiment")
            print("✅ Experiment creation works")
            
            # Test run creation and logging
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log parameters
                mlflow.log_param("test_param", "test_value")
                
                # Log metrics
                mlflow.log_metric("test_metric", 0.85)
                
                # Log a simple model
                model = torch.nn.Linear(10, 1)
                mlflow.pytorch.log_model(model, "test_model")
                
                print("✅ Parameter and metric logging works")
                print("✅ Model logging works")
            
            # Test model loading
            model_uri = f"runs:/{run.info.run_id}/test_model"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            print("✅ Model loading works")
            
            self.results['mlflow_functionality'] = True
            
        except Exception as e:
            print(f"❌ MLflow functionality test failed: {e}")
            self.results['mlflow_functionality'] = False
        
        finally:
            if self.temp_dir:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_unit_tests(self):
        """Run unit tests for MLflow integration."""
        print("\n🧪 Running Unit Tests...")
        
        test_path = Path(__file__).parent.parent / "tests" / "unit" / "test_mlflow_integration.py"
        
        if not test_path.exists():
            print(f"❌ Unit test file not found: {test_path}")
            self.results['unit_tests'] = False
            return
        
        try:
            # Run pytest on unit tests
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                str(test_path), 
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ Unit tests passed")
                self.results['unit_tests'] = True
            else:
                print(f"❌ Unit tests failed:")
                print(result.stdout)
                print(result.stderr)
                self.results['unit_tests'] = False
                
        except subprocess.TimeoutExpired:
            print("❌ Unit tests timed out")
            self.results['unit_tests'] = False
        except Exception as e:
            print(f"❌ Error running unit tests: {e}")
            self.results['unit_tests'] = False
    
    def run_integration_tests(self):
        """Run integration tests for MLflow."""
        print("\n🔗 Running Integration Tests...")
        
        test_path = Path(__file__).parent.parent / "tests" / "integration" / "test_mlflow_integration.py"
        
        if not test_path.exists():
            print(f"❌ Integration test file not found: {test_path}")
            self.results['integration_tests'] = False
            return
        
        try:
            # Run pytest on integration tests with longer timeout
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                str(test_path), 
                '-v', '--tb=short', '-x'  # Stop on first failure
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Integration tests passed")
                self.results['integration_tests'] = True
            else:
                print(f"⚠️  Integration tests had issues:")
                # Show only summary for brevity
                lines = result.stdout.split('\n')
                summary_started = False
                for line in lines:
                    if 'FAILURES' in line or 'ERRORS' in line or summary_started:
                        summary_started = True
                        print(line)
                self.results['integration_tests'] = False
                
        except subprocess.TimeoutExpired:
            print("❌ Integration tests timed out")
            self.results['integration_tests'] = False
        except Exception as e:
            print(f"❌ Error running integration tests: {e}")
            self.results['integration_tests'] = False
    
    def validate_deployment_preparation(self):
        """Test deployment preparation functionality."""
        print("\n🚀 Testing Deployment Preparation...")
        
        # Check if deployment script exists
        deploy_script = Path(__file__).parent.parent / "scripts" / "deploy_model.py"
        
        if not deploy_script.exists():
            print(f"❌ Deployment script not found: {deploy_script}")
            self.results['deployment'] = False
            return
        
        try:
            # Test deployment script help
            result = subprocess.run([
                sys.executable, str(deploy_script), '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and 'deployment-type' in result.stdout:
                print("✅ Deployment script is functional")
                self.results['deployment'] = True
            else:
                print(f"❌ Deployment script help failed")
                self.results['deployment'] = False
                
        except Exception as e:
            print(f"❌ Error testing deployment script: {e}")
            self.results['deployment'] = False
    
    def display_results(self):
        """Display validation results summary."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION RESULTS SUMMARY")
        print("=" * 60)
        
        total_checks = len(self.results)
        passed_checks = sum(self.results.values())
        
        for check, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check.replace('_', ' ').title():<25} {status}")
        
        print("-" * 60)
        print(f"Overall: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("🎉 All validations passed! MLflow integration is ready.")
        else:
            print("⚠️  Some validations failed. Check the logs above for details.")
            print("\n💡 Troubleshooting tips:")
            
            if not self.results.get('dependencies', True):
                print("- Install missing dependencies: pip install -r requirements/base.txt")
            
            if not self.results.get('configuration', True):
                print("- Check MLflow configuration files in configs/")
            
            if not self.results.get('mlflow_functionality', True):
                print("- Verify MLflow installation: pip install mlflow==2.8.1")
                print("- Check MLflow server accessibility if using remote tracking")
            
            if not self.results.get('unit_tests', True):
                print("- Review unit test failures and fix underlying issues")
            
            if not self.results.get('integration_tests', True):
                print("- Check integration test logs for specific failure points")
                print("- Ensure test environment has sufficient resources")


def main():
    """Main validation function."""
    print("🚀 MLflow Integration Validation Tool")
    print("This tool validates the MLflow integration setup for the MEGAN project.")
    
    validator = MLflowIntegrationValidator()
    results = validator.run_all_validations()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
