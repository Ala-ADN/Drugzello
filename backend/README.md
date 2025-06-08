# Drugzello ML Backend

This is the machine learning backend for the Drugzello solubility prediction project.

## Project Structure

- `configs/`: Configuration files for different environments and components
- `data/`: Data storage and organization
  - `raw/`: Original, immutable datasets
  - `processed/`: Cleaned and transformed data
  - `features/`: Feature engineering outputs
  - `external/`: External reference data
- `models/`: Model artifacts and registry
  - `trained/`: Saved model files
  - `experiments/`: Experiment tracking
  - `registry/`: Production model registry
- `src/`: Source code modules
  - `data/`: Data processing utilities
  - `models/`: Model implementations
  - `features/`: Feature engineering
  - `utils/`: Common utilities
  - `monitoring/`: Model monitoring and drift detection
- `scripts/`: Executable scripts for training, evaluation, etc.
- `tests/`: Test suite
- `requirements/`: Dependency management
- `docs/`: Documentation

## Getting Started

### Local Development

1. Install dependencies: `pip install -r requirements/development.txt`
2. Configure environment: Copy `.env.example` to `.env` and update settings
3. Run the development server: `uvicorn main:app --reload`
4. Access API documentation at http://localhost:8000/docs

### Docker Development

1. Build and start services: `docker-compose up -d`
2. Access API at http://localhost:8000
3. Access MLflow UI at http://localhost:5000

### Azure Deployment

1. Ensure prerequisites are installed (Azure CLI, Docker)
2. Copy `.env.azure` to `.env` and update with your Azure settings
3. Deploy using the provided scripts in the `azure` directory:
   
   For Container Apps (recommended for production):
   ```bash
   # For Linux/macOS
   chmod +x azure/deploy_container_apps.sh
   ./azure/deploy_container_apps.sh
   
   # For Windows
   .\azure\deploy_container_apps.bat
   ```
   
   For ARM template-based deployment:
   ```bash
   # For Linux/macOS
   chmod +x azure/deploy-azure.sh
   ./azure/deploy-azure.sh
     # For Windows
   .\azure\deploy-azure.bat
   ```

4. Alternative deployment methods:
   - Azure DevOps Pipeline: Use the `azure-pipelines.yml` file
   - Terraform: Use the configuration in the `terraform` directory
   - ARM Template: Use the template in `azure/arm-template.json`

See `docs/azure_deployment_guide.md` for detailed deployment instructions.
3. Run tests: `pytest tests/`
4. Train model: `python scripts/train_model.py`
