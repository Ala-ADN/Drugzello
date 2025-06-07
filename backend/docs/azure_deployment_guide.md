# Deploying Drugzello ML Backend to Azure

This guide provides step-by-step instructions for deploying the Drugzello ML Backend to Azure using Container Apps and Nginx.

## Prerequisites

- Azure CLI installed and configured (Download from https://aka.ms/installazurecli)
- Docker installed and configured (Download from https://www.docker.com/products/docker-desktop)
- Git repository with the latest code
- Active Azure subscription

## Setup Steps

### 1. Prepare Environment

First, clone the repository and navigate to the project root:

```bash
git clone https://github.com/your-org/drugzello.git
cd drugzello
```

### 2. Configure Environment Variables

Create a `.env` file in the backend directory based on the example:

```bash
cp backend/.env.example backend/.env
```

Edit the `.env` file to include your Azure-specific settings.

### 3. Build the Docker Image Locally (Optional)

Test the Docker build locally before deploying:

```bash
docker build -f backend/Dockerfile.azure -t drugzello/backend:latest ./backend
```

### 4. Azure Deployment

#### Using the Deployment Scripts

You have multiple deployment options depending on your needs:

##### Container Apps Deployment

For deploying to Azure Container Apps (recommended for production):

```bash
# For Linux/macOS
chmod +x backend/azure/deploy_container_apps.sh
./backend/azure/deploy_container_apps.sh

# For Windows
backend\azure\deploy_container_apps.bat
```

##### ARM Template Deployment

For deploying with Azure Resource Manager templates:

```bash
# For Linux/macOS
chmod +x backend/azure/deploy-azure.sh
./backend/azure/deploy-azure.sh

# For Windows
./backend/azure/deploy-azure.bat
```

#### Azure Resource Manager Template

For consistent and repeatable deployments, you can also use the Azure Resource Manager (ARM) template:

```json
{
    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "contentVersion": "1.0.0.0",
    "parameters": {
        "containerGroupName": {
            "type": "string",
            "defaultValue": "drugzello-backend",
            "metadata": {
                "description": "Name for the container group"
            }
        },
        "location": {
            "type": "string",
            "defaultValue": "[resourceGroup().location]",
            "metadata": {
                "description": "Location for all resources"
            }
        },
        "imageRegistry": {
            "type": "string",
            "defaultValue": "drugzelloregistry.azurecr.io",
            "metadata": {
                "description": "Container registry URL"
            }
        },
        "imageTag": {
            "type": "string",
            "defaultValue": "latest",
            "metadata": {
                "description": "Image tag"
            }
        }
    },
    "variables": {
        "containerName": "drugzello-backend",
        "nginxContainerName": "drugzello-nginx"
    },
    "resources": [
        {
            "type": "Microsoft.ContainerInstance/containerGroups",
            "apiVersion": "2021-09-01",
            "name": "[parameters('containerGroupName')]",
            "location": "[parameters('location')]",
            "properties": {
                "containers": [
                    {
                        "name": "[variables('nginxContainerName')]",
                        "properties": {
                            "image": "nginx:alpine",
                            "ports": [
                                {
                                    "port": 80,
                                    "protocol": "TCP"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": 0.5,
                                    "memoryInGB": 0.5
                                }
                            },
                            "volumeMounts": [
                                {
                                    "name": "nginx-config",
                                    "mountPath": "/etc/nginx/nginx.conf",
                                    "subPath": "nginx.conf"
                                }
                            ]
                        }
                    },
                    {
                        "name": "[variables('containerName')]",
                        "properties": {
                            "image": "[concat(parameters('imageRegistry'), '/drugzello-backend:', parameters('imageTag'))]",
                            "ports": [
                                {
                                    "port": 8000,
                                    "protocol": "TCP"
                                }
                            ],
                            "environmentVariables": [
                                {
                                    "name": "ENVIRONMENT",
                                    "value": "production"
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "value": "INFO"
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": 2,
                                    "memoryInGB": 4
                                }
                            }
                        }
                    }
                ],
                "osType": "Linux",
                "restartPolicy": "Always",
                "ipAddress": {
                    "type": "Public",
                    "ports": [
                        {
                            "protocol": "TCP",
                            "port": 80
                        }
                    ],
                    "dnsNameLabel": "[parameters('containerGroupName')]"
                },
                "volumes": [
                    {
                        "name": "nginx-config",
                        "configMap": {
                            "items": [
                                {
                                    "key": "nginx.conf",
                                    "path": "nginx.conf"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    ],
    "outputs": {
        "containerIPv4Address": {
            "type": "string",
            "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups/', parameters('containerGroupName'))).ipAddress.ip]"
        },
        "containerFQDN": {
            "type": "string",
            "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups/', parameters('containerGroupName'))).ipAddress.fqdn]"
        }
    }
}
```

Save this file as `backend/azure/arm-template.json` in your project.

#### Manual Deployment

If you prefer to deploy manually, follow these steps:

1. Login to Azure:
   ```bash
   az login
   ```

2. Create a Resource Group:
   ```bash
   az group create --name drugzello-rg --location eastus
   ```

3. Create an Azure Container Registry:
   ```bash
   az acr create --resource-group drugzello-rg --name drugzelloacr --sku Basic
   ```

4. Build and push Docker images to ACR:
   ```bash
   az acr build --registry drugzelloacr --image drugzello/backend:latest --file backend/Dockerfile.azure .
   ```

5. Create Log Analytics workspace:
   ```bash
   az monitor log-analytics workspace create --resource-group drugzello-rg --workspace-name drugzello-logs
   ```

6. Create Container App Environment:
   ```bash
   # Get Log Analytics workspace details
   LOG_ANALYTICS_WORKSPACE_CLIENT_ID=$(az monitor log-analytics workspace show \
     --resource-group drugzello-rg \
     --workspace-name drugzello-logs \
     --query customerId \
     --output tsv)
   
   LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET=$(az monitor log-analytics workspace get-shared-keys \
     --resource-group drugzello-rg \
     --workspace-name drugzello-logs \
     --query primarySharedKey \
     --output tsv)
   
   # Create the environment
   az containerapp env create \
     --resource-group drugzello-rg \
     --name drugzello-env \
     --location eastus \
     --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_CLIENT_ID \
     --logs-workspace-key $LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET
   ```

7. Deploy the Container App:
   ```bash
   az containerapp create \
     --resource-group drugzello-rg \
     --name drugzello-api \
     --environment drugzello-env \
     --image "drugzelloacr.azurecr.io/drugzello/backend:latest" \
     --target-port 8000 \
     --ingress external \
     --min-replicas 1 \
     --max-replicas 5 \
     --cpu 0.5 \
     --memory 1.0Gi \
     --registry-server "drugzelloacr.azurecr.io" \
     --env-vars "ENVIRONMENT=production" "LOG_LEVEL=INFO"
   ```

### 5. Set Up Monitoring and Logging

Execute the monitoring setup script:

```bash
# For Linux/macOS
chmod +x backend/azure/setup_monitoring.sh
./backend/azure/setup_monitoring.sh

# For Windows
backend\azure\setup_monitoring.sh
```

### 6. Environment Configuration

Create a production-ready environment configuration:

```bash
# For Linux/macOS
cp backend/.env.azure backend/.env
```

For Windows:
```powershell
Copy-Item -Path backend/.env.azure -Destination backend/.env
```

Edit the `.env` file to include your actual Azure-specific settings:
```
# Production environment variables for Azure deployment

ENVIRONMENT=production
LOG_LEVEL=INFO

# Database (use Azure Database for PostgreSQL)
DATABASE_URL=postgresql://username:password@server.postgres.database.azure.com:5432/drugzello

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=azure://mlflow-artifacts

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string_here

# Security
SECRET_KEY=your_super_secret_key_here
ALLOWED_HOSTS=your-app-domain.com,*.azurecontainer.io

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
```

### 7. SSL Configuration

For production deployments, you have three options:

1. **Azure-managed certificates**: Recommended for production.
2. **Let's Encrypt**: Good for custom domains.
3. **Self-signed certificates**: Only for development/testing.

To generate self-signed certificates for testing:

```bash
# For Linux/macOS
chmod +x backend/azure/generate_ssl.sh
./backend/azure/generate_ssl.sh your-domain.com

# For Windows
bash backend/azure/generate_ssl.sh your-domain.com
```

## Alternative Deployment Options

### Using Azure CLI Scripts

We've provided additional Azure CLI scripts for deployment in the `azure` directory:

1. For Linux/macOS:
```bash
chmod +x backend/azure/deploy-azure.sh
./backend/azure/deploy-azure.sh
```

2. For Windows:
```batch
.\backend\azure\deploy-azure.bat
```

### Post-Deployment Steps

1. **Configure DNS** (Optional):
   
   ```bash
   # Add custom domain
   az network dns record-set cname set-record \
     --resource-group drugzello-rg \
     --zone-name yourdomain.com \
     --record-set-name api \
     --cname your-container-group.eastus.azurecontainer.io
   ```

2. **Set up SSL Certificate** (Recommended):
   - Use Azure Application Gateway with SSL termination
   - Or configure Let's Encrypt in Nginx

3. **Monitor the Application**:
   
   ```bash
   # Check container logs
   az container logs --resource-group drugzello-rg --name drugzello-backend
   
   # Monitor container status
   az container show --resource-group drugzello-rg --name drugzello-backend
   ```

4. **Scale if needed**:
   
   ```bash
   # Update container resources
   az container update \
     --resource-group drugzello-rg \
     --name drugzello-backend \
     --cpu 4 \
     --memory 8
   ```

## Troubleshooting

### Common Issues

1. **Container fails to start**:
   - Check logs: `az containerapp logs show --name drugzello-api --resource-group drugzello-rg`
   - Verify environment variables

2. **SSL issues**:
   - Verify certificate paths in the Nginx configuration
   - Check certificate expiration dates

3. **Performance issues**:
   - Check CPU and memory usage in Azure Monitor
   - Consider scaling up resources

## Maintenance

### Updates and Rollbacks

To update your deployment:

```bash
# Build and push a new image version
az acr build --registry drugzelloacr --image drugzello/backend:v2 --file backend/Dockerfile.azure .

# Update the container app
az containerapp update \
  --name drugzello-api \
  --resource-group drugzello-rg \
  --image "drugzelloacr.azurecr.io/drugzello/backend:v2"
```

To rollback to a previous version:

```bash
az containerapp update \
  --name drugzello-api \
  --resource-group drugzello-rg \
  --image "drugzelloacr.azurecr.io/drugzello/backend:previous-version"
```

## Security Best Practices

1. Store secrets in Azure Key Vault
2. Use Azure Managed Identities for authentication
3. Regularly update dependencies and Docker base images
4. Enable Azure Security Center for your resources
5. Set up network security groups to restrict access

## Cost Optimization

1. Use consumption plan for low-traffic applications
2. Set appropriate scaling limits
3. Use Azure Advisor for cost recommendations
4. Consider reserved instances for stable workloads

## Deployment Summary

This setup provides you with:
- ✅ *Containerized ML backend* with FastAPI
- ✅ *Nginx reverse proxy* with rate limiting
- ✅ *MLflow integration* for experiment tracking
- ✅ *Azure Container Registry* for image management
- ✅ *Azure Blob Storage* for artifacts
- ✅ *Health checks* and monitoring
- ✅ *Production-ready configuration*
