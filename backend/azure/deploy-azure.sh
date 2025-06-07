#!/bin/bash

# Azure deployment script for Drugzello backend

set -e

# Configuration
RESOURCE_GROUP="drugzello-rg"
LOCATION="eastus"
ACR_NAME="drugzelloregistry"
CONTAINER_GROUP_NAME="drugzello-backend"
IMAGE_NAME="drugzello-backend"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

# Login to Azure
print_status "Checking Azure login status..."
if ! az account show &> /dev/null; then
    print_status "Please login to Azure..."
    az login
fi

# Create resource group
print_status "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
print_status "Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
print_status "ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push Docker image
print_status "Building Docker image..."
docker build -f Dockerfile.azure -t $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG .

print_status "Logging into ACR..."
az acr login --name $ACR_NAME

print_status "Pushing image to ACR..."
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG

# Create storage account for MLflow artifacts
STORAGE_ACCOUNT_NAME="drugzellosa$(date +%s)"
print_status "Creating storage account: $STORAGE_ACCOUNT_NAME"
az storage account create \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS

# Get storage connection string
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT_NAME \
    --resource-group $RESOURCE_GROUP \
    --output tsv)

# Create blob container for MLflow artifacts
print_status "Creating blob container for MLflow artifacts..."
az storage container create \
    --name mlflow-artifacts \
    --connection-string "$STORAGE_CONNECTION_STRING"

# Deploy using ARM template
print_status "Deploying containers using ARM template..."
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file azure/arm-template.json \
    --parameters \
        containerGroupName=$CONTAINER_GROUP_NAME \
        imageRegistry=$ACR_LOGIN_SERVER \
        imageTag=$IMAGE_TAG

# Get the public IP/FQDN
CONTAINER_FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_GROUP_NAME \
    --query ipAddress.fqdn \
    --output tsv)

print_status "Deployment completed successfully!"
print_status "Application URL: http://$CONTAINER_FQDN"
print_status "Health check: http://$CONTAINER_FQDN/health"
print_status "API docs: http://$CONTAINER_FQDN/docs"

# Save deployment info
cat > deployment-info.json << EOF
{
    "resource_group": "$RESOURCE_GROUP",
    "container_group": "$CONTAINER_GROUP_NAME",
    "acr_name": "$ACR_NAME",
    "acr_login_server": "$ACR_LOGIN_SERVER",
    "storage_account": "$STORAGE_ACCOUNT_NAME",
    "application_url": "http://$CONTAINER_FQDN",
    "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

print_status "Deployment information saved to deployment-info.json"
