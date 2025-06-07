#!/bin/bash

# Display header
echo "====================================="
echo "Drugzello Azure Deployment"
echo "====================================="
echo ""

# Function to check for Azure CLI
check_azure_cli() {
    if ! command -v az &> /dev/null; then
        echo "Azure CLI not found. Please install it first:"
        echo "Visit: https://aka.ms/installazurecli"
        exit 1
    fi
}

# Function for Azure login
handle_azure_login() {
    echo "Checking Azure login status..."
    
    # First try to get current account info
    SUBSCRIPTION_ID=$(az account show --query id -o tsv 2>/dev/null)
    
    if [ -z "$SUBSCRIPTION_ID" ]; then
        echo "Not logged in to Azure. Please log in now."
        
        # Try regular login first
        az login
        
        # Check again if login was successful
        SUBSCRIPTION_ID=$(az account show --query id -o tsv 2>/dev/null)
        
        # If still not logged in, try with device code
        if [ -z "$SUBSCRIPTION_ID" ]; then
            echo "Regular login failed. Trying with device code..."
            az login --use-device-code
            
            # Check again if login was successful
            SUBSCRIPTION_ID=$(az account show --query id -o tsv 2>/dev/null)
            
            if [ -z "$SUBSCRIPTION_ID" ]; then
                echo "ERROR: Failed to log in to Azure."
                echo "Please ensure your account has an active Azure subscription."
                echo "If you're using an educational account, you may need to:"
                echo "1. Create an Azure student account at https://azure.microsoft.com/free/students/"
                echo "2. Or use a different account with Azure subscription access"
                exit 1
            fi
        fi
    fi
    
    # List available subscriptions for user to choose
    echo ""
    echo "Available subscriptions:"
    az account list --output table
    echo ""
    
    # Let user select a subscription if there are multiple
    SUB_COUNT=$(az account list --query 'length([])')
    if [ "$SUB_COUNT" -gt 1 ]; then
        echo "Multiple subscriptions found. Please select one:"
        read -p "Enter subscription ID or name: " SELECTED_SUB
        az account set --subscription "$SELECTED_SUB"
        
        # Verify selection worked
        SUBSCRIPTION_ID=$(az account show --query id -o tsv)
        SUB_NAME=$(az account show --query name -o tsv)
        echo "Using subscription: $SUB_NAME ($SUBSCRIPTION_ID)"
    else
        SUB_NAME=$(az account show --query name -o tsv)
        echo "Using subscription: $SUB_NAME ($SUBSCRIPTION_ID)"
    fi
}

# Check prerequisites
check_azure_cli
handle_azure_login

# Prompt user for Azure settings if not already set
if [ -z "$RESOURCE_GROUP" ]; then
    read -p "Enter resource group name [drugzello-rg]: " RESOURCE_GROUP
    RESOURCE_GROUP=${RESOURCE_GROUP:-drugzello-rg}
fi

if [ -z "$LOCATION" ]; then
    echo "Available locations:"
    az account list-locations --query "[].name" -o tsv | head -10
    echo "..."
    read -p "Enter Azure region [eastus]: " LOCATION
    LOCATION=${LOCATION:-eastus}
fi

# Create resource group if it doesn't exist
echo "Creating resource group $RESOURCE_GROUP if it doesn't exist..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# Create Azure Container Registry
echo "Creating Azure Container Registry..."
ACR_NAME="${RESOURCE_GROUP//-/}acr"
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true

# Get ACR credentials
echo "Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query "loginServer" -o tsv)

# Build and push Docker image to ACR
echo "Building and pushing Docker image to ACR..."
IMAGE_NAME="$ACR_LOGIN_SERVER/drugzello-backend:latest"
az acr build --registry "$ACR_NAME" --image drugzello-backend:latest --file backend/Dockerfile.azure backend

# Create Log Analytics workspace
echo "Creating Log Analytics workspace..."
WORKSPACE_NAME="${RESOURCE_GROUP//-/}logs"
az monitor log-analytics workspace create --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" --output none

# Get Log Analytics workspace details
LOG_ANALYTICS_WORKSPACE_CLIENT_ID=$(az monitor log-analytics workspace show \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --query customerId \
  --output tsv)

LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET=$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --query primarySharedKey \
  --output tsv)

# Create Container App Environment
echo "Creating Container App Environment..."
ENV_NAME="${RESOURCE_GROUP//-/}env"
az containerapp env create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ENV_NAME" \
  --location "$LOCATION" \
  --logs-workspace-id "$LOG_ANALYTICS_WORKSPACE_CLIENT_ID" \
  --logs-workspace-key "$LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET"

# Deploy the Container App
echo "Deploying Container App..."
az containerapp create \
  --resource-group "$RESOURCE_GROUP" \
  --name drugzello-api \
  --environment "$ENV_NAME" \
  --image "$IMAGE_NAME" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --env-vars "ENVIRONMENT=production" "LOG_LEVEL=INFO"

# Get the application URL
APP_URL=$(az containerapp show \
  --resource-group "$RESOURCE_GROUP" \
  --name drugzello-api \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo ""
echo "====================================="
echo "Deployment Complete!"
echo "====================================="
echo "Application URL: https://$APP_URL"
echo ""
echo "To check logs, run:"
echo "az containerapp logs show --name drugzello-api --resource-group $RESOURCE_GROUP"
