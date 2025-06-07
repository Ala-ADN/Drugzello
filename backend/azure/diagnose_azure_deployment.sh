#!/bin/bash
# Azure deployment diagnostics script for Drugzello ML Backend
# This script helps troubleshoot Azure deployments by checking common issues

set -e

# Configuration (modify as needed)
RESOURCE_GROUP="drugzello-rg"
CONTAINER_GROUP="drugzello-backend"
ACR_NAME="drugzelloregistry"
STORAGE_ACCOUNT_NAME="drugzellosa"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================================${NC}\n"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if Azure CLI is installed
print_header "Checking Azure CLI Installation"
if command -v az &> /dev/null; then
    AZ_VERSION=$(az version | jq '."azure-cli"' -r)
    print_status "Azure CLI is installed (version $AZ_VERSION)"
else
    print_error "Azure CLI is not installed. Please install it and try again."
    exit 1
fi

# Check Azure login status
print_header "Checking Azure Login Status"
if ! az account show &> /dev/null; then
    print_warning "Not logged in to Azure. Please login."
    az login
else
    ACCOUNT_NAME=$(az account show --query 'user.name' -o tsv)
    SUBSCRIPTION=$(az account show --query 'name' -o tsv)
    print_status "Logged in to Azure as $ACCOUNT_NAME"
    print_status "Using subscription: $SUBSCRIPTION"
fi

# Check resource group
print_header "Checking Resource Group"
if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    LOCATION=$(az group show --name "$RESOURCE_GROUP" --query 'location' -o tsv)
    print_status "Resource group $RESOURCE_GROUP exists in $LOCATION"
else
    print_error "Resource group $RESOURCE_GROUP does not exist"
fi

# Check container registry
print_header "Checking Container Registry"
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    ACR_URL=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query 'loginServer' -o tsv)
    print_status "Container registry $ACR_NAME exists ($ACR_URL)"
    
    # Check our image in the registry
    print_status "Checking for drugzello-backend image in registry..."
    if az acr repository show --name "$ACR_NAME" --repository drugzello-backend &> /dev/null; then
        TAGS=$(az acr repository show-tags --name "$ACR_NAME" --repository drugzello-backend --output tsv)
        print_status "Image found. Available tags: $TAGS"
    else
        print_error "Image drugzello-backend not found in registry"
    fi
else
    print_error "Container registry $ACR_NAME does not exist"
fi

# Check container group deployment
print_header "Checking Container Group"
if az container show --name "$CONTAINER_GROUP" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    STATUS=$(az container show --name "$CONTAINER_GROUP" --resource-group "$RESOURCE_GROUP" --query 'instanceView.state' -o tsv)
    FQDN=$(az container show --name "$CONTAINER_GROUP" --resource-group "$RESOURCE_GROUP" --query 'ipAddress.fqdn' -o tsv 2>/dev/null || echo "N/A")
    IP=$(az container show --name "$CONTAINER_GROUP" --resource-group "$RESOURCE_GROUP" --query 'ipAddress.ip' -o tsv 2>/dev/null || echo "N/A")
    
    print_status "Container group $CONTAINER_GROUP exists (Status: $STATUS)"
    print_status "FQDN: $FQDN"
    print_status "IP Address: $IP"
    
    # Check container logs
    print_status "Fetching recent container logs..."
    az container logs --name "$CONTAINER_GROUP" --resource-group "$RESOURCE_GROUP" --tail 20
else
    print_error "Container group $CONTAINER_GROUP does not exist"
fi

# Check storage account
print_header "Checking Storage Account"
if az storage account show --name "$STORAGE_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    print_status "Storage account $STORAGE_ACCOUNT_NAME exists"
    
    # Get storage account key
    STORAGE_KEY=$(az storage account keys list --account-name "$STORAGE_ACCOUNT_NAME" --resource-group "$RESOURCE_GROUP" --query '[0].value' -o tsv)
    
    # List blob containers
    print_status "Listing blob containers..."
    az storage container list --account-name "$STORAGE_ACCOUNT_NAME" --account-key "$STORAGE_KEY" --output table
else
    print_warning "Storage account $STORAGE_ACCOUNT_NAME not found. Trying to find by prefix..."
    FOUND_ACCOUNTS=$(az storage account list --query "[?starts_with(name,'$STORAGE_ACCOUNT_NAME')].name" -o tsv)
    
    if [ -n "$FOUND_ACCOUNTS" ]; then
        print_status "Found similar storage accounts: $FOUND_ACCOUNTS"
    else
        print_error "No storage accounts with prefix $STORAGE_ACCOUNT_NAME found"
    fi
fi

# Check health endpoint
print_header "Checking Health Endpoint"
if [ "$IP" != "N/A" ]; then
    print_status "Testing health endpoint..."
    if command -v curl &> /dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$FQDN/health" || echo "Failed")
        if [ "$HTTP_CODE" == "200" ]; then
            print_status "Health check OK (HTTP 200)"
            print_status "Health endpoint response:"
            curl -s "http://$FQDN/health" | jq '.'
        else
            print_error "Health check failed (HTTP $HTTP_CODE)"
        fi
    else
        print_warning "curl not installed, skipping health check"
    fi
else
    print_warning "No IP address available, skipping health check"
fi

print_header "Diagnostics Complete"
echo -e "If you need help resolving any issues, see the troubleshooting section in docs/azure_deployment_guide.md"
