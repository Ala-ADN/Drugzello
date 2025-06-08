#!/bin/bash

# Script to set up monitoring and logging for Azure deployment

set -e  # Exit on any error

# Configuration
RESOURCE_GROUP="drugzello-rg"
CONTAINER_APP_NAME="drugzello-api"
LOG_ANALYTICS_WORKSPACE="drugzello-logs"
ALERT_RULE_NAME="drugzello-high-cpu"

# Print header
echo "====================================="
echo "Drugzello Monitoring and Logging Setup"
echo "====================================="
echo

# Login to Azure (if not already logged in)
echo "Checking Azure login status..."
az account show > /dev/null 2>&1 || az login

# Create Log Analytics Workspace if it doesn't exist
echo "Creating Log Analytics workspace if it doesn't exist..."
if ! az monitor log-analytics workspace show --resource-group $RESOURCE_GROUP --workspace-name $LOG_ANALYTICS_WORKSPACE &>/dev/null; then
    echo "Creating Log Analytics workspace $LOG_ANALYTICS_WORKSPACE..."
    az monitor log-analytics workspace create \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $LOG_ANALYTICS_WORKSPACE
else
    echo "Log Analytics workspace already exists."
fi

# Get Log Analytics workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --query customerId -o tsv)

# Enable container insights
echo "Setting up container insights..."
az monitor diagnostic-settings create \
    --name "ContainerInsights" \
    --resource $(az container show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query id -o tsv) \
    --workspace $WORKSPACE_ID \
    --logs '[{"category": "ContainerInstanceLog","enabled": true}]' \
    --metrics '[{"category": "AllMetrics","enabled": true}]'

# Create CPU alert rule
echo "Creating CPU utilization alert rule..."
az monitor metrics alert create \
    --name $ALERT_RULE_NAME \
    --resource-group $RESOURCE_GROUP \
    --scopes $(az container show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query id -o tsv) \
    --condition "avg Percentage CPU > 80" \
    --description "Alert when CPU usage exceeds 80% for 5 minutes" \
    --evaluation-frequency 1m \
    --window-size 5m \
    --severity 2

echo
echo "Monitoring setup complete!"
echo "Log Analytics Workspace: $LOG_ANALYTICS_WORKSPACE"
echo "Alert Rule: $ALERT_RULE_NAME"
