@echo off
setlocal enabledelayedexpansion

echo =====================================
echo Drugzello Azure Deployment
echo =====================================
echo.

:: Check for Azure CLI
where az >nul 2>&1
if %errorlevel% neq 0 (
    echo Azure CLI not found. Please install it first:
    echo Visit: https://aka.ms/installazurecli
    exit /b 1
)

:: Check if logged in to Azure
echo Checking Azure login status...
for /f "tokens=*" %%i in ('az account show --query id -o tsv 2^>nul') do set SUBSCRIPTION_ID=%%i
if "!SUBSCRIPTION_ID!" == "" (
    echo Not logged in to Azure. Please log in now.
    
    :: Try regular login first
    az login
    
    :: Check again if login was successful
    for /f "tokens=*" %%i in ('az account show --query id -o tsv 2^>nul') do set SUBSCRIPTION_ID=%%i
    
    :: If still not logged in, try with device code
    if "!SUBSCRIPTION_ID!" == "" (
        echo Regular login failed. Trying with device code...
        az login --use-device-code
        
        :: Check again if login was successful
        for /f "tokens=*" %%i in ('az account show --query id -o tsv 2^>nul') do set SUBSCRIPTION_ID=%%i
        
        if "!SUBSCRIPTION_ID!" == "" (
            echo ERROR: Failed to log in to Azure.
            echo Please ensure your account has an active Azure subscription.
            echo If you're using an educational account, you may need to:
            echo 1. Create an Azure student account at https://azure.microsoft.com/free/students/
            echo 2. Or use a different account with Azure subscription access
            exit /b 1
        )
    )
)

:: List available subscriptions for user to choose
echo.
echo Available subscriptions:
az account list --output table
echo.

:: Let user select a subscription if there are multiple
for /f "tokens=*" %%c in ('az account list --query "length([])"') do set SUB_COUNT=%%c
if !SUB_COUNT! gtr 1 (
    echo Multiple subscriptions found. Please select one:
    set /p SELECTED_SUB="Enter subscription ID or name: "
    az account set --subscription "!SELECTED_SUB!"
    
    :: Verify selection worked
    for /f "tokens=*" %%i in ('az account show --query id -o tsv') do set SUBSCRIPTION_ID=%%i
    for /f "tokens=*" %%i in ('az account show --query name -o tsv') do set SUB_NAME=%%i
    echo Using subscription: !SUB_NAME! (!SUBSCRIPTION_ID!)
) else (
    for /f "tokens=*" %%i in ('az account show --query name -o tsv') do set SUB_NAME=%%i
    echo Using subscription: !SUB_NAME! (!SUBSCRIPTION_ID!)
)

:: Prompt user for Azure settings if not already set
if "!RESOURCE_GROUP!" == "" (
    set /p "RESOURCE_GROUP=Enter resource group name [drugzello-rg]: "
    if "!RESOURCE_GROUP!" == "" set RESOURCE_GROUP=drugzello-rg
)

if "!LOCATION!" == "" (
    echo Available locations:
    az account list-locations --query "[].name" -o tsv | findstr /B "east west central north south" | findstr /V "france switzerland norway gov" | head -10
    echo ...
    set /p "LOCATION=Enter Azure region [eastus]: "
    if "!LOCATION!" == "" set LOCATION=eastus
)

:: Create resource group if it doesn't exist
echo Creating resource group !RESOURCE_GROUP! if it doesn't exist...
az group create --name "!RESOURCE_GROUP!" --location "!LOCATION!" --output none

:: Create Azure Container Registry
echo Creating Azure Container Registry...
set ACR_NAME=!RESOURCE_GROUP:-=!acr
az acr create --resource-group "!RESOURCE_GROUP!" --name "!ACR_NAME!" --sku Basic --admin-enabled true

:: Get ACR credentials
echo Getting ACR credentials...
for /f "tokens=*" %%i in ('az acr credential show --name "!ACR_NAME!" --query "username" -o tsv') do set ACR_USERNAME=%%i
for /f "tokens=*" %%i in ('az acr credential show --name "!ACR_NAME!" --query "passwords[0].value" -o tsv') do set ACR_PASSWORD=%%i
for /f "tokens=*" %%i in ('az acr show --name "!ACR_NAME!" --query "loginServer" -o tsv') do set ACR_LOGIN_SERVER=%%i

:: Build and push Docker image to ACR
echo Building and pushing Docker image to ACR...
set IMAGE_NAME=!ACR_LOGIN_SERVER!/drugzello-backend:latest
az acr build --registry "!ACR_NAME!" --image drugzello-backend:latest --file backend/Dockerfile.azure backend

:: Create Log Analytics workspace
echo Creating Log Analytics workspace...
set WORKSPACE_NAME=!RESOURCE_GROUP:-=!logs
az monitor log-analytics workspace create --resource-group "!RESOURCE_GROUP!" --workspace-name "!WORKSPACE_NAME!" --output none

:: Get Log Analytics workspace details
for /f "tokens=*" %%i in ('az monitor log-analytics workspace show --resource-group "!RESOURCE_GROUP!" --workspace-name "!WORKSPACE_NAME!" --query customerId --output tsv') do set LOG_ANALYTICS_WORKSPACE_CLIENT_ID=%%i
for /f "tokens=*" %%i in ('az monitor log-analytics workspace get-shared-keys --resource-group "!RESOURCE_GROUP!" --workspace-name "!WORKSPACE_NAME!" --query primarySharedKey --output tsv') do set LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET=%%i

:: Create Container App Environment
echo Creating Container App Environment...
set ENV_NAME=!RESOURCE_GROUP:-=!env
az containerapp env create --resource-group "!RESOURCE_GROUP!" --name "!ENV_NAME!" --location "!LOCATION!" --logs-workspace-id "!LOG_ANALYTICS_WORKSPACE_CLIENT_ID!" --logs-workspace-key "!LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET!"

:: Deploy the Container App
echo Deploying Container App...
az containerapp create --resource-group "!RESOURCE_GROUP!" --name drugzello-api --environment "!ENV_NAME!" --image "!IMAGE_NAME!" --target-port 8000 --ingress external --min-replicas 1 --max-replicas 5 --cpu 0.5 --memory 1.0Gi --registry-server "!ACR_LOGIN_SERVER!" --registry-username "!ACR_USERNAME!" --registry-password "!ACR_PASSWORD!" --env-vars "ENVIRONMENT=production" "LOG_LEVEL=INFO"

:: Get the application URL
for /f "tokens=*" %%i in ('az containerapp show --resource-group "!RESOURCE_GROUP!" --name drugzello-api --query properties.configuration.ingress.fqdn --output tsv') do set APP_URL=%%i

echo.
echo =====================================
echo Deployment Complete!
echo =====================================
echo Application URL: https://!APP_URL!
echo.
echo To check logs, run:
echo az containerapp logs show --name drugzello-api --resource-group !RESOURCE_GROUP!
