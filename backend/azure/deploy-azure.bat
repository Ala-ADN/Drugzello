@echo off
REM Azure deployment script for Drugzello backend (Batch version)

setlocal EnableDelayedExpansion

REM Default parameters
set ResourceGroup=drugzello-rg
set Location=eastus
set ACRName=drugzelloregistry
set ContainerGroupName=drugzello-backend
set ImageName=drugzello-backend
set ImageTag=latest

REM Parse command-line arguments (if provided)
:parse_args
if "%~1"=="" goto :end_parse_args
set arg=%~1
if "%arg:~0,2%"=="-R" (
    set ResourceGroup=%arg:~2%
) else if "%arg:~0,2%"=="-L" (
    set Location=%arg:~2%
) else if "%arg:~0,2%"=="-A" (
    set ACRName=%arg:~2%
) else if "%arg:~0,2%"=="-C" (
    set ContainerGroupName=%arg:~2%
) else if "%arg:~0,2%"=="-I" (
    set ImageName=%arg:~2%
) else if "%arg:~0,2%"=="-T" (
    set ImageTag=%arg:~2%
)
shift
goto :parse_args
:end_parse_args

REM Function to display status messages
call :write_status "Starting Azure deployment for Drugzello backend"

REM Check if Azure CLI is installed
where az >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :write_error "Azure CLI is not installed. Please install it first."
    exit /b 1
)

REM Check Azure login status
call :write_status "Checking Azure login status..."
az account show >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :write_status "Please login to Azure..."
    az login
)

REM Create resource group
call :write_status "Creating resource group: %ResourceGroup%"
call az group create --name %ResourceGroup% --location %Location%

REM Create Azure Container Registry
call :write_status "Creating Azure Container Registry: %ACRName%"
call az acr create --resource-group %ResourceGroup% --name %ACRName% --sku Basic --admin-enabled true

REM Get ACR login server
for /f "tokens=*" %%a in ('az acr show --name %ACRName% --resource-group %ResourceGroup% --query "loginServer" --output tsv') do set ACRLoginServer=%%a
call :write_status "ACR Login Server: %ACRLoginServer%"

REM Build and push Docker image
call :write_status "Building Docker image..."
docker build -f Dockerfile.azure -t "%ACRLoginServer%/%ImageName%:%ImageTag%" .

call :write_status "Logging into ACR..."
az acr login --name %ACRName%

call :write_status "Pushing image to ACR..."
docker push "%ACRLoginServer%/%ImageName%:%ImageTag%"

REM Create storage account for MLflow artifacts
set StorageAccountName=drugzellosa%RANDOM%%RANDOM%
call :write_status "Creating storage account: %StorageAccountName%"
call az storage account create ^
    --name %StorageAccountName% ^
    --resource-group %ResourceGroup% ^
    --location %Location% ^
    --sku Standard_LRS

REM Get storage connection string
for /f "tokens=*" %%a in ('az storage account show-connection-string --name %StorageAccountName% --resource-group %ResourceGroup% --output tsv') do set StorageConnectionString=%%a

REM Create blob container for MLflow artifacts
call :write_status "Creating blob container for MLflow artifacts..."
call az storage container create ^
    --name mlflow-artifacts ^
    --connection-string "%StorageConnectionString%"

REM Deploy using ARM template
call :write_status "Deploying containers using ARM template..."
call az deployment group create ^
    --resource-group %ResourceGroup% ^
    --template-file azure\arm-template.json ^
    --parameters ^
        containerGroupName=%ContainerGroupName% ^
        imageRegistry=%ACRLoginServer% ^
        imageTag=%ImageTag%

REM Get the public IP/FQDN
for /f "tokens=*" %%a in ('az container show --resource-group %ResourceGroup% --name %ContainerGroupName% --query ipAddress.fqdn --output tsv') do set ContainerFQDN=%%a

call :write_status "Deployment completed successfully!"
call :write_status "Application URL: http://%ContainerFQDN%"
call :write_status "Health check: http://%ContainerFQDN%/health"
call :write_status "API docs: http://%ContainerFQDN%/docs"

REM Save deployment info as JSON
echo { > deployment-info.json
echo   "resource_group": "%ResourceGroup%", >> deployment-info.json
echo   "container_group": "%ContainerGroupName%", >> deployment-info.json
echo   "acr_name": "%ACRName%", >> deployment-info.json
echo   "acr_login_server": "%ACRLoginServer%", >> deployment-info.json
echo   "storage_account": "%StorageAccountName%", >> deployment-info.json
echo   "application_url": "http://%ContainerFQDN%", >> deployment-info.json
echo   "deployed_at": "%DATE% %TIME%" >> deployment-info.json
echo } >> deployment-info.json

call :write_status "Deployment information saved to deployment-info.json"
goto :eof

:write_status
echo [INFO] %~1
goto :eof

:write_warning
echo [WARNING] %~1
goto :eof

:write_error
echo [ERROR] %~1
goto :eof
