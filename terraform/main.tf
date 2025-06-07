# Terraform configuration for Drugzello ML Backend Azure deployment

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  backend "azurerm" {
    resource_group_name  = "tf-state-rg"
    storage_account_name = "drugzellotfstate"
    container_name       = "tfstate"
    key                  = "terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# Variables
variable "resource_group_name" {
  description = "Name of the resource group"
  default     = "drugzello-rg"
}

variable "location" {
  description = "Azure region for resources"
  default     = "eastus"
}

variable "acr_name" {
  description = "Name of the Azure Container Registry"
  default     = "drugzelloregistry"
}

variable "container_image_tag" {
  description = "Tag for the Docker image"
  default     = "latest"
}

variable "environment" {
  description = "Deployment environment"
  default     = "production"
}

# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
}

# Azure Container Registry
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

# Storage Account for MLflow artifacts
resource "azurerm_storage_account" "storage" {
  name                     = "drugzellosa${formatdate("YYYYMMDDhhmmss", timestamp())}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "container" {
  name                  = "mlflow-artifacts"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "drugzello-logs"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# Container Instance - Nginx
resource "azurerm_container_group" "nginx" {
  name                = "drugzello-nginx"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  ip_address_type     = "Public"
  dns_name_label      = "drugzello-api"
  os_type             = "Linux"

  container {
    name   = "nginx"
    image  = "nginx:alpine"
    cpu    = "0.5"
    memory = "0.5"

    ports {
      port     = 80
      protocol = "TCP"
    }
    
    ports {
      port     = 443
      protocol = "TCP"
    }
    
    volume {
      name       = "nginx-config"
      mount_path = "/etc/nginx/nginx.conf"
      
      secret = {
        "nginx.conf" = filebase64("../backend/nginx/nginx.conf")
      }
    }
  }

  tags = {
    environment = var.environment
  }
}

# Container Instance - Backend
resource "azurerm_container_group" "backend" {
  name                = "drugzello-backend"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  ip_address_type     = "Private"
  os_type             = "Linux"

  container {
    name   = "backend"
    image  = "${azurerm_container_registry.acr.login_server}/drugzello-backend:${var.container_image_tag}"
    cpu    = "2"
    memory = "4"

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      "ENVIRONMENT" = var.environment
      "LOG_LEVEL"   = "INFO"
    }
    
    secure_environment_variables = {
      "DATABASE_URL" = "postgresql://username:password@server.postgres.database.azure.com:5432/drugzello"
      "AZURE_STORAGE_CONNECTION_STRING" = azurerm_storage_account.storage.primary_connection_string
    }
  }

  tags = {
    environment = var.environment
  }
}

# Output values
output "nginx_fqdn" {
  value       = azurerm_container_group.nginx.fqdn
  description = "The FQDN of the Nginx container group"
}

output "acr_login_server" {
  value       = azurerm_container_registry.acr.login_server
  description = "The login server for the Azure Container Registry"
}

output "storage_connection_string" {
  value       = azurerm_storage_account.storage.primary_connection_string
  description = "The connection string for the Azure Storage Account"
  sensitive   = true
}
