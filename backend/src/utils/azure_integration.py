"""
Azure integration utilities for Drugzello ML Backend.
Provides functionality for Azure Storage, Key Vault, and Container Apps.
"""
import os
import logging
from typing import Dict, Any, Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)

def get_azure_credential() -> DefaultAzureCredential:
    """
    Gets the default Azure credential using the DefaultAzureCredential provider.
    
    Returns:
        DefaultAzureCredential: The Azure credential object
    """
    try:
        credential = DefaultAzureCredential()
        return credential
    except Exception as e:
        logger.error(f"Failed to get Azure credentials: {e}")
        raise

def get_keyvault_secret(vault_url: str, secret_name: str) -> Optional[str]:
    """
    Gets a secret from Azure Key Vault.
    
    Args:
        vault_url (str): The URL of the Azure Key Vault
        secret_name (str): The name of the secret to retrieve
        
    Returns:
        Optional[str]: The secret value if found, None otherwise
    """
    try:
        credential = get_azure_credential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        return client.get_secret(secret_name).value
    except ResourceNotFoundError:
        logger.warning(f"Secret {secret_name} not found in Key Vault {vault_url}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name} from Key Vault: {e}")
        return None

def load_secrets_to_env(vault_url: str, secret_names: list) -> None:
    """
    Loads secrets from Azure Key Vault into environment variables.
    
    Args:
        vault_url (str): The URL of the Azure Key Vault
        secret_names (list): List of secret names to retrieve
    """
    for secret_name in secret_names:
        value = get_keyvault_secret(vault_url, secret_name)
        if value:
            os.environ[secret_name] = value
            logger.info(f"Loaded secret {secret_name} into environment")
        else:
            logger.warning(f"Failed to load secret {secret_name}")

def upload_to_blob_storage(container_name: str, blob_name: str, data: bytes, 
                          connection_string: Optional[str] = None) -> str:
    """
    Uploads data to Azure Blob Storage.
    
    Args:
        container_name (str): The container name
        blob_name (str): The blob name/path
        data (bytes): The data to upload
        connection_string (Optional[str]): Connection string. If None, will use 
                                          AZURE_STORAGE_CONNECTION_STRING env var
                                          
    Returns:
        str: URL of the uploaded blob
    """
    conn_str = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("No Azure Storage connection string provided")
    
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
            
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload data
        blob_client.upload_blob(data, overwrite=True)
        
        logger.info(f"Uploaded blob {blob_name} to container {container_name}")
        return blob_client.url
        
    except Exception as e:
        logger.error(f"Error uploading to blob storage: {e}")
        raise

def download_from_blob_storage(container_name: str, blob_name: str, 
                              connection_string: Optional[str] = None) -> bytes:
    """
    Downloads data from Azure Blob Storage.
    
    Args:
        container_name (str): The container name
        blob_name (str): The blob name/path
        connection_string (Optional[str]): Connection string. If None, will use 
                                          AZURE_STORAGE_CONNECTION_STRING env var
                                          
    Returns:
        bytes: The downloaded data
    """
    conn_str = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("No Azure Storage connection string provided")
    
    try:
        # Create the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Download data
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        
        logger.info(f"Downloaded blob {blob_name} from container {container_name}")
        return data
        
    except Exception as e:
        logger.error(f"Error downloading from blob storage: {e}")
        raise
