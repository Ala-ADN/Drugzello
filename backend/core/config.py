from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "MEGAN Inference API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Model Settings
    model_path: Optional[str] = None
    model_type: str = "megan"
    batch_size: int = 1
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # MLflow Settings
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "megan_inference"
      # Logging
    log_level: str = "INFO"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "MEGAN_",
        "protected_namespaces": ()
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        # Set default model path if not provided
        if not self.model_path:
            self.model_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "models", 
                "trained",
                "best_model.pth"
            )

# Global settings instance
settings = Settings()
