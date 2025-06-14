from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "Drugzello ML API"
    version: str = "1.0.0"
    debug: bool = False
    
    # CORS Settings
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Model Settings
    model_path: Optional[str] = None
    model_type: str = "megan"
    batch_size: int = 1
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Data Settings
    data_path: str = "data"
    
    # MLflow Settings
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "megan_inference"
    
    # Logging
    log_level: str = "INFO"
    
    # Environment
    environment: str = "development"
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "MEGAN_",
        "protected_namespaces": ()
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set default model path if not provided
        if not self.model_path:
            self.model_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "models", 
                "trained",
                "best_model.pth"
            )
        
        # Set default data path if not provided as absolute path
        if not os.path.isabs(self.data_path):
            self.data_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                self.data_path
            )

# Global settings instance
settings = Settings()
