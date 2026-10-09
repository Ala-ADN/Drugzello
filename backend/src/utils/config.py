"""
Configuration classes for MEGAN model training and hyperparameter management.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml
from pathlib import Path


@dataclass
class MEGANConfig:
    """
    Configuration class for MEGAN model training.
    Supports different predefined configurations and custom parameter overrides.
    """
    # Model architecture
    hidden_channels: int = 60
    num_layers: int = 4
    K: int = 2  # Number of explanation heads
    heads_gat: int = 1
    use_edge_features: bool = False
    dropout: float = 0.1
    layer_norm: bool = True
    residual: bool = True
    feature_encoding: str = "ordinal"
    embedding_dim: int = 16
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 200
    patience: int = 20  # Early stopping patience
    training_schedule: str = "joint"
    scheduler_metric: str = "total"
    
    # Loss function weights
    gamma_exp: float = 0.1      # Explanation loss weight
    beta_sparsity: float = 0.01  # Sparsity regularization
    delta_decor: float = 0.0    # Decorrelation regularization
    
    # Data processing
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Model saving and logging
    save_model: bool = True
    model_save_path: str = "models/trained"
    log_interval: int = 10
    
    def __init__(self, config_name: str = "default", **kwargs):
        """
        Initialize configuration with predefined settings or custom parameters.
        
        Args:
            config_name: Name of predefined configuration
            **kwargs: Custom parameter overrides
        """
        # Load predefined configuration
        self._load_predefined_config(config_name)
        
        # Apply any custom overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _load_predefined_config(self, config_name: str):
        """Load predefined configuration settings."""
        configs = {
            "default": {
                "hidden_channels": 60,
                "num_layers": 4,
                "K": 2,
                "learning_rate": 0.001,
                "dropout": 0.1,
            },
            "small": {
                "hidden_channels": 32,
                "num_layers": 2,
                "K": 2,
                "learning_rate": 0.005,
                "dropout": 0.2,
            },
            "large": {
                "hidden_channels": 128,
                "num_layers": 6,
                "K": 4,
                "learning_rate": 0.0005,
                "dropout": 0.1,
            },
            "edge_focused": {
                "hidden_channels": 80,
                "num_layers": 4,
                "K": 3,
                "use_edge_features": True,
                "learning_rate": 0.001,
                "dropout": 0.15,
                "gamma_exp": 0.15,
            },
            "regularized": {
                "hidden_channels": 60,
                "num_layers": 4,
                "K": 2,
                "learning_rate": 0.001,
                "dropout": 0.1,
                "beta_sparsity": 0.05,
                "delta_decor": 0.1,
            }
        }
        
        if config_name in configs:
            config_dict = configs[config_name]
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MEGANConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
