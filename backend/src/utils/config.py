"""
Configuration classes for MEGAN model training and hyperparameter management.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
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
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 200
    patience: int = 20  # Early stopping patience
    
    # Loss function weights
    gamma_exp: float = 0.1      # Explanation loss weight
    beta_sparsity: float = 0.01  # Sparsity regularization
    delta_decor: float = 0.05    # Decorrelation regularization
    
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


@dataclass
class SearchSpace:
    """
    Define hyperparameter search spaces for MEGAN hyperparameter optimization.
    
    This class allows for flexible hyperparameter tuning by specifying lists of
    possible values for each hyperparameter. The search space can be customized
    by providing specific values or left as None to use default ranges.
    """
    
    # Architecture parameters
    hidden_channels: Optional[List[int]] = None
    num_layers: Optional[List[int]] = None
    K: Optional[List[int]] = None
    heads_gat: Optional[List[int]] = None
    
    # Training parameters
    learning_rate: Optional[List[float]] = None
    weight_decay: Optional[List[float]] = None
    dropout: Optional[List[float]] = None
    batch_size: Optional[List[int]] = None
    
    # Regularization parameters
    gamma_exp: Optional[List[float]] = None
    beta_sparsity: Optional[List[float]] = None
    delta_decor: Optional[List[float]] = None
    
    # Boolean parameters
    use_edge_features: Optional[List[bool]] = None
    layer_norm: Optional[List[bool]] = None
    residual: Optional[List[bool]] = None
    
    def __post_init__(self):
        """Set default search spaces for None values."""
        if self.hidden_channels is None:
            self.hidden_channels = [32, 60, 80, 128]
        
        if self.num_layers is None:
            self.num_layers = [2, 3, 4, 5]
        
        if self.K is None:
            self.K = [2, 3, 4]
        
        if self.heads_gat is None:
            self.heads_gat = [1, 2, 4]
        
        if self.learning_rate is None:
            self.learning_rate = [0.0001, 0.0005, 0.001, 0.005]
        
        if self.weight_decay is None:
            self.weight_decay = [1e-6, 1e-5, 1e-4, 1e-3]
        
        if self.dropout is None:
            self.dropout = [0.0, 0.1, 0.2, 0.3]
        
        if self.batch_size is None:
            self.batch_size = [16, 32, 64]
        
        if self.gamma_exp is None:
            self.gamma_exp = [0.05, 0.1, 0.15, 0.2]
        
        if self.beta_sparsity is None:
            self.beta_sparsity = [0.0, 0.01, 0.05, 0.1]
        
        if self.delta_decor is None:
            self.delta_decor = [0.0, 0.05, 0.1, 0.2]
        
        if self.use_edge_features is None:
            self.use_edge_features = [True, False]
        
        if self.layer_norm is None:
            self.layer_norm = [True, False]
        
        if self.residual is None:
            self.residual = [True, False]
    
    def get_param_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations from the search space.
        
        Returns:
            List of dictionaries, each containing one parameter combination
        """
        from itertools import product
        
        # Get all parameter names and their possible values
        param_names = []
        param_values = []
        
        for field_name, field_value in asdict(self).items():
            if field_value is not None:
                param_names.append(field_name)
                param_values.append(field_value)
        
        # Generate all combinations
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def sample_random_config(self, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Sample a random configuration from the search space.
        
        Args:
            random_seed: Random seed for reproducible sampling
        
        Returns:
            Dictionary with sampled parameter values
        """
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        config = {}
        for field_name, field_value in asdict(self).items():
            if field_value is not None:
                config[field_name] = random.choice(field_value)
        
        return config


# Predefined search spaces for common scenarios
SEARCH_SPACES = {
    "quick": SearchSpace(
        hidden_channels=[32, 60],
        num_layers=[2, 3],
        K=[2, 3],
        learning_rate=[0.001, 0.005],
        dropout=[0.1, 0.2]
    ),
    
    "comprehensive": SearchSpace(
        hidden_channels=[32, 60, 80, 128],
        num_layers=[2, 3, 4, 5],
        K=[2, 3, 4],
        heads_gat=[1, 2, 4],
        learning_rate=[0.0001, 0.0005, 0.001, 0.005],
        weight_decay=[1e-6, 1e-5, 1e-4],
        dropout=[0.0, 0.1, 0.2, 0.3],
        gamma_exp=[0.05, 0.1, 0.15],
        beta_sparsity=[0.0, 0.01, 0.05],
        delta_decor=[0.0, 0.05, 0.1]
    ),
    
    "architecture_focus": SearchSpace(
        hidden_channels=[32, 60, 80, 128, 256],
        num_layers=[2, 3, 4, 5, 6],
        K=[2, 3, 4, 5],
        heads_gat=[1, 2, 4, 8],
        use_edge_features=[True, False],
        layer_norm=[True, False],
        residual=[True, False]
    )
}
