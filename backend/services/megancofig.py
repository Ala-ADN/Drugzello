import numpy as np
from itertools import product
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import random
import pandas as pd
import matplotlib.pyplot as plt

class MEGANConfig:
    """Configuration class for MEGAN hyperparameters with presets - ENHANCED VERSION."""

    def __init__(self, preset: str = "default"):
        if preset == "default":
            self.setup_default()
        elif preset == "large":
            self.setup_large()
        elif preset == "small":
            self.setup_small()
        elif preset == "edge_focused":
            self.setup_edge_focused()
        elif preset == "loss_balanced":
            self.setup_loss_balanced()
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def setup_default(self):
        """Default configuration similar to original implementation."""
        self.hidden_channels = 60
        self.num_layers = 4
        self.K = 2
        self.heads_gat = 1
        self.use_edge_features = True
        self.dropout = 0.1
        self.layer_norm = True
        self.residual = True
        self.learning_rate = 5e-4
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.epochs = 150
        # Loss weights
        self.gamma_exp = 0.1
        self.beta_sparsity = 0.01
        self.delta_decor = 0.05

    def setup_large(self):
        """Larger model for complex datasets."""
        self.hidden_channels = 128
        self.num_layers = 6
        self.K = 4
        self.heads_gat = 2
        self.use_edge_features = True
        self.dropout = 0.15
        self.layer_norm = True
        self.residual = True
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.batch_size = 16
        self.epochs = 200
        # Loss weights for larger model
        self.gamma_exp = 0.05
        self.beta_sparsity = 0.02
        self.delta_decor = 0.1

    def setup_small(self):
        """Smaller model for quick experiments."""
        self.hidden_channels = 32
        self.num_layers = 3
        self.K = 2
        self.heads_gat = 1
        self.use_edge_features = False
        self.dropout = 0.05
        self.layer_norm = False
        self.residual = False
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.batch_size = 64
        self.epochs = 100
        # Loss weights for small model
        self.gamma_exp = 0.2
        self.beta_sparsity = 0.005
        self.delta_decor = 0.02

    def setup_edge_focused(self):
        """Configuration optimized for edge feature utilization."""
        self.hidden_channels = 128
        self.num_layers = 3
        self.K = 2
        self.heads_gat = 4
        self.use_edge_features = True
        self.dropout = 0.1
        self.layer_norm = False
        self.residual = True
        self.learning_rate = 0.001
        self.weight_decay = 1e-06
        self.batch_size = 16
        self.epochs = 175
        # Loss weights for edge-focused model
        self.gamma_exp = 0.1
        self.beta_sparsity = 0.01
        self.delta_decor = 0.05

    def setup_loss_balanced(self):
        """Configuration focused on balanced loss components."""
        self.hidden_channels = 80
        self.num_layers = 4
        self.K = 3
        self.heads_gat = 2
        self.use_edge_features = True
        self.dropout = 0.12
        self.layer_norm = True
        self.residual = True
        self.learning_rate = 4e-4
        self.weight_decay = 5e-5
        self.batch_size = 24
        self.epochs = 160
        # Carefully balanced loss weights
        self.gamma_exp = 0.08
        self.beta_sparsity = 0.012
        self.delta_decor = 0.06
@dataclass
class SearchSpace:
    """Define hyperparameter search spaces for MEGAN - ENHANCED VERSION."""

    # Architecture parameters
    hidden_channels: List[int] = None
    num_layers: List[int] = None
    K: List[int] = None
    heads_gat: List[int] = None

    # Training parameters
    learning_rate: List[float] = None
    weight_decay: List[float] = None
    dropout: List[float] = None
    batch_size: List[int] = None

    # Feature parameters
    use_edge_features: List[bool] = None
    layer_norm: List[bool] = None
    residual: List[bool] = None

    # Loss weight parameters
    gamma_exp: List[float] = None
    beta_sparsity: List[float] = None
    delta_decor: List[float] = None

    def __post_init__(self):
        """Set default search spaces if not provided."""
        if self.hidden_channels is None:
            self.hidden_channels = [32, 60, 80, 128]
        if self.num_layers is None:
            self.num_layers = [3, 4, 5, 6]
        if self.K is None:
            self.K = [2]
        if self.heads_gat is None:
            self.heads_gat = [1, 2, 4]
        if self.learning_rate is None:
            self.learning_rate = [1e-4, 3e-4, 5e-4, 1e-3]
        if self.weight_decay is None:
            self.weight_decay = [1e-6, 1e-5, 1e-4]
        if self.dropout is None:
            self.dropout = [0.0, 0.05, 0.1, 0.15, 0.2]
        if self.batch_size is None:
            self.batch_size = [16, 32, 64]
        if self.use_edge_features is None:
            self.use_edge_features = [True, False]
        if self.layer_norm is None:
            self.layer_norm = [True, False]
        if self.residual is None:
            self.residual = [True, False]
        # Loss weight search spaces
        if self.gamma_exp is None:
            self.gamma_exp = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
        if self.beta_sparsity is None:
            self.beta_sparsity = [0.005, 0.01, 0.015, 0.02, 0.03]
        if self.delta_decor is None:
            self.delta_decor = [0.02, 0.05, 0.08, 0.1, 0.15]
class HyperparameterSearcher:
    """Advanced hyperparameter search with different strategies."""

    def __init__(self, search_space: SearchSpace, strategy: str = "random"):
        self.search_space = search_space
        self.strategy = strategy
        self.results = []
        self.best_config = None
        self.best_score = float('inf')

    def generate_configs(self, n_trials: int = 50) -> List[Dict]:
        """Generate hyperparameter configurations based on strategy."""
        if self.strategy == "grid":
            return self._grid_search()
        elif self.strategy == "random":
            return self._random_search(n_trials)
        elif self.strategy == "smart":
            return self._smart_search(n_trials)
        elif self.strategy == "bayesian":
            return self._bayesian_search(n_trials)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _grid_search(self) -> List[Dict]:
        """Exhaustive grid search (can be very large!)"""
        param_names = []
        param_values = []

        for field, value in asdict(self.search_space).items():
            if isinstance(value, list) and len(value) > 0:
                param_names.append(field)
                param_values.append(value)

        configs = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            configs.append(config)

        print(f"Grid search will test {len(configs)} configurations")
        return configs

    def _random_search(self, n_trials: int) -> List[Dict]:
        """Random sampling from search space."""
        configs = []
        search_dict = asdict(self.search_space)

        for _ in range(n_trials):
            config = {}
            for param, values in search_dict.items():
                if isinstance(values, list) and len(values) > 0:
                    config[param] = random.choice(values)
            configs.append(config)

        return configs

    def _smart_search(self, n_trials: int) -> List[Dict]:
        """Smart search combining good defaults with random exploration."""
        configs = []

        # Start with some good baseline configurations
        good_configs = [
            # Small, fast model
            {"hidden_channels": 32, "num_layers": 3, "K": 2, "heads_gat": 1,
             "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1,
             "batch_size": 64, "use_edge_features": True, "layer_norm": True, "residual": True},

            #Balanced model
            {"hidden_channels": 60, "num_layers": 4, "K": 2, "heads_gat": 1,
             "learning_rate": 5e-4, "weight_decay": 1e-5, "dropout": 0.1,
             "batch_size": 32, "use_edge_features": True, "layer_norm": True, "residual": True},

            # Large model
            {"hidden_channels": 128, "num_layers": 5, "K": 3, "heads_gat": 2,
             "learning_rate": 3e-4, "weight_decay": 1e-4, "dropout": 0.15,
             "batch_size": 16, "use_edge_features": True, "layer_norm": True, "residual": True},

            # Edge-focused model
            {"hidden_channels": 80, "num_layers": 4, "K": 3, "heads_gat": 2,
             "learning_rate": 4e-4, "weight_decay": 5e-5, "dropout": 0.12,
             "batch_size": 24, "use_edge_features": True, "layer_norm": True, "residual": True},
        ]

        # Add baseline configs
        configs.extend(good_configs[:min(len(good_configs), n_trials // 4)])

        # Fill remaining with random search
        remaining_trials = n_trials - len(configs)
        if remaining_trials > 0:
            configs.extend(self._random_search(remaining_trials))

        return configs

    def _bayesian_search(self, n_trials: int) -> List[Dict]:
        """Simplified Bayesian optimization using Gaussian Process."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            print("scikit-learn not available, falling back to random search")
            return self._random_search(n_trials)

        # Start with a few random configurations
        initial_configs = self._random_search(min(10, n_trials // 3))
        configs = initial_configs.copy()

        # For remaining trials, use acquisition function
        # (This is a simplified version; full implementation would require more sophisticated handling)
        remaining_trials = n_trials - len(configs)
        if remaining_trials > 0:
            configs.extend(self._random_search(remaining_trials))

        return configs

class SearchResultAnalyzer:
    """Analyze and visualize hyperparameter search results."""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.df = self._results_to_dataframe()

    def _results_to_dataframe(self):
        """Convert results to pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available for analysis")
            return None

        flattened_results = []
        for result in self.results:
            flat_result = {}
            # Add configuration parameters
            for key, value in result.get('config', {}).items():
                flat_result[f'config_{key}'] = value

            # Add performance metrics
            flat_result['val_mae'] = result.get('val_mae', float('inf'))
            flat_result['train_mae'] = result.get('train_mae', float('inf'))
            flat_result['training_time'] = result.get('training_time', 0)
            flat_result['trial_id'] = result.get('trial_id', -1)

            flattened_results.append(flat_result)

        return pd.DataFrame(flattened_results)

    def get_top_configs(self, n: int = 10, metric: str = 'val_mae'):
        """Get top N configurations based on specified metric."""
        if self.df is None or self.df.empty:
            return []

        # Filter out non-finite metric values before sorting
        df_finite_metric = self.df[np.isfinite(self.df[metric])].copy()
        if df_finite_metric.empty:
            return []

        sorted_df = df_finite_metric.sort_values(metric, ascending=True)
        return sorted_df.head(n).to_dict('records')

    def analyze_parameter_importance(self, metric: str = 'val_mae'):
        """Analyze which parameters have the most impact on performance."""
        if self.df is None or self.df.empty:
            return {}

        # Filter out rows where the metric is not finite
        finite_metric_df = self.df[np.isfinite(self.df[metric])].copy()
        if finite_metric_df.empty:
            return {}

        config_cols = [col for col in finite_metric_df.columns if col.startswith('config_')]
        importance_scores = {}

        for col in config_cols:
            param_name = col.replace('config_', '')
            if col not in finite_metric_df.columns: # Should not happen if config_cols from finite_metric_df
                continue

            # Ensure param column has more than one unique value to calculate importance
            if finite_metric_df[col].nunique(dropna=False) <= 1: # dropna=False to count NaN as a unique value if present
                importance_scores[param_name] = 0.0
                continue

            if finite_metric_df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                if finite_metric_df[metric].nunique() > 1: # Metric also needs variance
                    correlation = abs(finite_metric_df[col].corr(finite_metric_df[metric]))
                    importance_scores[param_name] = correlation if pd.notna(correlation) else 0.0
                else:
                    importance_scores[param_name] = 0.0
            else: # Categorical
                # Treat as string for grouping to handle mixed types or Nones robustly
                groups = finite_metric_df.groupby(finite_metric_df[col].astype(str))[metric]
                if len(groups) > 1 and finite_metric_df[metric].nunique() > 1:
                    group_means = groups.mean()
                    group_vars = groups.var()

                    if group_means.isna().any() or not np.all(np.isfinite(group_means.dropna())):
                        importance_scores[param_name] = 0.0
                        continue

                    variance_between = group_means.var(ddof=0) # Population variance for means
                    variance_within = group_vars.fillna(0).mean() # Fill NaN var (e.g. single item group) with 0

                    if pd.notna(variance_between) and pd.notna(variance_within):
                        if variance_within > 1e-9: # Avoid division by zero or tiny numbers
                            importance_scores[param_name] = variance_between / variance_within
                        elif variance_between > 1e-9 : # If variance_within is ~0 but variance_between is not
                             importance_scores[param_name] = variance_between * 1e9 # Large number
                        else:
                            importance_scores[param_name] = 0.0 # Both are zero or near zero
                    else:
                        importance_scores[param_name] = 0.0
                else:
                    importance_scores[param_name] = 0.0

        return dict(sorted([item for item in importance_scores.items() if pd.notna(item[1])], key=lambda x: x[1], reverse=True))

    def plot_search_progress(self, save_path: Optional[str] = None):
        """Plot search progress over trials."""
        if self.df is None or self.df.empty:
            print("DataFrame not available for plotting search progress.")
            # Optionally plot a message
            fig, ax = plt.subplots(1,1)
            ax.text(0.5, 0.5, "No data to plot search progress.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if save_path: plt.savefig(save_path)
            plt.show()
            return

        plt.figure(figsize=(12, 8))

        df_finite_mae = self.df[np.isfinite(self.df['val_mae'])].copy()

        # Plot validation MAE over trials
        plt.subplot(2, 2, 1)
        if not df_finite_mae.empty:
            plt.plot(df_finite_mae['trial_id'], df_finite_mae['val_mae'], 'o-', alpha=0.7)
        else:
            plt.plot([], [], 'o-') # Empty plot
            plt.text(0.5, 0.5, "No finite MAE data", transform=plt.gca().transAxes, ha="center", va="center")
        plt.xlabel('Trial ID')
        plt.ylabel('Validation MAE')
        plt.title('Search Progress (Finite MAEs)')

        # Plot best MAE so far
        plt.subplot(2, 2, 2)
        if not df_finite_mae.empty:
            sorted_finite_df = df_finite_mae.sort_values('trial_id')
            if not sorted_finite_df.empty:
                 best_so_far = sorted_finite_df['val_mae'].cummin()
                 plt.plot(sorted_finite_df['trial_id'], best_so_far, 'r-', linewidth=2)
            else: # Should not be reached if df_finite_mae is not empty
                 plt.plot([], [], 'r-')
                 plt.text(0.5, 0.5, "No finite MAE data", transform=plt.gca().transAxes, ha="center", va="center")
        else:
            plt.plot([], [], 'r-')
            plt.text(0.5, 0.5, "No finite MAE data", transform=plt.gca().transAxes, ha="center", va="center")
        plt.xlabel('Trial ID')
        plt.ylabel('Best Validation MAE (Finite)')
        plt.title('Best Performance Over Time (Finite MAEs)')

        # Distribution of validation MAE
        plt.subplot(2, 2, 3)
        if not df_finite_mae.empty:
            plt.hist(df_finite_mae['val_mae'], bins=20, alpha=0.7, edgecolor='black')
        else:
            plt.text(0.5, 0.5, "No finite MAE data for histogram", transform=plt.gca().transAxes, ha="center", va="center")
        plt.xlabel('Validation MAE')
        plt.ylabel('Frequency')
        plt.title('Distribution of Performance (Finite MAEs)')

        # Training time vs performance
        plt.subplot(2, 2, 4)
        if not df_finite_mae.empty:
            plt.scatter(df_finite_mae['training_time'], df_finite_mae['val_mae'], alpha=0.7)
        else:
            plt.text(0.5, 0.5, "No finite MAE data for scatter plot", transform=plt.gca().transAxes, ha="center", va="center")
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation MAE')
        plt.title('Training Time vs Performance (Finite MAEs)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_parameter_effects(self, top_n: int = 5, save_path: Optional[str] = None):
        """Plot effects of top parameters on performance."""
        if self.df is None or self.df.empty:
            print("DataFrame not available for plotting parameter effects.")
            fig, ax = plt.subplots(1,1)
            ax.text(0.5, 0.5, "No data to plot parameter effects.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            if save_path: plt.savefig(save_path)
            plt.show()
            return

        df_finite = self.df[np.isfinite(self.df['val_mae'])].copy()

        if df_finite.empty:
            print("No finite 'val_mae' data to plot parameter effects.")
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, "No finite 'val_mae' data for parameter effects plot.",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return

        importance = self.analyze_parameter_importance(metric='val_mae')
        top_params = list(importance.keys())[:top_n]

        if not top_params:
            print("No top parameters to plot (e.g., no variance after filtering).")
            fig, ax = plt.subplots(1, 1)
            ax.text(0.5, 0.5, "No top parameters to plot.",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return

        num_plots = len(top_params)
        ncols = min(3, num_plots) if num_plots > 0 else 1
        nrows = (num_plots + ncols - 1) // ncols if num_plots > 0 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        axes = axes.flatten()

        for i, param in enumerate(top_params):
            col_name = f'config_{param}'
            ax = axes[i]

            if col_name not in df_finite.columns:
                ax.text(0.5, 0.5, f"Param '{param}' not in data.", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f'{param} (Data N/A)')
                continue

            param_data = df_finite[col_name]
            metric_data = df_finite['val_mae']

            if param_data.dtype in ['int64', 'float64', 'int32', 'float32'] and param_data.nunique() > 5 : # Numeric scatter
                ax.scatter(param_data, metric_data, alpha=0.6, s=20)
            else: # Categorical, boolean, or few unique numerics -> box plot
                # Convert to string to handle mixed types or ensure discrete categories
                param_data_str = param_data.astype(str)
                unique_values = sorted(param_data_str.unique())

                data_by_value = [metric_data[param_data_str == val].dropna().values for val in unique_values]

                valid_data_by_value = [d for d in data_by_value if len(d) > 0]
                valid_labels = [unique_values[idx] for idx, d in enumerate(data_by_value) if len(d) > 0]

                if valid_data_by_value:
                    ax.boxplot(valid_data_by_value, labels=valid_labels, patch_artist=True, medianprops=dict(color="black"))
                else:
                    ax.text(0.5, 0.5, 'No data for boxplot', transform=ax.transAxes, ha="center", va="center")

            ax.set_xlabel(param)
            ax.set_ylabel('Validation MAE')
            ax.set_title(f'{param} (Importance: {importance.get(param, float("nan")):.3f})')
            ax.tick_params(axis='x', rotation=30) # Rotate x-labels if they are long

        for i in range(len(top_params), len(axes)): # Hide unused subplots
            axes[i].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()