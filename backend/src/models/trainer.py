"""
Training and evaluation logic for MEGAN model.
Contains the main training loop, loss computation, and model evaluation functions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
import pickle
from pathlib import Path

from ..utils.config import MEGANConfig
from ..utils.mlflow_integration import MLflowManager, MLflowExperimentTracker
from .megan_architecture import MEGANCore


class MEGANLoss:
    """
    Custom loss function for MEGAN that combines prediction loss with regularization terms.
    """
    
    def __init__(self, gamma_exp: float = 0.1, beta_sparsity: float = 0.01, 
                 delta_decor: float = 0.05):
        self.gamma_exp = gamma_exp
        self.beta_sparsity = beta_sparsity
        self.delta_decor = delta_decor
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 model: MEGANCore) -> Dict[str, torch.Tensor]:
        """
        Compute MEGAN loss with all components.
        
        Args:
            predictions: Model predictions [B, 1]
            targets: Ground truth targets [B, 1]
            model: MEGAN model for accessing attention logits
        
        Returns:
            Dictionary containing all loss components
        """
        # Primary prediction loss (MSE for regression)
        pred_loss = F.mse_loss(predictions, targets)
        
        # Sparsity regularization loss
        sparsity_loss = self._compute_sparsity_loss(model)
        
        # Decorrelation loss
        decor_loss = self._compute_decorrelation_loss(model)
        
        # Total loss
        total_loss = pred_loss + self.beta_sparsity * sparsity_loss + self.delta_decor * decor_loss
        
        return {
            'total': total_loss,
            'pred': pred_loss,
            'sparsity': sparsity_loss,
            'decor': decor_loss
        }
    
    def _compute_sparsity_loss(self, model: MEGANCore) -> torch.Tensor:
        """Compute sparsity regularization on attention weights."""
        sparsity_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for layer_logits in model.attention_logits:
            for head_logits in layer_logits:
                # Apply softmax to get attention weights
                attention_weights = torch.softmax(head_logits, dim=0)
                # L1 penalty on attention weights to encourage sparsity
                sparsity_loss += torch.mean(torch.abs(attention_weights))
        
        return sparsity_loss
    
    def _compute_decorrelation_loss(self, model: MEGANCore) -> torch.Tensor:
        """Compute decorrelation loss between different explanation heads."""
        decor_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for layer_logits in model.attention_logits:
            if len(layer_logits) > 1:  # Need at least 2 heads for decorrelation
                # Stack logits from different heads
                stacked_logits = torch.stack(layer_logits, dim=0)  # [K, E, H]
                
                # Compute correlation matrix between heads
                K = stacked_logits.size(0)
                for i in range(K):
                    for j in range(i + 1, K):
                        # Flatten logits for correlation computation
                        logits_i = stacked_logits[i].flatten()
                        logits_j = stacked_logits[j].flatten()
                        
                        # Compute correlation coefficient
                        correlation = torch.corrcoef(torch.stack([logits_i, logits_j]))[0, 1]
                        decor_loss += torch.abs(correlation)
        
        return decor_loss


class MEGANTrainer:
    """
    Trainer class for MEGAN model with comprehensive training and evaluation capabilities.
    """
    def __init__(self, config: MEGANConfig, device: torch.device, 
                 mlflow_manager: Optional[MLflowManager] = None):
        self.config = config
        self.device = device
        self.mlflow_manager = mlflow_manager
        self.loss_fn = MEGANLoss(
            gamma_exp=config.gamma_exp,
            beta_sparsity=config.beta_sparsity,
            delta_decor=config.delta_decor
        )
    
    def train_fold(self, model: MEGANCore, train_loader: DataLoader, 
                   val_loader: DataLoader, fold_idx: int) -> Dict:
        """
        Train model for one fold with early stopping and comprehensive logging.
        
        Args:
            model: MEGAN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            fold_idx: Current fold index for logging
        
        Returns:
            Dictionary with training results and best model state
        """
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Training tracking
        train_losses = {'total': [], 'pred': [], 'sparsity': [], 'decor': []}
        val_losses = {'total': [], 'pred': [], 'sparsity': [], 'decor': []}
        val_maes = []
        
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        
        print(f"\nTraining Fold {fold_idx + 1}")
        print("-" * 50)
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            epoch_train_losses = {'total': 0, 'pred': 0, 'sparsity': 0, 'decor': 0}
            num_train_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, batch.y.view(-1, 1), model)
                
                # Backward pass
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                for key in epoch_train_losses:
                    epoch_train_losses[key] += loss_dict[key].item()
                num_train_batches += 1
            
            # Average training losses
            for key in epoch_train_losses:
                epoch_train_losses[key] /= num_train_batches
                train_losses[key].append(epoch_train_losses[key])
            
            # Validation phase
            val_loss_dict, val_mae = self.evaluate(model, val_loader)
            
            for key in val_loss_dict:
                val_losses[key].append(val_loss_dict[key])
            val_maes.append(val_mae)
            
            # Learning rate scheduling
            scheduler.step(val_loss_dict['total'])
            
            # Early stopping check
            if val_loss_dict['total'] < best_val_loss:
                best_val_loss = val_loss_dict['total']
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % self.config.log_interval == 0 or epoch == self.config.num_epochs - 1:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {epoch_train_losses['total']:.4f} | "
                      f"Val Loss: {val_loss_dict['total']:.4f} | "
                      f"Val MAE: {val_mae:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'best_model_state': best_model_state,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_maes': val_maes,
            'fold_idx': fold_idx
        }
    
    def evaluate(self, model: MEGANCore, data_loader: DataLoader) -> Tuple[Dict, float]:
        """
        Evaluate model on given data loader.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation
        
        Returns:
            Tuple of (loss_dict, mae)
        """
        model.eval()
        total_losses = {'total': 0, 'pred': 0, 'sparsity': 0, 'decor': 0}
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                targets = batch.y.view(-1, 1)
                
                # Compute losses
                loss_dict = self.loss_fn(predictions, targets, model)
                
                for key in total_losses:
                    total_losses[key] += loss_dict[key].item()
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                num_batches += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # Compute MAE
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae = F.l1_loss(all_predictions, all_targets).item()
        
        return total_losses, mae
    
    def train_cross_validation(self, dataset, fold_splits: List[Tuple], 
                             save_models: bool = True, save_dir: str = None) -> List[Dict]:
        """
        Train model using cross-validation with comprehensive result tracking.
        
        Args:
            dataset: PyTorch Geometric dataset
            fold_splits: List of (train_idx, val_idx) tuples for each fold
            save_models: Whether to save trained models
            save_dir: Directory to save models (uses config default if None)
        
        Returns:
            List of results for each fold
        """
        if save_dir is None:
            save_dir = self.config.model_save_path
        
        if save_models:
            os.makedirs(save_dir, exist_ok=True)
        
        fold_results = []
        overall_best_val_loss = float("inf")
        overall_best_model_state = None
        overall_best_fold_idx = None
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold_idx + 1}/{len(fold_splits)}")
            print(f"{'='*60}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model
            model_kwargs = {
                "in_channels": dataset.num_node_features,
                "hidden_channels": self.config.hidden_channels,
                "out_channels": 1,
                "edge_dim": getattr(dataset, 'num_edge_features', 0),
                "num_layers": self.config.num_layers,
                "K": self.config.K,
                "heads_gat": self.config.heads_gat,
                "use_edge_features": self.config.use_edge_features,
                "dropout": self.config.dropout,
                "layer_norm": self.config.layer_norm,
                "residual": self.config.residual,
            }
            
            model = MEGANCore(**model_kwargs).to(self.device)
            
            # Train fold
            fold_result = self.train_fold(model, train_loader, val_loader, fold_idx)
            fold_results.append(fold_result)
            
            # Track overall best model
            if fold_result['best_val_loss'] < overall_best_val_loss:
                overall_best_val_loss = fold_result['best_val_loss']
                overall_best_model_state = fold_result['best_model_state']
                overall_best_fold_idx = fold_idx
            
            # Save fold model if requested
            if save_models:
                fold_save_path = Path(save_dir) / f"fold_{fold_idx}_model.pth"
                torch.save({
                    'model_state_dict': fold_result['best_model_state'],
                    'config': self.config.to_dict(),
                    'fold_results': fold_result,
                    'model_kwargs': model_kwargs
                }, fold_save_path)
        
        # Save overall best model
        if save_models and overall_best_model_state is not None:
            best_model_path = Path(save_dir) / "best_model.pth"
            torch.save({
                'model_state_dict': overall_best_model_state,
                'config': self.config.to_dict(),
                'best_fold_idx': overall_best_fold_idx,
                'overall_best_val_loss': overall_best_val_loss,
                'all_fold_results': fold_results,
                'model_kwargs': model_kwargs
            }, best_model_path)
        
        # Print summary
        self._print_cv_summary(fold_results)
        
        return fold_results
    
    def _print_cv_summary(self, fold_results: List[Dict]):
        """Print cross-validation summary statistics."""
        val_losses = [r['best_val_loss'] for r in fold_results]
        val_maes = [r['best_val_mae'] for r in fold_results]
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Validation MAE:  {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        print(f"Best Fold:       {np.argmin(val_losses) + 1} (Loss: {min(val_losses):.4f})")
        print(f"{'='*60}")


def train_megan(dataset, config: MEGANConfig, fold_splits: List[Tuple], 
                device: torch.device, save_models: bool = True, 
                save_dir: str = None, verbose: bool = True,
                use_mlflow: bool = True, experiment_name: str = None) -> List[Dict]:
    """
    Main training function for MEGAN model with cross-validation and MLflow tracking.
    
    Args:
        dataset: PyTorch Geometric dataset
        config: MEGAN configuration
        fold_splits: List of (train_idx, val_idx) tuples
        device: Device to train on
        save_models: Whether to save models
        save_dir: Directory to save models
        verbose: Whether to print training progress
        use_mlflow: Whether to use MLflow tracking
        experiment_name: MLflow experiment name
    
    Returns:
        List of training results for each fold
    """
    trainer = MEGANTrainer(config, device)
    
    if verbose:
        print(f"Training MEGAN with configuration: {config.__class__.__name__}")
        print(f"Device: {device}")
        print(f"Dataset: {len(dataset)} samples")
        print(f"Folds: {len(fold_splits)}")
    
    # MLflow tracking setup
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = MLflowExperimentTracker(experiment_name)
            
            # Prepare dataset info for logging
            dataset_info = {
                "name": getattr(dataset, 'name', 'unknown'),
                "size": len(dataset),
                "n_features": dataset.num_node_features,
                "splits": {
                    "n_folds": len(fold_splits),
                    "fold_splits": fold_splits
                }
            }
            
            if verbose:
                print("MLflow tracking enabled")
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")
            use_mlflow = False
    
    # Train with cross-validation
    fold_results = trainer.train_cross_validation(dataset, fold_splits, save_models, save_dir)
    
    # Log to MLflow if enabled
    if use_mlflow and mlflow_tracker:
        try:
            # Get the best model from cross-validation
            best_fold_idx = np.argmin([r['best_val_loss'] for r in fold_results])
            best_fold_result = fold_results[best_fold_idx]
            
            # Create a model instance with best configuration for logging
            model_kwargs = {
                "in_channels": dataset.num_node_features,
                "hidden_channels": config.hidden_channels,
                "out_channels": 1,
                "edge_dim": getattr(dataset, 'num_edge_features', 0),
                "num_layers": config.num_layers,
                "K": config.K,
                "heads_gat": config.heads_gat,
                "use_edge_features": config.use_edge_features,
                "dropout": config.dropout,
                "layer_norm": config.layer_norm,
                "residual": config.residual,
            }
            
            best_model = MEGANCore(**model_kwargs).to(device)
            best_model.load_state_dict(best_fold_result['best_model_state'])
            
            # Track the experiment
            run_id, model_uri = mlflow_tracker.track_training_experiment(
                config=config,
                dataset_info=dataset_info,
                fold_results=fold_results,
                model=best_model,
                additional_artifacts={
                    "fold_results": fold_results,
                    "model_kwargs": model_kwargs,
                    "training_summary": {
                        "best_fold": best_fold_idx,
                        "best_val_loss": best_fold_result['best_val_loss'],
                        "best_val_mae": best_fold_result['best_val_mae'],
                        "cv_mean_loss": np.mean([r['best_val_loss'] for r in fold_results]),
                        "cv_std_loss": np.std([r['best_val_loss'] for r in fold_results])
                    }
                }
            )
            
            if verbose:
                print(f"\nMLflow tracking completed:")
                print(f"Run ID: {run_id}")
                print(f"Model URI: {model_uri}")
                
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
    
    return trainer.train_cross_validation(
        dataset, fold_splits, save_models, save_dir
    )
