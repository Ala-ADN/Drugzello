"""
Training and evaluation logic for MEGAN model.
Contains the main training loop, loss computation, and model evaluation functions.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple

from ..utils.config import MEGANConfig
from .megan_architecture import MEGANCore


class MEGANLoss:
    """Prediction MSE, differentiable explanation co-training and node L1.

    For centered regression: channel 0 learns below-reference evidence magnitude,
    channel 1 above-reference magnitude. delta_decor is an optional variant,
    disabled in the maintained experiment preset (not an original-paper term).
    """

    def __init__(self, gamma_exp=0.1, beta_sparsity=0.01, delta_decor=0.0,
                 reference=0.0):
        self.gamma_exp, self.beta_sparsity = gamma_exp, beta_sparsity
        self.delta_decor, self.reference = delta_decor, reference

    def __call__(self, predictions, targets, model):
        from torch_geometric.nn import global_add_pool
        pred = F.mse_loss(predictions, targets)
        nodes = model.node_importances
        sparsity = nodes.abs().mean()
        exp = pred.new_zeros(())
        if self.gamma_exp:
            if model.K != 2 or predictions.size(-1) != 1:
                raise ValueError("Regression explanation co-training requires K=2 and one target")
            counts = global_add_pool(nodes, model.batch_indices)
            distance = targets.view(-1) - self.reference
            channel = (distance > 0).long()
            selected = counts.gather(1, channel[:, None]).view(-1)
            # Importances are nonnegative; the below-reference channel represents
            # reference - target, not a negative count. See the learning guide.
            exp = (((selected - distance.abs()) ** 2) * (distance != 0)).sum() / (2 * len(targets))
        decor = pred.new_zeros(())
        if self.delta_decor and model.K > 1:
            unit = F.normalize(nodes, p=2, dim=0)
            gram = unit.T @ unit
            mask = ~torch.eye(model.K, dtype=torch.bool, device=nodes.device)
            decor = gram[mask].square().mean()
        total = pred + self.gamma_exp * exp + self.beta_sparsity * sparsity + self.delta_decor * decor
        return {'total': total, 'pred': pred, 'exp': exp, 'sparsity': sparsity, 'decor': decor}


class MEGANTrainer:
    """
    Trainer class for MEGAN model with comprehensive training and evaluation capabilities.
    """
    def __init__(self, config: MEGANConfig, device: torch.device):
        self.config = config
        self.device = device
        self.loss_fn = MEGANLoss(
            gamma_exp=config.gamma_exp,
            beta_sparsity=config.beta_sparsity,
            delta_decor=config.delta_decor
        )
        if config.training_schedule not in ("joint", "split"):
            raise ValueError("training_schedule must be joint or split")
        if config.scheduler_metric not in ("total", "mae"):
            raise ValueError("scheduler_metric must be total or mae")

    @staticmethod
    def optimize_batch(model, batch, optimizer, loss_fn, schedule):
        """Split: explanation-only update, then a fresh prediction/sparsity update."""
        targets = batch.y.view(-1, 1)
        if schedule == "split" and loss_fn.gamma_exp:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            auxiliary = loss_fn(output, targets, model)
            (loss_fn.gamma_exp * auxiliary["exp"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        losses = loss_fn(output, targets, model)
        objective = (losses["pred"] + loss_fn.beta_sparsity * losses["sparsity"]
                     + loss_fn.delta_decor * losses["decor"] if schedule == "split" else losses["total"])
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return losses
    
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
        train_losses = {'total': [], 'pred': [], 'exp': [], 'sparsity': [], 'decor': []}
        val_losses = {'total': [], 'pred': [], 'exp': [], 'sparsity': [], 'decor': []}
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
            epoch_train_losses = {'total': 0, 'pred': 0, 'exp': 0, 'sparsity': 0, 'decor': 0}
            num_train_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                loss_dict = self.optimize_batch(model, batch, optimizer, self.loss_fn,
                                                self.config.training_schedule)
                
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
            scheduler.step(val_mae if self.config.scheduler_metric == "mae" else val_loss_dict['total'])
            
            # Early stopping check
            if val_mae < best_val_mae:
                best_val_loss = val_loss_dict['total']
                best_val_mae = val_mae
                best_epoch = epoch
                best_model_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
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
            'best_epoch': best_epoch,
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
        total_losses = {'total': 0, 'pred': 0, 'exp': 0, 'sparsity': 0, 'decor': 0}
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
