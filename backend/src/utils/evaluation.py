"""
Evaluation metrics and visualization utilities for MEGAN model analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


class ModelEvaluator:
    """
    Comprehensive evaluation utilities for MEGAN model performance.
    """
    
    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Dictionary of computed metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mape': mape,
            'max_error': max_error
        }
    
    @staticmethod
    def evaluate_fold_results(fold_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate and analyze results from cross-validation folds.
        
        Args:
            fold_results: List of fold result dictionaries
        
        Returns:
            Aggregated evaluation metrics
        """
        metrics = {}
        
        # Extract fold-level metrics
        val_losses = [r['best_val_loss'] for r in fold_results]
        val_maes = [r['best_val_mae'] for r in fold_results]
        
        # Compute aggregated statistics
        metrics['val_loss_mean'] = np.mean(val_losses)
        metrics['val_loss_std'] = np.std(val_losses)
        metrics['val_loss_min'] = np.min(val_losses)
        metrics['val_loss_max'] = np.max(val_losses)
        
        metrics['val_mae_mean'] = np.mean(val_maes)
        metrics['val_mae_std'] = np.std(val_maes)
        metrics['val_mae_min'] = np.min(val_maes)
        metrics['val_mae_max'] = np.max(val_maes)
        
        # Best performing fold
        best_fold_idx = np.argmin(val_losses)
        metrics['best_fold'] = best_fold_idx
        metrics['best_fold_loss'] = val_losses[best_fold_idx]
        metrics['best_fold_mae'] = val_maes[best_fold_idx]
        
        # Stability metrics
        metrics['cv_stability'] = 1 - (np.std(val_losses) / np.mean(val_losses))
        
        return metrics


class VisualizationUtils:
    """
    Visualization utilities for model training and evaluation analysis.
    """
    
    @staticmethod
    def plot_training_curves(fold_results: List[Dict], save_path: Optional[str] = None):
        """
        Plot training and validation curves for all folds.
        
        Args:
            fold_results: Results from cross-validation training
            save_path: Optional path to save the plot
        """
        n_folds = len(fold_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Dynamics Across Folds', fontsize=16)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_folds))
        
        # Plot 1: Total Loss
        ax = axes[0, 0]
        for i, result in enumerate(fold_results):
            train_losses = result['train_losses']['total']
            val_losses = result['val_losses']['total']
            epochs = range(len(train_losses))
            
            ax.plot(epochs, train_losses, '--', color=colors[i], alpha=0.7, 
                   label=f'Fold {i+1} Train' if i == 0 else '')
            ax.plot(epochs, val_losses, '-', color=colors[i], alpha=0.8,
                   label=f'Fold {i+1} Val' if i == 0 else '')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Loss
        ax = axes[0, 1]
        for i, result in enumerate(fold_results):
            pred_losses = result['train_losses']['pred']
            val_pred_losses = result['val_losses']['pred']
            epochs = range(len(pred_losses))
            
            ax.plot(epochs, pred_losses, '--', color=colors[i], alpha=0.7)
            ax.plot(epochs, val_pred_losses, '-', color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Prediction Loss')
        ax.set_title('Prediction Loss (MSE)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Validation MAE
        ax = axes[1, 0]
        for i, result in enumerate(fold_results):
            val_maes = result['val_maes']
            epochs = range(len(val_maes))
            ax.plot(epochs, val_maes, '-', color=colors[i], alpha=0.8, 
                   label=f'Fold {i+1}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation MAE')
        ax.set_title('Validation MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Regularization Losses
        ax = axes[1, 1]
        avg_sparsity = []
        avg_decor = []
        
        for result in fold_results:
            sparsity = result['train_losses']['sparsity']
            decor = result['train_losses']['decor']
            avg_sparsity.append(sparsity)
            avg_decor.append(decor)
        
        # Average across folds
        if avg_sparsity and avg_decor:
            avg_sparsity = np.mean(avg_sparsity, axis=0)
            avg_decor = np.mean(avg_decor, axis=0)
            epochs = range(len(avg_sparsity))
            
            ax.plot(epochs, avg_sparsity, 'b-', label='Sparsity Loss')
            ax.plot(epochs, avg_decor, 'r-', label='Decorrelation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Regularization Loss')
            ax.set_title('Average Regularization Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                              title: str = "Predictions vs True Values",
                              save_path: Optional[str] = None):
        """
        Create scatter plot of predictions vs true values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Create scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Add diagonal line (perfect prediction)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
        
        # Compute and display metrics
        metrics = ModelEvaluator.compute_regression_metrics(y_true, y_pred)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        
        # Add metrics text
        metrics_text = f"R² = {metrics['r2']:.3f}\n"
        metrics_text += f"MAE = {metrics['mae']:.3f}\n"
        metrics_text += f"RMSE = {metrics['rmse']:.3f}\n"
        metrics_text += f"Pearson r = {metrics['pearson_r']:.3f}"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Analysis",
                      save_path: Optional[str] = None):
        """
        Create residual plots for error analysis.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)
        
        # Residuals vs Predicted
        ax = axes[0]
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # Residual histogram
        ax = axes[1]
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax.text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_cross_validation_summary(fold_results: List[Dict], 
                                    save_path: Optional[str] = None):
        """
        Create summary visualization of cross-validation results.
        
        Args:
            fold_results: Results from cross-validation
            save_path: Optional path to save the plot
        """
        n_folds = len(fold_results)
        fold_nums = list(range(1, n_folds + 1))
        
        val_losses = [r['best_val_loss'] for r in fold_results]
        val_maes = [r['best_val_mae'] for r in fold_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validation Loss by Fold
        ax = axes[0]
        bars = ax.bar(fold_nums, val_losses, alpha=0.7, color='steelblue')
        ax.axhline(y=np.mean(val_losses), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(val_losses):.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Best Validation Loss')
        ax.set_title('Validation Loss by Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, val_losses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Validation MAE by Fold
        ax = axes[1]
        bars = ax.bar(fold_nums, val_maes, alpha=0.7, color='lightcoral')
        ax.axhline(y=np.mean(val_maes), color='red', linestyle='--',
                  label=f'Mean: {np.mean(val_maes):.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Best Validation MAE')
        ax.set_title('Validation MAE by Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, val_maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_comprehensive_report(fold_results: List[Dict], 
                                  dataset_info: Dict,
                                  config_info: Dict,
                                  save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report with all visualizations and metrics.
        
        Args:
            fold_results: Cross-validation results
            dataset_info: Information about the dataset
            config_info: Model configuration information
            save_dir: Directory to save visualizations
        
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute evaluation metrics
        evaluation_metrics = ModelEvaluator.evaluate_fold_results(fold_results)
        
        # Create visualizations
        plot_save_path = str(save_dir / "training_curves.png") if save_dir else None
        VisualizationUtils.plot_training_curves(fold_results, plot_save_path)
        
        plot_save_path = str(save_dir / "cv_summary.png") if save_dir else None
        VisualizationUtils.plot_cross_validation_summary(fold_results, plot_save_path)
        
        # Compile comprehensive report
        report = {
            'evaluation_metrics': evaluation_metrics,
            'dataset_info': dataset_info,
            'config_info': config_info,
            'fold_results': fold_results,
            'n_folds': len(fold_results),
            'total_epochs_trained': sum(len(r['train_losses']['total']) for r in fold_results)
        }
        
        # Save report if directory provided
        if save_dir:
            report_path = save_dir / "evaluation_report.pkl"
            with open(report_path, 'wb') as f:
                pickle.dump(report, f)
            
            # Also save as readable text summary
            text_report_path = save_dir / "evaluation_summary.txt"
            VisualizationUtils._save_text_summary(report, text_report_path)
        
        return report
    
    @staticmethod
    def _save_text_summary(report: Dict, save_path: Path):
        """Save a human-readable text summary of the evaluation."""
        with open(save_path, 'w') as f:
            f.write("MEGAN Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset information
            f.write("Dataset Information:\n")
            f.write("-" * 20 + "\n")
            dataset_info = report['dataset_info']
            f.write(f"Dataset: {dataset_info.get('dataset_name', 'Unknown')}\n")
            f.write(f"Total samples: {dataset_info.get('num_samples', 'Unknown')}\n")
            f.write(f"Node features: {dataset_info.get('num_node_features', 'Unknown')}\n")
            f.write(f"Edge features: {dataset_info.get('num_edge_features', 'Unknown')}\n\n")
            
            # Model configuration
            f.write("Model Configuration:\n")
            f.write("-" * 20 + "\n")
            config_info = report['config_info']
            for key, value in config_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Evaluation metrics
            f.write("Cross-Validation Results:\n")
            f.write("-" * 25 + "\n")
            metrics = report['evaluation_metrics']
            f.write(f"Validation Loss: {metrics['val_loss_mean']:.4f} ± {metrics['val_loss_std']:.4f}\n")
            f.write(f"Validation MAE:  {metrics['val_mae_mean']:.4f} ± {metrics['val_mae_std']:.4f}\n")
            f.write(f"Best Fold: {metrics['best_fold'] + 1} (Loss: {metrics['best_fold_loss']:.4f})\n")
            f.write(f"CV Stability: {metrics['cv_stability']:.3f}\n")
            f.write(f"Total Folds: {report['n_folds']}\n")
            f.write(f"Total Epochs: {report['total_epochs_trained']}\n")
