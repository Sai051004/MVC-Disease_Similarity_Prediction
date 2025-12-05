"""
High-quality visualization utilities for MVC model results.

This module provides organized plotting functions that save
figures in a structured manner with professional formatting.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

# Set professional style
try:
    plt.style.use('seaborn-v0_8-paper')  # Matplotlib style name
except OSError:
    plt.style.use('seaborn-paper')  # Fallback for older matplotlib versions
sns.set_palette("husl")

# High-quality figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FIGURE_SIZE = (8, 6)
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11


def ensure_results_dir(base_dir="results"):
    """
    Ensure results directory structure exists.
    
    Structure:
    results/
    ├── figures/
    │   ├── roc_curves/
    │   ├── pr_curves/
    │   ├── training_curves/
    │   └── contributions/
    ├── checkpoints/
    ├── metrics/
    └── logs/
    """
    dirs = [
        base_dir,
        os.path.join(base_dir, "figures"),
        os.path.join(base_dir, "figures", "roc_curves"),
        os.path.join(base_dir, "figures", "pr_curves"),
        os.path.join(base_dir, "figures", "training_curves"),
        os.path.join(base_dir, "figures", "contributions"),
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "metrics"),
        os.path.join(base_dir, "logs"),
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir


def plot_roc_curve(y_true, y_scores, model_name="model", save_dir="results", show_plot=False):
    """
    Plot ROC curve with professional formatting.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probability scores
        model_name: Name of the model (for filename)
        save_dir: Directory to save the plot
        show_plot: Whether to display the plot (default: False)
    
    Returns:
        Path to saved figure
    """
    base_dir = ensure_results_dir(save_dir)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    
    ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=FONT_SIZE, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(base_dir, "figures", "roc_curves", f"roc_curve_{model_name}.{FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"✅ ROC curve saved: {save_path}")
    return save_path


def plot_precision_recall(y_true, y_scores, model_name="model", save_dir="results", show_plot=False):
    """
    Plot Precision-Recall curve with professional formatting. 
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probability scores
        model_name: Name of the model (for filename)
        save_dir: Directory to save the plot
        show_plot: Whether to display the plot (default: False)
    
    Returns:
        Path to saved figure
    """
    base_dir = ensure_results_dir(save_dir)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    
    ax.plot(recall, precision, linewidth=2.5, label=f'PR curve (AP = {ap_score:.4f})', color='#A23B72')
    
    ax.set_xlabel('Recall', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=FONT_SIZE, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(base_dir, "figures", "pr_curves", f"pr_curve_{model_name}.{FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"✅ Precision-Recall curve saved: {save_path}")
    return save_path


def plot_training_curves(train_losses=None, val_losses=None, train_metrics=None, val_metrics=None,
                        model_name="model", save_dir="results"):
    """
    Plot training curves (loss and metrics) with professional formatting.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: Dict of training metrics {'f1': [...], 'auroc': [...], ...}
        val_metrics: Dict of validation metrics {'f1': [...], 'auroc': [...], ...}
        model_name: Name of the model (for filename)
        save_dir: Directory to save the plot
    
    Returns:
        List of paths to saved figures
    """
    base_dir = ensure_results_dir(save_dir)
    saved_paths = []
    
    # Plot losses
    if train_losses is not None or val_losses is not None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
        
        epochs = range(1, max(len(train_losses) if train_losses else 0,
                             len(val_losses) if val_losses else 0) + 1)
        
        if train_losses:
            ax.plot(epochs, train_losses, 'o-', linewidth=2, markersize=4, 
                   label='Training Loss', color='#F18F01', alpha=0.8)
        if val_losses:
            ax.plot(epochs, val_losses, 's-', linewidth=2, markersize=4,
                   label='Validation Loss', color='#C73E1D', alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(f'Training and Validation Loss - {model_name}', 
                    fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax.legend(fontsize=FONT_SIZE, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        save_path = os.path.join(base_dir, "figures", "training_curves", 
                                f"loss_curves_{model_name}.{FIGURE_FORMAT}")
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        saved_paths.append(save_path)
        print(f"✅ Loss curves saved: {save_path}")
    
    # Plot metrics
    if train_metrics or val_metrics:
        for metric_name in ['f1', 'auroc', 'ap']:
            if (train_metrics and metric_name in train_metrics) or \
               (val_metrics and metric_name in val_metrics):
                fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
                
                epochs = range(1, max(
                    len(train_metrics.get(metric_name, [])) if train_metrics else 0,
                    len(val_metrics.get(metric_name, [])) if val_metrics else 0
                ) + 1)
                
                if train_metrics and metric_name in train_metrics:
                    ax.plot(epochs, train_metrics[metric_name], 'o-', linewidth=2, markersize=4,
                           label=f'Training {metric_name.upper()}', color='#06A77D', alpha=0.8)
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 's-', linewidth=2, markersize=4,
                           label=f'Validation {metric_name.upper()}', color='#005F73', alpha=0.8)
                
                ax.set_xlabel('Epoch', fontsize=LABEL_SIZE, fontweight='bold')
                ax.set_ylabel(metric_name.upper(), fontsize=LABEL_SIZE, fontweight='bold')
                ax.set_title(f'{metric_name.upper()} During Training - {model_name}',
                           fontsize=TITLE_SIZE, fontweight='bold', pad=15)
                ax.legend(fontsize=FONT_SIZE, frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_ylim([0.0, 1.0])
                
                plt.tight_layout()
                
                save_path = os.path.join(base_dir, "figures", "training_curves",
                                       f"{metric_name}_curves_{model_name}.{FIGURE_FORMAT}")
                plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
                plt.close(fig)
                saved_paths.append(save_path)
                print(f"✅ {metric_name.upper()} curves saved: {save_path}")
    
    return saved_paths


def plot_metrics_comparison(metrics_dict, model_name="model", save_dir="results"):
    """
    Plot bar chart comparing different metrics.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        model_name: Name of the model
        save_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    base_dir = ensure_results_dir(save_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
    
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Performance Metrics - {model_name}', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
    
    plt.tight_layout()
    
    save_path = os.path.join(base_dir, "figures", f"metrics_comparison_{model_name}.{FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"✅ Metrics comparison saved: {save_path}")
    return save_path


def plot_contribution_bars(gene_ids, gene_percentages, mirna_ids, mirna_percentages,
                          disease_pair, model_name="model", save_dir="results"):
    """
    Plot contribution bars for a disease pair.
    
    Args:
        gene_ids: List of gene IDs
        gene_percentages: List of gene contribution percentages
        mirna_ids: List of miRNA IDs
        mirna_percentages: List of miRNA contribution percentages
        disease_pair: Tuple of (disease_i, disease_j)
        model_name: Name of the model
        save_dir: Directory to save the plot
    
    Returns:
        Path to saved figure
    """
    base_dir = ensure_results_dir(save_dir)
    di, dj = disease_pair
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)
    
    # Gene contributions
    axes[0].barh(range(len(gene_ids)), gene_percentages[::-1], 
                color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_yticks(range(len(gene_ids)))
    axes[0].set_yticklabels([str(x) for x in gene_ids[::-1]], fontsize=9)
    axes[0].set_xlabel('Contribution Percentage', fontsize=LABEL_SIZE, fontweight='bold')
    axes[0].set_title(f'Top Gene Contributions\n(Diseases {di} & {dj})', 
                     fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # miRNA contributions
    axes[1].barh(range(len(mirna_ids)), mirna_percentages[::-1],
                color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    axes[1].set_yticks(range(len(mirna_ids)))
    axes[1].set_yticklabels([str(x) for x in mirna_ids[::-1]], fontsize=9)
    axes[1].set_xlabel('Contribution Percentage', fontsize=LABEL_SIZE, fontweight='bold')
    axes[1].set_title(f'Top miRNA Contributions\n(Diseases {di} & {dj})',
                     fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.suptitle(f'Feature Contributions - {model_name}', fontsize=TITLE_SIZE+2, 
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(base_dir, "figures", "contributions",
                           f"contributions_pair_{di}_{dj}_{model_name}.{FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return save_path


def save_metrics_to_file(metrics_dict, model_name="model", save_dir="results"):
    """
    Save metrics to a text file in an organized manner.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        model_name: Name of the model
        save_dir: Directory to save the file
    
    Returns:
        Path to saved file
    """
    base_dir = ensure_results_dir(save_dir)
    
    save_path = os.path.join(base_dir, "metrics", f"metrics_{model_name}.txt")
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"MODEL PERFORMANCE METRICS - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        
        for metric_name, value in metrics_dict.items():
            if isinstance(value, float):
                f.write(f"{metric_name.upper():<25}: {value:.6f}\n")
            else:
                f.write(f"{metric_name.upper():<25}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✅ Metrics saved: {save_path}")
    return save_path

