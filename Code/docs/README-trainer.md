# trainer.py

Provides k-fold cross-validation training utilities for the MVC model, enabling robust hyperparameter tuning and model selection.

## Overview

The `CrossValidationTrainer` class implements a complete training pipeline with:
- K-fold cross-validation for hyperparameter search
- Multi-view contrastive learning loss
- Class-balanced binary cross-entropy loss
- Comprehensive metric computation
- Final model training on full dataset

---

## Classes

### CrossValidationTrainer

Main trainer class for cross-validation and model training.

**Constructor:**
```python
CrossValidationTrainer(model_class, tau=0.8, device='cuda', n_folds=5)
```

**Parameters:**
- `model_class`: Model class to instantiate (e.g., `MVC`)
- `tau`: Temperature parameter for contrastive loss (default: 0.8)
- `device`: Device for training ('cuda' or 'cpu')
- `n_folds`: Number of folds for cross-validation (default: 5)

**Key Methods:**

#### `load_data(g_data, m_data, labels, d2g, m2d, dis_path)`
Loads and stores data for training.

**Parameters:**
- `g_data`: Gene graph data (PyTorch Geometric Data object)
- `m_data`: miRNA graph data (PyTorch Geometric Data object)
- `labels`: Training labels (not used directly, loaded from files)
- `d2g`: Disease-to-gene mapping matrix
- `m2d`: miRNA-to-disease mapping matrix
- `dis_path`: Path to dataset directory

**Purpose:**
Prepares all necessary data structures for training and evaluation.

---

#### `nce_loss_g(gz, kgz, labels, tau=0.8)`
Contrastive loss for gene encoder.

**Parameters:**
- `gz`: Gene embeddings `[N_genes, h_dim]`
- `kgz`: miRNA embeddings `[N_mirnas, h_dim]`
- `labels`: Gene-miRNA association matrix `[N_genes, N_mirnas]`
- `tau`: Temperature parameter

**Returns:**
- Scalar loss value

**Purpose:**
Encourages gene and miRNA embeddings to be similar when they are associated.

**Mathematical Formulation:**
```
loss = -log(exp(sim(gz, kgz_pos) / τ) / Σ exp(sim(gz, kgz_i) / τ))
```

---

#### `nce_loss_m(gz, kgz, labels, tau=0.8)`
Contrastive loss for miRNA encoder (similar to `nce_loss_g` but from miRNA perspective).

---

#### `link_loss_dis(pre_value, d2d_link_labels)`
Binary cross-entropy loss for disease-disease link prediction.

**Parameters:**
- `pre_value`: Predicted logits `[N_pairs]`
- `d2d_link_labels`: Ground truth labels `[N_pairs]` (0 or 1)

**Returns:**
- Scalar BCE loss value

**Purpose:**
Supervised loss for disease-disease association prediction.

---

#### `compute_metrics(predictions, labels)`
Computes comprehensive evaluation metrics.

**Parameters:**
- `predictions`: Predicted logits `[N_pairs]`
- `labels`: Ground truth labels `[N_pairs]`

**Returns:**
Dictionary with:
- `accuracy`: Classification accuracy (%)
- `precision`: Positive predictive value
- `recall`: Sensitivity/True positive rate
- `f1`: F1-score (harmonic mean of precision and recall)
- `error_rate`: Classification error rate

**Purpose:**
Evaluates model performance on validation/test sets.

---

#### `train_fold(params, train_indices, val_indices, epochs, metric='f1')`
Trains model on a single fold.

**Parameters:**
- `params`: Dictionary of hyperparameters
  - `h_dim`: Hidden dimension
  - `lr1`, `lr2`: Learning rates for different parameter groups
  - `dropout`: Dropout rate
  - `num_layers`: Number of encoder layers
  - `residual`: Whether to use residual connections
  - `tau`: Contrastive loss temperature
- `train_indices`: Indices for training set
- `val_indices`: Indices for validation set
- `epochs`: Number of training epochs
- `metric`: Metric to optimize ('f1', 'accuracy', etc.)

**Returns:**
- Best validation score for the fold

**Training Process:**
1. Create model with given hyperparameters
2. Initialize optimizers (AdamW) with different learning rates
3. Training loop:
   - Forward pass through encoders
   - Compute contrastive loss (gene-miRNA alignment)
   - Compute link prediction loss
   - Combined loss: `0.3 * contrastive_loss + 0.7 * link_loss`
   - Backward pass and optimization
4. Validation after each epoch
5. Return best validation score

---

#### `cross_validate(param_grid, epochs=10, metric='f1')`
Orchestrates k-fold cross-validation over a hyperparameter grid.

**Parameters:**
- `param_grid`: Dictionary of hyperparameter lists to search
  ```python
  {
      'h_dim': [128, 256, 384],
      'lr1': [0.001, 0.0005],
      'dropout': [0.1, 0.15, 0.2],
      ...
  }
  ```
- `epochs`: Number of epochs per fold
- `metric`: Metric to optimize

**Returns:**
- `best_params`: Best hyperparameters found
- `best_cv_score`: Best cross-validation score

**Process:**
1. Generate all hyperparameter combinations
2. For each combination:
   - Split data into k folds
   - Train on k-1 folds, validate on 1 fold
   - Average validation scores across folds
3. Select combination with best average score
4. Store best parameters for final training

**Example:**
```python
param_grid = {
    'h_dim': [256, 384],
    'dropout': [0.1, 0.15],
    'lr1': [0.001, 0.0005]
}

best_params, best_score = trainer.cross_validate(param_grid, epochs=20, metric='f1')
```

---

#### `train_final_model(epochs=200)`
Trains final model on full training set using best hyperparameters.

**Parameters:**
- `epochs`: Number of training epochs

**Returns:**
- Trained model

**Process:**
1. Create model with best hyperparameters from cross-validation
2. Train on full training set (no validation split)
3. Return trained model for test evaluation

**Note:**
Must call `cross_validate()` first to set `best_params`.

---

#### `evaluate_on_test(model)`
Evaluates final model on held-out test set.

**Parameters:**
- `model`: Trained model

**Returns:**
- Dictionary of test metrics (accuracy, precision, recall, f1, error_rate)

**Purpose:**
Final evaluation on unseen test data for unbiased performance estimate.

---

## Helper Functions

### `pooling(x, y2x)`
Mean-pooling operation for aggregating node embeddings to disease level.

**Parameters:**
- `x`: Node embeddings `[N_nodes, h_dim]`
- `y2x`: Mapping matrix `[N_diseases, N_nodes]` (sparse)

**Returns:**
- Disease embeddings `[N_diseases, h_dim]`

**Mathematical Formulation:**
```
d_h = (y2x @ x) / row_sum(y2x)
```

**Purpose:**
Aggregates gene/miRNA node embeddings to disease-level representations using disease-gene and disease-miRNA associations.

---

## Usage Example

```python
from trainer import CrossValidationTrainer
from model import MVC, AGCN, GAT, ...

# Initialize trainer
trainer = CrossValidationTrainer(
    model_class=MVC,
    tau=0.8,
    device='cuda',
    n_folds=5
)

# Load data
trainer.load_data(g_data, m_data, None, d2g, m2d, '../Dataset')

# Define hyperparameter grid
param_grid = {
    'h_dim': [256, 384],
    'dropout': [0.1, 0.15],
    'lr1': [0.001],
    'lr2': [0.0005],
    'num_layers': [4],
    'residual': [True],
    'tau': [0.8]
}

# Cross-validation
best_params, best_score = trainer.cross_validate(
    param_grid, 
    epochs=20, 
    metric='f1'
)

print(f"Best CV Score: {best_score:.4f}")
print(f"Best Params: {best_params}")

# Train final model
final_model = trainer.train_final_model(epochs=100)

# Evaluate on test set
test_metrics = trainer.evaluate_on_test(final_model)
print(f"Test F1: {test_metrics['f1']:.4f}")
```

---

## Hyperparameter Tuning Tips

**Hidden Dimension (`h_dim`):**
- Larger values (384, 512) → Better capacity, slower training
- Smaller values (128, 256) → Faster training, may underfit
- Recommended: 256-384

**Dropout:**
- Higher (0.2-0.3) → More regularization, prevents overfitting
- Lower (0.1-0.15) → Less regularization, may overfit
- Recommended: 0.15

**Learning Rates:**
- `lr1`: For encoders (typically 0.001)
- `lr2`: For projection/attention heads (typically 0.0005)
- Lower learning rates → More stable, slower convergence

**Temperature (`tau`):**
- Lower (0.5-0.7) → Sharper contrastive loss
- Higher (0.8-1.0) → Softer contrastive loss
- Recommended: 0.8

---

## Best Practices

1. **Always use cross-validation** for hyperparameter tuning
2. **Use separate validation set** for early stopping during training
3. **Save best model** based on validation performance
4. **Evaluate on test set only once** after final model selection
5. **Use fixed random seeds** for reproducibility
6. **Monitor both training and validation losses** to detect overfitting

---

## Implementation Notes

- Supports both AGCN and GIN encoders
- Uses class-balanced loss weighting for imbalanced datasets
- Implements gradient clipping for training stability
- Compatible with GPU acceleration
- Saves checkpoints automatically during training
