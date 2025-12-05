# main.py

High-level training and evaluation script for the MVC model. Provides a complete pipeline from data loading to model evaluation with optional cross-validation.

## Overview

`main.py` serves as the main entry point for training and evaluating the MVC model. It orchestrates data loading, model creation, optional cross-validation, training, and comprehensive evaluation.

---

## Script Flow

### 1. Data Loading
Loads all necessary graph data and mappings:

- **Gene Network**: `hnet.npz` - Human gene-gene interaction network
- **miRNA Network**: `miRNA2miRNA.npz` - miRNA-miRNA similarity network
- **Disease-Gene Mapping**: `d2g.npz` - Disease-to-gene association matrix
- **miRNA-Disease Mapping**: `miRNA2disease.npz` - miRNA-to-disease association matrix
- **Gene-miRNA Mapping**: `gene2miRNA.npz` - Gene-miRNA interaction network
- **Training Data**: `train_for_ukb_ori_sample.txt` and `train_for_ukb_ori_label.txt`
- **Test Data**: `test_sample_ukb_interpre_high.txt` and `test_label_ukb_interpre_high.txt`

**Output:**
- `g_data`: Gene graph (PyTorch Geometric Data object)
- `m_data`: miRNA graph (PyTorch Geometric Data object)
- `d2g`: Disease-to-gene mapping matrix
- `m2d`: miRNA-to-disease mapping matrix
- `d2d_edge_index`: Training disease pairs
- `d2d_link_labels`: Training labels
- `d2d_edge_index_test`: Test disease pairs
- `d2d_link_labels_test`: Test labels

---

### 2. Model Construction
Creates MVC model components:

- **Gene Encoder**: `AGCN` (default) or `GIN` (with `--use-gin` flag)
- **miRNA Encoder**: `GAT`
- **Projection Layers**: `Projection_dis_gene`, `Projection_dis_miRNA`
- **Attention**: `Attention_dis`
- **Scoring**: `Dis_score`
- **Fusion**: `SpatialAttentionFusion` (optional)

**Model Architecture:**
```
Gene Graph → AGCN/GIN → Gene Embeddings
                              ↓
                         Disease Pooling
                              ↓
miRNA Graph → GAT → miRNA Embeddings → Disease Pooling
                              ↓
                    Pair Construction & Fusion
                              ↓
                    Attention & Scoring → Predictions
```

---

### 3. Optional Cross-Validation
If enabled, performs k-fold cross-validation:

- Defines hyperparameter grid
- Splits training data into k folds
- Trains and validates on each fold
- Selects best hyperparameters
- Uses `CrossValidationTrainer` from `trainer.py`

**Hyperparameters Tuned:**
- Hidden dimension (`h_dim`)
- Learning rates (`lr1`, `lr2`)
- Dropout rate
- Number of layers
- Residual connections
- Temperature parameter (`tau`)

---

### 4. Final Model Training
Trains model on full training set:

- Uses best hyperparameters from CV (or defaults)
- Multi-view contrastive loss (gene-miRNA alignment)
- Binary cross-entropy loss (disease-disease prediction)
- Combined loss: `0.3 * contrastive_loss + 0.7 * link_loss`
- Learning rate scheduling
- Early stopping based on validation performance
- Model checkpoint saving

**Training Features:**
- Class-balanced loss weighting
- Gradient clipping
- Multiple optimizers for different components
- Cosine annealing learning rate scheduling

---

### 5. Test Evaluation
Evaluates trained model on held-out test set:

**Metrics Computed:**
- Accuracy (%)
- Precision
- Recall
- F1-Score
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

**Visualizations Generated:**
- ROC curve
- Precision-Recall curve
- Additional analysis plots (if enabled)

---

### 6. Explainability Analysis (Optional)
If enabled, computes and exports explainability artifacts:

- Top-k contributing genes for each disease pair
- Top-k contributing miRNAs for each disease pair
- Contribution percentages
- Exports to CSV: `contributions.csv`

**Uses:**
- `compute_pair_contributions` from `explainability.py`
- Gradient-based attribution
- Attention weight analysis

---

## Command-Line Arguments

```bash
python main.py [OPTIONS]
```

**Options:**
- `--data PATH`: Path to dataset directory (default: `../Dataset`)
- `--h_dim INT`: Hidden dimension (default: 256)
- `--dropout FLOAT`: Dropout rate (default: 0.15)
- `--epochs INT`: Number of training epochs (default: 50)
- `--use-gin`: Use GIN encoder instead of AGCN
- `--cv`: Enable cross-validation
- `--n_folds INT`: Number of CV folds (default: 5)
- `--seed INT`: Random seed (default: 42)
- `--disable-cuda`: Disable CUDA (use CPU)

---

## Usage Examples

### Basic Training
```bash
python main.py --data ../Dataset --epochs 100
```

### Training with GIN Encoder
```bash
python main.py --data ../Dataset --epochs 100 --use-gin --h_dim 384
```

### With Cross-Validation
```bash
python main.py --data ../Dataset --cv --n_folds 5 --epochs 20
```

### Full Pipeline
```bash
python main.py \
    --data ../Dataset \
    --epochs 200 \
    --use-gin \
    --h_dim 384 \
    --dropout 0.1 \
    --cv \
    --n_folds 5 \
    --seed 42
```

---

## Output Files

After running `main.py`, you'll get:

1. **Model Checkpoint**: `best_model.pt` (if `--save-best` specified)
2. **Metrics**: Printed to console
3. **Plots**: ROC/PR curves (if plotting enabled)
4. **Contributions CSV**: `contributions.csv` (if explainability enabled)

---

## Helper Functions

### `pooling(x, y2x)`
Mean-pooling operation for aggregating node embeddings.

**Parameters:**
- `x`: Node embeddings `[N_nodes, h_dim]`
- `y2x`: Mapping matrix `[N_diseases, N_nodes]`

**Returns:**
- Disease embeddings `[N_diseases, h_dim]`

**Purpose:**
Aggregates gene/miRNA node embeddings to disease-level representations.

---

## Integration with Other Scripts

**Related Scripts:**
- `agcn.py`: Enhanced training script with better defaults
- `trainer.py`: Cross-validation utilities
- `model.py`: Model components
- `attention.py`: Attention mechanisms
- `explainability.py`: Explainability tools
- `utils.py`: Data loading and utilities

**Workflow:**
```
main.py
  ├── utils.py (data loading)
  ├── model.py (model components)
  ├── attention.py (fusion mechanisms)
  ├── trainer.py (CV utilities)
  └── explainability.py (analysis)
```

---

## Best Practices

1. **Always use a validation set** for model selection
2. **Use cross-validation** for hyperparameter tuning
3. **Save model checkpoints** regularly
4. **Evaluate on test set only once** after final model selection
5. **Use fixed random seeds** for reproducibility
6. **Monitor training/validation losses** to detect overfitting
7. **Use GIN encoder** (`--use-gin`) for better accuracy

---

## Troubleshooting

**Common Issues:**

1. **Out of Memory**: Reduce `--h_dim` or use CPU (`--disable-cuda`)
2. **Poor Performance**: Try GIN encoder (`--use-gin`) or increase `--h_dim`
3. **Overfitting**: Increase `--dropout` or reduce model capacity
4. **Slow Training**: Reduce `--h_dim` or number of layers

---

## Implementation Notes

- Fully compatible with PyTorch Geometric
- Supports GPU acceleration
- Implements early stopping
- Saves best model checkpoints
- Comprehensive metric reporting
- Optional explainability analysis
