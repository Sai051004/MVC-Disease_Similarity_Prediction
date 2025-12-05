# utils.py

Utility functions for data I/O, sparse matrix handling, metrics computation, and visualization. Provides essential helper functions used throughout the MVC framework.

## Overview

`utils.py` contains low-level utilities for:
- Loading and preprocessing graph data
- Sparse matrix operations
- Disease/gene/miRNA ID mappings
- Evaluation metrics
- Visualization helpers

---

## Data Loading Functions

### `load_dmap(file_path)`
Loads name-to-ID and ID-to-name mappings from a text file.

**File Format:**
```
name1    id1
name2    id2
...
```

**Parameters:**
- `file_path`: Path to mapping file (e.g., `dis2id.txt`, `gene2id.txt`)

**Returns:**
- `name_to_id`: Dictionary mapping names to IDs
- `id_to_name`: Dictionary mapping IDs to names

**Example:**
```python
name_to_id, id_to_name = load_dmap('../Dataset/dis2id.txt')
# name_to_id: {'C1153706': 0, 'C0157738': 1, ...}
# id_to_name: {0: 'C1153706', 1: 'C0157738', ...}
```

**Usage:**
Used for loading disease, gene, and miRNA ID mappings.

---

### `load_sparse(path)`
Loads a sparse matrix from NPZ format.

**Parameters:**
- `path`: Path to `.npz` file

**Returns:**
- `scipy.sparse.coo_matrix`: Sparse matrix in COO format

**Example:**
```python
hnadj = load_sparse('../Dataset/hnet.npz')
# Returns COO sparse matrix of gene-gene interactions
```

**Supported Formats:**
- NPZ files created with `scipy.sparse.save_npz()`

---

### `load_data(data_path)`
Convenience function that loads all graph data and returns formatted PyTorch Geometric Data objects.

**Parameters:**
- `data_path`: Path to dataset directory

**Returns:**
Tuple containing:
- `g_data`: Gene graph (PyTorch Geometric Data object)
- `m_data`: miRNA graph (PyTorch Geometric Data object)
- `d2d_edge_index`: Training disease pairs `[N_pairs, 2]`
- `d2d_link_labels`: Training labels `[N_pairs]`
- `d2d_edge_index_test`: Test disease pairs `[N_pairs_test, 2]`
- `d2d_link_labels_test`: Test labels `[N_pairs_test]`

**Process:**
1. Loads gene-gene interaction network (`hnet.npz`)
2. Loads miRNA-miRNA similarity network (`miRNA2miRNA.npz`)
3. Loads disease-gene mapping (`d2g.npz`)
4. Creates one-hot identity features for nodes
5. Normalizes edge weights to [0, 1]
6. Creates bidirectional edges
7. Loads training and test disease pairs

**Example:**
```python
g_data, m_data, train_pairs, train_labels, test_pairs, test_labels = load_data('../Dataset')
```

---

## Sparse Matrix Utilities

### `sparse_to_tuple(sparse_mx)`
Converts sparse matrix to coordinate tuple format.

**Parameters:**
- `sparse_mx`: Sparse matrix (scipy.sparse)

**Returns:**
- `(coords, values, shape)`: Tuple of coordinates, values, and shape

**Purpose:**
Converts sparse matrices to a format suitable for tensor operations.

---

### `mx_to_torch_sparse_tensor(mx)`
Converts scipy sparse matrix to PyTorch sparse tensor.

**Parameters:**
- `mx`: Sparse matrix (scipy.sparse.coo_matrix or csr_matrix)

**Returns:**
- `torch.sparse_coo_tensor`: PyTorch sparse tensor

**Example:**
```python
d2g_sparse = load_sparse('../Dataset/d2g.npz')
d2g_tensor = mx_to_torch_sparse_tensor(d2g_sparse)
# Returns torch sparse tensor
```

**Purpose:**
Enables efficient sparse matrix operations in PyTorch.

---

### `generate_sparse_one_hot(num_ents, dtype=torch.float32)`
Generates one-hot identity features for nodes.

**Parameters:**
- `num_ents`: Number of entities (genes, miRNAs, etc.)
- `dtype`: Data type (default: torch.float32)

**Returns:**
- `[num_ents, num_ents]`: Identity matrix as dense tensor

**Example:**
```python
gene_features = generate_sparse_one_hot(num_genes=5000)
# Returns [5000, 5000] identity matrix
```

**Purpose:**
Creates initial node features when no node attributes are available. Each node gets a unique one-hot encoding.

**Note:**
Despite the name "sparse", this returns a dense tensor. The name refers to the sparse nature of one-hot encoding.

---

## Evaluation Metrics

### `plot_roc_curve(y_true, y_scores)`
Plots Receiver Operating Characteristic (ROC) curve.

**Parameters:**
- `y_true`: Ground truth binary labels
- `y_scores`: Predicted probability scores

**Output:**
- Matplotlib figure showing ROC curve with AUC score

**Purpose:**
Visualizes model performance for binary classification.

---

### `plot_precision_recall(y_true, y_scores)`
Plots Precision-Recall (PR) curve.

**Parameters:**
- `y_true`: Ground truth binary labels
- `y_scores`: Predicted probability scores

**Output:**
- Matplotlib figure showing PR curve with AUPRC score

**Purpose:**
Visualizes model performance, especially useful for imbalanced datasets.

---

## Visualization Helpers

### `plot_auc_vs_feature_dimensionality()`
Plots AUC vs. feature dimensionality (if data available).

**Purpose:**
Analyzes how model performance changes with feature dimension.

---

### `plot_auc_vs_epochs()`
Plots AUC vs. training epochs (if data available).

**Purpose:**
Visualizes training progress and convergence.

---

### `plot_true_positive_pairs(y_true)`
Visualizes distribution of positive disease pairs.

**Parameters:**
- `y_true`: Ground truth labels

**Purpose:**
Analyzes class distribution in the dataset.

---

### `plot_auc_vs_weighting_coefficient()`
Plots AUC vs. loss weighting coefficient (if data available).

**Purpose:**
Analyzes optimal weighting between contrastive and link prediction losses.

---

## Data Preprocessing

### Edge Weight Normalization
Edge weights are normalized to [0, 1] range:

```python
edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
```

**Purpose:**
Ensures consistent scale across different networks.

---

### Bidirectional Edges
Graphs are made undirected by creating bidirectional edges:

```python
edge_index = torch.cat([forward_edges, reverse_edges], dim=1)
```

**Purpose:**
Enables information flow in both directions for undirected graphs.

---

## File Format Specifications

### Disease ID Mapping (`dis2id.txt`)
```
30170
C1153706    0
C0157738    1
C0036572    2
...
```

**Format:**
- First line: Total count
- Subsequent lines: `NIH_Disease_ID    Internal_Index`

**Note:**
- First column: NIH Disease ID (unique identifier from National Institutes of Health)
- Second column: Internal numeric index (0, 1, 2, ...) used by the model

---

### Training/Test Data Files
**Training pairs**: `train_for_ukb_ori_sample.txt`
- Format: Two columns, each row is a disease pair
- Shape: `[N_pairs, 2]`

**Training labels**: `train_for_ukb_ori_label.txt`
- Format: Single column, binary labels (0 or 1)
- Shape: `[N_pairs]`

**Test files**: Similar format with `test_sample_ukb_interpre_high.txt` and `test_label_ukb_interpre_high.txt`

---

## Usage Examples

### Loading Complete Dataset
```python
from utils import load_data

g_data, m_data, train_pairs, train_labels, test_pairs, test_labels = load_data('../Dataset')

print(f"Gene nodes: {g_data.x.shape[0]}")
print(f"miRNA nodes: {m_data.x.shape[0]}")
print(f"Training pairs: {len(train_pairs)}")
```

### Loading Mappings
```python
from utils import load_dmap

name_to_id, id_to_name = load_dmap('../Dataset/dis2id.txt')
print(f"Disease {id_to_name[0]}: {name_to_id['C1153706']}")
```

### Converting Sparse Matrices
```python
from utils import load_sparse, mx_to_torch_sparse_tensor

d2g_sparse = load_sparse('../Dataset/d2g.npz')
d2g_tensor = mx_to_torch_sparse_tensor(d2g_sparse)
```

---

## Implementation Notes

- All functions support both CPU and GPU tensors
- Sparse operations are optimized for memory efficiency
- File I/O handles large datasets efficiently
- Compatible with PyTorch Geometric data format
- Error handling for missing files and invalid formats

---

## Dependencies

- `torch`: PyTorch for tensor operations
- `numpy`: Numerical operations
- `scipy.sparse`: Sparse matrix handling
- `scikit-learn`: Metrics computation
- `matplotlib`: Visualization
- `torch_geometric`: Graph data structures
