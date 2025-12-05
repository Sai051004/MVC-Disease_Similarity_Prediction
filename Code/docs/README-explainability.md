# explainability.py

Provides explainability utilities for the MVC model using forward hooks and gradient-based attribution methods. Enables interpretation of model predictions without modifying core model internals.

## Overview

The explainability module helps answer:
- **Which genes contribute most** to disease-disease similarity predictions?
- **Which miRNAs are important** for specific disease associations?
- **What attention patterns** does the model learn?
- **How do features interact** across modalities?

---

## Key Concepts

### Attention Capture
Captures attention weights from `SpatialAttentionModule` during forward passes to understand what the model focuses on.

### Gradient-Based Saliency
Uses gradients to identify which input features are most important for predictions.

### Contribution Analysis
Identifies top-k contributing genes and miRNAs for each disease pair.

---

## Core Functions

### `enable_attention_capture(model)`
Registers forward hooks across the model to capture attention weights.

**Parameters:**
- `model`: MVC model instance

**Process:**
1. Recursively traverses model modules
2. Registers hooks on `SpatialAttentionModule` instances
3. Captures attention weights during forward passes
4. Stores weights as module attributes for later retrieval

**Usage:**
```python
from explainability import enable_attention_capture

enable_attention_capture(model)
# Now attention weights will be captured during forward passes
```

**Key Features:**
- Non-intrusive (doesn't modify model architecture)
- Automatically handles nested modules
- Detaches weights to CPU to save GPU memory

---

### `_attach_attention_hooks(module)`
Internal function that recursively registers hooks on model modules.

**Hooks Registered:**
1. **Pre-forward hook** (`compute_attn_hook`):
   - Recomputes attention scores/weights in `SpatialAttentionModule`
   - Caches weights for current input

2. **Forward hook** (`save_attn_hook`):
   - Persists cached attention weights as module attribute
   - Stores under `ATTN_ATTR_NAME` key

**Notes:**
- Only activates special logic for `SpatialAttentionModule`
- Other modules receive harmless pass-through hooks
- Attention weights are detached to CPU

---

### `collect_attention_weights(model) -> List[Tensor]`
Collects all captured attention weights from the model.

**Parameters:**
- `model`: MVC model instance

**Returns:**
- List of attention weight tensors

**Usage:**
```python
from explainability import collect_attention_weights

# After forward pass
attention_weights = collect_attention_weights(model)
# Returns list of [B, H, L, L] attention weight tensors
```

**Purpose:**
Retrieves attention weights captured during forward passes for analysis.

---

### `compute_fusion_saliency(model, d_train_1, d_train_2, loss_fn, target_labels, retain_graph=False)`
Computes gradient-based saliency for fused disease-pair inputs.

**Parameters:**
- `model`: MVC model instance
- `d_train_1`: Gene pair embeddings `[N_pairs, 2*h_dim]` with `requires_grad=True`
- `d_train_2`: miRNA pair embeddings `[N_pairs, 2*h_dim]` with `requires_grad=True`
- `loss_fn`: Loss function (typically `BCEWithLogitsLoss`)
- `target_labels`: Binary labels `[N_pairs]` for disease-disease pairs
- `retain_graph`: Whether to retain computation graph (default: False)

**Returns:**
- `saliency_1`: Saliency map for gene features `[N_pairs, 2*h_dim]`
- `saliency_2`: Saliency map for miRNA features `[N_pairs, 2*h_dim]`

**Process:**
1. Forward pass: `model.spatial_attention_fusion(d_train_1, d_train_2)`
2. Apply attention and scoring: `model.nonlinear_attention_dis` → `model.nonlinear_dis_score`
3. Compute loss: `loss_fn(predictions, target_labels)`
4. Backward pass: Compute gradients w.r.t. `d_train_1` and `d_train_2`
5. Return absolute gradients as saliency maps

**Usage:**
```python
from explainability import compute_fusion_saliency
import torch.nn as nn

d_train_1.requires_grad_(True)
d_train_2.requires_grad_(True)
loss_fn = nn.BCEWithLogitsLoss()

saliency_gene, saliency_mirna = compute_fusion_saliency(
    model, d_train_1, d_train_2, loss_fn, labels
)
```

**Purpose:**
Identifies which gene and miRNA features are most important for predictions.

---

### `summarize_saliency(saliency) -> Tensor`
Reduces 2D saliency map to per-pair importance scores.

**Parameters:**
- `saliency`: Saliency map `[N_pairs, D]`

**Returns:**
- Per-pair importance scores `[N_pairs]`

**Process:**
- Takes mean across feature dimension
- Produces single importance score per disease pair

**Usage:**
```python
from explainability import summarize_saliency

importance_scores = summarize_saliency(saliency_gene)
# Returns [N_pairs] importance scores
```

---

### `explanation_artifacts(model) -> Dict[str, Any]`
Aggregates all explainability artifacts from the model.

**Parameters:**
- `model`: MVC model instance

**Returns:**
Dictionary containing:
- `attention_weights`: List of captured attention tensors

**Usage:**
```python
from explainability import explanation_artifacts

artifacts = explanation_artifacts(model)
attn_weights = artifacts['attention_weights']
```

**Purpose:**
Centralized access to all explainability information.

---

### `compute_pair_contributions(model, g_h, m_h, d2g_dense, m2d_dense, pair_indices, top_k=10, id_to_gene=None, id_to_mirna=None)`
Computes top-k contributing genes and miRNAs for disease pairs.

**Parameters:**
- `model`: Trained MVC model
- `g_h`: Gene embeddings `[N_genes, h_dim]`
- `m_h`: miRNA embeddings `[N_mirnas, h_dim]`
- `d2g_dense`: Dense disease-gene mapping `[N_diseases, N_genes]`
- `m2d_dense`: Dense miRNA-disease mapping `[N_diseases, N_mirnas]`
- `pair_indices`: Disease pair indices `[N_pairs, 2]`
- `top_k`: Number of top contributors to return (default: 10)
- `id_to_gene`: Optional mapping from gene ID to name
- `id_to_mirna`: Optional mapping from miRNA ID to name

**Returns:**
List of dictionaries, one per disease pair:
```python
{
    'pair': (disease_i, disease_j),
    'genes': [
        {'id': gene_id, 'percent': contribution_percent},
        ...
    ],
    'mirnas': [
        {'id': mirna_id, 'percent': contribution_percent},
        ...
    ]
}
```

**Process:**
1. Computes disease embeddings via pooling
2. Identifies genes/miRNAs associated with each disease
3. Computes contribution scores based on embedding similarity
4. Ranks and returns top-k contributors

**Usage:**
```python
from explainability import compute_pair_contributions

contributions = compute_pair_contributions(
    model, g_h, m_h, d2g_dense, m2d_dense,
    pair_indices=d2d_edge_index_test,
    top_k=10
)

for contrib in contributions:
    print(f"Pair: {contrib['pair']}")
    print(f"Top genes: {contrib['genes']}")
    print(f"Top miRNAs: {contrib['mirnas']}")
```

**Purpose:**
Provides interpretable insights into which biological entities drive predictions.

---

## Typical Workflow

### 1. Enable Attention Capture
```python
from explainability import enable_attention_capture

enable_attention_capture(model)
```

### 2. Forward Pass
```python
model.eval()
with torch.no_grad():
    g_h, m_h = model(g_data, m_data)
    # Attention weights are automatically captured
```

### 3. Collect Artifacts
```python
from explainability import explanation_artifacts, collect_attention_weights

artifacts = explanation_artifacts(model)
attention_weights = collect_attention_weights(model)
```

### 4. Compute Saliency
```python
from explainability import compute_fusion_saliency
import torch.nn as nn

# Prepare inputs with gradients
d_train_1.requires_grad_(True)
d_train_2.requires_grad_(True)

# Compute saliency
loss_fn = nn.BCEWithLogitsLoss()
saliency_gene, saliency_mirna = compute_fusion_saliency(
    model, d_train_1, d_train_2, loss_fn, labels
)
```

### 5. Analyze Contributions
```python
from explainability import compute_pair_contributions

contributions = compute_pair_contributions(
    model, g_h, m_h, d2g_dense, m2d_dense,
    pair_indices=test_pairs,
    top_k=10
)
```

---

## Exporting Results

### CSV Export
```python
import csv

contributions = compute_pair_contributions(...)

with open('contributions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['disease_i', 'disease_j', 'gene_id', 'gene_pct', 'mirna_id', 'mirna_pct'])
    
    for contrib in contributions:
        di, dj = contrib['pair']
        for gene in contrib['genes']:
            writer.writerow([di, dj, gene['id'], gene['percent'], '', ''])
        for mirna in contrib['mirnas']:
            writer.writerow([di, dj, '', '', mirna['id'], mirna['percent']])
```

---

## Key Constants

### `ATTN_ATTR_NAME`
Attribute name used to store captured attention weights on modules.

**Value:** `'_captured_attention_weights'`

**Purpose:**
Provides consistent key for accessing attention weights across modules.

---

## Best Practices

1. **Enable hooks before training/evaluation** for consistent capture
2. **Use `model.eval()`** when computing saliency to disable dropout
3. **Detach gradients** when storing attention weights to save memory
4. **Batch processing** for efficiency when analyzing many pairs
5. **Validate contributions** against known biological knowledge

---

## Limitations

- Attention weights reflect model's learned patterns, not ground truth
- Saliency maps show relative importance, not absolute significance
- Contributions are based on embedding similarity, may not capture all interactions
- Computational cost increases with number of pairs analyzed

---

## Integration with Training

The explainability module is integrated into `agcn.py`:

```python
from explainability import compute_pair_contributions

# After training
contributions = compute_pair_contributions(
    model, g_h, m_h, d2g.to_dense(), m2d.to_dense(),
    d2d_edge_index_test, top_k=10
)
```

Results are automatically exported to CSV for analysis.

---

## Implementation Notes

- Uses PyTorch hooks for non-intrusive monitoring
- Supports both CPU and GPU computation
- Memory-efficient with automatic CPU offloading
- Compatible with gradient checkpointing
- Thread-safe for multi-threaded inference
