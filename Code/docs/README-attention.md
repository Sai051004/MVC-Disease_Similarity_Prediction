# attention.py

Implements spatial and cross-modal attention mechanisms used to fuse gene- and miRNA-derived disease-pair features in the MVC model.

## Overview

The attention module enables the model to learn which features from gene and miRNA modalities are most important for predicting disease-disease associations. It uses multi-head self-attention with residual connections and feed-forward networks.

---

## Core Components

### SpatialAttentionModule
Multi-head self-attention module with residuals and feed-forward network.

**Architecture:**
- **Input**: `x` `[B, L, D]` where:
  - `B`: Batch size (number of disease pairs)
  - `L`: Sequence length (1 or 2 depending on stacking)
  - `D`: Feature dimension (typically `2*h_dim`)

**Components:**
1. **Query/Key/Value Projections**: 
   - Linear projections to `num_heads` heads
   - Head dimension: `head_dim = D // num_heads`
   
2. **Attention Mechanism**:
   - Compute attention scores: `Q @ K^T / sqrt(head_dim)`
   - Apply softmax to get attention weights: `[B, H, L, L]`
   - Weighted sum: `weights @ V`
   
3. **Output Projection**:
   - Concatenate heads
   - Linear projection + dropout + residual connection
   
4. **Feed-Forward Network (FFN)**:
   - Linear → ReLU → Dropout → Linear
   - Residual connection + LayerNorm

**Output**: `[B, L, D]` - Attended features

**Key Features:**
- Multi-head attention captures diverse interaction patterns
- Residual connections facilitate gradient flow
- LayerNorm for training stability

---

### CrossModalSpatialAttention
Cross-modal attention for fusing gene and miRNA features.

**Architecture:**

1. **Input Projections**:
   - Gene projection: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`
   - miRNA projection: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`

2. **Stacking**:
   - Stack gene and miRNA as sequence: `[N_pairs, 2, 2*h_dim]`
   - Gene at position 0, miRNA at position 1

3. **Spatial Attention**:
   - Apply `SpatialAttentionModule` to the stacked sequence
   - Learns cross-modal interactions

4. **Split and Concatenate**:
   - Split attended outputs back to gene/miRNA positions
   - Concatenate: `[attended_gene, attended_miRNA] → [N_pairs, 4*h_dim]`

5. **Output Projection**:
   - Linear: `[N_pairs, 4*h_dim] → [N_pairs, 2*h_dim]`
   - LayerNorm

**Output**: `[N_pairs, 2*h_dim]` - Fused cross-modal features

**Purpose:**
Enables the model to learn which gene and miRNA features should be emphasized together for accurate disease similarity prediction.

---

### SpatialAttentionFusion
Top-level fusion block used by `MVC` to combine per-modality disease-pair vectors.

**Constructor:**
```python
SpatialAttentionFusion(gene_dim, mirna_dim, fusion_dim, num_heads=8, dropout=0.15)
```

**Architecture:**

1. **Cross-Modal Spatial Attention**:
   - Applies `CrossModalSpatialAttention` to gene and miRNA pair features
   - Output: `[N_pairs, 2*h_dim]`

2. **Disease-Specific Spatial Attention**:
   - Reshape: `[N_pairs, 2*h_dim] → [N_pairs, 1, 2*h_dim]`
   - Apply `SpatialAttentionModule` for disease-specific attention
   - Reshape back: `[N_pairs, 1, 2*h_dim] → [N_pairs, 2*h_dim]`

3. **Final Projection**:
   - Linear: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`

**Inputs:**
- `gene_features`: `[N_pairs, 2*h_dim]` - Gene-based disease pair features
- `mirna_features`: `[N_pairs, 2*h_dim]` - miRNA-based disease pair features

**Outputs:**
- `fused_features`: `[N_pairs, 2*h_dim]` - Fused multi-modal features
- `fusion_weights`: `[2]` - Learnable weights for gene/miRNA modalities

**Key Features:**
- Two-stage attention: cross-modal then disease-specific
- Learnable fusion weights for modality importance
- Preserves feature dimensionality for downstream processing

---

## Attention Flow Diagram

```
Gene Pair Features [N_pairs, 2*h_dim]
         ↓
    Projection
         ↓
miRNA Pair Features [N_pairs, 2*h_dim]
         ↓
    Projection
         ↓
    ┌────┴────┐
    │  Stack  │ → [N_pairs, 2, 2*h_dim]
    └────┬────┘
         ↓
Cross-Modal Spatial Attention
         ↓
    Split & Concat → [N_pairs, 4*h_dim]
         ↓
    Output Projection → [N_pairs, 2*h_dim]
         ↓
Disease-Specific Spatial Attention
         ↓
    Final Projection → [N_pairs, 2*h_dim]
         ↓
    Fused Features
```

---

## Usage Example

```python
from attention import SpatialAttentionFusion

# Create fusion module
fusion = SpatialAttentionFusion(
    gene_dim=512,      # 2 * h_dim
    mirna_dim=512,     # 2 * h_dim
    fusion_dim=512,    # Output dimension
    num_heads=8,
    dropout=0.15
)

# Forward pass
gene_pair_features = ...  # [N_pairs, 512]
mirna_pair_features = ... # [N_pairs, 512]

fused_features, fusion_weights = fusion(gene_pair_features, mirna_pair_features)
# fused_features: [N_pairs, 512]
# fusion_weights: [2] (gene_weight, mirna_weight)
```

---

## Explainability

Attention weights can be captured for explainability analysis:

```python
from explainability import enable_attention_capture, collect_attention_weights

# Enable attention capture
enable_attention_capture(model)

# Forward pass
fused_features, _ = model.spatial_attention_fusion(gene_features, mirna_features)

# Collect attention weights
attention_weights = collect_attention_weights(model)
```

The attention weights reveal which gene and miRNA features the model focuses on when making predictions.

---

## Hyperparameters

**Recommended Settings:**
- `num_heads`: 8 (good balance of expressiveness and efficiency)
- `dropout`: 0.15 (prevents overfitting)
- `fusion_dim`: Should match `2*h_dim` (typically 512 for h_dim=256)

**Tuning Tips:**
- Increase `num_heads` for more complex interactions (may slow training)
- Adjust `dropout` based on overfitting (increase if overfitting, decrease if underfitting)
- `fusion_dim` should match input dimensions for optimal performance

---

## Implementation Notes

- Attention weights are detached to CPU to reduce GPU memory pressure
- Supports batch processing for efficient inference
- Fully differentiable for end-to-end training
- Compatible with gradient-based explainability methods
