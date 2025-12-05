# MVC Architecture - Detailed Flow and Dimensions

This document describes the complete architecture flow of the MVC (Multi-View Contrastive) model, including the order of operations, number of layers at each step, and input/output dimensions.

## Architecture Overview

The MVC model processes disease-disease link prediction through a multi-modal pipeline that:
1. Encodes gene and miRNA graphs separately
2. Aggregates node embeddings to disease-level representations
3. Constructs disease-pair features
4. Fuses multi-modal information
5. Predicts disease-disease associations

---

## Step-by-Step Architecture Flow

### **Step 1: Input Data Preparation**

**Input:**
- Gene graph: `g_data` with one-hot identity features
  - Node features: `[N_genes, N_genes]` (sparse identity matrix)
  - Edge index: `[2, E_genes]` (bidirectional edges)
  - Edge weights: `[E_genes]` (normalized to [0, 1])
  
- miRNA graph: `m_data` with one-hot identity features
  - Node features: `[N_mirna, N_mirna]` (sparse identity matrix)
  - Edge index: `[2, E_mirna]` (bidirectional edges)
  - Edge weights: `[E_mirna]` (normalized to [0, 1])

**Output:**
- Prepared graph data structures ready for encoding

**Default Values:**
- `N_genes`: Number of genes in the network (dataset-dependent)
- `N_mirna`: Number of miRNAs in the network (dataset-dependent)

---

### **Step 2: Gene Encoder (AGCN or GIN)**

**Component:** `AGCN` - Adaptive Graph Convolutional Network (default)
**Alternative:** `GIN` - Graph Isomorphism Network (use `--use-gin` flag for better accuracy)

**Input:**
- `g_data.x`: `[N_genes, N_genes]` (one-hot identity features)
- `g_data.edge_index`: `[2, E_genes]`
- `g_data.edge_weight`: `[E_genes]` (optional)

**Architecture:**
- **Number of Layers:** 4 (default: `num_layers=4`)
- **Hidden Dimension:** `h_dim` (default: 256)

**AGCN Architecture:**

**Layer-by-Layer Structure:**

#### Layer 1 (Input Layer):
- **AGCNConv Layer:**
  - Input: `[N_genes, N_genes]`
  - Linear transform: `[N_genes, N_genes] → [N_genes, h_dim]`
  - Graph convolution with adaptive edge weighting
  - Output: `[N_genes, h_dim]`
- **BatchNorm1d:** `[N_genes, h_dim]`
- **Multi-head Attention:** 8 heads (default: `heads=8`)
- **LeakyReLU activation** (alpha=0.2)
- **Dropout** (default: 0.15)
- **Residual connection** (with projection if dimension mismatch)

#### Layers 2-3 (Hidden Layers):
- **AGCNConv Layer:**
  - Input: `[N_genes, h_dim]`
  - Output: `[N_genes, h_dim]`
- **BatchNorm1d:** `[N_genes, h_dim]`
- **Multi-head Attention:** 8 heads
- **LeakyReLU activation**
- **Dropout**
- **Residual connection**

#### Layer 4 (Output Layer):
- **AGCNConv Layer:**
  - Input: `[N_genes, h_dim]`
  - Output: `[N_genes, h_dim]`
- **BatchNorm1d:** `[N_genes, h_dim]`
- **LeakyReLU activation**
- **Dropout**
- **Residual connection**

#### Post-Processing:
- **LayerNorm:** `[N_genes, h_dim]`
- **Global Attention Pooling:** (if nodes ≤ 2048)
  - Multi-head attention with `heads//2` heads
  - Combines local and global features: `x + 0.1 * x_global`
- **Final Projection MLP:**
  - Linear: `[N_genes, h_dim] → [N_genes, 2*h_dim]`
  - LayerNorm + LeakyReLU + Dropout
  - Linear: `[N_genes, 2*h_dim] → [N_genes, h_dim]`
  - LayerNorm + LeakyReLU + Dropout (0.5 * dropout rate)

**GIN Architecture (Alternative, often better accuracy):**
- **GINConv Layers:** 4 layers with MLP-based aggregation
- **MLP Structure:** Each GIN layer uses a 2-layer MLP with BatchNorm and ReLU
- **Residual Connections:** Enabled by default
- **Learnable Epsilon:** Trainable parameter for node feature aggregation
- **Final Projection:** 2-layer MLP with LayerNorm

**Output:**
- `g_h`: Gene embeddings `[N_genes, h_dim]`

**AGCN Total Components:**
- 4 AGCNConv layers
- 4 BatchNorm1d layers
- 3 Multi-head Attention layers (Layers 1-3)
- 1 Global Attention layer (optional)
- 1 Final projection MLP (2 linear layers)

**GIN Total Components:**
- 4 GINConv layers (each with 2-layer MLP)
- 4 BatchNorm1d layers
- 1 LayerNorm
- 1 Final projection MLP (2 linear layers)

**Why GIN?**
- GIN is theoretically more expressive than GCN variants
- Uses MLPs instead of linear transformations for better feature learning
- Often achieves superior accuracy on graph classification and node embedding tasks
- Better at distinguishing non-isomorphic graph structures

---

### **Step 3: miRNA Encoder (GAT)**

**Component:** `GAT` - Graph Attention Network

**Input:**
- `m_data.x`: `[N_mirna, N_mirna]` (one-hot identity features)
- `m_data.edge_index`: `[2, E_mirna]`
- `m_data.edge_weight`: `[E_mirna]` (optional)

**Architecture:**
- **Number of GAT Layers:** 2
- **Hidden Dimension:** `h_dim` (default: 256)

**Layer-by-Layer Structure:**

#### Layer 1:
- **GATConv:**
  - Input: `[N_mirna, N_mirna]`
  - Output: `[N_mirna, h_dim]`
- **LeakyReLU activation**

#### Layer 2:
- **GATConv:**
  - Input: `[N_mirna, h_dim]`
  - Output: `[N_mirna, h_dim]`

**Output:**
- `m_h`: miRNA embeddings `[N_mirna, h_dim]`

**Total Components:**
- 2 GATConv layers

---

### **Step 4: Disease-Level Pooling**

**Operation:** Mean pooling using mapping matrices

**Gene → Disease:**
- **Input:** `g_h` `[N_genes, h_dim]`
- **Mapping:** `d2g` sparse matrix `[N_diseases, N_genes]`
- **Operation:** `pooling(g_h, d2g)`
  - Matrix multiplication: `d2g @ g_h` → `[N_diseases, h_dim]`
  - Row-wise normalization (mean pooling)
- **Output:** `d_h_gene` `[N_diseases, h_dim]`

**miRNA → Disease:**
- **Input:** `m_h` `[N_mirna, h_dim]`
- **Mapping:** `m2d` sparse matrix `[N_diseases, N_mirna]` (transposed)
- **Operation:** `pooling(m_h, m2d)`
  - Matrix multiplication: `m2d @ m_h` → `[N_diseases, h_dim]`
  - Row-wise normalization (mean pooling)
- **Output:** `d_h_mirna` `[N_diseases, h_dim]`

---

### **Step 5: Disease Pair Construction**

**Operation:** Concatenate disease embeddings for each pair

**Input:**
- `d_h_gene`: `[N_diseases, h_dim]`
- `d_h_mirna`: `[N_diseases, h_dim]`
- Disease pair indices: `d2d_edge_index` `[N_pairs, 2]`

**Gene Pair Features:**
- For each pair (i, j):
  - Concatenate: `[d_h_gene[i], d_h_gene[j]]`
- **Output:** `d_pair_gene` `[N_pairs, 2*h_dim]`

**miRNA Pair Features:**
- For each pair (i, j):
  - Concatenate: `[d_h_mirna[i], d_h_mirna[j]]`
- **Output:** `d_pair_mirna` `[N_pairs, 2*h_dim]`

---

### **Step 6: Projection Layers** (Optional, if not using spatial attention)

**Component:** `Projection_dis_gene` and `Projection_dis_miRNA`

#### Gene Projection:
- **Input:** `d_pair_gene` `[N_pairs, 2*h_dim]`
- **Layer 1:**
  - Linear: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`
  - ReLU activation
  - LayerNorm `[N_pairs, 2*h_dim]`
- **Layer 2:**
  - Linear: `[N_pairs, 2*h_dim] → [N_pairs, h_dim]`
  - ReLU activation
  - LayerNorm `[N_pairs, h_dim]`
- **Output:** `hid_vec_gene` `[N_pairs, h_dim]`

#### miRNA Projection:
- **Input:** `d_pair_mirna` `[N_pairs, 2*h_dim]`
- **Same structure as gene projection**
- **Output:** `hid_vec_mirna` `[N_pairs, h_dim]`

**Total Components:**
- 2 Linear layers per projection (4 total)

---

### **Step 7: Feature Fusion** (If using spatial attention)

**Component:** `SpatialAttentionFusion`

**Input:**
- Gene pair features: `[N_pairs, 2*h_dim]`
- miRNA pair features: `[N_pairs, 2*h_dim]`

#### Sub-step 7.1: Cross-Modal Spatial Attention

**Component:** `CrossModalSpatialAttention`

- **Gene Projection:**
  - Linear: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`
- **miRNA Projection:**
  - Linear: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`
- **Stack:** `[N_pairs, 2, 2*h_dim]` (gene and miRNA as sequence)
- **SpatialAttentionModule:**
  - Multi-head attention (8 heads)
  - FFN (2 linear layers: `2*h_dim → 8*h_dim → 2*h_dim`)
  - Residual connections
- **Concatenate:** `[N_pairs, 4*h_dim]`
- **Output Projection:**
  - Linear: `[N_pairs, 4*h_dim] → [N_pairs, 2*h_dim]`
  - LayerNorm
- **Output:** `[N_pairs, 2*h_dim]`

#### Sub-step 7.2: Disease-Specific Spatial Attention

- **Reshape:** `[N_pairs, 2*h_dim] → [N_pairs, 1, 2*h_dim]`
- **SpatialAttentionModule:**
  - Same structure as above
- **Reshape back:** `[N_pairs, 1, 2*h_dim] → [N_pairs, 2*h_dim]`
- **Final Projection:**
  - Linear: `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`

**Output:**
- `fused_features` `[N_pairs, 2*h_dim]`
- `fusion_weights` `[2]` (learnable gene/miRNA weights)

**Total Components:**
- 4 Linear layers (2 for cross-modal, 2 for disease-specific)
- 2 SpatialAttentionModules
- Each SpatialAttentionModule contains:
  - 3 Linear layers (Q, K, V projections)
  - 1 Output projection
  - 2 FFN Linear layers
  - 2 LayerNorm layers

**Alternative (No Spatial Attention):**
- Simple concatenation: `[N_pairs, 2*h_dim] + [N_pairs, 2*h_dim] → [N_pairs, 4*h_dim]`

---

### **Step 8: Disease Attention**

**Component:** `Attention_dis`

**Input:**
- `fused_features` or `concatenated_features`: `[N_pairs, 2*h_dim]` or `[N_pairs, 4*h_dim]`

**Structure:**
- **Linear Layer:**
  - `[N_pairs, 2*h_dim] → [N_pairs, 2*h_dim]`
- **Softmax attention weights** (dim=-2)
- **Element-wise multiplication:** `features * attention_weights`
- **ReLU activation**
- **LayerNorm:** `[N_pairs, 2*h_dim]`

**Output:**
- `atten_hid_vec` `[N_pairs, 2*h_dim]`

**Total Components:**
- 1 Linear layer
- 1 LayerNorm layer

---

### **Step 9: Final Scoring**

**Component:** `Dis_score`

**Input:**
- `atten_hid_vec` `[N_pairs, 2*h_dim]`

**Structure:**
- **Linear Layer:**
  - `[N_pairs, 2*h_dim] → [N_pairs, 1]`

**Output:**
- `score_logits` `[N_pairs, 1]`

**Final Prediction:**
- Apply sigmoid: `torch.sigmoid(score_logits)` → `[N_pairs, 1]` (probabilities)

**Total Components:**
- 1 Linear layer

---

## Complete Architecture Summary

### Total Layer Count (with Spatial Attention):

| Component | Layer Type | Count |
|-----------|-----------|-------|
| AGCN Encoder | AGCNConv | 4 |
| AGCN Encoder | BatchNorm1d | 4 |
| AGCN Encoder | Multi-head Attention | 4 (3 layer-wise + 1 global) |
| AGCN Encoder | Final MLP Linear | 2 |
| **GIN Encoder (alternative)** | **GINConv** | **4** |
| **GIN Encoder** | **MLP (per GIN layer)** | **8 (2 per layer)** |
| **GIN Encoder** | **BatchNorm1d** | **4** |
| **GIN Encoder** | **Final MLP Linear** | **2** |
| GAT Encoder | GATConv | 2 |
| Projection (if no fusion) | Linear | 4 |
| Spatial Fusion | Linear (projections) | 4 |
| Spatial Fusion | SpatialAttentionModule | 2 |
| Spatial Fusion | Linear (in SpatialAttentionModule) | 10 per module (20 total) |
| Disease Attention | Linear | 1 |
| Final Score | Linear | 1 |

### Dimension Flow Summary:

```
Gene Graph:     [N_genes, N_genes] 
    ↓ (AGCN or GIN, 4 layers)
Gene Embeddings: [N_genes, h_dim]
    ↓ (Pooling with d2g)
Disease Gene Rep: [N_diseases, h_dim]
    ↓ (Pair Construction)
Gene Pair Features: [N_pairs, 2*h_dim]
    ↓ (Spatial Fusion - optional)
Fused Features: [N_pairs, 2*h_dim]
    ↓ (Disease Attention)
Attended Features: [N_pairs, 2*h_dim]
    ↓ (Final Score)
Logits: [N_pairs, 1]

miRNA Graph:    [N_mirna, N_mirna]
    ↓ (GAT, 2 layers)
miRNA Embeddings: [N_mirna, h_dim]
    ↓ (Pooling with m2d)
Disease miRNA Rep: [N_diseases, h_dim]
    ↓ (Pair Construction)
miRNA Pair Features: [N_pairs, 2*h_dim]
    ↓ (Spatial Fusion)
```

### Default Hyperparameters:

- **h_dim (Hidden Dimension):** 256
- **num_layers (AGCN):** 4
- **GAT layers:** 2
- **Attention heads:** 8
- **Dropout:** 0.15
- **LeakyReLU alpha:** 0.2

---

## Key Architectural Features

1. **Multi-Modal Encoding:** Separate AGCN/GIN and GAT encoders for different biological networks
2. **Adaptive Graph Convolution (AGCN):** Learnable edge weights in AGCN layers
3. **Graph Isomorphism Network (GIN):** MLP-based aggregation for potentially better accuracy (use `--use-gin` flag)
4. **Residual Connections:** Facilitate gradient flow in deep encoders
5. **Multi-Head Attention:** Captures diverse interaction patterns (AGCN only)
6. **Spatial Attention Fusion:** Cross-modal attention for combining gene and miRNA information
7. **Disease-Level Pooling:** Aggregates node-level embeddings to disease representations
8. **Pair-Wise Prediction:** Concatenates disease pairs for link prediction

---

## Notes

- The architecture supports both **with** and **without** spatial attention fusion
- When spatial attention is disabled, simple concatenation is used
- All dimensions assume default `h_dim=256`
- The actual number of genes, miRNAs, and diseases depends on your dataset
- Edge weights are normalized to [0, 1] range before processing
- **GIN encoder is recommended** for better accuracy (use `--use-gin` flag)
- Model checkpoints are automatically saved with best validation performance
- Disease IDs: Use NIH Disease IDs (e.g., C1153706) - unique identifiers from National Institutes of Health

---

## Model Saving and Loading

### Checkpoint Format

Saved model checkpoints contain:
```python
{
    "model_state": model.state_dict(),
    "best_f1": float,      # Best validation F1 score
    "best_auroc": float,   # Best validation AUROC
    "best_ap": float,      # Best validation AUPRC
    "epoch": int           # Epoch number when best model was saved
}
```

### Loading a Saved Model

```python
import torch
from model_utils import load_model_and_data

# Load model and data
model, g_data, m_data, d2g, m2d, device = load_model_and_data(
    checkpoint_path="best_model.pt",
    data_path="../Dataset",
    device="cuda",
    use_gin=True,  # Must match training configuration
    h_dim=256,
    dropout=0.15,
    heads=8
)
```

---

## Web Application Integration

The architecture supports interactive prediction through a Streamlit web application:

1. **Model Loading**: Loads saved checkpoint with matching hyperparameters
2. **Disease Input**: Accepts NIH Disease IDs (e.g., C1153706, C0157738)
3. **Prediction Pipeline**:
   - Encodes gene and miRNA graphs
   - Pools to disease level
   - Constructs disease pair features
   - Applies spatial attention fusion
   - Computes similarity score
4. **Output**: Similarity probability (0-1) with confidence level

See `app.py` and `model_utils.py` for implementation details.

