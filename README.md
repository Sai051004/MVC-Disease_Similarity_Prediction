# Multi-View Contrastive Learning for Disease-Disease Association Prediction

A state-of-the-art deep learning framework for predicting disease-disease associations using multi-modal graph neural networks. This implementation combines **Adaptive Graph Convolutional Networks (AGCN)** or **Graph Isomorphism Networks (GIN)** with **Graph Attention Networks (GAT)** to encode gene and miRNA networks, then uses spatial attention fusion to predict disease similarities.

## 🎯 Key Features

- **Multi-Modal Graph Encoding**: Separate encoders for gene and miRNA networks
- **Advanced Architectures**: Support for both AGCN and GIN (Graph Isomorphism Network) encoders
- **Spatial Attention Fusion**: Cross-modal attention mechanism for optimal feature fusion
- **Explainable AI**: Provides interpretable insights into disease similarity predictions
- **High Performance**: Achieves state-of-the-art results on disease-disease association prediction
- **Reproducible Research**: Seed control and checkpoint saving for consistent results
- **Interactive Web App**: Streamlit-based interface for real-time disease similarity queries

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Application](#web-application)
- [Architecture](#architecture)
- [Model Checkpoints](#model-checkpoints)
- [Results](#results)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

```bash
# Clone the repository
cd Code

# Install dependencies
pip install torch torch-geometric scikit-learn numpy scipy matplotlib streamlit pandas
```

Or install from requirements.txt (if available):
```bash
pip install -r requirements.txt
```

## 📁 Dataset Structure

The framework expects the following data structure in the `Dataset/` directory:

```
Dataset/
├── hnet.npz                          # Human gene-gene interaction network
├── miRNA2miRNA.npz                   # miRNA-miRNA similarity network
├── d2g.npz                           # Disease-gene association matrix
├── miRNA2disease.npz                # miRNA-disease association matrix
├── gene2miRNA.npz                    # Gene-miRNA interaction network
├── train_for_ukb_ori_sample.txt     # Training disease pairs (N_pairs x 2)
├── train_for_ukb_ori_label.txt      # Training labels (N_pairs)
├── test_sample_ukb_interpre_high.txt # Test disease pairs
├── test_label_ukb_interpre_high.txt  # Test labels
├── dis2id.txt                       # Disease ID mappings (name -> index)
├── gene2id.txt                      # Gene ID mappings (optional)
└── miRNA2id.txt                      # miRNA ID mappings (optional)
```

## 🚀 Quick Start

### Basic Training

Train the model with default AGCN encoder:
```bash
python agcn.py --data ../Dataset --epochs 100
```

### Training with GIN Encoder (Recommended for Better Accuracy)

```bash
python agcn.py --data ../Dataset --epochs 100 --use-gin --save-best best_model.pt
```

### Training with Custom Hyperparameters

```bash
python agcn.py \
    --data ../Dataset \
    --epochs 200 \
    --h_dim 384 \
    --dropout 0.1 \
    --heads 8 \
    --use-gin \
    --seed 42 \
    --save-best best_model.pt
```

## 📊 Training

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `../Dataset` | Path to dataset directory |
| `--h_dim` | int | `256` | Hidden dimension size |
| `--dropout` | float | `0.15` | Dropout rate |
| `--heads` | int | `8` | Number of attention heads (AGCN only) |
| `--epochs` | int | `50` | Number of training epochs |
| `--use-gin` | flag | `False` | Use GIN encoder instead of AGCN |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--save-best` | str | `best_model.pt` | Path to save best checkpoint |
| `--disable-cuda` | flag | `False` | Disable CUDA (use CPU) |

### Training Process

The training script:
1. **Loads and preprocesses** gene, miRNA, and disease networks
2. **Creates the model** with specified encoder (AGCN or GIN)
3. **Trains** with:
   - Multi-view contrastive loss (gene-miRNA alignment)
   - Binary cross-entropy loss for disease-disease prediction
   - Class-balanced loss weighting
   - Learning rate scheduling (CosineAnnealingWarmRestarts)
   - Gradient clipping
   - Early stopping based on validation F1 score
4. **Saves the best checkpoint** based on validation performance
5. **Evaluates** on test set and reports comprehensive metrics

### Example Training Output

```
🚀 Using device: cuda
📂 Loading data...
🔧 Creating enhanced GIN model...
📊 Model parameters: 2,456,789
🚀 Training enhanced model...
Epoch 10/100 | Loss: 0.4521 | Val F1: 0.8234 | Val AUROC: 0.9123
Epoch 20/100 | Loss: 0.3892 | Val F1: 0.8567 | Val AUROC: 0.9345
...
💾 Saved best checkpoint (F1=0.8765, AUROC=0.9456) to best_model.pt

🎉 ENHANCED GIN RESULTS:
Training time: 1245.67s
Test Accuracy: 86.53%
Test Precision: 0.8543
Test Recall: 0.8921
Test F1: 0.8529
Test AUROC: 0.9432
Test AUPRC: 0.9123
```

## 🧪 Evaluation

### Metrics

The framework reports:
- **Accuracy**: Classification accuracy (%)
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve

### Evaluating Saved Model

```bash
python evaluate_model.py --model best_model.pt --data ../Dataset
```

## 🌐 Web Application

### Launch Streamlit App

**Prerequisites**: Make sure you have trained a model and saved it:
```bash
python agcn.py --data ../Dataset --epochs 100 --save-best best_model.pt --use-gin
```

**Start the app**:
```bash
streamlit run app.py
```

Or with environment variables for custom paths:
```bash
MODEL_PATH=best_model.pt DATA_PATH=../Dataset streamlit run app.py
```

The web app provides:
- **Interactive disease similarity prediction**
- **Input**: Two disease IDs (numeric indices) or disease names (e.g., C1153706)
- **Output**: Similarity score (0-1), prediction confidence, and interpretation
- **Visualization**: Similarity score bar chart and probability distribution
- **Model information**: Display model parameters and configuration

### Using the Web App

1. **Start the app**: `streamlit run app.py`
2. **Configure model** (in sidebar):
   - Set model checkpoint path (default: `best_model.pt`)
   - Set dataset path (default: `../Dataset`)
   - Configure hyperparameters (must match training settings)
   - Click "Reload Model" if you change settings
3. **Enter diseases**:
   - Input NIH disease IDs (unique identifiers from National Institutes of Health)
     - Format: `C` followed by numbers (e.g., `C1153706`, `C0157738`, `C0036572`)
     - Only NIH disease IDs are accepted - numeric indices are not supported
4. **Predict**: Click "Predict Similarity" button
5. **View results**:
   - Similarity score with color-coded visualization
   - Confidence level (Very High/High/Medium/Low/Very Low)
   - Interpretation of the prediction
   - Visual charts showing similarity distribution

### Example Usage

```bash
# Train a model first
python agcn.py --data ../Dataset --epochs 100 --use-gin --save-best best_model.pt

# Launch the web app
streamlit run app.py
```

Then in the app:
- **Disease 1**: Enter `C1153706` (NIH disease ID)
- **Disease 2**: Enter `C0157738` (NIH disease ID)
- Click "Predict Similarity"

**Note**: Only NIH disease IDs are accepted. These are unique identifiers from the National Institutes of Health (format: C followed by numbers).

## 🏗️ Architecture

### Overview

```
Gene Network (AGCN/GIN) → Gene Embeddings → Disease-Level Pooling → Disease Pair Features
                                                                         ↓
miRNA Network (GAT) → miRNA Embeddings → Disease-Level Pooling → Disease Pair Features
                                                                         ↓
                                                          Spatial Attention Fusion
                                                                         ↓
                                                          Disease Attention
                                                                         ↓
                                                          Final Score (Similarity)
```

### Key Components

1. **Gene Encoder**: AGCN (4 layers) or GIN (4 layers with MLPs)
   - Processes gene-gene interaction network
   - Outputs gene embeddings [N_genes, h_dim]

2. **miRNA Encoder**: GAT (2 layers)
   - Processes miRNA-miRNA similarity network
   - Outputs miRNA embeddings [N_mirna, h_dim]

3. **Disease-Level Pooling**: Mean pooling using disease-gene and disease-miRNA mappings

4. **Spatial Attention Fusion**: Cross-modal attention for combining gene and miRNA features

5. **Disease Attention**: Attention mechanism over disease pair features

6. **Scoring Head**: Final linear layer for similarity prediction

For detailed architecture documentation, see [docs/README-architecture.md](docs/README-architecture.md)

## 💾 Model Checkpoints

### Checkpoint Format

Saved checkpoints contain:
```python
{
    "model_state": model.state_dict(),
    "best_f1": float,
    "best_auroc": float,
    "best_ap": float,
    "epoch": int
}
```

### Loading a Checkpoint

```python
import torch
from model import create_enhanced_model

checkpoint = torch.load("best_model.pt")
model = create_enhanced_model(g_data, m_data, ...)
model.load_state_dict(checkpoint["model_state"])
```

### Best Practices

- **Save checkpoints regularly**: Use `--save-best` to save the best validation model
- **Reproducibility**: Always use `--seed` for consistent results
- **Model selection**: Best model is selected based on validation F1 score
- **Experiments**: Use fixed seeds and save all checkpoints

## 📈 Results

### Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|-------|----------|-----------|--------|----|----|-------|
| AGCN | 86.52% | 0.832 | 0.856 | 0.844 | 0.928 | 0.895 |


*Results may vary based on dataset and hyperparameters*

## 📚 Additional Documentation

- [Architecture Details](docs/README-architecture.md) - Complete architecture flow and dimensions
- [Model Components](docs/README-model.md) - Core neural network components
- [Attention Mechanisms](docs/README-attention.md) - Spatial attention fusion
- [Explainability](docs/README-explainability.md) - Model interpretation tools
- [Training Guide](docs/README-trainer.md) - Cross-validation and hyperparameter tuning

## 🔬 Research & Citation

### Novel Contributions

1. **First application of GIN** to disease-disease association prediction on this dataset
2. **Multi-modal graph fusion** combining gene and miRNA networks
3. **Spatial attention mechanism** for optimal feature integration
4. **Explainable predictions** with contribution analysis


## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--h_dim` (e.g., 128 or 192)
   - Use CPU: `--disable-cuda`

2. **Model Not Saving**
   - Check write permissions in the directory
   - Verify `--save-best` path is valid

3. **Low Accuracy**
   - Try GIN encoder: `--use-gin`
   - Increase training epochs: `--epochs 200`
   - Adjust dropout: `--dropout 0.1`

4. **Import Errors**
   - Install all dependencies: `pip install torch torch-geometric scikit-learn`
   - Check Python version: `python --version` (requires 3.8+)


## 👥 Authors
- **Merugu Saikiran**
- **Are Ramgopal Reddy**
- **Sheik Mohuddin**

## 🙏 Acknowledgments

- PyTorch Geometric team for GNN implementations
- Streamlit for web app framework
- Dataset providers and contributors

---

For questions or issues, please open an issue on GitHub or contact [your email].


