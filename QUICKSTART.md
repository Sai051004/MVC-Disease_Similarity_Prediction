# Quick Start Guide

## 🚀 Fast Track to Running the Model

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
# Basic training (10 epochs for quick test)
python agcn.py --data ../Dataset --epochs 10 --save-best best_model.pt

# Recommended training (GIN encoder, 100 epochs)
python agcn.py --data ../Dataset --epochs 100 --use-gin --save-best best_model.pt --h_dim 256
```

### Step 3: Use the Web App

```bash
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Enter two disease IDs (e.g., 0 and 1)
3. Click "Predict Similarity"
4. View results!

## 📝 Example Commands

### Training Examples

```bash
# Quick test (10 epochs)
python agcn.py --data ../Dataset --epochs 10

# Full training with GIN (best accuracy)
python agcn.py --data ../Dataset --epochs 200 --use-gin --h_dim 384 --dropout 0.1 --save-best best_model.pt

# Training with AGCN (faster)
python agcn.py --data ../Dataset --epochs 100 --heads 8 --save-best best_model_agcn.pt
```

### Web App Examples

```bash
# Default settings
streamlit run app.py

# Custom model path
MODEL_PATH=my_model.pt streamlit run app.py

# Custom data path
DATA_PATH=/path/to/Dataset streamlit run app.py
```

## 🎯 What You'll Get

After training:
- **Model checkpoint**: `best_model.pt` (saved automatically)
- **Test metrics**: Printed to console (Accuracy, F1, AUROC, AUPRC)
- **Contributions CSV**: `enhanced_agcn_contributions.csv` (explainability results)

After using web app:
- **Interactive predictions**: Real-time disease similarity scores
- **Visualizations**: Charts and graphs
- **Interpretations**: Confidence levels and explanations
- **Input**: Use NIH Disease IDs (e.g., C1153706, C0157738) - unique identifiers from National Institutes of Health

## ⚠️ Common Issues

1. **Model not found**: Make sure you've trained and saved a model first
2. **CUDA errors**: Use `--disable-cuda` flag for CPU-only training
3. **Out of memory**: Reduce `--h_dim` (try 128 instead of 256)
4. **Import errors**: Install all dependencies: `pip install -r requirements.txt`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [docs/README-architecture.md](docs/README-architecture.md) for architecture details
- Explore the code to understand the implementation

