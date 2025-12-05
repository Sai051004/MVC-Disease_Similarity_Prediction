"""
Streamlit Web Application for Disease-Disease Similarity Prediction

This app provides an interactive interface to predict similarity between two diseases
using the trained MVC (Multi-View Contrastive) model.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model_utils import (
    load_model_and_data,
    predict_disease_similarity,
    load_disease_mappings
)


# Page configuration
st.set_page_config(
    page_title="Disease Similarity Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .similarity-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-similarity {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-similarity {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-similarity {
        background-color: #f8d7da;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e7f3ff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached(checkpoint_path, data_path, use_gin, h_dim, dropout, heads):
    """Cache model loading to avoid reloading on every interaction."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return load_model_and_data(
        checkpoint_path, data_path, device=device,
        use_gin=use_gin, h_dim=h_dim, dropout=dropout, heads=heads
    )


@st.cache_data
def load_mappings_cached(data_path):
    """Cache disease mappings."""
    return load_disease_mappings(data_path)


def get_similarity_class(score):
    """Get similarity class based on score."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"


def main():
    # Default configuration (can be overridden via sidebar)
    default_model = os.getenv("MODEL_PATH", "best_model.pt")
    default_data = os.getenv("DATA_PATH", "../Dataset")
    default_use_gin = os.getenv("USE_GIN", "False").lower() == "true"
    default_h_dim = int(os.getenv("H_DIM", "256"))
    default_dropout = float(os.getenv("DROPOUT", "0.15"))
    default_heads = int(os.getenv("HEADS", "8"))
    
    # Header
    st.markdown('<div class="main-header">🏥 Disease-Disease Similarity Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_path = st.text_input("Model Path", value=default_model)
        data_path = st.text_input("Data Path", value=default_data)
        use_gin = st.checkbox("Use GIN Encoder", value=default_use_gin)
        h_dim = st.number_input("Hidden Dimension", min_value=64, max_value=512, value=default_h_dim, step=64)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, default_dropout, 0.05)
        heads = st.number_input("Attention Heads", min_value=1, max_value=16, value=default_heads)
        
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared! Model will reload on next prediction.")
    
    # Load model and data
    try:
        with st.spinner("Loading model and data..."):
            model, g_data, m_data, d2g, m2d, device = load_model_cached(
                model_path, data_path, use_gin, h_dim, dropout, heads
            )
            id_to_name, name_to_id = load_mappings_cached(data_path)
        
        st.success(f"✅ Model loaded successfully! Using device: {device}")
        
        # Display model info
        with st.expander("📊 Model Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
                st.metric("Encoder Type", "GIN" if use_gin else "AGCN")
            with col2:
                st.metric("Hidden Dimension", h_dim)
                st.metric("Attention Heads", heads if not use_gin else "N/A")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please check:")
        st.info("1. Model checkpoint path is correct")
        st.info("2. Dataset path is correct")
        st.info("3. Model hyperparameters match training settings")
        st.stop()
    
    # Main input section
    st.header("🔍 Disease Similarity Prediction")
    
    # Info box about NIH IDs
    st.info("""
    **📋 Input Format**: Enter **NIH Disease IDs** (unique identifiers from National Institutes of Health). 
    Format: `C` followed by numbers (e.g., `C1153706`, `C0157738`). 
    Only NIH disease IDs are accepted - numeric indices are not supported.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease 1")
        disease_input_1 = st.text_input(
            "Enter NIH Disease ID",
            key="disease1",
            placeholder="e.g., C1153706",
            help="Enter NIH disease ID (e.g., C1153706, C0157738). These are unique identifiers from the National Institutes of Health."
        )
    
    with col2:
        st.subheader("Disease 2")
        disease_input_2 = st.text_input(
            "Enter NIH Disease ID",
            key="disease2",
            placeholder="e.g., C0157738",
            help="Enter NIH disease ID (e.g., C1153706, C0157738). These are unique identifiers from the National Institutes of Health."
        )
    
    # Parse disease inputs - Only accept NIH Disease IDs (unique identifiers)
    disease_id_1 = None
    disease_id_2 = None
    disease_name_1 = None
    disease_name_2 = None
    
    # Parse Disease 1: Only accept NIH Disease ID
    if disease_input_1:
        disease_input_1_clean = disease_input_1.strip().upper()
        if disease_input_1_clean in name_to_id:
            disease_id_1 = name_to_id[disease_input_1_clean]
            disease_name_1 = disease_input_1_clean
        else:
            st.warning(f"⚠️ Disease 1 '{disease_input_1}' not found. Please enter a valid NIH disease ID (e.g., C1153706).")
    
    # Parse Disease 2: Only accept NIH Disease ID
    if disease_input_2:
        disease_input_2_clean = disease_input_2.strip().upper()
        if disease_input_2_clean in name_to_id:
            disease_id_2 = name_to_id[disease_input_2_clean]
            disease_name_2 = disease_input_2_clean
        else:
            st.warning(f"⚠️ Disease 2 '{disease_input_2}' not found. Please enter a valid NIH disease ID (e.g., C0157738).")
    
    # Validate disease IDs
    max_disease_id = d2g.shape[0] - 1
    if disease_id_1 is not None and (disease_id_1 < 0 or disease_id_1 > max_disease_id):
        st.error(f"❌ Disease 1 ID {disease_id_1} is out of range (0-{max_disease_id})")
        disease_id_1 = None
    
    if disease_id_2 is not None and (disease_id_2 < 0 or disease_id_2 > max_disease_id):
        st.error(f"❌ Disease 2 ID {disease_id_2} is out of range (0-{max_disease_id})")
        disease_id_2 = None
    
    # Predict button
    if st.button("🔮 Predict Similarity", type="primary", use_container_width=True):
        if disease_id_1 is None or disease_id_2 is None:
            st.error("❌ Please enter valid disease IDs or names")
        elif disease_id_1 == disease_id_2:
            st.warning("⚠️ Please enter two different diseases")
        else:
            try:
                with st.spinner("Computing similarity..."):
                    similarity, logit, confidence = predict_disease_similarity(
                        model, g_data, m_data, d2g, m2d,
                        disease_id_1, disease_id_2, device
                    )
                
                # Display results
                st.markdown("---")
                st.header("📈 Prediction Results")
                
                # Disease information
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Disease 1:** {disease_name_1} (NIH Disease ID)")
                    st.caption(f"Internal Model Index: {disease_id_1}")
                with col2:
                    st.success(f"**Disease 2:** {disease_name_2} (NIH Disease ID)")
                    st.caption(f"Internal Model Index: {disease_id_2}")
                
                # Similarity score with color coding
                similarity_class = get_similarity_class(similarity)
                if similarity_class == "high":
                    css_class = "high-similarity"
                elif similarity_class == "medium":
                    css_class = "medium-similarity"
                else:
                    css_class = "low-similarity"
                
                st.markdown(
                    f'<div class="similarity-score {css_class}">'
                    f'Similarity Score: {similarity:.4f} ({similarity*100:.2f}%)'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similarity Score", f"{similarity:.4f}", delta=f"{similarity*100:.2f}%")
                with col2:
                    st.metric("Confidence Level", confidence)
                with col3:
                    st.metric("Raw Logit", f"{logit:.4f}")
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                if similarity >= 0.7:
                    st.success(f"**High Similarity**: These diseases are likely to be associated. "
                              f"The model predicts a {similarity*100:.1f}% probability of association.")
                elif similarity >= 0.4:
                    st.warning(f"**Medium Similarity**: These diseases show moderate similarity. "
                              f"The model predicts a {similarity*100:.1f}% probability of association.")
                else:
                    st.info(f"**Low Similarity**: These diseases show low similarity. "
                            f"The model predicts a {similarity*100:.1f}% probability of association.")
                
                # Visualization
                st.markdown("### 📊 Visualization")
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Bar chart
                colors = ['#2ecc71' if similarity >= 0.7 else '#f39c12' if similarity >= 0.4 else '#e74c3c']
                ax.barh(['Similarity Score'], [similarity], color=colors, alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Similarity Score')
                ax.set_title('Disease-Disease Similarity Prediction')
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
                ax.legend()
                
                # Add value label
                ax.text(similarity + 0.02, 0, f'{similarity:.3f}', 
                       va='center', fontsize=12, fontweight='bold')
                
                st.pyplot(fig)
                
                # Probability distribution visualization
                st.markdown("#### Probability Distribution")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                
                # Create a distribution around the prediction
                x = np.linspace(0, 1, 100)
                # Use sigmoid of normal distribution centered at logit
                from scipy.stats import norm
                mu = logit
                sigma = 0.5  # Standard deviation for visualization
                y = norm.pdf((np.log(similarity / (1 - similarity + 1e-10)) - mu) / sigma) / sigma
                y = y / y.max()  # Normalize
                
                ax2.plot(x, y, 'b-', linewidth=2, label='Probability Density')
                ax2.axvline(x=similarity, color='r', linestyle='--', linewidth=2, label=f'Prediction: {similarity:.3f}')
                ax2.fill_between(x, 0, y, alpha=0.3)
                ax2.set_xlabel('Similarity Score')
                ax2.set_ylabel('Density')
                ax2.set_title('Prediction Confidence Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig2)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
    
    # Information section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Instructions
        
        1. **Enter NIH Disease IDs**: Input two NIH disease IDs (unique identifiers from National Institutes of Health)
        2. **Click Predict**: Click the "Predict Similarity" button to get the similarity score
        3. **Interpret Results**: 
           - **High Similarity (≥0.7)**: Diseases are likely associated
           - **Medium Similarity (0.4-0.7)**: Moderate association probability
           - **Low Similarity (<0.4)**: Low association probability
        
        ### Example NIH Disease IDs
        - **C1153706**: First disease in the dataset
        - **C0157738**: Second disease in the dataset
        - **C0036572**: Third disease in the dataset
        - **C0030846**: Fourth disease in the dataset
        
        **Note**: Only NIH disease IDs are accepted (format: C followed by numbers). 
        These are unique identifiers from the National Institutes of Health.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit | Multi-View Contrastive Learning Model"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

