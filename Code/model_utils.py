"""
Utility functions for loading and using trained models for disease similarity prediction.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from utils import load_sparse, mx_to_torch_sparse_tensor, generate_sparse_one_hot
from model import AGCN, GIN, GAT, Projection_dis_gene, Projection_dis_miRNA, Attention_dis, Dis_score, MVC
from attention import SpatialAttentionFusion


def pooling(x, y2x):
    """Mean pooling operation for aggregating node embeddings to disease level."""
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    return torch.div(x, row_sum)


def load_model_and_data(checkpoint_path, data_path, device='cpu', use_gin=False, h_dim=256, dropout=0.15, heads=8):
    """
    Load a trained model checkpoint and prepare data for inference.
    
    Args:
        checkpoint_path: Path to saved model checkpoint (.pt file)
        data_path: Path to dataset directory
        device: Device to load model on ('cpu' or 'cuda')
        use_gin: Whether the checkpoint uses GIN encoder (must match training)
        h_dim: Hidden dimension (must match training)
        dropout: Dropout rate (must match training)
        heads: Number of attention heads for AGCN (must match training)
    
    Returns:
        model: Loaded model in eval mode
        g_data: Gene graph data
        m_data: miRNA graph data
        d2g: Disease-to-gene mapping matrix
        m2d: miRNA-to-disease mapping matrix
        device: Device object
    """
    device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load graph data
    print("Loading graph data...")
    hnadj = load_sparse(data_path + "/hnet.npz")
    src = hnadj.row
    dst = hnadj.col
    hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
    hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min() + 1e-8)
    hn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)

    d2g = load_sparse(data_path + "/d2g.npz")
    d2g = mx_to_torch_sparse_tensor(d2g)

    x = generate_sparse_one_hot(d2g.shape[1])
    g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

    mnadj = load_sparse(data_path + "/miRNA2miRNA.npz")
    src = mnadj.row
    dst = mnadj.col
    mn_edge_weight = torch.tensor(np.hstack((mnadj.data, mnadj.data)), dtype=torch.float)
    mn_edge_weight = (mn_edge_weight - mn_edge_weight.min()) / (mn_edge_weight.max() - mn_edge_weight.min() + 1e-8)
    mn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)
    x_m = generate_sparse_one_hot(mnadj.shape[0])
    m_data = Data(x=x_m, edge_index=mn_edge_index, edge_weight=mn_edge_weight)

    m2d = load_sparse(data_path + "/miRNA2disease.npz")
    m2d = m2d.T
    m2d = mx_to_torch_sparse_tensor(m2d)

    # Move data to device
    g_data = g_data.to(device)
    m_data = m_data.to(device)
    d2g = d2g.to(device)
    m2d = m2d.to(device)

    # Create model architecture
    if use_gin:
        g_encoder = GIN(
            nfeat=g_data.x.shape[1],
            nhid=h_dim,
            dropout=dropout,
            num_layers=4,
            residual=True,
            mlp_hidden=h_dim,
            train_eps=True
        )
    else:
        g_encoder = AGCN(
            nfeat=g_data.x.shape[1],
            nhid=h_dim,
            dropout=dropout,
            num_layers=4,
            residual=True,
            heads=heads,
            alpha=0.2
        )
    
    m_encoder = GAT(nfeat=m_data.x.shape[1], nhid=h_dim)
    projection_dis_gene = Projection_dis_gene(h_dim, h_dim)
    projection_dis_miRNA = Projection_dis_miRNA(h_dim, h_dim)
    attention_dis = Attention_dis(h_dim)
    dis_score = Dis_score(h_dim)
    spatial_fusion = SpatialAttentionFusion(
        gene_dim=2*h_dim, mirna_dim=2*h_dim, fusion_dim=2*h_dim,
        num_heads=heads, dropout=dropout
    )
    
    model = MVC(
        g_encoder, m_encoder, projection_dis_gene,
        projection_dis_miRNA, attention_dis, dis_score, spatial_fusion=spatial_fusion
    )
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Best validation F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
    print(f"Best validation AUROC: {checkpoint.get('best_auroc', 'N/A'):.4f}")
    
    return model, g_data, m_data, d2g, m2d, device


def predict_disease_similarity(model, g_data, m_data, d2g, m2d, disease_id_1, disease_id_2, device):
    """
    Predict similarity between two diseases.
    
    Args:
        model: Trained MVC model
        g_data: Gene graph data
        m_data: miRNA graph data
        d2g: Disease-to-gene mapping matrix
        m2d: miRNA-to-disease mapping matrix
        disease_id_1: First disease ID (integer index)
        disease_id_2: Second disease ID (integer index)
        device: Device to run inference on
    
    Returns:
        similarity_score: Similarity probability (0-1)
        logit: Raw logit score
        confidence: Prediction confidence level
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings
        g_h = model.get_gene_embeddings(g_data)
        m_h = model.get_miRNA_embeddings(m_data)
        
        # Pool to disease level
        d_h_gene = pooling(g_h, d2g.to_dense())
        d_h_mirna = pooling(m_h, m2d.to_dense())
        
        # Create disease pair features
        d_pair_gene = torch.cat((d_h_gene[disease_id_1:disease_id_1+1], d_h_gene[disease_id_2:disease_id_2+1]), 1)
        d_pair_mirna = torch.cat((d_h_mirna[disease_id_1:disease_id_1+1], d_h_mirna[disease_id_2:disease_id_2+1]), 1)
        
        # Fuse features
        fused_features, _ = model.spatial_attention_fusion(d_pair_gene, d_pair_mirna)
        
        # Apply attention
        atten_hid_vec = model.nonlinear_attention_dis(fused_features)
        
        # Get score
        score_logit = model.nonlinear_dis_score(atten_hid_vec)
        similarity_score = torch.sigmoid(score_logit).item()
        logit = score_logit.item()
        
        # Determine confidence level
        if similarity_score > 0.8:
            confidence = "Very High"
        elif similarity_score > 0.6:
            confidence = "High"
        elif similarity_score > 0.4:
            confidence = "Medium"
        elif similarity_score > 0.2:
            confidence = "Low"
        else:
            confidence = "Very Low"
    
    return similarity_score, logit, confidence


def load_disease_mappings(data_path):
    """
    Load disease ID to name mappings.
    
    The dis2id.txt file contains:
    - Column 1: NIH Disease ID (unique identifier from National Institutes of Health, e.g., C1153706)
    - Column 2: Internal numeric index (0, 1, 2, ...) used by the model
    
    Args:
        data_path: Path to dataset directory
    
    Returns:
        id_to_name: Dictionary mapping internal numeric index -> NIH disease ID
        name_to_id: Dictionary mapping NIH disease ID -> internal numeric index
    """
    id_to_name = {}
    name_to_id = {}
    
    try:
        with open(data_path + "/dis2id.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip first line (header with total count)
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    nih_disease_id = parts[0].strip()  # NIH disease ID (e.g., C1153706)
                    internal_index = int(parts[1])      # Internal numeric index (0, 1, 2, ...)
                    id_to_name[internal_index] = nih_disease_id
                    name_to_id[nih_disease_id.upper()] = internal_index  # Store uppercase for case-insensitive lookup
    except Exception as e:
        print(f"Warning: Could not load disease mappings: {e}")
    
    return id_to_name, name_to_id

