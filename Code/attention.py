import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module for Attention-Based Fusion
    This module learns spatial attention weights to focus on important regions
    in the feature space when fusing gene and miRNA information.
    """
    
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(SpatialAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        # Adjust head_dim to ensure it's divisible
        self.head_dim = max(1, feature_dim // num_heads)
        # Adjust feature_dim to be divisible by num_heads
        self.adjusted_feature_dim = self.head_dim * num_heads
        
        # Multi-head attention components
        self.query_proj = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.key_proj = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.value_proj = nn.Linear(feature_dim, self.adjusted_feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.adjusted_feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            Output tensor with spatial attention applied
        """
        batch_size, seq_len, _ = x.shape
        
        # Layer normalization
        x_norm = self.layer_norm1(x)
        
        # Multi-head attention
        Q = self.query_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.adjusted_feature_dim)
        attention_output = self.output_proj(attention_output)
        
        # Residual connection
        x = x + self.dropout(attention_output)
        
        # Feed-forward network
        x_norm2 = self.layer_norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output
        
        return x


class CrossModalSpatialAttention(nn.Module):
    """
    Cross-Modal Spatial Attention for fusing gene and miRNA information
    This module learns attention weights between different modalities
    """
    
    def __init__(self, gene_dim, mirna_dim, fusion_dim, num_heads=8, dropout=0.1):
        super(CrossModalSpatialAttention, self).__init__()
        self.gene_dim = gene_dim
        self.mirna_dim = mirna_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project gene and miRNA features to common space
        self.gene_proj = nn.Linear(gene_dim, fusion_dim)
        self.mirna_proj = nn.Linear(mirna_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = SpatialAttentionModule(fusion_dim, num_heads, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim * 2, fusion_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, gene_features, mirna_features):
        """
        Args:
            gene_features: Gene embeddings (batch_size, gene_dim)
            mirna_features: miRNA embeddings (batch_size, mirna_dim)
        Returns:
            Fused features with spatial attention
        """
        # Project to common space
        gene_proj = self.gene_proj(gene_features)
        mirna_proj = self.mirna_proj(mirna_features)
        
        # Stack features for cross-modal attention
        # Treat each sample as a sequence of 2 elements (gene, miRNA)
        stacked_features = torch.stack([gene_proj, mirna_proj], dim=1)  # (batch_size, 2, fusion_dim)
        
        # Apply cross-modal spatial attention
        attended_features = self.cross_attention(stacked_features)
        
        # Extract gene and miRNA features after attention
        attended_gene = attended_features[:, 0, :]  # (batch_size, fusion_dim)
        attended_mirna = attended_features[:, 1, :]  # (batch_size, fusion_dim)
        
        # Concatenate and project
        fused_features = torch.cat([attended_gene, attended_mirna], dim=-1)
        output = self.output_proj(fused_features)
        output = self.layer_norm(output)
        
        return output


class AdaptiveSpatialFusion(nn.Module):
    """
    Adaptive Spatial Fusion with learnable fusion weights
    This module adaptively combines gene and miRNA features using spatial attention
    """
    
    def __init__(self, gene_dim, mirna_dim, fusion_dim, num_heads=8, dropout=0.1):
        super(AdaptiveSpatialFusion, self).__init__()
        self.gene_dim = gene_dim
        self.mirna_dim = mirna_dim
        self.fusion_dim = fusion_dim
        
        # Cross-modal spatial attention
        self.cross_attention = CrossModalSpatialAttention(gene_dim, mirna_dim, fusion_dim, num_heads, dropout)
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # Learnable weights for gene and miRNA
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, gene_features, mirna_features):
        """
        Args:
            gene_features: Gene embeddings
            mirna_features: miRNA embeddings
        Returns:
            Fused features with adaptive spatial attention
        """
        # Apply cross-modal spatial attention
        attended_features = self.cross_attention(gene_features, mirna_features)
        
        # Apply feature transformation
        transformed_features = self.feature_transform(attended_features)
        
        # Adaptive fusion with learnable weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Final output
        output = self.output_proj(transformed_features)
        
        return output, weights


class SpatialAttentionFusion(nn.Module):
    """
    Main Spatial Attention Fusion module for MVC
    Integrates spatial attention mechanisms into the disease prediction pipeline
    """
    
    def __init__(self, gene_dim, mirna_dim, fusion_dim, num_heads=8, dropout=0.1):
        super(SpatialAttentionFusion, self).__init__()
        self.adaptive_fusion = AdaptiveSpatialFusion(gene_dim, mirna_dim, fusion_dim, num_heads, dropout)
        
        # Spatial attention for disease-specific features
        self.disease_spatial_attention = SpatialAttentionModule(fusion_dim, num_heads, dropout)
        
        # Output projection for disease prediction
        self.disease_proj = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, gene_features, mirna_features):
        """
        Args:
            gene_features: Gene embeddings
            mirna_features: miRNA embeddings
        Returns:
            Fused features with spatial attention for disease prediction
        """
        # Adaptive fusion with spatial attention
        fused_features, fusion_weights = self.adaptive_fusion(gene_features, mirna_features)
        
        # Apply spatial attention for disease-specific features
        # Reshape for spatial attention (treat as sequence of length 1)
        fused_reshaped = fused_features.unsqueeze(1)  # (batch_size, 1, fusion_dim)
        attended_features = self.disease_spatial_attention(fused_reshaped)
        attended_features = attended_features.squeeze(1)  # (batch_size, fusion_dim)
        
        # Final disease-specific projection
        output = self.disease_proj(attended_features)
        
        return output, fusion_weights
