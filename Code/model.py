import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, global_add_pool
import numpy as np
from attention import SpatialAttentionFusion


class AGCNConv(nn.Module):
    """
    Enhanced Adaptive Graph Convolutional Network Layer
    This layer adaptively learns the graph structure and edge weights with improved architecture
    """
    def __init__(self, in_channels, out_channels, bias=True, dropout=0.1, alpha=0.2, heads=1, max_attn_nodes: int = 2048):
        super(AGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.alpha = alpha
        self.heads = heads
        self.max_attn_nodes = max_attn_nodes
        
        # Learnable parameters for adaptive graph structure
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.adaptive_weight = nn.Parameter(torch.Tensor(1))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        
        # Multi-head attention for better feature learning
        if heads > 1:
            self.attention = nn.MultiheadAttention(out_channels, heads, dropout=dropout, batch_first=True)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(alpha)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.uniform_(self.adaptive_weight, 0, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Apply linear transformation
        out = torch.mm(x, self.weight)
        
        # Adaptive edge weight computation
        if edge_weight is not None:
            # Learn adaptive edge weights
            adaptive_edge_weight = edge_weight * torch.sigmoid(self.adaptive_weight)
        else:
            # If no edge weights provided, create uniform weights
            adaptive_edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            adaptive_edge_weight = adaptive_edge_weight * torch.sigmoid(self.adaptive_weight)
        
        # Normalize edge weights
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, adaptive_edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * adaptive_edge_weight * deg_inv_sqrt[col]
        
        # Apply graph convolution
        out = self.propagate(edge_index, x=out, norm=norm)
        
        # Apply bias
        if self.bias is not None:
            out += self.bias
        
        # Apply layer normalization
        out = self.layer_norm(out)
        
        # Apply activation
        out = self.leaky_relu(out)
        
        # Apply multi-head attention if enabled
        if self.heads > 1 and hasattr(self, 'attention') and out.size(0) <= self.max_attn_nodes:
            # Reshape for attention: [batch_size, seq_len, features]
            out_reshaped = out.unsqueeze(0)  # [1, num_nodes, out_channels]
            attn_out, _ = self.attention(out_reshaped, out_reshaped, out_reshaped)
            out = attn_out.squeeze(0)  # Back to [num_nodes, out_channels]
        
        # Apply dropout
        out = self.dropout_layer(out)
        
        return out
    
    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        out = torch.zeros_like(x)
        # Fix: Properly handle tensor dimensions for scatter_add_
        # row should be expanded to match the feature dimension of x
        row_expanded = row.unsqueeze(-1).expand(-1, x.size(1))
        out.scatter_add_(0, row_expanded, x[col] * norm.unsqueeze(-1))
        return out


class ChebyshevAGCNConv(nn.Module):
    """
    Chebyshev-based Adaptive Graph Convolutional Network Layer
    Uses Chebyshev polynomials for better spectral approximation and accuracy
    """
    def __init__(self, in_channels, out_channels, K=3, bias=True, dropout=0.1):
        super(ChebyshevAGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K  # Order of Chebyshev polynomials
        self.dropout = dropout
        
        # Learnable parameters for Chebyshev coefficients
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.adaptive_weight = nn.Parameter(torch.Tensor(1))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.uniform_(self.adaptive_weight, 0, 1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Apply Chebyshev polynomial-based graph convolution
        out = self.chebyshev_conv(x, edge_index, edge_weight)
        
        if self.bias is not None:
            out += self.bias
            
        return out
    
    def chebyshev_conv(self, x, edge_index, edge_weight=None):
        # Compute normalized Laplacian matrix
        L = self.compute_normalized_laplacian(x.size(0), edge_index, edge_weight)
        
        # Compute Chebyshev polynomials
        Tx_0 = x
        Tx_1 = torch.mm(L, x)
        
        # Initialize output
        out = torch.mm(Tx_0, self.weight[0]) + torch.mm(Tx_1, self.weight[1])
        
        # Higher order Chebyshev polynomials (K > 2)
        if self.K > 2:
            for k in range(2, self.K):
                Tx_k = 2 * torch.mm(L, Tx_1) - Tx_0
                out = out + torch.mm(Tx_k, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_k
        
        return out
    
    def compute_normalized_laplacian(self, num_nodes, edge_index, edge_weight=None):
        # Create adjacency matrix
        if edge_weight is not None:
            adaptive_edge_weight = edge_weight * torch.sigmoid(self.adaptive_weight)
        else:
            adaptive_edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            adaptive_edge_weight = adaptive_edge_weight * torch.sigmoid(self.adaptive_weight)
        
        # Build sparse adjacency matrix
        row, col = edge_index
        A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        A[row, col] = adaptive_edge_weight
        
        # Add self-loops
        A = A + torch.eye(num_nodes, device=edge_index.device)
        
        # Compute degree matrix
        D = torch.diag(torch.sum(A, dim=1))
        D_inv_sqrt = torch.diag(torch.pow(torch.sum(A, dim=1), -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        L = torch.eye(num_nodes, device=edge_index.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        
        return L


class Projection_dis_gene(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection_dis_gene, self).__init__()
        self.fc1 = Linear(2*input_dim, 2*hid_dim)
        self.fc2 = Linear(2*hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.layernorm_1 = nn.LayerNorm(2*hid_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(hid_dim, eps=1e-6)
        self.act_fn_score = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm_1(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.layernorm_2(x)
        return x

class Projection_dis_miRNA(nn.Module):

    def __init__(self, input_dim, hid_dim):
        super(Projection_dis_miRNA, self).__init__()
        self.fc1 = Linear(2*input_dim, 2*hid_dim)
        self.fc2 = Linear(2*hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.layernorm_1 = nn.LayerNorm(2*hid_dim, eps=1e-6)
        self.layernorm_2 = nn.LayerNorm(hid_dim, eps=1e-6)
        self.act_fn_score = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layernorm_1(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.layernorm_2(x)
        return x

class Attention_dis(nn.Module):

    def __init__(self, hid_dim):
        super(Attention_dis, self).__init__()
        self.fc1 = Linear( 2*hid_dim, 2*hid_dim)
        self.layernorm_1 = nn.LayerNorm( 2*hid_dim, eps=1e-6)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        attention_weights = torch.softmax(self.fc1(x), dim=-2)  
        x = torch.mul(x,attention_weights)
        x = self.act_fn(x)
        x = self.layernorm_1(x)
        return x

class Dis_score(nn.Module):
    def __init__(self,  hid_dim):
        super(Dis_score, self).__init__()
        self.fc2 = Linear( 2*hid_dim,1)

    def forward(self, x):
        x = self.fc2(x)
        return x

class AGCN(nn.Module):
    """
    Enhanced Adaptive Graph Convolutional Network
    Improved architecture with residual connections, attention, and better regularization
    """
    
    def __init__(self, nfeat, nhid, dropout=0.1, num_layers=4, residual=True, heads=8, alpha=0.2, max_attn_nodes: int = 2048):
        super(AGCN, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.dropout = dropout
        self.alpha = alpha
        self.max_attn_nodes = max_attn_nodes
        
        # Create multiple AGCN layers with increasing complexity
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # Input layer
        self.convs.append(AGCNConv(nfeat, nhid, bias=True, dropout=dropout, alpha=alpha, heads=heads, max_attn_nodes=max_attn_nodes))
        self.batch_norms.append(nn.BatchNorm1d(nhid))
        self.attention_layers.append(nn.MultiheadAttention(nhid, num_heads=heads, dropout=dropout, batch_first=True))
        self.residual_projs.append(nn.Linear(nfeat, nhid) if nfeat != nhid else nn.Identity())
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(AGCNConv(nhid, nhid, bias=True, dropout=dropout, alpha=alpha, heads=heads, max_attn_nodes=max_attn_nodes))
            self.batch_norms.append(nn.BatchNorm1d(nhid))
            self.attention_layers.append(nn.MultiheadAttention(nhid, num_heads=heads, dropout=dropout, batch_first=True))
            self.residual_projs.append(nn.Identity())
        
        # Output layer
        if num_layers > 1:
            self.convs.append(AGCNConv(nhid, nhid, bias=True, dropout=dropout, alpha=alpha, heads=heads, max_attn_nodes=max_attn_nodes))
            self.batch_norms.append(nn.BatchNorm1d(nhid))
            self.residual_projs.append(nn.Identity())
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(nhid)
        
        # Enhanced final projection layer
        self.final_proj = nn.Sequential(
            nn.Linear(nhid, nhid * 2),
            nn.LayerNorm(nhid * 2),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(nhid * 2, nhid),
            nn.LayerNorm(nhid),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout * 0.5)
        )
        
        # Global attention pooling (optional, guarded by node count)
        self.global_attention = nn.MultiheadAttention(nhid, num_heads=max(1, heads//2), dropout=dropout, batch_first=True)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        
        x_initial = x
        
        for i in range(self.num_layers):
            # Store input for residual connection
            x_in = x
            
            # Apply AGCN convolution
            try:
                x = self.convs[i](x, edge_index, edge_weight)
            except (TypeError, RuntimeError):
                # Fallback if edge_weight is not supported
                x = self.convs[i](x, edge_index)
            
            # Apply batch normalization
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply attention mechanism
            if i < len(self.attention_layers):
                x_attended, _ = self.attention_layers[i](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
                x = x_attended.squeeze(0)
            
            # Apply activation
            if i < self.num_layers - 1:  # Don't apply ReLU to the last layer
                x = F.leaky_relu(x, negative_slope=self.alpha)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Enhanced residual connection with projection
            if self.residual:
                if x_in.size(-1) == x.size(-1):
                    x = x + x_in
                else:
                    # Use projection for dimension mismatch
                    x = x + self.residual_projs[i](x_in)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global attention pooling for better representation (skip if too large)
        if x.size(0) <= self.max_attn_nodes:
            x_global, _ = self.global_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x_global = x_global.squeeze(0)
            # Combine local and global features
            x = x + 0.1 * x_global  # Small weight for global features
        
        # Final projection
        x = self.final_proj(x)
        
        return x


class ChebyshevAGCN(nn.Module):
    """
    Chebyshev-based Adaptive Graph Convolutional Network
    Enhanced version using Chebyshev polynomials for better accuracy
    """
    
    def __init__(self, nfeat, nhid, K=3):
        super(ChebyshevAGCN, self).__init__()
        self.conv1 = ChebyshevAGCNConv(nfeat, nhid, K=K, bias=True)
        self.conv2 = ChebyshevAGCNConv(nhid, nhid, K=K, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except (TypeError, RuntimeError):
            # Fallback if edge_weight is not supported
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            x = self.conv2(x, data.edge_index)
        return x

class GAT(nn.Module):

    def __init__(self, nfeat, nhid):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, bias=True)
        self.conv2 = GATConv(nhid, nhid, bias=True)

    def forward(self, data):
        try:
            x = self.conv1(data.x, data.edge_index, data.edge_weight)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index, data.edge_weight)
        except (TypeError, RuntimeError):
            # Fallback if edge_weight is not supported
            x = self.conv1(data.x, data.edge_index)
            x = F.leaky_relu(x)
            x = self.conv2(x, data.edge_index)
        return x


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) Encoder
    GIN is more expressive than GCN variants and can achieve better accuracy.
    Uses MLPs instead of linear transformations for better feature learning.
    
    Key advantages:
    - More powerful than GCN (as powerful as Weisfeiler-Lehman test)
    - Better at distinguishing non-isomorphic graphs
    - Often achieves superior accuracy on graph tasks
    """
    
    def __init__(self, nfeat, nhid, dropout=0.1, num_layers=4, residual=True, 
                 mlp_hidden=None, eps=0.0, train_eps=True):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.dropout = dropout
        
        # Default MLP hidden dimension
        if mlp_hidden is None:
            mlp_hidden = nhid
        
        # Create GIN layers with MLPs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # Learnable epsilon parameter for GIN
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_buffer('eps', torch.zeros(num_layers))
            self.eps.fill_(eps)
        
        # Input layer
        mlp1 = nn.Sequential(
            nn.Linear(nfeat, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, nhid),
            nn.BatchNorm1d(nhid),
            nn.ReLU()
        )
        self.convs.append(GINConv(mlp1, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(nhid))
        self.residual_projs.append(nn.Linear(nfeat, nhid) if nfeat != nhid else nn.Identity())
        
        # Hidden layers
        for i in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(nhid, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, nhid),
                nn.BatchNorm1d(nhid),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(nhid))
            self.residual_projs.append(nn.Identity())
        
        # Output layer
        if num_layers > 1:
            mlp_out = nn.Sequential(
                nn.Linear(nhid, mlp_hidden),
                nn.BatchNorm1d(mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, nhid),
                nn.BatchNorm1d(nhid)
            )
            self.convs.append(GINConv(mlp_out, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(nhid))
            self.residual_projs.append(nn.Identity())
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(nhid)
        
        # Final projection MLP for enhanced representation
        self.final_proj = nn.Sequential(
            nn.Linear(nhid, nhid * 2),
            nn.LayerNorm(nhid * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid * 2, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        
        x_initial = x
        
        for i in range(self.num_layers):
            # Store input for residual connection
            x_in = x
            
            # Apply GIN convolution
            # GINConv doesn't directly support edge_weight, so we handle it differently
            if edge_weight is not None:
                # For GIN, we can pass edge_weight if the implementation supports it
                # Otherwise, we'll use the default aggregation
                try:
                    x = self.convs[i](x, edge_index, edge_weight)
                except (TypeError, RuntimeError):
                    # Fallback if edge_weight is not supported by GINConv
                    x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            
            # Apply batch normalization
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if self.residual:
                if x_in.size(-1) == x.size(-1):
                    x = x + x_in
                else:
                    x = x + self.residual_projs[i](x_in)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Final projection
        x = self.final_proj(x)
        
        return x


class MVC(nn.Module):
    
    def __init__(self, 
                 g_encoder, 
                 m_encoder,
                 projection_dis_gene,
                 projection_dis_miRNA,
                 attention_dis,
                 dis_score,
                 spatial_fusion=None):
        super(MVC, self).__init__()
        self.g_encoder = g_encoder
        self.m_encoder = m_encoder
        self.projection_dis_gene = projection_dis_gene
        self.projection_dis_miRNA = projection_dis_miRNA
        self.attention_dis = attention_dis
        self.dis_score = dis_score
        self.spatial_fusion = spatial_fusion
        self.use_spatial_attention = spatial_fusion is not None

    def forward(self, g_data, m_data):
        g_h = self.g_encoder(g_data)
        m_h = self.m_encoder(m_data)
        return g_h,m_h
    
    def nonlinear_transformation_dis_gene(self, h):
        z = self.projection_dis_gene(h)
        return z


    def nonlinear_transformation_dis_miRNA(self, h):
        z = self.projection_dis_miRNA(h)
        return z

    def nonlinear_attention_dis(self, h):
        z = self.attention_dis(h)
        return z

    def nonlinear_dis_score(self, h):
        z = self.dis_score(h)
        return z

    def spatial_attention_fusion(self, gene_features, mirna_features):
        """
        Apply spatial attention fusion to gene and miRNA features
        """
        if self.use_spatial_attention:
            fused_features, fusion_weights = self.spatial_fusion(gene_features, mirna_features)
            return fused_features, fusion_weights
        else:
            # Fallback to simple concatenation if spatial attention is not used
            fused_features = torch.cat([gene_features, mirna_features], dim=-1)
            return fused_features, None

    def get_gene_embeddings(self, g_data):
        return self.g_encoder(g_data)

    def get_miRNA_embeddings(self, m_data):
        return self.m_encoder(m_data)
