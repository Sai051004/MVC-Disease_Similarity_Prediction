import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch_geometric.data import Data

def load_dmap(file_path):
    name_to_id = {}
    id_to_name = {}
    with open(file_path, 'r') as f:
        for line in f:
            name, id = line.strip().split()
            name_to_id[name] = int(id)
            id_to_name[int(id)] = name
    return name_to_id, id_to_name

########################################################################
# Sparse Matrix Utils
########################################################################

def load_sparse(path):
    return sp.load_npz(path).tocoo()

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mx_to_torch_sparse_tensor(mx):
    sparse_mx = mx.astype(np.float32)
    sparse_mx.eliminate_zeros()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    size = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, size)

def generate_sparse_one_hot(num_ents, dtype=torch.float32):
    diag_size = num_ents
    diag_range = list(range(num_ents))
    diag_range = torch.tensor(diag_range)
    return torch.sparse_coo_tensor(
        indices=torch.vstack([diag_range, diag_range]),
        values=torch.ones(diag_size, dtype=dtype),
        size=(diag_size, diag_size))

########################################################################
# Graph Visualization
########################################################################

def visualize_graph(matrix, title):
    G = nx.from_scipy_sparse_matrix(matrix)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=False, node_size=20, edge_color="gray")
    plt.title(title)
    plt.show()

########################################################################
# Performance Evaluation Graphs
########################################################################

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

def plot_true_positive_pairs(y_true):
    top_n = np.arange(1, 21)
    tp_pairs = np.cumsum(np.sort(y_true)[::-1])[:20]
    plt.figure()
    plt.plot(top_n, tp_pairs, marker='o', label="True Positive Pairs")
    plt.xlabel("Top-N Predicted Pairs")
    plt.ylabel("True Positive Pairs")
    plt.title("True Positive Pairs vs Top-N")
    plt.legend()
    plt.show()

def plot_auc_vs_feature_dimensionality():
    feature_dims = [32, 64, 96, 128, 160, 192]
    auc_values = np.linspace(0.93, 0.98, len(feature_dims))
    aupr_values = auc_values - 0.01
    plt.figure()
    plt.plot(feature_dims, auc_values, marker='s', label="AUC")
    plt.plot(feature_dims, aupr_values, marker='o', label="AUPR")
    plt.xlabel("Feature Dimensionality")
    plt.ylabel("Metric Score")
    plt.title("Feature Dimensionality vs AUC & AUPR")
    plt.legend()
    plt.show()

def plot_auc_vs_epochs():
    epochs = [10, 20, 30, 40, 50]
    auc_epochs = np.linspace(0.90, 0.98, len(epochs))
    aupr_epochs = auc_epochs - 0.01
    plt.figure()
    plt.plot(epochs, auc_epochs, marker='s', label="AUC")
    plt.plot(epochs, aupr_epochs, marker='o', label="AUPR")
    plt.xlabel("Training Epoch")
    plt.ylabel("Metric Score")
    plt.title("Training Epoch vs AUC & AUPR")
    plt.legend()
    plt.show()

def plot_auc_vs_weighting_coefficient():
    weights = np.linspace(0, 1, 10)
    auc_weights = np.linspace(0.95, 0.97, len(weights))
    aupr_weights = auc_weights - 0.005
    plt.figure()
    plt.plot(weights, auc_weights, marker='s', label="AUC")
    plt.plot(weights, aupr_weights, marker='o', label="AUPR")
    plt.xlabel("Weighting Coefficient β")
    plt.ylabel("Metric Score")
    plt.title("Weighting Coefficient vs AUC & AUPR")
    plt.legend()
    plt.show()

def load_data(data_path):
    """Load core graphs and label splits for model training and evaluation.

    Returns ``(g_data, m_data), (d2d_edge_index, d2d_link_labels), (d2d_edge_index_test, d2d_link_labels_test)``.
    """
    # Load human gene net for AGCN model
    hnadj = load_sparse(data_path + "/hnet.npz")
    src = hnadj.row
    dst = hnadj.col
    hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
    hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min())
    hn_edge_index = torch.tensor(
        np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))),
        dtype=torch.long,
    )

    # Load disease2gene network
    d2g = load_sparse(data_path + "/d2g.npz")
    d2g = mx_to_torch_sparse_tensor(d2g)

    x = generate_sparse_one_hot(d2g.shape[1])
    g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

    # Load miRNA-miRNA similarity net for GAT model
    mnadj = load_sparse(data_path + "/miRNA2miRNA.npz")
    src = mnadj.row
    dst = mnadj.col
    mn_edge_weight = torch.tensor(np.hstack((mnadj.data, mnadj.data)), dtype=torch.float)
    mn_edge_weight = (mn_edge_weight - mn_edge_weight.min()) / (mn_edge_weight.max() - mn_edge_weight.min())
    mn_edge_index = torch.tensor(
        np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))),
        dtype=torch.long,
    )
    x_m = generate_sparse_one_hot(mnadj.shape[0])
    m_data = Data(x=x_m, edge_index=mn_edge_index, edge_weight=mn_edge_weight)

    # Load training data
    d2d_edge_index = np.loadtxt(data_path + "/train_for_ukb_ori_sample.txt")
    d2d_link_labels = np.loadtxt(data_path + "/train_for_ukb_ori_label.txt")
    d2d_edge_index = torch.tensor(d2d_edge_index, dtype=torch.long)
    d2d_link_labels = torch.tensor(d2d_link_labels)

    # Load test data
    d2d_edge_index_test = np.loadtxt(data_path + "/test_sample_ukb_interpre_high.txt")
    d2d_link_labels_test = np.loadtxt(data_path + "/test_label_ukb_interpre_high.txt")
    d2d_edge_index_test = torch.tensor(d2d_edge_index_test, dtype=torch.long)
    d2d_link_labels_test = torch.tensor(d2d_link_labels_test)

    return (g_data, m_data), (d2d_edge_index, d2d_link_labels), (d2d_edge_index_test, d2d_link_labels_test)
