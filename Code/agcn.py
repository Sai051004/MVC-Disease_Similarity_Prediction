#!/usr/bin/env python3
"""
Test script for enhanced AGCN model performance
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
import time
from utils import (
    load_sparse,
    mx_to_torch_sparse_tensor,
    generate_sparse_one_hot
)
from model import AGCN, GIN, GAT, Projection_dis_gene, Projection_dis_miRNA, Attention_dis, Dis_score, MVC
from attention import SpatialAttentionFusion
from explainability import compute_pair_contributions
import csv
import os
import matplotlib.pyplot as plt
import random


def pooling(x, y2x):
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    return torch.div(x, row_sum)


def set_seed(seed: int = 42):
    """Set seeds for reproducibility (important for consistent results)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class-balancing weight for BCEWithLogitsLoss.
    pos_weight = (#negative / #positive)
    """
    labels = labels.float()
    pos = labels.sum()
    neg = labels.numel() - pos
    # Avoid division by zero
    if pos == 0:
        return torch.tensor(1.0, device=labels.device)
    return torch.tensor(neg / pos, device=labels.device)


def create_enhanced_model(g_data, m_data, h_dim=256, dropout=0.15, heads=8, use_gin=False):
    """Create enhanced model with AGCN or GIN encoder
    
    Args:
        g_data: Gene graph data
        m_data: miRNA graph data
        h_dim: Hidden dimension
        dropout: Dropout rate
        heads: Number of attention heads (for AGCN)
        use_gin: If True, use GIN encoder instead of AGCN for better accuracy
    """
    if use_gin:
        # Use GIN (Graph Isomorphism Network) for potentially better accuracy
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
        # Use AGCN (Adaptive Graph Convolutional Network)
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
    return model


def train_enhanced_model(model, g_data, m_data, d2g, m2d, d2d_edge_index, d2d_link_labels, g2m, device, epochs=50):
    """Train enhanced model with advanced techniques. Returns (model, best_checkpoint_dict)."""
    # Enhanced optimizers
    params_group1 = [
        {'params': model.g_encoder.parameters(), 'weight_decay': 1e-4},
        {'params': model.m_encoder.parameters(), 'weight_decay': 1e-4}
    ]
    params_group2 = [
        {'params': model.projection_dis_gene.parameters(), 'weight_decay': 1e-4},
        {'params': model.projection_dis_miRNA.parameters(), 'weight_decay': 1e-4},
        {'params': model.attention_dis.parameters(), 'weight_decay': 1e-4},
        {'params': model.dis_score.parameters(), 'weight_decay': 1e-4}
    ]
    params_group3 = [{'params': model.spatial_fusion.parameters(), 'weight_decay': 1e-4}]

    optimizer1 = optim.AdamW(params_group1, lr=0.001, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(params_group2, lr=0.0005, betas=(0.9, 0.999))
    optimizer3 = optim.AdamW(params_group3, lr=0.0005, betas=(0.9, 0.999))
    
    # Learning rate schedulers
    scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler3 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer3, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Loss functions
    def nce_loss_g(gz, kgz, labels, tau=0.8):
        gz = torch.nn.functional.normalize(gz, dim=1)
        kgz = torch.nn.functional.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, 1)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss

    def nce_loss_m(gz, kgz, labels, tau=0.8):
        gz = torch.nn.functional.normalize(gz, dim=1)
        kgz = torch.nn.functional.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, -2, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, -2)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss

    def link_loss_dis(pre_value, d2d_link_labels, pos_weight):
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_function(pre_value.squeeze(), d2d_link_labels)
        return loss

    def compute_metrics(predictions, labels):
        probs = torch.sigmoid(predictions).squeeze()
        preds = (probs >= 0.5).float()
        labels = labels.float().squeeze()

        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = (correct / total) * 100

        from sklearn import metrics
        precision = metrics.precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = metrics.recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = metrics.f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        auroc = metrics.roc_auc_score(labels.cpu(), probs.cpu())
        ap = metrics.average_precision_score(labels.cpu(), probs.cpu())

        return accuracy, precision, recall, f1, auroc, ap

    # Training loop
    best_val_score = 0.0
    best_checkpoint = None
    patience = 20
    patience_counter = 0
    pos_weight = compute_pos_weight(d2d_link_labels).detach()
    
    for epoch in range(epochs):
        model.train()
        
        g_h, m_h = model(g_data, m_data)
        loss_1_g = nce_loss_g(g_h, m_h, g2m)
        loss_1_m = nce_loss_m(g_h, m_h, g2m)
        loss_1 = 0.2 * loss_1_g + 0.8 * loss_1_m

        d_h_1 = pooling(g_h, d2g.to_dense())
        d_h_2 = pooling(m_h, m2d.to_dense())
        d_train_1 = torch.cat((d_h_1[d2d_edge_index[:, 0]], d_h_1[d2d_edge_index[:, 1]]), 1)
        d_train_2 = torch.cat((d_h_2[d2d_edge_index[:, 0]], d_h_2[d2d_edge_index[:, 1]]), 1)

        fused_features, _ = model.spatial_attention_fusion(d_train_1, d_train_2)
        atten_hid_vec = model.nonlinear_attention_dis(fused_features)
        atten_score_vec = model.nonlinear_dis_score(atten_hid_vec)
        loss_2 = link_loss_dis(atten_score_vec, d2d_link_labels, pos_weight)

        loss = 0.3 * loss_1 + 0.7 * loss_2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        # Validation
        model.eval()
        with torch.no_grad():
            g_h_val, m_h_val = model(g_data, m_data)
            d_h_1_val = pooling(g_h_val, d2g.to_dense())
            d_h_2_val = pooling(m_h_val, m2d.to_dense())
            d_val_1 = torch.cat((d_h_1_val[d2d_edge_index[:, 0]], d_h_1_val[d2d_edge_index[:, 1]]), 1)
            d_val_2 = torch.cat((d_h_2_val[d2d_edge_index[:, 0]], d_h_2_val[d2d_edge_index[:, 1]]), 1)

            fused_val_features, _ = model.spatial_attention_fusion(d_val_1, d_val_2)
            atten_hid_vec_val = model.nonlinear_attention_dis(fused_val_features)
            atten_score_vec_val = model.nonlinear_dis_score(atten_hid_vec_val)

            val_acc, val_prec, val_rec, val_f1, val_auroc, val_ap = compute_metrics(atten_score_vec_val, d2d_link_labels)

            # Composite validation objective: prioritize F1, break ties with AUROC
            composite = val_f1 + 0.01 * val_auroc
            if composite > best_val_score:
                best_val_score = composite
                patience_counter = 0
                best_checkpoint = {
                    "model_state": model.state_dict(),
                    "best_f1": val_f1,
                    "best_auroc": val_auroc,
                    "best_ap": val_ap,
                    "epoch": epoch + 1,
                }
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val AUROC: {val_auroc:.4f}")

    return model, best_checkpoint


def evaluate_model(model, g_data, m_data, d2g, m2d, d2d_edge_index, d2d_link_labels, device):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        g_h = model.get_gene_embeddings(g_data)
        m_h = model.get_miRNA_embeddings(m_data)

        d_1 = pooling(g_h, d2g.to_dense())
        d_2 = pooling(m_h, m2d.to_dense())
        d_test_1 = torch.cat((d_1[d2d_edge_index[:, 0]], d_1[d2d_edge_index[:, 1]]), 1)
        d_test_2 = torch.cat((d_2[d2d_edge_index[:, 0]], d_2[d2d_edge_index[:, 1]]), 1)

        fused_test_features, _ = model.spatial_attention_fusion(d_test_1, d_test_2)
        atten_hid_vec_test = model.nonlinear_attention_dis(fused_test_features)
        atten_score_vec_test = model.nonlinear_dis_score(atten_hid_vec_test)

        y_true = d2d_link_labels.cpu().numpy()
        y_scores = torch.sigmoid(atten_score_vec_test).cpu().numpy().flatten()
        y_pred = (y_scores >= 0.5).astype(int)
        
        from sklearn import metrics
        accuracy = metrics.accuracy_score(y_true, y_pred) * 100
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        auroc = metrics.roc_auc_score(y_true, y_scores)
        ap = metrics.average_precision_score(y_true, y_scores)

        return {
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'auroc': auroc, 'ap': ap,
            'y_true': y_true, 'y_scores': y_scores, 'y_pred': y_pred
        }


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced AGCN/GIN Model")
    parser.add_argument("--data", default="../Dataset", help="path to dataset")
    parser.add_argument("--h_dim", default=256, type=int, help="dimension of hidden layer")
    parser.add_argument("--dropout", default=0.15, type=float, help="dropout rate")
    parser.add_argument("--heads", default=8, type=int, help="number of attention heads (for AGCN)")
    parser.add_argument("--epochs", default=50, type=int, help="training epochs")
    parser.add_argument("--use-gin", action="store_true", help="Use GIN encoder instead of AGCN for better accuracy")
    parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
    parser.add_argument("--save-best", default="best_model.pt", help="path to save best validation checkpoint")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load data
    print("📂 Loading data...")
    hnadj = load_sparse(args.data+"/hnet.npz")
    src = hnadj.row
    dst = hnadj.col
    hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
    hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min())
    hn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)

    d2g = load_sparse(args.data+"/d2g.npz")
    d2g = mx_to_torch_sparse_tensor(d2g)

    x = generate_sparse_one_hot(d2g.shape[1])
    g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

    g2m = load_sparse(args.data+"/gene2miRNA.npz")
    g2m = mx_to_torch_sparse_tensor(g2m).to_dense()

    mnadj = load_sparse(args.data+"/miRNA2miRNA.npz")
    src = mnadj.row
    dst = mnadj.col
    mn_edge_weight = torch.tensor(np.hstack((mnadj.data, mnadj.data)), dtype=torch.float)
    mn_edge_weight = (mn_edge_weight - mn_edge_weight.min()) / (mn_edge_weight.max() - mn_edge_weight.min())
    mn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)
    x_m = generate_sparse_one_hot(mnadj.shape[0])
    m_data = Data(x=x_m, edge_index=mn_edge_index, edge_weight=mn_edge_weight)

    m2d = load_sparse(args.data+"/miRNA2disease.npz")
    m2d = m2d.T
    m2d = mx_to_torch_sparse_tensor(m2d)

    # Move data to device
    g_data = g_data.to(device)
    m_data = m_data.to(device)
    g2m = g2m.to(device)
    d2g = d2g.to(device)
    m2d = m2d.to(device)

    # Load training and test data
    d2d_edge_index = np.loadtxt(args.data + "/train_for_ukb_ori_sample.txt")
    d2d_link_labels = np.loadtxt(args.data + "/train_for_ukb_ori_label.txt")
    d2d_edge_index = torch.tensor(d2d_edge_index, dtype=torch.long).to(device)
    d2d_link_labels = torch.tensor(d2d_link_labels).to(device)
    
    d2d_edge_index_test = np.loadtxt(args.data + "/test_sample_ukb_interpre_high.txt")
    d2d_link_labels_test = np.loadtxt(args.data + "/test_label_ukb_interpre_high.txt")
    d2d_edge_index_test = torch.tensor(d2d_edge_index_test, dtype=torch.long).to(device)
    d2d_link_labels_test = torch.tensor(d2d_link_labels_test).to(device)

    # Create and train enhanced model
    encoder_name = "GIN" if args.use_gin else "AGCN"
    print(f"🔧 Creating enhanced {encoder_name} model...")
    model = create_enhanced_model(g_data, m_data, args.h_dim, args.dropout, args.heads, use_gin=args.use_gin)
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("🚀 Training enhanced model...")
    start_time = time.time()
    model, best_ckpt = train_enhanced_model(
        model, g_data, m_data, d2g, m2d, 
        d2d_edge_index, d2d_link_labels, g2m, device, args.epochs
    )

    # Save best validation checkpoint (state_dict + metrics)
    if best_ckpt is not None:
        if args.save_best:
            torch.save(best_ckpt, args.save_best)
            print(f"💾 Saved best checkpoint (F1={best_ckpt['best_f1']:.4f}, AUROC={best_ckpt['best_auroc']:.4f}, AUPRC={best_ckpt.get('best_ap', 0):.4f}) to {args.save_best}")
        # Load best weights for final evaluation
        model.load_state_dict(best_ckpt["model_state"])
    else:
        print("⚠️ Warning: No best checkpoint found. Saving current model state...")
        if args.save_best:
            torch.save({
                "model_state": model.state_dict(),
                "epoch": args.epochs,
                "note": "Final model state (no validation checkpoint found)"
            }, args.save_best)
            print(f"💾 Saved final model state to {args.save_best}")
    training_time = time.time() - start_time
    
    # Evaluate model
    print("🧪 Evaluating enhanced model...")
    test_metrics = evaluate_model(
        model, g_data, m_data, d2g, m2d,
        d2d_edge_index_test, d2d_link_labels_test, device
    )
    
    print(f"\n🎉 ENHANCED {encoder_name} RESULTS:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test AUPRC: {test_metrics['ap']:.4f}")
    
    # Generate high-quality visualizations
    print("\n📊 Generating visualizations...")
    try:
        from visualization import (
            plot_roc_curve, plot_precision_recall, plot_metrics_comparison,
            save_metrics_to_file, ensure_results_dir
        )
        
        model_name = f"{encoder_name.lower()}_{args.h_dim}dim"
        results_dir = ensure_results_dir("results")
        
        # Plot ROC curve
        plot_roc_curve(
            test_metrics['y_true'], 
            test_metrics['y_scores'],
            model_name=model_name,
            save_dir=results_dir
        )
        
        # Plot Precision-Recall curve
        plot_precision_recall(
            test_metrics['y_true'],
            test_metrics['y_scores'],
            model_name=model_name,
            save_dir=results_dir
        )
        
        # Plot metrics comparison
        metrics_dict = {
            'Accuracy': test_metrics['accuracy'] / 100.0,
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1-Score': test_metrics['f1'],
            'AUROC': test_metrics['auroc'],
            'AUPRC': test_metrics['ap']
        }
        plot_metrics_comparison(metrics_dict, model_name=model_name, save_dir=results_dir)
        
        # Save metrics to file
        save_metrics_to_file(metrics_dict, model_name=model_name, save_dir=results_dir)
        
        print(f"✅ All visualizations saved to {results_dir}/figures/")
        
    except ImportError:
        print("⚠️ Visualization module not found. Skipping plot generation.")
    except Exception as e:
        print(f"⚠️ Error generating visualizations: {e}")
    
    # Compute and export contributions
    print("📤 Computing contributions...")
    try:
        model.eval()
        with torch.no_grad():
            g_h = model.get_gene_embeddings(g_data)
            m_h = model.get_miRNA_embeddings(m_data)

        contrib = compute_pair_contributions(
            model, g_h, m_h, d2g.to_dense(), m2d.to_dense(),
            d2d_edge_index_test, top_k=10, id_to_gene=None, id_to_mirna=None
        )

        # Export CSV to organized results directory
        try:
            from visualization import ensure_results_dir
            results_dir = ensure_results_dir("results")
        except ImportError:
            results_dir = "results"
            os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        
        csv_path = os.path.join(results_dir, "metrics", f"contributions_{encoder_name.lower()}.csv")
        header = ["disease_i", "disease_j"] + \
                 [f"gene_id_{k}" for k in range(1, 11)] + \
                 [f"gene_pct_{k}" for k in range(1, 11)] + \
                 [f"mirna_id_{k}" for k in range(1, 11)] + \
                 [f"mirna_pct_{k}" for k in range(1, 11)]
        
        with open(csv_path, mode='w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(header)
            for item in contrib:
                try:
                    di, dj = item['pair']
                    genes = item.get('genes', [])[:10]
                    mirnas = item.get('mirnas', [])[:10]
                    while len(genes) < 10:
                        genes.append({"id": 'NA', "percent": 0.0})
                    while len(mirnas) < 10:
                        mirnas.append({"id": 'NA', "percent": 0.0})
                    gene_ids = [g.get('id', 'NA') for g in genes]
                    gene_pcts = [round(float(g.get('percent', 0.0)), 6) for g in genes]
                    mir_ids = [m.get('id', 'NA') for m in mirnas]
                    mir_pcts = [round(float(m.get('percent', 0.0)), 6) for m in mirnas]
                    row = [di, dj] + gene_ids + gene_pcts + mir_ids + mir_pcts
                except Exception:
                    row = ['NA', 'NA'] + ['NA']*10 + [0.0]*10 + ['NA']*10 + [0.0]*10
                writer.writerow(row)
        
        print(f"✅ Contributions exported to {csv_path}")
        
    except Exception as e:
        print(f"❌ Error computing contributions: {e}")


if __name__ == "__main__":
    main()


