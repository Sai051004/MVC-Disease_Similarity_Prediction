import time
import os.path as osp
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from sklearn.model_selection import KFold
import itertools

torch.manual_seed(0)

# Pooling function
def pooling(x, y2x):
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    return torch.div(x, row_sum)


class CrossValidationTrainer(object):
    """
    Cross-Validation Trainer for Hyperparameter Tuning
    Performs 5-fold CV on training set only, keeping test set untouched
    """
    
    def __init__(self, model_class, tau, device, n_folds=5):
        self.model_class = model_class
        self.tau = tau
        self.device = device
        self.n_folds = n_folds
        self.writer = SummaryWriter()
        self.best_params = None
        self.best_cv_score = 0.0

    def load_data(self, g_data, m_data, labels, d2g, m2d, dis_path):
        self.g_data = g_data.to(self.device)
        self.m_data = m_data.to(self.device)
        self.labels = labels.to(self.device)
        self.d2g = d2g
        self.m2d = m2d
        self.dis_path = dis_path

    # Contrastive Loss for Gene
    def nce_loss_g(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, 1)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss

    # Contrastive Loss for miRNA
    def nce_loss_m(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, -2, keepdim=True)
        positives_sum = torch.sum(similarity_matrix * labels, -2)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss

    # Link Loss for Disease Prediction
    def link_loss_dis(self, pre_value, d2d_link_labels):
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(pre_value.squeeze(), d2d_link_labels)
        return loss

    # Metric Calculation
    def compute_metrics(self, predictions, labels):
        probs = torch.sigmoid(predictions).squeeze()
        preds = (probs >= 0.5).float()
        labels = labels.float().squeeze()

        # Accuracy Calculation
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = (correct / total) * 100

        # Classification Metrics
        precision = metrics.precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = metrics.recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = metrics.f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        error_rate = 100 - accuracy

        return accuracy, precision, recall, f1, error_rate

    def train_fold(self, params, train_indices, val_indices, epochs, metric='f1'):
        """Train and validate on a single fold; returns validation score for chosen metric"""
        # Build a fresh model for this fold with provided hyperparameters
        from model import AGCN, GAT, Projection_dis_gene, Projection_dis_miRNA, Attention_dis, Dis_score, MVC
        h_dim = params.get('h_dim', 128)
        dropout = params.get('dropout', 0.1)
        num_layers = params.get('num_layers', 3)
        residual = params.get('residual', True)
        lr1 = params.get('lr1', 0.003)
        lr2 = params.get('lr2', 0.001)

        # Temporarily override tau for this fold if provided
        original_tau = self.tau
        self.tau = params.get('tau', self.tau)

        g_encoder = AGCN(
            nfeat=self.g_data.x.shape[1], 
            nhid=h_dim, 
            dropout=dropout, 
            num_layers=4, 
            residual=residual,
            heads=params.get('heads', 8),
            alpha=0.2
        )
        m_encoder = GAT(nfeat=self.m_data.x.shape[1], nhid=h_dim)
        projection_dis_gene = Projection_dis_gene(h_dim, h_dim)
        projection_dis_miRNA = Projection_dis_miRNA(h_dim, h_dim)
        attention_dis = Attention_dis(h_dim)
        dis_score = Dis_score(h_dim)
        model = MVC(g_encoder, m_encoder, projection_dis_gene, projection_dis_miRNA, attention_dis, dis_score)
        model = model.to(self.device)
        model.train()
        
        # Split training data for this fold
        d2d_edge_index = np.loadtxt(self.dis_path + "/train_for_ukb_ori_sample.txt")
        d2d_link_labels = np.loadtxt(self.dis_path + "/train_for_ukb_ori_label.txt")
        
        # Use only training indices for this fold
        fold_train_indices = train_indices
        fold_val_indices = val_indices
        
        # Split into train/val for this fold
        train_edge_index = torch.tensor(d2d_edge_index[fold_train_indices], dtype=torch.long)
        train_labels = torch.tensor(d2d_link_labels[fold_train_indices])
        val_edge_index = torch.tensor(d2d_edge_index[fold_val_indices], dtype=torch.long)
        val_labels = torch.tensor(d2d_link_labels[fold_val_indices])

        # Enhanced optimizers with weight decay
        weight_decay = params.get('weight_decay', 1e-4)
        params_group1 = [
            {'params': model.g_encoder.parameters(), 'weight_decay': weight_decay},
            {'params': model.m_encoder.parameters(), 'weight_decay': weight_decay}
        ]
        params_group2 = [
            {'params': model.projection_dis_gene.parameters(), 'weight_decay': weight_decay},
            {'params': model.projection_dis_miRNA.parameters(), 'weight_decay': weight_decay},
            {'params': model.attention_dis.parameters(), 'weight_decay': weight_decay},
            {'params': model.dis_score.parameters(), 'weight_decay': weight_decay}
        ]

        # Use AdamW for better generalization
        optimizer1 = optim.AdamW(params_group1, lr1, betas=(0.9, 0.999))
        optimizer2 = optim.AdamW(params_group2, lr=lr2, betas=(0.9, 0.999))
        
        # Learning rate schedulers
        scheduler1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2, eta_min=1e-6)
        scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2, eta_min=1e-6)

        best_val_score = 0.0
        patience = params.get('patience', 15)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            g_h, m_h = model(self.g_data, self.m_data)
            loss_1_g = self.nce_loss_g(g_h, m_h, self.labels)
            loss_1_m = self.nce_loss_m(g_h, m_h, self.labels)
            loss_1 = 0.3 * loss_1_g + 0.7 * loss_1_m

            d_h_1 = pooling(g_h, self.d2g.to_dense())
            d_h_2 = pooling(m_h, self.m2d.to_dense())
            d_train_1 = torch.cat((d_h_1[train_edge_index[:, 0]], d_h_1[train_edge_index[:, 1]]), 1)
            d_train_2 = torch.cat((d_h_2[train_edge_index[:, 0]], d_h_2[train_edge_index[:, 1]]), 1)

            hid_vec_1 = model.nonlinear_transformation_dis_gene(d_train_1)
            hid_vec_2 = model.nonlinear_transformation_dis_miRNA(d_train_2)
            hid_vec = torch.cat((hid_vec_1, hid_vec_2), 1)
            atten_hid_vec = model.nonlinear_attention_dis(hid_vec)
            atten_score_vec = model.nonlinear_dis_score(atten_hid_vec)
            loss_2 = self.link_loss_dis(atten_score_vec, train_labels)

            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer1.step()
            optimizer2.step()
            
            # Update learning rates
            scheduler1.step()
            scheduler2.step()

            # Validation
            model.eval()
            with torch.no_grad():
                g_h_val, m_h_val = model(self.g_data, self.m_data)
                d_h_1_val = pooling(g_h_val, self.d2g.to_dense())
                d_h_2_val = pooling(m_h_val, self.m2d.to_dense())
                d_val_1 = torch.cat((d_h_1_val[val_edge_index[:, 0]], d_h_1_val[val_edge_index[:, 1]]), 1)
                d_val_2 = torch.cat((d_h_2_val[val_edge_index[:, 0]], d_h_2_val[val_edge_index[:, 1]]), 1)

                hid_vec_1_val = model.nonlinear_transformation_dis_gene(d_val_1)
                hid_vec_2_val = model.nonlinear_transformation_dis_miRNA(d_val_2)
                hid_vec_val = torch.cat((hid_vec_1_val, hid_vec_2_val), 1)
                atten_hid_vec_val = model.nonlinear_attention_dis(hid_vec_val)
                atten_score_vec_val = model.nonlinear_dis_score(atten_hid_vec_val)

                val_acc, val_prec, val_rec, val_f1, val_err = self.compute_metrics(atten_score_vec_val, val_labels)
                # Compute AUROC as well for metric selection
                try:
                    val_auroc = metrics.roc_auc_score(val_labels.cpu(), atten_score_vec_val.detach().cpu())
                except Exception:
                    val_auroc = 0.0

                current_score = val_f1 if metric == 'f1' else val_auroc
                if current_score > best_val_score:
                    best_val_score = current_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                    break
                    
            model.train()

        # Restore original tau
        self.tau = original_tau

        return best_val_score

    def cross_validate(self, param_grid, epochs=10, metric='f1'):
        """
        Perform 5-fold cross-validation for hyperparameter tuning
        
        param_grid: dict with hyperparameters to tune
        Example: {
            'h_dim': [64, 128, 256],
            'dropout': [0.1, 0.2, 0.3],
            'lr1': [0.001, 0.003, 0.005],
            'lr2': [0.0005, 0.001, 0.002]
        }
        """
        print(f"Starting {self.n_folds}-fold cross-validation for hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")
        
        # Load training data for CV
        d2d_edge_index = np.loadtxt(self.dis_path + "/train_for_ukb_ori_sample.txt")
        n_samples = len(d2d_edge_index)
        
        # Create K-fold splits
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"Total parameter combinations to test: {len(param_combinations)}")
        
        best_params = None
        best_cv_score = 0.0
        
        for i, param_combo in enumerate(param_combinations):
            print(f"\n--- Testing combination {i+1}/{len(param_combinations)} ---")
            
            # Create parameter dict
            params = dict(zip(param_names, param_combo))
            print(f"Parameters: {params}")
            
            # Perform cross-validation
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
                print(f"  Fold {fold+1}/{self.n_folds}")
                
                # Train on this fold with fresh model
                fold_score = self.train_fold(params, train_idx, val_idx, epochs, metric)
                fold_scores.append(fold_score)
                print(f"    Fold {fold+1} {metric.upper()}: {fold_score:.4f}")
            
            # Calculate mean CV score
            mean_cv_score = np.mean(fold_scores)
            std_cv_score = np.std(fold_scores)
            
            print(f"  Mean CV {metric.upper()}: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
            
            # Update best parameters
            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_params = params.copy()
                print(f"  🎯 New best! CV {metric.upper()}: {best_cv_score:.4f}")
        
        print(f"\n🎉 Cross-validation completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best CV {metric.upper()} score: {best_cv_score:.4f}")
        
        self.best_params = best_params
        self.best_cv_score = best_cv_score
        
        return best_params, best_cv_score

    def train_final_model(self, epochs=200):
        """Train final model with best parameters on full training set"""
        if self.best_params is None:
            print("No best parameters found. Run cross_validate first.")
            return None
            
        print(f"Training for {epochs} epochs...")
        
        # Create final model with best parameters
        h_dim = self.best_params.get('h_dim', 128)
        lr1 = self.best_params.get('lr1', 0.001)
        lr2 = self.best_params.get('lr2', 0.0005)
        dropout = self.best_params.get('dropout', 0.1)
        num_layers = self.best_params.get('num_layers', 3)
        residual = self.best_params.get('residual', True)
        # use tuned tau if provided
        self.tau = self.best_params.get('tau', self.tau)
        
        from model import AGCN, GAT, Projection_dis_gene, Projection_dis_miRNA, Attention_dis, Dis_score, MVC
        
        g_encoder = AGCN(
            nfeat=self.g_data.x.shape[1], 
            nhid=h_dim, 
            dropout=dropout, 
            num_layers=4, 
            residual=residual,
            heads=self.best_params.get('heads', 8),
            alpha=0.2
        )
        m_encoder = GAT(nfeat=self.m_data.x.shape[1], nhid=h_dim)
        projection_dis_gene = Projection_dis_gene(h_dim, h_dim)
        projection_dis_miRNA = Projection_dis_miRNA(h_dim, h_dim)
        attention_dis = Attention_dis(h_dim)
        dis_score = Dis_score(h_dim)
        
        final_model = MVC(g_encoder, m_encoder, projection_dis_gene, 
                             projection_dis_miRNA, attention_dis, dis_score)
        final_model = final_model.to(self.device)
        
        # Train on full training set
        t0 = time.time()
        
        # Load full training data
        d2d_edge_index = np.loadtxt(self.dis_path + "/train_for_ukb_ori_sample.txt")
        d2d_link_labels = np.loadtxt(self.dis_path + "/train_for_ukb_ori_label.txt")
        d2d_edge_index = torch.tensor(d2d_edge_index, dtype=torch.long)
        d2d_link_labels = torch.tensor(d2d_link_labels)

        # Optimizers
        params_group1 = [{'params': final_model.g_encoder.parameters()},
                         {'params': final_model.m_encoder.parameters()}]
        params_group2 = [{'params': final_model.projection_dis_gene.parameters()},
                         {'params': final_model.projection_dis_miRNA.parameters()},
                         {'params': final_model.attention_dis.parameters()},
                         {'params': final_model.dis_score.parameters()}]

        optimizer1 = optim.RMSprop(params_group1, lr1)
        optimizer2 = optim.SGD(params_group2, lr=lr2)

        for epoch in range(epochs):
            final_model.train()
            g_h, m_h = final_model(self.g_data, self.m_data)
            loss_1_g = self.nce_loss_g(g_h, m_h, self.labels)
            loss_1_m = self.nce_loss_m(g_h, m_h, self.labels)
            loss_1 = 0.3 * loss_1_g + 0.7 * loss_1_m

            d_h_1 = pooling(g_h, self.d2g.to_dense())
            d_h_2 = pooling(m_h, self.m2d.to_dense())
            d_train_1 = torch.cat((d_h_1[d2d_edge_index[:, 0]], d_h_1[d2d_edge_index[:, 1]]), 1)
            d_train_2 = torch.cat((d_h_2[d2d_edge_index[:, 0]], d_h_2[d2d_edge_index[:, 1]]), 1)

            hid_vec_1 = final_model.nonlinear_transformation_dis_gene(d_train_1)
            hid_vec_2 = final_model.nonlinear_transformation_dis_miRNA(d_train_2)
            hid_vec = torch.cat((hid_vec_1, hid_vec_2), 1)
            atten_hid_vec = final_model.nonlinear_attention_dis(hid_vec)
            atten_score_vec = final_model.nonlinear_dis_score(hid_vec)
            loss_2 = self.link_loss_dis(atten_score_vec, d2d_link_labels)

            loss = loss_1 + loss_2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        print(f"Training completed in {time.time() - t0:.2f}s")
        
        return final_model

    def evaluate_on_test(self, model):
        """Evaluate the final model on the untouched test set"""
        model.eval()
        with torch.no_grad():
            g_h = model.get_gene_embeddings(self.g_data)
            m_h = model.get_miRNA_embeddings(self.m_data)

            # Load Test Data (untouched)
            d2d_edge_index_test = np.loadtxt(self.dis_path + "/test_sample_ukb_interpre_high.txt")
            d2d_link_labels_test = np.loadtxt(self.dis_path + "/test_label_ukb_interpre_high.txt")
            d2d_edge_index_test = torch.tensor(d2d_edge_index_test, dtype=torch.long)
            d2d_link_labels_test = torch.tensor(d2d_link_labels_test)

            d_1 = pooling(g_h, self.d2g.to_dense())
            d_2 = pooling(m_h, self.m2d.to_dense())
            d_test_1 = torch.cat((d_1[d2d_edge_index_test[:, 0]], d_1[d2d_edge_index_test[:, 1]]), 1)
            d_test_2 = torch.cat((d_2[d2d_edge_index_test[:, 0]], d_2[d2d_edge_index_test[:, 1]]), 1)

            hid_vec_test_1 = model.nonlinear_transformation_dis_gene(d_test_1)
            hid_vec_test_2 = model.nonlinear_transformation_dis_miRNA(d_test_2)
            hid_vec_test = torch.cat((hid_vec_test_1, hid_vec_test_2), 1)
            atten_hid_vec_test = model.nonlinear_attention_dis(hid_vec_test)
            atten_score_vec_test = model.nonlinear_dis_score(hid_vec_test)

            # Test Metrics
            test_loss = self.link_loss_dis(atten_score_vec_test, d2d_link_labels_test).item()
            acc, prec, rec, f1, err = self.compute_metrics(atten_score_vec_test, d2d_link_labels_test)
            auroc = metrics.roc_auc_score(d2d_link_labels_test, atten_score_vec_test)
            ap = metrics.average_precision_score(d2d_link_labels_test, atten_score_vec_test)

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Accuracy: {acc:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f}")
            print(f"F1: {f1:.4f} | AUROC: {auroc:.4f} | AUPRC: {ap:.4f}")

        return auroc, ap
