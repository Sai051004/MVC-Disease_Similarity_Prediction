import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Lightweight utilities to add explainability without changing model internals

ATTN_ATTR_NAME = "_last_attention_weights"


def _attach_attention_hooks(module: torch.nn.Module) -> None:
    """Register lightweight forward hooks to capture attention weights.

    This inspects child modules; for spatial attention modules it computes
    attention weights pre-forward and stores them on the module, then a
    post-forward hook copies the cached weights to a public attribute so they
    can be collected later without impacting the main forward outputs.
    """
    try:
        from attention import SpatialAttentionModule
    except Exception:
        SpatialAttentionModule = None

    def save_attn_hook(mod, inputs, output):
        # Capture attention weights if present on the module during forward
        if hasattr(mod, "_cached_attention_weights"):
            setattr(mod, ATTN_ATTR_NAME, mod._cached_attention_weights)

    def compute_attn_hook(mod, inputs):
        # For SpatialAttentionModule, reconstruct attention weights before forward
        if not SpatialAttentionModule or not isinstance(mod, SpatialAttentionModule):
            return
        x = inputs[0]
        batch_size, seq_len, _ = x.shape
        x_norm = mod.layer_norm1(x)
        Q = mod.query_proj(x_norm).view(batch_size, seq_len, mod.num_heads, mod.head_dim).transpose(1, 2)
        K = mod.key_proj(x_norm).view(batch_size, seq_len, mod.num_heads, mod.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (mod.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Cache on module so the save hook can store it after forward
        mod._cached_attention_weights = attention_weights.detach().cpu()

    # Recursively register hooks
    for child in module.modules():
        # Pre-forward to compute attention; post-forward to persist it
        child.register_forward_pre_hook(compute_attn_hook)
        child.register_forward_hook(save_attn_hook)


def enable_attention_capture(model: torch.nn.Module) -> None:
    """Enable attention weight capture across the model by registering hooks."""
    _attach_attention_hooks(model)


def collect_attention_weights(model: torch.nn.Module) -> List[torch.Tensor]:
    """Collect attention weights previously captured via registered hooks.

    Returns
    -------
    List[torch.Tensor]
        A list of attention weight tensors captured from attention modules.
    """
    attn_list: List[torch.Tensor] = []
    for m in model.modules():
        if hasattr(m, ATTN_ATTR_NAME):
            weights = getattr(m, ATTN_ATTR_NAME)
            if weights is not None:
                attn_list.append(weights)
    return attn_list


def compute_fusion_saliency(model: torch.nn.Module,
                            d_train_1: torch.Tensor,
                            d_train_2: torch.Tensor,
                            loss_fn,
                            target_labels: torch.Tensor,
                            retain_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient-based saliency for fusion inputs.

    This performs a forward pass through the fusion + scoring heads and then a
    backward pass to obtain absolute gradients with respect to the fusion
    inputs. The result highlights which input elements are most influential
    for the current predictions under the provided loss.
    """
    d_train_1 = d_train_1.detach().requires_grad_(True)
    d_train_2 = d_train_2.detach().requires_grad_(True)

    fused_features, _ = model.spatial_attention_fusion(d_train_1, d_train_2)
    atten_hid_vec = model.nonlinear_attention_dis(fused_features)
    logits = model.nonlinear_dis_score(atten_hid_vec)

    loss = loss_fn(logits.squeeze(), target_labels)
    loss.backward(retain_graph=retain_graph)

    saliency_1 = d_train_1.grad.detach().abs()
    saliency_2 = d_train_2.grad.detach().abs()
    return saliency_1, saliency_2


def summarize_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """Summarize saliency by averaging across feature dimension when 2D.

    Useful for quick per-sample visualization.
    """
    if saliency.dim() == 2:
        return saliency.mean(dim=1)
    return saliency


def explanation_artifacts(model: torch.nn.Module) -> Dict[str, Any]:
    """Return a dictionary of available explainability artifacts.

    Currently includes attention weights captured from attention modules.
    """
    return {
        "attention_weights": collect_attention_weights(model)
    }


def _compute_single_pair_contributions(
    model: torch.nn.Module,
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair: torch.Tensor,
    top_k: int = 10,
    id_to_gene: Dict[int, str] = None,
    id_to_mirna: Dict[int, str] = None,
) -> Dict[str, Any]:
    """Compute top-k gene/miRNA contribution percentages for one pair.

    Uses gradients with respect to disease embeddings and aggregates per-entity
    contributions weighted by pooling coefficients, then normalizes to percent.
    """
    device = g_h.device
    i = int(pair[0].item())
    j = int(pair[1].item())

    # Build disease embeddings from pooled encoders
    # di, dj for gene branch
    di_g = torch.matmul(d2g_dense[i].to(device), g_h)  # [h]
    dj_g = torch.matmul(d2g_dense[j].to(device), g_h)  # [h]
    # di, dj for miRNA branch
    di_m = torch.matmul(m2d_dense[i].to(device), m_h)
    dj_m = torch.matmul(m2d_dense[j].to(device), m_h)

    # Require grad to obtain saliency on disease embeddings
    di_g = di_g.detach().requires_grad_(True)
    dj_g = dj_g.detach().requires_grad_(True)
    di_m = di_m.detach().requires_grad_(True)
    dj_m = dj_m.detach().requires_grad_(True)

    # Build pair inputs for fusion: concatenate (i,j)
    d_pair_gene = torch.cat([di_g, dj_g], dim=0).unsqueeze(0)  # [1, 2h]
    d_pair_mir = torch.cat([di_m, dj_m], dim=0).unsqueeze(0)   # [1, 2h]

    # Forward through fusion and heads to get a single logit
    fused_features, _ = model.spatial_attention_fusion(d_pair_gene, d_pair_mir)
    atten_hid_vec = model.nonlinear_attention_dis(fused_features)
    logit = model.nonlinear_dis_score(atten_hid_vec).squeeze()  # scalar

    # Backprop to get gradients wrt disease embeddings
    model.zero_grad(set_to_none=True)
    if di_g.grad is not None:
        di_g.grad.zero_()
    if dj_g.grad is not None:
        dj_g.grad.zero_()
    if di_m.grad is not None:
        di_m.grad.zero_()
    if dj_m.grad is not None:
        dj_m.grad.zero_()
    logit.backward(retain_graph=True)

    grad_di_g = di_g.grad.detach()  # [h]
    grad_dj_g = dj_g.grad.detach()
    grad_di_m = di_m.grad.detach()
    grad_dj_m = dj_m.grad.detach()

    # Contribution scores per gene: combine (i and j) sides
    # Use elementwise |grad| * |embedding| aggregated by the pooling weights
    g_abs = g_h.detach().abs()  # [N_genes, h]
    m_abs = m_h.detach().abs()  # [N_mirna, h]

    def _scores_from_side(pool_row: torch.Tensor, grad_vec: torch.Tensor, emb_abs: torch.Tensor) -> torch.Tensor:
        # pool_row: [N_entities]
        # grad_vec: [h]
        # emb_abs: [N_entities, h]
        weighted = (emb_abs * grad_vec.abs().unsqueeze(0))  # [N, h]
        contrib = weighted.mean(dim=1)  # [N]
        # scale by pooling weights magnitude
        contrib = contrib * pool_row.abs()
        return contrib

    contrib_gene_i = _scores_from_side(d2g_dense[i].to(device), grad_di_g, g_abs)
    contrib_gene_j = _scores_from_side(d2g_dense[j].to(device), grad_dj_g, g_abs)
    contrib_gene = contrib_gene_i + contrib_gene_j  # [N_genes]

    contrib_mir_i = _scores_from_side(m2d_dense[i].to(device), grad_di_m, m_abs)
    contrib_mir_j = _scores_from_side(m2d_dense[j].to(device), grad_dj_m, m_abs)
    contrib_mir = contrib_mir_i + contrib_mir_j  # [N_mir]

    # Normalize to percentages
    def _topk_percentages(scores: torch.Tensor, id_to_name: Dict[int, str]):
        scores_cpu = scores.detach().cpu()
        total = float(scores_cpu.sum().item()) if scores_cpu.sum().item() > 0 else 1.0
        k = min(top_k, scores_cpu.numel())
        vals, idx = torch.topk(scores_cpu, k)
        out = []
        for v, ix in zip(vals.tolist(), idx.tolist()):
            pct = 100.0 * (v / total)
            name = id_to_name.get(ix, str(ix)) if id_to_name else str(ix)
            out.append({"id": ix, "name": name, "score": v, "percent": pct})
        return out

    genes_top = _topk_percentages(contrib_gene, id_to_gene)
    mirnas_top = _topk_percentages(contrib_mir, id_to_mirna)

    return {
        "pair": [i, j],
        "genes": genes_top,
        "mirnas": mirnas_top,
    }


def compute_pair_contributions(
    model: torch.nn.Module,
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair_indices: torch.Tensor,
    top_k: int = 10,
    id_to_gene: Dict[int, str] = None,
    id_to_mirna: Dict[int, str] = None,
) -> List[Dict[str, Any]]:
    """Compute per-pair top-k contribution percentages for many pairs.

    Parameters
    ----------
    pair_indices : torch.Tensor
        Tensor of shape [M, 2] listing (disease_i, disease_j) pairs.
    """
    results: List[Dict[str, Any]] = []
    for pair in pair_indices:
        try:
            result = _compute_single_pair_contributions(
                model, g_h, m_h, d2g_dense, m2d_dense, pair, top_k, id_to_gene, id_to_mirna
            )
            results.append(result)
        except Exception:
            continue
    return results


def compute_pair_contributions_full(
    model: torch.nn.Module,
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair: torch.Tensor,
) -> Dict[str, Any]:
    """Compute full-length percentage contributions for one pair.

    Returns a dict containing per-gene and per-miRNA percentage vectors that
    sum to ~100 each (subject to numerical precision).
    """
    device = g_h.device
    i = int(pair[0].item())
    j = int(pair[1].item())

    di_g = torch.matmul(d2g_dense[i].to(device), g_h)
    dj_g = torch.matmul(d2g_dense[j].to(device), g_h)
    di_m = torch.matmul(m2d_dense[i].to(device), m_h)
    dj_m = torch.matmul(m2d_dense[j].to(device), m_h)

    di_g = di_g.detach().requires_grad_(True)
    dj_g = dj_g.detach().requires_grad_(True)
    di_m = di_m.detach().requires_grad_(True)
    dj_m = dj_m.detach().requires_grad_(True)

    d_pair_gene = torch.cat([di_g, dj_g], dim=0).unsqueeze(0)
    d_pair_mir = torch.cat([di_m, dj_m], dim=0).unsqueeze(0)

    fused_features, _ = model.spatial_attention_fusion(d_pair_gene, d_pair_mir)
    atten_hid_vec = model.nonlinear_attention_dis(fused_features)
    logit = model.nonlinear_dis_score(atten_hid_vec).squeeze()

    model.zero_grad(set_to_none=True)
    if di_g.grad is not None:
        di_g.grad.zero_()
    if dj_g.grad is not None:
        dj_g.grad.zero_()
    if di_m.grad is not None:
        di_m.grad.zero_()
    if dj_m.grad is not None:
        dj_m.grad.zero_()
    logit.backward(retain_graph=True)

    grad_di_g = di_g.grad.detach()
    grad_dj_g = dj_g.grad.detach()
    grad_di_m = di_m.grad.detach()
    grad_dj_m = dj_m.grad.detach()

    g_abs = g_h.detach().abs()
    m_abs = m_h.detach().abs()

    def _scores_from_side(pool_row: torch.Tensor, grad_vec: torch.Tensor, emb_abs: torch.Tensor) -> torch.Tensor:
        weighted = (emb_abs * grad_vec.abs().unsqueeze(0))
        contrib = weighted.mean(dim=1)
        contrib = contrib * pool_row.abs()
        return contrib

    contrib_gene = _scores_from_side(d2g_dense[i].to(device), grad_di_g, g_abs) + \
                   _scores_from_side(d2g_dense[j].to(device), grad_dj_g, g_abs)
    contrib_mir = _scores_from_side(m2d_dense[i].to(device), grad_di_m, m_abs) + \
                  _scores_from_side(m2d_dense[j].to(device), grad_dj_m, m_abs)

    gene_sum = float(contrib_gene.sum().item()) if contrib_gene.sum().item() > 0 else 1.0
    mir_sum = float(contrib_mir.sum().item()) if contrib_mir.sum().item() > 0 else 1.0
    genes_percent = (100.0 * contrib_gene.cpu() / gene_sum).tolist()
    mirnas_percent = (100.0 * contrib_mir.cpu() / mir_sum).tolist()

    return {
        'pair': [i, j],
        'genes_percent': genes_percent,
        'mirnas_percent': mirnas_percent,
    }


# ==============================
# SHAP / LIME (model-agnostic)
# ==============================

def _prepare_pair_features(
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair_indices: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build concatenated pair features [di_g||dj_g||di_m||dj_m] for pairs.

    Output shape is [num_pairs, 4*h]. These features are used by SHAP/LIME.
    """
    if device is None:
        device = g_h.device
    pairs = pair_indices.to(torch.long)
    # Disease embeddings via pooling
    di_g = torch.matmul(d2g_dense[pairs[:, 0]].to(device), g_h)  # [N, h]
    dj_g = torch.matmul(d2g_dense[pairs[:, 1]].to(device), g_h)  # [N, h]
    di_m = torch.matmul(m2d_dense[pairs[:, 0]].to(device), m_h)  # [N, h]
    dj_m = torch.matmul(m2d_dense[pairs[:, 1]].to(device), m_h)  # [N, h]
    x_gene = torch.cat([di_g, dj_g], dim=1)  # [N, 2h]
    x_mir = torch.cat([di_m, dj_m], dim=1)  # [N, 2h]
    x_all = torch.cat([x_gene, x_mir], dim=1)  # [N, 4h]
    return x_all


class _PairPredictor:
    """Wrap model to map pair features to probabilities for SHAP/LIME.

    Accepts numpy arrays of shape [N, 4*h] and returns probability scores.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device).float()  # [N, 4h]
            # Split into gene and mir branches
            half = x.shape[1] // 2
            x_gene = x[:, :half]
            x_mir = x[:, half:]
            fused, _ = self.model.spatial_attention_fusion(x_gene, x_mir)
            z = self.model.nonlinear_attention_dis(fused)
            logit = self.model.nonlinear_dis_score(z).squeeze(-1)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        # KernelExplainer expects 2D outputs; ensure shape [N]
        return prob


def shap_explain_pairs(
    model: torch.nn.Module,
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair_indices: torch.Tensor,
    background_size: int = 50,
    shap_nsamples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP KernelExplainer values for multiple disease pairs.

    Returns a dict with original pairs, input matrix X, expected value and the
    SHAP value matrix aligned with X.
    """
    try:
        import shap  # type: ignore
    except Exception as e:
        raise RuntimeError("SHAP is not installed. Please install 'shap'.") from e

    device = next(model.parameters()).device
    X_all = _prepare_pair_features(g_h, m_h, d2g_dense.to(device), m2d_dense.to(device), pair_indices, device)
    X_all_np = X_all.detach().cpu().numpy()

    # Choose a small background subset for efficiency
    num_bg = min(background_size, X_all_np.shape[0])
    bg_idx = np.random.choice(X_all_np.shape[0], size=num_bg, replace=False)
    X_bg = X_all_np[bg_idx]

    predictor = _PairPredictor(model, device)
    explainer = shap.KernelExplainer(predictor, X_bg)
    shap_values = explainer.shap_values(X_all_np, nsamples=shap_nsamples)

    # KernelExplainer returns list for multi-output; ours is single-output, normalize to array
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    return {
        'pairs': pair_indices.detach().cpu().tolist(),
        'X': X_all_np,
        'expected_value': explainer.expected_value,
        'shap_values': shap_values,
    }


def lime_explain_pair(
    model: torch.nn.Module,
    g_h: torch.Tensor,
    m_h: torch.Tensor,
    d2g_dense: torch.Tensor,
    m2d_dense: torch.Tensor,
    pair: torch.Tensor,
    num_features: int = 20,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """Run LIME Tabular explanation for a single (disease_i, disease_j) pair.

    Returns a dict with the target pair and a list of feature weights as
    (feature_name, weight) tuples from LIME.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception as e:
        raise RuntimeError("LIME is not installed. Please install 'lime'.") from e

    device = next(model.parameters()).device
    # Build a small synthetic dataset around the target pair using random pairs as background
    pair_indices = pair.view(1, 2)
    X_target = _prepare_pair_features(g_h, m_h, d2g_dense.to(device), m2d_dense.to(device), pair_indices, device)
    X_target_np = X_target.detach().cpu().numpy()

    # Create background by sampling random pairs from all diseases
    num_diseases = d2g_dense.shape[0]
    rand_i = torch.randint(0, num_diseases, (max(200, num_samples // 2),))
    rand_j = torch.randint(0, num_diseases, (max(200, num_samples // 2),))
    bg_pairs = torch.stack([rand_i, rand_j], dim=1)
    X_bg = _prepare_pair_features(g_h, m_h, d2g_dense.to(device), m2d_dense.to(device), bg_pairs.to(device), device)
    X_bg_np = X_bg.detach().cpu().numpy()

    predictor = _PairPredictor(model, device)
    explainer = LimeTabularExplainer(
        X_bg_np,
        mode='classification',
        discretize_continuous=False,
        feature_names=[f'f_{k}' for k in range(X_bg_np.shape[1])],
        class_names=['score'],
    )

    # LIME expects predict_proba returning [N, 2] for binary; wrap accordingly
    def predict_proba(x: np.ndarray) -> np.ndarray:
        p = predictor(x).reshape(-1, 1)
        return np.hstack([1 - p, p])

    exp = explainer.explain_instance(
        X_target_np[0],
        predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )

    return {
        'pair': pair_indices.squeeze(0).detach().cpu().tolist(),
        'weights': exp.as_list(),
    }

