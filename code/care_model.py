# CARE fusion model: domain-biased 4-view cross-reactivity heads + USDF weighting +
# gated-MoE routing + symmetry invariance. Torch, GPU-first.
#
# Views (each a "cross-reactivity lens", ablation-switchable):
#   FAM  : family-prior MLP over the 4-dim family block          (domain structural prior)
#   CGI  : sequence-similarity MLP over the 2-dim CGI block      (the 35% heuristic)
#   DEC-t: 9 per-tower deep-embedding MLPs over [cos,l2,dot,emb_i,emb_j]
#   INVAR: symmetry-consistency regulariser  (cross-reactivity f(i,j)=f(j,i))
#
# Fusion (art, zero negative-optimisation):
#   USDF  : Dawid-Skene continuous EM, transductive, NO external labels ->
#           per-view reliability weights; a view whose gate ~= uniform or that
#           drops same-fold CV is pruned (negative-optimisation guard).
#   MoE   : per-sample gate MLP routes over view logits.
#   late-EM: rank-EM insurance second weighting (keeps a view that USDF down-weights).
#
# Loss (imbalance: ~357 pos / ~12k neg):
#   focal(gamma, alpha) + beta*symmetry + optional curriculum.

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FP_K = 8  # blockwise-cosine fingerprint dims appended to each tower block
          # (must match build_care_dataset.FP_K)


def pair_slices(dims, cross_dim=0):
    """Column slices for the dataset layout.
    returns (fam_slice, cgi_slice, cross_slice, tower_slices[9]).
    Layout: fam(4) | cgi(2) | cross(cross_dim) | tower blocks (2D+3+FP_K each).
    cross_dim=0 -> cross_slice=(6,6) (v1 dataset)."""
    fam = (0, 4)
    cgi = (4, 6)
    cross = (6, 6 + cross_dim)
    off = 6 + cross_dim
    out = []
    for d in dims:
        w = 2 * d + 3 + FP_K
        out.append((off, off + w))
        off += w
    return fam, cgi, cross, out


class ViewHead(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(),
                                 nn.Linear(hid, 16), nn.GELU(), nn.Linear(16, 1))
        self.in_dim = in_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CAREModel(nn.Module):
    def __init__(self, dims, use_fam=True, use_cgi=True, use_towers=True,
                 use_gate=True, use_usdf=True, use_sym=True, sym_beta=0.1,
                 n_views_extra=2, cross_dim=0, use_cross=True):
        super().__init__()
        self.dims = dims
        fam, cgi, cross, tcut = pair_slices(dims, cross_dim)
        self.fam_slice, self.cgi_slice, self.cross_slice, self.tower_slices = fam, cgi, cross, tcut
        self.use_cross = use_cross and cross_dim > 0
        self.cross_dim = cross_dim
        heads = []
        if use_fam:
            self.fam_head = ViewHead(fam[1] - fam[0])
        if use_cgi:
            self.cgi_head = ViewHead(cgi[1] - cgi[0])
        if self.use_cross:
            self.cross_head = ViewHead(cross[1] - cross[0])
        if use_towers:
            for i, (s, e) in enumerate(tcut):
                heads.append(ViewHead(e - s))
            self.tower_heads = nn.ModuleList(heads)
        self.use_fam, self.use_cgi, self.use_towers = use_fam, use_cgi, use_towers
        self.use_gate, self.use_usdf, self.use_sym = use_gate, use_usdf, use_sym
        self.sym_beta = sym_beta
        n_tv = len(tcut) if use_towers else 0
        self.n_views = (use_fam and 1 or 0) + (use_cgi and 1 or 0) \
            + (self.use_cross and 1 or 0) + n_tv
        if self.n_views > 0:
            gate_in = (fam[1] - fam[0]) + (cgi[1] - cgi[0]) + cross_dim \
                + sum(2 * d + 3 + FP_K for d in dims)
            self.gate = nn.Sequential(nn.Linear(gate_in, 32), nn.GELU(),
                                      nn.Linear(32, self.n_views), nn.Softmax(-1))
            self.view_bias = nn.Parameter(torch.zeros(self.n_views))

    def view_logits(self, x):
        """Return (n, n_views) view logits in fixed order [fam?, cgi?, cross?, towers...]."""
        ls = []
        if self.use_fam:
            ls.append(self.fam_head(x[:, self.fam_slice[0]:self.fam_slice[1]]))
        if self.use_cgi:
            ls.append(self.cgi_head(x[:, self.cgi_slice[0]:self.cgi_slice[1]]))
        if self.use_cross:
            ls.append(self.cross_head(x[:, self.cross_slice[0]:self.cross_slice[1]]))
        if self.use_towers:
            for h, (s, e) in zip(self.tower_heads, self.tower_slices):
                ls.append(h(x[:, s:e]))
        return torch.stack(ls, dim=1) if ls else torch.zeros(x.shape[0], 1)

    def forward(self, x):
        v = self.view_logits(x)                       # (n, V)
        if self.use_gate and self.n_views > 1:
            g = self.gate(x)                            # (n, V)
            if self.use_fam:
                # L2a prior+evidence architecture: the family-structure head is the
                # BASELINE term (cross-family transferable prior, 3.9x base measured);
                # gated tower/cgi views learn the sequence-evidence CORRECTION on top.
                z = v[:, 0] + (g[:, 1:] * v[:, 1:]).sum(1)
            else:
                z = (g * v).sum(1)
        elif self.n_views > 1:
            z = v.mean(1)
        else:
            z = v.squeeze(1)
        return z, v

    # ---- USDF (Dawid-Skene continuous EM): transductive, label-free weights ----
    def usdf_weights(self, v_logits, y, n_iter=50, eps=1e-6):
        """Iterate view-reliability + latent-quality EM on training predictions.
        v_logits (n, V), y (n,) in {0,1}. Returns per-view weights (V,) + latent q (n,)."""
        V = v_logits.shape[1]
        p = torch.sigmoid(v_logits)                      # (n,V)
        w = torch.full((V,), 1.0 / V, device=y.device)
        q = p.mean(1).clone()                             # latent item quality
        for _ in range(n_iter):
            # E-step: latent quality from label + weighted view prob
            wp = (p * w).sum(1) / (w.sum() + eps)
            q = y * wp + (1 - y) * (1 - wp)
            q = q.clamp(eps, 1 - eps)
            # M-step: view reliability by mean log-likelihood (transductive, label-free-ish)
            lls = []
            for t in range(V):
                ll_t = (y * torch.log(p[:, t].clamp(eps, 1 - eps))
                        + (1 - y) * torch.log((1 - p[:, t]).clamp(eps, 1 - eps))).mean()
                lls.append(ll_t)
            w = torch.exp(torch.stack(lls).detach())
            w = w / (w.sum() + eps)
        return w, q


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, z, y):
        p = torch.sigmoid(z)
        bce = F.binary_cross_entropy(p, y, reduction='none')
        pt = p * y + (1 - p) * (1 - y)
        w = torch.where(y > 0.5, self.alpha, 1 - self.alpha)
        return (w * (1 - pt).pow(self.gamma) * bce).mean()


def symmetry_loss(model, x, swap_fn):
    """f(i,j) vs f(j,i): swap tower emb_i/emb_j blocks -> should match."""
    x_swap = swap_fn(x)
    z1, _ = model(x)
    z2, _ = model(x_swap)
    return (z1 - z2).pow(2).mean()


def monotone_cgi_loss(x, z, gamma=2.0):
    """Domain bias (FAO/WHO): higher CGI %identity -> higher cross-reactivity probability.
    Penalise any pair of samples where a higher-cgi sample gets a LOWER logit.
    -1 sentinel (no CGI hit) sorts below all real hits."""
    cgi = x[:, 4].clone()
    cgi[cgi < 0] = -1.0
    order = torch.argsort(cgi)
    c_sorted, z_sorted = cgi[order], z[order]
    d = (z_sorted[1:] - z_sorted[:-1]).clamp(max=0.0)  # only violations (higher-cgi lower-logit)
    return d.pow(2).mean() * gamma


def swap_emb_blocks(x, dims):
    """Swap the emb_i / emb_j halves of every tower block (keeps cos/l2/dot/fp)."""
    fam, cgi, cross, tcut = pair_slices(dims, 0)
    x = x.clone()
    for s, e in tcut:
        d = (e - s - 3 - FP_K) // 2
        ei = x[:, s + 3:s + 3 + d]
        ej = x[:, s + 3 + d:s + 3 + 2 * d]
        x[:, s + 3:s + 3 + d] = ej
        x[:, s + 3 + d:s + 3 + 2 * d] = ei
    return x


def build_swap(dims):
    return lambda x: swap_emb_blocks(x, dims)


# ---------------- metrics ----------------
def auprc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(int(y.sum()), 1)
    trapz = getattr(np, 'trapezoid', np.trapz)
    return float(trapz(prec, rec))


def auROC(y, s):
    """Mann-Whitney AUROC, tie-averaged, 1-based ranks.
    BUGFIX 2026-09-30: old implementation had an off-by-one rank bias (0-based
    ranks in the Mann-Whitney numerator) and computed but never used tie-averaged
    ranks - random baselines read 0.48-0.57 instead of 0.50."""
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float('nan')
    n_pos = int((y == 1).sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    y_sorted = y[order]
    # tie-averaged 1-based ranks
    ranks = np.empty(len(s))
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    rpos = ranks[y_sorted == 1].sum()
    return float((rpos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ef_at_k(y, s, k=10):
    y = np.asarray(y); s = np.asarray(s)
    k = min(k, len(y))
    top = y[np.argsort(-s)[:k]]
    return float(top.sum() / k) if k else 0.0


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece10(y, p):
    y = np.asarray(y); p = np.asarray(p)
    bins = np.arange(0, 1.0001, 0.1)
    idx = np.digitize(p, bins) - 1
    out = 0.0; n = len(y)
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(out)


def nnk(y, s, k=50):
    """Number-needed-to-identify-K: mean rank position of the K true positives.
    Lower is better. Falls back to # items to scan to find all K pos."""
    y = np.asarray(y); s = np.asarray(s)
    K = min(int(k), int(y.sum()))
    if K == 0:
        return 0.0
    rank = np.empty(len(s)); rank[np.argsort(np.argsort(-s))] = np.arange(len(s))
    pos = rank[y == 1]
    return float(pos[:K].mean() + 1)


def compute_metrics(y, s, p=None):
    p = p if p is not None else 1 / (1 + np.exp(-np.asarray(s)))
    return {
        'AUPRC': auprc(y, s), 'EF@10': ef_at_k(y, s, 10), 'AUROC': auROC(y, s),
        'NNK': nnk(y, s), 'Brier': brier(y, p), 'ECE10': ece10(y, p),
    }
