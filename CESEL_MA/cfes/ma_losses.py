"""CESel-MA objectives and the multi-axis selection gap (decisions/2026-10-26-final-model-spec.md section 2).

u (N,K) cue utilities of the CES pool (ces_losses.utilities), y (N,) class (1 = fake), cond (N,) manipulation condition,
M (N,5) per-image axes in ma_labels.COLS order: gender, age_bin, ita, ita_bin, group (gender x race), missing = -1/NaN.
cue_mask (K,) = validated cues (floor.cues). delta = 0.1, n_min = 4, class weight w_real = 2 (ma.real_weight), w_fake = 1.

L_floor-MA = sum_c w_c [ GxR_c + sum_{a in gender, age} D_{a,c} + S_c ]
  GxR_c  = the CESel floor kept as is: sum over (c, condition m, gender x race group g) cells with n >= n_min of
           sum_k relu( sg(ubar_{k,c,m}) - ubar_{g,k,c,m} - delta )^2          (ces_losses.floor_loss, per class)
  D_{a,c}= sum over groups v of axis a with n >= n_min (labelled images of class c) of
           sum_k relu( sg(ubar_{k,c}) - ubar_{v,k,c} - delta )^2,      ubar_{k,c} = mean over ALL images of class c
  S_c    = sum over the 8 anchors t_j = linspace(lo, hi, 8) of the TRAIN ITA range [lo, hi] with effective sample size
           n_eff(t) = (sum_i w_i)^2 / sum_i w_i^2 >= n_eff_min (4), w_i(t) = exp(-(ITA_i - t)^2 / 2h^2), h = (hi - lo)/8,
           of sum_k relu( sg(ubar_{k,c}) - ubar_{k,c}(t) - delta )^2,  ubar_{k,c}(t) = sum_i w_i u_ik / sum_i w_i
           (images of class c with a finite ITA)
Frequency warm-up: L_freq_aux = balanced BCE( a^T c_freq + a_0 , y ) over the CES pool (own linear head, weight 0.2),
           active for the first ma.freq_warmup.epochs (1) epochs only (step < epochs x steps_per_epoch), 0 afterwards.
Selection gap (validation only): FPR per group at the fitted common intercept, for each axis gender / age_bin /
           ita_bin / group; a group is eligible when it holds >= min_videos (2) distinct REAL validation videos (the FPR
           denominator; cluster key = video); gap_a = max - min eligible FPR (0 with < 2 eligible groups);
           max_gap = max_a gap_a. Selection value = val AUC - 0.5 max_gap inside the unchanged Amendment-9 eligibility.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import ces_losses as CL

G_, A_, T_, TB_, GR_ = 0, 1, 2, 3, 4          # columns of M


def _zero(t):
    return t.sum() * 0.0


def kernel_anchors(lo, hi, n=8):
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"ITA range must have hi > lo, got {lo}, {hi}")
    return torch.linspace(lo, hi, n, dtype=torch.float64), (hi - lo) / n


def discrete_floor(u, sel_c, labels, cue_mask, pooled, n_min, delta):
    """sum over groups v >= 0 with n >= n_min (within sel_c) of sum_k relu(pooled - ubar_v - delta)^2."""
    terms = []
    for v in torch.unique(labels[sel_c]).tolist():
        if v < 0:
            continue
        s = sel_c & (labels == v)
        if int(s.sum()) < n_min:
            continue
        d = F.relu(pooled - u[s].mean(0) - delta) ** 2
        terms.append(d[cue_mask].sum())
    return terms


def kernel_floor(u, sel_c, ita, cue_mask, pooled, anchors, h, n_eff_min, delta):
    """continuous skin-tone floor of one class; returns (terms, n_anchor_used, n_eff per anchor)."""
    ok = sel_c & torch.isfinite(ita)
    if int(ok.sum()) == 0:
        return [], []
    t = ita[ok].double()
    uu = u[ok]
    W = torch.exp(-(t[None, :] - anchors.to(t.device)[:, None]) ** 2 / (2.0 * h * h))          # (A, n)
    neff = W.sum(1) ** 2 / (W ** 2).sum(1).clamp(min=1e-300)
    terms, used = [], []
    for j in range(W.shape[0]):
        if float(neff[j]) < n_eff_min:
            continue
        w = (W[j] / W[j].sum()).to(uu.dtype)
        ub = (w[:, None] * uu).sum(0)
        d = F.relu(pooled - ub - delta) ** 2
        terms.append(d[cue_mask].sum())
        used.append(j)
    return terms, used


def floor_ma_loss(u, y, cond, M, cue_mask, fcfg, macfg, anchors, h):
    """L_floor-MA (module docstring). Returns (loss, stats)."""
    n_min, delta = int(fcfg.get("n_min", 4)), float(fcfg.get("delta", 0.1))
    n_eff_min = float(macfg.get("n_eff_min", 4))
    wr = float(macfg.get("real_weight", 2.0))
    axes = macfg.get("floor_axes", ["gxr", "gender", "age", "skin"])
    cue_mask = cue_mask.to(u.device)
    M = M.to(u.device)
    gr = M[:, GR_].long()
    total, st = [], {}
    if not cue_mask.any():
        return _zero(u), {"floor_ma_cells": 0}
    for c in torch.unique(y).tolist():
        wc = wr if int(c) == 0 else 1.0
        sel_c = y == c
        name = "real" if int(c) == 0 else "fake"
        pooled = u[sel_c].mean(0).detach()
        parts = {}
        if "gxr" in axes:                                     # CESel floor (class x condition x group), this class only
            l, s = CL.floor_loss(u[sel_c], y[sel_c], gr[sel_c], cond[sel_c], cue_mask, fcfg)
            parts["gxr"] = (l, int(s.get("floor_cells", 0)))
        if "gender" in axes:
            t = discrete_floor(u, sel_c, M[:, G_].long(), cue_mask, pooled, n_min, delta)
            parts["gender"] = (torch.stack(t).sum() if t else _zero(u), len(t))
        if "age" in axes:
            t = discrete_floor(u, sel_c, M[:, A_].long(), cue_mask, pooled, n_min, delta)
            parts["age"] = (torch.stack(t).sum() if t else _zero(u), len(t))
        if "skin" in axes:
            t, used = kernel_floor(u, sel_c, M[:, T_], cue_mask, pooled, anchors, h, n_eff_min, delta)
            parts["skin"] = (torch.stack(t).sum() if t else _zero(u), len(t))
        for k, (l, n) in parts.items():
            total.append(wc * l)
            st[f"floor_ma_{k}_{name}"] = float(l)
            st[f"floor_ma_{k}_{name}_cells"] = n
    loss = torch.stack(total).sum() if total else _zero(u)
    st["floor_ma_cells"] = int(sum(v for k, v in st.items() if k.endswith("_cells")))
    return loss, st


def freq_aux_factor(step, warm_steps):
    """1 during the frequency warm-up (step < warm_steps), 0 afterwards."""
    return 1.0 if (step is not None and int(step) < int(warm_steps)) else 0.0


# ------------------------------------------------------------------------------------------------ selection gap
def group_fprs(dec, y, lab, clusters, min_videos=2):
    """{group: FPR %} over groups (lab >= 0) with >= min_videos distinct real videos."""
    out = {}
    for v in np.unique(lab):
        if v < 0:
            continue
        r = (lab == v) & (y == 0)
        if not r.any() or len(set(clusters[r].tolist())) < min_videos:
            continue
        out[int(v)] = float(100 * dec[r].mean())
    return out


def ma_gaps(y, z, intercept, M, clusters, min_videos=2):
    """Multi-axis FPR gaps at the fitted common intercept (decision z + b >= 0). M (N,5) numpy (COLS order)."""
    y, z = np.asarray(y), np.asarray(z, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    dec = (z + float(intercept)) >= 0
    res = {"min_videos": int(min_videos), "axes": {}}
    for name, j in (("gender", G_), ("age_bin", A_), ("ita_bin", TB_), ("group", GR_)):
        lab = np.asarray(M[:, j]).astype(np.int64)
        f = group_fprs(dec, y, lab, clusters, min_videos)
        gap = (max(f.values()) - min(f.values())) if len(f) >= 2 else 0.0
        res["axes"][name] = {"group_FPR": f, "gap": float(gap), "n_groups": len(f)}
    res["max_gap"] = float(max(a["gap"] for a in res["axes"].values()))
    res["argmax_axis"] = max(res["axes"], key=lambda k: res["axes"][k]["gap"])
    return res
