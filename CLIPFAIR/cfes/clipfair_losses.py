"""Losses of the four matched-backbone fairness baselines (decisions/2026-10-25-cesel-plan.md, Amendment 3), ported
from the official code. Every function cites the source file:line it reproduces. Paths:

  AIF = agents/CFES/third_party/ff2id/AI-Face-FairnessBench/training      (Ju et al. WACV'24: DAW-FDD, DAG-FDD)
  FG  = agents/CFES/third_party/ff2id/Fairness-Generalization/training    (Lin et al. CVPR'24: PG-FDD)
  DM  = agents/DINGREPRO_LIN/code/Ding_mod                                 (Ding et al. CVPR'26: SFD + GDA)

The only systematic change: the official dual searches evaluate their objective with `.mean().item()` on the GPU
tensor; here the same float32 arithmetic runs on a detached CPU copy (identical values, no per-evaluation GPU sync).
The searched dual variable then enters the differentiable loss exactly as in the official code.
"""
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize


# ------------------------------------------------------------------------------------------------ CVaR duals
def threshplus(x):
    """AIF detectors/daw_fdd.py:55-59 (= dag_fdd.py:56-60 = FG detectors/fair_df_detector.py:175-179): relu(x)."""
    return torch.relu(x.clone())


def search_func(losses, alpha):
    """AIF daw_fdd.py:61-62 / dag_fdd.py:62-63 / FG fair_df_detector.py:181-182:
    x -> x + (1/alpha) * mean(relu(losses - x)). `losses` = detached CPU float32 tensor (see module docstring)."""
    return lambda x: x + (1.0 / alpha) * (threshplus(losses - x).mean().item())


def searched_lamda_loss(losses, lam, alpha):
    """AIF daw_fdd.py:64-65 / dag_fdd.py:65-66 / FG fair_df_detector.py:187-188 (differentiable in `losses`)."""
    return lam + ((1.0 / alpha) * torch.mean(threshplus(losses - lam)))


def search_func_smooth(losses, alpha, tau1, tau2):
    """FG fair_df_detector.py:184-185, verbatim (including the inf guard returning float max)."""
    return lambda x: x + (tau2 / (2 * alpha)) * (x ** 2) + (1.0 / alpha) * threshplus(
        tau1 * torch.log(1 + torch.exp((losses - x) / tau1))).mean().item() \
        if not torch.isinf(torch.exp((losses - x) / tau1)).any() else np.finfo(float).max


def searched_lamda_loss_smooth(losses, lam, alpha, tau1, tau2):
    """FG fair_df_detector.py:190-191, verbatim."""
    return lam + (tau2 / (2 * alpha)) * (lam ** 2) + (1.0 / alpha) * (
        torch.mean(threshplus(tau1 * torch.log(1 + torch.exp((losses - lam) / tau1)))))


def _fminbound(fun, losses_np):
    """The official call: optimize.fminbound(f, min(losses) - 1000.0, max(losses)) (scipy defaults)."""
    return optimize.fminbound(fun, np.min(losses_np) - 1000.0, np.max(losses_np))


def cvar(losses, alpha):
    """CVaR_alpha of per-sample losses by the official dual search (dag_fdd.py:80-82). Returns (loss, lambda*)."""
    lc = losses.detach().float().cpu()
    lam = _fminbound(search_func(lc, alpha), lc.numpy())
    return searched_lamda_loss(losses, float(lam), alpha), float(lam)


# ------------------------------------------------------------------------------------------------ DAW / DAG
def per_sample_bce(logit, y):
    """AIF loss/daw_bce_loss.py:10,24: BCEWithLogitsLoss(reduction='none')(inputs.squeeze(1), targets.float())."""
    return F.binary_cross_entropy_with_logits(logit.float(), y.float(), reduction="none")


def daw_fdd_loss(logit, y, groups, inner_alpha=0.9, outer_alpha=0.5):
    """DAW-FDD, AIF detectors/daw_fdd.py:73-94 (defaults :75-76 inner 0.9, outer 0.5): per intersectional group the
    CVaR_inner of the per-sample BCE (dual by fminbound), then CVaR_outer over the groups' values."""
    loss_entropy = per_sample_bce(logit, y)
    outer = []
    inter_index = list(torch.unique(groups))
    lams = []
    for index in inter_index:
        ori = loss_entropy[groups == index]
        inner, lam = cvar(ori, inner_alpha)
        outer.append(inner)
        lams.append(lam)
    outer = torch.stack(outer)
    loss, lam_o = cvar(outer, outer_alpha)
    stats = {"daw_lambda_outer": lam_o, "daw_bce_mean": float(loss_entropy.mean()),
             "daw_group_max": float(outer.max()), "daw_group_min": float(outer.min()), "daw_n_groups": len(inter_index)}
    return loss, stats


def dag_fdd_loss(logit, y, alpha=0.5):
    """DAG-FDD, AIF detectors/dag_fdd.py:74-84 (default :76 dag_alpha 0.5): CVaR_alpha of the per-sample BCE."""
    loss_entropy = per_sample_bce(logit, y)
    loss, lam = cvar(loss_entropy, alpha)
    return loss, {"dag_lambda": lam, "dag_bce_mean": float(loss_entropy.mean())}


# ------------------------------------------------------------------------------------------------ PG-FDD
def ldam_loss(x, target, cls_num_list, max_m=0.5, s=30.0, weight=None):
    """FG loss/bal_loss.py:14-35 (LDAMLoss; .cuda FloatTensor -> device-agnostic). cls_num_list = the per-group
    training counts (fair_df_detector.py:136-140; the official FF++ value equals our FF++ train counts)."""
    m_list = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=np.float64)))
    m_list = m_list * (max_m / np.max(m_list))
    m_list = torch.as_tensor(m_list, dtype=torch.float32, device=x.device)
    x = x.float()
    index = torch.zeros_like(x, dtype=torch.uint8)
    index.scatter_(1, target.data.view(-1, 1), 1)
    index_float = index.float()
    batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
    batch_m = batch_m.view((-1, 1))
    x_m = x - batch_m
    output = torch.where(index.bool(), x_m, x)
    return F.cross_entropy(s * output, target, weight=weight)


def _swap_spe_features(type_list, value_list, rng):
    """FG loss/contrastive_regularization.py:10-35 (python random -> a seeded random.Random)."""
    type_list = type_list.cpu().numpy().tolist()
    index_list = list(range(len(type_list)))
    spe_dict = defaultdict(list)
    for i, one_type in enumerate(type_list):
        spe_dict[one_type].append(index_list[i])
    for keys in spe_dict.keys():
        rng.shuffle(spe_dict[keys])
    new_index_list = []
    for one_type in type_list:
        new_index_list.append(spe_dict[one_type].pop())
    return value_list[new_index_list]


def _triplet(anchor, positive, negative, margin):
    """FG contrastive_regularization.py:44-50."""
    dist_pos = F.pairwise_distance(anchor, positive)
    dist_neg = F.pairwise_distance(anchor, negative)
    return torch.mean(torch.clamp(dist_pos - dist_neg + margin, min=0.0))


def contrastive_loss(common, specific, spe_label, rng, margin=3.0):
    """FG contrastive_regularization.py:52-87 (margin 3.0 from fair_df_detector.py:134). Batch = [real half; fake
    half] of equal size, as the official pair loader delivers it."""
    common, specific = common.float(), specific.float()
    bs = common.shape[0]
    real_common, fake_common = common.chunk(2)
    idx_list = list(range(0, bs // 2))
    rng.shuffle(idx_list)
    real_common_anchor = common[idx_list]
    idx_list = list(range(bs // 2, bs))
    rng.shuffle(idx_list)
    fake_common_anchor = common[idx_list]
    specific_anchor = _swap_spe_features(spe_label, specific, rng)
    real_specific_anchor, fake_specific_anchor = specific_anchor.chunk(2)
    real_specific, fake_specific = specific.chunk(2)
    loss_realcommon = _triplet(real_common, real_common_anchor, fake_common_anchor, margin)
    loss_fakecommon = _triplet(fake_common, fake_common_anchor, real_common_anchor, margin)
    loss_realspecific = _triplet(real_specific, real_specific_anchor, fake_specific_anchor, margin)
    loss_fakespecific = _triplet(fake_specific, fake_specific_anchor, real_specific_anchor, margin)
    return loss_realcommon + loss_fakecommon + loss_fakespecific + loss_realspecific


def bilevel_fuse_loss(pred_fuse, label, groups, inner_alpha=0.9, outer_alpha=0.5, tau1=0.001, tau2=0.0001):
    """FG fair_df_detector.py:248-268: per-group CVaR_0.9 of the per-sample CE of the fused head (loss/bi_level_CE.py,
    CrossEntropyLoss(reduction='none')), then the smoothed CVaR_0.5 across groups (tau1 1e-3, tau2 1e-4)."""
    loss_fuse_entropy = F.cross_entropy(pred_fuse.float(), label, reduction="none")
    outer = []
    for index in list(torch.unique(groups)):
        ori = loss_fuse_entropy[groups == index]
        inner, _ = cvar(ori, inner_alpha)
        outer.append(inner)
    outer = torch.stack(outer)
    oc = outer.detach().float().cpu()
    lam = optimize.fminbound(search_func_smooth(oc, outer_alpha, tau1, tau2), np.min(oc.numpy()) - 1000.0,
                             np.max(oc.numpy()))
    loss = searched_lamda_loss_smooth(outer, float(lam), outer_alpha, tau1, tau2)
    return loss, {"fuse_lambda": float(lam), "fuse_group_max": float(outer.max()), "fuse_ce_mean": float(loss_fuse_entropy.mean())}


# ------------------------------------------------------------------------------------------------ Ding SFD / GDA
def snnl_per_channel(feat, labels, temperature=1.0, eps=1e-9):
    """Soft-nearest-neighbour loss of EVERY channel at once, DM SFD.py:15-58 (cos distance, T = 1, eps 1e-9, same-class
    mask INCLUDING the diagonal, all-class mask excluding it), applied per channel as SFD.py:148-160 does
    (split_feature.view(B, -1) = the channel's spatial map). feat: (B, C, P) channel maps; labels: (B,). Returns (C,)."""
    x = F.normalize(feat.float().transpose(0, 1), dim=2)                 # (C, B, P)
    dist = (1 - torch.bmm(x, x.transpose(1, 2))) / temperature          # (C, B, B)
    negexpd = torch.exp(-dist)
    b = labels.shape[0]
    pairs_y = labels.repeat(b, 1)
    mask = (pairs_y == pairs_y.T).float()
    dmask = torch.ones(b, b, device=feat.device) - torch.eye(b, device=feat.device)
    alcn = torch.sum(negexpd * dmask, dim=2)
    sacn = torch.sum(negexpd * mask, dim=2)
    return -torch.log((sacn + eps) / alcn).mean(1)


def select_sfd_channels(avg_snnl, prc=2.0):
    """DM SFD.py:166-188: n = max(1, round(prc% * C)); threshold = (n)-th order statistic (get_min_n, :84-87);
    channels with avg < threshold, sorted by value."""
    data = np.asarray(avg_snnl, dtype=np.float64)
    n = max(1, round(prc / 100.0 * data.shape[0]))
    tmp = data.copy()
    tmp.sort()
    threshold = tmp[n]
    x = np.arange(0, data.shape[0])
    ch = x[data < threshold]
    order = np.argsort(data[ch])
    return [int(c) for c in ch[order]], float(threshold), int(n)


_SINKHORN = {}


def sinkhorn(x, y, blur=5e-4, p=2):
    """DM GDA.py:191: geomloss SamplesLoss(loss='sinkhorn', p=2, blur=sinkhorn_blur) (geomloss 0.3.1, vendored in
    CLIPFAIR/third_party because the training env has no geomloss)."""
    key = (float(blur), int(p))
    if key not in _SINKHORN:
        import os
        import sys
        tp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
        if tp not in sys.path:
            sys.path.insert(0, tp)
        from geomloss import SamplesLoss
        _SINKHORN[key] = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    return _SINKHORN[key](x.float().contiguous(), y.float().contiguous())


def gda_fair_loss(logits, groups, gen, blur=5e-4, grad=False, n_groups=8):
    """DM GDA.py:218-235. One group index drawn uniformly (GDA.py:219, random.randint(0, 7)); fair = Sinkhorn between
    the main batch's 2-way logits and that group's logits. RELEASED code: computed inside torch.no_grad() (GDA.py:232-235)
    so it contributes NO gradient (agents/DINGREPRO_LIN/results/GDA_GRAD_CHECK.json); grad=True is the separately
    labelled gradient-enabled variant (not run unless the decision file is amended). Adaptation: the group's logits
    are the pool images of that group (no extra forward pass of a separate 64-image group batch)."""
    present = torch.unique(groups).tolist()
    gi = int(torch.randint(0, n_groups, (1,), generator=gen))
    if gi not in present:                       # cannot happen with the stratified batch (4 per label x group x cond)
        gi = present[int(torch.randint(0, len(present), (1,), generator=gen))]
    sel = groups == gi
    if grad:
        return sinkhorn(logits, logits[sel], blur), gi
    with torch.no_grad():
        return sinkhorn(logits.detach(), logits[sel].detach(), blur), gi
