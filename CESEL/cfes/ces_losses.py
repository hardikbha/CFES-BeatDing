"""CES objectives on cue tensors. Every loss returns (loss, stats); a term with no eligible item returns a zero that
is still attached to the graph.

Notation: C (N,K,d) cues, s (N,) score = H(C), y in {0,1} (1 = fake), y' = 2y - 1, g group (0..G-1), m manipulation
condition (0 = real, 1.. = manipulation family), sg = stop-gradient, K = 4 cues (tex, freq, bound, sem).

Margin            m(s, y) = y' * kappa * asinh(s / kappa)          (ces.margin: asinh | linear -> y' s; kappa = 1)
Cue removal       R (N,K,d) = replacement cues, always detached (ces.replace):
                    fake with its paired real source in the same forward (pair chunk): R_k = c_k(source)
                    other fake : R_k = mu[real, g, k], the EMA real-cue mean of the fake's OWN group (ces.fake_baseline
                                 group, default since Amendment 1); mu[real, k] (class mean) while that group has
                                 no estimate, or with ces.fake_baseline class
                    real       : R_k = mu[fake, k]
                    control "random": R_k = c_k(j), j a uniformly random OTHER image of the step (any label)
Removed score     s^(-k) = H(C with row k replaced by R_k)
                    additive head: s^(-k) = s - w_k^T c_k + w_k^T R_k  (closed form, exact)
Utility           u_{i,k} = m(s_i, y_i) - m(s_i^(-k), y_i)
L_sens  = sum_{k validated} mean_i relu(tau_k - u_{i,k})
L_floor = sum_k sum_{(g,c,m): n_{g,c,m} >= n_min} relu( sg(ubar_{k,c,m}) - ubar_{g,k,c,m} - delta )^2
          ubar_{k,c,m} = mean of u_{.,k} over every image of class c and condition m (all groups pooled),
          ubar_{g,k,c,m} = mean over the cell's images; k over validated cues (floor.cues: validated | all);
          floor.reduce: sum (default, as specified) | mean over cells; control floor.shuffle_groups permutes g.
L_inv (feature space): partner j(i) = random image of the step with y_j = y_i and g_j != g_i;
          c'_{i,k} = c_{i,k} - P_k c_{i,k} + P_k sg(c_{j,k})   (P_k = U_k U_k^T, cues with a usable subspace; others kept)
          s'_i = H(C'_i);  valid_i = q(g_j | C'_i) > q(g_i | C'_i)  (online probe, no grad)
          L_inv = mean_{i valid} Huber_delta( asinh s_i , asinh s'_i )   (inv.stopgrad: none | clean | cf)
L_inv (matched, inv.source = matched): per matched cross-group set with >= 2 valid groups, for fakes and for reals,
          mean_g Huber( asinh s_{set,g}, mean_g' asinh s_{set,g'} ); mean over (set, role) units.
"""
import torch
import torch.nn.functional as F


def _zero(t):
    return t.sum() * 0.0


def margin(s, y, cfg):
    yp = 2.0 * y.float() - 1.0
    if cfg.get("margin", "asinh") == "linear":
        return yp * s.float()
    kap = float(cfg.get("kappa", 1.0))
    return yp * kap * torch.asinh(s.float() / kap)


def random_other(n, gen, device):
    """Index j(i) != i, uniform over the other n - 1 images (n >= 2)."""
    off = torch.randint(1, n, (n,), generator=gen).to(device)
    return (torch.arange(n, device=device) + off) % n


def replacement(C, y, state, pair_src=None, mode="counterfactual", gen=None, g=None, fake_baseline="group"):
    """(N,K,d) detached replacement cues (see module docstring). pair_src (N,) long: row index of the image's paired
    real source in C, or -1. g (N,) groups: with fake_baseline 'group' an UNPAIRED fake uses the EMA real-cue mean of
    its own group (class mean while that group has no estimate); g None -> class mean."""
    if mode == "random":
        return C[random_other(C.shape[0], gen, C.device)].detach()
    if mode != "counterfactual":
        raise ValueError(f"ces.replace must be counterfactual | random, got {mode}")
    R = state.baseline(y, g, fake_baseline).to(C.dtype).clone()
    if pair_src is not None:
        has = pair_src >= 0
        if has.any():
            R[has] = C[pair_src[has]].detach()
    return R.detach()


def removed_scores(model, C, R, s, path="auto"):
    """(N,K) s^(-k). path: auto (closed form for the additive head), closed, general (one head pass per cue)."""
    head = model.head
    if path == "closed" or (path == "auto" and head.kind == "additive"):
        if head.kind != "additive":
            raise ValueError("closed-form removal needs the additive head")
        return s.float()[:, None] - head.per_cue(C) + head.per_cue(R)
    out = []
    for k in range(C.shape[1]):
        sel = torch.zeros(1, C.shape[1], 1, dtype=torch.bool, device=C.device)
        sel[0, k] = True
        out.append(model.score_from(torch.where(sel, R, C)))
    return torch.stack(out, 1)


def utilities(model, C, s, y, R, cfg, path="auto"):
    """u (N,K) = m(s, y) - m(s^(-k), y), and s^(-k)."""
    s_minus = removed_scores(model, C, R, s, path)
    u = margin(s, y, cfg)[:, None] - margin(s_minus, y[:, None].expand_as(s_minus), cfg)
    return u, s_minus


def sens_loss(u, valid, target):
    """valid (K,) bool, target (K,) float; or per image (N,K) / (N,K) (per-family validation, Amendment 2):
    loss = sum_k mean_i valid_ik relu(tau_ik - u_ik), identical to the (K,) form when every row is the same."""
    valid = valid.to(u.device)
    if valid.dim() == 2:
        if not valid.any():
            return _zero(u), {"sens_ces_cues": 0}
        viol = F.relu(target.to(u.device) - u) * valid.to(u.dtype)
        return viol.mean(0).sum(), {"sens_ces_cues": int(valid.any(0).sum()),
                                    "sens_ces_viol_frac": float((viol > 0).sum() / valid.sum().clamp(min=1))}
    if not valid.any():
        return _zero(u), {"sens_ces_cues": 0}
    viol = F.relu(target.to(u.device)[None, :] - u)                       # (N,K)
    per = viol.mean(0)
    loss = per[valid].sum()
    st = {"sens_ces_cues": int(valid.sum()), "sens_ces_viol_frac": float((viol[:, valid] > 0).float().mean())}
    return loss, st


def floor_loss(u, y, g, cond, cue_mask, cfg, gen=None):
    n_min, delta = int(cfg.get("n_min", 4)), float(cfg.get("delta", 0.1))
    cue_mask = cue_mask.to(u.device)
    if cfg.get("shuffle_groups", False):                    # control: group labels permuted within the step
        g = g[torch.randperm(g.shape[0], generator=gen).to(g.device)]
    terms, worst = [], 0.0
    for c in torch.unique(y).tolist():
        for mm in torch.unique(cond[y == c]).tolist():
            scm = (y == c) & (cond == mm)
            pooled = u[scm].mean(0).detach()                                  # (K,)
            for gg in torch.unique(g[scm]).tolist():
                sel = scm & (g == gg)
                if int(sel.sum()) < n_min:
                    continue
                d = F.relu(pooled - u[sel].mean(0) - delta) ** 2
                terms.append(d[cue_mask].sum())
                worst = max(worst, float(d[cue_mask].max()) if cue_mask.any() else 0.0)
    if not terms or not cue_mask.any():
        return _zero(u), {"floor_cells": 0}
    T = torch.stack(terms)
    loss = T.sum() if cfg.get("reduce", "sum") == "sum" else T.mean()
    return loss, {"floor_cells": len(terms), "floor_active_cells": int((T > 0).sum()), "floor_worst": worst}


def demographic_partner(y, g, gen):
    """partner (N,) long and has (N,) bool: a uniformly random image with the same label and a different group."""
    n = y.shape[0]
    M = (y[:, None] == y[None, :]) & (g[:, None] != g[None, :])
    r = torch.rand(n, n, generator=gen).to(y.device).masked_fill(~M, -1.0)
    return r.argmax(1), M.any(1)


def demographic_swap(C, state, partner, has, detach_donor=True):
    """C' = C - P C + P C[partner] on the cues whose subspace is usable; rows without a partner are unchanged."""
    U = state.U.to(C.dtype)                                                  # (K,d,r)
    P = U @ U.transpose(1, 2)                                                # (K,d,d)
    ok = state.U_ok.to(C.device)
    donor = C[partner].detach() if detach_donor else C[partner]
    delta = torch.einsum("nkd,kde->nke", donor - C, P)
    delta = delta * ok[None, :, None].to(C.dtype) * has[:, None, None].to(C.dtype)
    return C + delta


@torch.no_grad()
def swap_validity(state, C_swap, g, partner, has):
    lp = state.probe_logits(C_swap)
    i = torch.arange(g.shape[0], device=g.device)
    return has & (lp[i, g[partner]] > lp[i, g])


def inv_loss(s, s_swap, valid, cfg):
    a, b = torch.asinh(s.float()), torch.asinh(s_swap.float())
    sg = cfg.get("stopgrad", "none")
    if sg == "clean":
        a = a.detach()
    elif sg == "cf":
        b = b.detach()
    elif sg != "none":
        raise ValueError(f"loss.inv.stopgrad must be none | clean | cf, got {sg}")
    if not valid.any():
        return _zero(s_swap), {"inv_items": 0}
    h = F.huber_loss(a, b, delta=float(cfg.get("huber_delta", 1.0)), reduction="none")
    return h[valid].mean(), {"inv_items": int(valid.sum()), "inv_abs_dasinh": float((a - b).abs()[valid].mean())}


def matched_inv_loss(sf, sr, mask, cfg):
    """sf, sr (S,Kmax) fake / real scores of matched cross-group sets, mask (S,Kmax) valid slots."""
    delta = float(cfg.get("huber_delta", 1.0))
    terms = []
    for sc in (sf, sr):
        a = torch.asinh(sc.float())
        n = mask.sum(1)
        use = n >= 2
        if not use.any():
            continue
        mu = (a * mask).sum(1, keepdim=True) / n.clamp(min=1)[:, None]
        h = F.huber_loss(a, mu.expand_as(a), delta=delta, reduction="none") * mask
        terms.append((h.sum(1) / n.clamp(min=1))[use])
    if not terms:
        return _zero(sf), {"inv_matched_units": 0}
    t = torch.cat(terms)
    return t.mean(), {"inv_matched_units": int(t.numel())}
