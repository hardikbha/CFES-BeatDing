"""CFES training objectives. Every function returns (loss, stats). A term that has no eligible items in the
batch returns a zero that is still attached to the graph, so the total is always differentiable.

Notation: s = image logit (fake > 0), P = (N,G,G) patch evidence logits.
"""
import torch
import torch.nn.functional as F


def _zero(t):
    return t.sum() * 0.0


def detection_loss(score, y, mode="balanced"):
    """BCE. 'balanced' = 0.5 mean over reals + 0.5 mean over fakes (batches are 50/50 anyway; this keeps the
    weighting fixed when a pair or view batch is unbalanced)."""
    bce = F.binary_cross_entropy_with_logits(score, y.to(score.dtype), reduction="none")
    if mode == "plain":
        return bce.mean(), {}
    r, f = bce[y == 0], bce[y == 1]
    parts = [t.mean() for t in (r, f) if len(t)]
    return sum(parts) / len(parts), {}


def stability_loss(clean, view, kind, geom, cfg, background_kinds):
    """Counterfactual stability under label-preserving nuisance interventions.

    score   : Huber(s(x') - sg[s(x)])                                   all views
    map     : mean over face cells of (sigmoid P(x') - sg sigmoid P(x))^2  background views (face pixels intact)
    bg_ev   : BCE(P(x') on background cells, 0)                           background views
    The clean prediction is the anchor (stop-gradient), so the view is pulled to the clean prediction and the
    clean prediction is trained only by the detection loss."""
    s_term = F.huber_loss(view["score"], clean["score"].detach(), delta=cfg["huber_delta"])
    stats = {"stab_score": float(s_term)}
    total = s_term
    bgk = torch.isin(kind, torch.tensor(background_kinds, device=kind.device))
    if view["patch"] is not None and bgk.any():
        pv, pc = view["patch"][bgk], clean["patch"][bgk].detach()
        fc, bc = geom.face_cells, geom.bg_cells
        m_term = ((torch.sigmoid(pv) - torch.sigmoid(pc)) ** 2)[:, fc].mean() if fc.any() else _zero(pv)
        b = pv[:, bc]
        b_term = F.binary_cross_entropy_with_logits(b, torch.zeros_like(b)) if bc.any() else _zero(pv)
        total = total + cfg["map_weight"] * m_term + cfg["bg_evidence_weight"] * b_term
        stats.update(stab_map=float(m_term), stab_bg_ev=float(b_term))
    return total, stats


def sensitivity_loss(out_f, out_r, frac, aligned, geom, cfg):
    """Evidence sensitivity on (fake, real source) pairs that received the SAME intervention.

    score     : softplus(margin - (s_f - s_r))          the fake must score above its own source
    map (aligned pairs only, frac = per-cell share of face pixels the manipulation changed):
      changed : softplus(map_margin - (P_f - P_r))      evidence rises where the face was manipulated
      same    : Huber(P_f - P_r)                         and stays equal where the two images are identical,
                                                         which is what controls for intervention artefacts
    NOTE: the map terms are heuristic patch supervision. `frac` comes from raw pixel differences between the fake
    and its source frame, which are NOT validated manipulation masks (registration is only checked by the
    background-difference test, pairs.align_tau). They are active whenever this loss is on (every CFES phase-2 run
    so far), independently of loss.loc.weight. loss.sens.map_terms=false keeps only the score term.
    """
    d = out_f["score"] - out_r["score"]
    s_term = F.softplus(cfg["margin"] - d).mean()
    stats = {"sens_score": float(s_term), "sens_gap": float(d.mean())}
    total = s_term
    al = aligned.bool()
    if out_f["patch"] is not None and al.any() and cfg.get("map_terms", True):
        pf, pr, fr = out_f["patch"][al], out_r["patch"][al], frac[al]
        face = geom.face_cells.unsqueeze(0).expand_as(fr)
        changed = (fr >= cfg["changed_thr"]) & face
        same = (fr == 0) & face
        dp = pf - pr
        c_term = F.softplus(cfg["map_margin"] - dp[changed]).mean() if changed.any() else _zero(dp)
        u_term = F.huber_loss(dp[same], torch.zeros_like(dp[same])) if same.any() else _zero(dp)
        total = total + cfg["map_weight"] * c_term + cfg["unchanged_weight"] * u_term
        stats.update(sens_changed=float(c_term), sens_same=float(u_term),
                     sens_changed_cells=int(changed.sum()), sens_aligned_pairs=int(al.sum()))
    return total, stats


def localization_loss(p_fake, p_real, frac, aligned, cfg):
    """Optional direct evidence supervision (ELF L_loc): aligned fakes' changed cells -> 1, unchanged -> 0,
    source reals -> 0 everywhere; positives and negatives balanced within each image."""
    al = aligned.bool()
    if p_fake is None or not al.any():
        return _zero(p_fake if p_fake is not None else frac), {}
    tgt = (frac[al] >= cfg["changed_thr"]).float()
    p = torch.cat([p_fake[al], p_real[al]]).flatten(1)
    t = torch.cat([tgt, torch.zeros_like(tgt)]).flatten(1)
    bce = F.binary_cross_entropy_with_logits(p, t, reduction="none")
    pos, neg = t > 0.5, t < 0.5
    lp = (bce * pos).sum(1) / pos.sum(1).clamp(min=1)
    ln = (bce * neg).sum(1) / neg.sum(1).clamp(min=1)
    has = (pos.sum(1) > 0).float()
    loss = (0.5 * lp * has + (1 - 0.5 * has) * ln).mean()
    return loss, {"loc": float(loss)}


def alignment_loss(score, y, group, cfg):
    """Optional demographic alignment. Within each class, the soft positive rate sigmoid(s / T) is averaged per
    subgroup (groups with fewer than n_min items in the batch are skipped) and the variance of those group
    rates is penalised: soft FPR parity on reals, soft TPR parity on fakes. Uses demographic labels, so it is
    only enabled when the dataset's annotations are marked reliable."""
    rate = torch.sigmoid(score / cfg["temperature"])
    terms, used = [], 0
    for c in (0, 1):
        means = []
        for g in torch.unique(group[y == c]):
            sel = (y == c) & (group == g)
            if int(sel.sum()) >= cfg["n_min"]:
                means.append(rate[sel].mean())
        if len(means) >= 2:
            terms.append(torch.stack(means).var(unbiased=False))
            used += len(means)
    if not terms:
        return _zero(score), {"align_groups": 0}
    loss = sum(terms)
    return loss, {"align": float(loss), "align_groups": used}


def scale_loss(score, cfg):
    """Logit-scale control (option 2, 18 Oct): mean hinge max(0, |s| - cap) over the detection-labelled scores of the
    step. Keeps scores out of sigmoid saturation so that a soft-rate term (alignment) keeps a usable gradient and the
    0.5 operating point is set by unsaturated scores. Linear beyond the cap: constant per-item pull."""
    over = F.relu(score.float().abs() - cfg["cap"])
    return over.mean(), {"scale": float(over.mean()), "scale_frac_over_cap": float((over > 0).float().mean()),
                         "score_absmax": float(score.detach().abs().max())}


def dist_align_loss(score, y, group, cfg):
    """Class-conditional distribution alignment on standardised logits (review 3; a score-level ADAPTATION of Ding et
    al.'s global distribution alignment, not a reproduction of it).

    For each class c, scores are standardised with the class's pooled mean and std (stop-gradient):
    u = (s - sg mean_c) / sg std_c, so the loss is linear in the logit and cannot saturate (unlike sigmoid soft rates).
    Each group with >= n_min items of class c is pulled toward the pooled class-c distribution by the 1-D
    Wasserstein-1 distance between the two EMPIRICAL distributions, W1 = integral_0^1 |Q_g(t) - Q_pool(t)| dt, with the
    exact step quantile function Q(t) = x_(ceil(t n)) evaluated at `quantiles` grid midpoints (dist = "w1"; zero for
    identical empirical distributions whatever the sample sizes), or by matching the first two moments
    (dist = "moments"). The loss is the mean over (class, group) cells; real-class cells target
    false-positive parity, fake-class cells target true-positive parity."""
    s = score.float()
    K = int(cfg.get("quantiles", 32))
    t = (torch.arange(K, device=s.device, dtype=torch.float32) + 0.5) / K

    def qf(v):                                        # exact empirical (inverted-CDF) quantiles, differentiable
        v = torch.sort(v).values
        idx = (torch.ceil(t * v.numel()) - 1).long().clamp(0, v.numel() - 1)
        return v[idx]
    terms, used = [], 0
    for c in (0, 1):
        sc, gc = s[y == c], group[y == c]
        if sc.numel() < 2 * cfg["n_min"]:
            continue
        mu, sd = sc.mean().detach(), sc.std(unbiased=False).detach().clamp_min(1e-6)
        u = (sc - mu) / sd
        pooled_q = qf(u)
        for g in torch.unique(gc):
            ug = u[gc == g]
            if ug.numel() < cfg["n_min"]:
                continue
            if cfg.get("dist", "w1") == "w1":
                terms.append((qf(ug) - pooled_q).abs().mean())
            else:
                terms.append((ug.mean() - u.mean()) ** 2 + (ug.std(unbiased=False) - u.std(unbiased=False)) ** 2)
            used += 1
    if not terms:
        return _zero(score), {"dalign_cells": 0}
    loss = torch.stack(terms).mean()
    return loss, {"dalign": float(loss), "dalign_cells": used}


def real_group_risk_loss(score, y, group, cfg):
    """Review 4 item 4: real-image subgroup risk (a standard group-DRO-style objective; not claimed as novel).
    R_g = mean BCE(s, real) = mean softplus(s) over the REAL images of group g with >= n_min reals (ordinary + stratified
    batch and, when computed, their nuisance views). Group weights w_g = softmax_g(log R_g / tau), stop-gradient: with
    tau = 1, w_g is proportional to R_g, so the group with the highest real-image risk dominates whatever the absolute
    risk level (on training reals R_g can be 1e-5 or smaller). Loss = sum_g w_g R_g / sg(mean_g R_g): dividing by the
    stop-gradient mean risk gives the term an O(1) gradient on the riskiest reals. Detection supervision for both classes
    is kept separately."""
    s = score.float()
    risks, counts = [], []
    for g in torch.unique(group[y == 0]):
        sel = (y == 0) & (group == g)
        if int(sel.sum()) >= cfg["n_min"]:
            risks.append(F.softplus(s[sel]).mean())
            counts.append(int(sel.sum()))
    if len(risks) < 2:
        return _zero(score), {"risk_groups": 0}
    R = torch.stack(risks)
    if cfg.get("size_aware", False):
        # arm F (pre-registration 2026-10-22 C2): w_g proportional to R_g * (n_g / mean n)^size_power, n_g = the group's
        # real count in this risk pool (reals of the ordinary + stratified batch and, when included, their views).
        # The pre-registered formula w_g ~ R_g / sqrt(n_g / mean n) is size_power = -0.5.
        n = torch.tensor(counts, dtype=R.dtype, device=R.device)
        w = torch.softmax(torch.log(R.detach() + 1e-30) / cfg["tau"]
                          + float(cfg.get("size_power", -0.5)) * torch.log(n / n.mean()), 0)
    else:
        w = torch.softmax(torch.log(R.detach() + 1e-30) / cfg["tau"], 0)
    loss = (w * R).sum() / R.detach().mean().clamp_min(1e-30)
    return loss, {"risk": float(loss), "risk_max_group": float(R.max()), "risk_mean_group": float(R.mean()),
                  "risk_weight_max": float(w.max()), "risk_groups": len(risks)}


class RunningQuantile:
    """Running q-quantile of the most recent `window` values (no gradient). Arm E's threshold: values of the current
    step are appended first, then the quantile of the window is returned. state()/load() carry the window across a
    resume (train.py stores it in last.pt)."""

    def __init__(self, q, window):
        self.q, self.window, self.buf = float(q), int(window), None

    @torch.no_grad()
    def update(self, v):
        v = v.detach().float().flatten()
        self.buf = v if self.buf is None else torch.cat([self.buf.to(v.device), v])
        self.buf = self.buf[-self.window:]
        return torch.quantile(self.buf, self.q)

    def state(self):
        return [] if self.buf is None else self.buf.cpu().tolist()

    def load(self, vals):
        self.buf = torch.tensor(vals, dtype=torch.float32) if len(vals) else None


def fpr_gap_loss(score, y, group, cfg, rq):
    """Arm E (pre-registration 2026-10-22 B): soft group-FPR-gap penalty on REAL images (ordinary + stratified batch
    and, when included, their nuisance views). thr = running q-quantile of the real scores (rq, no gradient; q = 0.9
    gives ~10% soft FPR, a non-degenerate signal); soft FPR_g = mean_g sigmoid((s - thr) / tau);
    loss = sum_g |FPR_g - FPR_pooled| over groups with >= n_min reals (Ding et al.'s F_FPR form). FPR_pooled is the
    mean over all reals of the pool and keeps its gradient."""
    s = score.float()
    real = y == 0
    if int(real.sum()) < 2:
        return _zero(score), {"fprgap_groups": 0}
    thr = rq.update(s[real])
    soft = torch.sigmoid((s[real] - thr) / cfg["tau"])
    gr = group[real]
    pooled = soft.mean()
    gaps = []
    for g in torch.unique(gr):
        sel = gr == g
        if int(sel.sum()) >= cfg["n_min"]:
            gaps.append((soft[sel].mean() - pooled).abs())
    if len(gaps) < 2:
        return _zero(score), {"fprgap_groups": len(gaps)}
    loss = torch.stack(gaps).sum()
    return loss, {"fprgap": float(loss), "fprgap_thr": float(thr), "fprgap_pooled_softfpr": float(pooled),
                  "fprgap_max_abs": float(torch.stack(gaps).max()), "fprgap_groups": len(gaps)}


def reliability_supervision_loss(rel_logit, target):
    """C-FER R4 (decisions/2026-10-24-cfer-pilot-spec.md): cross-fitted reliability supervision, image level. BCE between
    the reliability head's r_global = sigmoid(rel_logit) and the soft correctness target t = 1 - |sigmoid(z_oof) - y|
    (a constant, i.e. detached) of the out-of-fold model (scripts/cfer_oof.py). Patch-level targets are not used: the
    registered-pair fraction of TTD (data/cfer_ttd/folds.json) is below 0.30, the declared fallback."""
    loss = F.binary_cross_entropy_with_logits(rel_logit.float(), target.float())
    with torch.no_grad():
        r = torch.sigmoid(rel_logit.float())
    return loss, {"relsup": float(loss), "relsup_target_mean": float(target.mean()), "relsup_r_mean": float(r.mean()),
                  "relsup_abs_err": float((r - target.float()).abs().mean())}
