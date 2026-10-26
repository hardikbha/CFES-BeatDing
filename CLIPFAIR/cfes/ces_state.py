"""Cross-step CES state (no gradient into the detector): cue baselines, demographic subspaces, online group probe.

  class means   mu[y,k]      EMA over training images of class y of the (detached) cue c_k        (2,K,d)
  group means   mu[y,g,k]    EMA over images of class y and group g                              (2,G,K,d)
                EMA: first observation initialises; afterwards mu <- (1 - m) mu + m * batch mean  (m = ces.ema)
  subspace      U_k (d,r)    top-r right singular vectors of the stacked between-group differences within a class
                             { mu[y,g,k] - mean_{g' init} mu[y,g',k] : y in {0,1}, g initialised },
                             recomputed every `subspace_every` steps; usable only when >= 2 groups are initialised in
                             some class and the stack has rank >= r. P_k = U_k U_k^T.
  probe         q(g | c)     multinomial logistic regression on the concatenated detached cues [c_1..c_K] (K*d -> G),
                             one Adam step per training step on the step's images (its own optimiser, lr probe_lr).
The probe and the EMA statistics never receive gradient from the detector loss. state_dict() carries everything
(including the probe optimiser) for resume; train_ces.py stores it in the checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CESState(nn.Module):
    def __init__(self, K, d, n_groups, cfg):
        super().__init__()
        c = cfg.get("ces", {})
        inv = cfg.get("loss", {}).get("inv", {})
        self.K, self.d, self.G = K, d, n_groups
        self.m = float(c.get("ema", 0.02))
        self.r = int(inv.get("rank", 2))
        self.subspace_every = int(inv.get("subspace_every", 1))
        self.register_buffer("cls_mean", torch.zeros(2, K, d))
        self.register_buffer("cls_init", torch.zeros(2, dtype=torch.bool))
        self.register_buffer("grp_mean", torch.zeros(2, n_groups, K, d))
        self.register_buffer("grp_init", torch.zeros(2, n_groups, dtype=torch.bool))
        self.register_buffer("U", torch.zeros(K, d, self.r))
        self.register_buffer("U_ok", torch.zeros(K, dtype=torch.bool))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
        self.probe = nn.Linear(K * d, n_groups)
        self.probe_opt = torch.optim.Adam(self.probe.parameters(), lr=float(inv.get("probe_lr", 1e-3)))

    # ---- EMA statistics -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _ema(self, mean, init, idx, value):
        if bool(init[idx]):
            mean[idx].mul_(1 - self.m).add_(self.m * value)
        else:
            mean[idx].copy_(value)
            init[idx] = True

    @torch.no_grad()
    def update(self, C, y, g):
        """C (N,K,d) detached cues of clean labelled images, y (N,), g (N,)."""
        C = C.detach().float()
        for c in (0, 1):
            sel = y == c
            if sel.any():
                self._ema(self.cls_mean, self.cls_init, c, C[sel].mean(0))
                for gg in torch.unique(g[sel]).tolist():
                    s2 = sel & (g == gg)
                    self._ema(self.grp_mean, self.grp_init, (c, int(gg)), C[s2].mean(0))
        self.steps += 1
        if int(self.steps) % self.subspace_every == 0:
            self.recompute_subspace()

    @torch.no_grad()
    def recompute_subspace(self):
        for k in range(self.K):
            rows = []
            for c in (0, 1):
                ok = self.grp_init[c]
                if int(ok.sum()) >= 2:
                    M = self.grp_mean[c, ok, k]                       # (G_ok, d)
                    rows.append(M - M.mean(0, keepdim=True))
            if not rows:
                self.U_ok[k] = False
                continue
            A = torch.cat(rows)
            _, S, Vh = torch.linalg.svd(A, full_matrices=False)
            rank = int((S > 1e-6 * max(float(S[0]), 1e-12)).sum()) if S.numel() else 0
            if rank < self.r:
                self.U_ok[k] = False
                continue
            self.U[k] = Vh[:self.r].T
            self.U_ok[k] = True

    def baseline(self, y, g=None, fake_mode="group"):
        """(N,K,d) removal baseline from the EMA means (detached buffers; no gradient).
        reals: EMA mean of the FAKE class's cues.
        fakes: fake_mode 'group' (user decision 25 Oct, Amendment 1) = EMA mean of the REAL cues of the fake's OWN
               group, mu[real, g]; falls back to the real class mean when that group has no estimate yet (or g is
               None / outside 0..G-1). fake_mode 'class' = the real class mean (the original rule)."""
        y = y.long()
        R = self.cls_mean[1 - y].clone()
        if fake_mode not in ("group", "class"):
            raise ValueError(f"ces.fake_baseline must be group | class, got {fake_mode}")
        if fake_mode == "group" and g is not None:
            g = g.long().to(y.device)
            inr = (g >= 0) & (g < self.G)
            gi = g.clamp(0, self.G - 1)
            use = (y == 1) & inr & self.grp_init[0, gi]
            if use.any():
                R[use] = self.grp_mean[0, gi[use]]
        return R

    def ready(self):
        return bool(self.cls_init.all())

    # ---- online probe ---------------------------------------------------------------------------------------------
    def probe_logits(self, C):
        with torch.no_grad():
            return self.probe(C.detach().float().flatten(1))

    def probe_step(self, C, g):
        """One Adam step of cross-entropy on detached cues (enable_grad, so it also works inside no_grad callers)."""
        with torch.enable_grad():
            with torch.autocast(device_type=C.device.type, enabled=False):
                loss = F.cross_entropy(self.probe(C.detach().float().flatten(1)), g.long())
            self.probe_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.probe_opt.step()
        return float(loss)

    def full_state(self):
        return {"buffers": self.state_dict(), "probe_opt": self.probe_opt.state_dict()}

    def load_full_state(self, st):
        self.load_state_dict(st["buffers"])
        if st.get("probe_opt"):
            self.probe_opt.load_state_dict(st["probe_opt"])
