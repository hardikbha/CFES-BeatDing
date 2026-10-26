"""CFES detector: frozen CLIP ViT-L/14 with trainable rank-1 residual adapters (FairReL/Effort SVD form, reused
from related_work/FairReL so the baseline is our existing B backbone), a global head on the CLS token, and a
patch-level evidence head on the 16x16 patch tokens.

Image score  s = s_cls + alpha * topk_mean(P)       (score_path = "fused"; alpha is learned)
             alpha_param = raw      : alpha is an unconstrained parameter, initialised at alpha_init (0 = the patch
                                      path starts with no influence). All CFES runs up to 16 Oct use this; their
                                      logged alpha stayed positive but nothing enforced it.
             alpha_param = softplus : alpha = softplus(beta) >= 0, beta stored as `alpha_raw`. alpha_init must be
                                      > 0. A raw-alpha checkpoint loads into it via alpha_raw = softplus^-1(alpha).
             alpha_param = sigmoid  : alpha = alpha_max * sigmoid(beta) in (0, alpha_max), smooth and bounded (CFES
                                      revision study). A raw-alpha checkpoint (alpha_1 > 0) is loaded EXACTLY: beta = 0
                                      (alpha = alpha_max / 2, the steepest point of the gate) and the patch head's last
                                      linear layer is multiplied by alpha_1 / (alpha_max / 2). topk_mean is positively
                                      homogeneous, so every score is unchanged; only the patch-logit scale changes.
             alpha_param = tanh     : alpha = alpha_max * tanh(beta) in (-alpha_max, alpha_max): smooth, bounded and
                                      signed (used for a phase-1 checkpoint whose alpha is negative, CLIP seed 27).
                                      Exact load: alpha = sign(alpha_1) * alpha_max / 2, last patch layer x |alpha_1| /
                                      (alpha_max / 2).
             s = s_cls                              (score_path = "cls";  the baseline)
             s = topk_mean(P)                       (score_path = "topk")
P is the (N,16,16) map of patch evidence logits; sigmoid(P) is the forensic evidence map.
"""
import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _fairrel(repo):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from fairrel.models.backbone import build_backbone
    from fairrel.models.svd_linear import total_orthogonality_loss
    return build_backbone, total_orthogonality_loss


def topk_mean(p, k):
    flat = p.flatten(1)
    return flat.topk(max(1, min(int(k), flat.shape[1])), dim=1).values.mean(1)


def inverse_softplus(a):
    a = float(a)
    if a <= 0:
        raise ValueError(f"alpha_param=softplus needs alpha > 0 (got {a})")
    return a + math.log(-math.expm1(-a))


def init_fusion_weight(module, m):
    """Creates module.alpha (raw) or module.alpha_raw (softplus); the unused one is None."""
    module.alpha_param = m.get("alpha_param", "raw")
    module.alpha, module.alpha_raw = None, None
    if module.score_path != "fused":
        return
    if module.alpha_param == "raw":
        module.alpha = nn.Parameter(torch.tensor(float(m["alpha_init"])))
    elif module.alpha_param == "softplus":
        module.alpha_raw = nn.Parameter(torch.tensor(inverse_softplus(m["alpha_init"])))
    elif module.alpha_param == "tanh":
        module.alpha_max = float(m.get("alpha_max", 1.0))
        a0 = float(m["alpha_init"])
        if not (0 < abs(a0) < module.alpha_max):
            raise ValueError(f"alpha_param=tanh needs 0 < |alpha_init| < alpha_max (got {a0}, {module.alpha_max})")
        module.alpha_raw = nn.Parameter(torch.tensor(math.atanh(a0 / module.alpha_max)))
    elif module.alpha_param == "sigmoid":
        module.alpha_max = float(m.get("alpha_max", 1.0))
        a0 = float(m["alpha_init"])
        if not 0 < a0 < module.alpha_max:
            raise ValueError(f"alpha_param=sigmoid needs 0 < alpha_init < alpha_max (got {a0}, {module.alpha_max})")
        module.alpha_raw = nn.Parameter(torch.tensor(math.log(a0 / (module.alpha_max - a0))))
    else:
        raise ValueError(f"model.alpha_param must be raw or softplus, got {module.alpha_param}")


def fusion_weight(module):
    """The effective alpha (a tensor), or None when the score path is not fused."""
    if module.alpha_raw is not None:
        if module.alpha_param == "sigmoid":
            return module.alpha_max * torch.sigmoid(module.alpha_raw)
        if module.alpha_param == "tanh":
            return module.alpha_max * torch.tanh(module.alpha_raw)
        return F.softplus(module.alpha_raw)
    return module.alpha


def last_patch_linear(module):
    """Name prefix of the patch head's final nn.Linear (e.g. 'patch_head.3')."""
    names = [n for n, m in module.patch_head.named_modules() if isinstance(m, nn.Linear)]
    return "patch_head" + ("." + names[-1] if names[-1] else "")


def convert_alpha_state(sd, module):
    """Loading a raw-alpha checkpoint into a constrained model.
    softplus: alpha_raw = softplus^-1(alpha). sigmoid: exact rescaling (see module docstring). Raises if alpha <= 0.
    Records what was done in module.alpha_conversion."""
    if getattr(module, "alpha_raw", None) is None or "alpha" not in sd or "alpha_raw" in sd:
        return sd
    sd = dict(sd)
    a1 = float(sd.pop("alpha"))
    if module.alpha_param == "softplus":
        sd["alpha_raw"] = torch.tensor(inverse_softplus(a1))
        module.alpha_conversion = {"rule": "softplus_inverse", "alpha_ckpt": a1}
    elif module.alpha_param == "tanh":
        if a1 == 0:
            raise ValueError("cannot convert alpha = 0 exactly into the tanh gate")
        c = module.alpha_max / 2
        lin = last_patch_linear(module)
        for k in (lin + ".weight", lin + ".bias"):
            sd[k] = sd[k] * (abs(a1) / c)
        sd["alpha_raw"] = torch.tensor(math.atanh(math.copysign(0.5, a1)))
        module.alpha_conversion = {"rule": "tanh_rescale_patch_head", "alpha_ckpt": a1,
                                   "alpha_start": math.copysign(c, a1), "patch_scale": abs(a1) / c, "layer": lin}
    elif module.alpha_param == "sigmoid":
        if a1 <= 0:
            raise ValueError(f"cannot convert alpha={a1} <= 0 exactly into the sigmoid gate")
        c = module.alpha_max / 2
        lin = last_patch_linear(module)
        for k in (lin + ".weight", lin + ".bias"):
            sd[k] = sd[k] * (a1 / c)
        sd["alpha_raw"] = torch.tensor(0.0)
        module.alpha_conversion = {"rule": "sigmoid_rescale_patch_head", "alpha_ckpt": a1, "alpha_start": c,
                                   "patch_scale": a1 / c, "layer": lin}
    return sd


class CFESDetector(nn.Module):
    def __init__(self, cfg, fairrel_repo):
        super().__init__()
        m = cfg
        build_backbone, self._orth = _fairrel(fairrel_repo)
        self.backbone = build_backbone(m["backbone"], m["pretrained"], m["rank"], m["pooling"])
        d = self.backbone.feat_dim
        self.grid = self.backbone.grid_size[0]
        self.cls_head = nn.Linear(d, 1)
        ph = m["patch_head"]
        self.patch_head = None
        if ph["enabled"]:
            h = ph["hidden"]
            self.patch_head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
                               if h else nn.Linear(d, 1))
        self.score_path = m["score_path"]
        # review 4 stabilisation candidate: the patch head reads stop-gradient backbone features, so no loss can grow the
        # patch logits through the backbone adapters (the forward pass, and hence every starting score, is unchanged)
        self.patch_detach = bool(m.get("patch_detach", False))
        # review 4 stabilisation (chosen from the training diagnosis): normalise the LOCAL contribution. With
        # t = topk_mean(P), the fused score uses  alpha * t * min(1, c0 / r)  where r is a running RMS of t over training
        # batches (stop-gradient buffer, momentum `local_norm_momentum`) and c0 is r on the first phase-2 batch. The
        # running RMS of the local contribution therefore cannot exceed alpha * c0 (its phase-1 level): inflating P gains
        # nothing. Before the first training step the factor is 1, so every starting score is unchanged; at inference
        # the stored buffers are used, so the score of an image does not depend on the batch.
        self.local_norm = bool(m.get("local_norm", False))
        self.local_norm_momentum = float(m.get("local_norm_momentum", 0.01))
        self.register_buffer("loc_c0", torch.tensor(float("nan")))
        self.register_buffer("loc_r", torch.tensor(float("nan")))
        if self.score_path != "cls" and self.patch_head is None:
            raise ValueError("score_path topk/fused needs model.patch_head.enabled=true")
        self.k = m["topk"]
        init_fusion_weight(self, m)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        # C-FER R3/R4 (decisions/2026-10-24-cfer-pilot-spec.md): reliability-aware fusion, OFF by default. When off,
        # nothing below is created and forward() runs the unchanged code path: same modules, parameters, state_dict
        # keys and RNG draws as before (bit-identity check: results/ttd/2026-10-24-cfer-build.md).
        rel = m.get("reliability") or {}
        self.rel_enabled = bool(rel.get("enabled", False))
        self.rel_head = None
        if self.rel_enabled:
            if self.score_path != "fused" or self.patch_head is None:
                raise ValueError("model.reliability needs score_path=fused and an enabled patch head")
            if not rel.get("gate") or len(rel["gate"]) != 2:        # bounds come from the config, no default
                raise ValueError("model.reliability.gate must be [lo, hi] (C-FER: chosen on TTD-val from "
                                 "{[0.10, 0.90], [0.25, 0.75]})")
            lo, hi = (float(v) for v in rel["gate"])
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(f"model.reliability.gate must satisfy 0 <= lo < hi <= 1, got {rel['gate']}")
            self.rel_gate = (lo, hi)
            self.rel_view = dict(rel.get("view") or {"kind": "bg_blur", "sigma": 4.0})
            if self.rel_view.get("kind") != "bg_blur":
                raise ValueError(f"model.reliability.view.kind must be bg_blur, got {self.rel_view.get('kind')}")
            self.rel_face, self._rel_blend = None, {}     # face template from cf.face_mask (engine.build_model)
            with torch.random.fork_rng(devices=[]):        # own seed: the global RNG stream is the same as R2's
                torch.manual_seed(int(rel.get("init_seed", 0)))
                self.rel_head = ReliabilityHead(d, int(rel.get("hidden", 256)))
            self._gate_mean = None

    def set_reliability_face(self, face_mask):
        self.rel_face = dict(face_mask)

    @torch.no_grad()
    def reliability_view(self, x01):
        """The DETERMINISTIC evaluation view (spec: background blur, rho = sigma = 4 px at 224, face kept): the bg_blur
        intervention of counterfactuals.py with the cf face template, so face-core pixels are untouched."""
        from .counterfactuals import FaceGeometry, gaussian_blur
        if self.rel_face is None:
            raise RuntimeError("reliability view needs the face template (engine.build_model sets it from cf.face_mask)")
        size = x01.shape[-1]
        key = (size, str(x01.device))
        if key not in self._rel_blend:
            f = self.rel_face
            self._rel_blend[key] = FaceGeometry(size, self.grid, f["cx"], f["cy"], f["ax"], f["ay"], f.get("dilate", 1.15),
                                                x01.device).blend
        a = self._rel_blend[key].to(x01.dtype)
        return a * x01 + (1 - a) * gaussian_blur(x01, float(self.rel_view["sigma"]))

    @torch.no_grad()
    def view_logit(self, x01):
        """z_view: the model's own global logit on the deterministic view (no gradient)."""
        _, zv = self.backbone((self.reliability_view(x01) - self.mean) / self.std)
        return self.cls_head(zv).squeeze(-1).float()

    def forward_parts(self, x01):
        """Backbone + both heads, no fusion: pooled feature z, patch features, z_global (= s_cls) and z_patches (= P)."""
        feat, z = self.backbone((x01 - self.mean) / self.std)
        s_cls = self.cls_head(z).squeeze(-1).float()
        patch = self.patch_head(feat.detach() if self.patch_detach else feat).squeeze(-1).float()
        return dict(feat=feat, z=z, s_cls=s_cls, patch=patch)

    def fuse(self, parts, z_view):
        """C-FER R3 fusion (spec 2026-10-24):
          inst      = |sg(z_global) - sg(z_view)|
          r_global, r_patch (logits) = ReliabilityHead(pooled feature, patch features, inst, z_global, z_patches)
          alpha     = lo + (hi - lo) * sigmoid(a(r_global)),  a = learned scalar affine map (init: identity)
          t         = sum over the top-8 cells of z_patches of softmax(r_patch over those 8 cells) * z_patches
          t        <- CFES-D's local normalisation (running-RMS cap, model.local_norm) applied to the pooled t
          z_fused   = alpha * z_global + (1 - alpha) * t
        Zero-initialised output layers: r_global logit = 0 and r_patch uniform at the start, so alpha = (lo + hi) / 2
        and t = topk_mean(P)."""
        s_cls, patch, feat, z = parts["s_cls"], parts["patch"], parts["feat"], parts["z"]
        inst = (s_cls.detach() - z_view.detach().float()).abs()
        rg, rp = self.rel_head(z, feat, inst, s_cls, patch)
        lo, hi = self.rel_gate
        alpha = lo + (hi - lo) * torch.sigmoid(self.rel_head.gate_w * rg + self.rel_head.gate_b)
        flat = patch.flatten(1)
        top, idx = flat.topk(max(1, min(int(self.k), flat.shape[1])), dim=1)
        w = torch.softmax(rp.flatten(1).gather(1, idx), dim=1)
        t = (w * top).sum(1)
        if self.local_norm:
            if self.training:
                with torch.no_grad():
                    rb = t.detach().float().pow(2).mean().sqrt()
                    if torch.isnan(self.loc_c0):
                        self.loc_c0.fill_(float(rb)); self.loc_r.fill_(float(rb))
                    else:
                        self.loc_r.mul_(1 - self.local_norm_momentum).add_(self.local_norm_momentum * rb)
            if not torch.isnan(self.loc_c0):
                t = t * torch.clamp(self.loc_c0 / self.loc_r, max=1.0)
        score = alpha * s_cls + (1 - alpha) * t
        self._gate_mean = alpha.detach().mean()
        return dict(score=score, s_cls=s_cls, patch=patch, rel_global=rg, rel_patch=rp, gate=alpha, inst=inst, t=t)

    def forward(self, x01, z_view=None):
        if self.rel_enabled:
            # z_view given (training: the logit of the stability view drawn for that image, engine.compute_losses);
            # otherwise the deterministic evaluation view (one extra no-grad backbone pass: 2x inference cost)
            parts = self.forward_parts(x01)
            return self.fuse(parts, self.view_logit(x01) if z_view is None else z_view)
        feat, z = self.backbone((x01 - self.mean) / self.std)          # (N,G,G,D), (N,D)
        s_cls = self.cls_head(z).squeeze(-1).float()
        if self.patch_head is not None:
            patch = self.patch_head(feat.detach() if self.patch_detach else feat).squeeze(-1).float()
        else:
            patch = None
        if self.score_path == "cls":
            score = s_cls
        elif self.score_path == "topk":
            score = topk_mean(patch, self.k)
        else:
            t = topk_mean(patch, self.k)
            if self.local_norm:
                if self.training:
                    with torch.no_grad():
                        rb = t.detach().float().pow(2).mean().sqrt()
                        if torch.isnan(self.loc_c0):
                            self.loc_c0.fill_(float(rb)); self.loc_r.fill_(float(rb))
                        else:
                            self.loc_r.mul_(1 - self.local_norm_momentum).add_(self.local_norm_momentum * rb)
                if not torch.isnan(self.loc_c0):
                    t = t * torch.clamp(self.loc_c0 / self.loc_r, max=1.0)
            score = s_cls + fusion_weight(self).float() * t
        return dict(score=score, s_cls=s_cls, patch=patch)

    def fusion_alpha(self):
        if self.rel_enabled:            # per-image gate: the mean of the last forward pass (logged as "alpha")
            return self._gate_mean
        return fusion_weight(self)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints written before the local_norm buffers existed: treat them as "not yet set" (factor 1, i.e. the
        # original model), so every earlier checkpoint loads strictly and scores exactly as before
        for k in ("loc_c0", "loc_r"):
            state_dict.setdefault(prefix + k, torch.tensor(float("nan")))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def orthogonality_loss(self):
        return self._orth(self.backbone)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """Only the parameters that training changes (adapters + heads); the frozen part is rebuilt from the
        OpenAI weights at load time."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    def parameter_report(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parts = {"adapters": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
                 "cls_head": sum(p.numel() for p in self.cls_head.parameters()),
                 "patch_head": sum(p.numel() for p in self.patch_head.parameters()) if self.patch_head else 0}
        if self.rel_head is not None:
            parts["rel_head"] = sum(p.numel() for p in self.rel_head.parameters())
        return dict(total=tot, trainable=tr, **parts)


class ReliabilityHead(nn.Module):
    """C-FER R3 reliability head (spec 2026-10-24). Inputs: pooled backbone feature z (N,D), patch features (N,G,G,D),
    instability inst = |sg(z_global) - sg(z_view)| (N,), z_global (N,) and z_patches (N,G,G). Logits enter through
    asinh (sign-preserving log squash; FF++ logits reach |z| ~ 20), the instability through log1p.
        ctx      = A(LN z) + B(LN mean_cells feat) + C([log1p inst, asinh z_global])
        r_global = out_g(GELU(ctx))                                   (logit; r = sigmoid)
        r_patch  = out_p(GELU(Pp(LN feat) + Q(asinh z_patch) + ctx))   (per-cell logit)
    gate_w, gate_b: the scalar affine map a(.) of the fusion gate, alpha = lo + (hi - lo) sigmoid(gate_w r_global + gate_b),
    initialised to the identity (1, 0). out_g / out_p start at zero (r_global logit 0, uniform r_patch). Demographic
    and domain labels are never inputs."""

    def __init__(self, d, h):
        super().__init__()
        self.ln_z, self.ln_m, self.ln_p = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.a, self.b, self.c, self.pp, self.q = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(2, h), nn.Linear(d, h), nn.Linear(1, h)
        self.out_g, self.out_p = nn.Linear(h, 1), nn.Linear(h, 1)
        for lin in (self.out_g, self.out_p):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.gate_w = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z, feat, inst, z_global, z_patch):
        sc = torch.stack([torch.log1p(inst.float()), torch.asinh(z_global.float())], 1).to(z.dtype)
        ctx = self.a(self.ln_z(z)) + self.b(self.ln_m(feat.mean((1, 2)))) + self.c(sc)
        rg = self.out_g(F.gelu(ctx)).squeeze(-1).float()
        cell = self.pp(self.ln_p(feat)) + self.q(torch.asinh(z_patch.float()).to(feat.dtype)[..., None])
        rp = self.out_p(F.gelu(cell + ctx[:, None, None, :])).squeeze(-1).float()
        return rg, rp
