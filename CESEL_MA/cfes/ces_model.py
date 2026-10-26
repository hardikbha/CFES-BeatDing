"""CES detector: four support-defined cue branches on a swappable backbone, and a cue head.

Notation: N images, K = 4 cues in the fixed order CUES = (tex, freq, bound, sem), d = cue dimension (default 128).
feat (N,G,G,D), z (N,D) = backbone(x01) (ces_backbones.py); masks from ces_geometry.CueGeometry.

  texture   c_tex   = MLP_T( Pool_T( LN(feat[:, core cells]) ) )      interior face-core cells
  boundary  c_bound = MLP_B( Pool_B( LN(feat[:, ring cells]) ) )      ring cells at the face border
  semantic  c_sem   = MLP_S( LN(z) )                                   CLS token (CNN backends: avg-pooled feature)
  frequency c_freq  = FreqNet( log(1 + |FFTshift(FFT2_ortho(crop(x01)))|) )  separate pixel-space CNN, no backbone
  Pool = attention pooling a = softmax_cells(q^T h_c), sum_c a_c h_c (ces.pool: attn | mean);
  MLP = Linear(D, h) - GELU - Linear(h, d); every cue then passes a LayerNorm(d) (ces.cue_norm: layernorm | none).

Head (ces.head):
  additive (default):  s_k = w_k^T c_k,   s = sum_k s_k + b             per-cue logits are exact contributions
  mlp               :  s = MLP([c_tex; c_freq; c_bound; c_sem])          contributions only by replacement
`score_from(C)` evaluates the head on any (N,K,d) cue tensor, which is how cue removal (ces_losses.replace) and the
demographic intervention (ces_losses.demographic_swap) are scored: head-level only, no second backbone pass.
The head runs in float32 outside autocast; cues are cast to float32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ces_backbones import build_cue_backbone
from .ces_geometry import CueGeometry

CUES = ("tex", "freq", "bound", "sem")


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class TokenCue(nn.Module):
    """Pooling over a fixed cell mask, then MLP. mask is a (G,G) bool buffer."""

    def __init__(self, D, d, hidden, mask, pool="attn"):
        super().__init__()
        self.register_buffer("mask", mask.clone(), persistent=False)
        self.ln = nn.LayerNorm(D)
        self.pool = pool
        self.q = nn.Linear(D, 1) if pool == "attn" else None
        self.mlp = _mlp(D, hidden, d)

    def forward(self, feat):
        t = self.ln(feat[:, self.mask])                          # (N, M, D)
        if self.q is not None:
            a = torch.softmax(self.q(t).squeeze(-1).float(), 1).to(t.dtype)
            h = (a.unsqueeze(-1) * t).sum(1)
        else:
            h = t.mean(1)
        return self.mlp(h)


class FreqNet(nn.Module):
    """Pixel-space frequency cue: log-amplitude spectrum of the windowed face crop (3 colour channels), then
    4 x [Conv3x3 stride 2 - GroupNorm - GELU] (channels 3-16-32-64-128), global average pool, MLP(128 - h - d).
    About 0.12 M parameters with the defaults (asserted < 1 M in tests)."""

    def __init__(self, geom, d, hidden=128, widths=(16, 32, 64, 128)):
        super().__init__()
        self.geom = geom
        self.register_buffer("window", geom.window.clone(), persistent=False)
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, 2, 1), nn.GroupNorm(min(8, w // 4), w), nn.GELU()]
            c = w
        self.cnn = nn.Sequential(*layers)
        self.mlp = _mlp(c, hidden, d)

    @staticmethod
    def spectrum(crop):
        """(N,3,P,P) -> (N,3,P,P) log(1 + |F|), F = fftshift(fft2(crop, norm='ortho')), float32."""
        Fq = torch.fft.fftshift(torch.fft.fft2(crop.float(), norm="ortho"), dim=(-2, -1))
        return torch.log1p(Fq.abs())

    def forward(self, x01):
        with torch.autocast(device_type=x01.device.type, enabled=False):
            a = self.spectrum(self.geom.face_crop(x01.float(), self.window))
        return self.mlp(self.cnn(a.to(next(self.cnn.parameters()).dtype)).mean((2, 3)))


class CueHead(nn.Module):
    def __init__(self, K, d, kind="additive", hidden=256):
        super().__init__()
        self.kind, self.K, self.d = kind, K, d
        if kind == "additive":
            self.w = nn.Parameter(torch.randn(K, d) * d ** -0.5)
            self.b = nn.Parameter(torch.zeros(()))
        elif kind == "mlp":
            self.net = nn.Sequential(nn.LayerNorm(K * d), nn.Linear(K * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(f"ces.head must be additive | mlp, got {kind}")

    def per_cue(self, C):
        """(N,K,d) -> (N,K) per-cue logits s_k (additive head only, else None)."""
        if self.kind != "additive":
            return None
        return (C.float() * self.w).sum(-1)

    def score_from(self, C):
        with torch.autocast(device_type=C.device.type, enabled=False):
            C = C.float()
            if self.kind == "additive":
                return self.per_cue(C).sum(1) + self.b
            return self.net(C.flatten(1)).squeeze(-1)


class CESDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.get("ces", {})
        self.backbone = build_cue_backbone(cfg)
        D, self.grid = self.backbone.feat_dim, self.backbone.grid
        self.geom = CueGeometry.from_cfg(cfg, self.grid)
        d, h = int(c.get("dim", 128)), int(c.get("hidden", 256))
        pool = c.get("pool", "attn")
        self.cue_names = CUES
        self.branches = nn.ModuleDict({
            "tex": TokenCue(D, d, h, self.geom.core_cells, pool),
            "freq": FreqNet(self.geom, d, int(c.get("freq_hidden", 128)), tuple(c.get("freq_widths", (16, 32, 64, 128)))),
            "bound": TokenCue(D, d, h, self.geom.ring_cells, pool),
            "sem": nn.Sequential(nn.LayerNorm(D), _mlp(D, h, d)),
        })
        self.cue_norm = nn.ModuleDict({k: nn.LayerNorm(d) for k in CUES}) if c.get("cue_norm", "layernorm") == "layernorm" \
            else None
        self.head = CueHead(len(CUES), d, c.get("head", "additive"), int(c.get("head_hidden", 256)))
        # CESel-MA frequency warm-up: own auxiliary linear head on c_freq (loss.freq_aux, first epoch only); it never
        # enters the score s, so the detector's output and per-cue contributions are unchanged by its presence
        if float(cfg.get("loss", {}).get("freq_aux", {}).get("weight", 0.0)) > 0:
            self.freq_aux = nn.Linear(d, 1)
        self.dim = d
        self.k = 8          # unused by CES; kept so CFES helpers that read model.k do not fail

    def cues(self, x01):
        feat, z = self.backbone(x01)
        out = {"tex": self.branches["tex"](feat), "freq": self.branches["freq"](x01),
               "bound": self.branches["bound"](feat), "sem": self.branches["sem"](z)}
        if self.cue_norm is not None:
            out = {k: self.cue_norm[k](v.float()) for k, v in out.items()}
        return torch.stack([out[k].float() for k in CUES], 1)               # (N,K,d)

    def forward(self, x01):
        C = self.cues(x01)
        return dict(score=self.head.score_from(C), cues=C, s_k=self.head.per_cue(C), patch=None, s_cls=None)

    def score_from(self, C):
        return self.head.score_from(C)

    def fusion_alpha(self):
        return None

    def orthogonality_loss(self):
        return self.backbone.orthogonality_loss()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr, lr_new, wd):
        """Backbone adapters at lr (CFES phase-2 lr), new modules (branches, norms, head) at lr_new."""
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        ids = {id(p) for p in bb}
        new = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": new, "lr": lr_new, "weight_decay": wd, "name": "new"}]
        if bb:
            groups.insert(0, {"params": bb, "lr": lr, "weight_decay": wd, "name": "backbone"})
        return groups

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    def parameter_report(self):
        cnt = lambda ps: sum(p.numel() for p in ps)
        rep = {"total": cnt(self.parameters()), "trainable": cnt(p for p in self.parameters() if p.requires_grad),
               "backbone_trainable": cnt(p for p in self.backbone.parameters() if p.requires_grad),
               "head": cnt(self.head.parameters())}
        for k in CUES:
            rep[f"branch_{k}"] = cnt(self.branches[k].parameters()) + (cnt(self.cue_norm[k].parameters())
                                                                       if self.cue_norm is not None else 0)
        return rep


def load_cfes_backbone(model, path):
    """Warm start the CLIP backbone from a CFES checkpoint (e.g. runs/clip_s7_p1/last.pt): every key 'backbone.*'
    maps to 'backbone.net.*'; all backbone keys must be present (strict on the backbone). Heads are not loaded."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {"net." + k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"init_backbone_ckpt {path}: missing {missing[:5]} unexpected {unexpected[:5]}")
    return len(sd)
