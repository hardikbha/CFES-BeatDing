"""Beat-Ding on Ding et al.'s Xception (Tables 1-2): CFESXceptionBD (plan Amendments 4, 4a, 4b; design
agents/BEAT_DING/XCEPTION_DESIGN.md sections 3-5). Backend name: model.backend = xception_bd (cfes/engine.build_model
imports this module lazily; cfes/model_xception.py is NOT used or changed).

Network = Ding's Xception, mode 'adjust_channel' (cfes/ding_xception.py: Ding's classes copied verbatim, bit-identity
tested) -> adjust_channel map F (N,512,7,7 at 224 px) -> (a) Ding's classifier avg-pool + last_linear (512->2);
s_cls = logit_fake - logit_real, so sigmoid(s_cls) = Ding's softmax P(fake); (b) a patch head LN -> 512->h -> GELU -> h->1
on every one of the 7 x 7 feature vectors, fused by score = s_cls + alpha * topk_mean(P) (alpha_param tanh, |alpha| <= 1).
Input: x in [0,1] at 224 px (PIL bilinear resize = torchvision Resize), (x - 0.5) / 0.5 inside the model = Ding's
resnet_default_data_transforms (ToTensor + Normalize 0.5 / 0.5). fp32.

Start (D1 = X0): model.xception.init_from = DINGREPRO_LIN/results/DING_base_s<seed>/49.pth, loaded STRICTLY (Ding's own
phase-1 checkpoint, run by us with his official train_ori_model.py). ImageNet weights (model.xception.weights) are loaded
only when init_from is null (X1 fallback; unused). With model.xception.patch_zero_init the patch head's LAST Linear is
zero-initialised, so P == 0 and every starting score equals s_cls exactly (start equivalence, max |dscore| = 0).

BatchNorm (design 3.3): bn_recal() = one-shot cumulative re-estimation of EVERY BatchNorm's running statistics
(reset, momentum=None, train-mode forward under no_grad over K clean 32 real + 32 fake FF++ train batches, Ding's
transform, no augmentation, sampler seed = run seed + 999 with its own DataLoader generator: the global RNG and the
training data order are untouched); weights untouched; momentum restored afterwards. Then freeze_all_bn_stats keeps every
BN in eval mode during training (the mixed BD step batch never touches BN statistics), and the trunk + adjust_channel
(incl. the BN affine parameters) are frozen under trainable: head -> no trainable layer feeds a frozen-statistics BN.

trainable: head (BD) = last_linear (+ patch head, alpha, + C heads and gate); last_block / all as in model_xception.py.

Component C on Xception (design 3.6; OFF unless model.multilayer.enabled): forward hooks on Ding's block2 (N,256,28,28,
end of the stride-8 entry stage), block11 (N,728,14,14, last middle-flow block) and block12 (N,1024,7,7, exit block);
per stage a 1x1 conv (C -> 1) = an evidence-logit map, adaptive-avg-pooled to the 7 x 7 grid, averaged over stages = P2;
score += beta * topk_mean(P2, k), beta = gate_max * tanh(ml_gate_raw), ml_gate_raw = 0 at the start (no local norm, as on
ResNet). Heads come from their own seed (init_seed), so the global RNG stream is unchanged. The hooked outputs are only
read by non-inplace ops (block3.rep[0] / block12.rep[0] are ReLU(inplace=False), conv3 is a convolution).
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ding_xception import build_ding_xception, load_detector, load_imagenet
from .model import _keep_block_output, fusion_weight, init_fusion_weight, topk_mean

DING_MEAN, DING_STD = 0.5, 0.5
STAGES = {"block1": 128, "block2": 256, "block3": 728, "block4": 728, "block5": 728, "block6": 728, "block7": 728,
          "block8": 728, "block9": 728, "block10": 728, "block11": 728, "block12": 1024}


def _bns(module):
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class CFESXceptionBD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        x = cfg["xception"]
        self.net = build_ding_xception()
        self.init_report = {}
        if x.get("init_from"):
            load_detector(self.net, x["init_from"])
            self.init_report = {"init_from": x["init_from"]}
        elif x.get("weights"):
            self.init_report = {"imagenet": x["weights"], "missing": load_imagenet(self.net, x["weights"])}
        d = self.net.last_linear.in_features
        self.grid = int(x.get("grid", 7))
        ph = cfg["patch_head"]
        self.patch_head = None
        if ph["enabled"]:
            h = ph["hidden"]
            self.patch_head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
                               if h else nn.Linear(d, 1))
            if x.get("patch_zero_init"):
                last = [m for m in self.patch_head.modules() if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        self.score_path = cfg["score_path"]
        if self.score_path != "cls" and self.patch_head is None:
            raise ValueError("score_path topk/fused needs model.patch_head.enabled=true")
        self.k = cfg["topk"]
        init_fusion_weight(self, cfg)
        for p in self.net.parameters():
            p.requires_grad_(False)
        mode = x["trainable"]
        mods = {"all": [self.net], "last_block": [self.net.adjust_channel, self.net.last_linear],
                "head": [self.net.last_linear]}
        if mode not in mods:
            raise ValueError(mode)
        for m in mods[mode]:
            for p in m.parameters():
                p.requires_grad_(True)
        self.freeze_frozen_bn = bool(x.get("freeze_frozen_bn", True))
        self.freeze_all_bn_stats = bool(x.get("freeze_all_bn_stats", False))
        self.recal_cfg = dict(x.get("bn_recal") or {})
        ml = cfg.get("multilayer") or {}
        self.ml_enabled = bool(ml.get("enabled", False))
        self.ml_heads = None
        if self.ml_enabled:
            if self.score_path != "fused" or self.patch_head is None:
                raise ValueError("model.multilayer needs score_path=fused and an enabled patch head")
            self.ml_stages = list(ml.get("stages", ["block2", "block11", "block12"]))
            if not self.ml_stages or set(self.ml_stages) - set(STAGES):
                raise ValueError(f"model.multilayer.stages must be among {sorted(STAGES)}, got {self.ml_stages}")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(ml.get("init_seed", 0)))
                self.ml_heads = nn.ModuleList([nn.Conv2d(STAGES[st], 1, 1) for st in self.ml_stages])
            self.ml_gate_max = float(ml.get("gate_max", 1.0))
            self.ml_gate_raw = nn.Parameter(torch.tensor(0.0))
            self._ml_hook_handles = [getattr(self.net, st).register_forward_hook(_keep_block_output)
                                     for st in self.ml_stages]

    # ---------------------------------------------------------------------------------------------- component C
    def ml_gate(self):
        return self.ml_gate_max * torch.tanh(self.ml_gate_raw)

    def _ml_clear(self):
        if self.ml_enabled:
            for st in self.ml_stages:
                getattr(self.net, st).__dict__.pop("_bd_tokens", None)

    def _ml_map(self, hw):
        """(N, h, w) average of the per-stage evidence maps, each pooled to the existing map's h x w grid."""
        maps = []
        for st, head in zip(self.ml_stages, self.ml_heads):
            f = getattr(self.net, st).__dict__.pop("_bd_tokens", None)
            if f is None:
                raise RuntimeError(f"multi-stage evidence: {st} captured no feature map in this forward pass")
            maps.append(F.adaptive_avg_pool2d(head(f), hw).squeeze(1).float())
        return torch.stack(maps).mean(0)

    # ---------------------------------------------------------------------------------------------- BatchNorm
    def train(self, mode=True):
        super().train(mode)
        if mode and (self.freeze_frozen_bn or self.freeze_all_bn_stats):
            for m in _bns(self.net):
                if self.freeze_all_bn_stats or not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    @torch.no_grad()
    def bn_recal(self, batches, device, max_batches=None):
        """One-shot cumulative re-estimation of every BatchNorm's running statistics (weights untouched).
        batches: iterable of dicts with "image" (N,3,H,W) in [0,1] or of tensors. Returns a report dict."""
        bns = _bns(self.net)
        was_training = self.training
        moms = [m.momentum for m in bns]
        self.eval()
        for m in bns:
            m.reset_running_stats()
            m.momentum = None                     # cumulative moving average = exact mean of the per-batch statistics
            m.train()
        t0, n_b, n_img, labels = time.time(), 0, 0, []
        try:
            for b in batches:
                if max_batches is not None and n_b >= max_batches:
                    break
                xb = b["image"] if isinstance(b, dict) else b
                if isinstance(b, dict) and "label" in b:
                    labels.append(b["label"])
                self.net.features((xb.to(device, non_blocking=True) - DING_MEAN) / DING_STD)
                self._ml_clear()
                n_b += 1
                n_img += int(xb.shape[0])
        finally:
            for m, mo in zip(bns, moms):
                m.momentum = mo
                m.eval()
            self._ml_clear()
            self.train(was_training)
        if n_b == 0:
            raise RuntimeError("bn_recal: no batch")
        finite = all(bool(torch.isfinite(m.running_mean).all() and torch.isfinite(m.running_var).all()) for m in bns)
        if not finite:
            raise FloatingPointError("bn_recal: non-finite BatchNorm statistics")
        lab = torch.cat(labels) if labels else None
        named = dict((n, m) for n, m in self.net.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
        return {"batches": n_b, "images": n_img, "n_bn": len(bns), "seconds": round(time.time() - t0, 2), "finite": finite,
                "n_real": int((lab == 0).sum()) if lab is not None else None,
                "n_fake": int((lab == 1).sum()) if lab is not None else None,
                "num_batches_tracked": int(bns[0].num_batches_tracked),
                "running_var_mean": {k: float(named[k].running_var.mean()) for k in
                                     ("bn1", "block12.skipbn", "bn3", "bn4", "adjust_channel.1") if k in named}}

    # ---------------------------------------------------------------------------------------------- forward
    def forward(self, x01):
        self._ml_clear()
        feat = self.net.features((x01 - DING_MEAN) / DING_STD)          # (N,512,h,w)
        logits = self.net.classifier(feat).float()
        s_cls = logits[:, 1] - logits[:, 0]
        patch = None
        if self.patch_head is not None:
            patch = self.patch_head(feat.permute(0, 2, 3, 1)).squeeze(-1).float()   # (N,h,w)
        if self.score_path == "cls":
            score = s_cls
        elif self.score_path == "topk":
            score = topk_mean(patch, self.k)
        else:
            score = s_cls + fusion_weight(self).float() * topk_mean(patch, self.k)
            if self.ml_enabled:                 # Beat-Ding C (Xception): + beta * top-k mean of the multi-stage map
                patch_ml = self._ml_map(tuple(feat.shape[-2:]))
                t2 = topk_mean(patch_ml, self.k)
                score = score + self.ml_gate().float() * t2
                return dict(score=score, s_cls=s_cls, patch=patch, patch_ml=patch_ml, t_ml=t2)
        self._ml_clear()
        return dict(score=score, s_cls=s_cls, patch=patch)

    def fusion_alpha(self):
        return fusion_weight(self)

    def orthogonality_loss(self):
        return torch.zeros((), device=self.net.last_linear.weight.device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    def parameter_report(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rep = dict(total=tot, trainable=tr,
                   patch_head=sum(p.numel() for p in self.patch_head.parameters()) if self.patch_head else 0,
                   backbone_trainable=sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        if self.ml_heads is not None:
            rep["ml_heads"] = sum(p.numel() for p in self.ml_heads.parameters()) + 1     # + the gate
        return rep


# -------------------------------------------------------------------------------------------------- step 0 (train.py)
def recal_loader(cfg, seed):
    """The step-0 BN re-estimation stream: clean FF++ train frames (no augmentation, no views) at the model input size,
    BalancedRealFakeSampler(per_class real + per_class fake per batch, `batches` batches, seed = run seed + seed_offset),
    with its OWN DataLoader generator (the global torch RNG is not consumed)."""
    from torch.utils.data import DataLoader

    from .data import BalancedRealFakeSampler, ImageTable, read_split
    rc = cfg["model"]["xception"].get("bn_recal") or {}
    d = cfg["data"]
    df = read_split({**d["train"], "max_items": cfg["debug"].get("max_train_items")})
    per = int(rc.get("per_class", 32))
    s = int(seed) + int(rc.get("seed_offset", 999))
    sampler = BalancedRealFakeSampler(df["label"].values, 2 * per, steps=int(rc["batches"]), seed=s)
    ds = ImageTable(df, d["input_size"], d.get("resample", "bicubic"))
    g = torch.Generator().manual_seed(s)
    return DataLoader(ds, batch_size=2 * per, sampler=sampler, num_workers=d["num_workers"], pin_memory=True,
                      drop_last=True, generator=g)


def bn_recal_from_cfg(model, cfg, device):
    """train.py step 0 (fresh runs only): run model.bn_recal on recal_loader(cfg, seed). Returns the report."""
    rc = cfg["model"]["xception"].get("bn_recal") or {}
    rep = model.bn_recal(recal_loader(cfg, cfg["seed"]), device, max_batches=int(rc["batches"]))
    rep.update(per_class=int(rc.get("per_class", 32)), seed=int(cfg["seed"]) + int(rc.get("seed_offset", 999)),
               momentum="cumulative (None), restored afterwards")
    return rep


def bn_recal_enabled(cfg):
    m = cfg.get("model") or {}
    return m.get("backend") == "xception_bd" and int(((m.get("xception") or {}).get("bn_recal") or {}).get("batches", 0)) > 0
