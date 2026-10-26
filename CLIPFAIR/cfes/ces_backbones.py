"""Swappable backbones for CES. Every backbone maps x01 (N,3,S,S) in [0,1] to

    feat (N, G, G, D)  token / feature-map grid         z (N, D)  global vector ("semantic" cue input)

and exposes feat_dim (D), grid (G), orthogonality_loss() and its own input normalisation. Nothing downstream
hard-codes CLIP shapes: cue masks are built for (input_size, grid) by ces_geometry.CueGeometry.

  clip   : FairReL CLIP ViT-L/14, frozen, rank-1 SVD residual adapters on every attention projection (the CFES-D
           backbone, cfes/model.py). feat = last block after ln_post (16x16x1024 at 224), z = CLS token.
  resnet : Ding et al.'s ResNet-50 + adjust_channel (cfes/model_resnet.py); feat = (7x7x512 at 224), z = avg pool.
  xception: Ding et al.'s Xception (cfes/model_xception.py, needs paths.ding_repo); feat = adjust_channel map,
           z = avg pool.
  stub   : tiny patchify conv, for CPU tests only (all parameters trainable).
"""
import torch
import torch.nn as nn

from .model import CLIP_MEAN, CLIP_STD, _fairrel


class CLIPCueBackbone(nn.Module):
    def __init__(self, m, fairrel_repo):
        super().__init__()
        build_backbone, self._orth = _fairrel(fairrel_repo)
        self.net = build_backbone(m["backbone"], m["pretrained"], m["rank"], m["pooling"])
        self.feat_dim, self.grid = self.net.feat_dim, self.net.grid_size[0]
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01):
        return self.net((x01 - self.mean) / self.std)

    def orthogonality_loss(self):
        return self._orth(self.net)

    def adapter_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]


class _CNNCueBackbone(nn.Module):
    """Shared logic of the two Ding CNNs: freeze according to `trainable`, keep frozen BatchNorm in eval mode."""

    def _setup_trainable(self, net, mode, mods, x):
        for p in net.parameters():
            p.requires_grad_(False)
        if mode not in mods:
            raise ValueError(f"trainable must be one of {sorted(mods)}, got {mode}")
        for mod in mods[mode]:
            for p in mod.parameters():
                p.requires_grad_(True)
        self.freeze_frozen_bn = bool(x.get("freeze_frozen_bn", True))
        self.freeze_all_bn_stats = bool(x.get("freeze_all_bn_stats", False))

    def train(self, mode=True):
        super().train(mode)
        if mode and (self.freeze_frozen_bn or self.freeze_all_bn_stats):
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and (
                        self.freeze_all_bn_stats or not any(p.requires_grad for p in m.parameters())):
                    m.eval()
        return self

    def forward(self, x01):
        f = self.features((x01 - 0.5) / 0.5)                       # Ding's Normalize([0.5]*3, [0.5]*3)
        return f.permute(0, 2, 3, 1), f.mean((2, 3))

    def orthogonality_loss(self):
        return next(self.parameters()).sum() * 0.0

    def adapter_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class ResNetCueBackbone(_CNNCueBackbone):
    def __init__(self, m):
        super().__init__()
        from .model_resnet import DingResNet50
        x = m.get("resnet", {})
        self.net = DingResNet50(x.get("weights"))
        if x.get("init_from"):
            self.net.load_state_dict(torch.load(x["init_from"], map_location="cpu", weights_only=True), strict=True)
        self.feat_dim, self.grid = self.net.fc.in_features, int(x.get("grid", 7))
        n = self.net
        self._setup_trainable(n, x.get("trainable", "last_block"),
                              {"all": [n.resnet, n.adjust_channel], "last_block": [n.adjust_channel], "none": []}, x)

    def features(self, x):
        return self.net.features(x)


class XceptionCueBackbone(_CNNCueBackbone):
    def __init__(self, m, ding_repo):
        super().__init__()
        from .model_xception import _ding
        x = m.get("xception", {})
        det = _ding(ding_repo)["XceptionDetector"]()
        if x.get("init_from"):
            det.load_state_dict(torch.load(x["init_from"], map_location="cpu", weights_only=True), strict=True)
        self.net = det.backbone
        self.feat_dim, self.grid = self.net.last_linear.in_features, int(x.get("grid", 8))
        self.net.last_linear.requires_grad_(False)                   # Ding's classifier is not used by CES
        n = self.net
        mods = {"all": [n], "last_block": [n.adjust_channel], "none": []}
        self._setup_trainable(n, x.get("trainable", "last_block"), mods, x)
        n.last_linear.requires_grad_(False)

    def features(self, x):
        return self.net.features(x)


class StubCueBackbone(nn.Module):
    """CPU-test backbone: patchify conv (stride S/G) -> GELU -> 1x1 conv; z = Linear(mean of tokens). The 1x1 conv
    plays the role of the "adapters" in gradient tests."""

    def __init__(self, m, size):
        super().__init__()
        s = m.get("stub", {})
        self.feat_dim, self.grid = int(s.get("dim", 32)), int(s.get("grid", 8))
        k = size // self.grid
        self.patch = nn.Conv2d(3, self.feat_dim, k, k)
        self.adapter = nn.Conv2d(self.feat_dim, self.feat_dim, 1)
        self.cls = nn.Linear(self.feat_dim, self.feat_dim)

    def forward(self, x01):
        f = self.adapter(torch.nn.functional.gelu(self.patch(x01 - 0.5)))
        f = f.permute(0, 2, 3, 1)
        return f, self.cls(f.mean((1, 2)))

    def orthogonality_loss(self):
        return self.adapter.weight.sum() * 0.0

    def adapter_parameters(self):
        return list(self.adapter.parameters())


def build_cue_backbone(cfg):
    m = cfg["model"]
    b = m.get("backend", "clip")
    if b == "clip":
        return CLIPCueBackbone(m, cfg["paths"]["fairrel_repo"])
    if b == "resnet":
        return ResNetCueBackbone(m)
    if b == "xception":
        return XceptionCueBackbone(m, cfg["paths"]["ding_repo"])
    if b == "stub":
        return StubCueBackbone(m, cfg["data"]["input_size"])
    raise ValueError(f"model.backend must be clip | resnet | xception | stub, got {b}")
