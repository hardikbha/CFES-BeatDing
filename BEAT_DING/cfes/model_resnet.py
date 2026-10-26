"""CFES on Ding et al.'s ResNet-50 network (their Table 3 backbone).

Network = Ding's utils/resnet50.py ResNet50 in mode 'adjust_channel' (dropout off), rebuilt here so no weights are
downloaded: torchvision ResNet-50 trunk (conv1 ... layer4) -> adjust_channel (1x1 conv 2048->512 + BN + ReLU) ->
feature map F (N,512,h,w) -> avg-pool -> fc (512->2). s_cls = logit_fake - logit_real, so sigmoid(s_cls) equals
Ding's softmax P(fake). A patch evidence head reads each of the h x w feature vectors (7 x 7 at 224 px).
Score s = s_cls + alpha * topk_mean(P), alpha as in model.py (raw | softplus).

Initialisation: torchvision's ImageNet-1k V1 ResNet-50 weights (resnet50-19c8e357.pth, the weights Ding's code gets
from torchvision resnet50(pretrained=True)); adjust_channel, fc and the patch head start from their default init.
Ding's trained checkpoints are never loaded. `trainable` and the BatchNorm switches work as in model_xception.py.
"""
import torch
import torch.nn as nn
import torchvision

import torch.nn.functional as F

from .model import _keep_block_output, fusion_weight, init_fusion_weight, topk_mean

MEAN, STD = 0.5, 0.5          # Ding's ResNet transforms: Normalize([0.5]*3, [0.5]*3)


class DingResNet50(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)   # legacy-format official file
            r.load_state_dict(sd, strict=True)
        self.resnet = nn.Sequential(*list(r.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.adjust_channel = nn.Sequential(nn.Conv2d(2048, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, 2)

    def features(self, x):
        return self.adjust_channel(self.resnet(x))

    def classifier(self, f):
        return self.fc(self.avgpool(f).flatten(1))


class CFESResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        x = cfg["resnet"]
        self.net = DingResNet50(x.get("weights"))
        if x.get("init_from"):
            self.net.load_state_dict(torch.load(x["init_from"], map_location="cpu", weights_only=True), strict=True)
        d = self.net.fc.in_features
        self.grid = int(x.get("grid", 7))
        ph = cfg["patch_head"]
        self.patch_head = None
        if ph["enabled"]:
            h = ph["hidden"]
            self.patch_head = (nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
                               if h else nn.Linear(d, 1))
        self.score_path = cfg["score_path"]
        if self.score_path != "cls" and self.patch_head is None:
            raise ValueError("score_path topk/fused needs model.patch_head.enabled=true")
        self.k = cfg["topk"]
        init_fusion_weight(self, cfg)
        for p in self.net.parameters():
            p.requires_grad_(False)
        mode = x["trainable"]
        mods = {"all": [self.net], "last_block": [self.net.adjust_channel, self.net.fc], "head": [self.net.fc]}
        if mode not in mods:
            raise ValueError(mode)
        for m in mods[mode]:
            for p in m.parameters():
                p.requires_grad_(True)
        self.freeze_frozen_bn = bool(x.get("freeze_frozen_bn", True))
        self.freeze_all_bn_stats = bool(x.get("freeze_all_bn_stats", False))
        # Beat-Ding component C on ResNet-50 (plan Amendment 2): multi-STAGE evidence, OFF by default (nothing created).
        # Outputs of the trunk stages layer2 / layer3 / layer4 (forward hooks on net.resnet[5] / [6] / [7]; at 224 px
        # (N,512,28,28), (N,1024,14,14), (N,2048,7,7)) -> per stage a 1x1 conv (C -> 1) = an evidence logit map ->
        # adaptive average pooling to the patch grid of the existing map (7 x 7) -> average over stages = P2, fused like
        # the existing ResNet map (top-k mean, no local norm on ResNet):  score = s_cls + alpha t + beta topk_mean(P2),
        # beta = gate_max * tanh(ml_gate_raw), ml_gate_raw = 0 -> every starting score equals the base's. Heads from
        # their own seed (global RNG stream unchanged). With the phase-2 scope (trainable: head) the trunk stays frozen.
        ml = cfg.get("multilayer") or {}
        self.ml_enabled = bool(ml.get("enabled", False))
        self.ml_heads = None
        if self.ml_enabled:
            if self.score_path != "fused" or self.patch_head is None:
                raise ValueError("model.multilayer needs score_path=fused and an enabled patch head")
            idx, chans = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}, {"layer1": 256, "layer2": 512,
                                                                                    "layer3": 1024, "layer4": 2048}
            self.ml_stages = list(ml.get("stages", ["layer2", "layer3", "layer4"]))
            if not self.ml_stages or set(self.ml_stages) - set(idx):
                raise ValueError(f"model.multilayer.stages must be among {sorted(idx)}, got {self.ml_stages}")
            self.ml_idx = [idx[st] for st in self.ml_stages]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(ml.get("init_seed", 0)))
                self.ml_heads = nn.ModuleList([nn.Conv2d(chans[st], 1, 1) for st in self.ml_stages])
            self.ml_gate_max = float(ml.get("gate_max", 1.0))
            self.ml_gate_raw = nn.Parameter(torch.tensor(0.0))
            self._ml_hook_handles = [self.net.resnet[i].register_forward_hook(_keep_block_output) for i in self.ml_idx]

    def ml_gate(self):
        return self.ml_gate_max * torch.tanh(self.ml_gate_raw)

    def _ml_clear(self):
        for i in self.ml_idx:
            self.net.resnet[i].__dict__.pop("_bd_tokens", None)

    def _ml_map(self, hw):
        """(N, h, w) average of the per-stage evidence maps, each pooled to the existing map's h x w grid."""
        maps = []
        for st, i, head in zip(self.ml_stages, self.ml_idx, self.ml_heads):
            f = self.net.resnet[i].__dict__.pop("_bd_tokens", None)
            if f is None:
                raise RuntimeError(f"multi-stage evidence: {st} captured no feature map in this forward pass")
            maps.append(F.adaptive_avg_pool2d(head(f), hw).squeeze(1).float())
        return torch.stack(maps).mean(0)

    def train(self, mode=True):
        super().train(mode)
        if mode and (self.freeze_frozen_bn or self.freeze_all_bn_stats):
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and (
                        self.freeze_all_bn_stats or not any(p.requires_grad for p in m.parameters())):
                    m.eval()
        return self

    def forward(self, x01):
        if self.ml_enabled:
            self._ml_clear()
        feat = self.net.features((x01 - MEAN) / STD)                      # (N,512,h,w)
        logits = self.net.classifier(feat).float()
        s_cls = logits[:, 1] - logits[:, 0]
        patch = None
        if self.patch_head is not None:
            patch = self.patch_head(feat.permute(0, 2, 3, 1)).squeeze(-1).float()
        if self.score_path == "cls":
            score = s_cls
        elif self.score_path == "topk":
            score = topk_mean(patch, self.k)
        else:
            score = s_cls + fusion_weight(self).float() * topk_mean(patch, self.k)
            if self.ml_enabled:                     # Beat-Ding C (ResNet): + beta * top-k mean of the multi-stage map
                patch_ml = self._ml_map(tuple(feat.shape[-2:]))
                t2 = topk_mean(patch_ml, self.k)
                score = score + self.ml_gate().float() * t2
                return dict(score=score, s_cls=s_cls, patch=patch, patch_ml=patch_ml, t_ml=t2)
        if self.ml_enabled:
            self._ml_clear()
        return dict(score=score, s_cls=s_cls, patch=patch)

    def fusion_alpha(self):
        return fusion_weight(self)

    def orthogonality_loss(self):
        return torch.zeros((), device=self.net.fc.weight.device)

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
