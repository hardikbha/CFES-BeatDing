"""CFES on Ding et al.'s Xception network (XceptionDetector, official code; ImageNet initialisation unless
model.xception.init_from is set -- the phase-1/phase-2 configs never load Ding's trained checkpoints).

Network: Ding's XceptionDetector (official code in DINGREPRO_LIN/code/Ding_mod) -> adjust_channel feature map
F (N,512,h,w) -> (a) Ding's own classifier: avg-pool + last_linear(512->2); s_cls = logit_fake - logit_real, so
sigmoid(s_cls) is exactly Ding's softmax P(fake); (b) a patch evidence head on each of the h x w feature vectors.
Score s = s_cls + alpha * topk_mean(P) (score_path fused; alpha as in model.py: raw or softplus).

`trainable` picks what is updated:
  head       : last_linear (+ patch head, alpha)               ~ Ding's GDA (last_linear only)
  last_block : adjust_channel + last_linear (+ patch head, alpha)
  all        : the whole network
BatchNorm layers inside frozen parts keep their stage-1 running statistics (eval mode) when
freeze_frozen_bn is true.
"""
import sys

import torch
import torch.nn as nn

from .model import fusion_weight, init_fusion_weight, topk_mean

DING_MEAN, DING_STD = 0.5, 0.5


def _ding(repo):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from detectors import DETECTOR
    return DETECTOR


class CFESXception(nn.Module):
    def __init__(self, cfg, ding_repo):
        super().__init__()
        x = cfg["xception"]
        det = _ding(ding_repo)["XceptionDetector"]()
        if x.get("init_from"):
            sd = torch.load(x["init_from"], map_location="cpu", weights_only=True)
            det.load_state_dict(sd, strict=True)
        self.net = det.backbone
        d = self.net.last_linear.in_features
        self.grid = int(x.get("grid", 8))      # feature-map side at the training input size (256/32)
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
        if mode == "all":
            mods = [self.net]
        elif mode == "last_block":
            mods = [self.net.adjust_channel, self.net.last_linear]
        elif mode == "head":
            mods = [self.net.last_linear]
        else:
            raise ValueError(mode)
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(True)
        self.freeze_frozen_bn = bool(x.get("freeze_frozen_bn", True))
        # "all": every BatchNorm keeps its running statistics (affine weights still train). Needed when batches mix
        # clean images, counterfactual views and pairs: updating running stats from such batches collapsed the
        # adjust_channel BN variance (~0) and saturated every eval score (seed 27 phase 2, 16 Oct).
        self.freeze_all_bn_stats = bool(x.get("freeze_all_bn_stats", False))

    def train(self, mode=True):
        super().train(mode)
        if mode and (self.freeze_frozen_bn or self.freeze_all_bn_stats):
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and (
                        self.freeze_all_bn_stats or not any(p.requires_grad for p in m.parameters())):
                    m.eval()
        return self

    def forward(self, x01):
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
        return dict(total=tot, trainable=tr,
                    patch_head=sum(p.numel() for p in self.patch_head.parameters()) if self.patch_head else 0,
                    backbone_trainable=sum(p.numel() for p in self.net.parameters() if p.requires_grad))
