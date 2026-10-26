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

from .model import fusion_weight, init_fusion_weight, topk_mean

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

    def train(self, mode=True):
        super().train(mode)
        if mode and (self.freeze_frozen_bn or self.freeze_all_bn_stats):
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and (
                        self.freeze_all_bn_stats or not any(p.requires_grad for p in m.parameters())):
                    m.eval()
        return self

    def forward(self, x01):
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
        return dict(total=tot, trainable=tr,
                    patch_head=sum(p.numel() for p in self.patch_head.parameters()) if self.patch_head else 0,
                    backbone_trainable=sum(p.numel() for p in self.net.parameters() if p.requires_grad))
