"""CFES on FS-VFM ViT-S/16 (arXiv 2510.10663, FSFM-CVPR25 branch FSVFM-extension-R1; weights HF Wolowolo/fsfm-3c
pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth, CC BY-NC 4.0). Provenance: agents/LIGHT_BACKBONE/PROVENANCE.md.

Network = the FS-VFM fine-tuning network (their models_vit.VisionTransformer with global_pool=True), rebuilt on timm
blocks so it does not depend on their timm version: patch_embed -> [CLS] + pos_embed -> 12 blocks (drop_path) ->
(a) global head: mean of the 196 patch tokens (CLS excluded) -> fc_norm (LayerNorm, fresh) -> Linear(384, 1)
    (their head is Linear(384, 2) + softmax; one logit is the same model for two classes); weight init
    trunc_normal(std 2e-5), as in their main_finetune_DfD.py;
(b) patch evidence head on each of the 14 x 14 patch tokens (as in model.py).
The encoder's final `norm` is dropped (their global_pool path deletes it). Everything is trainable (full fine-tune).
Score s = s_cls + alpha * topk_mean(P) (score_path fused; alpha as in model.py), or s_cls (score_path cls).

Also here (used only when a config asks for them, so no other backend changes):
  optim_param_groups()  layer-wise lr decay groups (BEiT/MAE rule, their util/lr_decay.py)
  FSVFMTrainTable       their training augmentation (timm create_transform: RandomResizedCrop, hflip, RandAugment
                        rand-m9-mstd0.5-inc1, random erasing), returned un-normalised so the model normalises as usual
  MixupBCE              timm Mixup/CutMix with label smoothing -> soft fake-probability targets for BCE
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from .model import fusion_weight, init_fusion_weight, topk_mean

# VGGFace2 pretraining statistics (pretrain_ds_mean_std.txt shipped with the checkpoint; their default)
FSVFM_MEAN = (0.5482207536697388, 0.42340534925460815, 0.3654651641845703)
FSVFM_STD = (0.2789176106452942, 0.2438540756702423, 0.23493893444538116)
ARCH = {"vit_small_patch16": dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4)}


class CFESFSVFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from functools import partial
        from timm.models.vision_transformer import VisionTransformer
        f = cfg["fsvfm"]
        a = ARCH[f.get("arch", "vit_small_patch16")]
        self.vit = VisionTransformer(img_size=224, num_classes=0, global_pool="", class_token=True, qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=float(f.get("drop_path", 0.1)),
                                     **a)
        self.vit.norm = nn.Identity()          # removed by their global_pool path; fc_norm below replaces it
        d = a["embed_dim"]
        self.grid = 224 // a["patch_size"]
        self.fc_norm = nn.LayerNorm(d, eps=1e-6)
        self.cls_head = nn.Linear(d, 1)
        nn.init.trunc_normal_(self.cls_head.weight, std=2e-5)
        nn.init.zeros_(self.cls_head.bias)
        self.load_report = None
        if f.get("weights"):
            self.load_report = self._load_pretrained(f["weights"])
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
        self.register_buffer("mean", torch.tensor(FSVFM_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(FSVFM_STD).view(1, 3, 1, 1), persistent=False)

    def _load_pretrained(self, path):
        """Encoder weights of the pretraining checkpoint ('model' dict); decoders, projector/predictor, mask token
        and the final norm are not part of the fine-tuning network. Every encoder key must load."""
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        enc = {k: v for k, v in sd.items()
               if k in ("cls_token", "pos_embed") or k.startswith("patch_embed.") or k.startswith("blocks.")}
        if enc["pos_embed"].shape != self.vit.pos_embed.shape:
            raise ValueError(f"pos_embed {tuple(enc['pos_embed'].shape)} != {tuple(self.vit.pos_embed.shape)}")
        missing, unexpected = self.vit.load_state_dict(enc, strict=False)
        missing = [k for k in missing if not k.startswith("norm.")]
        if missing or unexpected:
            raise RuntimeError(f"FS-VFM load: missing {missing[:5]} unexpected {unexpected[:5]}")
        return {"loaded": len(enc), "ignored": sorted({k.split(".")[0] for k in sd if k not in enc})}

    def tokens(self, x):
        v = self.vit
        x = v.patch_embed(x)
        x = torch.cat([v.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + v.pos_embed
        x = v.pos_drop(x)
        for blk in v.blocks:
            x = blk(x)
        return x                                             # (N, 1 + G*G, D)

    def forward(self, x01):
        t = self.tokens((x01 - self.mean) / self.std)
        pt = t[:, 1:, :]
        s_cls = self.cls_head(self.fc_norm(pt.mean(dim=1))).squeeze(-1).float()
        patch = None
        if self.patch_head is not None:
            n = pt.shape[0]
            patch = self.patch_head(pt.reshape(n, self.grid, self.grid, -1)).squeeze(-1).float()   # (N,G,G)
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
        return torch.zeros((), device=self.cls_head.weight.device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    def parameter_report(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(total=tot, trainable=tr, backbone=sum(p.numel() for p in self.vit.parameters()),
                    cls_head=sum(p.numel() for p in self.cls_head.parameters()) + sum(p.numel() for p in self.fc_norm.parameters()),
                    patch_head=sum(p.numel() for p in self.patch_head.parameters()) if self.patch_head else 0)

    def optim_param_groups(self, weight_decay, layer_decay):
        """Their util/lr_decay.param_groups_lrd: layer id 0 = cls_token/pos_embed/patch_embed, i+1 = block i, 13 = all
        heads (fc_norm, cls_head, patch_head, alpha); lr scale layer_decay^(13 - id); no weight decay on 0-/1-D params (alpha too) and
        on cls_token/pos_embed. Each group carries lr_scale; the caller multiplies the base lr by it."""
        nl = len(self.vit.blocks) + 1
        groups = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.ndim <= 1 or n in ("vit.cls_token", "vit.pos_embed")
            if n in ("vit.cls_token", "vit.pos_embed") or n.startswith("vit.patch_embed"):
                lid = 0
            elif n.startswith("vit.blocks."):
                lid = int(n.split(".")[2]) + 1
            else:
                lid = nl
            key = (lid, nd)
            if key not in groups:
                groups[key] = {"params": [], "weight_decay": 0.0 if nd else weight_decay,
                               "lr_scale": layer_decay ** (nl - lid), "layer": lid}
            groups[key]["params"].append(p)
        return [groups[k] for k in sorted(groups)]


class FSVFMTrainTable(Dataset):
    """Their training transform (main_finetune_DfD.py --apply_simple_augment -> timm create_transform, bicubic,
    RandAugment rand-m9-mstd0.5-inc1, random erasing reprob/pixel/1, no colour jitter since AA is on), applied to the
    image file as stored (our crops). Output is un-normalised back to pixel scale (x * std + mean): the model
    normalises again, so the network input equals theirs exactly (random-erasing noise may leave [0, 1])."""

    def __init__(self, df, size, aug):
        from timm.data import create_transform
        self.paths = df["img_path"].tolist()
        self.labels = torch.tensor(df["label"].values)
        self.groups = torch.tensor(df["group"].values)
        self.tf = create_transform(input_size=size, is_training=True, color_jitter=None,
                                   auto_augment=aug.get("aa", "rand-m9-mstd0.5-inc1"),
                                   interpolation="bicubic", re_prob=float(aug.get("reprob", 0.25)),
                                   re_mode="pixel", re_count=1, mean=FSVFM_MEAN, std=FSVFM_STD)
        self.m = torch.tensor(FSVFM_MEAN).view(3, 1, 1)
        self.s = torch.tensor(FSVFM_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.tf(Image.open(self.paths[i]).convert("RGB")) * self.s + self.m
        return dict(image=x, label=self.labels[i], group=self.groups[i], index=i)


class MixupBCE:
    """timm Mixup (mixup_alpha, cutmix_alpha, prob 1, switch prob 0.5, mode batch, label smoothing) on 2 classes; the
    soft target of class 'fake' is the BCE target of the single logit (2-class softmax over (0, s) = sigmoid(s), so this
    equals their SoftTargetCrossEntropy). Seeded from the training generator so runs are reproducible."""

    def __init__(self, aug):
        from timm.data.mixup import Mixup
        self.mix = Mixup(mixup_alpha=float(aug.get("mixup", 0.8)), cutmix_alpha=float(aug.get("cutmix", 1.0)),
                         prob=float(aug.get("mixup_prob", 1.0)), switch_prob=float(aug.get("mixup_switch_prob", 0.5)),
                         mode="batch", label_smoothing=float(aug.get("smoothing", 0.1)), num_classes=2)

    def __call__(self, x, y, gen):
        st = np.random.get_state()
        np.random.seed(int(torch.randint(2 ** 31 - 1, (1,), generator=gen)))   # timm Mixup draws from np.random
        try:
            xm, t = self.mix(x.clone(), y.long())
        finally:
            np.random.set_state(st)
        return xm, t[:, 1].float()


def gmacs(model, size=224):
    """Analytic multiply-accumulates of one forward of the ViT encoder at size x size (conv patch embedding,
    qkv/proj/MLP linears, attention matmuls); heads excluded (< 0.1 MMAC)."""
    v = model.vit
    d = v.embed_dim
    p = v.patch_embed.patch_size[0]
    n = (size // p) ** 2
    t = n + 1
    hid = v.blocks[0].mlp.fc1.out_features
    patch = n * 3 * p * p * d
    per_block = t * d * 3 * d + 2 * t * t * d + t * d * d + 2 * t * d * hid
    return (patch + len(v.blocks) * per_block) / 1e9
