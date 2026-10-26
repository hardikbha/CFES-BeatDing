"""Detectors of the four matched-backbone fairness baselines (Amendment 3). Every one uses the SAME backbone as CESel B0:
FairReL CLIP ViT-L/14, frozen, rank-1 SVD residual adapters on every attention projection (cfes/ces_backbones.py
CLIPCueBackbone; feat = 16x16x1024 patch tokens after ln_post, z = CLS token after ln_post), warm-started from the
CFES phase-1 checkpoint runs/clip_s{seed}_p1/last.pt (backbone keys only, as B0; Ding also takes its CLS head).

  LinearHeadDetector  DAW-FDD / DAG-FDD: one logit = Linear(1024, 1)(z)  (official: Xception classifier = relu,
                      global avg pool, last_linear(2048 -> num_classes = 1), AIF detectors/daw_fdd.py:36-38, 70-71)
  DingDetector        SFD + GDA: z with the decoupled channels zeroed -> Linear(1024, 2); score = l1 - l0
                      (official: XceptionDetector, 2-way CE, DM detectors/xception_detector.py:68-83; pruned
                      adjust_channel out-channels, DM GDA.py:63-74,157-158)
  PGFDDDetector       PG-FDD (FG detectors/fair_df_detector.py): three "encoders", blocks, AdaIN fusion, conditional
                      UNet decoder, four heads; inference = fused head, score = l1 - l0

score = l1 - l0 for the 2-way heads: sigmoid(score) is exactly softmax(.)[:, 1] (the official test probability), so
every downstream metric and the 0.5 decision are those of the official 2-way head.
Heads run in float32 outside autocast (as the CES head); the trunk and the PG-FDD conv modules run under the caller's
bf16 autocast.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ces_backbones import build_cue_backbone


class _Base(nn.Module):
    k = 8                                    # unused; CFES helpers read model.k

    def orthogonality_loss(self):
        return self.backbone.orthogonality_loss()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr, lr_new, wd):
        """Backbone adapters at lr (CFES phase-2 / B0 lr 1e-4), every new module at lr_new (1e-3), as CESDetector."""
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
        return {"total": cnt(self.parameters()), "trainable": cnt(self.trainable_parameters()),
                "backbone_trainable": cnt(p for p in self.backbone.parameters() if p.requires_grad),
                "new": cnt(p for n, p in self.named_parameters() if not n.startswith("backbone."))}

    def fusion_alpha(self):
        return None


def _fp32(mod, x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return mod(x.float())


# ------------------------------------------------------------------------------------------------ DAW / DAG
class LinearHeadDetector(_Base):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_cue_backbone(cfg)
        self.head = nn.Linear(self.backbone.feat_dim, 1)
        self.grid = self.backbone.grid

    def forward(self, x01):
        feat, z = self.backbone(x01)
        s = _fp32(self.head, z).squeeze(-1)
        return dict(score=s, z=z, patch=None, s_cls=s)


# ------------------------------------------------------------------------------------------------ Ding SFD + GDA
class DingDetector(_Base):
    """keep: (D,) persistent buffer, 0 on the SFD channels (set by set_sfd_channels; saved in every checkpoint)."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_cue_backbone(cfg)
        D = self.backbone.feat_dim
        self.head = nn.Linear(D, 2)
        self.register_buffer("keep", torch.ones(D))
        self.grid = self.backbone.grid
        self.sfd_channels = []
        if not bool(cfg["clipfair"]["ding"].get("train_backbone", False)):
            # DM GDA.py:121-124: every parameter frozen except backbone.last_linear (the classifier)
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def set_sfd_channels(self, channels):
        self.keep.fill_(1.0)
        if channels:
            self.keep[torch.as_tensor(channels, dtype=torch.long)] = 0.0
        self.sfd_channels = list(channels)

    def init_head_from_logit(self, w, b):
        """2-way head equal to a 1-logit head (w, b): rows (-w/2, w/2), biases (-b/2, b/2) -> l1 - l0 = w.z + b."""
        with torch.no_grad():
            self.head.weight.copy_(torch.stack([-w / 2, w / 2]))
            self.head.bias.copy_(torch.stack([-b / 2, b / 2]))

    def forward(self, x01):
        feat, z = self.backbone(x01)
        zm = z.float() * self.keep
        out = _fp32(self.head, zm)
        s = out[:, 1] - out[:, 0]
        return dict(score=s, logits2=out, z=zm, feat=feat, patch=None, s_cls=s)


# ------------------------------------------------------------------------------------------------ PG-FDD
class AdaIN(nn.Module):
    """FG fair_df_detector.py:453-476, verbatim."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def c_norm(self, x, bs, ch, eps=1e-7):
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        with torch.autocast(device_type=x.device.type, enabled=False):     # float32 statistics (bf16 trunk)
            return self._forward(x.float(), y.float())

    def _forward(self, x, y):
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        return ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(size) + y_mean.expand(size)


def r_double_conv(in_channels, out_channels):
    """FG fair_df_detector.py:444-450."""
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))


class ConditionalUNet(nn.Module):
    """FG fair_df_detector.py:479-538, verbatim except the final upsampling: the official up_last is x4 from an 8x8
    Xception map (8 -> 64 -> x4 = 256 px input); on the 16x16 CLIP grid the three x2 stages give 128, so up_last
    resamples to the input size (224) instead of a fixed x4 (CLIP adaptation)."""

    def __init__(self, out_size=224):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        self.adain3, self.adain2, self.adain1 = AdaIN(), AdaIN(), AdaIN()
        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(size=(out_size, out_size), mode="bilinear", align_corners=True)
        self.activation = nn.Tanh()

    def forward(self, c, x):
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)
        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up2(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up2(c)
        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        out = self.up_last(x)
        return self.activation(out)


class Conv2d1x1(nn.Module):
    """FG fair_df_detector.py:557-568."""

    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1), nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1), nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, out_f, 1, 1))

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    """FG fair_df_detector.py:571-585 (dropout 0.2 on the logits, as released). Run in float32."""

    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim), nn.LeakyReLU(inplace=True), nn.Linear(hidden_dim, out_f))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            bs = x.size()[0]
            x_feat = self.pool(x).view(bs, -1)
            x = self.mlp(x_feat)
            x = self.do(x)
        return x, x_feat


def adjust_channel(in_f, out_f=512):
    """Analog of the Xception adjust_channel neck (FG networks/xception.py:172-176: Conv1x1 2048->512, BN, ReLU)
    that turns each Xception into a 512-channel 'encoder' in PG-FDD (fair_df_detector.py:107-122, mode
    adjust_channel). Here: Conv1x1 1024->512, BN, ReLU on the 16x16 CLIP token map."""
    return nn.Sequential(nn.Conv2d(in_f, out_f, 1, 1), nn.BatchNorm2d(out_f), nn.ReLU(inplace=True))


class PGFDDDetector(_Base):
    """FG fair_df_detector.py:34-105 (module layout) and :336-431 (forward). CLIP adaptation: the three Xception
    encoders (encoder_f / encoder_c / encoder_fair, :44-46) become three adjust_channel necks on ONE shared adapted
    CLIP trunk (three ViT-L/14 passes of 176 images do not fit one 80 GB GPU)."""

    def __init__(self, cfg):
        super().__init__()
        p = cfg["clipfair"]["pgfdd"]
        self.backbone = build_cue_backbone(cfg)
        D = self.backbone.feat_dim
        self.grid = self.backbone.grid
        self.num_classes = 2
        self.encoder_feat_dim = 512                                  # :40
        self.half_fingerprint_dim = self.encoder_feat_dim // 2       # :41
        e, h = self.encoder_feat_dim, self.half_fingerprint_dim
        self.encoder_f, self.encoder_c, self.encoder_fair = adjust_channel(D, e), adjust_channel(D, e), adjust_channel(D, e)
        self.con_gan = ConditionalUNet(cfg["data"]["input_size"])    # :58
        self.adain = AdaIN()                                         # :59
        self.head_spe = Head(h, e, int(p.get("specific_task_number", 6)))   # :63, :66-70
        self.head_sha = Head(h, e, self.num_classes)                 # :71-75
        self.head_fair = Head(h, e, int(p.get("fair_task_number", 8)))      # :64, :76-80
        self.head_fused = Head(h, e, self.num_classes)               # :81-85
        self.block_spe = Conv2d1x1(e, h, h)                          # :86-90
        self.block_sha = Conv2d1x1(e, h, h)                          # :91-95
        self.block_fair = Conv2d1x1(e, h, h)                         # :96-100
        self.block_fused = Conv2d1x1(e, h, h)                        # :101-105 (constructed, never called, as released)

    def encode(self, x01):
        feat, _ = self.backbone(x01)                                 # (N,G,G,D)
        fmap = feat.permute(0, 3, 1, 2).contiguous()                 # (N,D,G,G): token map as a conv feature map
        return self.encoder_f(fmap), self.encoder_c(fmap), self.encoder_fair(fmap)

    def forward(self, x01):
        """Every head on every image (:336-346, :397-400). The reconstruction branch is in clipfair_engine (it needs the
        [real; fake] halves). Inference (eval) = the fused head (:348-381)."""
        f, c, d = self.encode(x01)
        f_spe, f_share = self.block_spe(f), self.block_sha(f)         # classifier(), :161-166
        f_fair = self.block_fair(d)
        fused = self.adain(f_fair, f_share)                          # :346
        out_fused, fused_feat = self.head_fused(fused)
        s = out_fused[:, 1] - out_fused[:, 0]
        if not self.training:
            return dict(score=s, patch=None, s_cls=s)
        out_spe, spe_feat = self.head_spe(f_spe)
        out_sha, sha_feat = self.head_sha(f_share)
        out_fair, _ = self.head_fair(f_fair)
        return dict(score=s, cls=out_sha, feat=sha_feat, cls_spe=out_spe, feat_spe=spe_feat, cls_fair=out_fair,
                    cls_fused=out_fused, f_all=torch.cat((f_spe, f_share), dim=1), content=c, fair_map=d,
                    patch=None, s_cls=s)

    def reconstruct(self, out, real_idx, fake_idx):
        """:384-394 with the official chunk convention: the batch is [real; fake], f2/c2/d2 = real half, f1/c1/d1 = fake
        half. Returns (rec_1, rec_2, self_rec_1, self_rec_2) = (G(f1, cd2), G(f2, cd1), G(f1, cd1), G(f2, cd2))."""
        f_all, c, d = out["f_all"], out["content"], out["fair_map"]
        f2, f1 = f_all[real_idx], f_all[fake_idx]
        c2, c1 = c[real_idx], c[fake_idx]
        d2, d1 = d[real_idx], d[fake_idx]
        cd1 = self.adain(d1, c1)
        cd2 = self.adain(d2, c2)
        self_rec_1 = self.con_gan(f1, cd1)
        self_rec_2 = self.con_gan(f2, cd2)
        rec_1 = self.con_gan(f1, cd2)
        rec_2 = self.con_gan(f2, cd1)
        return rec_1, rec_2, self_rec_1, self_rec_2


METHODS = {"daw": LinearHeadDetector, "dag": LinearHeadDetector, "ding": DingDetector, "pgfdd": PGFDDDetector}


def load_p1_backbone(model, path):
    """CFES checkpoint -> backbone (strict on the backbone; same mapping as ces_model.load_cfes_backbone).
    Returns the checkpoint's model state dict (for Ding's head)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {"net." + k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"init_backbone_ckpt {path}: missing {missing[:5]} unexpected {unexpected[:5]}")
    return ck["model"], len(sd)


def build_model(cfg, device):
    """Model of cfg.clipfair.method. model.init_backbone_ckpt (p1) -> backbone; Ding also: head <- p1 cls_head and
    SFD channels <- clipfair.ding.channels_json. model.init_ckpt (a CLIPFAIR checkpoint) -> whole model, strict."""
    import json
    method = cfg["clipfair"]["method"]
    model = METHODS[method](cfg)
    m = cfg["model"]
    model.init_info = {}
    if m.get("init_backbone_ckpt"):
        sd, n = load_p1_backbone(model, m["init_backbone_ckpt"])
        model.init_info["backbone_keys"] = n
        if method == "ding":
            if "cls_head.weight" not in sd:
                raise RuntimeError("Ding: p1 checkpoint has no cls_head")
            model.init_head_from_logit(sd["cls_head.weight"][0].float(), sd["cls_head.bias"][0].float())
            model.init_info["head"] = "p1 cls_head (2-way equivalent)"
    if method == "ding":
        cj = cfg["clipfair"]["ding"].get("channels_json")
        if cj:
            model.set_sfd_channels(json.load(open(cj))["channels"])
            model.init_info["sfd_channels"] = model.sfd_channels
    if m.get("init_ckpt"):
        ck = torch.load(m["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=bool(ck.get("full", True)))
    return model.to(device)
