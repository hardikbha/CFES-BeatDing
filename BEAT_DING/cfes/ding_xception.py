"""Ding et al.'s Xception network, VENDORED for the Beat-Ding Xception arm (plan Amendment 4b; design
agents/BEAT_DING/XCEPTION_DESIGN.md section 3.2).

The classes SeparableConv2d, Block and Xception below are copied VERBATIM (byte for byte, between the
"# ---- BEGIN VERBATIM" / "# ---- END VERBATIM" markers) from Ding's official code
  agents/DINGREPRO_LIN/code/Ding_mod/utils/xception.py   (md5 d176ef63da09121bb7f5c0e02c6b0f05; lines 25-93 and 102-289)
WITHOUT its registry decorator (@BACKBONE.register_module) and its unused imports (networkx, utils.registry, model_zoo),
because the BD training env (pulid, torch 2.4.1) cannot import Ding's `detectors` package (efficientnet_pytorch is
missing). This mirrors how cfes/model_resnet.py rebuilds Ding's ResNet-50. tests/bd_checks.py x1 asserts: the copied
source is identical to Ding's, identical state_dict keys / shapes, identical ImageNet load (same pointwise patch, same
missing keys), identical DING_base load, and a bit-identical CPU forward against Ding's own module (run under qwen38).

The helpers after the END marker are NOT Ding's code: DING_CONFIG is the config XceptionDetector.build_backbone passes;
load_imagenet() replays build_backbone's weight load (incl. the DINGREPRO_LIN disclosed pointwise patch); load_detector()
loads a Ding XceptionDetector state dict ("backbone." prefix, e.g. DING_base_s7/49.pth) strictly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VERBATIM_SOURCE = ("/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/code/Ding_mod/"
                   "utils/xception.py")
VERBATIM_LINES = ((25, 93), (102, 289))      # 1-based inclusive line ranges of the copied blocks in VERBATIM_SOURCE


# ---- BEGIN VERBATIM 1 (Ding_mod/utils/xception.py lines 25-93)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

# ---- END VERBATIM 1


# ---- BEGIN VERBATIM 2 (Ding_mod/utils/xception.py lines 102-289; decorator line 101 dropped)
class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(2048, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.last_linear = nn.Linear(512, self.num_classes)
        # else:
        #     self.last_linear = nn.Linear(2048, self.num_classes)

        # dag
        self.last_linear = nn.Linear(512, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )




        # if dropout:
        #     self.last_linear = nn.Sequential(
        #         nn.Dropout(p=dropout),
        #         nn.Linear(2048, self.num_classes)
        #     )

    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            # print(x.shape)
            x = self.block12(x)
            # print(x.shape, 12)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)

        return x

    def classifier(self, features):
        # x = self.relu(features) #([128, 512, 7, 7])
        x = F.adaptive_avg_pool2d(features, (1, 1)) #([128, 512, 1, 1])
        x = x.view(x.size(0), -1) #([128, 512])
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x

# ---- END VERBATIM 2


# ------------------------------------------------------------------------------------------------ helpers (not Ding's code)
DING_CONFIG = {'mode': 'adjust_channel', 'num_classes': 2, 'inc': 3, 'dropout': False}   # XceptionDetector.build_backbone


def build_ding_xception():
    """Ding's backbone as XceptionDetector builds it, randomly initialised (no weights loaded)."""
    return Xception(DING_CONFIG)


def load_imagenet(net, path):
    """XceptionDetector.build_backbone's ImageNet load, replayed: pointwise weights unsqueezed to 4-D only when 2-D (the
    DINGREPRO_LIN disclosed patch for the substitute checkpoint), 'fc' keys dropped, strict=False, no unexpected keys.
    Returns the missing keys (expected: adjust_channel.* and last_linear.*)."""
    state_dict = torch.load(path, map_location="cpu")
    for name, weights in state_dict.items():
        if 'pointwise' in name and weights.dim() == 2:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    res = net.load_state_dict(state_dict, False)
    assert len(res.unexpected_keys) == 0, res.unexpected_keys
    return list(res.missing_keys)


def load_detector(net, path):
    """Strict load of a Ding XceptionDetector state dict (keys 'backbone.<...>'; e.g. DINGREPRO_LIN DING_base_s7/49.pth)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    bad = [k for k in sd if not k.startswith("backbone.")]
    if bad:
        raise RuntimeError(f"{path}: not an XceptionDetector state dict (keys without 'backbone.': {bad[:5]})")
    net.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()}, strict=True)
    return net
