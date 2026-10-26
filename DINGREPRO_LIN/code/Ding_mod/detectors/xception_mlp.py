# detectors/xception_detector_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR
from .base_detector import AbstractDetector
from loss import LOSSFUNC

from utils.xception import Xception



@DETECTOR.register_module(module_name='xception_mlp')
class xception_mlp(AbstractDetector):
    """
    在原 XceptionDetector 基础上：backbone.features -> GAP -> MLP -> Head
    - 旧权重仅命中 backbone；MLP 与 Head 随机初始化
    - forward 不再调用 backbone.classifier，而是使用自己的 MLP+Head
    """

    def __init__(
        self,
        mlp_hidden=(1024, 512),      # 你可以按需改层数/维度
        mlp_dropout=0.1,
        num_classes=2,
        backbone_mode='adjust_channel',    # 'original' 或 'adjust_channel'
        inc=3,
        dropout=False
    ):
        super().__init__()
        self.backbone = self.build_backbone(
            mode=backbone_mode, inc=inc, dropout=dropout, num_classes=num_classes
        )
        self.loss_func = self.build_loss()

        # 推断特征维度（对 features 输出做 GAP+flatten 后的维度）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)  # 与训练预处理尺寸一致
            feat_map = self.backbone.features(dummy)
            vec = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(1, -1)
            in_dim = vec.shape[1]

        # 构建 MLP（多层）
        layers = []
        last = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h, bias=True), nn.ReLU(inplace=True)]
            if mlp_dropout and mlp_dropout > 0:
                layers += [nn.Dropout(p=mlp_dropout)]
            last = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # 新的分类头（不复用 backbone.last_linear）
        self.head = nn.Linear(last, num_classes, bias=True)

    # ---------- 复用原 detector 的套路 ----------
    def build_backbone(self, mode='original', inc=3, dropout=False, num_classes=2):
        # 与你现有 XceptionDetector.build_backbone 一致
        config = {'mode': mode, 'num_classes': num_classes, 'inc': inc, 'dropout': dropout}
        backbone = Xception(config)

        # 载入你项目里的 xception 预训练权重（保持原有的 pointwise 维度修正）
        state_dict = torch.load('/home/user/local/zhouyunpeng/SinkhornWithPrune/pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, strict=False)
        print('[XceptionDetector_MLP] Load pretrained backbone successfully.')
        return backbone

    def build_loss(self):
        loss_class = LOSSFUNC['cross_entropy']
        return loss_class()

    # ---------- 与原 detector 的接口保持一致 ----------
    def features(self, data_dict: dict) -> torch.Tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, feat_map: torch.Tensor) -> torch.Tensor:
        # GAP -> flatten -> MLP -> 新的 Head
        x = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
        z = self.mlp(x)
        logits = self.head(z)
        return logits

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        feat_map = self.features(data_dict)
        pred = self.classifier(feat_map)
        return {'cls': pred}

    # 供训练脚本调用的冻结函数（只训 MLP+Head）
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
