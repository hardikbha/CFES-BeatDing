import sys, torch
sys.path.insert(0, "/home/sachin.chaudhary/ICLR_2027/related_work/Ding_Fairness-Optimization")
from utils.xception import Xception
config = {'mode': 'adjust_channel', 'num_classes': 2, 'inc': 3, 'dropout': False}
backbone = Xception(config)
sd = torch.load("/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/xception-b5690688.pth", map_location='cpu')
print("ckpt keys sample:", list(sd.keys())[:5])
print("pointwise already 4D:", sd[[k for k in sd if 'pointwise' in k][0]].dim())
for name, w in sd.items():
    if 'pointwise' in name and w.dim() == 2:
        sd[name] = w.unsqueeze(-1).unsqueeze(-1)
sd = {k:v for k,v in sd.items() if 'fc' not in k}
res = backbone.load_state_dict(sd, False)
print("MISSING:", res.missing_keys)
print("UNEXPECTED:", res.unexpected_keys)
x = torch.randn(2,3,256,256)
out = backbone.features({'image': x} if False else x)
print("feat shape:", out.shape if hasattr(out,'shape') else type(out))
