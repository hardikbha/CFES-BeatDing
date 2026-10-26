"""Reference side of tests/bd_checks.py x1 / x2, run under Ding's own code env (qwen38; it can import Ding's `detectors`
package). CPU only. Compares, IN THIS ENV, Ding's own XceptionDetector backbone with our vendored cfes/ding_xception.py
(imported by file path) and writes Ding's transform outputs and forward outputs for the pulid side to compare against.

  qwen38 python tests/x_ding_ref.py <paths.json> <out.npz>
paths.json: {"train": [16 paths], "test": [16 paths], "fwd": [8 paths], "ckpt": DING_base_s7/49.pth}
Prints one JSON line (REF ...)."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

DING = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/code/Ding_mod"
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/xception-b5690688.pth"


def main():
    paths, out = json.load(open(sys.argv[1])), sys.argv[2]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.chdir(DING)
    sys.path.insert(0, DING)
    from detectors import DETECTOR                                   # Ding's registry (needs Ding's env)
    from transform import fair_df_default_data_transforms, resnet_default_data_transforms
    spec = importlib.util.spec_from_file_location("ding_xception_vendored", os.path.join(A, "cfes", "ding_xception.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    rep = {"torch": torch.__version__}
    # transforms exactly as Ding's datasets apply them: PIL open + convert('RGB') + transform
    tr = torch.stack([resnet_default_data_transforms["train"](Image.open(p).convert("RGB")) for p in paths["train"]])
    te = torch.stack([fair_df_default_data_transforms["test"](Image.open(p).convert("RGB")) for p in paths["test"]])
    fx = torch.stack([resnet_default_data_transforms["test"](Image.open(p).convert("RGB")) for p in paths["fwd"]])
    # (a) ImageNet construction: Ding's XceptionDetector() vs vendored Xception + load_imagenet
    torch.manual_seed(0)
    det = DETECTOR["XceptionDetector"]()
    torch.manual_seed(0)
    net = V.build_ding_xception()
    missing = V.load_imagenet(net, IMAGENET)
    sd_d, sd_v = det.backbone.state_dict(), net.state_dict()
    rep["keys_equal"] = list(sd_d.keys()) == list(sd_v.keys())
    rep["shapes_equal"] = all(sd_d[k].shape == sd_v[k].shape for k in sd_d)
    loaded = [k for k in sd_d if not k.startswith(("adjust_channel.", "last_linear."))]
    rep["imagenet_tensors_equal"] = all(torch.equal(sd_d[k], sd_v[k]) for k in loaded)
    rep["imagenet_missing_keys"] = sorted(missing)
    rep["n_state"] = len(sd_d)
    # (b) DING_base load: Ding's detector (strict, as model_xception.py does) vs vendored load_detector
    ck = torch.load(paths["ckpt"], map_location="cpu", weights_only=True)
    det.load_state_dict(ck, strict=True)
    V.load_detector(net, paths["ckpt"])
    rep["ding_base_tensors_equal"] = all(torch.equal(v, net.state_dict()[k]) for k, v in det.backbone.state_dict().items())
    # (c) forward, eval mode, CPU: Ding's own module vs vendored, same env -> must be bit-identical
    det.eval(); net.eval()
    with torch.no_grad():
        ld = det({"image": fx})["cls"]
        fd = det.backbone.features(fx)
        lv, fv = net(fx)
    rep["fwd_same_env_logits_max_abs"] = float((ld - lv).abs().max())
    rep["fwd_same_env_feat_max_abs"] = float((fd - fv).abs().max())
    rep["fwd_same_env_bitwise"] = bool(torch.equal(ld, lv) and torch.equal(fd, fv))
    np.savez(out, train_tf=tr.numpy(), test_tf=te.numpy(), fwd_in=fx.numpy(), ding_logits=ld.numpy(),
             ding_feat=fd.numpy())
    print("REF " + json.dumps(rep))


if __name__ == "__main__":
    main()
