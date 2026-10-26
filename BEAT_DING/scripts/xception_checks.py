"""Beat-Ding Xception run checks (plan Amendment 4b; design 3.3-3.4). Used by run/bd_run.sh xception_smoke (GPU) and by
tests/bd_checks.py x3 / x4 (CPU). Prints one JSON line; exit 1 if a check fails.

  python scripts/xception_checks.py --run runs/X_base_s7 [--n 32] [--out file.json]

Checks (validation images only; nothing is trained):
  start_equivalence : the step-0 model (recal.pt) in eval mode vs Ding's own network (the vendored Xception holding the
                      SAME net.* state, i.e. Ding's model + the same step-0 BN statistics): max |score - (l1 - l0)| == 0
                      (patch head last Linear zero-initialised, C gate 0);
  bn_frozen         : every BatchNorm buffer (running_mean / running_var / num_batches_tracked) and every frozen net.*
                      parameter is bit-identical between recal.pt and last.pt (BN statistics never change after step 0);
  trunk_unchanged   : every net.* parameter except last_linear in last.pt is bit-identical to the init_from checkpoint
                      (Ding's DING_base), and every BN running statistic differs from it only through step 0;
  no_mode_mismatch  : on a clean batch, the last.pt model's scores in train mode equal its eval-mode scores (|d| <= 1e-5)
                      (frozen BN statistics are used in both modes; no dropout).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import engine  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.data import load_image, read_split  # noqa: E402
from cfes.ding_xception import build_ding_xception  # noqa: E402


def clean_batch(cfg, n):
    df = read_split(cfg["data"]["val"])
    import pandas as pd
    sel = pd.concat([df[df.label == k].sample(min(int((df.label == k).sum()), n // 2), random_state=0) for k in (0, 1)])
    d = cfg["data"]
    return torch.stack([load_image(p, d.get("eval_size", d["input_size"]), d.get("resample", "bicubic"))
                        for p in sel.img_path])


def load(run, cfg, ckpt, device):
    m = engine.build_model(cfg, device)
    ck = torch.load(os.path.join(run, ckpt), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    return m.to(device), ck


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(os.path.join(a.run, "config.yaml"))
    if cfg["model"].get("backend") != "xception_bd":
        raise SystemExit("xception_checks: not an xception_bd run")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    x = clean_batch(cfg, a.n).to(dev)
    rep, ok = {"run": a.run, "device": str(dev), "n_images": int(x.shape[0])}, True
    # start equivalence: recal.pt vs Ding's network with the same net.* state
    m0, ck0 = load(a.run, cfg, "recal.pt", dev)
    m0.eval()
    ding = build_ding_xception()
    ding.load_state_dict({k[4:]: v for k, v in ck0["model"].items() if k.startswith("net.")}, strict=True)
    ding = ding.to(dev).eval()
    s_model = m0(x)["score"].double()
    out, _ = ding((x - 0.5) / 0.5)
    s_ding = (out[:, 1] - out[:, 0]).double()
    d0 = float((s_model - s_ding).abs().max())
    rep["start_equivalence_max_abs"] = d0
    rep["bn_recal"] = (ck0.get("state") or {}).get("bn_recal")
    ok &= d0 == 0.0
    if os.path.exists(os.path.join(a.run, "last.pt")):
        ck1 = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=False)
        sd0, sd1 = ck0["model"], ck1["model"]
        bn_keys = [k for k in sd0 if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        trainable = {n for n, p in m0.named_parameters() if p.requires_grad}
        frozen = [k for k in sd0 if k.startswith("net.") and k not in trainable and k not in bn_keys]
        bn_changed = [k for k in bn_keys if not torch.equal(sd0[k], sd1[k])]
        fr_changed = [k for k in frozen if not torch.equal(sd0[k], sd1[k])]
        rep.update(bn_buffers=len(bn_keys), bn_buffers_changed=bn_changed[:10], frozen_params=len(frozen),
                   frozen_params_changed=fr_changed[:10],
                   trained_tensors_changed=sorted(k for k in trainable if not torch.equal(sd0[k], sd1[k])))
        ok &= not bn_changed and not fr_changed
        init = cfg["model"]["xception"].get("init_from")
        if init:
            di = torch.load(init, map_location="cpu", weights_only=True)
            di = {"net." + k[len("backbone."):]: v for k, v in di.items()}
            tr_diff = [k for k in frozen if not torch.equal(di[k], sd1[k])]
            rep["trunk_params_changed_vs_init_from"] = tr_diff[:10]
            rep["bn_stats_changed_by_step0"] = sum(not torch.equal(di[k], sd0[k]) for k in bn_keys)
            ok &= not tr_diff
        m1, _ = load(a.run, cfg, "last.pt", dev)
        m1.eval()
        se = m1(x)["score"].double()
        m1.train()
        st = m1(x)["score"].double()
        dm = float((se - st).abs().max())
        rep["train_vs_eval_mode_max_abs"] = dm
        rep["score_absmax_last"] = float(se.abs().max())
        ok &= dm <= 1e-5
    rep["ok"] = bool(ok)
    line = json.dumps(rep, default=float)
    print("XCHECK " + line)
    if a.out:           # the marker file only when every check passed (a failed check never looks done)
        dst = a.out if ok else a.out + ".failed"
        tmp = dst + ".tmp"
        open(tmp, "w").write(line + "\n")
        os.replace(tmp, dst)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
