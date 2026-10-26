"""Validation pass of the STARTING model: per-cue utilities on the validation split -> validated-cue JSON for L_sens.

  python scripts/ces_validate_cues.py --config configs/ffpp/B0.yaml --ckpt runs/ffpp_B0_s7/best.pt \
         --out runs/ffpp_B0_s7/validated_cues.json [--max_items 4000]

Utilities use the training margin and the training replacement rule with the EMA baselines stored in the checkpoint's
CES state (no pairs on validation): reals -> fake class mean; fakes -> real-cue mean of their own group
(ces.fake_baseline group; class mean for a group without an estimate). If the checkpoint has no initialised baselines, the class means of the validation cues themselves
are used and the JSON says so. The decision rule is cfes/ces_validation.decide_validated (declared there).
Not run yet: needs a trained B0 checkpoint (GPU).
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import ces_engine as CE  # noqa: E402
from cfes import ces_losses as CL  # noqa: E402
from cfes import engine as E  # noqa: E402
from cfes.ces_data import condition_codes  # noqa: E402
from cfes.ces_validation import decide_validated, decide_validated_per_family  # noqa: E402
from cfes.config import load_config  # noqa: E402


@torch.no_grad()
def collect(model, loader, device, amp):
    Cs, ss, ys, gs = [], [], [], []
    model.eval()
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(b["image"].to(device))
        Cs.append(out["cues"].float().cpu()); ss.append(out["score"].float().cpu()); ys.append(b["label"])
        gs.append(b["group"])
    return torch.cat(Cs), torch.cat(ss), torch.cat(ys), torch.cat(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, list(a.set) + [f"model.init_ckpt={a.ckpt}", "model.init_backbone_ckpt=null"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, st = CE.build_model(cfg, device)
    state = CE.build_state(cfg, model, device, st).cpu()
    df, loader = E.eval_loader(cfg["data"]["val"], cfg, a.max_items)
    C, s, y, g = collect(model.to(device), loader, device, bool(cfg["optim"]["amp"]))
    model.cpu()
    src = "ckpt_ema"
    if not state.ready():
        for c in (0, 1):
            state.cls_mean[c] = C[y == c].mean(0)
        state.cls_init[:] = True
        src = "validation_class_means"
    with torch.no_grad():
        fb = cfg.get("ces", {}).get("fake_baseline", "group")
        R = CL.replacement(C, y, state, None, "counterfactual", g=g.long(), fake_baseline=fb)
        U, _ = CL.utilities(model, C, s, y, R, cfg.get("ces", {}))
    vcfg = cfg.get("ces", {}).get("validation", {})
    res = decide_validated(U.numpy(), y.numpy(), vcfg)
    if vcfg.get("per_family", False):          # Amendment 2 (FairForge-2 only): per-family masks / targets
        cond = condition_codes(df, cfg)
        assert len(cond) == len(y)
        res["per_family"] = decide_validated_per_family(U.numpy(), y.numpy(), cond, list(cfg["data"]["families"]),
                                                        vcfg, res)
    res.update(ckpt=a.ckpt, n_val=int(len(y)), baseline_source=src, margin=cfg.get("ces", {}).get("margin", "asinh"),
               fake_baseline=fb, fake_group_baseline_groups=[int(x) for x in torch.nonzero(state.grp_init[0]).flatten()])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("valid", "target")}))


if __name__ == "__main__":
    main()
