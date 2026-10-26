"""Score a trained CESel run on every test set of its config, exactly like CFES scripts/evaluate.py, plus the per-cue
logits of the additive head.

  python scripts/evaluate_ces.py --run runs/ffpp_B0_s7 --ckpt best.pt --leak_free [--max_items N]
  python scripts/calibrate.py --eval runs/ffpp_B0_s7/eval            # unchanged copy of CFES calibrate.py

Same logic and outputs as CFES/scripts/evaluate.py (read only; this is the CESel port): <run>/eval/<name>_scores.npz,
<name>.json, summary.csv; validation-calibrated threshold (TPR = eval.val_tpr) recomputed from the loaded checkpoint,
never from test data; --leak_free adds the _leakfree_video and (FF++-style names) _leakfree_identity variants;
metrics from rank_prob(logit) when eval.rank_metrics (CFES-D setting). Differences, all additive:
  * the model is the CESDetector (cfes.ces_engine.build_model), loaded from <run>/<ckpt> (CES checkpoint);
  * every *_scores.npz (val included) also holds s_k (N,4) = per-cue logits w_k^T c_k in CUES order (tex, freq,
    bound, sem); s = sum_k s_k + b (b in <eval>/head.json) -- used by scripts/ces_posteval.py;
  * the CES state (EMA means, probe) is stripped from checkpoint_state in the json files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import ces_engine as CE  # noqa: E402
from cfes import engine  # noqa: E402
from cfes.ces_model import CUES  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.data import ffpp_identities, read_split  # noqa: E402
from cfes.metrics import evaluate_scores, headline, rank_prob, threshold_at_tpr  # noqa: E402
from cfes.utils import load_checkpoint  # noqa: E402


@torch.no_grad()
def predict(model, loader, device, amp, prob64=True):
    model.eval()
    zs, sks = [], []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(b["image"].to(device, non_blocking=True))
        zs.append(out["score"].float().cpu().numpy())
        sks.append(out["s_k"].float().cpu().numpy() if out.get("s_k") is not None
                   else np.full((len(zs[-1]), len(CUES)), np.nan, np.float32))
    z = np.concatenate(zs) if zs else np.zeros(0, np.float32)
    sk = np.concatenate(sks) if sks else np.zeros((0, len(CUES)), np.float32)
    p = 1.0 / (1.0 + np.exp(-z.astype(np.float64))) if prob64 else 1.0 / (1.0 + np.exp(-z))
    return p, z, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--leak_free", action="store_true")
    ap.add_argument("--max_items", type=int, default=None, help="debug: subsample each set (label-stratified)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(os.path.join(a.run, "config.yaml"), ["model.init_ckpt=null", "model.init_backbone_ckpt=null"])
    out_dir = a.out_dir or os.path.join(a.run, "eval" if a.ckpt == "best.pt" else f"eval_{a.ckpt.split('.')[0]}")
    os.makedirs(out_dir, exist_ok=True)
    rank = bool(cfg["eval"].get("rank_metrics"))
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = CE.build_model(cfg, device)
    state = load_checkpoint(os.path.join(a.run, a.ckpt), model)
    state = {k: v for k, v in (state or {}).items() if k not in ("ces_state", "fprgap_buf")}
    model.to(device)
    amp = bool(cfg["optim"]["amp"])
    prob64 = bool(cfg["eval"].get("prob_float64", True))
    head = {"cues": list(CUES), "kind": model.head.kind,
            "b": float(model.head.b) if model.head.kind == "additive" else None, "checkpoint": a.ckpt,
            "checkpoint_state": state}
    json.dump(head, open(os.path.join(out_dir, "head.json"), "w"), indent=1, default=float)

    val_df, val_loader = engine.eval_loader(cfg["data"]["val"], cfg, a.max_items)
    pv, zv, skv = predict(model, val_loader, device, amp, prob64)
    # Codex review F4: the validation-TPR threshold is chosen AND applied on raw logits (one fixed logit threshold for
    # every test set); rank_prob is per-dataset normalised, so it is used for AUC only.
    thr = threshold_at_tpr(val_df["label"].values, zv.astype(np.float64), cfg["eval"]["val_tpr"])
    np.savez(os.path.join(out_dir, "val_scores.npz"), prob=pv, logit=zv, s_k=skv, label=val_df["label"].values,
             group=val_df["group"].values, cluster=val_df["cluster"].values.astype(str),
             img_path=val_df["img_path"].values.astype(str))
    train_keys, train_ids = set(), set()
    if a.leak_free:
        tr = read_split(cfg["data"]["train"])
        train_keys = set(tr["cluster"])
        train_ids = set().union(*tr["img_path"].map(ffpp_identities))

    rows = []
    for name, spec in cfg["eval"]["test_sets"].items():
        if a.only and name not in a.only:
            continue
        df, loader = engine.eval_loader(spec, cfg, a.max_items)
        p, z, sk = predict(model, loader, device, amp, prob64)
        np.savez(os.path.join(out_dir, f"{name}_scores.npz"), prob=p, logit=z, s_k=sk, label=df["label"].values,
                 group=df["group"].values, method=df["method"].values.astype(str),
                 cluster=df["cluster"].values.astype(str), img_path=df["img_path"].values.astype(str))
        pm = rank_prob(z) if rank else p
        variants = {name: np.ones(len(df), bool)}
        if a.leak_free:
            variants[f"{name}_leakfree_video"] = ~df["cluster"].isin(train_keys).values
            ids = df["img_path"].map(ffpp_identities)
            if ids.map(len).sum() > 0:
                variants[f"{name}_leakfree_identity"] = ~ids.map(lambda s: bool(s & train_ids)).values
        for vname, keep in variants.items():
            res = evaluate_scores(df["label"].values[keep], pm[keep], df["group"].values[keep],
                                  spec.get("scheme", "eight"), methods=df["method"].values[keep], val_threshold=thr,
                                  thr_scores=z.astype(np.float64)[keep])
            res.update(checkpoint=a.ckpt, checkpoint_state=state, n_dropped=int((~keep).sum()))
            with open(os.path.join(out_dir, f"{vname}.json"), "w") as f:
                json.dump(res, f, indent=1, default=float)
            rows.append({"test_set": vname, "n": res["n"], **headline(res), "acc@0.5": res["acc@0.5"],
                         "acc@val": res["acc@val"]})
            print(f"{vname}: AUC {res['auc']:.2f}  n={res['n']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
