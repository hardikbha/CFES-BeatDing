"""Score a trained CLIPFAIR baseline on every test set of its config: scripts/evaluate_ces.py (the CESel evaluator,
itself the CFES evaluate.py logic) with the CLIPFAIR model builder and no per-cue logits.

  python scripts/evaluate_clipfair.py --run runs/ffpp_daw_s7 --ckpt best.pt --leak_free [--max_items N]
  python scripts/calibrate.py --eval runs/ffpp_daw_s7/eval        # unchanged copy
  python scripts/clipfair_posteval.py --run runs/ffpp_daw_s7      # A1 threshold (fair_a1.py, verbatim CFES rule)

Outputs (as evaluate_ces.py): <eval>/val_scores.npz, <set>_scores.npz (prob, logit, label, group, method, cluster,
img_path), <variant>.json, summary.csv. Validation-calibrated threshold (TPR = eval.val_tpr) from validation only;
--leak_free adds _leakfree_video / _leakfree_identity variants; metrics from rank_prob(logit) (eval.rank_metrics).
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

from cfes import engine  # noqa: E402
from cfes.clipfair_models import build_model  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.data import ffpp_identities, read_split  # noqa: E402
from cfes.metrics import evaluate_scores, headline, rank_prob, threshold_at_tpr  # noqa: E402
from cfes.utils import load_checkpoint  # noqa: E402


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
    model = build_model(cfg, device)
    state = load_checkpoint(os.path.join(a.run, a.ckpt), model)
    model.to(device)
    amp = bool(cfg["optim"]["amp"])
    prob64 = bool(cfg["eval"].get("prob_float64", True))
    info = {"method": cfg["clipfair"]["method"], "checkpoint": a.ckpt, "checkpoint_state": state}
    if cfg["clipfair"]["method"] == "ding":
        info["sfd_channels_zeroed"] = [int(i) for i in torch.nonzero(model.keep == 0).flatten().tolist()]
    json.dump(info, open(os.path.join(out_dir, "model.json"), "w"), indent=1, default=float)

    def predict(loader):
        p, z = engine.predict(model, loader, device, amp=amp, return_logits=True, prob64=prob64)
        return p, z

    val_df, val_loader = engine.eval_loader(cfg["data"]["val"], cfg, a.max_items)
    pv, zv = predict(val_loader)
    # Codex review F4 (CESel, 25 Oct): validation-TPR threshold chosen AND applied on raw logits; ranks for AUC only
    thr = threshold_at_tpr(val_df["label"].values, np.asarray(zv, dtype=np.float64), cfg["eval"]["val_tpr"])
    np.savez(os.path.join(out_dir, "val_scores.npz"), prob=pv, logit=zv, label=val_df["label"].values,
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
        p, z = predict(loader)
        np.savez(os.path.join(out_dir, f"{name}_scores.npz"), prob=p, logit=z, label=df["label"].values,
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
                                  thr_scores=np.asarray(z, dtype=np.float64)[keep])
            res.update(checkpoint=a.ckpt, checkpoint_state=state, n_dropped=int((~keep).sum()))
            with open(os.path.join(out_dir, f"{vname}.json"), "w") as f:
                json.dump(res, f, indent=1, default=float)
            rows.append({"test_set": vname, "n": res["n"], **headline(res), "acc@0.5": res["acc@0.5"],
                         "acc@val": res["acc@val"]})
            print(f"{vname}: AUC {res['auc']:.2f}  n={res['n']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
