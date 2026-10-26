"""[CESEL_MA copy of agents/CFES/scripts/evaluate.py, unchanged except that it imports the CESEL_MA cfes package (Codex F8 cluster keys); used for the FF2-public CFES-D run.]
Score a trained run on every test set in its config and write metrics.

  python scripts/evaluate.py --run runs/ff2_cfes_s7                  # best.pt, all cfg.eval.test_sets
  python scripts/evaluate.py --run runs/ff2_cfes_s7 --ckpt last.pt --only ffpp_test dfdc

Outputs in <run>/eval/: <name>_scores.npz (probabilities, labels, groups, methods, clusters), <name>.json (all
metrics), summary.csv (one row per test set, Ding Table-2 column order). The validation-calibrated threshold
(TPR = cfg.eval.val_tpr on the validation split) is recomputed from the loaded checkpoint, never from test data.
With --leak_free, two extra variants are reported: _leakfree_video drops test frames whose video key occurs in
training (Lin et al.'s FF++ split shares 174 videos), _leakfree_identity drops FF++ test frames involving any
identity seen in training (68% of FF++ test frames; see agents/DATA_AUDIT).
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
    ap.add_argument("--max_items", type=int, default=None, help="debug: subsample each test set")
    ap.add_argument("--extra_set", nargs="*", default=[],
                    help="extra test sets NAME=CSV[:SCHEME[:META]] (e.g. FairForge-2 TEST); use with --out_dir")
    ap.add_argument("--out_dir", default=None, help="write to this eval dir instead of eval/ or eval_<ckpt>/")
    ap.add_argument("--from_scores", action="store_true",
                    help="recompute metrics from the saved <set>_scores.npz logits (no inference)")
    a = ap.parse_args()

    cfg = load_config(os.path.join(a.run, "config.yaml"))
    out_dir = a.out_dir or os.path.join(a.run, "eval" if a.ckpt == "best.pt" else f"eval_{a.ckpt.split('.')[0]}")
    if a.extra_set:
        if not a.out_dir:
            raise SystemExit("--extra_set needs --out_dir (keeps the main eval directory untouched)")
        sets = {}
        for item in a.extra_set:
            name, spec = item.split("=", 1)
            parts = spec.split(":")
            sets[name] = {"csv": parts[0], "scheme": parts[1] if len(parts) > 1 else "eight",
                          **({"meta": parts[2]} if len(parts) > 2 else {})}
        cfg["eval"]["test_sets"] = sets
    os.makedirs(out_dir, exist_ok=True)
    rank = bool(cfg["eval"].get("rank_metrics"))     # metrics from rank_prob(logit): exact AUC, same 0.5 decisions
    state = None
    if not a.from_scores:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = engine.build_model(cfg, device)
        state = load_checkpoint(os.path.join(a.run, a.ckpt), model)
        amp = bool(cfg["optim"]["amp"])

    val_df, val_loader = engine.eval_loader(cfg["data"]["val"], cfg, a.max_items)
    if a.from_scores:
        zz = np.load(os.path.join(out_dir, "val_scores.npz"), allow_pickle=True)
        pv, zv = zz["prob"], zz["logit"]
    else:
        pv, zv = engine.predict(model, val_loader, device, amp, return_logits=True,
                                prob64=bool(cfg["eval"].get("prob_float64")))
    thr = threshold_at_tpr(val_df["label"].values, rank_prob(zv) if rank else pv, cfg["eval"]["val_tpr"])
    if not a.from_scores:
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
        if a.from_scores:
            zz = np.load(os.path.join(out_dir, f"{name}_scores.npz"), allow_pickle=True)
            assert (zz["img_path"] == df["img_path"].values.astype(str)).all(), f"{name}: saved rows differ"
            p, z = zz["prob"], zz["logit"]
        else:
            p, z = engine.predict(model, loader, device, amp, return_logits=True,
                                  prob64=bool(cfg["eval"].get("prob_float64")))
            np.savez(os.path.join(out_dir, f"{name}_scores.npz"), prob=p, logit=z, label=df["label"].values,
                     group=df["group"].values, method=df["method"].values.astype(str),
                     cluster=df["cluster"].values.astype(str), img_path=df["img_path"].values.astype(str))
        pm = rank_prob(z) if rank else p
        variants = {name: np.ones(len(df), bool)}
        if a.leak_free:
            variants[f"{name}_leakfree_video"] = ~df["cluster"].isin(train_keys).values
            ids = df["img_path"].map(ffpp_identities)
            if ids.map(len).sum() > 0:      # FF++-style names only
                variants[f"{name}_leakfree_identity"] = ~ids.map(lambda s: bool(s & train_ids)).values
        for vname, keep in variants.items():
            res = evaluate_scores(df["label"].values[keep], pm[keep], df["group"].values[keep],
                                  spec.get("scheme", "eight"), methods=df["method"].values[keep], val_threshold=thr)
            res.update(checkpoint=a.ckpt, checkpoint_state=state, n_dropped=int((~keep).sum()))
            with open(os.path.join(out_dir, f"{vname}.json"), "w") as f:
                json.dump(res, f, indent=1, default=float)
            rows.append({"test_set": vname, "n": res["n"], **headline(res), "acc@0.5": res["acc@0.5"],
                         "acc@val": res["acc@val"]})
            print(f"{vname}: AUC {res['auc']:.2f}  n={res['n']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
