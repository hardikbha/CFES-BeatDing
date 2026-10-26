"""A1 operating point of a CLIPFAIR run from its saved scores (CPU): the A1 part of scripts/ces_posteval.py, unchanged
(fair_a1.py = the CFES fair_oppoint rule, verbatim): t chosen on FF++ validation only (min val intersectional F_FPR
s.t. val TPR >= 80%, FPR <= 20%); test metrics at t, plus the reference points t = 0 (raw 0.5) and t = -b
(validation-calibrated common intercept, calibrate.py). No per-cue analysis (the baselines have no cue head).

  python scripts/clipfair_posteval.py --run runs/ffpp_daw_s7      -> <run>/eval/a1.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import fair_a1 as A  # noqa: E402
from cfes.config import load_config  # noqa: E402


def ld(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval_dir", default=None)
    a = ap.parse_args()
    ev = a.eval_dir or os.path.join(a.run, "eval")
    cfg = load_config(os.path.join(a.run, "config.yaml"))
    V = ld(os.path.join(ev, "val_scores.npz"))
    T = {k: ld(os.path.join(ev, f"{k}_scores.npz")) for k in cfg["eval"]["test_sets"]
         if os.path.exists(os.path.join(ev, f"{k}_scores.npz"))}
    schemes = {k: s.get("scheme", "eight") for k, s in cfg["eval"]["test_sets"].items()}
    tri = lambda d: (d["logit"].astype(np.float64), d["label"].astype(np.int64), d["group"].astype(np.int64))
    rec = A.a1_for(tri(V), {k: tri(d) for k, d in T.items()}, schemes)      # every configured test set (F7)
    cal = os.path.join(ev, "calibration.json")
    b_int = json.load(open(cal))["intercept_b"] if os.path.exists(cal) else None
    ref = {"raw0": 0.0}
    if b_int is not None:
        ref["calibrated"] = -float(b_int)
    rec["reference"] = {}
    for nm, t in ref.items():
        rec["reference"][nm] = {"t": t, "val": A.full_metrics(*tri(V), "eight", t),
                                "test": {k: A.full_metrics(*tri(d), schemes[k], t) for k, d in T.items()}}
    tmp = os.path.join(ev, "a1.json.tmp")
    json.dump(rec, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, os.path.join(ev, "a1.json"))
    print(f"A1 t={rec['t']}")


if __name__ == "__main__":
    main()
