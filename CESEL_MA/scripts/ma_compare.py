"""Step-2 check of the final-model spec: CESel-MA seed 7 must NOT be worse than the single-axis CESel (the screen's
winning arm, gate.json single_axis_run) on the validation selection value. Validation only (Amendment 6):

  value = the CESel-MA selection value (val_oppoint_ma: val AUC - 0.5 x max over the gender / age-bin / ITA-bin /
          gender x race FPR gaps, groups with >= 2 real validation videos, at the fitted common intercept; eligibility =
          pooled TPR >= 80 / FPR <= 20 and each run's own start-checkpoint reference checks, TRAINED["select_ref"]),
          recomputed for BOTH runs from their saved validation logits (<run>/eval/val_scores.npz of best.pt) with the
          same code and the same labels (the MA config's validation label file).
  not_worse = value(MA) >= value(single-axis).
The single-axis (gender x race) value of both is reported alongside (descriptive). Output: results/ma_vs_single_s<seed>.json.

  ~/.conda/envs/pulid/bin/python scripts/ma_compare.py [--out_root <CESEL_MA root>] [--seed 7]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import train as T  # noqa: E402
from cfes import ma_labels as MAL  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.ma_losses import ma_gaps  # noqa: E402
from cfes.metrics import evaluate_scores, rank_prob  # noqa: E402


def score_run(run, ma_cfg):
    z = np.load(os.path.join(run, "eval", "val_scores.npz"), allow_pickle=True)
    y, logit, g = z["label"].astype(int), z["logit"].astype(np.float64), z["group"].astype(int)
    df = pd.DataFrame({"img_path": z["img_path"].astype(str), "label": y, "group": g, "cluster": z["cluster"].astype(str)})
    df, rep = MAL.attach(df, ma_cfg, "val")
    res = evaluate_scores(y, rank_prob(logit), g, "eight")
    res["oppoint"] = T.oppoint_stats(y, logit.astype(np.float32), g, "eight")
    res["oppoint_ma"] = ma_gaps(y, logit.astype(np.float32), res["oppoint"]["intercept"], MAL.ma_matrix(df),
                                df["cluster"].values, int(ma_cfg["ma"].get("min_val_videos", 2)))
    tr = json.load(open(os.path.join(run, "TRAINED")))
    ref = tr.get("select_ref")
    return {"run": run, "value_ma": T.selection_value(res, "val_oppoint_ma", ref),
            "value_single_axis": T.selection_value(res, "val_oppoint", ref), "val_auc": res["auc"],
            "max_gap": res["oppoint_ma"]["max_gap"], "argmax_axis": res["oppoint_ma"]["argmax_axis"],
            "axis_gaps": {k: v["gap"] for k, v in res["oppoint_ma"]["axes"].items()},
            "gxr_gap_lin": res["oppoint"]["fpr_gap"], "pooled_TPR": res["oppoint"]["pooled_TPR"],
            "pooled_FPR": res["oppoint"]["pooled_FPR"], "trained_best": tr.get("best"), "best_epoch": tr.get("best_epoch"),
            "labels": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=os.path.dirname(HERE))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    g = json.load(open(os.path.join(a.out_root, "results", "gate.json")))
    if g.get("status") != "PASS":
        sys.exit(f"gate is {g.get('status')}: no MA comparison")
    cfgname = g["ma_config"]
    ma_run = os.path.join(a.out_root, "runs", f"ffpp_{cfgname}_s{a.seed}")
    cfg = load_config(os.path.join(ma_run, "config.yaml"))      # the MA run's resolved config (its label files)
    single = score_run(g["single_axis_run"], cfg)
    ma = score_run(ma_run, cfg)
    out = {"seed": a.seed, "rule": "not_worse = value_ma(CESel-MA) >= value_ma(single-axis winner), validation only",
           "single_axis_arm": g["single_axis_arm"], "ma_config": cfgname, "single_axis": single, "ma": ma,
           "not_worse": bool(ma["value_ma"] >= single["value_ma"]), "test_metrics_used": False}
    p = os.path.join(a.out_root, "results", f"ma_vs_single_s{a.seed}.json")
    tmp = p + f".tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, p)
    print(f"MA {ma['value_ma']:.3f} vs single-axis {single['value_ma']:.3f} -> not_worse={out['not_worse']} ({p})")


if __name__ == "__main__":
    main()
