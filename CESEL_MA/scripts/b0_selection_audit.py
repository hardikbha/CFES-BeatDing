"""B0 checkpoint-selection audit for Codex review F3 (decisions/2026-10-25-cesel-plan.md Amendment 4). CPU, read only.

B0 (runs/ffpp_B0_s7, launched 19:10 25 Oct, before the fix) selected its checkpoint with validation AUC computed from
float64 sigmoid probabilities, which tie every logit above ~37 (B0's validation fake logits: mean ~23, SD ~17 at
epoch 3). B0 is NOT retrained. Per-epoch validation logits were NOT saved by train_ces.py (only the per-epoch
summaries in train_log.jsonl), so the per-epoch choice cannot be recomputed exactly. What this script does:
  * lists the recorded per-epoch (probability-based) val AUC / selection values;
  * from <run>/eval/val_scores.npz (logits of best.pt, written by evaluate_ces.py) recomputes the best checkpoint's
    validation AUC / group AUCs / es-AUC from logits (rank_prob) and its Amendment-9 selection value with the fixed
    validate() rule (same reference = the start checkpoint's operating point stored in TRAINED["select_ref"]), and
    counts saturated validation logits (> 36.7) and cross-class sigmoid ties;
  * writes results/b0_selection_audit_F3.json (+ .md) (nothing is written into the run directory).
  ~/.conda/envs/pulid/bin/python scripts/b0_selection_audit.py --run runs/ffpp_B0_s7
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
from cfes.metrics import evaluate_scores, rank_prob  # noqa: E402
import train as T  # noqa: E402

SAT = 36.7      # float64 sigmoid(z) == 1.0 for z above ~36.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(os.path.dirname(HERE), "runs", "ffpp_B0_s7"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "b0_selection_audit_F3.json"))
    a = ap.parse_args()
    tr = json.load(open(os.path.join(a.run, "TRAINED")))
    ep = [json.loads(l) for l in open(os.path.join(a.run, "train_log.jsonl"))]
    ep = [e for e in ep if e.get("event") == "epoch"]
    per_epoch_logits = sorted(f for f in os.listdir(a.run) if f.startswith(("val_logits_epoch", "val_probs_epoch")))
    out = {"run": a.run, "best_epoch": tr["best_epoch"], "best_select_recorded": tr["best"],
           "per_epoch_val_logits_saved": bool(per_epoch_logits),
           "epochs": [{"epoch": e["epoch"], "val_auc_prob": e.get("val_auc"), "select": e.get("select_value"),
                       "fake_logit_mean": (e.get("val_stats") or {}).get("logit_mean_fake"),
                       "logit_std": (e.get("val_stats") or {}).get("logit_std")} for e in ep]}
    vs = os.path.join(a.run, "eval", "val_scores.npz")
    if os.path.exists(vs):
        v = np.load(vs, allow_pickle=True)
        z, y, g = v["logit"].astype(np.float64), v["label"].astype(int), v["group"].astype(int)
        p = 1.0 / (1.0 + np.exp(-z))
        res = evaluate_scores(y, rank_prob(z), g, "eight")
        res_p = evaluate_scores(y, p, g, "eight")
        res["oppoint"] = T.oppoint_stats(y, z, g, "eight")
        sel_new = T.selection_value(res, "val_oppoint", tr.get("select_ref"))
        zr, zf = z[y == 0], z[y == 1]
        ties = int(((p[y == 0] == 1.0).sum()) * ((p[y == 1] == 1.0).sum()))
        out["best_ckpt_from_eval"] = {
            "val_auc_logit": res["auc"], "val_auc_prob_recomputed": res_p["auc"],
            "inter_es_auc_logit": res["ding@0.5"]["intersection"].get("es_AUC"),
            "inter_es_auc_prob": res_p["ding@0.5"]["intersection"].get("es_AUC"),
            "select_value_fixed_rule": sel_new, "select_value_recorded": tr["best"],
            "n_real_saturated": int((zr > SAT).sum()), "n_fake_saturated": int((zf > SAT).sum()),
            "n_real": int(len(zr)), "n_fake": int(len(zf)), "cross_class_sigmoid_tie_pairs": ties}
    else:
        out["best_ckpt_from_eval"] = None
    js = a.out                                    # never written into the run directory
    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(out, open(js, "w"), indent=1, default=float)
    L = [f"# B0 selection audit (Codex F3) -- {a.run}", "",
         f"Per-epoch validation logits saved: {'yes' if per_epoch_logits else 'NO (only best.pt is re-scorable)'}; "
         f"best epoch {tr['best_epoch']} (recorded selection value {tr['best']:.3f}, probability-based AUC).", "",
         "| epoch | val AUC (prob, recorded) | selection (recorded) | fake logit mean | logit SD |", "|---|---|---|---|---|"]
    for e in out["epochs"]:
        L.append(f"| {e['epoch']} | {e['val_auc_prob']:.3f} | {e['select']:.3f} | "
                 f"{e['fake_logit_mean'] if e['fake_logit_mean'] is None else round(e['fake_logit_mean'], 2)} | "
                 f"{e['logit_std'] if e['logit_std'] is None else round(e['logit_std'], 2)} |")
    b = out["best_ckpt_from_eval"]
    L += ["", "best.pt re-scored from eval/val_scores.npz: " + (json.dumps(b) if b else "eval not available yet")]
    open(js.replace(".json", ".md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
