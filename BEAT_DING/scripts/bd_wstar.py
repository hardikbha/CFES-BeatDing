"""Beat-Ding plan step 3: base Arm-E weight w* (decisions/2026-10-26-beat-ding-plan.md section 2 "Base E weight" and
section 4). VALIDATION only: for the existing Arm E runs w in {1, 3, 10} (seed 7) and CFES-D s7 (reference), the
val Ding-score = number of Ding's 10 columns on FF++ val, at the matched threshold (val TPR = the Ding reproduction's
val TPR at 0.5), on which the run beats the Ding reproduction's FF++ val columns at its 0.5; ties -> val AUC.
w* = argmax over the three Arm E runs. DISCLOSED: this rule postdates seeing these runs' test results.

  /home/sachin.chaudhary/.conda/envs/aiface/bin/python scripts/bd_wstar.py      # -> results/wstar.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import bd_metrics as BM  # noqa: E402

RUNS = BM.R + "/agents/CFES/runs_rev4"
CANDS = {f"armE_w{w}_s7": f"{RUNS}/clip_E_w{w}_s7" for w in (1, 3, 10)}
REFS = {"CFES-D_s7": f"{RUNS}/clip_D_w0.3_s7"}


def main():
    ref = BM.ding_val_reference()
    print(f"Ding repro on FF++ val at 0.5: TPR {100 * ref['tpr']:.4f} FPR {100 * ref['fpr']:.4f} "
          f"(n_fake {ref['n_fake']}, n_real {ref['n_real']})")
    print("Ding repro val columns: " + " ".join(f"{c} {ref['cols'][c]:.2f}" for c in BM.K))
    out = {"rule": "val Ding-score at the matched val TPR, ties -> val AUC (plan section 4)", "ding_val": ref, "runs": {}}
    for name, run in {**CANDS, **REFS}.items():
        zv, yv, gv, _ = BM.load(f"{run}/eval/val_scores.npz")
        s = BM.val_ding_score(zv, yv, gv, ref)
        s["run"] = run
        out["runs"][name] = s
        print(f"{name:12s} val Ding-score {s['score']:2d}/10  val AUC {s['val_auc']:.3f}  t {s['t']:+.4f}  "
              f"val TPR {s['val_TPR']:.2f} FPR {s['val_FPR']:.2f} | wins {s['wins']} | "
              + " ".join(f"{c} {s['cols'][c]:.2f}" for c in BM.K))
    best = max(CANDS, key=lambda n: BM.rank_key(out["runs"][n]))
    out["w_star"] = int(best.split("_w")[1].split("_")[0])
    out["w_star_run"] = best
    out["best_reference_score"] = max(v["score"] for v in out["runs"].values())
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "wstar.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"w* = {out['w_star']} ({best})")


if __name__ == "__main__":
    main()
