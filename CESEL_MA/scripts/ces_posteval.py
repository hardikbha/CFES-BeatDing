"""Post-evaluation of a CESel run from its saved scores (CPU, no inference): A1 threshold metrics and per-cue
interpretability. Needs <run>/eval/{val,<set>}_scores.npz with s_k (scripts/evaluate_ces.py) and <run>/<ckpt>.

  python scripts/ces_posteval.py --run runs/ffpp_B0_s7        -> <run>/eval/a1.json, <run>/eval/interp.json

A1 (fair_a1.py = the CFES fair_oppoint rule, verbatim): t chosen on FF++ validation only; test metrics at t, plus the
reference points t = 0 (raw 0.5) and t = -b (validation-calibrated common intercept, calibrate.py).

Interpretability (additive head, s = sum_k s_k + b, s_k = w_k^T c_k; nothing here is used for selection):
  beta[c,k] = w_k^T mu[c,k], beta[c,g,k] = w_k^T mu[c,g,k] from the checkpoint's EMA means (CES state); neutral
  baseline beta0_k = (beta[0,k] + beta[1,k]) / 2 (label-free).
  * removal utility u_ik = m(s_i, y_i) - m(s_i - s_ik + beta_ik, y_i), m = training margin (asinh, kappa), with the
    TRAINING replacement rule (reals: beta[fake,k]; fakes: beta[real,g,k] of their own group, six-group sets mapped by
    group name, else beta[real,k]; scripts/cue_stats.py). Uses labels: analysis only.
    -> per (set, class, group): mean u_k, mean s_k, mean centred contribution s_k - beta0_k, n.
  * per-cue deletion curve: s(alpha) = s - alpha (s_k - beta0_k), alpha in {0, .25, .5, .75, 1}: AUC, acc at
    logit >= 0, and TPR / FPR / i-F_FPR at the run's A1 threshold (threshold kept fixed).
  * cumulative deletion: cues ordered by validation importance mean_i |s_ik - beta0_k| (label-free, FF++ val), removed
    1..4 at a time (alpha = 1) -> AUC; reference = mean over all 24 orders.
  * solo AUC of each s_k.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cue_stats as C  # noqa: E402
import fair_a1 as A  # noqa: E402
from cfes.ces_model import CUES  # noqa: E402
from cfes.config import load_config  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def ld(f):
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def auc(y, s):
    return 100 * float(roc_auc_score(y, s.astype(np.float64))) if len(np.unique(y)) == 2 else float("nan")


def intervene(s, sk, b, beta0, alpha):
    """Codex review F9: the intervened score rebuilt from the RETAINED cue terms + replacement constants + head bias,
    never as s - (removed terms): s is a float32 model sum, so subtracting saved cue terms leaves image-dependent
    rounding residuals that AUC would rank (full deletion must be EXACTLY the constant b + sum_k beta0_k).
    alpha: (K,) per-cue deletion fraction; cue k contributes (1 - alpha_k) s_k + alpha_k beta0_k (alpha_k = 1: beta0_k
    only, alpha_k = 0: s_k only). All-zero alpha returns the saved model score s itself (no intervention)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    s, sk, beta0 = np.asarray(s, np.float64), np.asarray(sk, np.float64), np.asarray(beta0, np.float64)
    if not alpha.any():
        return s
    rm = alpha > 0
    out = np.full(len(s), float(b) + float((alpha[rm] * beta0[rm]).sum()), dtype=np.float64)
    if (~rm).any():
        out = out + sk[:, ~rm].sum(1)
    part = rm & (alpha < 1.0)
    if part.any():
        out = out + ((1.0 - alpha[part])[None] * sk[:, part]).sum(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--eval_dir", default=None)
    a = ap.parse_args()
    ev = a.eval_dir or os.path.join(a.run, "eval")
    cfg = load_config(os.path.join(a.run, "config.yaml"))
    ces = cfg.get("ces", {})
    train_scheme = cfg["data"]["train"].get("scheme", "eight")
    V = ld(os.path.join(ev, "val_scores.npz"))
    T = {k: ld(os.path.join(ev, f"{k}_scores.npz")) for k in cfg["eval"]["test_sets"]
         if os.path.exists(os.path.join(ev, f"{k}_scores.npz"))}
    schemes = {k: s.get("scheme", "eight") for k, s in cfg["eval"]["test_sets"].items()}

    # ---------------------------------------------------------------- A1 (+ reference points)
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
    json.dump(rec, open(os.path.join(ev, "a1.json"), "w"), indent=1, default=float)
    t_a1 = rec["t"]

    # ---------------------------------------------------------------- interpretability
    ref = C.load_ref(os.path.join(a.run, a.ckpt))
    b, beta_c = ref["b"], ref["beta_c"]
    beta0 = 0.5 * (beta_c[0] + beta_c[1])
    fb = ces.get("fake_baseline", "group")
    out = {"cues": list(CUES), "head_b": b, "beta_class": beta_c.tolist(), "beta0": beta0.tolist(),
           "fake_baseline": fb, "A1_t": t_a1, "sets": {}}
    imp = np.abs(V["s_k"] - beta0[None]).mean(0)
    order = [int(i) for i in np.argsort(-imp)]
    out["val_importance"] = {CUES[k]: float(imp[k]) for k in range(len(CUES))}
    out["deletion_order"] = [CUES[k] for k in order]
    for name, d in {"val": V, **T}.items():
        s, sk, y, g = d["logit"].astype(np.float64), d["s_k"].astype(np.float64), d["label"].astype(int), d["group"].astype(int)
        sch = "eight" if name == "val" else schemes[name]
        if np.isnan(sk).any():
            raise SystemExit(f"{name}: s_k missing (re-run evaluate_ces.py)")
        err = float(np.abs(sk.sum(1) + b - s).max())
        u = C.utilities(s, sk, y, C.to_train_groups(g, sch, train_scheme), ref, ces)
        cen = sk - beta0[None]
        S = {"n": int(len(y)), "additivity_max_err": err, "solo_auc": {CUES[k]: auc(y, sk[:, k]) for k in range(len(CUES))},
             "by_group": {}, "deletion": {}, "cumulative": {}}
        for c in (0, 1):
            for gg in np.unique(g[y == c]).tolist():
                m = (y == c) & (g == gg)
                S["by_group"][f"{'real' if c == 0 else 'fake'}|{gg}"] = {
                    "n": int(m.sum()), "u": {CUES[k]: float(u[m, k].mean()) for k in range(len(CUES))},
                    "s_k": {CUES[k]: float(sk[m, k].mean()) for k in range(len(CUES))},
                    "centred": {CUES[k]: float(cen[m, k].mean()) for k in range(len(CUES))}}
            m = y == c
            S["by_group"][f"{'real' if c == 0 else 'fake'}|all"] = {
                "n": int(m.sum()), "u": {CUES[k]: float(u[m, k].mean()) for k in range(len(CUES))},
                "s_k": {CUES[k]: float(sk[m, k].mean()) for k in range(len(CUES))},
                "centred": {CUES[k]: float(cen[m, k].mean()) for k in range(len(CUES))}}
        def intervened(alpha):
            return intervene(s, sk, b, beta0, alpha)

        def onehot(ks, al=1.0):
            v = np.zeros(len(CUES))
            v[list(ks)] = al
            return v

        for k in range(len(CUES)):
            curve = []
            for al in ALPHAS:
                sa = intervened(onehot([k], al))
                pt = {"alpha": al, "AUC": auc(y, sa), "acc0": 100 * float(((sa >= 0) == (y == 1)).mean())}
                if t_a1 is not None:
                    fm = A.full_metrics(sa, y, g, sch, t_a1)
                    pt.update(TPR_A1=fm["TPR"], FPR_A1=fm["FPR"], i_F_FPR_A1=fm["i_F_FPR"])
                curve.append(pt)
            S["deletion"][CUES[k]] = curve
        cum = [auc(y, s)]
        for n_rm in range(1, len(CUES) + 1):
            cum.append(auc(y, intervened(onehot(order[:n_rm]))))
        ref_curve = []
        for n_rm in range(0, len(CUES) + 1):
            # mean over distinct subsets of size n_rm (= mean over all 24 orders, each subset equally often)
            subs = list(itertools.combinations(range(len(CUES)), n_rm))
            ref_curve.append(float(np.mean([auc(y, intervened(onehot(q))) for q in subs])) if n_rm else cum[0])
        S["cumulative"] = {"order": [CUES[k] for k in order], "AUC": cum, "AUC_mean_all_orders": ref_curve}
        out["sets"][name] = S
    json.dump(out, open(os.path.join(ev, "interp.json"), "w"), indent=1, default=float)
    print(f"A1 t={t_a1} | val importance {out['val_importance']} | additivity err "
          f"{max(v['additivity_max_err'] for v in out['sets'].values()):.2e}")


if __name__ == "__main__":
    main()
