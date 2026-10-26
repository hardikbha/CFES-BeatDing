"""Beat-Ding evaluation of one run from its SAVED scores (CPU; plan decisions/2026-10-26-beat-ding-plan.md sections 4-5).

  python scripts/bd_eval.py --run runs/BD2_s7                       # needs <run>/eval/{val,<sets>}_scores.npz
  python scripts/bd_eval.py --run ../CFES/runs_rev4/clip_E_w1_s7 --out results/refs/armE_w1_s7.json --name armE_w1_s7
  python scripts/bd_eval.py --ding --out results/refs/ding_repro_s7.json    # the Ding reproduction (its own 0.5)

Operating points, each ONE global logit threshold chosen on FF++ validation only (cfes/bd_metrics.py):
  matched : val TPR = the Ding reproduction's val TPR at 0.5 (94.18 %);   native : t = -b (scripts/calibrate.py
  class-balanced common intercept, eval/calibration.json; refitted from val logits if the file is missing);
  A1      : scripts/fair_a1.py.        (--ding: the reproduction's own 0.5, t = 0, is reported as "matched" and "native".)
Per test set (ffpp_test, dfdc, celebdf, dfd) and operating point: Ding's 10 columns + TPR / FPR, win marks against Ding
PUBLISHED Tables 1-2 and the win count. Also: the val Ding-score (section 4), the F_DP oracle per set (perfect
detector) and the FF++ test noise floor of intersectional F_FPR (video-permutation null, code/noise_floor.py) at the
matched threshold and at pooled real FPR 12.5 % (Ding's operating point). Writes <out> (json) and <out>.md.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cfes import bd_metrics as BM  # noqa: E402


def set_files(a):
    if a.ding:
        val = BM.DING_VAL_NPZ
        sets = {k: f"{BM.DING_DIR}/full_s{a.ding_seed}_ep49_{v}.npz" for k, v in BM.DING_TEST_NPZ.items()}
        return val.replace("_s7_", f"_s{a.ding_seed}_"), sets
    ev = a.eval_dir or os.path.join(a.run, "eval")
    return os.path.join(ev, "val_scores.npz"), {k: os.path.join(ev, f"{k}_scores.npz") for k in BM.SETS}


def backend_of(a):
    """model.backend of the run (clip | resnet | ...) from <run>/config.yaml; --table overrides."""
    if a.table:
        return "resnet" if a.table == 3 else "xception"
    f = os.path.join(a.run, "config.yaml") if a.run else None
    if f and os.path.exists(f):
        import yaml
        return (yaml.safe_load(open(f)).get("model") or {}).get("backend", "clip")
    return "clip"


def native_threshold(a, zv, yv):
    if a.ding:
        return 0.0, "own 0.5 (logit 0)"
    f = os.path.join(a.eval_dir or os.path.join(a.run, "eval"), "calibration.json")
    if os.path.exists(f):
        return -float(json.load(open(f))["intercept_b"]), f
    from calibrate import fit_intercept
    return -fit_intercept(zv, yv)[0], "refitted from val logits (calibrate.fit_intercept)"


def block(z, y, g, scheme, t, k, pub):
    d = BM.cols(y, z, g, scheme, t, with_groups=True)
    w = BM.wins(d, pub[k])
    return {"t": float(t), "cols": {c: round(float(d[c]), 4) for c in BM.K + ["TPR", "FPR"]}, "wins": w,
            "n_wins": int(sum(w)), "groups": {lv: d[lv + "_groups"] for lv in ("g", "r", "i")}}


def evaluate_run(a):
    vf, sets = set_files(a)
    zv, yv, gv, _ = BM.load(vf)
    backend = "xception" if a.ding else backend_of(a)
    table, pub = BM.published_for(backend)
    ref = BM.ding_val_reference()
    vds = BM.val_ding_score(zv, yv, gv, ref)
    if a.ding:          # the reference itself, at its own 0.5 (the quantile rule would land a hair away from logit 0)
        d0 = BM.cols(yv, zv, gv, "eight", 0.0)
        vds.update(t=0.0, cols=d0, wins=BM.wins(d0, ref["cols"]), val_auc=d0["AUC"], val_TPR=d0["TPR"], val_FPR=d0["FPR"])
        vds["score"] = int(sum(vds["wins"]))
    t_nat, nat_src = native_threshold(a, zv, yv)
    ops = {"matched": 0.0 if a.ding else vds["t"], "native": t_nat, "A1": 0.0 if a.ding else BM.a1_threshold(zv, yv, gv)}
    out = {"name": a.name or (os.path.basename(os.path.normpath(a.run)) if a.run else f"ding_repro_s{a.ding_seed}"),
           "run": a.run, "val_file": vf, "backend": backend, "published_table": table, "published": pub, "ding_val_reference": {k: ref[k] for k in ("file", "tpr", "fpr", "cols")},
           "val_ding_score": {k: vds[k] for k in ("score", "wins", "t", "val_auc", "val_TPR", "val_FPR", "cols")},
           "thresholds": ops, "native_source": nat_src, "val": {}, "test": {}, "fdp_oracle": {}, "noise_floor": {}}
    for op, t in ops.items():
        if t is None:
            out["val"][op] = None
            continue
        d = BM.cols(yv, zv, gv, "eight", t)
        out["val"][op] = {"t": float(t), "cols": {c: round(float(d[c]), 4) for c in BM.K + ["TPR", "FPR"]},
                          "wins_vs_ding_val": BM.wins(d, ref["cols"])}
    for k, f in sets.items():
        if not os.path.exists(f):
            out["test"][k] = None
            continue
        z, y, g, paths = BM.load(f)
        sch = BM.SETS[k]
        out["test"][k] = {op: (block(z, y, g, sch, t, k, pub) if t is not None else None) for op, t in ops.items()}
        out["fdp_oracle"][k] = BM.fdp_oracle(y, g, sch)
        if k == "ffpp_test" and paths is not None and not a.no_noise:
            nf = {"matched": BM.noise_floor(z, y, g, paths, ops["matched"], n_perm=a.n_perm)}
            t125 = float(np.quantile(z[y == 0], 1 - 0.125))
            nf["realFPR12.5"] = BM.noise_floor(z, y, g, paths, t125, n_perm=a.n_perm)
            out["noise_floor"] = nf
    tot = {}
    for op in ops:
        tot[op] = 0
        for k in BM.SETS:
            b = (out["test"].get(k) or {}).get(op)
            tot[op] += b["n_wins"] if b else 0
    out["total_wins_vs_ding_published"] = tot
    out["complete"] = all(out["test"].get(k) is not None for k in BM.SETS)
    return out


def fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) and v == v else str(v)


def render(out):
    L = [f"# Beat-Ding evaluation: {out['name']} (vs Ding published {out['published_table']})", "",
         f"val Ding-score (matched val TPR {100 * out['ding_val_reference']['tpr']:.2f} %): "
         f"**{out['val_ding_score']['score']}/10**, val AUC {out['val_ding_score']['val_auc']:.3f}; thresholds "
         + ", ".join(f"{k} {fmt(v)}" for k, v in out["thresholds"].items()), ""]
    hdr = "| set | op | TPR | FPR | " + " | ".join(BM.K) + " | wins |"
    L += [hdr, "|" + "---|" * (len(BM.K) + 5)]
    for k in BM.SETS:
        L.append(f"| {BM.NAMES[k]} | Ding published | | | " + " | ".join(fmt(v) for v in out["published"][k]) + " | |")
        for op in out["thresholds"]:
            b = (out["test"].get(k) or {}).get(op)
            if not b:
                continue
            cells = [f"**{fmt(b['cols'][c])}**" if b["wins"][i] else fmt(b["cols"][c]) for i, c in enumerate(BM.K)]
            L.append(f"| {BM.NAMES[k]} | {op} | {fmt(b['cols']['TPR'])} | {fmt(b['cols']['FPR'])} | "
                     + " | ".join(cells) + f" | {b['n_wins']}/10 |")
    L += ["", "Bold = beats Ding published. Totals: " + ", ".join(f"{op} {v}/40" for op, v in out["total_wins_vs_ding_published"].items()),
          "", "F_DP oracle (perfect detector, g/r/i): " + "; ".join(
              f"{BM.NAMES[k]} {fmt(v['gF_DP'])}/{fmt(v['rF_DP'])}/{fmt(v['iF_DP'])}" for k, v in out["fdp_oracle"].items())]
    for op, nf in (out.get("noise_floor") or {}).items():
        L.append(f"FF++ noise floor ({op}, real FPR {fmt(nf['real_FPR'])} %): observed reals-only i-F_FPR "
                 f"{fmt(nf['observed_reals_only_iF_FPR'])}, null median {fmt(nf['null_median'])} "
                 f"[{fmt(nf['null_q05'])}, {fmt(nf['null_q95'])}], P(null >= obs) {nf['p_null_ge_obs']:.3f}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--eval_dir", default=None, help="score directory (default <run>/eval)")
    ap.add_argument("--ding", action="store_true", help="evaluate the Ding reproduction's saved scores instead")
    ap.add_argument("--ding_seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--no_noise", action="store_true")
    ap.add_argument("--table", type=int, choices=[2, 3], default=None,
                    help="published Ding table to compare with (default: 3 for a ResNet run, Tables 1-2 otherwise)")
    a = ap.parse_args()
    if not a.ding and not a.run:
        raise SystemExit("--run or --ding required")
    out = evaluate_run(a)
    dst = a.out or os.path.join(a.eval_dir or os.path.join(a.run, "eval"), "bd_eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=float)
    open(tmp + ".md", "w").write(render(out))
    os.replace(tmp + ".md", os.path.splitext(dst)[0] + ".md")
    os.replace(tmp, dst)
    print(f"{out['name']}: val Ding-score {out['val_ding_score']['score']}/10 (val AUC {out['val_ding_score']['val_auc']:.3f}); "
          "test wins vs Ding published " + ", ".join(f"{op} {v}/40" for op, v in out["total_wins_vs_ding_published"].items())
          + f" -> {dst}")


if __name__ == "__main__":
    main()
