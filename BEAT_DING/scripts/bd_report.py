"""Beat-Ding reports (decisions/2026-10-26-beat-ding-plan.md sections 3-5; CPU, saved scores only).

  python scripts/bd_report.py refs                  # reference rows (CFES-D s7, Arm E w1/w3/w10 s7, Ding repro) -> results/refs/
  python scripts/bd_report.py screen                # seed-7 screen: BD0..BD4 vs references, winner + promotion
                                                    #   -> results/bd_screen.json / .md
  python scripts/bd_report.py final                 # 3-seed means of the promoted recipe vs Ding PUBLISHED, success
                                                    #   criteria -> results/bd_final.json / .md
Options: --runs_root (default runs/), --results (default results/), --arms BD0,...; for tests.

Screen (section 4): per row the val Ding-score (FF++ val, matched val TPR, vs the Ding reproduction's val columns at its
0.5), val AUC, and test columns vs Ding published at the matched operating point (also native and A1 totals).
Winner = the BD arm with the highest val Ding-score, ties -> val AUC. Promotion requires winner score >= the best
existing reference's (CFES-D s7, Arm E w1/w3/w10 s7; Amendment 1: 6/10). Not promoted -> the paper uses the best
reference (highest val Ding-score, ties -> val AUC). The test sets never choose.
Final (section 5): mean +- SD over seeds 7/17/27 of the promoted recipe (or of the best reference) at the matched
operating point; a column wins if the 3-seed MEAN beats Ding published. Claim "beats Ding" iff >= 25/40 columns AND
AUC + the three es-AUC columns better on all four test sets AND intersectional F_FPR better on >= 3 of 4. Every column
is reported, with TPR / FPR, the F_DP oracle and the FF++ noise floor next to FF++ F_FPR.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import bd_metrics as BM  # noqa: E402

CF = BM.R + "/agents/CFES/runs_rev4"
REFS = {"CFES-D_s7": f"{CF}/clip_D_w0.3_s7", "armE_w1_s7": f"{CF}/clip_E_w1_s7", "armE_w3_s7": f"{CF}/clip_E_w3_s7",
        "armE_w10_s7": f"{CF}/clip_E_w10_s7"}
# reference recipes -> how their seed s run is found (existing CFES runs; BD-tree copies trained by run/bd_run.sh winner)
REF_RECIPES = {"CFES-D_s7": ("CFES-D", f"{CF}/clip_D_w0.3_s{{s}}"), "armE_w1_s7": ("E_w1", f"{CF}/clip_E_w1_s{{s}}"),
               "armE_w3_s7": ("E_w3", f"{CF}/clip_E_w3_s{{s}}"), "armE_w10_s7": ("E_w10", f"{CF}/clip_E_w10_s{{s}}")}
ARMS = ["BD0", "BD1", "BD2", "BD3", "BD4"]
SEEDS = (7, 17, 27)
PY = sys.executable


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def atomic_text(txt, path):
    tmp = f"{path}.tmp{os.getpid()}"
    open(tmp, "w").write(txt)
    os.replace(tmp, path)


def bd_eval(args):
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "bd_eval.py"), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bd_eval {args} failed: {r.stderr[-2000:]}")
    print(r.stdout.strip())


def refs(a):
    d = os.path.join(a.results, "refs")
    os.makedirs(d, exist_ok=True)
    for name, run in REFS.items():
        if not os.path.exists(os.path.join(d, f"{name}.json")):
            bd_eval(["--run", run, "--out", os.path.join(d, f"{name}.json"), "--name", name])
    if not os.path.exists(os.path.join(d, "ding_repro_s7.json")):
        bd_eval(["--ding", "--out", os.path.join(d, "ding_repro_s7.json")])


def row(r):
    v = r["val_ding_score"]
    return {"name": r["name"], "val_ding_score": v["score"], "val_auc": v["val_auc"], "val_wins": v["wins"],
            "val_TPR": v["val_TPR"], "val_FPR": v["val_FPR"], "matched_t": r["thresholds"]["matched"],
            "test_wins": r["total_wins_vs_ding_published"],
            "per_set": {k: (r["test"][k]["matched"]["n_wins"] if r["test"].get(k) else None) for k in BM.SETS}}


def winner_rule(arm_rows, ref_rows):
    """-> dict(winner, promoted, best_reference, recipe). Validation only."""
    key = lambda x: (x["val_ding_score"], x["val_auc"])      # noqa: E731
    w = max(arm_rows, key=key) if arm_rows else None
    best_ref = max(ref_rows, key=key)
    promoted = w is not None and w["val_ding_score"] >= best_ref["val_ding_score"]
    return {"winner": w["name"] if w else None, "winner_score": w["val_ding_score"] if w else None,
            "winner_val_auc": w["val_auc"] if w else None, "best_reference": best_ref["name"],
            "best_reference_score": best_ref["val_ding_score"], "best_reference_val_auc": best_ref["val_auc"],
            "promoted": bool(promoted), "recipe": (w["name"].rsplit("_s", 1)[0] if promoted else best_ref["name"])}


def table_rows(r, op="matched"):
    out = []
    for k in BM.SETS:
        b = (r["test"].get(k) or {}).get(op)
        if b:
            out.append((k, b))
    return out


def screen(a):
    refs(a)
    rows, missing = {}, []
    for arm in a.arms:
        r = jl(os.path.join(a.runs_root, f"{arm}_s7", "eval", "bd_eval.json"))
        if r is None or not r.get("complete"):
            missing.append(arm)
            continue
        rows[f"{arm}_s7"] = r
    if missing and not a.allow_missing:
        raise SystemExit(f"screen: missing / incomplete bd_eval.json for {missing}")
    ref = {n: jl(os.path.join(a.results, "refs", f"{n}.json")) for n in REFS}
    ding = jl(os.path.join(a.results, "refs", "ding_repro_s7.json"))
    dec = winner_rule([row(r) for r in rows.values()], [row(r) for r in ref.values()])
    out = {"plan": "decisions/2026-10-26-beat-ding-plan.md sections 3-4 + Amendment 1", "decision": dec,
           "arms": {n: row(r) for n, r in rows.items()}, "references": {n: row(r) for n, r in ref.items()},
           "ding_repro": row(ding), "missing": missing}
    atomic_json(out, os.path.join(a.results, "bd_screen.json"))
    L = ["# Beat-Ding screen (seed 7)", "",
         f"Winner {dec['winner']} (val Ding-score {dec['winner_score']}/10, val AUC {fmt(dec['winner_val_auc'], 3)}); "
         f"best existing reference {dec['best_reference']} ({dec['best_reference_score']}/10, val AUC "
         f"{fmt(dec['best_reference_val_auc'], 3)}). **Promoted to seeds 17/27: {dec['promoted']}** "
         f"(rule: winner val Ding-score >= best reference). Recipe for the 3-seed claim: **{dec['recipe']}**.", "",
         "| row | val Ding-score | val AUC | val TPR / FPR | test wins @matched (FF++/DFDC/CDF/DFD) | total @matched | @native | @A1 |",
         "|---|---|---|---|---|---|---|---|"]
    for n, x in list(out["arms"].items()) + list(out["references"].items()) + [("Ding repro s7 (own 0.5)", out["ding_repro"])]:
        ps = "/".join(str(x["per_set"][k]) for k in BM.SETS)
        L.append(f"| {n} | {x['val_ding_score']}/10 | {fmt(x['val_auc'], 3)} | {fmt(x['val_TPR'])} / {fmt(x['val_FPR'])} | "
                 f"{ps} | {x['test_wins']['matched']}/40 | {x['test_wins']['native']}/40 | {x['test_wins']['A1']}/40 |")
    L += ["", "Test columns are reported only; selection used validation only. Per-run tables: runs/<arm>_s7/eval/bd_eval.md, "
          "results/refs/*.md."]
    for n, r in rows.items():
        L += ["", f"## {n} at the matched operating point (bold = beats Ding published)", ""] + col_table([r])
    atomic_text("\n".join(L) + "\n", os.path.join(a.results, "bd_screen.md"))
    print(f"screen: winner {dec['winner']} promoted {dec['promoted']} recipe {dec['recipe']} -> {a.results}/bd_screen.json")
    return out


def fmt(v, n=2):
    return f"{v:.{n}f}" if isinstance(v, (int, float)) and v == v else "-"


def col_table(runs, op="matched"):
    """Markdown table: Ding published vs the mean (+- SD) of `runs` (bd_eval records) at `op`."""
    L = ["| set | row | TPR | FPR | " + " | ".join(BM.K) + " | wins |", "|" + "---|" * (len(BM.K) + 5)]
    for k in BM.SETS:
        bs = [r["test"][k][op] for r in runs if r["test"].get(k) and r["test"][k].get(op)]
        if not bs:
            continue
        mean = {c: float(np.mean([b["cols"][c] for b in bs])) for c in BM.K + ["TPR", "FPR"]}
        sd = {c: float(np.std([b["cols"][c] for b in bs], ddof=1)) if len(bs) > 1 else None for c in BM.K + ["TPR", "FPR"]}
        pub = runs[0].get("published") or BM.DING_PUB
        w = BM.wins(mean, pub[k])
        L.append(f"| {BM.NAMES[k]} | Ding published | | | " + " | ".join(fmt(v) for v in pub[k]) + " | |")
        cell = lambda c, i: (("**%s**" if w[i] else "%s") % fmt(mean[c])) + (f" ± {fmt(sd[c])}" if sd[c] is not None else "")  # noqa: E731
        L.append(f"| {BM.NAMES[k]} | ours (n={len(bs)}) | {fmt(mean['TPR'])} | {fmt(mean['FPR'])} | "
                 + " | ".join(cell(c, i) for i, c in enumerate(BM.K)) + f" | {sum(w)}/10 |")
    return L


def recipe_runs(recipe, a):
    """bd_eval records of seeds 7/17/27 for a BD arm (runs/<arm>_s<seed>) or a reference recipe."""
    out, miss = [], []
    for s in SEEDS:
        if recipe in ARMS:
            f = os.path.join(a.runs_root, f"{recipe}_s{s}", "eval", "bd_eval.json")
        else:
            f = os.path.join(a.results, "refs", f"{recipe.rsplit('_s', 1)[0]}_s{s}.json")
            if not os.path.exists(f):
                run = REF_RECIPES[recipe][1].format(s=s)
                local = os.path.join(a.runs_root, f"{REF_RECIPES[recipe][0]}_s{s}")
                src = run if os.path.exists(os.path.join(run, "eval", "val_scores.npz")) else local
                if os.path.exists(os.path.join(src, "eval", "val_scores.npz")):
                    bd_eval(["--run", src, "--out", f, "--name", f"{recipe.rsplit('_s', 1)[0]}_s{s}"])
        r = jl(f)
        (out if r and r.get("complete") else miss).append(r if r else f)
    return out, miss


def resnet_components(sc):
    """Plan Amendment 2: the ResNet-50 runs use the CLIP screen's winning component set: the promoted BD arm (BD0..BD4),
    else no Beat-Ding component ('base' = CFES-ResNet + group-FPR-gap w* = 1)."""
    d = sc["decision"]
    return d["winner"].rsplit("_s", 1)[0] if d["promoted"] else "base"


def final(a):
    sc = jl(os.path.join(a.results, "bd_screen.json"))
    if sc is None:
        raise SystemExit("final: results/bd_screen.json missing (run the screen first)")
    if a.family == "resnet":
        recipe = "R50_" + resnet_components(sc)
        runs, miss = [], []
        for s in SEEDS:
            f = os.path.join(a.runs_root, f"{recipe}_s{s}", "eval", "bd_eval.json")
            r = jl(f)
            (runs if r and r.get("complete") else miss).append(r if r else f)
    else:
        recipe = sc["decision"]["recipe"]
        runs, miss = recipe_runs(recipe, a)
    if miss and not a.allow_missing:
        raise SystemExit(f"final: missing seed evaluations for {recipe}: {miss}")
    if not runs:
        raise SystemExit(f"final: no complete seed evaluation for {recipe}")
    pub = runs[0].get("published") or BM.DING_PUB
    per = {}
    for k in BM.SETS:
        bs = [r["test"][k]["matched"] for r in runs]
        mean = {c: float(np.mean([b["cols"][c] for b in bs])) for c in BM.K + ["TPR", "FPR"]}
        sd = {c: float(np.std([b["cols"][c] for b in bs], ddof=1)) if len(bs) > 1 else None for c in BM.K + ["TPR", "FPR"]}
        per[k] = {"mean": mean, "sd": sd, "wins": BM.wins(mean, pub[k]), "seed_cols": [b["cols"] for b in bs]}
    total = sum(sum(v["wins"]) for v in per.values())
    i_auc, i_es = BM.K.index("AUC"), [BM.K.index(c) for c in ("gesAUC", "resAUC", "iesAUC")]
    auc_es_all = all(per[k]["wins"][i_auc] and all(per[k]["wins"][j] for j in i_es) for k in BM.SETS)
    iff = sum(per[k]["wins"][BM.K.index("iF_FPR")] for k in BM.SETS)
    claim = total >= 25 and auc_es_all and iff >= 3
    nf = [r.get("noise_floor", {}).get("matched") for r in runs]
    oracle = runs[0]["fdp_oracle"] if runs else {}
    out = {"family": a.family, "published_table": runs[0].get("published_table"), "published": pub,
           "recipe": recipe, "promoted": sc["decision"]["promoted"], "seeds": [r["name"] for r in runs], "missing": miss,
           "per_set": per, "total_wins": total, "criteria": {"wins_ge_25_of_40": total >= 25,
                                                             "auc_and_all_es_auc_better_on_all_4_sets": auc_es_all,
                                                             "iF_FPR_better_on_ge_3_of_4": iff >= 3, "iF_FPR_sets_better": iff},
           "claim_beats_ding": bool(claim), "fdp_oracle": oracle, "ffpp_noise_floor_matched": nf,
           "tpr_fpr": {k: {"TPR": per[k]["mean"]["TPR"], "FPR": per[k]["mean"]["FPR"]} for k in BM.SETS}}
    tag = "bd_final" if a.family == "clip" else "bd_final_resnet"
    atomic_json(out, os.path.join(a.results, f"{tag}.json"))
    L = [f"# Beat-Ding final ({a.family}): {recipe}, {len(runs)} seed(s) at the matched operating point, vs Ding "
         f"published {out['published_table']}", "",
         f"{'Promoted BD recipe' if sc['decision']['promoted'] else 'No BD arm reached the best reference; best reference'}: "
         f"**{recipe}**. Seeds: {', '.join(out['seeds'])}" + (f"; MISSING {miss}" if miss else ""), "",
         f"Columns won (3-seed mean vs Ding published): **{total}/40**. Criteria: >= 25/40 {total >= 25}; AUC and all "
         f"three es-AUC better on all four sets {auc_es_all}; intersectional F_FPR better on {iff}/4 sets (need >= 3). "
         f"**Claim 'beats Ding': {claim}.**", ""] + col_table(runs)
    L += ["", "F_DP oracle (perfect detector; g/r/i): " + "; ".join(
        f"{BM.NAMES[k]} {fmt(v['gF_DP'])}/{fmt(v['rF_DP'])}/{fmt(v['iF_DP'])}" for k, v in oracle.items())]
    for r, x in zip(runs, nf):
        if x:
            L.append(f"FF++ i-F_FPR noise floor ({r['name']}, matched t, real FPR {fmt(x['real_FPR'])} %): reals-only "
                     f"observed {fmt(x['observed_reals_only_iF_FPR'])}, video-permutation null median {fmt(x['null_median'])} "
                     f"[{fmt(x['null_q05'])}, {fmt(x['null_q95'])}]")
    if a.family == "resnet":
        L += ["", "Matched operating point: val TPR 94.18 %, the Xception reproduction's (no Ding ResNet reproduction "
              "exists; disclosed proxy, plan Amendment 2)."]
    atomic_text("\n".join(L) + "\n", os.path.join(a.results, f"{tag}.md"))
    print(f"final ({a.family}): {recipe} {total}/40, claim {claim} -> {a.results}/{tag}.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "screen", "final"])
    ap.add_argument("--runs_root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--allow_missing", action="store_true", help="interim report (never used by the task list)")
    ap.add_argument("--family", choices=["clip", "resnet"], default="clip", help="final: CLIP (Tables 1-2) or ResNet (Table 3)")
    a = ap.parse_args()
    a.arms = [x for x in a.arms.split(",") if x]
    os.makedirs(a.results, exist_ok=True)
    {"refs": refs, "screen": screen, "final": final}[a.what](a)


if __name__ == "__main__":
    main()
