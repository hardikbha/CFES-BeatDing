"""Re-score every saved model on the multi-axis label tables (CPU, saved per-image logits only; no inference).

  OMP_NUM_THREADS=8 ~/.conda/envs/aiface/bin/python rescore.py [--B 1000]

Thresholds (decision fake iff logit >= t; nothing is re-selected with the new labels):
  t05  the model's deployed P(fake) >= 0.5 rule: raw logit >= 0, except CFES-D (paper-deployed with its FF++-val
       intercept folded in: logit + b >= 0, as in fair_oppoint.py / the paper's calibrated CFES-D rows)
  A1   the 22-Oct rule (FF++ val, Xu labels, intersectional F_FPR min s.t. val TPR >= 80 %, FPR <= 20 %; grid = every
       distinct val logit). Stored thresholds are used where they exist (CFES-D s7/17/27, Ding full: fair_oppoint.json;
       CESel B0: eval/a1.json) and the rule is re-run on the model's own saved val logits otherwise (flag a1_src).
       The stored values are re-derived as a check. FF2-full-trained models have no FF++-val A1 (t05 only).
Outputs (results/): rescore_axes.csv (model x seed x set x threshold x axis: pooled AUC/FPR/TPR + min-support metrics +
ding_compat metrics + bootstrap CIs), rescore_groups.csv (per-group counts and rates), thresholds.csv,
rescore_meta.json, sanity_checks.csv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
A = f"{R}/agents/CFES"
C = f"{R}/agents/CESEL"
DS = f"{R}/agents/DINGREPRO_LIN/results/audit_scores"
DE = f"{R}/agents/DINGREPRO_LIN/results/eval_ding_protocol"
sys.path.insert(0, A)
sys.path.insert(0, f"{A}/scripts")
sys.path.insert(0, f"{V}/code")
import multiaxis_metrics as MM  # noqa: E402
from fair_oppoint import pick_a1, val_curves  # noqa: E402

TEST = ["ffpp_test", "dfdc", "celebdf", "dfd"]
SCHEME = {"ffpp_test": "eight", "dfdc": "eight", "celebdf": "six", "dfd": "six", "ff2_test": "eight"}
DING_KEY = {"ffpp_test": "ff++", "dfdc": "dfdc", "celebdf": "celebdf", "dfd": "dfd", "val": "ffval"}


# ------------------------------------------------------------------------------------------------ registry
def registry():
    M = []

    def cfes_run(name, fam, pat, seeds, root="runs_rev4", deployed_b=False, ff2=True, ff2_ev="eval_ff2full_best"):
        for s in seeds:
            r = f"{A}/{root}/{pat.format(s)}"
            if not os.path.exists(f"{r}/eval/ffpp_test_scores.npz"):
                continue
            b = json.load(open(f"{r}/eval/calibration.json"))["intercept_b"] if deployed_b else 0.0
            sets = {k: f"{r}/eval/{k}_scores.npz" for k in TEST}
            f2 = f"{r}/{ff2_ev}/ff2_test_scores.npz"
            if ff2 and os.path.exists(f2):
                sets["ff2_test"] = f2
            M.append(dict(model=name, family=fam, seed=s, val=f"{r}/eval/val_scores.npz", sets=sets, b05=b, kind="logit",
                          run=r))

    cfes_run("CFES-D", "ours", "clip_D_w0.3_s{}", (7, 17, 27), deployed_b=True)
    cfes_run("arm C", "ours", "clip_C_w0.3_s{}", (7, 17, 27))
    for w in (1, 3, 10):
        cfes_run(f"arm E w{w}", "ours", f"clip_E_w{w}_s{{}}", (7,))
    cfes_run("control (det-only)", "control", "clip_A_s{}", (7, 17, 27), root="runs_study")
    cfes_run("CFES-ResNet", "ours-resnet", "resnet_A_s{}", (7, 17, 27))
    cfes_run("CFES A (rev4)", "ours", "clip_A_s{}", (7, 17, 27))
    # Ding reproductions (official code; logits)
    for tag, nm, seeds in (("full", "Ding repro (SFD+GDA)", (7,)), ("base", "Ding Ori repro (SFD)", (7, 17, 27))):
        for s in seeds:
            sets = {k: f"{DS}/{tag}_s{s}_ep49_{DING_KEY[k]}.npz" for k in TEST}
            f2 = f"{DE}/{tag}_s{s}_ep49_ff2full.npz"
            if os.path.exists(f2):
                sets["ff2_test"] = f2
            M.append(dict(model=nm, family="Ding", seed=s, val=f"{DS}/{tag}_s{s}_ep49_ffval.npz", sets=sets, b05=0.0,
                          kind="logit", run=DS))
    # CESel B0 (and any p1 arm / CLIPFAIR baseline with a finished eval)
    for r, nm in [(f"{C}/runs/ffpp_B0_s7", "CESel B0")] + \
                 [(f"{C}/runs/{d}", f"CESel p1 {d[8:-3]}") for d in sorted(os.listdir(f"{C}/runs")) if d.startswith("ffpp_p1_") and not d.endswith("RUNLOCK")] + \
                 [(f"{R}/agents/CLIPFAIR/runs/{d}", f"CLIPFAIR {d}") for d in sorted(os.listdir(f"{R}/agents/CLIPFAIR/runs"))]:
        if all(os.path.exists(f"{r}/eval/{k}_scores.npz") for k in TEST + ["val"]):
            M.append(dict(model=nm, family="CESel" if "CESel" in nm else "CLIP baseline", seed=7,
                          val=f"{r}/eval/val_scores.npz", sets={k: f"{r}/eval/{k}_scores.npz" for k in TEST}, b05=0.0,
                          kind="logit", run=r))
    # FF2-full-trained (FF2 TEST only, t05 only)
    for lab, run in (("FF2-trained phase 1", "clip_s{}_p1/eval_all_last"), ("FF2-trained control", "clip_A_s{}/eval_all"),
                     ("FF2-trained CFES", "clip_B_s{}/eval_all")):
        for s in (7, 17, 27):
            f = f"{A}/runs_ff2full/{run.format(s)}/ff2_test_scores.npz"
            if os.path.exists(f):
                M.append(dict(model=lab, family="FF2-full-trained", seed=s, val=None, sets={"ff2_test": f}, b05=0.0,
                              kind="logit", run=os.path.dirname(f)))
    return M


def load_scores(path, n_expected_paths=None):
    z = np.load(path, allow_pickle=True)
    if "logit" in z.files:
        lg = z["logit"].astype(np.float64)
    else:                                                         # Ding FF2 npz: probabilities only
        p = np.clip(z["preds"].astype(np.float64), 1e-12, 1 - 1e-12)
        lg = np.log(p) - np.log1p(-p)
    y = z["label"] if "label" in z.files else z["labels"]
    paths = z["img_path"].astype(str) if "img_path" in z.files else None
    return lg, y.astype(int), paths


# ------------------------------------------------------------------------------------------------ axes
AGE_AIF = {0: "<25", 1: "25-44", 2: "45-64", 3: "65+"}
AGE_IF = {0: "<30", 1: "30-44", 2: "45-59", 3: "60+"}
AGE_V2 = {0: "young", 1: "middle", 2: "senior"}
ITA4 = {0: "ITA-Q1 (darkest)", 1: "ITA-Q2", 2: "ITA-Q3", 3: "ITA-Q4 (lightest)"}
MST3 = {0: "MST 1-3", 1: "MST 4-6", 2: "MST 7-10"}


def _map(x, d):
    return np.array([d.get(int(v)) if np.isfinite(v) else None for v in np.asarray(x, dtype=float)], dtype=object)


FAIRFACE = f"{V}/labels_fairface"          # written by the FairForensics-protocol agent (read only here)
FF_FILE = {"ffpp_test": "ffpp_test.csv", "dfdc": "dfdc.csv", "celebdf": "celebdf.csv", "dfd": "dfd.csv", "ff2_test": "ff2pub_test.csv"}


def fairface_age(L, s):
    """FairFace-classifier age, 0-29 / 30+ (FairForensics/GenFace protocol): video majority `vid_age2` on video sets,
    per-image `age2` on FF2. None when the file or the row is missing."""
    f = f"{FAIRFACE}/{FF_FILE[s]}"
    if not os.path.exists(f):
        return None
    col = "age2" if s == "ff2_test" else "vid_age2"
    d = pd.read_csv(f, usecols=["img_path", col]).drop_duplicates("img_path").set_index("img_path")[col]
    v = d.reindex(L.img_path.values).values
    return np.array([("age " + x) if isinstance(x, str) else None for x in v], dtype=object)


def axes_for(L, s):
    six = SCHEME[s] == "six"
    g = np.where(L.cl_male.values == 1, "male", "nonmale").astype(object)
    r = L.cl_race.values.astype(object)
    if six:
        r = np.where(r == "asian", "others", r)
    gx = np.where(L.xu_male.values == 1, "male", "nonmale").astype(object)
    rx = L.xu_race.values.astype(object)
    ax = {"gender": g, "race": r, "gender_x_race": np.array([a + "," + b for a, b in zip(g, r)], dtype=object),
          "skin_ita4": _map(L.ita4, ITA4)}
    if s == "ff2_test":
        ax["age"] = _map(L.age4_if, AGE_IF)
    else:
        ax["age"] = _map(L.age4_aif, AGE_AIF)
        ax["age_if (sens)"] = _map(L.age4_if, AGE_IF)
        ax["age_v2 (sens)"] = _map(L.cl_age_v2, AGE_V2)
        ax["skin_mst3_aif (sens)"] = _map(L.mst3_vid, MST3)
        ax["gender_xu (sens)"] = gx
        ax["race_xu (sens)"] = rx
        ax["gender_x_race_xu (sens)"] = np.array([a + "," + b for a, b in zip(gx, rx)], dtype=object)
    if s != "ff2_test":
        gv = np.where(L.cl_male_vid.values == 1, "male", "nonmale").astype(object)
        rv = L.cl_race_vid.values.astype(object)
        if six:
            rv = np.where(rv == "asian", "others", rv)
        ax["gender_vidmaj (sens)"] = gv
        ax["race_vidmaj (sens)"] = rv
        ax["gender_x_race_vidmaj (sens)"] = np.array([a + "," + b for a, b in zip(gv, rv)], dtype=object)
    fa = fairface_age(L, s)
    if fa is not None:
        ax["age_fairface2 (sens)"] = fa
    ax["skin_ita4_raw (sens)"] = _map(L.ita4_raw, ITA4)
    ax["skin_ita4_gwexp (sens)"] = _map(L.ita4_gwexp, ITA4)
    ax["skin_ita4_local (sens)"] = _map(L.ita4_local, ITA4)
    ax["gender_x_age (expl)"] = np.array([a + "," + b if b else None for a, b in zip(g, ax["age"])], dtype=object)
    ax["gender_x_skin (expl)"] = np.array([a + "," + b if b else None for a, b in zip(g, ax["skin_ita4"])], dtype=object)
    return ax


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(f"{V}/results", exist_ok=True)
    t0 = time.time()
    SETS = [s for s in TEST + ["ff2_test"] if os.path.exists(f"{V}/labels/labels_{s}.csv")]
    labels = {s: pd.read_csv(f"{V}/labels/labels_{s}.csv", low_memory=False) for s in SETS}
    if "ff2_test" in labels:
        labels["ff2_test"] = labels["ff2_test"][labels["ff2_test"].cell_valid.astype(bool)].reset_index(drop=True)
    M = registry()
    fo = json.load(open(f"{A}/results/fair_oppoint/fair_oppoint.json"))["thresholds"]
    stored = {("CFES-D", s): fo[f"CFES-D s{s}"]["t"]["A1"] for s in (7, 17, 27)}
    stored[("Ding repro (SFD+GDA)", 7)] = fo["Ding repro"]["t"]["A1"]
    for m in M:                                                    # CESel runs store their own A1 (eval/a1.json)
        if m["family"] == "CESel" and os.path.exists(f"{m['run']}/eval/a1.json"):
            stored[(m["model"], m["seed"])] = json.load(open(f"{m['run']}/eval/a1.json"))["t"]
    # ---- thresholds
    thr, sanity = [], []
    for m in M:
        m["t05"] = -m["b05"]
        m["A1"], m["a1_src"] = None, "none (no FF++ val scores)"
        if m["val"]:
            z, y, _ = load_scores(m["val"])
            g = np.load(m["val"], allow_pickle=True)["group"].astype(int)
            u, fpr, tpr, f = val_curves(z, y, g, "eight")
            j = pick_a1(u, fpr, tpr, f)
            t_rule = float(u[j]) if j is not None else None
            key = (m["model"], m["seed"])
            if key in stored:
                m["A1"], m["a1_src"] = float(stored[key]), "stored"
                sanity.append({"check": f"A1 re-derived == stored: {key}", "stored": stored[key], "recomputed": t_rule,
                               "pass": t_rule is not None and abs(t_rule - stored[key]) < 1e-4})
            else:
                m["A1"], m["a1_src"] = t_rule, "rule re-run on saved val logits (Xu labels)"
            if j is not None:
                m["val_at_A1"] = {"FPR": 100 * float(fpr[j]), "TPR": 100 * float(tpr[j]), "i_F_FPR": float(f[j])}
        thr.append({"model": m["model"], "seed": m["seed"], "t05_logit": m["t05"], "A1_logit": m["A1"], "a1_src": m["a1_src"],
                    **{f"val_{k}@A1": v for k, v in m.get("val_at_A1", {}).items()}})
    pd.DataFrame(thr).to_csv(f"{V}/results/thresholds.csv", index=False, float_format="%.5g")
    print("thresholds done", round(time.time() - t0, 1), "s")
    # ---- per set
    rng = np.random.default_rng(2026)
    rows_ax, rows_g, boots = [], [], {}
    for s in SETS:
        L = labels[s]
        ax = axes_for(L, s)
        cl = L.video.astype(str).values
        person = L.person.astype(str).values
        u, cl_idx = np.unique(cl, return_inverse=True)
        K = len(u)
        W = MM.multinomial_weights(K, a.B, rng)
        boots[s] = {}
        for m in M:
            if s not in m["sets"]:
                continue
            z, y, paths = load_scores(m["sets"][s])
            if s == "ff2_test":
                full = pd.read_csv(f"{R}/data_ding_schema/fairforge2_full/test.csv", usecols=["img_path"]).img_path.values
                if paths is None:
                    paths = full
                idx = pd.Series(np.arange(len(paths)), index=paths)
                sel = idx.reindex(L.img_path.values).values
                assert np.isfinite(sel).all(), (m["model"], "ff2 rows missing")
                sel = sel.astype(int)
                z, y = z[sel], y[sel]
            else:
                if paths is None:
                    raise SystemExit(f"{m['sets'][s]} has no img_path")
                assert len(paths) == len(L) and all(os.path.basename(p) == os.path.basename(q) for p, q in zip(paths, L.img_path.values)), (m["model"], s, "row order")
            assert (y == L.label.values).all(), (m["model"], s, "labels")
            for tn in ("t05", "A1"):
                t = m[tn]
                if t is None:
                    continue
                pred = z >= t
                for an, grp in ax.items():
                    res = MM.axis_metrics(y, z, t, grp, cl, person)
                    names = sorted(res["groups"])
                    fe = np.array([res["groups"][n]["fpr_elig"] for n in names])
                    te = np.array([res["groups"][n]["tpr_elig"] for n in names])
                    Cc, P = MM.cluster_counts(y, pred, grp, cl_idx, K, names)
                    bt = MM.boot_rate_metrics(Cc, P, W, fe, te)
                    boots[s][(m["model"], m["seed"], tn, an)] = {k: bt[k] for k in ("F_FPR", "fpr_gap", "tpr_gap", "EO_gap", "FPR", "TPR")}
                    row = {"set": s, "model": m["model"], "family": m["family"], "seed": m["seed"], "threshold": tn, "t": t,
                           "a1_src": m["a1_src"], "axis": an, **res["pooled"]}
                    for k, v in res["min_support"].items():
                        row[k] = v
                    for k in ("F_FPR", "fpr_gap", "tpr_gap", "EO_gap"):
                        row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanpercentile(bt[k], 2.5), np.nanpercentile(bt[k], 97.5)) \
                            if np.isfinite(bt[k]).any() else (np.nan, np.nan)
                    for k, v in res["ding_compat"].items():
                        row[f"dc_{k}"] = v
                    rows_ax.append(row)
                    for j, n in enumerate(names):
                        d = res["groups"][n]
                        gf, gt_ = bt["group_FPR"][:, j], bt["group_TPR"][:, j]
                        rows_g.append({"set": s, "model": m["model"], "seed": m["seed"], "threshold": tn, "axis": an, "group": n,
                                       **{k: v for k, v in d.items()},
                                       "FPR_lo": np.nanpercentile(gf, 2.5) if np.isfinite(gf).any() else np.nan,
                                       "FPR_hi": np.nanpercentile(gf, 97.5) if np.isfinite(gf).any() else np.nan,
                                       "TPR_lo": np.nanpercentile(gt_, 2.5) if np.isfinite(gt_).any() else np.nan,
                                       "TPR_hi": np.nanpercentile(gt_, 97.5) if np.isfinite(gt_).any() else np.nan})
        print(s, "done", round(time.time() - t0, 1), "s")
    df = pd.DataFrame(rows_ax)
    for c in ("FPR", "TPR", "PR", "AUC"):
        pass
    df.to_csv(f"{V}/results/rescore_axes.csv", index=False, float_format="%.4f")
    G = pd.DataFrame(rows_g)
    for c in ("FPR", "TPR", "PR", "AUC"):
        G[c] = 100 * G[c]
    G.to_csv(f"{V}/results/rescore_groups.csv", index=False, float_format="%.4f")
    # ---- sanity: ding_compat reproduces the stored A1 numbers (Xu labels, fair_oppoint.json)
    fj = json.load(open(f"{A}/results/fair_oppoint/fair_oppoint.json"))["sets"]
    for s in TEST:
        for nm, key in (("CFES-D", "CFES-D s{}"), ("Ding repro (SFD+GDA)", "Ding repro")):
            for sd in ((7, 17, 27) if nm == "CFES-D" else (7,)):
                q = df[(df.set == s) & (df.model == nm) & (df.seed == sd) & (df.threshold == "A1")]
                pm = fj[s]["per_model"][key.format(sd)]["A1"]
                for ax_, k in (("gender_x_race_xu (sens)", "i_F_FPR"), ("gender_xu (sens)", "g_F_FPR"), ("race_xu (sens)", "r_F_FPR")):
                    v = float(q[q.axis == ax_].dc_F_FPR.iloc[0])
                    sanity.append({"check": f"{s} {nm} s{sd} A1 {k} (Xu, ding_compat) vs fair_oppoint.json", "stored": pm[k],
                                   "recomputed": v, "pass": abs(v - pm[k]) < 1e-6})
                v = float(q.TPR.iloc[0])
                sanity.append({"check": f"{s} {nm} s{sd} A1 pooled TPR", "stored": pm["TPR"], "recomputed": v, "pass": abs(v - pm["TPR"]) < 1e-6})
    pd.DataFrame(sanity).to_csv(f"{V}/results/sanity_checks.csv", index=False)
    print("sanity pass", sum(x["pass"] for x in sanity), "/", len(sanity))
    np.savez_compressed(f"{V}/results/boot_cache.npz", **{f"{s}|{k[0]}|{k[1]}|{k[2]}|{k[3]}|{mk}": v[mk].astype(np.float32)
                                                          for s, d in boots.items() for k, v in d.items() for mk in v})
    json.dump({"B": a.B, "rng": 2026, "models": [{k: v for k, v in m.items() if k in ("model", "family", "seed", "run", "a1_src", "A1", "t05")} for m in M],
               "runtime_s": time.time() - t0}, open(f"{V}/results/rescore_meta.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
