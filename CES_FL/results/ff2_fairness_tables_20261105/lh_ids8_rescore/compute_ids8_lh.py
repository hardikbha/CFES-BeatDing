"""8-IDS FairForensics-style rows for CES-FL L_A / L_H (Xception plain head, FF++-trained, seed 17).

Re-scores ALREADY SAVED per-image test logits (lh_tables_s17_20261103T133000Z, locked 3-Nov evaluation). No inference,
no training, CPU only. Nothing is selected on test: thresholds are (a) the checkpoint's stored FF++ val-real 10%-FPR
threshold, (b) t05 = logit >= 0, (c) A1 = the 22-Oct rule re-run on the checkpoint's own FF++ validation logits with Xu
labels (rescore.py / fair_oppoint.py code, imported unchanged).
Labels / metric: DEMO_AXES/code/fairforensics_ids.py load_labels + ids8_metrics (imported unchanged), labels "main"
(video-majority FairFace IDS), rule min_support. Bootstrap CIs replicate fairforensics_ids.py main() (B=1000, rng 2026,
same set order, so the same cluster weights).
Writes only into this directory: IDS8_LH_ROWS.csv (aggregates).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
sys.path.insert(0, f"{V}/code")
import fairforensics_ids as FI  # noqa: E402
import rescore as RS  # noqa: E402  (pick_a1, val_curves, load_scores)
import multiaxis_metrics as MM  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
LT = "/home/sachin.chaudhary/ces_fl_runs/lh_tables_s17_20261103T133000Z"
PH = "/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z"
SETS = {"ffpp_test": "ffpp", "dfdc": "dfdc", "celebdf": "celebdf", "dfd": "dfd"}
PUB_AUC = {("L_A", "ffpp_test"): 98.471, ("L_H", "ffpp_test"): 98.489, ("L_A", "dfdc"): 61.769, ("L_H", "dfdc"): 60.456,
           ("L_A", "celebdf"): 70.747, ("L_H", "celebdf"): 71.404, ("L_A", "dfd"): 81.156, ("L_H", "dfd"): 80.879}
ARMS = ["L_A", "L_H"]
log = {"checks": {}}


def chk(k, v):
    log["checks"][k] = v
    print(k, v, flush=True)


# ---------------------------------------------------------------- Xu val groups (same field the other rows' A1 used)
xu_src = f"{RS.A}/runs_rev4/clip_D_w0.3_s7/eval/val_scores.npz"
xz = np.load(xu_src, allow_pickle=True)
xu = pd.Series(xz["group"].astype(int), index=xz["img_path"].astype(str))
xl = pd.Series(xz["label"].astype(int), index=xz["img_path"].astype(str))
assert xu.index.is_unique
alt = np.load(f"{RS.A}/runs_rev4/clip_C_w0.3_s7/eval/val_scores.npz", allow_pickle=True)
alt_g = pd.Series(alt["group"].astype(int), index=alt["img_path"].astype(str)).reindex(xu.index)
chk("xu_val_groups_equal_across_two_registry_val_npz", bool((alt_g.values == xu.values).all()))

# code check: re-derive one registry model's A1 with the imported rule and compare with thresholds_used.csv
z, y, _ = RS.load_scores(f"{RS.A}/runs_rev4/clip_C_w0.3_s7/eval/val_scores.npz")
j = RS.pick_a1(*RS.val_curves(z, y, alt["group"].astype(int), "eight"))
tu = pd.read_csv(f"{V}/results/fairforensics_ids/thresholds_used.csv")
ref = float(tu[(tu.model == "arm C") & (tu.seed == 7)].A1_logit.iloc[0])
chk("a1_rule_rederives_armC_s7", {"ours": float(np.unique(z)[j]), "thresholds_used.csv": ref,
                                  "match_5sig": bool(abs(float(np.unique(z)[j]) - ref) <= 1e-4 * max(1, abs(ref)))})

XU8 = {("male", "asian"): 0, ("male", "white"): 1, ("male", "black"): 2, ("male", "others"): 3,
       ("female", "asian"): 4, ("female", "white"): 5, ("female", "black"): 6, ("female", "others"): 7}

thr = {}
for a in ARMS:
    vd = json.load(open(f"{PH}/{a}/validation_scores_PRIVATE.json"))
    assert vd["split"] == "validation" and vd["test_read"] is False
    paths = np.array([r["image"] for r in vd["rows"]])
    yv = np.array([int(r["label"]) for r in vd["rows"]])
    zv = np.asarray(vd["logits"], dtype=np.float64)
    gv = xu.reindex(paths)
    cov = float(gv.notna().mean())
    assert pd.Index(paths).is_unique and cov == 1.0, (a, cov)
    gv = gv.values.astype(int)
    assert (xl.reindex(paths).values == yv).all()
    # independent cross-check against the Ding/Lin group string stored in the validation rows
    gr = np.array([XU8.get(tuple(s.lower().split("-")), -1) for s in (r["gender_race"] for r in vd["rows"])])
    chk(f"{a}_val_rows", {"n": int(len(yv)), "xu_join_coverage": cov,
                          "agree_with_row_gender_race": float((gr == gv).mean())})
    u, fpr, tpr, f = RS.val_curves(zv, yv, gv, "eight")
    jj = RS.pick_a1(u, fpr, tpr, f)
    a1 = float(u[jj]) if jj is not None else None
    # stored threshold re-derived on val as a check
    st = float(vd["threshold"])
    chk(f"{a}_thresholds", {"stored_valreal10": st, "val_FPR_at_stored": float((zv[yv == 0] >= st).mean()),
                            "A1": a1, "val_FPR_at_A1": float(fpr[jj]) if jj is not None else None,
                            "val_TPR_at_A1": float(tpr[jj]) if jj is not None else None,
                            "val_xu_FFPR_at_A1": float(f[jj]) if jj is not None else None})
    thr[a] = {"valreal10": st, "t05": 0.0, "A1": a1}

# ---------------------------------------------------------------- test re-scoring
rng = np.random.default_rng(2026)
rows = []
for s, d in SETS.items():
    L = FI.load_labels(s)
    cl = L.video.astype(str).values
    person = L.person.astype(str).values
    uu, cl_idx = np.unique(cl, return_inverse=True)
    W = MM.multinomial_weights(len(uu), 1000, rng)     # same draw order as fairforensics_ids.py main()
    for a in ARMS:
        sd = json.load(open(f"{LT}/inference/{a}/{d}/scores_PRIVATE.json"))
        assert abs(float(sd["threshold"]) - thr[a]["valreal10"]) < 1e-12
        sp = np.array([r["image"] for r in sd["rows"]])
        ys = np.array([int(r["label"]) for r in sd["rows"]])
        zs = np.asarray(sd["logits"], dtype=np.float64)
        same_order = len(sp) == len(L) and bool((sp == L.img_path.values).all())
        if same_order:
            z, y = zs, ys
            cov = 1.0
        else:                                           # join on img_path (unique paths must carry one logit)
            df = pd.DataFrame({"p": sp, "z": zs, "y": ys})
            gz = df.groupby("p").agg(z=("z", "first"), zmax=("z", "max"), zmin=("z", "min"), y=("y", "first"))
            assert (gz.zmax - gz.zmin).abs().max() < 1e-6
            m = gz.reindex(L.img_path.values)
            cov = float(m.z.notna().mean())
            assert cov >= 0.99, (a, s, cov)
            z, y = m.z.values, m.y.values
        assert (y == L.label.values).all(), (a, s, "labels")
        # agreement of the score file's own ids8 field with the FairFace video-majority IDS (informational)
        rid = np.array([str(r.get("ids8", "")) for r in sd["rows"]])
        known = np.array([v.isdigit() for v in rid])
        ffi = L.vid_ids8.astype(int).astype(str).values if same_order else None
        ids_agree = float((rid[known] == ffi[known]).mean()) if same_order and known.any() else None
        ids = L["ids_main"].values.astype(object)
        auc_chk = None
        for tn in ("valreal10", "t05", "A1"):
            t = thr[a][tn]
            if t is None:
                continue
            res = FI.ids8_metrics(y, z, t, ids, cl, person)
            names = [n for n in FI.IDS8 if n in res["groups"]]
            fe = np.array([res["groups"][n]["fpr_elig"] for n in names])
            te = np.array([res["groups"][n]["tpr_elig"] for n in names])
            Cc, P = MM.cluster_counts(y, z >= t, ids, cl_idx, len(uu), names)
            bt = MM.boot_rate_metrics(Cc, P, W, fe, te)
            ms = res["min_support"]
            row = {"set": s, "model": f"CES-FL {a} (Xception plain head, FF++)", "arm": a, "seed": 17, "threshold": tn,
                   "t_logit": t, "labels": "main", "rule": "min_support", "n_seeds": 1,
                   **{k: res["pooled"][k] for k in ("AUC", "FPR", "TPR", "n", "n_real", "n_fake", "n_unlabeled")},
                   **{k: ms[k] for k in ("F_FPR", "F_MEO", "F_DP", "F_OAE", "es_AUC", "fpr_gap", "tpr_gap", "worst_fpr",
                                         "worst_tpr", "n_fpr_elig", "n_tpr_elig", "n_both_elig")}}
            for k in ("F_FPR", "fpr_gap", "tpr_gap"):
                v = bt[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) \
                    if np.isfinite(v).any() else (np.nan, np.nan)
            row["join_coverage"] = cov
            row["AUC_published"] = PUB_AUC[(a, s)]
            row["AUC_abs_diff"] = abs(res["pooled"]["AUC"] - PUB_AUC[(a, s)])
            rows.append(row)
            auc_chk = row["AUC_abs_diff"]
        chk(f"{a}_{s}", {"rows": int(len(sp)), "same_row_order_as_protocol": same_order, "join_coverage": cov,
                         "AUC_abs_diff_vs_published": round(auc_chk, 5), "AUC_within_0.01": auc_chk <= 0.01,
                         "score_file_ids8_known_frac": float(known.mean()), "score_file_ids8_agree_fairface": ids_agree})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/IDS8_LH_ROWS.csv", index=False, float_format="%.3f")
json.dump({"thresholds": thr, **log}, open(f"{OUT}/checks.json", "w"), indent=1, default=str)
cols = ["set", "arm", "threshold", "AUC", "FPR", "TPR", "F_FPR", "F_MEO", "F_DP", "F_OAE", "n_fpr_elig", "n_tpr_elig",
        "n_both_elig"]
print(df[cols].round(2).to_string(index=False))
