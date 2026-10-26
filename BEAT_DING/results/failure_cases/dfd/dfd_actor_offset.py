"""Follow-up: is the DFD error identity-driven? Per actor (real videos 0_<id>__*, fakes with id_a = <id>, i.e. the npz
'method' column; and fakes with id_b = <id>): median logit, FPR / FNR at the matched threshold, within-actor AUC
(actor's reals vs actor's fakes), plus Spearman of actor FPR vs actor FNR. Same for the Ding repro (logit >= 0).
Writes actor_offset.csv / actor_offset.json here. Read-only otherwise."""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
BD = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING"
AG = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
OUT = f"{BD}/results/failure_cases/dfd"
d = np.load(f"{BD}/runs/BD0_s7/eval/dfd_scores.npz", allow_pickle=True)
e = np.load(f"{AG}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_dfd.npz", allow_pickle=True)
t = json.load(open(f"{BD}/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]["matched"]
df = pd.DataFrame({"video": d["cluster"], "label": d["label"], "logit": d["logit"].astype(float), "ding": e["logit"].astype(float)})
ids = df.video.str[2:].str.split("__").str[0].str.split("_")
df["a"], df["b"] = ids.str[0], ids.str[1]
rows = []
for a in sorted(df[df.label == 0].a.unique()):
    r = df[(df.label == 0) & (df.a == a)]
    f = df[(df.label == 1) & (df.a == a)]
    fb = df[(df.label == 1) & (df.b == a)]
    yy = np.r_[np.zeros(len(r)), np.ones(len(f))]
    rows.append({"actor": a, "n_real": len(r), "n_fake_a": len(f), "n_fake_b": len(fb),
                 "real_med_logit": r.logit.median(), "fake_a_med_logit": f.logit.median(),
                 "FPR": 100 * (r.logit >= t).mean(), "FNR_a": 100 * (f.logit < t).mean(), "FNR_b": 100 * (fb.logit < t).mean(),
                 "AUC_within_a": 100 * roc_auc_score(yy, np.r_[r.logit, f.logit]),
                 "ding_FPR": 100 * (r.ding >= 0).mean(), "ding_FNR_a": 100 * (f.ding < 0).mean(),
                 "ding_AUC_within_a": 100 * roc_auc_score(yy, np.r_[r.ding, f.ding])})
T = pd.DataFrame(rows).sort_values("FPR", ascending=False)
T.round(2).to_csv(f"{OUT}/actor_offset.csv", index=False)
real_all = df[df.label == 0]
fake_known = df[(df.label == 1) & df.a.isin(T.actor)]
y = np.r_[np.zeros(len(real_all)), np.ones(len(fake_known))]
res = {"n_actors_with_real_videos": int(len(T)),
       "spearman_FPR_vs_FNR_a": list(map(float, spearmanr(T.FPR, T.FNR_a))),
       "spearman_real_med_vs_fake_a_med_logit": list(map(float, spearmanr(T.real_med_logit, T.fake_a_med_logit))),
       "ding_spearman_FPR_vs_FNR_a": list(map(float, spearmanr(T.ding_FPR, T.ding_FNR_a))),
       "mean_within_actor_AUC": float(T.AUC_within_a.mean()),
       "pooled_AUC_same_rows": 100 * float(roc_auc_score(y, np.r_[real_all.logit, fake_known.logit])),
       "ding_mean_within_actor_AUC": float(T.ding_AUC_within_a.mean()),
       "ding_pooled_AUC_same_rows": 100 * float(roc_auc_score(y, np.r_[real_all.ding, fake_known.ding])),
       "spread_real_med_logit_across_actors": [float(T.real_med_logit.min()), float(T.real_med_logit.max())]}
json.dump(res, open(f"{OUT}/actor_offset.json", "w"), indent=1)
print(T.round(1).to_string()); print(json.dumps(res, indent=1))
