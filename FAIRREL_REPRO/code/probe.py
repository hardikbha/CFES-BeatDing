"""FAIRREL_REPRO demographic probe (PREREG section 8). FF++ VAL features only.
Feature files: runs/valfeats_frozen_clip.npz (untrained FairReLDetector == frozen CLIP), runs/{FAIRREL,CTRL}_s*/
valfeats_best.npz. Probe = StandardScaler + multinomial LogisticRegression(C=1, lbfgs, max_iter 3000), 5-fold
GroupKFold on identity components. Out-of-fold acc / balanced acc vs training-fold-majority baseline and 1/K chance;
2,000-rep component-cluster bootstrap CIs.
usage: pulid python probe.py [nproc]
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"
NPROC = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SUBGROUPS = ["M-W", "M-A", "M-B", "F-W", "F-A", "F-B"]
REPS = 2000


def components(clusters):
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for c in set(clusters):
        p = c.split("_")
        if p[0] == "youtube":
            f(p[1])
        else:
            par.setdefault(p[-2], p[-2]); par.setdefault(p[-1], p[-1])
            par[f(p[-2])] = f(p[-1])
    return np.array([f(c.split("_")[1]) if c.startswith("youtube") else f(c.split("_")[-1]) for c in clusters])


val = pd.read_csv(f"{A}/data/ffpp_val.csv")
COMP = components(val["cluster"].astype(str).tolist())
SG = np.array([SUBGROUPS.index(s) for s in val["subgroup"]])
TARGETS = {"subgroup": SG, "gender": (SG >= 3).astype(int), "race": SG % 3}   # race: 0 W, 1 A, 2 B
Y = val["label"].values.astype(int)


def load_feats():
    F = {}
    p = f"{A}/runs/valfeats_frozen_clip.npz"
    z = np.load(p)
    assert (z["path"].astype(str) == val["path"].values).all() and (z["g"] == SG).all()
    F[("frozen_clip", 0, "z")] = z["z"]
    for arm in ["FAIRREL", "CTRL"]:
        for sd in [7, 17, 27]:
            z = np.load(f"{A}/runs/{arm}_s{sd}/valfeats_best.npz")
            assert (z["path"].astype(str) == val["path"].values).all() and (z["g"] == SG).all()
            F[(arm, sd, "z")] = z["z"]
            F[(arm, sd, "zd")] = z["zd"]
            s = np.clip(z["score"].astype(np.float64), 1e-7, 1 - 1e-7)
            F[(arm, sd, "logit")] = np.log(s / (1 - s))[:, None]
    return F


def oof(X, t, groups):
    pred = np.empty(len(t), int)
    maj = np.empty(len(t), int)
    for tr, te in GroupKFold(5).split(X, t, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
        clf.fit(X[tr], t[tr])
        pred[te] = clf.predict(X[te])
        maj[te] = np.bincount(t[tr]).argmax()
    return pred, maj


def bal_acc(t, p, K):
    r = [np.mean(p[t == k] == k) for k in range(K) if (t == k).any()]
    return float(np.mean(r))


def job(key, tname, subset, X):
    t_all = TARGETS[tname]
    m = np.ones(len(t_all), bool) if subset == "all" else (Y == 0)
    X, t, comp = X[m], t_all[m], COMP[m]
    K = len(np.unique(t_all))
    pred, maj = oof(X, t, comp)
    uc, inv = np.unique(comp, return_inverse=True)
    mem = [np.where(inv == c)[0] for c in range(len(uc))]
    rng = np.random.default_rng([2026, len(t)])
    bs = []
    for _ in range(REPS):
        idx = np.concatenate([mem[i] for i in rng.integers(0, len(mem), len(mem))])
        a, b = np.mean(pred[idx] == t[idx]), np.mean(maj[idx] == t[idx])
        ba = bal_acc(t[idx], pred[idx], K)
        bs.append((a, b, ba, a - b, ba - 1.0 / K))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    acc, mj, ba = float(np.mean(pred == t)), float(np.mean(maj == t)), bal_acc(t, pred, K)
    return dict(features=key[0], seed=key[1], view=key[2], target=tname, frames=subset, n=int(len(t)),
                n_components=len(uc), K=K, acc=acc, acc_lo=lo[0], acc_hi=hi[0], majority_acc=mj,
                acc_minus_majority=acc - mj, amm_lo=lo[3], amm_hi=hi[3], bal_acc=ba, bal_lo=lo[2], bal_hi=hi[2],
                chance=1.0 / K, bal_minus_chance=ba - 1.0 / K, bmc_lo=lo[4], bmc_hi=hi[4],
                present=bool(lo[4] > 0.05 and lo[3] > 0))


def main():
    F = load_feats()
    tasks = [(k, tn, ss) for k in F for tn in TARGETS for ss in ["all", "real"]]
    print(f"{len(tasks)} probe fits; val n={len(Y)} components={len(np.unique(COMP))}", flush=True)
    res = Parallel(n_jobs=NPROC, verbose=5)(delayed(job)(k, tn, ss, F[k]) for k, tn, ss in tasks)
    df = pd.DataFrame(res)
    df.to_csv(f"{A}/results/probe_results.csv", index=False)
    L = ["Demographic linear probe on FF++ val (5-fold GroupKFold on identity components; OOF; 2000-rep component "
         "bootstrap). present = (bal-chance) CI lo > 0.05 AND (acc-majority) CI lo > 0", ""]
    for ss in ["all", "real"]:
        for tn in TARGETS:
            L.append(f"== target {tn}, frames {ss} ==")
            sub = df[(df.target == tn) & (df.frames == ss)].sort_values(["view", "features", "seed"])
            for _, r in sub.iterrows():
                L.append(f"  {r.features:11s} s{r.seed:<2d} {r.view:5s} acc {r.acc:.3f} [{r.acc_lo:.3f},{r.acc_hi:.3f}] "
                         f"maj {r.majority_acc:.3f} (d {r.acc_minus_majority:+.3f} [{r.amm_lo:+.3f},{r.amm_hi:+.3f}]) | "
                         f"bal {r.bal_acc:.3f} [{r.bal_lo:.3f},{r.bal_hi:.3f}] chance {r.chance:.3f} | "
                         f"{'PRESENT' if r.present else 'absent'}")
    open(f"{A}/results/PROBE_SUMMARY.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
