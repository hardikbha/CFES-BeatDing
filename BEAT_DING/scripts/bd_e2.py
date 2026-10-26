"""E2 of the Beat-Ding Xception arm (design agents/BEAT_DING/XCEPTION_DESIGN.md 3.8; plan Amendment 4b): our X-BD run
against OUR REPRODUCTION of Ding's official code (DINGREPRO_LIN), paired, on the same test rows. CPU, saved scores only.

Per seed s and test set: Ding's 10 columns of X-BD s (its FF++-val matched threshold, plan section 4) and of the Ding
reproduction seed s (DING_full s7 = the released, unmodified run, at its own 0.5 = the matched point by definition;
s17 / s27 = the D2 "fast-loader" reruns, at the same matched rule applied to their own FF++ val scores when present,
else at their own 0.5), the delta X - Ding per column and a video-cluster PAIRED bootstrap 95 % CI of the delta:
5,000 resamples (seed 2026) of the test set's video clusters (cfes.data.cluster_key, Codex F8) with replacement, the
SAME resampled rows for both models; both thresholds stay fixed (chosen on validation). Under X0 both models start from
the same phase-1 network (DING_base_s<s>/49.pth), so the comparison isolates Ding's phase 2 (SFD + GDA) vs BD's.

Bootstrap metrics use multiplicity weights (a resampled row counted k times has weight k); the weighted columns equal
cfes.bd_metrics.cols exactly up to float rounding at unit weights (checked by self_check(), run in tests/bd_checks.py x6).

  python scripts/bd_e2.py --x_eval runs/X_BD3_s7/eval --x_t <matched t> --seed 7 [--n_boot 5000]     (normally called by
                                                                                                   bd_report.py final --family xception)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import bd_metrics as BM  # noqa: E402
from cfes.data import cluster_key  # noqa: E402
from cfes.metrics import partitions  # noqa: E402

DING_PATTERN = BM.DING_DIR + "/full_s{seed}_ep49_{set}.npz"      # DINGREPRO_LIN audit_scores naming (s7 exists)
DING_VAL_KEY = "ffval"
FAST_LOADER_SEEDS = (17, 27)                                     # plan Amendment 4a, D2 = (b)
N_BOOT, BOOT_SEED = 5000, 2026


def ding_files(seed, pattern=DING_PATTERN):
    """{set: npz} for the 4 test sets + 'val', or None when any test file is missing."""
    out = {k: pattern.format(seed=seed, set=v) for k, v in BM.DING_TEST_NPZ.items()}
    if not all(os.path.exists(f) for f in out.values()):
        return None
    v = pattern.format(seed=seed, set=DING_VAL_KEY)
    out["val"] = v if os.path.exists(v) else None
    return out


def ding_threshold(seed, files):
    """s7: its own 0.5 (logit 0; the matched point is defined by it). Others: the matched rule on their own val scores."""
    if seed == 7:
        return 0.0, "own 0.5 (defines the matched point)"
    if files.get("val"):
        zv, yv, _, _ = BM.load(files["val"])
        return BM.matched_threshold(zv, yv, BM.ding_val_reference()["tpr"]), "matched val TPR (own FF++ val)"
    return 0.0, "own 0.5 (no FF++ val scores found)"


class Model:
    """One model's logits on the common rows, pre-sorted once (ascending) with tie groups, for fast weighted AUCs."""

    def __init__(self, z, t):
        self.z = np.asarray(z, dtype=np.float64)
        self.pred = self.z >= t
        self.order = np.argsort(self.z, kind="mergesort")
        _, self.inv = np.unique(self.z[self.order], return_inverse=True)
        self.nt = int(self.inv.max()) + 1

    def auc(self, wf, wr):
        """Weighted Mann-Whitney AUC (ties count 1/2) = sklearn roc_auc_score(sample_weight) = roc_auc_score at w = 1."""
        F = np.bincount(self.inv, wf[self.order], minlength=self.nt)
        R = np.bincount(self.inv, wr[self.order], minlength=self.nt)
        sf, sr = F.sum(), R.sum()
        below = np.cumsum(R) - R
        return float((F * (below + 0.5 * R)).sum() / (sf * sr))


def wcols(m, y, g, scheme, w):
    """cfes.bd_metrics.cols with row weights w (multiplicities); only the 10 columns + TPR / FPR."""
    real, fake = (y == 0).astype(np.float64) * w, (y == 1).astype(np.float64) * w
    p = m.pred.astype(np.float64)
    FPR, TPR = (p * real).sum() / real.sum(), (p * fake).sum() / fake.sum()
    auc = m.auc(fake, real)
    out = {}
    for lv, pre in (("gender", "g"), ("race", "r"), ("intersection", "i")):
        st = []
        for name, ids in partitions(scheme)[lv].items():
            msk = np.isin(g, ids).astype(np.float64)
            rr, ff = real * msk, fake * msk
            if rr.sum() <= 0 or ff.sum() <= 0:
                continue
            st.append(((p * rr).sum() / rr.sum(), (p * msk * w).sum() / (msk * w).sum(), m.auc(ff, rr)))
        st = np.array(st)
        if len(st) == 0:                 # no group with both classes in this resample (tiny sets only): undefined
            out.update({pre + "F_FPR": np.nan, pre + "F_DP": np.nan, pre + "esAUC": np.nan})
            continue
        out[pre + "F_FPR"] = 100 * float(np.abs(st[:, 0] - FPR).sum())
        out[pre + "F_DP"] = 100 * float(max(st[:, 1].max() - st[:, 1].min(), (1 - st[:, 1]).max() - (1 - st[:, 1]).min()))
        out[pre + "esAUC"] = 100 * float(auc / (1 + np.abs(st[:, 2] - auc).sum()))
    out.update(AUC=100 * auc, TPR=100 * float(TPR), FPR=100 * float(FPR))
    return out


def paired_set(x_npz, x_t, d_npz, d_t, scheme, n_boot=N_BOOT, seed=BOOT_SEED):
    zx, yx, gx, px = BM.load(x_npz)
    zd, yd, gd, pd_ = BM.load(d_npz)
    if px is None or pd_ is None:
        raise ValueError("E2 needs img_path in both score files")
    pos = {p: i for i, p in enumerate(pd_)}
    idx = np.array([pos.get(p, -1) for p in px])
    if (idx < 0).any() or len(px) != len(pd_):
        raise ValueError(f"E2: rows differ between {x_npz} and {d_npz} ({int((idx < 0).sum())} unmatched)")
    zd, yd, gd = zd[idx], yd[idx], gd[idx]
    if not ((yd == yx).all() and (gd == gx).all()):
        raise ValueError("E2: labels / groups differ between the two score files")
    y, g = yx, gx
    mx, md = Model(zx, x_t), Model(zd, d_t)
    one = np.ones(len(y))
    cx, cd = wcols(mx, y, g, scheme, one), wcols(md, y, g, scheme, one)
    keys = np.array([cluster_key(p) for p in px])
    uk, kidx = np.unique(keys, return_inverse=True)
    rng = np.random.default_rng(seed)
    D = np.empty((n_boot, len(BM.K)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uk), len(uk)), minlength=len(uk)).astype(np.float64)[kidx]
        bx, bd = wcols(mx, y, g, scheme, w), wcols(md, y, g, scheme, w)
        D[b] = [bx[c] - bd[c] for c in BM.K]
    out = {"n": int(len(y)), "n_clusters": int(len(uk)), "x": {c: cx[c] for c in BM.K + ["TPR", "FPR"]},
           "ding": {c: cd[c] for c in BM.K + ["TPR", "FPR"]}, "delta": {}, "ci95": {}, "x_better": {}, "ci_excludes_0": {}}
    for i, c in enumerate(BM.K):
        d = cx[c] - cd[c]
        lo, hi = (float(v) for v in np.nanquantile(D[:, i], [0.025, 0.975]))
        out["delta"][c], out["ci95"][c] = d, [lo, hi]
        out["x_better"][c] = bool(d > 0) if BM.UP[i] else bool(d < 0)
        out["ci_excludes_0"][c] = bool(lo > 0 or hi < 0)
    out["n_boot_undefined"] = int(np.isnan(D).any(1).sum())
    out["n_x_better"] = int(sum(out["x_better"].values()))
    out["n_x_better_ci"] = int(sum(out["x_better"][c] and out["ci_excludes_0"][c] for c in BM.K))
    out["n_x_worse_ci"] = int(sum((not out["x_better"][c]) and out["ci_excludes_0"][c] for c in BM.K))
    return out


def e2_seed(x_eval_dir, x_t, seed, n_boot=N_BOOT, pattern=DING_PATTERN):
    """Paired E2 for one seed over the 4 test sets, or None when the Ding reproduction of that seed has no scores."""
    files = ding_files(seed, pattern)
    if files is None:
        return None
    d_t, d_src = ding_threshold(seed, files)
    res = {"seed": seed, "ding_run": "DING_full (released, unmodified)" if seed == 7 else "DING_full fast-loader (D2 = b)",
           "fast_loader": seed in FAST_LOADER_SEEDS, "ding_t": d_t, "ding_t_source": d_src, "x_t": float(x_t),
           "n_boot": n_boot, "boot_seed": BOOT_SEED, "sets": {}}
    for k, sch in BM.SETS.items():
        res["sets"][k] = paired_set(os.path.join(x_eval_dir, f"{k}_scores.npz"), x_t, files[k], d_t, sch, n_boot)
    res["n_x_better"] = sum(v["n_x_better"] for v in res["sets"].values())
    res["n_x_better_ci"] = sum(v["n_x_better_ci"] for v in res["sets"].values())
    res["n_x_worse_ci"] = sum(v["n_x_worse_ci"] for v in res["sets"].values())
    return res


def self_check(n=4000, seed=0):
    """Weighted columns at unit weights == cfes.bd_metrics.cols (|d| < 1e-9), incl. ties and absent groups."""
    rng = np.random.default_rng(seed)
    out = {}
    for sch, ng in (("eight", 8), ("six", 6)):
        y = rng.integers(0, 2, n)
        g = rng.integers(0, ng, n)
        z = np.round(rng.normal(y * 1.0 + 0.2 * g, 1.0), 1)          # rounding -> many ties
        y[(g == 2) & (y == 0)] = 1                                      # one group without reals -> skipped
        for t in (0.0, 0.7):
            a = BM.cols(y, z, g, sch, t)
            b = wcols(Model(z, t), y, g, sch, np.ones(n))
            out[f"{sch}_t{t}"] = max(abs(a[c] - b[c]) for c in BM.K + ["TPR", "FPR"])
    # integer weights == explicit row duplication
    y = rng.integers(0, 2, 500); g = rng.integers(0, 8, 500); z = np.round(rng.normal(y, 1.0), 1)
    w = rng.integers(0, 3, 500)
    rep = np.repeat(np.arange(500), w)
    a = BM.cols(y[rep], z[rep], g[rep], "eight", 0.3)
    b = wcols(Model(z, 0.3), y, g, "eight", w.astype(np.float64))
    out["weights_vs_duplication"] = max(abs(a[c] - b[c]) for c in BM.K + ["TPR", "FPR"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_eval", default=None)
    ap.add_argument("--x_t", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--self_check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        print(json.dumps(self_check()))
        return
    print(json.dumps(e2_seed(a.x_eval, a.x_t, a.seed, a.n_boot), default=float, indent=1))


if __name__ == "__main__":
    main()
