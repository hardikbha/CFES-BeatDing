"""Builds results/MULTI_AXIS_RESULTS.md (+ results/seedmean_axes.csv, results/descriptive_H1_H4.csv) from the CSVs written
by build_labels.py, rescore.py and mechanism.py. Seed means: metric averaged over seeds; CI = percentile interval of the
seed-averaged bootstrap replicate (same cluster resample for every seed; thresholds fixed; test-sample uncertainty only).
"""
import json
import os

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
V = f"{R}/agents/DEMO_AXES"
RES = f"{V}/results"
LAB = f"{V}/labels"
ORDER = ["CFES-D", "arm C", "arm E w1", "arm E w3", "arm E w10", "CFES A (rev4)", "control (det-only)", "CFES-ResNet",
         "CESel B0", "CESel p1 full", "CESel p1 sens", "CESel p1 inv", "CESel p1 floor", "CESel p1 C_randcue",
         "CESel p1 C_shufgroup", "CESel p1 C_novalid", "CESel p1 B0_cfesD", "Ding repro (SFD+GDA)", "Ding Ori repro (SFD)", "FF2-trained phase 1", "FF2-trained control", "FF2-trained CFES"]
SETN = {"ffpp_test": "FF++ test", "dfdc": "DFDC", "celebdf": "Celeb-DF", "dfd": "DFD", "ff2_test": "FF2-public TEST"}
MAIN_AXES = ["gender", "race", "gender_x_race", "age", "skin_ita4"]
AXN = {"gender": "gender (AI-Face clean, frame level; FF2: FF2 labels)", "race": "race (AI-Face clean, frame level; FF2: FF2 labels)",
       "gender_x_race": "gender x race (AI-Face clean, frame level; FF2: FF2 labels)",
       "age": "age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)", "skin_ita4": "skin tone (ITA, 4 bins)"}


def f(x, d=1):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def ci(lo, hi, d=1):
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f" [{lo:.{d}f}, {hi:.{d}f}]"


def seed_means(ax, boot):
    keys = ["AUC", "FPR", "TPR", "F_FPR", "fpr_gap", "tpr_gap", "EO_gap", "EO_maxdev", "F_DP", "es_AUC", "worst_fpr",
            "worst_tpr", "n_groups", "n_fpr_elig", "n_tpr_elig", "n_both_elig", "dc_F_FPR"]
    rows = []
    for (s, m, tn, a), g in ax.groupby(["set", "model", "threshold", "axis"], sort=False):
        r = {"set": s, "model": m, "threshold": tn, "axis": a, "n_seeds": len(g), "seeds": ",".join(map(str, g.seed)),
             "a1_src": g.a1_src.iloc[0]}
        for k in keys:
            r[k] = float(g[k].mean())
            if len(g) > 1:
                r[k + "_sd"] = float(g[k].std(ddof=1))
        for k in ("F_FPR", "fpr_gap", "tpr_gap", "EO_gap"):
            arrs = [boot.get(f"{s}|{m}|{sd}|{tn}|{a}|{k}") for sd in g.seed]
            arrs = [x for x in arrs if x is not None]
            if arrs:
                mb = np.nanmean(np.stack(arrs), 0) if len(arrs) > 1 else arrs[0]
                r[k + "_lo"], r[k + "_hi"] = (float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))) \
                    if np.isfinite(mb).any() else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def mean_boot(boot, s, m, seeds, tn, a, k):
    arrs = [boot.get(f"{s}|{m}|{sd}|{tn}|{a}|{k}") for sd in seeds]
    arrs = [x for x in arrs if x is not None]
    return np.nanmean(np.stack(arrs), 0) if arrs else None


def main():
    ax = pd.read_csv(f"{RES}/rescore_axes.csv")
    G = pd.read_csv(f"{RES}/rescore_groups.csv")
    thr = pd.read_csv(f"{RES}/thresholds.csv")
    bc = np.load(f"{RES}/boot_cache.npz")
    boot = {k: bc[k] for k in bc.files}
    SM = seed_means(ax, boot)
    SM.to_csv(f"{RES}/seedmean_axes.csv", index=False, float_format="%.4f")
    seeds_of = ax.groupby("model").seed.apply(lambda s: sorted(set(s))).to_dict()
    edges = json.load(open(f"{LAB}/ita_bin_edges.json"))
    stab = pd.read_csv(f"{LAB}/ita_stability.csv")
    cov = pd.read_csv(f"{LAB}/coverage.csv")
    cons = pd.read_csv(f"{LAB}/ita_construct_checks.csv")
    hs = pd.read_csv(f"{LAB}/human_check_sample.csv")
    san = pd.read_csv(f"{RES}/sanity_checks.csv")
    mech = pd.read_csv(f"{RES}/mechanism_b0.csv") if os.path.exists(f"{RES}/mechanism_b0.csv") else None
    L = []
    w = L.append
    w("# Multi-axis (gender / race / age / skin tone) evaluation of every saved model")
    w("")
    w("Agent DEMO_AXES, 26 Oct 2026. CPU only: saved per-image logits, no retraining, no GPU, no existing result modified. "
      "Code `agents/DEMO_AXES/code/`, labels `labels/`, numbers `results/` (README.md there). "
      "Pre-registration: `decisions/2026-10-26-final-model-spec.md`.")
    w("")
    w("> **Status of every number here: EXPLORATORY / DESCRIPTIVE.** H1-H4 are the confirmatory family of the final-model "
      "spec and are *not tested* in this document: the final model (CESel-MA, or CFES-D on fallback) and the matched-CLIP "
      "baselines do not exist yet. Section 7 computes the H1-H4 *quantities* for existing models as descriptive values "
      "only (no Holm, no seed resampling, no NI decision). Intervals are video-cluster (FF2: person-cluster) percentile "
      "bootstraps with thresholds and models fixed (test-sample uncertainty only).")
    w("")
    if os.path.exists(f"{RES}/_summary_block.md"):                 # hand-written reading of the tables below
        w(open(f"{RES}/_summary_block.md").read().rstrip())
        w("")
    # ---------------------------------------------------------------- 1 labels
    w("## 1. Labels")
    w("")
    w("One CSV per split, keyed exactly like the score npz `img_path` (`labels/labels_<set>.csv`; column dictionary in "
      "`code/build_labels.py` docstring).")
    w("")
    w("| axis | source used here (primary) | sensitivity versions |")
    w("|---|---|---|")
    w("| gender, race | AI-Face cleaner labels (Lin et al. robust-fairness release, v2), frame level; Xu labels where "
      "missing (`cl_src = xu_fallback`: all of FF++ val). Celeb-DF/DFD: Asian -> Others (Ding six-group code book) | Xu labels (= protocol) |")
    w("| age | AI-Face v1 `Predicted Age`, video majority, 4 bins <25 / 25-44 / 45-64 / 65+ (child+youth / adult / middle / senior). "
      "FF2: manifest `if_age` (InsightFace, per image), bins <30 / 30-44 / 45-59 / 60+ | InsightFace buffalo_l age (ours, per-video median, "
      "spec bins <30/30-44/45-59/60+); AI-Face v2 young/middle/senior |")
    w("| skin tone | ITA of face-skin pixels (cheek + forehead ROI from SCRFD landmarks, YCrCb skin mask, 10-90 % L* trim), "
      "gray-world white balance estimated on the crop surround (chromatic only), per-video median; 4 bins at TRAIN quartiles | "
      "ITA without white balance; ITA with exposure normalisation; AI-Face MST 3-group (video majority) |")
    w("")
    w("**Deviation from the spec (flag).** The spec asks for age bins <30/30-44/45-59/60+ *from AI-Face predicted age*, but "
      "AI-Face v1 only provides a 5-class age (0-14/15-24/25-44/45-64/65+), so those edges cannot be formed from it. "
      "Primary on FF++/DFDC/CDF/DFD = AI-Face classes collapsed to 4 bins; the spec's numeric bins are reported with our "
      "InsightFace buffalo_l estimate (the same estimator that produced FF2's `if_age`). The coordinator should choose "
      "which becomes the selection/training age axis before 2 Nov.")
    w("")
    c2 = cov[cov.column.isin(["cl_src==aiface_v2|ff2", "aif_age5", "aif_mst", "if_age", "ita_gw", "ita4", "age4_aif", "age4_if"])]
    pv = c2.pivot_table(index="set", columns="column", values="frac").reindex(["ffpp_train", "ffpp_val", "ffpp_test", "dfdc", "celebdf", "dfd", "ff2_test"])
    w("Coverage (fraction of rows with a value):")
    w("")
    cols = [c for c in ["cl_src==aiface_v2|ff2", "aif_age5", "age4_aif", "if_age", "age4_if", "aif_mst", "ita_gw", "ita4"] if c in pv.columns]
    w("| set | rows | " + " | ".join(cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    nrows = cov.drop_duplicates("set").set_index("set").rows
    for s, r in pv.iterrows():
        w(f"| {s} | {int(nrows.get(s, 0))} | " + " | ".join(f(100 * r[c], 1) for c in cols) + " |")
    w("")
    w("`cl_src==aiface_v2|ff2` = share of rows with the cleaner (or FF2) gender/race label; the rest fall back to Xu. "
      "`ita_gw` = frames with a valid frame ITA; `ita4` = rows whose video has a valid median ITA.")
    w("")
    w("### ITA bin edges (declared from TRAIN quartiles before any score was read)")
    w("")
    w("| protocol | variant | q25 | q50 | q75 | TRAIN units |")
    w("|---|---|---|---|---|---|")
    for pk, unit in (("ffpp_protocol", "n_real_train_videos"), ("ff2", "n_real_train_persons")):
        for vk, d in edges[pk].items():
            e = d["edges"]
            w(f"| {pk} | {vk}{' (primary)' if vk == 'ita_gw' else ''} | {e[0]:.1f} | {e[1]:.1f} | {e[2]:.1f} | {d[unit]} |")
    w("")
    w("Bins: Q1 = ITA < q25 (darkest) ... Q4 = ITA >= q75 (lightest). FF++/DFDC/CDF/DFD use the FF++ TRAIN edges; FF2 uses the FF2-public TRAIN edges.")
    w("")
    w("### ITA stability within a video (frame ITA, primary variant)")
    w("")
    w("| set | videos (valid) | valid frames | frames/video | within-video SD (median) | within-video IQR (median) | between-video SD | ICC(1) | videos whose frames span >1 bin | frame-majority bin = median bin |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in stab.iterrows():
        w(f"| {r.set} | {int(r.videos)} ({int(r.videos_with_valid_ita)}) | {100 * r.frames_valid_frac:.1f}% | {r.median_frames_per_video:.0f} | "
          f"{r.within_video_sd_median:.1f} | {r.within_video_iqr_median:.1f} | {r.between_video_sd:.1f} | {r.icc1:.2f} | "
          f"{100 * r.videos_frames_span_gt1_bin:.0f}% | {100 * r.videos_frame_majority_bin_eq_median_bin:.0f}% |")
    w("")
    w("### ITA construct checks (label-only; NOT validation: race is not skin tone)")
    w("")
    w("| set | ITA variant | Spearman(ITA, -AI-Face MST) | median ITA White | Black | Asian | Others | AUC White-vs-Black |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in cons.iterrows():
        w(f"| {r.set} | {r.ita} | {f(r.get('spearman_vs_aif_mst(video median)', np.nan), 2)} | {f(r['median_white(cl_race)'])} | "
          f"{f(r['median_black(cl_race)'])} | {f(r['median_asian(cl_race)'])} | {f(r['median_others(cl_race)'])} | {f(r.auc_white_vs_black, 3)} |")
    w("")
    w(f"Human skin-tone check sample: `labels/human_check_sample.csv`, {len(hs)} reals (one frame per video/person, the "
      "frame closest to the video median ITA), stratified dataset x ITA bin x gender (seed 20261026). To be added to "
      "`release_ff2_public/docs/HUMAN_VALIDATION_PROTOCOL.md` as an MST-10 swatch question with an 'unsure' option; "
      "no study has been run.")
    w("")
    w("| dataset | " + " | ".join(f"Q{i + 1}" for i in range(4)) + " | male | non-male |")
    w("|---|---|---|---|---|---|---|")
    for s, g in hs.groupby("dataset"):
        b = g.ita_bin.value_counts()
        w(f"| {s} | " + " | ".join(str(int(b.get(i, 0))) for i in range(4)) + f" | {int((g.cl_male == 1).sum())} | {int((g.cl_male == 0).sum())} |")
    w("")
    # ---------------------------------------------------------------- 2 metrics
    w("## 2. Metrics module")
    w("")
    w("`code/multiaxis_metrics.py` (definitions = `agents/CFES/cfes/metrics.py`, on any partition). Per axis: F_FPR "
      "(Ding Eq. 8), F_DP, es-AUC, max-min FPR gap, TPR gap, equalised odds (EO_gap = max of the two gaps; EO_maxdev), "
      "worst-group FPR/TPR, with **FPR-specific eligibility** (a group enters FPR terms if it has >= 3 real videos, TPR "
      "terms if >= 3 fake videos, AUC/DP terms if both; FF2: persons) and per-group frame / video / person counts "
      "(`results/rescore_groups.csv`). Validation selection uses >= 2 videos (`MIN_CLUSTERS_VAL`). "
      "`dc_*` columns = the historical rule (groups with both classes, no minimum) for reproduction checks.")
    w("")
    npass = int(san["pass"].sum())
    w(f"Reproduction check: {npass}/{len(san)} checks pass (`results/sanity_checks.csv`): stored A1 thresholds re-derived "
      "from saved val logits, and Xu-label `dc_` gender/race/intersection F_FPR and pooled TPR at A1 equal "
      "`CFES/results/fair_oppoint/fair_oppoint.json` for CFES-D s7/17/27 and the Ding reproduction on all four sets.")
    if npass < len(san):
        w("")
        w("FAILED checks: " + "; ".join(san[~san["pass"]].check.tolist()))
    w("")
    # ---------------------------------------------------------------- 3 models
    w("## 3. Models and thresholds")
    w("")
    w("| model | seeds | t05 rule | A1 source | val TPR / FPR at A1 (mean) |")
    w("|---|---|---|---|---|")
    for m in ORDER:
        t = thr[thr.model == m]
        if not len(t):
            continue
        rule = "logit + b >= 0 (deployed intercept)" if (t.t05_logit != 0).any() else "logit >= 0"
        vt = t.get("val_TPR@A1", pd.Series(dtype=float)).mean()
        vf = t.get("val_FPR@A1", pd.Series(dtype=float)).mean()
        w(f"| {m} | {','.join(map(str, t.seed))} | {rule} | {t.a1_src.iloc[0]} | {f(vt)} / {f(vf)} |")
    w("")
    w("Note: `CFES A (rev4)` and `control (det-only)` at seed 17 both kept their start checkpoint (TRAINED best_epoch = -1), "
      "i.e. both are the phase-1 model `runs/clip_s17_p1`; their logits agree to 1.5e-5. Their 3-seed means therefore share "
      "one seed.")
    w("")
    p1 = sorted(m for m in thr.model.unique() if m.startswith("CESel p1"))
    w(f"CESel p1 arms (Amendment-5 screen) with a finished eval when this ran: {', '.join(p1) if p1 else 'none'}; the "
      "other arms and the CLIPFAIR baselines had no eval on disk (checked `CESEL/runs/ffpp_p1_*/eval`, `CLIPFAIR/runs/`). "
      "Rerun `code/rescore.py` + `code/make_report.py` when they finish (picked up automatically). **The p1 screen is "
      "validation-gated and not decided; the test-set numbers of p1 arms here must not inform the gate.** FF2-trained "
      "models were trained on FF2-full TRAIN (incl. RFW), so their public-TEST numbers are 'trained on full, tested on public'.")
    w("")
    # ---------------------------------------------------------------- 4 results per set and axis (A1)
    w("## 4. Results at A1 (per test set and axis; seed means)")
    w("")
    w("Columns: pooled AUC / TPR / FPR (at the threshold), then for the axis: F_FPR [95% CI], FPR gap [CI], TPR gap [CI], "
      "EO gap, F_DP, es-AUC, eligible groups FPR/TPR (of all groups). All in points. A1 is not defined for FF2-trained models.")
    w("")
    w("Reading the intervals: F_FPR and the gaps are sums / ranges of absolute deviations, so resampling noise inflates "
      "them (upward bias of the bootstrap distribution). The percentile interval can therefore sit above a small point "
      "estimate (e.g. a CI that excludes its own point). Use the intervals as a statement of width (how unstable the "
      "number is), not as tests; no model comparison here is a test. A pooled TPR far below another model's at the same "
      "rule usually explains a lower FPR disparity (fewer positives, fewer false positives): read every fairness number "
      "next to the TPR / FPR columns.")
    for s in ["ffpp_test", "dfdc", "celebdf", "dfd", "ff2_test"]:
        w("")
        w(f"### {SETN[s]}")
        for a in MAIN_AXES:
            sub = SM[(SM.set == s) & (SM.threshold == "A1") & (SM.axis == a)]
            if not len(sub):
                continue
            w("")
            w(f"**{AXN[a]}**")
            w("")
            w("| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |")
            w("|---|---|---|---|---|---|---|---|---|---|---|")
            for m in ORDER:
                r = sub[sub.model == m]
                if not len(r):
                    continue
                r = r.iloc[0]
                nm = f"{m} (mean of {int(r.n_seeds)})" if r.n_seeds > 1 else m
                w(f"| {nm} | {f(r.AUC)} | {f(r.TPR)} | {f(r.FPR)} | {f(r.F_FPR)}{ci(r.get('F_FPR_lo', np.nan), r.get('F_FPR_hi', np.nan))} | "
                  f"{f(r.fpr_gap)}{ci(r.get('fpr_gap_lo', np.nan), r.get('fpr_gap_hi', np.nan))} | "
                  f"{f(r.tpr_gap)}{ci(r.get('tpr_gap_lo', np.nan), r.get('tpr_gap_hi', np.nan))} | {f(r.EO_gap)} | {f(r.F_DP)} | "
                  f"{f(r.es_AUC)} | {int(round(r.n_fpr_elig))}/{int(round(r.n_tpr_elig))} ({int(round(r.n_groups))}) |")
    w("")
    # ---------------------------------------------------------------- 4b sensitivity
    w("## 4b. Label-source and label-unit sensitivity (A1, seed means)")
    w("")
    w("The AI-Face cleaner labels are per frame and change inside a video: they split 9 % (FF++ test reals) to 36 % "
      "(DFDC reals) of videos across gender x race groups (Xu labels: 0 %). The spec's min-support rule counts videos, "
      "which presumes one group per video, so the video-majority version (`*_vidmaj`, ties -> Xu label) is reported next "
      "to the frame-level primary. **The coordinator should fix the unit for H4 (recommendation: video majority) before "
      "the final model is scored.** Cells: F_FPR (eligible groups FPR/TPR).")
    KEY = ["CFES-D", "arm C", "arm E w3", "control (det-only)", "CESel B0", "CESel p1 full", "Ding repro (SFD+GDA)",
           "Ding Ori repro (SFD)"]
    for title, axes in (("gender x race", [("gender_x_race", "clean, frame"), ("gender_x_race_vidmaj (sens)", "clean, video maj."),
                                            ("gender_x_race_xu (sens)", "Xu (protocol)")]),
                        ("age", [("age", "AI-Face 4-bin"), ("age_if (sens)", "InsightFace <30/30-44/45-59/60+"),
                                 ("age_v2 (sens)", "AI-Face v2 young/middle/senior"),
                                 ("age_fairface2 (sens)", "FairFace 0-29/30+ (video majority)")]),
                        ("skin tone (FPR gap)", [("skin_ita4", "ITA gw (primary)"), ("skin_ita4_raw (sens)", "ITA raw"),
                                                 ("skin_ita4_gwexp (sens)", "ITA gw+exposure"), ("skin_ita4_local (sens)", "ITA within-set quartiles"),
                                                 ("skin_mst3_aif (sens)", "AI-Face MST 3-group")])):
        k = "fpr_gap" if title.startswith("skin") else "F_FPR"
        w("")
        w(f"**{title}** ({k})")
        w("")
        w("| set | model | " + " | ".join(n for _, n in axes) + " |")
        w("|---|---|" + "---|" * len(axes))
        for s_ in ["ffpp_test", "dfdc", "celebdf", "dfd"]:
            for m in KEY:
                cells = []
                for a_, _ in axes:
                    q = SM[(SM.set == s_) & (SM.model == m) & (SM.threshold == "A1") & (SM.axis == a_)]
                    cells.append(f"{f(q[k].iloc[0])} ({int(round(q.n_fpr_elig.iloc[0]))}/{int(round(q.n_tpr_elig.iloc[0]))})" if len(q) else "-")
                w(f"| {SETN[s_]} | {m} | " + " | ".join(cells) + " |")
    w("")
    # ---------------------------------------------------------------- 5 at 0.5
    w("## 5. Results at threshold 0.5 (compact: FPR gap per axis, with pooled TPR / FPR)")
    w("")
    w("Full metric set at 0.5 in `results/seedmean_axes.csv` (threshold = t05).")
    for s in ["ffpp_test", "dfdc", "celebdf", "dfd", "ff2_test"]:
        sub = SM[(SM.set == s) & (SM.threshold == "t05")]
        if not len(sub):
            continue
        w("")
        w(f"**{SETN[s]}**")
        w("")
        w("| model | AUC | TPR | FPR | " + " | ".join(f"{a} F_FPR / FPR gap" for a in MAIN_AXES) + " |")
        w("|---|---|---|---|" + "---|" * len(MAIN_AXES))
        for m in ORDER:
            r = sub[sub.model == m]
            if not len(r):
                continue
            r0 = r.iloc[0]
            cells = []
            for a in MAIN_AXES:
                q = r[r.axis == a]
                cells.append(f"{f(q.F_FPR.iloc[0])} / {f(q.fpr_gap.iloc[0])}" if len(q) else "-")
            w(f"| {m} | {f(r0.AUC)} | {f(r0.TPR)} | {f(r0.FPR)} | " + " | ".join(cells) + " |")
    w("")
    # ---------------------------------------------------------------- 6 per-group
    w("## 6. Per-group rates on the new axes (A1; CFES-D seed 7 / Ding repro / CESel B0)")
    w("")
    w("FPR [video-cluster CI] and TPR per group; counts = real frames / real clusters / fake clusters (clusters = videos; "
      "FF2: persons). `*` = not FPR-eligible (< 3 real clusters). Seed 7 shown so that the CI is a single-model interval.")
    for s in ["ffpp_test", "dfdc", "celebdf", "dfd", "ff2_test"]:
        for a in ("age", "skin_ita4"):
            sub = G[(G.set == s) & (G.axis == a) & (G.threshold == "A1") & (G.seed == 7)
                    & G.model.isin(["CFES-D", "Ding repro (SFD+GDA)", "CESel B0"])]
            if not len(sub):
                continue
            w("")
            w(f"**{SETN[s]}, {AXN[a]}**")
            w("")
            w("| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |")
            w("|---|---|---|---|---|---|---|---|")
            for gname in sorted(sub.group.unique()):
                q = sub[sub.group == gname]
                q0 = q.iloc[0]
                star = "" if bool(q0.fpr_elig) else "*"
                cells = []
                for m in ("CFES-D", "Ding repro (SFD+GDA)", "CESel B0"):
                    x = q[q.model == m]
                    if len(x):
                        x = x.iloc[0]
                        cells += [f"{f(x.FPR)}{ci(x.FPR_lo, x.FPR_hi)}", f(x.TPR)]
                    else:
                        cells += ["-", "-"]
                w(f"| {gname}{star} | {int(q0.n_real)} / {int(q0.real_clusters)} / {int(q0.fake_clusters)} | " + " | ".join(cells) + " |")
    w("")
    # ---------------------------------------------------------------- 7 descriptive H1-H4
    w("## 7. H1-H4 quantities for existing models (DESCRIPTIVE ONLY; not the confirmatory test)")
    w("")
    w("H1 = FF2-public TEST gender x race F_FPR (FF2 labels); H2 = FF2 age max-min FPR gap (InsightFace age, 4 bins); "
      "H3 = FF2 ITA 4-bin max-min FPR gap; H4 = mean over DFDC/Celeb-DF/DFD of gender x race F_FPR (AI-Face cleaner "
      "labels, >= 3 real videos per group). A1 threshold, seed means. NI columns = pooled TPR and AUC (FF2 for H1-H3; "
      "mean of the three sets for H4). The final-model comparator (best matched-CLIP baseline) does not exist yet.")
    w("")
    H = []
    for m in ORDER:
        r = {"model": m}
        for hk, s, a, k in (("H1", "ff2_test", "gender_x_race", "F_FPR"), ("H2", "ff2_test", "age", "fpr_gap"),
                            ("H3", "ff2_test", "skin_ita4", "fpr_gap")):
            q = SM[(SM.set == s) & (SM.model == m) & (SM.threshold == "A1") & (SM.axis == a)]
            r[hk] = float(q[k].iloc[0]) if len(q) else np.nan
            if len(q):
                r["FF2_TPR"], r["FF2_AUC"] = float(q.TPR.iloc[0]), float(q.AUC.iloc[0])
        q = SM[SM.set.isin(["dfdc", "celebdf", "dfd"]) & (SM.model == m) & (SM.threshold == "A1") & (SM.axis == "gender_x_race")]
        r["H4"] = float(q.F_FPR.mean()) if len(q) == 3 else np.nan
        qv = SM[SM.set.isin(["dfdc", "celebdf", "dfd"]) & (SM.model == m) & (SM.threshold == "A1") & (SM.axis == "gender_x_race_vidmaj (sens)")]
        r["H4_vidmaj"] = float(qv.F_FPR.mean()) if len(qv) == 3 else np.nan
        r["XD_TPR"], r["XD_AUC"] = (float(q.TPR.mean()), float(q.AUC.mean())) if len(q) == 3 else (np.nan, np.nan)
        H.append(r)
    H = pd.DataFrame(H)
    H.to_csv(f"{RES}/descriptive_H1_H4.csv", index=False, float_format="%.4f")
    w("| model | H1 i-F_FPR | H2 age FPR gap | H3 ITA FPR gap | FF2 TPR / AUC | H4 i-F_FPR (mean 3 sets; frame labels) | H4 with video-majority labels | cross-domain TPR / AUC |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in H.iterrows():
        if not np.isfinite([r.H1, r.H4]).any():
            continue
        w(f"| {r.model} | {f(r.H1)} | {f(r.H2)} | {f(r.H3)} | {f(r.get('FF2_TPR', np.nan))} / {f(r.get('FF2_AUC', np.nan))} | "
          f"{f(r.H4)} | {f(r.H4_vidmaj)} | {f(r.XD_TPR)} / {f(r.XD_AUC)} |")
    w("")
    # paired descriptive diffs
    w("Paired descriptive differences (model - Ding repro (SFD+GDA), seed-mean, same cluster resample; percentile 95% interval; "
      "Ding has one seed; no multiplicity correction; NOT a test of H1-H4):")
    w("")
    w("| model | H1 diff | H2 diff | H3 diff | H4 diff | H4 diff (video-majority labels) | dTPR FF2 / cross-domain |")
    w("|---|---|---|---|---|---|---|")
    for m in ["CFES-D", "arm C", "control (det-only)", "CESel B0", "Ding Ori repro (SFD)"]:
        cells = []
        for hk, sets, a, k in (("H1", ["ff2_test"], "gender_x_race", "F_FPR"), ("H2", ["ff2_test"], "age", "fpr_gap"),
                               ("H3", ["ff2_test"], "skin_ita4", "fpr_gap"), ("H4", ["dfdc", "celebdf", "dfd"], "gender_x_race", "F_FPR"),
                               ("H4v", ["dfdc", "celebdf", "dfd"], "gender_x_race_vidmaj (sens)", "F_FPR")):
            ds, pts = [], []
            for s in sets:
                x = mean_boot(boot, s, m, seeds_of.get(m, []), "A1", a, k)
                y = mean_boot(boot, s, "Ding repro (SFD+GDA)", [7], "A1", a, k)
                p1 = SM[(SM.set == s) & (SM.model == m) & (SM.threshold == "A1") & (SM.axis == a)]
                p0 = SM[(SM.set == s) & (SM.model == "Ding repro (SFD+GDA)") & (SM.threshold == "A1") & (SM.axis == a)]
                if x is None or y is None or not len(p1) or not len(p0):
                    ds = None
                    break
                ds.append(x - y)
                pts.append(float(p1[k].iloc[0]) - float(p0[k].iloc[0]))
            if ds is None:
                cells.append("-")
                continue
            d = np.nanmean(np.stack(ds), 0)
            cells.append(f"{np.mean(pts):+.1f} [{np.nanpercentile(d, 2.5):+.1f}, {np.nanpercentile(d, 97.5):+.1f}]")
        hm, hd = H[H.model == m], H[H.model == "Ding repro (SFD+GDA)"]
        dt = f"{f(float(hm.get('FF2_TPR', pd.Series([np.nan])).iloc[0]) - float(hd.FF2_TPR.iloc[0]))} / {f(float(hm.XD_TPR.iloc[0]) - float(hd.XD_TPR.iloc[0]))}" if len(hm) else "-"
        w(f"| {m} | " + " | ".join(cells) + f" | {dt} |")
    w("")
    # ---------------------------------------------------------------- 8 mechanism
    w("## 8. Mechanism: B0 per-group cue deficits vs per-group errors")
    w("")
    if mech is None:
        w("Not computed.")
    else:
        w("`code/mechanism.py`. Per (test set, group) point: cue deficit (set class mean - group mean of the removal "
          "utility u_k; positive = the group gets less evidence from cue k) vs the group's error deviation (reals: "
          "FPR_g - FPR_set; fakes: FNR_g - FNR_set). Spearman over all eligible points pooled across FF++ test, DFDC, "
          "Celeb-DF, DFD (>= 3 videos per group and side), video-cluster bootstrap 95% CI (B = 1000). `margin` row = "
          "the same correlation using the group's mean training-margin deficit (the arithmetic reference: summed "
          "utilities are margin differences). Threshold A1 (B0 a1.json).")
        w("")
        w("| grouping | side | n points | tex | freq | bound | sem | total | margin (reference) |")
        w("|---|---|---|---|---|---|---|---|---|")
        mm = mech[mech.threshold == "A1"]
        for (gname, side), g in mm.groupby(["grouping", "side"], sort=False):
            g = g.set_index("cue")
            cells = [f"{g.loc[c, 'spearman']:+.2f} [{g.loc[c, 'ci_lo']:+.2f}, {g.loc[c, 'ci_hi']:+.2f}]" if c in g.index else "-"
                     for c in ("tex", "freq", "bound", "sem", "total", "margin")]
            w(f"| {gname} | {side} | {int(g.n_points.iloc[0])} | " + " | ".join(cells) + " |")
        w("")
        w("Partial Spearman of (cue deficit, error deviation) given the margin deficit (does the cue carry information "
          "beyond the group's overall margin?), same bootstrap:")
        w("")
        w("| grouping | side | tex | freq | bound | sem | total |")
        w("|---|---|---|---|---|---|---|")
        for (gname, side), g in mm.groupby(["grouping", "side"], sort=False):
            g = g.set_index("cue")
            cells = [f"{g.loc[c, 'partial_given_margin']:+.2f} [{g.loc[c, 'pci_lo']:+.2f}, {g.loc[c, 'pci_hi']:+.2f}]"
                     if c in g.index and np.isfinite(g.loc[c, "partial_given_margin"]) else "-" for c in ("tex", "freq", "bound", "sem", "total")]
            w(f"| {gname} | {side} | " + " | ".join(cells) + " |")
    w("")
    # ---------------------------------------------------------------- appendix: group support
    w("## Appendix A. Group support per test set and axis")
    w("")
    w("Counts are model-independent: real frames / real videos / real persons | fake frames / fake videos / fake persons "
      "(FF2: video = person; de-novo fakes without a person are their own cluster); F = FPR-eligible (>= 3 real "
      "clusters), T = TPR-eligible (>= 3 fake clusters). With frame-level cleaner labels one video can contribute frames "
      "to several groups (Section 4b), so per-group video counts can add up to more than the set's videos.")
    ref = G[(G.model == "Ding repro (SFD+GDA)") & (G.seed == 7) & (G.threshold == "t05")]
    for s in ["ffpp_test", "dfdc", "celebdf", "dfd", "ff2_test"]:
        w("")
        w(f"**{SETN[s]}**")
        w("")
        w("| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |")
        w("|---|---|---|---|---|")
        for a in MAIN_AXES:
            q = ref[(ref.set == s) & (ref.axis == a)]
            for _, r in q.iterrows():
                el = ("F" if r.fpr_elig else "") + ("T" if r.tpr_elig else "")
                w(f"| {a} | {r.group} | {int(r.n_real)} / {int(r.real_clusters)} / {int(r.real_persons)} | {int(r.n_fake)} / {int(r.fake_clusters)} / {int(r.fake_persons)} | {el or '-'} |")
    w("")
    open(f"{RES}/MULTI_AXIS_RESULTS.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{RES}/MULTI_AXIS_RESULTS.md", len(L), "lines")


if __name__ == "__main__":
    main()
