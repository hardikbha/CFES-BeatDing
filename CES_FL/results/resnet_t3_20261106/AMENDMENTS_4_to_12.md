## Amendment 4 (06 Nov ~03:45 IST): ResNet priority, Codex (gpt-6-luna xhigh) strategy adopted
- Source: agents/RESNET_STRAT/2026-11-06-codex-resnet-strategy.md. Codex forecast: 25/40 on Table 3 is not credible;
  realistic 10-16/40, optimistic 18.
- **Arms (ResNet-50, Ding recipe):** SGD 1e-3, mom .9, wd 5e-3, 64+64 natural sampler, no aug, 50 ep.
  - **R0-SWA:** Ding's CE path unchanged, plus SWA over epochs 40-49 with BN recomputed on FF++ train.
  - **R3-SWA:** R0-SWA plus, every 4th update, a separate fair batch (8 real + 8 fake per gender x race cell,
    independent RNG). Its loss is 4 x [0.5 L_C + 0.02 L_R]:
    - L_C = class-conditional group CORAL (mean + population covariance) on L2-normalised pooled 2048-d features;
    - L_R = mean of the two largest per-cell real means of softplus(s + 1), with s = z_fake - z_real.
  - No balanced CE sampler, no W1, no full-strength tail.
- **Seeds:** 37, 47, 57, all fresh confirmatory. Primary metric: op05 vs Ding published T3. Secondary: one global
  FF++-val-real 90th-percentile threshold.
- **Criteria:**
  - target >= 25/40 mean wins AND mean AUC >= Ding on all 4 sets;
  - method evidence: >= 3/40 more mean wins than paired R0-SWA, and <= 0.5 AUC loss vs R0-SWA on any set.
- **Kill rule:** kill an R3 seed if its FF++ val AUC is > 2 pt below its paired R0 at both epochs 10 and 15, or if
  the aux gradient norm exceeds CE on > 20% of fair steps. Non-finite values = immediate fail. Never substitute a
  best-val epoch.
- **Execution:** direct SSH on gpu01/gpu02 (user, 06 Nov):
  - per-epoch checkpoints, plus a login-node watchdog that resumes after PBS kills ssh processes on a node;
  - allowed GPUs only.

## Amendment 5 (06 Nov ~10:45 IST): user goal "beat Ding on ResNet, esp. Celeb-DF, very soon"
- **Amendment-4 results** (seeds 37/47/57, op05, vs Ding T3 published):
  - R0-SWA: 14.0 ± 1.0 /40 per seed; mean-row 15/40 (FF++ 5, DFDC 9, Celeb-DF 0, DFD 2). AUC below Ding on all
    4 sets (Celeb-DF 75.72 vs 77.35).
  - R3-SWA: all 3 seeds KILLED in epoch 1 by the aux > CE gradient rule. Design flaw: the CE gradient shrinks as
    training fits, so the ratio rule fires even at lambda 0.02. No R3 result exists.
- **Celeb-DF diagnosis** (reported, not selected on):
  - Ding's Celeb-DF intersection scheme has 3 cells: male-black n=1922, nonmale-white n=25432, nonmale-others
    n=120 (90 reals). Our nonmale-others FPR is 0.51-0.67 vs overall 0.34.
  - es-AUC loses on the group-AUC gap: Black ~0.66, Others ~0.59-0.68, vs overall ~0.757.
  - The F_DP columns are label-blind and favour flat predictors (oracle CDF i-F_DP 59.6 vs Ding 9.11).
  - Threshold choice cannot change AUC or es-AUC (4 of the 10 columns).
- **New arms (ResNet-50, Ding recipe + SWA 40-49), seeds 37/47/57:**
  - **R0-SBI-SWA:** Self-Blended Images (Shiohara & Yamasaki CVPR'22) generated online from FF++ TRAIN reals only.
    They replace exactly 32 of the 64 fake slots per batch (label fake). This is the recipe-level generalisation
    lever, a matched control.
  - **R3-SBI-SWA:** R0-SBI-SWA plus Amendment-4's R3 aux loss, unchanged (4 x [0.5 L_C + 0.02 L_R], every 4th
    update).
- **Kill rule (amended before any R3-SBI run):** the aux > CE rule is DROPPED (shown to be degenerate); aux/CE stays
  logged as a diagnostic. Kept: the val-AUC rule (> 2 pt below paired R0 at both ep10 and ep15), and non-finite =
  fail.
- **Reporting:** op05 (primary) and the val-FPR10 threshold (secondary), vs Ding T3 published, per-seed plus mean.
  No test-based selection of seeds, epochs or thresholds; all three seeds are reported.
- **Comparisons:** R0-SWA (no SBI) vs R0-SBI-SWA = SBI effect; R0-SBI-SWA vs R3-SBI-SWA = our fairness term.

## Amendment 5a (06 Nov ~11:05 IST, user)
- The SBI arms run on 3 seeds (37/47/57). The user wants to "select the one which gives the best score thereafter".
- Recorded handling:
  - per-seed rows plus the 3-seed mean stay the primary report (paper rule);
  - a best-of-3 row may be shown only if labelled "best seed, selected post hoc on test", next to the mean.

## Amendment 6 (06 Nov ~15:20 IST): target 25-30/40 on Table 3 (user)
- **Status at op05, SWA, vs Ding T3:**
  - R0-SWA 15/40 (3-seed mean row);
  - R0-SBI-SWA 14/40 (per-seed 15/10/16; Celeb-DF AUC 78.79 > 77.35);
  - R3-SBI-SWA s37 15, s47 17 (s57 pending); s47 beats Ding AUC on all 4 sets.
- **Where columns are lost:**
  - (i) F_DP (label-blind; most cross-domain F_DP columns lost);
  - (ii) cross-domain race/intersection F_FPR on Celeb-DF/DFD (small groups over-flagged: Celeb-DF nonmale-others,
    90 reals);
  - (iii) race/intersection es-AUC on Celeb-DF/DFD (Black/Others group AUC gap).
- **New arms** (ResNet-50, Ding recipe, SWA 40-49, SBI on 32 of 64 fake slots, seeds 37/47/57, everything fixed now):
  - **R4 = R3-SBI-SWA + group-balanced SBI:** SBI source reals are drawn uniformly over gender x race cells (8 ids
    cells, train reals only; uniform cell, then uniform image within the cell) instead of uniformly over images.
    Mechanism: equal blend-artefact exposure for every demographic group, aimed at per-group AUC (es-AUC) and
    per-group real-score calibration (F_FPR) on unseen sets.
  - **R5 = R4 with a stronger real-tail term:** L_R coefficient 0.02 -> 0.10 (CORAL 0.5 unchanged; x4 every 4th
    update). A dose check, not test-tuned.
- **Kill rule** as in Amendment 5 (val-AUC vs paired R0-SBI; non-finite).
- **Reporting:** as in Amendment 5 / 5a: per-seed plus mean primary; best-of-3 only labelled post hoc.
- **Comparisons:** R3-SBI vs R4 = effect of balanced SBI; R4 vs R5 = dose.

## Amendment 7 (06 Nov ~15:45 IST): Codex memo 2 arms, under the user's /goal (beat Ding T3, esp. Celeb-DF, fast)
- Source: agents/RESNET_STRAT/2026-11-06-codex-resnet-strategy2.md. Forecast: about 16-19 wins per seed for the best
  arm; 25+ unlikely.
- **Final SBI** (op05, SWA): R0-SBI 15/10/16; R3-SBI 15/17/21 (mean 17.7 ± 3.1); method evidence R3-SBI vs R0-SBI
  MET.
- **Loss mix of R3-SBI s47:** 12 F_DP, 7 F_FPR, 4 es-AUC. All new arms build on R4 (balanced SBI sources), seeds
  37/47/57, Ding recipe, SWA 40-49, all coefficients fixed now:
  - **R0-BSBI:** R0-SBI-SWA with balanced SBI sources and no aux loss. Matched control for R4.
  - **R6:** R4 plus L_DP = mean_g (q_g - q̄)², with q_g = mean sigmoid(s/0.5) over each gender x race cell of the
    fair batch (8 real + 8 fake). Fair-step loss 4[0.5 L_C + 0.02 L_R + 0.05 L_DP]. Targets F_DP.
  - **R7:** R4 plus L_rank = mean over cells (>= 4 real and 4 fake) of the mean softplus(1 - (s_f - s_r)) over
    real-fake pairs. Fair-step loss 4[0.5 L_C + 0.02 L_R + 0.05 L_rank]. Targets es-AUC.
- Kill and reporting rules as in Amendments 5/5a.
- **Comparisons:**
  - R0-BSBI vs R0-SBI: effect of source balancing;
  - R4 vs R0-BSBI: fairness loss under balanced SBI;
  - R6 vs R4: rate term;
  - R7 vs R4: ranking term.

## Amendment 7a (06 Nov ~15:55 IST, before any R6 result)
- Smoke: R6's L_DP ≈ 0.0044 raw, ≈ 0.001 weighted at dp_coef 0.05, i.e. ~100x below CORAL/tail. R6 is likely
  near-inert.
- **Pre-declared dose arm R6b:** identical to R6 except dp_coef = 2.0 (weighted ≈ 0.04 per fair step before the x4,
  the same order as the other aux terms). Seeds 37/47/57. Launch when the R4/R5 GPUs free (~20:00).
- Fixed now; not test-chosen.

## Amendment 7b (06 Nov 16:50 IST, user; compute only, no result seen)
- R0-BSBI and R6 seeds 47/57 are PAUSED at epochs ~6-7 to free gpu01:0 and gpu01:6 for another project. They resume
  later from their checkpoints, with no change in protocol.
- Until they resume, the R4 vs R0-BSBI and R6 vs R4 comparisons have seed 37 only. They must not be read as
  3-seed results.

## Amendment 8 (06 Nov ~18:40 IST, user GO): post-hoc variants, no retraining, rules fixed BEFORE any test read
- **Variants**, each applied identically to every arm (R0-SWA, R0-SBI, R3-SBI, then R4/R5/R7, and the R0-BSBI/R6
  seeds that finish):
  - **V-TTA:** horizontal-flip test-time augmentation; the score is the mean of the logits for the image and its
    mirror.
  - **V-SOUP:** uniform weight average of an arm's 3 SWA checkpoints (seeds 37/47/57), BN recomputed on FF++ TRAIN
    only. One ResNet-50, same parameters and inference cost. Also SOUP+TTA.
  - **V-OP:** one global operating point, chosen on the DEV seed-17 R0 Ding-recipe run with FF++ val only
    (dev_celebdf_diag.py, 06 Nov 10:55):
    - candidates op05, val-real P90/P95/P99, valEER and priorFF were compared on that dev run; val-real P99
      ("valFPR1") gave the most wins there, and is FROZEN now as the secondary rule;
    - per model, the threshold is the logit at the 99th percentile of that model's own FF++ val real scores.
- **Reporting:**
  - op05 (Ding protocol) stays primary; valFPR1 is a labelled secondary (a protocol deviation from Ding);
  - TTA and SOUP rows are labelled as such;
  - per-seed rows plus mean always shown.
- **User intends to headline the single best seed.** Allowed only as a row labelled "best seed (selected post hoc
  on test)" next to the per-seed rows and the mean (Amendment 5a).

## Amendment 8a (06 Nov ~20:55 IST, before any R6 result exists)
- R4 (15.7) and R5 (14.7) are below R3-SBI (17.7), so balanced SBI sources don't help.
- The pre-declared R6b is therefore re-based on **R3-SBI** (natural SBI sources) instead of R4:
  - R6b = R3-SBI-SWA plus L_DP with dp_coef 2.0, inside the existing x4 fair step:
    4[0.5 L_C + 0.02 L_R + 2.0 L_DP].
  - Seeds 37/47/57, on the GPUs R4/R5 freed (gpu01:2/3/4).
  - Target: the 12 F_DP columns.
- The coefficient is unchanged from Amendment 7a; only the base arm changes. Disclosed.

## Amendment 9 (07 Nov ~02:30 IST): extra seeds for R3-SBI (user: "try different seeds", /goal active)
- R3-SBI-SWA, unchanged recipe (Amendment 5), new seeds 67/77/87/97. Together with 37/47/57 that makes 7 seeds.
- **Reporting:**
  - ALL 7 seeds are reported (per-seed rows), with the 7-seed mean ± sd as the primary summary;
  - the best seed may appear only as a row labelled "best seed (selected post hoc on test)";
  - variants (TTA, valFPR1) are reported for every seed, as in Amendment 8.
- **No kill rule:** these seeds have no paired R0 control. Non-finite values are still a failure.

## Amendment 10 (07 Nov ~04:15 IST, user): 3-seed Xception for Tables 1-2
- The Xception L_A and L_H best (24/40, seed 17 only) need seeds 7 and 27 under the paper's 3-seed rule.
- **Per seed s in {7, 27}, rerun the FULL original pipeline unchanged:**
  - stage P: 50 ep, all params, enc lr 1e-4, wd 1e-4, batch 16, select max FF++ val AUC; config =
    ~/ces_fl_runs/manip_response_ssh_20261101T115720Z/P/config.json with seed = s;
  - then L_A and L_H: 10 ep from P_s best.pt; configs = plain_head_s17 L_A/L_H configs with seed = s and init = P_s;
  - same source commits/hashes as the s17 runs.
- **Scoring:** the same locked evaluator (lh_tables_s17 style) at op05 (primary) and the val-FPR10 calibration,
  vs Ding T12 published.
- **Reporting:** per-seed rows (7/17/27), mean ± sd, mean-row wins; the best seed only as a row labelled post hoc.

## Amendment 11 (07 Nov ~04:25 IST, user): SBI arms on Xception (Tables 1-2)
- The same recipe as the ResNet SBI arms (Amendment 5), with Ding's XceptionDetector (his official Tables 1-2
  network) in place of Res50Detector:
  - Ding recipe: SGD 1e-3, mom .9, wd 5e-3, 64 real + 64 fake, 50 ep, 224;
  - SWA over epochs 40-49 with BN recomputed on FF++ train;
  - SBI on 32 of 64 fake slots, from FF++ train reals with landmarks (same landmark shards, natural sources).
- **Arms:**
  - R0-SBI-SWA-X: the control;
  - R3-SBI-SWA-X: plus the R3 aux loss 4[0.5 L_C + 0.02 L_R] every 4th update, with CORAL on the L2-normalised
    pooled 2048-d Xception features (before Ding's adjust layer).
- **Seeds** 37/47/57, the same seeds as the ResNet SBI arms.
- **Scoring:** vs Ding T12 published at op05 (primary) and valFPR1 (Amendment 8 rule), plus flip-TTA variants;
  per-seed plus mean; best seed labelled post hoc.
- **Kill rule:** val-AUC (R3 more than 2 pt below paired R0-SBI-X at both ep10 and ep15); non-finite = fail.

## Amendment 12 (07 Nov ~07:10 IST, /goal active; declared before computing any result)
- **F_DP bound** (agents/RESNET_STRAT/cdf_fdp_bound.py):
  - Celeb-DF fake prevalence per group: gender 0.42/0.84; race White 0.82, Black 0.63, Others 0.17;
    intersection 0.63/0.85/0.25.
  - For an equal-opportunity detector, F_DP = J x prevalence spread. Ding's Celeb-DF F_DP (12.0/8.16/9.11) then
    implies J <= 0.28/0.13/0.15, while ours is 0.46.
  - So these 3 columns are unreachable without degrading detection, and the Celeb-DF ceiling is ~7/10.
- **New post-processing variant GPP** (group-aware post-processing; Hardt et al. 2016 style), scores unchanged:
  - per test image, threshold = the 99th percentile of FF++ VAL real logits of the same Ding gender x race cell
    (the per-group version of the frozen valFPR1 rule); cells with < 20 val reals fall back to the global valFPR1
    threshold;
  - Celeb-DF/DFD/DFDC/FF++ cells are mapped by their (gender, race) labels.
- **Disclosed protocol deviation:** it uses demographic labels at inference, unlike Ding. It's reported as a
  separate, labelled row; AUC and es-AUC are unchanged by construction.
- **Applied to:** all 7 R3-SBI seeds and the 3 R0-SBI seeds (control), per seed + mean, best seed labelled post hoc.
