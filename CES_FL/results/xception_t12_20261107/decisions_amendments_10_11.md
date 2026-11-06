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
