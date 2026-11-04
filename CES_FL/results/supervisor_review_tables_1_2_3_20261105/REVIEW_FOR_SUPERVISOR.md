# Supervisor review: Tables 1–3 and next CES-FL training decision

**Snapshot:** 5 November 2026
**Base evidence commit:** `e993315a853aa278ec649010fa11daef1e601e44`
**Purpose:** review the current Ding-format tables, verify what the displayed numbers do and do not show, and advise which training weakness to prioritize. This is an analysis handoff only. It does not change training code/configuration or authorize a new run.

## Short answer: how many displayed metrics are better than Ding?

I counted a result as favorable when lower F_FPR/F_DP or higher es-AUC/AUC is better, following the table arrows. To avoid counting the same overall AUC three times just because Table 1 repeats it on each demographic row, the primary count treats AUC once per dataset and counts the three fairness metrics for each of the three axes.

| Table | Scope | CES-FL L_A | CES-FL L_H | At least one arm |
|---|---|---:|---:|---:|
| 1 | FF++ Xception | 7/10 | 6/10 | 7/10 |
| 2 | DFDC, Celeb-DF, DFD Xception | 16/30 | 15/30 | 16/30 |
| 3 | FF++, DFDC, Celeb-DF, DFD ResNet-50 | 9/40 | 12/40 | 13/40 |
| **All three** | **80 unique metric comparisons** | **32/80** | **33/80** | **36/80** |

Table 1 repeats the same AUC on each demographic row; Tables 2 and 3 list AUC once per dataset. Counting every displayed metric cell, including Table 1’s duplicate AUC rows, gives 82 cells: L_A 34/82, L_H 35/82, and at least one arm 38/82.

AUC alone: each arm is above Ding’s published reference on **2/8** dataset/backbone entries—FF++ Xception and FF++ ResNet-50—and below it on the six DFDC, Celeb-DF and DFD entries. The Xception DFDC L_A difference is only −0.091 percentage points; Celeb-DF is the largest AUC gap.

| Dataset | Backbone | Ding published AUC | L_A AUC (Δ pp) | L_H AUC (Δ pp) |
|---|---|---:|---:|---:|
| FF++ | Xception | 97.710% | 98.471% (+0.761) | 98.489% (+0.779) |
| DFDC | Xception | 61.860% | 61.769% (−0.091) | 60.456% (−1.404) |
| Celeb-DF | Xception | 76.750% | 70.747% (−6.003) | 71.404% (−5.346) |
| DFD | Xception | 81.460% | 81.156% (−0.304) | 80.879% (−0.581) |
| FF++ | ResNet-50 | 95.530% | 96.210% (+0.680) | 96.054% (+0.524) |
| DFDC | ResNet-50 | 62.540% | 59.535% (−3.005) | 60.068% (−2.472) |
| Celeb-DF | ResNet-50 | 77.350% | 73.497% (−3.853) | 73.811% (−3.539) |
| DFD | ResNet-50 | 78.630% | 73.655% (−4.975) | 73.472% (−5.158) |

These are counts of rounded point estimates in tables with different training/threshold protocols. They are **not** a statistical win rate, a significance test, or evidence of a protocol-matched win over Ding.

## What was compared

- **Table 1:** Xception on FF++. Our L_A/L_H seed-17 rows use the frozen FF++ validation-real 10% FPR thresholds. Ding’s values are published references under the paper’s protocol.
- **Table 2:** the same Xception checkpoints on DFDC, Celeb-DF and DFD, retaining DFDC’s native duplicate-row multiplicity. Our threshold remains calibrated on FF++ validation; Ding’s published values retain the paper operating protocol.
- **Table 3:** new ResNet-50 L_A/L_H runs, each tested once under a locked plan. The table’s Ding values are published ResNet-50 references; no Ding model was newly trained or evaluated for this extension. Our checkpoints use frozen FF++ validation thresholds; there was no target-domain calibration or TEST tuning.

For Tables 1–2, the Xception L_A checkpoint was selected at downstream epoch 3 and L_H at epoch 1 of 10. Table 3 uses the shared ResNet P50 pretraining checkpoint and downstream L_A selected at epoch 5/10, L_H at epoch 1/10. The table files document the remaining training/preprocessing/support-policy mismatches with Ding.

The full source-validation-threshold tables, fixed-0.5 views, support, thresholds, protocol descriptions and uncertainty are linked here:

- [Current Tables 1–3 index](https://github.com/hardikbha/CFES-BeatDing/blob/run/lh-finalization-resnet50-20261104/CES_FL/results/lh_finalization_celebdf_20261104_r1/TABLES_1_2_3_CURRENT_20261104.md)
- [Table 1 — FF++ Xception](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_1_SOURCE_VAL10.md)
- [Table 2 — cross-domain Xception](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_2_SOURCE_VAL10.md)
- [Table 3 — ResNet-50 point estimates](https://github.com/hardikbha/CFES-BeatDing/blob/run/lh-finalization-resnet50-20261104/CES_FL/results/lh_finalization_celebdf_20261104_r1/TABLE_3_RESNET_POINT_ESTIMATES_20261104.md)
- [Table 3 uncertainty and support limitations](https://github.com/hardikbha/CFES-BeatDing/blob/run/lh-finalization-resnet50-20261104/CES_FL/results/lh_finalization_celebdf_20261104_r1/TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md)

The disparity picture is also mixed: Table 1 L_A is better on gender F_FPR but worse on race and intersection F_FPR; L_H is worse on all three F_FPR rows. Table 3 FF++ F_FPR is lower than Ding’s published reference for L_A and L_H, while Table 3 Celeb-DF gender/race/intersection F_FPR values are substantially higher. These are point comparisons under differing protocols, not matched significance tests.

## Findings that matter for training

### 1. Cross-domain detection is the clearest weakness

Both Xception arms exceed the published FF++ AUC reference, but neither exceeds Ding’s published DFDC, Celeb-DF or DFD AUC. The ResNet-50 arms show the same pattern: both exceed the FF++ reference, and both trail on the three cross-domain datasets. This is the most consistent detection gap in the tables.

**Review direction:** decide whether the next controlled training study should prioritize domain robustness. A controlled codec/resize augmentation is a plausible candidate, but the tables alone do not show which augmentation will transfer. Choose it using a prespecified TRAIN/validation design, then freeze the checkpoint and thresholds before the one-time external TEST evaluation.

### 2. Fairness gains depend on axis and metric

The Xception L_A/L_H rows improve some disparity and es-AUC values while worsening others. The ResNet results are also mixed. Celeb-DF has especially poor CES-FL fairness point estimates relative to Ding’s published reference, and some ResNet subgroup intervals are unavailable under the 900/1000-replicate support rule. Do not summarize the results as a uniform fairness improvement.

**Review direction:** select one primary fairness outcome and keep FPR, FNR, worst-group AUC, F_DP and es-AUC as guardrails. The method should target subgroup error tails under a global validation-calibrated threshold; aligning mean scores or synthetic responses alone does not guarantee equal FPR.

### 3. The nuisance pilot is a useful mechanism diagnostic, not a final-method win

The four-arm pilot is documented in the [pilot report](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-nuisance-pilot-20261104/CES_FL/results/lh_nuisance_pilot_20261104_r1/REPORT.md). H1 reduced resize-induced score/response drift on a fixed TRAIN diagnostic sample. Its validation fairness changes were mixed, and the primary gender×race bootstrap had only 746/1000 valid draws, below the fixed support floor. Its procedures are synthetic local warp/resample probes, not validated forensic interventions. The screening was AI-only, not human review.

**Review direction:** do not scale H1 as the final fairness method from this pilot. A follow-up requires a separately reviewed proposal, adequate subgroup support, and a clear detection/fairness decision rule. Do not loosen the 900-draw floor or use TEST results to choose the next arm.

## Suggested files for code review

Please review these source/config files alongside the result tables:

- `CES_FL/src/ces_fl_head_ablation.py` and `CES_FL/src/ces_fl_head_ablation_resnet50.py` — plain-head training and evaluation paths used for L_A/L_H.
- `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/` — ResNet P50/L_A/L_H configurations and locked Table 3 evaluation protocol.
- `CES_FL/src/ces_fl_nuisance_resnet50.py` and `CES_FL/proposals/lh_nuisance_pilot_20261104_r1/` — the four-arm nuisance pilot source, exact configs, CPU checks and review requirements.
- `CES_FL/results/lh_nuisance_pilot_20261104_r1/` — aggregate pilot histories, subgroup metrics, diagnostics, bootstrap and GPU ledger.

The scientific source for the four-arm pilot is pinned to `84ad9838ac14f9e252d292146df5b1e1b1121d7c` (SHA256 `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`). The Tables 1–3 result sources remain on their separately linked branches; their protocol differences are recorded in each table.

## Supervisor decision requested

1. Confirm whether the published-reference win count and protocol caveats are represented fairly.
2. Choose the next priority: cross-domain detection, fairness under a global threshold, or a carefully controlled study of both.
3. If another training run is warranted, request a new dated proposal with the exact hypothesis, config/hashes, validation selection/calibration, subgroup safeguards, uncertainty threshold, compute cap and TEST freeze. No training or new TEST inference was launched by this review commit.
