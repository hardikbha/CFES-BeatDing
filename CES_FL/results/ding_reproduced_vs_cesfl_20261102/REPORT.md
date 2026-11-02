# Actual local Ding reproduction compared with CES-FL

**These are measured local reproduction results, not the numbers published in Ding's paper.** Existing saved scores were checked against native metric JSONs; no detector inference, retraining, test tuning or recalibration was performed.

## Held-out detection AUC (%)

| Method | Seed | FF++ | DFDC | Celeb-DF | DFD |
|---|---|---:|---:|---:|---:|
| Ding classification (local) | 17 | 96.3210 | 55.5993 | 69.1593 | 68.0080 |
| Ding SFD+GDA (local archive) | 7; GDA code121 | 97.4443 | 56.0468 | 71.2196 | 72.5374 |
| CES-FL A: ImageNet frozen | 17 | 67.4828 | 50.6318 | 64.2825 | 59.3104 |
| CES-FL B: detection frozen + BCE | 17 | 98.3467 | 59.4712 | 74.3252 | 82.3578 |
| CES-FL C: detection frozen + invariance | 17 | 98.2216 | 59.6667 | 73.6992 | 81.2362 |
| CES-FL upgrade R0: frozen BCE | 17 | 98.3467 | 59.4712 | 74.3252 | 82.3578 |
| CES-FL upgrade R1: partial BCE | 17 | 98.4013 | 58.9726 | 74.2086 | 81.8093 |
| CES-FL upgrade R2: partial DRO | 17 | 98.3962 | 59.5189 | 74.6706 | 82.1211 |
| CES-FL upgrade RN: no-change diagnostic | 17 | 97.8472 | 59.6744 | 71.0136 | 81.5317 |
| CES-FL upgrade RD: age diagnostic | 17 | 98.1965 | 59.8601 | 72.7958 | 82.2801 |

Both CES-FL B and C have higher measured AUC on all four datasets than the archived full seed7 reproduction. C minus that reproduction: FF++ +0.7773pp, DFDC +3.6199pp, Celeb-DF +2.4797pp, DFD +8.6988pp. These are descriptive differences across distinct protocols/seeds, not causal method or statistical superiority claims.

## Actual reproduction scope and limitations

- Classification reference: seed17, native final checkpoint49.pth after50classification epochs. Full reference: classification seed7 followed by TRAIN-only SFD2% channel masking and50GDA epochs, native final49.pth. GDA resets its own RNG to121. Existing seed17/27 full checkpoints are complete, but a corresponding complete four-dataset evaluation export was not found in the inspected evaluation archive; their test values are not invented.

- Our A/B/C and upgrade rows use seed17, ten shared detection-pretraining epochs where applicable and ten downstream epochs, selected by validation AUC. A has only ImageNet initialization. B/R0 repeat identical saved scores; they are not independent replications.

- Both native TEST reports here use probability0.5. CES-FL values come from the already stored fixed0.5 report; its validation-calibrated main reports remain unchanged. A shared numerical threshold does not ensure matched FPR or calibrated score distributions.

- TEST image+label membership and native duplicate multiplicity match exactly for all four datasets. TRAIN/VAL membership audit is recorded below; this does not prove hidden-donor/person independence.

- Ding evaluation uses aspect-preserving Resize(224)+CenterCrop(224); CES-FL directly resizes to224×224. Ding classification training uses256×256, a two-logit head and SGD, whereas CES-FL uses224×224 and AdamW. Native checkpoint selection, optimizer budgets and seeds differ. These preprocessing/protocol differences are preserved and disclosed, not silently repaired using test outcomes.

- Archived Ding GDA remains the known code variant with fairness/total-loss computation under no_grad, last_linear-only parameter optimization and possible BN updates. The previous exact-source gradient probe found detached fairness and dependency-sensitive grad-mode behavior. These scores do not certify paper-faithful differentiable GDA. [Prior evidence](../../proposals/ff2_indomain_s17_20261101T042828Z/DING_BASELINE_STATUS_20261101.md).

- C is invariance-only; RN/RD are six-source diagnostics, not the full proposed manipulation-response model. Current50+50 runs have VAL results only and are excluded. No actual local ResNet-50 Ding reproduction was found here; Table3 cannot be filled with the paper's ResNet numbers and called reproduced.

## Membership checks

| Split | Ding rows | CES rows | Exact image+label multiplicity match |
|---|---:|---:|---|
| FF++ train | 76139 | 76139 | True |
| FF++ val | 25386 | 25386 | True |
| FF++ TEST | 25401 | 25401 | True |
| DFDC TEST | 22857 | 22857 | True |
| Celeb-DF TEST | 28458 | 28458 | True |
| DFD TEST | 9385 | 9385 | True |

## Full fairness and detection tables

All values below use the fixed0.5 reports. F_FPR is summed absolute disparity, not the max–min FPR gap. Lower F_FPR/F_DP and higher es-AUC/AUC are preferred. Native support filtering differences remain documented; they are not automatically harmonized by these tables.

### FF++

| Method | Backbone | Seed | Gender F_FPR↓ | Gender F_DP↓ | Gender es-AUC↑ | Race F_FPR↓ | Race F_DP↓ | Race es-AUC↑ | Intersection F_FPR↓ | Intersection F_DP↓ | Intersection es-AUC↑ | AUC↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ding classification (local) | Xception | 17 | 0.19 | 4.48 | 94.67 | 10.19 | 4.39 | 94.62 | 26.65 | 12.81 | 83.17 | 96.32 |
| Ding SFD+GDA (local archive) | Xception | 7; GDA code121 | 1.48 | 1.70 | 96.23 | 6.29 | 6.74 | 94.09 | 24.24 | 12.90 | 88.15 | 97.44 |
| CES-FL A: ImageNet frozen | Xception | 17 | 3.06 | 7.06 | 65.06 | 11.23 | 6.11 | 60.79 | 50.23 | 13.24 | 53.13 | 67.48 |
| CES-FL B: detection frozen + BCE | Xception | 17 | 0.18 | 4.21 | 96.94 | 11.78 | 2.88 | 96.04 | 34.35 | 13.14 | 90.90 | 98.35 |
| CES-FL C: detection frozen + invariance | Xception | 17 | 0.85 | 4.66 | 96.73 | 11.07 | 3.02 | 95.72 | 29.33 | 13.18 | 90.30 | 98.22 |
| CES-FL upgrade R0: frozen BCE | Xception | 17 | 0.18 | 4.21 | 96.94 | 11.78 | 2.88 | 96.04 | 34.35 | 13.14 | 90.90 | 98.35 |
| CES-FL upgrade R1: partial BCE | Xception | 17 | 0.12 | 3.98 | 96.99 | 10.22 | 2.66 | 95.98 | 29.36 | 12.98 | 90.50 | 98.40 |
| CES-FL upgrade R2: partial DRO | Xception | 17 | 0.95 | 3.98 | 97.02 | 15.02 | 3.70 | 96.07 | 29.22 | 12.61 | 90.57 | 98.40 |
| CES-FL upgrade RN: no-change diagnostic | Xception | 17 | 0.11 | 4.03 | 96.67 | 8.84 | 3.97 | 95.06 | 22.74 | 13.55 | 90.55 | 97.85 |
| CES-FL upgrade RD: age diagnostic | Xception | 17 | 0.67 | 3.87 | 96.73 | 8.98 | 2.73 | 95.77 | 28.36 | 13.05 | 90.09 | 98.20 |
### DFDC

| Method | Backbone | Seed | Gender F_FPR↓ | Gender F_DP↓ | Gender es-AUC↑ | Race F_FPR↓ | Race F_DP↓ | Race es-AUC↑ | Intersection F_FPR↓ | Intersection F_DP↓ | Intersection es-AUC↑ | AUC↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ding classification (local) | Xception | 17 | 3.20 | 4.28 | 55.04 | 28.90 | 12.17 | 44.32 | 57.59 | 15.80 | 33.70 | 55.60 |
| Ding SFD+GDA (local archive) | Xception | 7; GDA code121 | 3.06 | 3.84 | 55.77 | 20.75 | 10.80 | 47.43 | 42.56 | 13.86 | 34.86 | 56.05 |
| CES-FL A: ImageNet frozen | Xception | 17 | 0.22 | 1.51 | 46.98 | 9.82 | 4.55 | 39.33 | 22.92 | 7.62 | 32.06 | 50.63 |
| CES-FL B: detection frozen + BCE | Xception | 17 | 5.89 | 7.60 | 58.05 | 18.86 | 5.58 | 49.82 | 35.03 | 12.83 | 40.29 | 59.47 |
| CES-FL C: detection frozen + invariance | Xception | 17 | 5.41 | 7.07 | 58.84 | 18.22 | 6.37 | 51.33 | 34.53 | 12.47 | 42.87 | 59.67 |
| CES-FL upgrade R0: frozen BCE | Xception | 17 | 5.89 | 7.60 | 58.05 | 18.86 | 5.58 | 49.82 | 35.03 | 12.83 | 40.29 | 59.47 |
| CES-FL upgrade R1: partial BCE | Xception | 17 | 6.31 | 7.48 | 57.40 | 19.18 | 6.08 | 48.33 | 37.61 | 12.64 | 39.24 | 58.97 |
| CES-FL upgrade R2: partial DRO | Xception | 17 | 6.90 | 8.42 | 58.63 | 20.65 | 7.46 | 50.24 | 39.98 | 14.64 | 42.16 | 59.52 |
| CES-FL upgrade RN: no-change diagnostic | Xception | 17 | 5.69 | 7.44 | 59.51 | 19.46 | 7.30 | 49.29 | 40.29 | 13.61 | 40.22 | 59.67 |
| CES-FL upgrade RD: age diagnostic | Xception | 17 | 7.41 | 8.50 | 58.49 | 19.44 | 6.02 | 50.26 | 36.41 | 13.79 | 42.86 | 59.86 |
### Celeb-DF

| Method | Backbone | Seed | Gender F_FPR↓ | Gender F_DP↓ | Gender es-AUC↑ | Race F_FPR↓ | Race F_DP↓ | Race es-AUC↑ | Intersection F_FPR↓ | Intersection F_DP↓ | Intersection es-AUC↑ | AUC↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ding classification (local) | Xception | 17 | 7.87 | 18.03 | 63.47 | 34.13 | 21.46 | 48.83 | 39.41 | 23.41 | 45.99 | 69.16 |
| Ding SFD+GDA (local archive) | Xception | 7; GDA code121 | 6.96 | 19.69 | 59.78 | 45.47 | 29.60 | 58.07 | 47.59 | 32.10 | 53.38 | 71.22 |
| CES-FL A: ImageNet frozen | Xception | 17 | 4.51 | 11.79 | 59.17 | 15.03 | 11.91 | 59.55 | 20.62 | 12.28 | 58.18 | 64.28 |
| CES-FL B: detection frozen + BCE | Xception | 17 | 9.17 | 27.29 | 63.85 | 37.98 | 26.98 | 50.57 | 39.93 | 28.34 | 50.24 | 74.33 |
| CES-FL C: detection frozen + invariance | Xception | 17 | 11.06 | 30.83 | 63.39 | 41.89 | 29.66 | 50.52 | 42.80 | 30.80 | 50.26 | 73.70 |
| CES-FL upgrade R0: frozen BCE | Xception | 17 | 9.17 | 27.29 | 63.85 | 37.98 | 26.98 | 50.57 | 39.93 | 28.34 | 50.24 | 74.33 |
| CES-FL upgrade R1: partial BCE | Xception | 17 | 8.06 | 26.97 | 63.87 | 37.44 | 26.44 | 51.48 | 39.28 | 27.77 | 51.19 | 74.21 |
| CES-FL upgrade R2: partial DRO | Xception | 17 | 9.83 | 25.18 | 64.48 | 33.40 | 25.19 | 51.26 | 36.82 | 27.41 | 51.11 | 74.67 |
| CES-FL upgrade RN: no-change diagnostic | Xception | 17 | 10.28 | 28.99 | 62.27 | 34.90 | 28.01 | 52.55 | 36.87 | 29.07 | 50.06 | 71.01 |
| CES-FL upgrade RD: age diagnostic | Xception | 17 | 8.12 | 27.15 | 63.46 | 40.49 | 28.49 | 52.65 | 46.72 | 33.49 | 50.23 | 72.80 |
### DFD

| Method | Backbone | Seed | Gender F_FPR↓ | Gender F_DP↓ | Gender es-AUC↑ | Race F_FPR↓ | Race F_DP↓ | Race es-AUC↑ | Intersection F_FPR↓ | Intersection F_DP↓ | Intersection es-AUC↑ | AUC↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ding classification (local) | Xception | 17 | 0.48 | 0.93 | 67.71 | 22.09 | 10.36 | 61.84 | 36.14 | 12.74 | 55.94 | 68.01 |
| Ding SFD+GDA (local archive) | Xception | 7; GDA code121 | 3.27 | 2.99 | 72.50 | 12.02 | 0.56 | 66.02 | 49.82 | 14.33 | 59.53 | 72.54 |
| CES-FL A: ImageNet frozen | Xception | 17 | 4.11 | 3.28 | 58.85 | 13.93 | 17.71 | 56.40 | 24.26 | 26.73 | 52.70 | 59.31 |
| CES-FL B: detection frozen + BCE | Xception | 17 | 6.20 | 2.35 | 81.97 | 8.12 | 1.58 | 81.36 | 31.17 | 7.33 | 73.86 | 82.36 |
| CES-FL C: detection frozen + invariance | Xception | 17 | 2.74 | 1.48 | 80.66 | 2.60 | 4.79 | 80.49 | 22.80 | 12.08 | 71.50 | 81.24 |
| CES-FL upgrade R0: frozen BCE | Xception | 17 | 6.20 | 2.35 | 81.97 | 8.12 | 1.58 | 81.36 | 31.17 | 7.33 | 73.86 | 82.36 |
| CES-FL upgrade R1: partial BCE | Xception | 17 | 6.04 | 2.38 | 81.23 | 8.96 | 0.87 | 80.55 | 35.53 | 6.74 | 72.48 | 81.81 |
| CES-FL upgrade R2: partial DRO | Xception | 17 | 6.41 | 2.43 | 81.91 | 10.59 | 0.97 | 81.14 | 43.51 | 5.37 | 73.28 | 82.12 |
| CES-FL upgrade RN: no-change diagnostic | Xception | 17 | 3.25 | 2.04 | 80.62 | 6.96 | 0.91 | 80.82 | 33.13 | 6.60 | 73.06 | 81.53 |
| CES-FL upgrade RD: age diagnostic | Xception | 17 | 6.07 | 1.74 | 82.00 | 10.48 | 0.82 | 81.26 | 36.09 | 4.93 | 76.52 | 82.28 |

## Exact evidence

- Actual Ding evaluation files: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/eval_ding_protocol/{base_s17_ep49,full_s7_ep49}_{ff++,dfdc,celebdf,dfd}.{json,npz}`. Full seed7 FINAL_CKPT agrees with the evaluated49.pth.

- [Raw numeric comparison, sources and hashes](comparison.json) · [Full metrics CSV](FIXED05_ALL_METRICS.csv) · [CES-FL stored test provenance](../ding_format_tables_s17_20261102/README.md).

- This reporting command recomputed AUC from eight existing probability archives and checked their row order/labels against the actual TEST CSVs. No checkpoint was selected by the new comparison. Large checkpoints were not loaded or rehashed.

## FairForge-2 is a separate comparison

The existing Ding FF2 classification/final AUC99.2619/99.3573% comes from the internal502,035TRAIN/74,974VAL/182,884TEST build. It is not comparable as a public-split baseline. The old public-split CFES-D98.2598% belongs to a different historical model. No matching public-split Ding result is invented. [Authorized new original-only CES-FL pretraining](../../proposals/ff2_originals_s17_20261102_r1/README.md).

## Additional consistency checks

Full seed7 saved softmax probabilities exactly match its independently saved audit-score archive; AUC recomputed from raw logit margins differs by less than 0.000004 percentage points on each dataset. The main table preserves the native probability-based measurements. The separate classification audit-score export has small differences from its native evaluation export; it was not substituted for the checked native JSON/NPZ pair.

Eligible subgroup names and row counts agree with CES-FL under the explicitly declared male/nonmale→Male/Female mapping for these four datasets. This checks grouping consistency, not demographic-label ground truth. [Native aggregate Ding metric files](native_ding/) are copied unchanged for review; image rows, raw score arrays and checkpoint binaries remain private.

At the already stored fixed0.5 operating point, CES-FL B and C each exceed the local full seed7 reproduction on 28/40 unique numeric metric cells, including 24/36 fairness cells and 4/4 overall AUCs. This is a different reference and operating-point report from the earlier 17/40 comparison with published Ding numbers at validation-calibrated thresholds. It neither changes those historical results nor establishes superiority of C over B.
