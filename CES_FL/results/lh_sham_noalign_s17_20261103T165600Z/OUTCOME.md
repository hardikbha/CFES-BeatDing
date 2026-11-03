# Independent three-epoch L_H loss ablations — completed

Observed 4 November 2026, 01:06 IST (3 November 19:36 UTC). FF++ TRAIN/validation, seed 17. Scientific execution commit `ab7f36798cfa06226e17e31c2171e0c02404a216`; scientific source SHA256 `1b6dc1b0e6d2b2220da897500a4d1bffb7764280df642abf6befdd1cf511ed01`. PBS job `45361.mgmt01` finished with exit status 0 on gpu01, two H100 cards, `ngpus=2`, 10 CPUs. Both native runs completed 3/3 epochs, passed selected-checkpoint validation replay and the independent saved-score comparison. No benchmark TEST access.

| Selected epoch 3, validation only | Lower-LR control | SHAM-FIX | NO-ALIGN |
|---|---:|---:|---:|
| AUC (%) | 98.185660 | 98.162281 | 98.208318 |
| Achieved FPR (%) | 9.995112 | 9.995112 | 9.995112 |
| FNR (%) | 4.611628 | 4.494224 | 4.461351 |
| Gender F_FPR (fraction) | 0.043699 | 0.038799 | 0.044679 |
| Race F_FPR | 0.094555 | 0.070286 | 0.094555 |
| Gender×race F_FPR | 0.418939 | 0.430919 | 0.420300 |
| Age F_FPR | 0.192721 | 0.144968 | 0.207825 |
| Skin F_FPR | 0.036871 | 0.028359 | 0.035019 |
| ids8 F_FPR | 0.337456 | 0.377597 | 0.336647 |

SHAM-FIX improved gender, race, age and skin disparities and FNR, while AUC fell by 0.023379 percentage points and the primary gender×race disparity increased by 0.011981. NO-ALIGN improved AUC by 0.022658 percentage points and FNR by 0.150277 points, while gender, gender×race and age disparities rose. Neither is an unqualified fairness improvement over the lower-LR control. These are single-seed validation observations, not statistical proof or a comparison against Ding's TEST figures.

SHAM-FIX changed only the original-real and auxiliary-real sham terms to `mean(abs(asinh(z_sham-z_original)))`. NO-ALIGN kept the legacy sham formula and set only `lambda_alignment=0`. Both independently reloaded the same P50 encoder checkpoint (SHA256 `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5`) and fresh identical seed-17 head. They used batch 32, frozen BN, head LR 1e-4, encoder LR 1e-6, the same TRAIN/validation manifests, processing, auxiliary sampling, three epochs and 7,140 updates. Neither initialized from a trained L_H checkpoint. Earliest maximum validation AUC selected epoch 3 in both; no fairness-based checkpoint swap.

Focused CPU regression tests passed 6/6 and existing head tests 11/11. CUDA loss tests passed on both assigned devices; source-bound native smoke and real-input backward checks passed for each arm. Fresh byte/decoding preflight passed on all 76,139 TRAIN and 25,386 validation images per configuration. The 88-case procedural review was rebound only after exact image, sheet, case and settings equivalence, preserving its original named AI review scope. It does not prove biological identity or forensic-cue validity. The 163 declared contributor overlaps and proxy-label limitations remain.

Selected checkpoint SHA256: SHAM-FIX `78c7cd8300fe84a2c91ba85b0b2ad6916040bbab4eea2c65af9e2406a257f867`; NO-ALIGN `3b3c61a6dade0d5858a89223416b5edbae79d933712c7a8c40de64155c03785f`. Full epoch tables, subgroup counts/rates, all fairness measures and checkpoints' aggregate receipts are in each arm's `comparison/` and `training/` directories. Image data, row-level scores and checkpoint binaries remain on cluster storage.

PBS reported 00:40:30 for two allocated GPUs: 1.35 GPU-hours including one arm's idle allocation after its earlier finish. Per-arm monotonic times were 2,361.108 seconds and 1,422.504 seconds; the remaining 1,076.389 GPU-seconds were added to the shared ledger as cohort allocation overhead. The ledger now records 17.286613/24 GPU-hours used, 6.713387 remaining. PBS `stime`/`etime` disagree because of node clock skew, so accounting uses reported walltime and monotonic receipts. The allocation cap was respected. The earlier held/queued requests 45350, 45351, 45352 were preserved as superseded evidence and did not train.

Decision: retain both as bounded diagnostic results. SHAM-FIX is not supported as an improvement to the primary fairness metric, and NO-ALIGN shows only a small AUC/FNR gain with a small primary fairness regression. Do not extend either run or infer TEST generalization from this validation batch.
