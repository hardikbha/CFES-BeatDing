# FairForge-2 public validation in Ding-style layout

Baseline observation: 2026-11-03T16:56:35.705523+00:00; Ding status observation: 2026-11-03T17:13:19.169970+00:00.

Seed17; 218,266 TRAIN images and31,107 validation images (2,438real /28,669fake). All three baselines are still training toward50epochs. Every metric for a method uses its single highest-validation-AUC checkpoint so far, with earliest ties. No per-metric checkpoint selection. This is an interim in-domain validation table, not a final TEST or published-paper comparison.

The column order follows Ding Table1. Values are100times native fractions. F_FPR is a sum of group-versus-overall FPR deviations; it is not the overall FPR or a probability. F_DP is the gap between maximum and minimum positive prediction rates. es-AUC = overall AUC / (1 + sum of absolute group AUC deviations). Eligibility requires at least20real and20fake examples per group. Each selected model uses its own global validation-real10%target-FPR threshold; actual FPR is243/2,438=9.967186%. No TEST calibration or inference was performed.

## Detection and training progress

| Method | Completed / planned | Selected epoch | AUC % | FPR % | FNR % | Worst gender×race AUC % |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 18/50 | 17 | 99.344 | 9.967 | 1.228 | 98.854 |
| DAG-FDD | 21/50 | 19 | 99.645 | 9.967 | 0.617 | 99.272 |
| DAW-FDD | 17/50 | 15 | 99.521 | 9.967 | 0.879 | 99.077 |

## Gender, race and intersection

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|
| Gender | BCE | Xception | 1.734 | 0.024 | 99.224 | 99.344 |
| Gender | DAG-FDD | Xception | 3.259 | 0.185 | 99.630 | 99.645 |
| Gender | DAW-FDD | Xception | 2.581 | 0.010 | 99.500 | 99.521 |
| Race | BCE | Xception | 4.541 | 0.464 | 98.445 | 99.344 |
| Race | DAG-FDD | Xception | 13.567 | 1.441 | 99.071 | 99.645 |
| Race | DAW-FDD | Xception | 14.861 | 1.369 | 98.821 | 99.521 |
| Gender x race | BCE | Xception | 11.406 | 0.742 | 97.119 | 99.344 |
| Gender x race | DAG-FDD | Xception | 29.092 | 1.497 | 98.505 | 99.645 |
| Gender x race | DAW-FDD | Xception | 34.041 | 1.679 | 98.017 | 99.521 |

## Additional proxy-label axes

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|
| Age bins (proxy) | BCE | Xception | 9.037 | 4.839 | 98.864 | 99.344 |
| Age bins (proxy) | DAG-FDD | Xception | 8.108 | 4.143 | 99.062 | 99.645 |
| Age bins (proxy) | DAW-FDD | Xception | 8.469 | 4.701 | 99.192 | 99.521 |
| Skin-tone bins (proxy) | BCE | Xception | 6.889 | 1.242 | 98.688 | 99.344 |
| Skin-tone bins (proxy) | DAG-FDD | Xception | 7.208 | 0.910 | 99.175 | 99.645 |
| Skin-tone bins (proxy) | DAW-FDD | Xception | 6.223 | 1.408 | 98.934 | 99.521 |

## Additional group diagnostics

| Axis | Method | Worst-group AUC % | FPR gap (pp) | FNR gap (pp) |
|---|---|---:|---:|---:|
| Gender | BCE | 99.294 | 1.734 | 0.037 |
| Gender | DAG-FDD | 99.635 | 3.259 | 0.082 |
| Gender | DAW-FDD | 99.505 | 2.581 | 0.071 |
| Race | BCE | 99.099 | 2.880 | 0.836 |
| Race | DAG-FDD | 99.376 | 10.168 | 0.115 |
| Race | DAW-FDD | 99.329 | 11.275 | 0.375 |
| Gender x race | BCE | 98.854 | 5.810 | 1.085 |
| Gender x race | DAG-FDD | 99.272 | 13.049 | 0.446 |
| Gender x race | DAW-FDD | 99.077 | 12.830 | 0.668 |
| Age bins (proxy) | BCE | 99.123 | 7.475 | 0.328 |
| Age bins (proxy) | DAG-FDD | 99.394 | 6.243 | 0.184 |
| Age bins (proxy) | DAW-FDD | 99.352 | 8.148 | 0.269 |
| Skin-tone bins (proxy) | BCE | 99.131 | 5.408 | 0.849 |
| Skin-tone bins (proxy) | DAG-FDD | 99.473 | 5.652 | 0.279 |
| Skin-tone bins (proxy) | DAW-FDD | 99.305 | 4.904 | 0.680 |

Age uses automatic age estimates; skin-tone bins use automatic ITA. These are not independently human-verified demographic ground truth. Unknown skin bins remain in the per-group export but are excluded from eligible-group fairness aggregation. Per-group class counts and rates are in PER_GROUP_METRICS.csv. Intervals are pending completed-run validation replay and the prescribed paired contributor bootstrap.

## Existing Ding run: status only

The existing job is RUNNING on gpu01, with two live GPU worker PIDs2755001/2755002. Classification completed33/50epochs; last training event records epoch34. SFD and corrected GDA have no production receipts/directories yet. The planned chain is classification50 → TRAIN-only SFD2% → corrected GDA50 → selected validation replay. The full method is not complete.

Best classification validation AUC so far: 99.810363% at epoch21. Latest completed classification epoch33 AUC: 99.398600%. These are classification-stage observations, not a completed SFD+GDA result.

The existing GDA configuration is a class-conditioned, differentiable empirical-OT, head-only adaptation with frozen BatchNorm. It must be named as a corrected adaptation, not an exact published Ding reproduction. The TRAIN/VAL manifest hashes match this baseline wave, but sampling, staged budgets and evaluation geometry differ. No inference, training, calibration or job modification was performed for this status check.

Training log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.

Baseline source commit: `815b4d4bc679c55d4ee6323cc7284335ce210568`; Ding source commit: `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b`. Hashes, stage-file evidence and actual process/device evidence are in SNAPSHOT.json.
