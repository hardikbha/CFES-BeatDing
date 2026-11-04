# FairForge-2 public validation in Ding-style layout

Baseline observation: 2026-11-04T05:18:36.508444+00:00; Ding status observation: 2026-11-04T05:18:36.923871+00:00.

Seed17; 218,266 TRAIN images and31,107 validation images (2,438real /28,669fake). All three baselines are still training toward50epochs. Every metric for a method uses its single highest-validation-AUC checkpoint so far, with earliest ties. No per-metric checkpoint selection. This is an interim in-domain validation table, not a final TEST or published-paper comparison.

The column order follows Ding Table1. Values are100times native fractions. F_FPR is a sum of group-versus-overall FPR deviations; it is not the overall FPR or a probability. F_DP is the gap between maximum and minimum positive prediction rates. es-AUC = overall AUC / (1 + sum of absolute group AUC deviations). Eligibility requires at least20real and20fake examples per group. Each selected model uses its own global validation-real10%target-FPR threshold; actual FPR is243/2,438=9.967186%. No TEST calibration or inference was performed.

## Detection and training progress

| Method | Completed / planned | Selected epoch | AUC % | FPR % | FNR % | Worst gender×race AUC % |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 50/50 | 40 | 99.805 | 9.967 | 0.300 | 99.595 |
| DAG-FDD | 50/50 | 50 | 99.814 | 9.967 | 0.220 | 99.367 |
| DAW-FDD | 50/50 | 41 | 99.836 | 9.967 | 0.244 | 99.488 |

## Gender, race and intersection

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|
| Gender | BCE | Xception | 3.429 | 0.106 | 99.740 | 99.805 |
| Gender | DAG-FDD | Xception | 4.276 | 0.182 | 99.657 | 99.814 |
| Gender | DAW-FDD | Xception | 3.429 | 0.086 | 99.804 | 99.836 |
| Race | BCE | Xception | 19.541 | 1.522 | 99.220 | 99.805 |
| Race | DAG-FDD | Xception | 18.324 | 1.509 | 99.263 | 99.814 |
| Race | DAW-FDD | Xception | 12.866 | 1.286 | 99.380 | 99.836 |
| Gender x race | BCE | Xception | 40.524 | 1.624 | 98.612 | 99.805 |
| Gender x race | DAG-FDD | Xception | 37.891 | 1.680 | 98.504 | 99.814 |
| Gender x race | DAW-FDD | Xception | 30.695 | 1.560 | 98.854 | 99.836 |

## Additional proxy-label axes

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|
| Age bins (proxy) | BCE | Xception | 8.740 | 4.603 | 99.612 | 99.805 |
| Age bins (proxy) | DAG-FDD | Xception | 10.447 | 4.682 | 99.579 | 99.814 |
| Age bins (proxy) | DAW-FDD | Xception | 10.453 | 4.706 | 99.588 | 99.836 |
| Skin-tone bins (proxy) | BCE | Xception | 14.567 | 0.398 | 99.499 | 99.805 |
| Skin-tone bins (proxy) | DAG-FDD | Xception | 12.778 | 0.440 | 99.400 | 99.814 |
| Skin-tone bins (proxy) | DAW-FDD | Xception | 6.796 | 0.597 | 99.618 | 99.836 |

## Additional group diagnostics

| Axis | Method | Worst-group AUC % | FPR gap (pp) | FNR gap (pp) |
|---|---|---:|---:|---:|
| Gender | BCE | 99.766 | 3.429 | 0.017 |
| Gender | DAG-FDD | 99.718 | 4.276 | 0.007 |
| Gender | DAW-FDD | 99.815 | 3.429 | 0.038 |
| Race | BCE | 99.607 | 12.463 | 0.142 |
| Race | DAG-FDD | 99.655 | 11.819 | 0.291 |
| Race | DAW-FDD | 99.580 | 8.958 | 0.125 |
| Gender x race | BCE | 99.595 | 15.175 | 0.180 |
| Gender x race | DAG-FDD | 99.367 | 15.658 | 0.390 |
| Gender x race | DAW-FDD | 99.488 | 9.989 | 0.157 |
| Age bins (proxy) | BCE | 99.687 | 7.626 | 0.146 |
| Age bins (proxy) | DAG-FDD | 99.724 | 9.302 | 0.203 |
| Age bins (proxy) | DAW-FDD | 99.718 | 8.673 | 0.117 |
| Skin-tone bins (proxy) | BCE | 99.661 | 8.555 | 0.322 |
| Skin-tone bins (proxy) | DAG-FDD | 99.692 | 8.011 | 0.162 |
| Skin-tone bins (proxy) | DAW-FDD | 99.732 | 4.649 | 0.241 |

Age uses automatic age estimates; skin-tone bins use automatic ITA. These are not independently human-verified demographic ground truth. Unknown skin bins remain in the per-group export but are excluded from eligible-group fairness aggregation. Per-group class counts and rates are in PER_GROUP_METRICS.csv. Intervals are pending completed-run validation replay and the prescribed paired contributor bootstrap.

## Existing Ding run: status only

The existing job is RUNNING on gpu01, with two live GPU worker PIDs2755001/2755002. Classification completed45/50epochs; last training event records epoch46. SFD and corrected GDA have no production receipts/directories yet. The planned chain is classification50 → TRAIN-only SFD2% → corrected GDA50 → selected validation replay. The full method is not complete.

Best classification validation AUC so far: 99.810363% at epoch21. Latest completed classification epoch45 AUC: 99.664198%. These are classification-stage observations, not a completed SFD+GDA result.

The existing GDA configuration is a class-conditioned, differentiable empirical-OT, head-only adaptation with frozen BatchNorm. It must be named as a corrected adaptation, not an exact published Ding reproduction. The TRAIN/VAL manifest hashes match this baseline wave, but sampling, staged budgets and evaluation geometry differ. No inference, training, calibration or job modification was performed for this status check.

Training log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.

Baseline source commit: `815b4d4bc679c55d4ee6323cc7284335ce210568`; Ding source commit: `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b`. Hashes, stage-file evidence and actual process/device evidence are in SNAPSHOT.json.
