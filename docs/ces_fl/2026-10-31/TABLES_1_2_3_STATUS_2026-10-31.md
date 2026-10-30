# CES-FL: B, B/C, C-invariance and Ding Tables 1–3

Verified 31 October 2026. All current experimental results are seed 17.

## What the names mean

| Label | Meaning | What it tests |
|---|---|---|
| P | End-to-end BCE detection pretraining on original FF++ TRAIN images. | Produces a detection-trained Xception encoder. |
| A | Frozen ImageNet encoder plus newly trained CES-FL heads/gates/fusion; BCE only. | Reference before detection pretraining. |
| B | Load the selected P encoder, freeze it and its BatchNorm statistics, train fresh heads/gates/fusion using BCE on originals. | Strong matched baseline. |
| C-invariance | Exactly the same P encoder and head initialization as B; add demographic counterfactual consistency to original-image BCE. | Exploratory contribution of demographic invariance. |
| B/C | Shorthand for comparing B versus C. | This is not another model or training arm. |

C used 36 user-reviewed Qwen edits from six TRAIN originals (three real, three fake), covering younger/older, gender presentation and lighter/darker skin edits. They are variants of six sources, not 36 independent identities. Recorded exposure was 753 pair draws: 717 from real sources and 36 from fake sources. User visual approval does not independently establish preservation of forensic evidence.

The current executable invariance term uses |asinh(z_original) − asinh(z_edit)| on valid demographic pairs, with lambda_d=1. The current arm has lambda_f=0 and lambda_map=0. It does not test forensic sensitivity or the full selectivity objective.

“ARMAT” was not found as a saved method name in the audit logs and current launch records searched. In this document, “our method” means the current CES-FL v0.2.1 C-invariance run; ARMAT is not assumed to be an independently implemented method.

## Why the full matched comparison is missing

The current B and C runs have completed training, checkpoint selection and validation calibration. Both RUN_VERIFIED.json files explicitly record test_evaluated=false. There are no current real-test results to place beneath Ding's published results. Current ResNet-50 CES-FL results are also unavailable.

Below are the published reference rows and explicit missing-result rows, using the paper's fairness columns. These are a status report, not a completed benchmark comparison. Dash (—) means unavailable, never zero. All values are percentages; F_FPR and F_DP are better when smaller, and es-AUC and AUC when larger.

The Ding rows were checked against [the official paper, Tables 1–3](https://arxiv.org/html/2511.10150v5). They are paper results, not our reproduction results.

## Table 1 — FF++ held-out TEST, Xception

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Gender×race F_FPR ↓ | Gender×race F_DP ↓ | Gender×race es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ | Ding published | 0.53 | 3.61 | 96.45 | 9.29 | 4.35 | 94.86 | 20.18 | 9.47 | 86.91 | 97.71 |
| FF++ | Our B — TEST not evaluated | — | — | — | — | — | — | — | — | — | — |
| FF++ | Our C-invariance — TEST not evaluated | — | — | — | — | — | — | — | — | — | — |

## Available current results — FF++ VALIDATION, not Table 1 TEST results

These values were reconstructed from the saved group-metric CSV and selected-checkpoint RUN_VERIFIED.json files. Both models used the same validation-only empirical FPR target of 10%; achieved validation FPR was 409/4092=9.9951%. B selected epoch 7 and C epoch 3 by validation AUC. Their calibrated logit thresholds are −0.6314775943756102 and −1.5438921451568601, respectively. This operating-point protocol must not be silently equated with Ding's published protocol. AUC is threshold-free, but validation versus test remains a material distinction.

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Gender×race F_FPR ↓ | Gender×race F_DP ↓ | Gender×race es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ VAL | B: detection-pretrained BCE | 7.31 | 1.03 | 96.15 | 9.79 | 7.11 | 93.20 | 44.91 | 15.89 | 87.03 | 98.19 |
| FF++ VAL | C: BCE + invariance | 6.92 | 1.21 | 95.69 | 10.64 | 6.26 | 92.44 | 45.92 | 15.15 | 86.35 | 98.07 |

B→C validation AUC changes by −0.1288 percentage points. Gender F_FPR improves slightly; race and gender×race F_FPR worsen, and all three es-AUC values decrease. This single-seed pilot does not establish an overall fairness improvement. No significance claim is made. Small subgroup support and editor no-change drift remain concerns documented in the existing fairness assessment.

## Table 2 — Cross-domain TEST, Xception

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Gender×race F_FPR ↓ | Gender×race F_DP ↓ | Gender×race es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFDC | Ding published | 1.76 | 3.67 | 57.94 | 17.93 | 7.58 | 52.33 | 37.37 | 11.92 | 39.03 | 61.86 |
| DFDC | Our B — not evaluated | — | — | — | — | — | — | — | — | — | — |
| DFDC | Our C-invariance — not evaluated | — | — | — | — | — | — | — | — | — | — |
| Celeb-DF | Ding published | 6.41 | 10.81 | 69.68 | 16.60 | 7.29 | 69.96 | 23.32 | 12.01 | 63.27 | 76.75 |
| Celeb-DF | Our B — not evaluated | — | — | — | — | — | — | — | — | — | — |
| Celeb-DF | Our C-invariance — not evaluated | — | — | — | — | — | — | — | — | — | — |
| DFD | Ding published | 4.72 | 5.49 | 78.98 | 2.09 | 18.66 | 77.44 | 27.32 | 22.81 | 69.73 | 81.46 |
| DFD | Our B — not evaluated | — | — | — | — | — | — | — | — | — | — |
| DFD | Our C-invariance — not evaluated | — | — | — | — | — | — | — | — | — | — |

## Table 3 — Held-out and cross-domain TEST, ResNet-50

Our current B/C runs use Xception. We cannot insert those numbers into a ResNet-50 table. A current CES-FL ResNet-50 run is not available; older CFES-ResNet results are a different method.

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Gender×race F_FPR ↓ | Gender×race F_DP ↓ | Gender×race es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ | Ding published | 3.94 | 5.14 | 94.46 | 18.67 | 2.43 | 92.69 | 34.30 | 8.61 | 82.87 | 95.53 |
| FF++ | Our CES-FL ResNet-50 — not trained | — | — | — | — | — | — | — | — | — | — |
| DFDC | Ding published | 3.97 | 3.61 | 56.96 | 11.94 | 5.15 | 50.99 | 32.14 | 10.27 | 37.55 | 62.54 |
| DFDC | Our CES-FL ResNet-50 — not trained | — | — | — | — | — | — | — | — | — | — |
| Celeb-DF | Ding published | 4.53 | 12.00 | 68.98 | 6.76 | 8.16 | 71.03 | 5.56 | 9.11 | 68.25 | 77.35 |
| Celeb-DF | Our CES-FL ResNet-50 — not trained | — | — | — | — | — | — | — | — | — | — |
| DFD | Ding published | 3.12 | 1.76 | 77.68 | 6.51 | 3.03 | 70.93 | 20.99 | 4.85 | 66.19 | 78.63 |
| DFD | Our CES-FL ResNet-50 — not trained | — | — | — | — | — | — | — | — | — | — |

## Exact evidence and earlier tables

- B selected-run verification: `/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/full_experiment_runs/B_detection_frozen/RUN_VERIFIED.json`
- C selected-run verification: `/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/full_experiment_runs/C_detection_invariance/RUN_VERIFIED.json`
- B and C configurations and selected checkpoints: `config.json` and `best.pt` in those same directories.
- Validation group metrics: `/home/sachin.chaudhary/cesel_ma_audit_logs/ces_fl_fairness_BC_group_metrics_20261031.csv`
- Completed controlled-run results: `/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/launches/seed17_parallel_20261030T223530/REPORT_RESULTS.md`
- Detailed fairness assessment and control diagnostics: `/home/sachin.chaudhary/cesel_ma_audit_logs/ces_fl_fairness_assessment_and_next_steps_20261031.md`
- Executed code snapshot: `/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/source/ces_fl_diagram_model_v021.py`
- Historical Tables 1–3: `/home/sachin.chaudhary/cesel_ma_audit_logs/ding_tables_1_2_3_seed7_20261028.md`. Those rows concern older CLIP CESel/CFES experiments and a separate CFES-ResNet run; they must not be relabelled as current CES-FL B/C results.

## Remaining dependency

Finish the validation error/control audit and lock the reporting protocol before final evaluation. Once models and protocol are fixed, evaluate selected checkpoints on held-out FF++ TEST and the cross-domain datasets, retaining the declared thresholds and documenting differences from Ding. A ResNet-50 experiment is a separate requirement if a current-method Table 3 is desired. Do not interpret the empty cells as a reason to relax the fairness criteria or tune on test results.

No training, image generation, test inference or scheduler modifications were performed to prepare this explanation.
