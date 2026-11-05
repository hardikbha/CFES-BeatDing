# CES-FL L_H objective-priority update — 5 November 2026

**Scope:** seed 17, FF++ only, validation only. No Ding computation, TEST read, or FairForge-2 run. Results are not evidence that CES-FL beats Ding.

## Completed original-image fairness and component-ablation arms

The exact pinned scientific source is commit `985877b11b5bd77cbcec534592eb6d14509a0af1`, file SHA256 `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`. B0/C0 used config SHA256 `34800f42f2cc81df32508b7a732700e45b4df5f6ddfebca1a109aad1db1407ef`, digest `76575eb018ca4167e7acd3de1660eae80484da6191d3b7b75e97579168773630`. A0/H0 were reused from verified result commit `e993315a853aa278ec649010fa11daef1e601e44`; no controls were retrained. All four used the same FF++ split and three-epoch budget. B0 was run first; C0 followed its source/config-bound procedural check.

| Arm | Objective | Selected epoch | VAL AUC | VAL FPR | VAL FNR | Gender F_FPR | Race F_FPR | Gender×race F_FPR | Age F_FPR | Skin F_FPR | Worst gender×race AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | Original weighted BCE | 3 | 96.6217% | 9.9951% | 9.6036% | 0.0604 | 0.1235 | 0.4059 | 0.1292 | 0.0777 | 90.2678% |
| B0 | BCE + original-image hierarchical fairness | 3 | 96.6124% | 9.9951% | 9.6929% | 0.0613 | 0.1273 | 0.3988 | 0.1316 | 0.0865 | 90.1965% |
| C0 | B0 + procedural sensitivity/sham terms, no response alignment | 3 | 96.5088% | 9.9951% | 9.7727% | 0.0535 | 0.0794 | 0.3560 | 0.1314 | 0.0687 | 89.9060% |
| H0 | C0 + conditional response alignment | 1 | 96.4214% | 9.9951% | 9.9934% | 0.0564 | 0.0999 | 0.3337 | 0.1227 | 0.0501 | 90.2108% |

`F_FPR` values are the reported fraction units (sum of supported group FPR deviations from the overall FPR); smaller is better under that metric. The same global empirical validation-real 10% calibration rule was applied separately to each selected checkpoint. Equal overall FPRs are expected from that protocol; they are not a fairness result.

Point estimates suggest B0 barely changes the intersection metric versus A0 (0.4059→0.3988; AUC 96.6217%→96.6124%). C0 reduces that point estimate to 0.3560 while losing another 0.1036 AUC percentage points versus B0 and increasing FNR. H0 has the lowest gender×race F_FPR point estimate, 0.3337, with the lowest AUC and highest FNR of these four. Other axes are mixed; the full group counts, gaps, FPR/FNR and AUCs are in `STAGE1_RESULTS_AGGREGATE.json`.

At fixed probability 0.5, overall FPR/FNR were: A0 14.7849%/6.4854%; B0 14.2962%/6.7859%; C0 12.8055%/7.7727%; H0 13.0743%/8.0070%. Gender×race F_FPR at fixed 0.5 was 0.5407, 0.5348, 0.4119 and 0.3821, respectively. This operating point also shows the detection/fairness trade-off; it does not replace calibrated results.

The fixed-0.5 paired bootstrap used the same 1,000 contributor-cluster draws on private cache copies with logit threshold 0 (the original caches were unchanged). Gender×race F_FPR differences were B0−A0 −0.0058 [95% CI −0.0315, 0.0526], C0−B0 −0.1230 [−0.2876, 0.0170], H0−C0 −0.0298 [−0.0969, 0.0848], B0−H0 0.1528 [−0.0132, 0.2819], and H0−A0 −0.1586 [−0.2615, 0.0161]. Each had 746 valid draws, so all remain inconclusive under the 900-draw rule. The aggregate intervals are in `FIXED05_PAIRED_BOOTSTRAP_AGGREGATE.json`.

## Uncertainty and intervention validity

Paired seed-17 validation bootstrap resampled 209 connected recorded-contributor components for 1,000 draws. The primary gender×race F_FPR interval has only **746/1,000** valid draws, below the prespecified 900-draw minimum, so the primary fairness comparisons are **inconclusive**. Age-axis intervals had 976 valid draws. The connected recorded IDs are not ground-truth proof of person independence, and these intervals measure validation-sample uncertainty, not training-seed uncertainty.

B0’s exact objective is original-image fairness only; its fairness point change is small. C0 used 12,926 real training originals per epoch for the procedural pair terms. Its reused 88-case qualification is an AI visual screen; `real_forgery_validity_proven=false`. C0 therefore tests the configured procedural sensitivity/sham package, not validated sensitivity to genuine forensic evidence.

## Compute and artifacts

B0 and C0 completed three epochs and passed selected-checkpoint validation replay. Both ran sequentially on gpu02 H100 UUID `GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8`; B0 consumed 0.2335 GPU-hours and C0 0.2615. CUDA checks consumed 0.0500 GPU-hours total. The B0/C0 cohort used 0.5450 of its 3 GPU-hour cap; no reservation remains. Selected checkpoint hashes are recorded in the aggregate JSON. The original failed B0 operator attempt is preserved and used zero GPU time.

## Remaining work at this publication

- **Complete-classifier P50 audit:** CUDA smoke passed, but the zero-update comparison against complete P50 and A0/B0/C0/H0 did not finish within the separate 0.25 GPU-hour cap. The private process was stopped at the cap before it wrote an audit result; no score-distribution, nuisance-response or gradient comparison is claimed. The runner recorded 0.250314 GPU-hours, 1.13 seconds over the nominal cap during process teardown. This remains a blocker; no retry is authorized under the exhausted cap.
- **Existing checkpoint metrics (partial attribution only):** the verified P50 checkpoint receipt and A0/B0/C0/H0 validation receipts use the same FF++ VAL hash. P50’s stored full-classifier metrics are AUC 96.5348%, FPR 9.9951%, FNR 9.9042%; A0 is 96.6217%, 9.9951%, 9.6036%. These are pre-existing saved metrics, not a completed new audit. They do not isolate the encoder from the changed heads/training recipe, and no paired P50 score cache exists for uncertainty or class-score-distribution analysis. The aggregate group metrics are in `PRETRAINED_CLASSIFIER_EXISTING_METRICS.json`.
- **TRAIN-only source generalization:** the deterministic Deepfakes-held-out fold is prepared and image-decoded. Fit has 3,745 rows/160 recorded components; development has 12,504 held-out-family fakes and only 269 reals from 41 components. Several demographic real cells are sparse or empty. Both real-input CUDA smokes passed on separate gpu02 H100s; the fresh seed-17 initialization hashes match. However, measured profiles project **10.5848 GPU-hours for S0 and 10.4965 for S1** before a safety margin (21.0813 aggregate GPU-hours), exceeding the fixed 5 GPU-hour cohort cap. Training was not started; no truncated schedule, smaller batch, or extra budget was substituted. The result can only support a detection-generalization experiment if a new budget/proposal is authorized.

The separate proposal, exact protocol and scripts are in [`source_generalization_20261105_r1`](../../proposals/source_generalization_20261105_r1/README.md). The complete-classifier audit script and runbook are in [`lh_objective_priority_20261105_r1`](../../proposals/lh_objective_priority_20261105_r1/PRETRAINED_CLASSIFIER_AUDIT.md). Row-level scores, fold manifests and checkpoints remain private.
