# Source-tail head-training cohort: completed results

**Execution/export verdict: PASS. Scientific conclusion: no meaningful fairness benefit established; primary fairness uncertainty INCONCLUSIVE. Stop after this cohort—no automatic extension or tuning.**

Both exact three-epoch seed-17 arms completed, both selected epoch 1 under the unchanged rule, and both reconstructed full students passed exact native SELECT replay. All 274 encoder parameter/BatchNorm tensors match the teacher bit for bit. The source-tail coefficient changed learned head parameters and reduced its TRAIN surrogate relative to matched control, but its hard validation fairness effect is tiny and not statistically resolved. This does not beat Ding or validate the full L_H/counterfactual method.

## Native SELECT comparison

All rates, AUC and summed disparity below are multiplied by 100. `100F_FPR` is a sum, not overall FPR. Rows are coherent selected checkpoints, not best cells across epochs/seeds. There are 31,107 SELECT images (2,438 real / 28,669 fake). Teacher and students use their own already declared global validation-real FPR10 calibration; no group-specific threshold is introduced.

| Model | Selected epoch | AUC% | FPR% | FNR% | Gender×race100F_FPR | Worst-group AUC% |
|---|---|---|---|---|---|---|
| teacher | existing | 99.8057 | 9.9672 | 0.2825 | 37.3715 | 99.5181 |
| control | 1 | 99.8073 | 9.9672 | 0.2895 | 34.8501 | 99.5186 |
| treatment | 1 | 99.8069 | 9.9672 | 0.2895 | 34.8206 | 99.5187 |

Treatment minus control: AUC −0.000449 percentage points; FNR unchanged at 0.289511%; primary100F_FPR −0.029543 points (about 0.085% relative). The teacher-to-control change must not be credited to source-tail training: both downstream arms share further BCE+fake-score-guard adaptation, and only treatment adds the source term.

**Fixed probability0.5 view, kept separate:**

| Model | Selected epoch | AUC% | FPR% | FNR% | Gender×race100F_FPR | Worst-group AUC% |
|---|---|---|---|---|---|---|
| teacher | existing | 99.8057 | 2.7482 | 1.4999 | 11.6068 | 99.5181 |
| control | 1 | 99.8073 | 2.9532 | 1.4999 | 11.4714 | 99.5186 |
| treatment | 1 | 99.8069 | 2.7071 | 1.5278 | 11.4030 | 99.5187 |

At fixed0.5 the treatment has lower FPR but higher FNR than control. These decisions differ from the primary validation-calibrated operating point; the better-looking view was not used to select a checkpoint or substitute a result.

## All reported demographic axes and regressions

| Axis | Teacher100F_FPR | Control100F_FPR | Treatment100F_FPR | Treatment−control | Point reading |
|---|---|---|---|---|---|
| gender | 4.1065 | 3.9370 | 3.9370 | +0.0000 | unchanged |
| race | 17.5414 | 16.3977 | 16.3611 | -0.0365 | lower point estimate |
| gender_race | 37.3715 | 34.8501 | 34.8206 | -0.0295 | lower point estimate |
| age_bin | 9.5760 | 9.8828 | 9.8828 | +0.0000 | unchanged |
| skin_bin | 14.2273 | 13.3652 | 13.3778 | +0.0127 | worse |
| ids8 | 37.6754 | 37.3027 | 37.8110 | +0.5083 | worse |
| age_x_skin16 | 80.6409 | 76.1496 | 73.9342 | -2.2154 | lower point estimate |
| coarse_gender_age_race8 | 24.8878 | 23.6113 | 24.1222 | +0.5109 | worse |

Gender/age disparity does not change; race/joint point reductions are very small. Skin, inherited `ids8`, and the explicitly defined coarse gender×age×race eight-group view worsen. Age×skin point disparity decreases, while its worst-group AUC declines. `ids8` is the inherited model-estimated label, not the coarse eight-group axis. Age/skin labels are estimates and unknowns remain in native totals. Extra-intersection estimates are descriptive point-only; no newly selected objective/threshold is justified by a favorable cell. Full per-group FPR/FNR/AUC/support and both fixed0.5/VAL10 views are in the published CSVs.

## Acquisition-source false alarms and TRAIN surrogate

| Model | FFHQ false positives | FFHQ FPR% | FairFace false positives | FairFace FPR% |
|---|---|---|---|---|
| teacher | 164/242 | 67.7686 | 79/2196 | 3.5974 |
| control | 158/242 | 65.2893 | 85/2196 | 3.8707 |
| treatment | 159/242 | 65.7025 | 84/2196 | 3.8251 |

The treatment has one more FFHQ false positive and one fewer FairFace false positive than the control at their stored validation-calibrated thresholds. The hard source gap is not improved despite the smoothed TRAIN penalty change. This is consistent with the limitations of a fixed teacher-tail surrogate and changed score calibration; it does not establish a unique failure mechanism.

Full-TRAIN descriptive source-tail penalties at the selected heads:

| Model | Full-TRAIN source-tail penalty | Full-TRAIN fake-score guard |
|---|---|---|
| teacher | 0.050589863 | 0.000000000 |
| control | 0.057426944 | 0.000099998 |
| treatment | 0.052906651 | 0.000100133 |

Treatment penalty is 7.87% below the control, but remains above the original teacher. These are in-sample TRAIN quantities computed by CPU head arithmetic on existing cached features, not new encoder inference or independent evidence. The fake-score guard discourages fake-score decreases; it does not guarantee zero fake errors. References stay fixed at TRAIN-real 85/90/95 score quantiles, teacher-real IQR scale and pooled soft risks. All family/source FNR and utility rows are included rather than hiding regressions.

No source-standardized race/joint metric is manufactured: FFHQ-real Female-Others and Male-Others remain absent. Source breakdowns of fakes refer to recorded contributing-reference acquisition, not proven pixel origin.

## Epoch histories and actual exposure

| Arm | Epoch | Feasible | AUC% | FNR% | Joint100F_FPR | BCE step mean | Source-tail step mean | Fake guard step mean |
|---|---|---|---|---|---|---|---|---|
| control | 1 | True | 99.8073 | 0.2895 | 34.8501 | 0.102815 | 0.056816 | 0.000184 |
| control | 2 | True | 99.8008 | 0.3035 | 35.0203 | 0.101168 | 0.062204 | 0.000329 |
| control | 3 | True | 99.7883 | 0.3070 | 35.0203 | 0.100433 | 0.066347 | 0.000507 |
| treatment | 1 | True | 99.8069 | 0.2895 | 34.8206 | 0.103121 | 0.054477 | 0.000163 |
| treatment | 2 | True | 99.8073 | 0.2895 | 34.9454 | 0.101683 | 0.055747 | 0.000328 |
| treatment | 3 | True | 99.8065 | 0.2930 | 34.9454 | 0.100850 | 0.056923 | 0.000549 |

Every arm completed 20,463 optimizer updates: 6,821 per epoch, all 218,266 original TRAIN rows once per epoch (654,798 original-image exposures per arm). Both used the same deterministic auxiliary draws:16 reals from each source per update. That is 109,136 auxiliary exposures/source/epoch, or 654,816 auxiliary exposures across both sources over three epochs per arm. Sampling reuse is not unique-source coverage or more independent identities.

Reconstructing the exact auxiliary schedule from the fixed seed confirms every FFHQ real appears in each epoch; FairFace unique rows seen were 13,595 /13,599 /13,596 of 13,601. Average source reuse is 34.36 exposures per FFHQ real per epoch versus 8.02 per FairFace real. Per-source×joint-group exposure, including structural zero cells, and actual unique coverage are in AUXILIARY_SOURCE_JOINT_EXPOSURE.csv and ACTUAL_AUXILIARY_COVERAGE.csv. Main BCE coverage was identical across arms; no difficult original was excluded.

## Paired recorded-contributor uncertainty

The 2,000-draw analysis uses seed 17 and the audited 45 SELECT-bearing recorded identity/master/donor components, same draws for both selected heads, fixed thresholds and fixed group eligibility. It requires at least 1,900 valid draws. These are dependency components, not 45 biological people; derived frames are not treated as independent subjects. Intervals condition on the selected checkpoints and shared pretrained encoder and do not account for repeated validation selection or label error. All intervals are exploratory and unadjusted across axes.

| Treatment−control metric | Difference×100 | 95% interval×100 | Valid /2000 | Status |
|---|---|---|---|---|
| AUC | -0.000449 | [-0.0016, 0.0008] | 2000 | EXPLORATORY_INTERVAL |
| FPR | +0.000000 | [-0.1129, 0.1360] | 2000 | EXPLORATORY_INTERVAL |
| FNR | +0.000000 | [-0.0102, 0.0099] | 2000 | EXPLORATORY_INTERVAL |
| F_FPR/gender | +0.000000 | [-0.1894, 0.2694] | 2000 | EXPLORATORY_INTERVAL |
| F_FPR/race | -0.036550 | [-0.4996, 2.1221] | 1938 | EXPLORATORY_INTERVAL |
| F_FPR/age_bin | +0.000000 | [-0.5798, 0.8307] | 2000 | EXPLORATORY_INTERVAL |
| F_FPR/skin_bin | +0.012687 | [-0.5441, 0.6205] | 2000 | EXPLORATORY_INTERVAL |
| F_FPR/gender_race | -0.029543 | withheld | 999 | INCONCLUSIVE_COMPONENT_SUPPORT |
| F_FPR/ids8 | +0.508251 | [-0.3143, 2.3837] | 1997 | EXPLORATORY_INTERVAL |

The primary joint-disparity CI is withheld because only 999/2000 draws retain all joint-group denominators. It is INCONCLUSIVE; neither significance nor zero effect is established. The other reported intervals span zero. Sparse support is not repaired by dropping cells, splitting components or changing the 95% rule.

## Exact implementation and verification

Scientific commit: **9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754**, clean detached source at `/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109`. Complete trained L_A head is retained at initialization; only its 2,049 parameters are optimized. One Xception encoder is frozen and shared through a float32 TRAIN/SELECT cache. No original L_H probe/alignment terms, JPEG or synthetic demographic edits are included. Batch 32, seed 17, three epochs, AdamW head LR 1e-4, weight_decay 1e-4, clip 5, lambda_fake 1; source coefficient 0 versus 0.1 is the only arm change.

- Supplied CPU synthetic pipeline checks PASS, including resume identity, objective gradients, cache tampering and feature/native-forward equivalence. Actual CUDA fixture lifecycle and full 224 feature/native-forward checks PASS; neither is a benchmark result.
- Actual teacher hash, source sidecar/provenance and manifest bindings PASS; TRAIN-real source labels exactly 3,176 FFHQ /13,601 FairFace. Source labels derive from canonical recorded keys, not appearance. No mixed TEST/CAL master bank was used.
- Shared production cache hashes PASS; teacher feature/head/native forward max difference 0 and exact stored native validation replay PASS. The cache contains 218266×2048 TRAIN and 31107×2048 SELECT float32 pooled features, with checkpoint-bound image SHA checking during every actual image read.
- Independent outcome verification matches all nine cache blob hashes, exact TRAIN reference recomputation, complete teacher head, fixed epoch-selection logic, exact actual steps/auxiliary schedule, selected checkpoint bytes, full-export head and 274 unchanged encoder/BN tensors. Both exports record source labels unnecessary at inference.
- A presentation-only float32/float64 comparison issue was caught before publication: comparing cached float32 teacher logits directly against a nextafter float64 threshold rounded the boundary. Reporting now explicitly uses float64 decisions as the native evaluator does. The final teacher and both student metrics independently match native reports. The initial presentation attempt is preserved privately; no scientific cache/config/result or threshold was altered.

All exact source/config/cache/head/full-student hashes are in FINAL_VERIFICATION.json, CACHE_VERIFIED.json, *_RUN_VERIFIED.json and *_EXPORT_VERIFIED.json. Full-weight/feature/row-score files remain private.

## Compute and operational accounting

| Stage | Wall minutes while one GPU lease held | Outcome |
|---|---|---|
| qualify_profile | 3.071 | COMPLETED |
| prepare | 16.572 | COMPLETED |
| train_control | 1.760 | COMPLETED |
| train_treatment | 1.530 | COMPLETED |
| export_control | 0.576 | COMPLETED |
| export_treatment | 0.512 | COMPLETED |

The actual complete lease was **0.401480 GPU-hours (24.089 minutes)**, including qualification, loading, I/O, cache, both head runs, exports, held idle and bookends; the cap was 2 GPU-hours and was not exceeded. Device: gpu01 physical GPU 6, H100 80GB, UUID GPU-61f54ca7-3cee-1c93-def3-118523e95241. Existing user-authorized shared-pool-v2 capacity lease, not an exclusive/PBS allocation claim; all CUDA commands were bound to that one UUID and inside the existing ledger/deadline wrapper. No other job was displaced. No source-tail process or ledger reservation is active now.

Historical missing inference was conservatively charged 3.227956 GPU-hours over the full recorded first-to-last three-card window. This is an upper-bound accounting charge, not fabricated measured actual GPU time. Existing older manual measured charges were preserved and their missing state fields normalized for the strict ledger wrapper. Total shared charge now **68.230538/72**, leaving **3.769462 recorded GPU-hours**; none is automatically granted to another cohort. No original L_H budget was transferred.

Initial launch failed before leasing because a linked-worktree .git path lived in login-node-local /tmp. It was preserved and replaced by a self-contained shared detached checkout of the identical commit; source checks were not disabled. An idle-only capacity attempt also stopped before acquisition. Both consumed zero GPU lease time. The successful shared-pool route is documented separately, with actual binding/lease evidence.

## Final decision and stop

**Software/execution/export PASS; source-tail fairness contribution remains unproven and the primary interval INCONCLUSIVE. Do not scale or retune automatically.** The treatment meets the fixed utility guards but has virtually the same selected primary disparity as its matched control, some regressions, and no identified improvement in hard source FPR. There is no evidence from this cohort that we beat Ding. No new Ding computation or TEST/CAL images, predictions, tuning or evaluation occurred; only TRAIN/SELECT were scored in the authorized stages. Historical execution metadata was inspected solely to reconcile accounting.

The cohort is finished. Existing logs/results, frozen scientific source and all private artifacts are preserved. Any new study requires a separate reviewed proposal; remaining hours are not permission for another candidate.
