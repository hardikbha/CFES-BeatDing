# Interim FF2 common-recipe results for review

**Seed 17 · SELECT/validation only · five epochs recorded out of ten planned.** This is the exact frozen snapshot discussed before publication. Training continues separately; this GitHub packet is a timestamped review artifact, not a continuously updated leaderboard. It contains no final TEST result, three-seed mean, confidence interval, significance claim or established win.

All seven rows use one coherent selected-so-far checkpoint: the earliest maximum overall validation AUC within the five recorded epochs. Later best checkpoints can change. Historical interim checkpoint binaries are not certified by this packet; final selected-model replay is a separate end-of-training step.

Packaged UTC: 2026-11-10T04:38:16.747429+00:00. Metric snapshot UTC: 2026-11-10T04:31:31+00:00.

| Method | Selected epoch so far | AUC ↑ | FPR ↓ | FNR ↓ | Worst-group AUC ↑ |
|---|---:|---:|---:|---:|---:|
| BCE | 5 | 99.82% | 9.97% | 0.30% | 99.57% |
| DAG-FDD | 2 | 99.81% | 9.97% | 0.28% | 99.56% |
| DAW-FDD | 2 | 99.80% | 9.97% | 0.28% | 99.53% |
| Ding SFD+GDA (common representation) | 2 | 99.81% | 9.97% | 0.29% | 99.53% |
| L_H | 2 | 99.70% | 9.97% | 0.53% | 99.36% |
| L_H ramp | 2 | 99.82% | 9.97% | 0.30% | 99.57% |
| L_H low response | 2 | 99.82% | 9.97% | 0.30% | 99.57% |

Fairness below is on the eight gender×race groups. Rates are fractions internally, multiplied by 100 for display. `100 F_FPR` sums absolute group deviations from pooled FPR; it is neither an error probability nor the highest-minus-lowest group gap. `F_MEO_max` means the maximum of the FPR and FNR group gaps; `F_OAE` is the maximum-minus-minimum group accuracy.

| Method | 100 F_FPR ↓ | FPR gap, pp ↓ | 100 F_DP ↓ | F_OAE, pp ↓ | es-AUC ↑ |
|---|---:|---:|---:|---:|---:|
| BCE | 44.69 | 17.01 | 1.76 | 1.24 | 98.78% |
| DAG-FDD | 38.62 | 15.37 | 1.64 | 1.15 | 98.73% |
| DAW-FDD | 38.77 | 15.21 | 1.58 | 1.12 | 98.64% |
| Ding SFD+GDA (common representation) | 42.01 | 15.69 | 1.77 | 1.15 | 98.68% |
| L_H | 35.24 | 12.95 | 1.55 | 1.40 | 98.16% |
| L_H ramp | 38.43 | 14.85 | 1.59 | 1.10 | 98.77% |
| L_H low response | 38.43 | 14.85 | 1.59 | 1.10 | 98.77% |

Overall validation FPR is 243/2,438 = 9.9672% for every row because each checkpoint is calibrated on these same validation reals to a 10% empirical budget. Equal validation FPR is by construction and does not establish held-out calibration. Every model must ultimately report achieved TEST FPR and FNR.

At this snapshot L_H trades lower joint false-positive disparity for lower AUC and a higher miss rate. The ramp/low rows select epoch 2, before response losses activate; they are not evidence that the later response-learning schedule improved the selected model. Age/skin labels are proxies. All 42 method/axis rows and all group denominators are retained below.

The controlled track holds architecture, initializer, trainable layers, BatchNorm, optimizer, update budget, sample order, original/probe forward exposure, selection and evaluator fixed. Only the registered objective and Ding's declared TRAIN-derived feature mask differ. These are **controlled adaptations**, not faithful reproductions of the native published training recipes. L_H uses common unweighted BCE and joint-only training auxiliaries here; Ding uses the common 2,048-feature/one-logit architecture and AdamW rather than its native 512-feature/two-logit/head-only-SGD recipe. Those adaptations need methodological review.

Common data: TRAIN 218,266 / SELECT 31,107; fixed public TEST 87,564 reserved for end-of-training scoring. All rows share one verified seed-17 P50 encoder and a fresh seed-17 linear head. The planned 7/17/27 repetitions are downstream seeds conditional on that shared encoder, not independent complete pretraining repetitions. Pretraining took 50 epochs; the 10-epoch budget is downstream only.

Review files:

- [Joint metrics](metrics_joint.csv), [all six axes](metrics_all_axes.csv), [group support and rates](group_support_and_rates.csv), and [five-epoch selection history](selection_history_prefix.csv).
- [Native selected validation aggregates](selected_validation_aggregates.json) and [snapshot scope](SNAPSHOT.json). AUC is checked against recorded history; it cannot be independently recomputed from aggregate group rates without private predictions.
- [Complete protocol](source/proposals/ff2_fully_matched_20261110_r1/README.md), [common recipe](evidence/COMMON_RECIPE.json), and [all 21 fixed configurations](source/proposals/ff2_fully_matched_20261110_r1).
- [Scientific runner](source/src/shared.py), [unchanged functional core](source/src/core.py), [pinned loss ports](source/src/ported_losses.py), [input-cache audit](source/data_cache.py), and [software tests](source/tests/test_shared.py). Source copies are byte-verified against the cluster execution record; the execution commit is local to the isolated cluster repository.
- [Nine software-check receipts](evidence/UNIT_CHECKS.json), [test log](evidence/unit_tests.log), [fresh full image audit](evidence/INPUT_AUDIT.json), [CUDA checks for all seven seed-17 methods](evidence/cuda_checks), and [TRAIN-only SFD mask](evidence/SFD_FROZEN.json). The procedural operator is an unchanged previously reviewed recipe; no new human study or approval of every training view is claimed.
- [Status at publication](evidence/STATUS_AT_PUBLICATION.json), [source hashes](evidence/SOURCE_CHECKOUT.json), and [artifact hashes](ARTIFACT_SHA256.json). No dataset, face image, binary weight, private row-level score or per-case reviewer decision is included.

Run the portable aggregate check from this folder:

```bash
python3 verify_snapshot.py
```

It checks selection consistency, recomputes all reported group fairness/gap metrics, verifies 21 common-recipe configs and equal recorded CUDA input/parameter fingerprints, and verifies exact source/artifact copies. It runs no models and requires no datasets or GPUs. Its [verification result](VERIFICATION.json) is separate from final model replay.

For the reviewer: assess whether the objective ports and shared-representation SFD preserve a useful comparison; whether the common unweighted classification/joint-only auxiliaries adequately control confounds; whether calibration and metric definitions are correct; and which conclusions are warranted before complete downstream-seed and TEST results. Please distinguish coding defects, protocol limitations and statistical uncertainty. This packet authorises no new training, TEST tuning or release-setting changes.
