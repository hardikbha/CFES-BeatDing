# Final FF2 controlled training comparison: 21/21 complete

Observed management UTC: 2026-11-10T11:47:11.033219+00:00. Seven controlled method adaptations × three downstream seeds (7,17,27), each with ten complete fine-tuning epochs and frozen-checkpoint TEST evaluation on 87,564 images. Shared SFD preparation is separate from these 21 training runs. Zero failed, queued or active runs are recorded in this suite. No new inference or model execution was performed to publish this snapshot.

## Primary TEST results

Mean ± sample standard deviation over the three downstream seeds. AUC, FPR and FNR are percentages. Joint disparity is **100F_FPR**, the sum of absolute group-FPR deviations from pooled FPR across the eight gender×race cells; it is not an error rate or max–min gap. Lower disparity is better.

| Method | AUC % ↑ | Achieved FPR % ↓ | FNR % ↓ | Joint 100F_FPR ↓ |
|---|---:|---:|---:|---:|
| BCE | 99.026 ± 0.019 | 15.35 ± 0.27 | 1.26 ± 0.15 | 68.55 ± 2.89 |
| DAG-FDD | 99.006 ± 0.006 | 15.10 ± 0.09 | 1.37 ± 0.05 | 65.71 ± 0.71 |
| DAW-FDD | 99.009 ± 0.010 | 14.91 ± 0.13 | 1.42 ± 0.04 | 64.45 ± 0.34 |
| Ding SFD+GDA (controlled adaptation) | 99.011 ± 0.010 | 15.17 ± 0.12 | 1.32 ± 0.04 | 66.00 ± 0.75 |
| Our full L_H | 98.936 ± 0.095 | 15.62 ± 1.63 | 1.43 ± 0.98 | 79.47 ± 11.83 |
| Our L_H ramp | 99.014 ± 0.007 | 15.13 ± 0.41 | 1.38 ± 0.18 | 65.30 ± 3.44 |
| Our L_H low response | 99.014 ± 0.007 | 15.13 ± 0.41 | 1.38 ± 0.18 | 65.30 ± 3.44 |

BCE has the highest mean AUC and lowest mean miss rate. DAW has the lowest mean primary intersectional false-positive disparity. Full L_H has lower AUC, higher primary disparity and substantially more seed variation than the controlled Ding adaptation. Ramp/low improve over full L_H and reduce primary disparity versus BCE, but do not establish a significant advantage over Ding or DAW. These completed TEST results do not support an overall fairness/detection superiority claim for the response objective.

**Every ramp/low run selected epoch 2**, before the response losses activate at epoch 3. Ramp and low have identical selected-checkpoint aggregate metrics for every seed; their serialized checkpoint hashes differ. Their favorable rows cannot be attributed to later response learning. Full L_H does receive its response objective during the selected training trajectory. Prior five-epoch validation observations must not substitute for these TEST findings.

Every checkpoint used its frozen global SELECT-real threshold, targeting 10% empirical validation FPR. Achieved TEST FPR is approximately 15%, revealing calibration shift; there was no TEST threshold retuning. Per-group thresholds were not used. Age and skin labels are automatic estimates, and source composition can confound demographic differences.

## Scope and verification

All methods share the data, Xception/linear architecture, transferred encoder, trainable layers, optimization/update schedule, original/probe input exposure, selection policy and evaluator. Objective ports and Ding's declared TRAIN-derived feature mask differ. These are controlled adaptations, not faithful reproductions of each native published recipe. L_H uses shared unweighted BCE and joint-only auxiliaries in this track; it differs from other historical full-L_H studies.

The three downstream seeds share one seed-17 P50 encoder. Their SD measures conditional downstream training variation; it is neither a confidence interval nor independent pretraining replication. Benchmark TEST had been exposed historically; this study is descriptive and not an untouched confirmation. The separate no/random/informed channel-selection pilot has no production result and contributes no row here.

[Per-seed results across all six axes](TEST_PER_SEED_ALL_AXES.csv), [all primary means/SDs](TEST_PRIMARY_MEAN_SD.csv), [per-family results](TEST_PER_FAMILY.csv), [selected checkpoints](SELECTED_CHECKPOINTS.csv), [all 210 epoch selection summaries](SELECTION_HISTORY.csv) and [response-schedule provenance](SCHEDULE_PROVENANCE.json), [native training/frozen-checkpoint/TEST/CUDA/config receipts](native), [common recipe](COMMON_RECIPE.json), [fixed matrix](MATRIX_VERIFIED.json).

The exact training source is local detached commit `b7fa561b43a659b24efaf1e928d6bbf41a839b63`; [recorded hashes](SOURCE_CHECKOUT.json) and the [earlier exact source/config copies](../ff2_fully_matched_interim_20261110/source) retain its implementation. All 31 tracked source/config hashes were checked unchanged at publication. The portable verifier below checks native receipt consistency, selection chronology, full original coverage, same-seed initial parameters/forward inputs, all 126 method/axis rows and all 63 primary means/SDs. It executes no models and reads no private predictions. Checkpoint binary hashes are linked between existing receipts; binaries were not rehashed during publication.

```bash
python3 verify_final_results.py
```

No images, binary weights, private row-level predictions or per-case review notes are published. Historical snapshots remain unchanged; this GitHub packet is a dated final-results artifact, not an unattended live service. [Verification](VERIFICATION.json), [artifact hashes](SHA256.json).
