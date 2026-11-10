# Fully matched FF2 progress and completed-seed TEST results

Observed management UTC: 2026-11-10T06:47:22.308162+00:00.

**8 of 21 training runs complete with verified TEST evaluation; 10 active; 3 queued; 0 failed.**
The shared TRAIN-only SFD preparation is separate from the 21-run count.
All seven seed-17 methods and Ding seed7 have completed ten epochs and frozen-checkpoint TEST evaluation.
The remaining queued runs are L_H, L_H ramp, and L_H low response at seed27.

The additional standby workers have increased active capacity from eight to ten physical GPUs: six on gpu01 and four on gpu02. All ten leases, worker PIDs and training child process groups were checked alive. Neither node had an additional eligible GPU. Both PBS queues were at their per-user running limit (2); no additional PBS submission was made. Existing consumers continue to pick up the finite queue as eligible resources appear.

[Training status](STATUS.md), [resource/process observations](SNAPSHOT.json), [three-seed table](RESULTS.md).
These are dated observations, not continuously published GitHub monitoring.

## Completed-seed TEST metrics

Global thresholds were frozen from SELECT-real calibration at 10% empirical FPR. TEST FPR can differ. No threshold was selected or recalibrated using TEST. AUC/FPR/FNR below are percentages; joint disparity is the sum of absolute group FPR deviations multiplied by 100, not a max–min gap.

| Method | Seed | Selected epoch | TEST AUC | TEST FPR | TEST FNR | Joint 100F_FPR |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 17 | 7 | 99.0146 | 15.4200 | 1.2664 | 67.9232 |
| DAG-FDD | 17 | 7 | 99.0065 | 15.0891 | 1.3809 | 65.4379 |
| DAW-FDD | 17 | 7 | 99.0083 | 14.9237 | 1.4381 | 64.3660 |
| Ding SFD+GDA (common representation) | 17 | 2 | 99.0169 | 15.2178 | 1.2786 | 66.3557 |
| L_H | 17 | 10 | 98.8270 | 13.7842 | 2.5547 | 67.4903 |
| L_H ramp | 17 | 2 | 99.0114 | 14.8870 | 1.4734 | 63.7480 |
| L_H low response | 17 | 2 | 99.0114 | 14.8870 | 1.4734 | 63.7480 |
| Ding SFD+GDA (common representation) | 7 | 9 | 98.9996 | 15.2545 | 1.3443 | 66.5026 |

These are completed individual seeds, not final three-seed means or confidence intervals. The earlier [five-epoch validation snapshot](../ff2_fully_matched_interim_20261110/README.md) remains unchanged. It must not be substituted for these TEST measurements. The results do not establish an overall fairness or detection win for L_H. Ramp/low seed17 both select epoch2, before response losses activate, and have identical reported metrics. Their checkpoint file hashes differ. Equal metrics are not evidence that response learning improved the selected model.

All methods share one seed17 P50 pretrained encoder, with downstream seeds7/17/27. The study controls the shared architecture, initializer, optimizer, update budget, training input exposure, selection and evaluation, with declared objective and Ding feature-mask differences. These are controlled adaptations, not native-paper reproductions. Demographic age/skin labels are estimates. Prior benchmark TEST results had been exposed historically; this is a descriptive controlled comparison. [Exact source and protocol](../ff2_fully_matched_interim_20261110/source/proposals/ff2_fully_matched_20261110_r1/README.md).

[All 48 method/axis TEST rows](TEST_ALL_AXES.csv), [joint rows](TEST_JOINT.csv), [native aggregate/selection receipts](native), [publication checks](VERIFICATION.json). The checks use saved aggregates and receipts; they do not execute models, read private predictions or rehash checkpoint binaries. No dataset images, weights or personal row-level outputs are included. [Artifact SHA-256 hashes](SHA256.json).
