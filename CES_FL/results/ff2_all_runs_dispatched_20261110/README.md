# All 21 FF2 runs dispatched

Observed management UTC: 2026-11-10T09:02:02.202427+00:00.

**15/21 complete with verified TEST evaluations; 6 active; 0 queued; 0 failed.** Shared SFD preparation is excluded from this training-run count.

The final two queued runs are now training on gpu01: L_H ramp seed27 on GPU4 and L_H low response seed27 on GPU6. Both passed the actual CUDA backward, parameter-count, BatchNorm, same-seed initialization and actual forward-input checks. Their worker and training process groups were checked alive; both had reached step1000 of epoch1 at verification. [Launch evidence](FINAL_TWO_LAUNCH_RECEIPT.json).

Existing finite workers dispatched them when eligible GPUs became available. No training source, model configuration or allocation policy changed. No duplicate training or additional PBS job was launched. [Actual per-run state](STATUS.csv).

## Available results

[All completed-seed TEST metrics across six axes](TEST_ALL_AXES.csv), [complete-three-seed mean and sample SD](THREE_SEED_MEAN_SD.csv), [selection/checkpoint/threshold consistency checks and native aggregate hashes](SNAPSHOT.json).

All seven methods have completed seeds17 and7; Ding has also completed seed27. The three-seed Ding mean is available; final three-seed comparisons for the other methods remain pending. Means represent downstream seeds conditional on one shared seed17 pretrained encoder, not independent pretraining repetitions or confidence intervals.

These are controlled objective adaptations under the [fixed common protocol](../ff2_fully_matched_interim_20261110/source/proposals/ff2_fully_matched_20261110_r1/README.md), not native-paper reproductions. Frozen SELECT-real calibration thresholds are used on TEST, where achieved FPR can differ. No TEST-based threshold tuning or checkpoint selection is performed. Prior TEST benchmarks had already been exposed historically. The complete response objective has not established an overall fairness/detection advantage. Ramp/low seed17 select epoch2 before response losses activate; matching scores are not evidence for improved response learning.

This is a dated GitHub snapshot, not a continuously updated publication service. Historic result snapshots are preserved. Packaging reads native aggregate reports and receipts, not private predictions or checkpoint binaries, and executes no model. No images, weights or personal row-level outputs are included.
