# Completed seed17 original-only baselines: detection similar, no primary fairness win

Observed management UTC: **2026-11-01T20:53:50.835835+00:00** (2 November 2026 IST). **P, BCE and fair all completed50 epochs and passed their native independent selected-checkpoint validation replay.** No corresponding worker/training PIDs remain live. This report concerns **FF++ validation**, not TEST or the proposed manipulation-response arm.

## Selected-checkpoint results

Each downstream arm used the same verified P encoder, identical fresh head/gate/fusion seed17, original ordering and fixed label×ids8 weighting, preprocessing, optimizer budget and validation-AUC selection/calibration policy. Both completed50 epochs; checkpoint selection remained validation AUC only. No checkpoint was changed after viewing fairness. Each calibrated threshold belongs to its selected checkpoint and uses validation real images only; both achieved overall validationFPR9.995112% (409/4092). No testFPR is guaranteed.

| Arm | Status | Epochs completed | Selected epoch | Val AUC (%) | Val FNR (%) | gender_race worst-group AUC (%) | gender_race FPR gap (pp) | gender_race F_FPR (fraction) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | COMPLETED | 50 | 22 | 98.448762 | 3.841458 | 95.032007 | 47.935606 | 0.657819251 |
| fair | COMPLETED | 50 | 37 | 98.487114 | 3.587865 | 94.049637 | 50.619890 | 0.684208602 |

Fair minus BCE: AUC **+0.038352pp**, FPR max–min gap **+2.684284pp**, summed F_FPR **+0.026389352**, and worst-group AUC **-0.982370pp**. The fair baseline slightly increases AUC and lowers FNR, but its **primary F_FPR is worse**, its FPR gap is larger and worst-group AUC is lower. F_DP improves slightly (0.15250→0.14875) and es_AUC improves (88.9101→89.5524%), so outcomes are mixed rather than uniformly adverse. **This single-seed validation result does not support a primary fairness-improvement claim.** It does not test or refute the unrun manipulation-response objective.

FPR gap is max–min group FPR, in percentage points. F_FPR is the sum of absolute group deviations from the overall FPR, in fraction units; these are different measures. No confidence intervals, population superiority, test improvement or direct win over Ding is established here. Do not compare these validation AUCs to Ding's published test AUCs.

## Selected gender_race subgroup support and FPR

| Group | VAL real n | VAL fake n | BCE FPR (%) | fair FPR (%) | BCE AUC (%) | fair AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| Female-Asian | 275 | 1153 | 8.3636 | 13.0909 | 98.7839 | 98.7618 |
| Female-Black | 30 | 120 | 53.3333 | 56.6667 | 97.7500 | 97.8611 |
| Female-Others | 89 | 810 | 14.6067 | 7.8652 | 99.3203 | 99.6962 |
| Female-White | 1753 | 8874 | 5.6475 | 6.0468 | 99.5526 | 99.5740 |
| Male-Asian | 352 | 1733 | 5.3977 | 6.2500 | 99.1222 | 98.8484 |
| Male-Black | 90 | 390 | 10.0000 | 6.6667 | 96.0057 | 97.4359 |
| Male-Others | 239 | 1170 | 9.2050 | 10.4603 | 95.0320 | 94.0496 |
| Male-White | 1264 | 7044 | 16.4557 | 15.0316 | 97.2628 | 97.5567 |

Female-Black has only **30real validation images**: BCE makes16 false positives (53.33%) and fair 17(56.67%). A one-image change is 3.33percentage points in that group. Support/noisy proxy annotations and known benchmark contributor overlap limit interpretation; do not equate these frame counts with independently identified people. Full gender/race/age/skin/ids8/family metrics and supports are in aggregate.json. Eligibility remains min 20 per class, unknown groups excluded from eligible aggregates, as coded.

## Exposure and surrogate diagnostics

Both arms used all **76,139original TRAIN images once per epoch**: **3,806,950original exposures per arm**. There were **zero quartets and zero paired draws**. Their real-group-count histograms match epoch by epoch, supporting the intended matched original sampling. Seed, input manifest hashes, initialization and source are fixed; no extra seed was run.

The fair term is a TRAIN-only soft real-group disparity surrogate at fixed logit 0, not the eventual calibrated FPR metric. At the selected checkpoints, batches with fewer than two real groups occurred in **33.3053%** of BCE's selected epoch and **32.2337%** of fair's selected epoch. The fairness term is zero on such batches by design. This is a limitation worth investigating; it is not a proved cause of the fairness outcome or permission to change batch size/sampling after results. The original development/selection rule is preserved.

## What was actually verified now

The native training source independently replayed each selected checkpoint on all validation images and required exact metric equality before writing RUN_VERIFIED and DONE. This operator audit **did not rerun another full validation forward**. It rehashed each actual selected checkpoint; loaded it with weights_only on CPU; verified finite model tensors, exact source/config/arm/P dependency, selected epoch versus all 50 history records, stored validation/calibration and zero quartet use. Both actual checkpoint hashes match native replay receipts. Command/output evidence is preserved under the cluster check record below. No test data or scores were read by the new runs or this verification.

| Arm | Actual checkpoint SHA256 |
|---|---|
| Shared P | `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5` |
| BCE | `71d9b50377d3e0812c2721d9271ca3f4d59aaeedfa798a69853d9bff769bfb0b` |
| fair | `21a9fedc42be28b261e59c8bcb6f969bb422744453a9e0217fd5a32f6ab6d974` |

Both arms use bound config SHA256 `b3d19e4937897ca568836dc67587f4d5f3b14222963b9bfb676a19be82627ece` and science commit **58773bbef9d9fe79dd313610f694ddeea12c6037**. Trainer SHA256 `0e82a522a22e641dc5588b6a15317e6842896ab2bdcc5364833af25e207f257e`; architecture SHA256 `3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab`. No scientific source/config change, new training or generated-image run occurred for this status/verification task.

## Current per-arm state and next dependency

| Arm | Actual status | Remaining work |
|---|---|---|
| P50 | COMPLETED | Verified, selected epoch29; no rerun |
| BCE | COMPLETED | Verified selected checkpoint and validation metrics preserved |
| fair | COMPLETED | Verified selected checkpoint and validation metrics preserved |
| nochange | BLOCKED_INPUTS | Independently reviewed complete matched manipulation quartet bank, bind and own backward check |
| score | BLOCKED_INPUTS | Same bank/coverage/P requirements and own backward check |
| response | BLOCKED_INPUTS | Same bank/coverage/P requirements and own backward check |

Qwen32 completed **224/224outputs** across32 originals, but the pending224-row review artifact still has **zero approved rows and zero named reviewers**. No bound_quartets output exists. Qwen candidates provide x0/xa at most; they are not the missing t0/ta matched manipulation roles. No approval is inferred from earlier messages. The prospective source/donor record and cached SimSwap discovery are not a qualified bank.

**Next dependency:** prepare/qualify the TRAIN-only matched manipulation operator and collect substantive independent quartet review, preserving the declared group×age-direction coverage, contributor exclusions and processing controls. Only after actual bank validation and per-arm backward PASS may paired training proceed. No bank threshold, coverage requirement, loss or selection criterion was relaxed after these results. FairForge-2 remains DO NOT RUN. No TEST evaluation is attached.

## Exact cluster outputs and publication

- Native run root: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z`.
- BCE history/config/checkpoints/replay receipt: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/bce/`.
- fair history/config/checkpoints/replay receipt: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/fair/`.
- Final operator audit: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/checks/BASELINES_FINAL_OPERATOR_VERIFIED.json`.
- Logs: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/bce.log`, `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/fair.log`.
- Qwen pending review: `/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z/quality_review_pending.csv`.
- Public aggregate supports/metrics/reconciliation: `aggregate.json`; per-arm native receipts/config/provenance and50epoch summaries: `bce/`, `fair/`.

The finite read-only training observer published completion and stopped normally (`ORIGINAL_TRACK_COMPLETED_PAIRED_TRACK_STILL_GATED`). It cannot advance the paired track. Its last status publication was 2026-11-01T19:45:42.465025+00:00. [Timestamped stage history](../../status/manip_response_ssh_20261101T115720Z_live.md). Subsequent updates are explicit; an old status timestamp is not continuous monitoring. Historical checkpoints, outputs, config and logs were preserved; binary checkpoints, face images, row-level reviews and credentials remain private.
