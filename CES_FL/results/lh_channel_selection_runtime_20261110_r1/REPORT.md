# Channel-selection runtime verification: BLOCKED before production training

Observed management UTC: 2026-11-10T11:18:08.742143+00:00. Execution uses the unchanged detached scientific commit [1a6d63d](https://github.com/hardiksharma6555/CFES-BeatDing/commit/1a6d63db6baf7f6ad2a8e842a31341d2c128de25).

**CPU/CUDA, teacher/data, qualification, mask audit and all three real-input backward checks passed. The remaining runtime gates FAILED. No production arm was launched.**

## Fixed cohort and passed gates

The user explicitly permitted the no-mask/random-mask/informed-mask configurations to run in parallel. The three-epoch budget, thirty-minute training-arm cap and two aggregate GPU-hour allowance were retained. Parallel execution changes wall time; it does not make three device-hours count as one.

A source-bound qualification receipt passed after the user stated the 344-case panel was reviewed and named Aashish Negi. The assistant transcribed that panel-level chat approval. Individual case observations were not supplied; the record must not be described as independent per-case notes, human validation of dataset annotations, AI visual approval or real-forgery causal validation. [Review provenance](../lh_channel_selection_parallel_20261110_r1/HUMAN_REVIEW_PROVENANCE.json).

The CUDA smoke passed on an explicit shared-pool UUID lease. The frozen selector used 1,024 TRAIN utility images and every one of the 344 qualified cases from 86 sources. It protected the same high-utility half for both masks; random and informed each remove 41 of 2,048 channels, while the control removes none. All configurations retain full L_H and the same completed L_A encoder/head warm start. Validation scores were not used for mask selection. [Masks and diagnostics](MASKS.json), [three bound configurations](configs), [binding and backward checks](MASK_AND_DRY_VERIFICATION.json).

All three real-input dry runs passed, each with one discarded update, finite full-L_H losses, a nonzero hierarchical-error gradient and the expected 20,809,001 total / 6,790,433 trainable parameters. Peak allocated CUDA memory was approximately 0.95 GiB. These are software/feasibility checks, not scientific results.

## Actual mixed-schedule timing

The first controller extrapolated a single cold joint-auxiliary dry run, including an extra fairness-gradient diagnostic, as if every training step incurred it. Its conservative 7.6–14.9-hour per-arm extrapolation is preserved in [the original gate](COST_GATE.json); it is not an observed training duration.

Following the user's continuation request, one separately identified profile measured the real registered shuffled original stream and the periodic five-axis auxiliary schedule on disposable copies of all three fixed-mask models. Each arm performed 40 warmup and 80 measured steps: all five axes contributed four measured auxiliary events, with identical measured main-source/class exposure. All 120 profile updates per arm were discarded; no production checkpoint or new validation metric was computed. Masks stayed frozen. [Representative profile](REPRESENTATIVE_PROFILE_attempt2.json).

| Configuration | Measured mean seconds/step | Optimizer-only projection, minutes | Conservative complete-arm projection, minutes | Allowed minutes |
|---|---:|---:|---:|---:|
| none | 0.0997 | 33.99 | 60.79 | 30 |
| random | 0.0985 | 33.61 | 56.05 | 30 |
| informed | 0.0981 | 33.46 | 55.73 | 30 |

Each configuration requires 20,463 optimizer steps for three full epochs. **Even the observed mean step rate, with no safety margin and before validation/loading/checkpoints, projects above the thirty-minute arm cap.** This is why the remaining gate cannot pass under the current recipe and limits.

The conservative estimate uses the slowest measured twenty-step block plus 10%, measured input/model setup and TRAIN-real calibration costs, four previously timed full VAL forwards, a checkpoint/I/O allowance and another 10% margin. The VAL timing profile discarded its logits and computed no metric. The complete-cohort projection, including charged checks, is **3.117 aggregate GPU-hours**, above the fixed 2-hour cap. [Exact refined cost gate](REFINED_COST_GATE_attempt2.json), [original VAL timing receipt](RUNTIME_PROFILE.json).

These are cost projections from a short profile, not completed three-epoch runtimes or performance measurements. No epoch count, batch, masks, loss, fairness guard, selector rule or sampling policy was changed to force a pass. No extra budget or FF2-ledger transfer was used.

## Preserved helper failure and accounting

The initial profile helper was named `profile.py`, which shadowed Python's standard module when AdamW imported cProfile. It failed before any optimizer update. That attempt and its 180.81 held-GPU seconds are preserved. A recorded operator decision renamed the timing helper and added two passing CPU tests for actual AdamW/cProfile import and inert helper import. No scientific source changed. [Correction decision](PROFILE_ENVIRONMENT_CORRECTION_DECISION.json), [regression checks](PROFILE_ENVIRONMENT_CPU_TESTS.txt), [failed traceback](failed_profile_attempt1.txt), [corrected helper](operator_profile).

The corrected profile completed in 143.23 held-GPU seconds. Total charged work, including initial CUDA/audit/dry-run/VAL timing and the failed helper attempt, is **0.240438 GPU-hours** of the 2-hour allowance. Ledger reservations are closed and both diagnostic leases are released; the finished controllers are no longer alive. [Ledger reconciliation](FINAL_BUDGET_VERIFICATION.json), [process/lease verification](FINAL_PROCESS_AND_RELEASE_VERIFICATION.json).

## Scientific outcome and next-action boundary

| Arm | Production status | Validation result / uncertainty |
|---|---|---|
| Full L_H, no mask | NOT STARTED | Not available |
| Full L_H, random mask | NOT STARTED | Not available |
| Full L_H, informed mask | NOT STARTED | Not available |

There is no trained-cohort fairness benefit, AUC/FNR utility-guard outcome, paired-bootstrap comparison or superiority evidence. Discarded dry/profile losses must not become result-table rows. TEST/CAL data were not accessed by this task; no new Ding experiment was launched and existing jobs were not stopped. The historical AUC ≥99.5%, FNR ≤0.5651% and ≥95% valid-bootstrap guards remain unchanged.

The blocker is the documented per-arm and aggregate runtime allowance, not a missing GPU or a coding defect in the mask/loss checks. Any future execution requires an explicit revised resource/experiment decision and a new bound launch; this report starts no retry or extension. The earlier snapshots remain preserved. [Machine-readable gate state](GATE_STATUS.json), [portable verifier](verify_runtime_audit.py), [artifact hashes](SHA256.json).
