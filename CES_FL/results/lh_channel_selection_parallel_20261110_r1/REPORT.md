# Channel-selection continuation: qualified and CUDA setup started

Observed management UTC: 2026-11-10T10:39:52.733591+00:00. This is a dated snapshot; inspect native timestamps for progress.

The user reported reviewing the specified panel, instructed parallel execution on free GPUs and supplied the reviewer name Aashish Negi. A source-bound human-review record was transcribed from those panel-level chat statements. No individually authored case observations were supplied; no AI visual approval was substituted. The pinned qualifier accepted all 344 cases. This is procedural-probe qualification, not validation of demographic annotations or real-forgery causality. [Review provenance](HUMAN_REVIEW_PROVENANCE.json), [qualification receipt](QUALIFIED.json).

The user's latest instruction supersedes the earlier one-GPU/sequential scheduling rule: up to three physical GPUs may run the three registered arms concurrently. The **two aggregate GPU-hour limit, three epochs and thirty-minute per-arm limits remain unchanged**. No new budget grant or transfer is made. [Recorded scheduling override](USER_PARALLEL_SCHEDULING_OVERRIDE.json).

A separate, detached operator controller preserves scientific source commit `1a6d63db6baf7f6ad2a8e842a31341d2c128de25`. Its exact code is in [operator](operator) and its local commit/hashes are recorded in [the binding](PARALLEL_CONTROLLER_BINDING.json). Eight CPU controller tests passed, including summed multi-device accounting, budget-fit rejection and a real process-group watchdog test that leaves an unregistered process alive. The inherited 14 scientific CPU tests and input/checkpoint checks remain valid for the unchanged scientific source.

The first controller startup stopped before any budget reservation, GPU lease or model update because Git was unavailable on the compute node. A recorded operator decision corrected execution metadata verification to read detached HEAD and bound file hashes directly; the failed attempt remains preserved. This was not an automatic scientific retry. [Correction decision](STARTUP_CORRECTION_DECISION.json).

At this snapshot: **TRAIN_ONLY_MASK_AUDIT** on gpu02. The initial lease is GPU6 (`GPU-3a2f4cf3-9b32-14c9-3deb-29e55d1123bb`), with actual CUDA UUID visibility verified and CUDA software smoke passed. The TRAIN-only selector is preparing one frozen mask set. Production training has not yet started. [Allocation](ALLOCATION_primary.json), [status](PARALLEL_STATUS.json), [actual aggregate accounting](PARALLEL_GPU_ACCOUNTING.json).

The controller reserves 2 hours in the existing original L_H ledger, with its historic charges preserved. All device-held time, including idle/loading/checks, is added across cards; parallel wall time is not mistaken for aggregate time. A watchdog covers owned children independently of budget-file locks. It retains cleanup margin inside the two-hour/30minute bounds and never terminates other jobs.

After frozen masks and all three real-input dry runs, a timing-only TRAIN-real/VAL forward profile discards logits and computes no validation metric. No VAL scores select masks. Every arm must fit the conservative individual/aggregate estimate before the remaining GPUs are acquired and training begins. Failure blocks the cohort; it does not authorize smaller epochs, missing controls, retries or extra hours. Only after all three native selected-checkpoint replays complete can CPU comparisons run. No TEST/CAL access, new Ding work or automatic extension is authorized.

This continuation supersedes the earlier missing-review block while preserving that historical [CPU audit](../lh_channel_selection_operator_20261110_r1/REPORT.md). No scientific benefit is claimed from qualification, CUDA smoke, selector diagnostics or discarded profiling updates.
