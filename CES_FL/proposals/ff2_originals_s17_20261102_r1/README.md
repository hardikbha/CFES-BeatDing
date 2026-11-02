# FairForge-2 original-image pretraining, seed17 — new scoped release

Authorization: user on 2 November 2026 requested that FairForge-2 training start. The previous explicit direct SSH gpu01/gpu02 route authorization persists. This request releases **only original-image CES-FL pretraining P**, with the exact previously fixed ten-epoch configuration. It does not release the entire FF2 comparison or change the earlier review's findings.

The earlier full-comparison [DO NOT RUN review](../ff2_supervisor_review_20261101_r1/REVIEW.md) remains applicable to Ding adaptation, paired downstream arms, common evaluation and uncertainty reporting. Those components are not dependencies of original-only BCE pretraining. This new version explicitly isolates that prerequisite: no synthetic pairs, no unrecorded approvals, no Ding rerun, no full-comparison or novelty claim, no image generation, no automatic TEST evaluation or downstream launch.

## Exact configuration

`P_ff2_bce_s17.json` is byte-identical to the previously prepared FF2 configuration (SHA256 `db0c886527635295f0394dddbca9da1a8a3c40807609ab78f324b75b382fbee8`), originally locked before the FF2 transfer results. It is not the FF++50+50 configuration.

- Xception CES-FL v0.2.1; pretrained ImageNet initialization already cached locally.
- Seed17; ten epochs; batch16;224pixels;4data workers; balanced label×ids8 sampling with replacement.
- Encoder trainable, BN updates; encoder LR0.0001; heads/gates/fusion LR0.001; AdamW, weight decay0.0001.
- BCE only on original TRAIN images. No counterfactual loss or spatial-map consistency.
- TRAIN218,266 (16,777real/201,489fake); VAL31,107 (2,438real/28,669fake); public TEST87,564 held out from training/selection/calibration.
- Earliest highest validation AUC; selected-checkpoint threshold calibrated only on validation reals with empirical FPR target0.1. Actual achieved FPR must be reported.

True-person/hidden-donor independence is not established by recorded contributor disjointness. Demographic age/skin annotations remain automatic proxies, not newly human-reviewed ground truth. Stratified decode is not a claim that every image was decoded before training. Existing transfer scores and historical internal/public runs remain separate.

## Pinned source and preserved outputs

New detached worktree: `/home/sachin.chaudhary/ces_fl_sources/ff2_p_s17_20261102_875bce0`, commit `875bce025084871438ef799ca9086985c4375672`. Core hash `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`. No loss, architecture or research-code edits.

New run root: `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1`. Config, source, runtime and manifest hashes are in `request.json`. Operational scripts are copied and pinned under its `runtime/` before execution; active job sources are not pulled or edited.

## Explicit milestones

1. `stage_program.py --stage inputs`: verify source/config/weights/manifest hashes, actual model-reader parsing/path existence/class coverage, recorded contributor separation and stratified TRAIN/VAL image decoding. Test metadata is checked for separation only; no test inference.
2. `gpu_worker.py --stage smoke`: claim a currently idle device using the existing cooperative registry and project leases, verify UUID/CUDA visibility, run the native CUDA synthetic suite, and use a disposable model for two real TRAIN-only backward/update checks. Verify encoder/BN changes and finite head/gate/fusion gradients, plus signed-map decomposition. The smoke model never initializes actual training.
3. Only after actual input and smoke PASS receipts, explicitly launch `gpu_worker.py --stage train`. The unchanged native `train_run` writes checkpoints/history. Completion verification reloads the validation-selected checkpoint and independently replays only VAL with its frozen threshold. It never evaluates TEST.

The route is direct SSH with cooperative locking and a20% allocator cap, **not a PBS reservation**; no `ngpus` allocation is claimed. A busy GPU is skipped and no existing process is stopped. Each milestone has separate exclusive logs, process/binding/exit receipts and failure handling; there is no automatic retry or next-arm launch. Runtime is measured with a monotonic clock because node clocks can differ.

Actual milestone status, device/PID, logs and blockers are recorded in [STATUS.md](STATUS.md) after execution evidence exists. Paired arms and the full matched Ding comparison remain BLOCKED_INPUTS / unresolved review prerequisites.
