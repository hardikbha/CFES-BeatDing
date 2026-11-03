# Operator handoff: complete L_H evidence and audit Celeb-DF

Work on `hardikbha/CFES-BeatDing`. Fetch the exact supervisor commit supplied with this prompt from `https://github.com/hardiksharma6555/CFES-BeatDing.git` into a NEW detached worktree; do not pull into active job sources. Read CES_FL/AGENTS.md, WORKFLOW.md, this proposal's README/request/diagnosis, and the earlier work order `CES_FL/proposals/lh_tables_ding10_s17_20261103T133000Z/README.md`.

This package contains ready CPU analysis and implementation specifications. The ResNet L_H and Ding-10 adapters still need implementation, exact configuration binding and smoke tests. Do not describe them as already runnable. Continue autonomously through authorized preparation, checks and training that fit the remaining original budget; report genuine dependency/budget failures without inventing evidence.

## 1. Preserve the original candidate and avoid duplicate work

Retain original Xception L_H epoch1 checkpoint SHA `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`, selected by maximum FF++validation AUC after ten epochs. Its P50 encoder SHA is `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5`. Use the original H objective/rates for backbone confirmation; not lower-LR, SHAM-FIX or NO-ALIGN. Completed lower-LR and both ablations are diagnostics, not replacement winners. Tables1–2 at results commit `916999c8e54d0eeaa86bd2b09d9d3ceaadea0bb5` already contain all eight frozen L_A/L_H evaluations and uncertainty. Do not rerun them, reselect checkpoints, change thresholds or launch the deferred cosine experiment.

## 2. Run the CPU Celeb-DF audit and verify private provenance

```bash
python CES_FL/proposals/lh_finalization_celebdf_20261104_r1/analyze_published.py --out /tmp/cesfl_celebdf_audit_operator_new
```

Use a new output directory. Then inspect the already saved private L_A/L_H Celeb-DF score caches under `/home/sachin.chaudhary/ces_fl_runs/lh_tables_s17_20261103T133000Z/inference/`, using native command receipts to locate actual files. Match checkpoint/manifest/image hashes, row order and labels. Trace the inherited male/nonmale mapping and race/joint labels to the canonical upstream annotation files, including donor/target conventions and joins. The manifest has28,458rows,5,713real/22,745fake;94.5395%of fakes are labeled Female-White. Verify whether this concentration and the real-only/absent groups are present upstream. Do not infer or alter sensitive labels from image appearance.

Report per-video/recorded-contributor support and the source of missing age/skin labels. Check existing preprocessing code/receipts for RGB ordering, normalization, crop policy and input bytes without testing competing transforms on TEST. Inspect a small, deterministic private sample of errors and matched correct examples for image-quality issues. Keep faces/paths/predictions/checkpoints private; publish only aggregate findings. Preserve the existing587/1000joint and609/1000race bootstrap limitation. No new inference, threshold search or model/augmentation tuning from Celeb-DF TEST is authorized by this audit.

## 3. Implement and check the missing Table3 ResNet-50 comparison

Create new uniquely named source/configs; preserve original Xception source hash `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177` and outputs. Use ResNet50 spatial2048features, GAP and Linear2048→1 for downstream L_A/L_H. Fine-tune layer4+linear, freeze all BN running statistics. Use documented ResNet ImageNet normalization consistently for originals, shams and treated images; use already available local weights, no automatic downloads.

Bind ResNet P50 to the actual saved native Xception P50 configuration:50original-only pretraining epochs, same dataset/exposure, optimizer, selection and BN policy, with the declared backbone/normalization substitutions. Inspect its exact pretraining head/architecture rather than guessing from a P10example. Archive that native configuration and all changes. Reuse a verified ResNet P50 only if its exact protocol matches; historical ResNet P10/B/C does not satisfy this dependency.

From the SAME verified ResNet P50 encoder, run matched ResNet L_A and L_H with fresh identical seed17heads,10downstream epochs,batch32,head_lr1e-3,encoder_lr1e-5,AdamW weight_decay1e-4,clip5,original224resize geometry,bf16TRAIN/fp32VAL,frozen BN. Keep H losses/sampling/coefficients identical to original L_H; L_A is its BCE-only control. Every original once/epoch; H covers every TRAIN-real source with the existing procedural sham/probe recipe. Keep all manifests, overlap declarations and source/config/checkpoint hashes. Do not add layers, demographic edits, new losses or seeds.

Require Xception no-regression parity, ResNet parameter counts and gradients, all-view normalization checks, frozen BN, wrong-backbone checkpoint rejection, same initialization, native resume/selection/replay, CPU smoke, actual CUDA smoke and real-input backward checks. Bind procedural review only from valid unchanged-byte evidence; never fabricate approvals. Commit the exact tested source/configs before launching from an immutable checkout.

After both downstream models complete and pass native validation replay, freeze their selected checkpoints and source-VAL10%thresholds. Extend the locked evaluator deliberately for the new checkpoint format/backbone and test its guards. Evaluate each frozen model ONCE on the four existing TEST manifests with fixed0.5 and unchanged source-validation threshold views; no target-domain calibration. Publish Table3 including A and H controls, all demographic metrics, achieved FPR/FNR, supports and connected-contributor uncertainty. Do not fill missing rows from older architectures or partial runs.

## 4. Prepare the authorized Ding ten-epoch control

Complete the prior work order as a separate controlled adaptation, not a claimed paper reproduction. Reuse verified seed17 native classification P50/final checkpoint and TRAIN-only SFD2%mask if available and provenance-compatible. If unavailable, report the missing dependency and cost before any replacement pretraining. Implement connected-gradient class-conditioned GDA; do not launch the detached archived fairness term unchanged.

Use10downstream epochs,32main images/update,each original once/epoch,2,380updates/epoch,seed17,224direct preprocessing and FF++validation-AUC selection with both reporting thresholds. Retain Ding's512-dimensional classifier/head-only optimization,SGD lr1e-3,momentum0.9,weight_decay0.005,fairness weight0.005 and verified native SFD recipe. Freeze encoder BN buffers in this declared controlled variant to avoid hidden encoder adaptation, explicitly documenting this difference from archived BN-updating GDA. Preserve native class-conditioned fairness sampling/loss; audit actual epsilon/normalization against the published/native implementation and record any approximation. Do not claim equal effective compute or optimizers with L_H.

Tests must demonstrate nonzero finite FAIRNESS gradients to classifier weights, changing the fairness coefficient changes an optimizer update, frozen encoder parameters/BN buffers, fixed mask, correct class/group sampling, checkpoint/resume and validation-only selection/calibration. Implement and smoke-test before training. The final row must read “Ding SFD+GDA, connected-gradient,10epoch controlled adaptation”; preserve separate paper-published and archived-code rows. Evaluate a locked selected model only after the new protocol is fully bound; do not optimize it against known TEST scores.

## 5. Compute, publication and stopping

Refresh actual job state and the original24GPU-hour ledger. Latest observation after both ablations is17.286613used,6.713387remaining. This is a shared remaining allowance, not a new grant. The separate FF2 baseline72hours must not be used. Use at most two legitimately allocated GPUs with verified device bindings; preserve other active jobs. Count task-held GPU walltime including checks/idle/replay.

Priority is CPU provenance audit, then the ResNet Table3 cohort, then Ding-10 if its dependencies and remaining budget permit. Profile before launching, account for the full remaining P50/A/H/evaluation cost, and publish the estimate. Do not start a cohort that cannot finish within the verified remaining allowance; report the budget blocker and exact completed preparation. Do not silently shorten50to10epochs or declare an incomplete run complete. No new hyperparameter search or automatic retry/extension.

Push source/configs, real smoke receipts, launch commit/job IDs, stage/epoch progress, aggregate results, failures and the final ledger after milestones. Keep original result directories and preregistration immutable. Return exact commits and commands, updated Tables1–3 with pending cells honestly marked, Celeb-DF findings separated into verified facts/hypotheses/unavailable evidence, and the remaining blockers. No SOTA claim based on cell counts alone; prior TEST exposure remains disclosed.
