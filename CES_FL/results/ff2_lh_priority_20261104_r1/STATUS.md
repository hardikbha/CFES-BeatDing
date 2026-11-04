# FairForge-2 L_H priority run: execution status

**Status: BLOCKED before CUDA qualification and training.** Observation: 2026-11-04, GPU nodes queried at approximately 17:38 IST. This report records a failed prerequisite; it does not report L_H or L_A model results.

## Pinned source and bound inputs

- Requested source commit: `128e34cd53bed2f14bc7f6397cee030a4851d9b2`.
- Training implementation: `CES_FL/src/ces_fl_head_ablation.py`, SHA256 `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.
- L_H and matched L_A configuration SHA256: `4028cd433cf1b1efc01bef2e3da3996541560ccd448a920e89be5898dee56d20` (same configuration digest `f94b4b40bed4ed6e76b36cda97165b764e0438392d1755019065abbb2d87be90`). Seed 17, batch 32, 10 epochs, exit-block encoder tuning, frozen BatchNorm; head LR `1e-3`, encoder LR `1e-5`.
- TRAIN manifest SHA256: `4f83e7dc8042533329d7047fefd0f30cbd96f45d42bde2666dd0edc7d58f29cc`; VAL manifest SHA256: `aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8`.
- Input inventory SHA256: `9f5d4b587d5dd87fe04a7db9b353c3aabc043b892c931dfd7f2b059112276621`: 218,266 TRAIN rows, 31,107 VAL rows, 249,373 decoded images, zero TRAIN/VAL content overlap. TEST was not read.
- Completed FairForge-2 BCE50 checkpoint hash: `a62af6bc1ff02f4280fcb38e9b9a4b035ccee3427472d160880b4b9e0c31d835`, selected epoch 40. Encoder-only transfer hash: `741a8eb5b60ac8efe2ca34be985b423de12facac053acac7f05803df23248a23`; tensor equality was verified. The new downstream heads were not trained. This is FF2 BCE pretraining, not FF++ pretraining.

## Stage outcomes

| Stage | Status | Evidence |
|---|---|---|
| Encoder preparation/transfer | PASS | Transfer receipt; source and exported hashes above; encoder tensor equality true. |
| TRAIN/VAL preflight | PASS, reused | The exact TRAIN/VAL manifests and full ordered image inventory match the completed BCE/DAG/DAW preflights. The redundant full decode attempt was stopped and preserved; no split membership was changed. |
| Intervention preview binding | PASS | The new 88-case preview matches the previous preview in all 88 source image hashes, all settings, and all rendered contact-sheet hashes. Previous and new preview hashes: `f4fc30cf66df82d7d4e1aac36c14da261c218992714e51dc263f30f3a4270f2e` and `c4c6f6acdd899f87e80355adb9071a85fcee3d6b7a5cbd8076944bbfe17bbe38`. |
| Independent intervention review | BLOCKED | Existing decisions are 72 approved, 8 uncertain, 8 rejected. They were AI visual inspections, not human approval; the review does not establish that the intervention preserves or removes the intended evidence. |
| Qualification | FAIL | `commands.sh qualify` exited 1: `Substantive named operator review incomplete/rejected; revise recipe under new proposal`. No qualification receipt was created. |
| CUDA checks / 20-step cost profile | NOT RUN | The required qualification did not pass and no L_H GPU allocation was obtained. No L_H GPU-hours were charged. |
| L_H training | BLOCKED_INPUTS | No training process, checkpoint, epoch result, or L_H fairness metric exists for this request. |
| Matched L_A training | BLOCKED_INPUTS | Same preview and qualification gate; not started. |
| TEST evaluation | NOT RUN | No TEST manifest, labels, or scores were read. |

## Review failures and next valid remedy

The 16 unresolved variants come from four difficult sources, retained in the review set:

- 8 uncertain variants: two sources with substantial face occlusion (one by clothing/face covering and one by a hand) and weak visible texture; current previews cannot establish a clean intervention.
- 8 rejected variants: one source where rotation/resampling changes artificial black padding at the image boundary, and one source where headwear occludes the region being altered.

These are substantive confounds, not missing bookkeeping. The review artifact was rebound only after verifying the preview content was byte-identical; no decision was upgraded and no source was removed. A valid continuation needs an independently reviewed, source-bound recipe that handles these sources while retaining them, with matched no-change and resize/preprocessing controls. If that requires changing the intervention or preprocessing, it needs a separately versioned proposal/configuration and new checks; the current unchanged L_H must not be relabeled as qualified.

## Budget and active allocations

The shared FairForge-2 72-GPU-hour ledger showed `28.010872` GPU-hours charged, `42.0` reserved for active F3Net and CORE runs, and `1.989128` uncommitted. Both F3Net and CORE were live when checked; their allocations and outputs were left untouched. L_H has no allocation. Its configured cap is 2.0 GPU-hours, so the remaining uncommitted budget is already below that cap; actual full-run cost was not measured. No additional budget was assumed or borrowed.

## Metrics

No L_H/L_A AUC, FPR, FNR, fairness gap, worst-group metric, subgroup support, confidence interval, selected checkpoint, or threshold is available because training never started. Existing FF2 baseline results and the BCE50 pretraining checkpoint are not substitutes for these matched downstream results.

## Private audit artifacts

The cluster-only review sheets, row-level decisions, manifest inventory, logs, and checkpoint remain private under `/home/sachin.chaudhary/ces_fl_runs/ff2_lh_priority_20261104_r1/`. The qualification failure log is `qualify_attempt.log`; the aggregate private machine-readable audit is `INTERVENTION_VALIDITY_BLOCKED_PRIVATE.json`. No images, row-level labels/scores, or weights are committed here.
