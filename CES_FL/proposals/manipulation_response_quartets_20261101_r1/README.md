# Matched age/manipulation quartet preparation — BLOCKED_INPUTS

This is an additive preparation record for source commit `58773bbef9d9fe79dd313610f694ddeea12c6037`. It does not change the parent protocol and does not launch generation or training. The actual command `/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/runtime/prepare_quartet_inputs.py` completed with exit 0. It selected and hashed prospective source/donor inputs; it produced **zero manipulation images and zero approvals**.

## Actual candidate audit

The prescribed bank needs 16 different original real TRAIN sources: one older and one younger quartet for each of the eight real TRAIN `gender_race` groups. The active Qwen32 diagnostic was balanced by **ids8**, a different partition. Only 15 of its 16 real source candidates exclude recorded VAL contributors, and those cover five `gender_race` groups. It cannot be relabeled as a complete quartet bank.

Eight potential source candidates can be drawn from that existing run after individual substantive review; eight other eligible original sources were selected prospectively for supplementation. Missing source counts: Female-Black 2; Female-Others 1; Male-Asian 2; Male-Black 2; Male-Others 1. No source was dropped to relax these quotas. Completion of ongoing Qwen generation does not resolve the missing manipulation roles or reviews.

The prospective manifest contains **16 different target proxies and 16 different donor proxies**, all original real TRAIN paths, zero recorded VAL identity/source contributor overlap, one fixed donor from the same `gender_race` cell per quartet and no donor reuse. Proxy distinctness is not verified person identity. Paths and hashes are in the private manifest `/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/quartet_preparation/prospective_quartets_PRIVATE.csv`; only aggregate coverage/hash is public. All reviewer/approval fields remain empty/0. It is not a production bank.json.

## Existing operator located and actually hashed

Candidate: `/home/sachin.chaudhary/models_external/SimSwap`, Git HEAD `bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780`. Actual generator and ArcFace checkpoints and detector assets exist and were SHA256-hashed. The shared checkout has modified detector files; those modifications were preserved and their actual bytes recorded. A future run must freeze the entire relevant code bundle, including these patches, rather than claim the Git HEAD alone identifies the operator. No external source or environment was modified, downloaded or executed for generation.

The inspected historical driver `/home/sachin.chaudhary/ICLR_2027/fairforge2/src/gen_swap_arm.py` implements SimSwap-224 with replicate padding, explicit paste-back and no watermark. The old control driver `src/audit_v1/ws3_swapgen.py` includes no-generator sham controls, but its main function selects TEST/VAL identities from a different dataset. **Neither driver was run. Do not execute them as-is for this study.** Only their implementation was inspected. A new isolated FF++ TRAIN-only adapter is required, retaining current donor/seed/settings controls and recording geometry-dependent changes.

The old `pulid` environment metadata reports Torch 2.4.1+cu121, torchvision 0.19.1+cu121, NumPy 2.2.6, OpenCV-headless 5.0.0.93, InsightFace 2.0 and ONNX Runtime 1.23.2. These are metadata/import-spec observations, not a target forward-pass PASS. The first metadata query asked for `opencv-python` and failed because the installed distribution is `opencv-python-headless`; both receipts are preserved. No package was installed. Classifier training/smoke continues to use **aiface**, not this candidate generator environment.

## Proposed operation and qualification

See `operator_settings_proposed.json` for exact proposed settings, actual model hashes and required receipts. Apply the same SimSwap code/settings/donor/seed to x0 and xa from each original; log both face detections, transforms, masks and encode steps. Save matched no-generator warp/blend controls for both conditions. These controls are diagnostic and add **no training loss**. They are not counted as genuine manipulated faces. An operator may transfer identity/age appearance or add blur/boundary cues; independent review must assess that explicitly. No claim that edited bases are authentic, or that a swap preserves all demographic attributes, is made.

The minimum missing editor work is eight no-change plus eight assigned-age outputs if an explicit age-only adapter is implemented/tested. The current unmodified Qwen generator always produces seven conditions; using it would instead require 56 outputs and must be disclosed. No such supplementation is running, and the existing Qwen job/plan is untouched.

## Review workflow and blockers

For every proposed source, inspect original/donor/x0/xa/t0/ta and both sham controls. Require named qualified independent review of recognizable identity/target age, actual donor manipulation, matched processing and unintended changes, with evidence references and uncertainty. Use a second review/adjudication where feasible; do not turn detector agreement into approval or automatically fill missing flags. Preserve rejected candidates and missingness. All actual inputs must remain TRAIN-only, including donors and geometry contributors.

Only actual complete, reviewed quartet assets can be assembled into bank.json with the parent schema and then audited/bound. The parent pinned code must reject missing bytes/provenance, inappropriate contributors and unmet group/direction quotas; do not weaken it. No-change/score/response remain **BLOCKED_INPUTS**. Original-only P/BCE/fair can proceed after their own software/data/P dependency gates, without waiting for this bank. FairForge-2 and test evaluation are outside scope.
