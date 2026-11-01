# Qwen FF++ pilot: useful feasibility result, do not scale the current recipe

Observed UTC: 2026-11-01T09:24:53.570761+00:00. **Decision: REVISE / DO NOT SCALE.** Software generation passed; semantic counterfactual validity and fairness benefit are unverified.

## What completed

CUDA smoke passed all seven checks on the unchanged source. The 76,139-original TRAIN byte inventory and one real-data 16-image backward dry-run passed; the latter made zero optimizer updates. The real Qwen pilot then completed one original per class, seven independent diffusion conditions each, plus geometric controls. No test data was used.

| Item | Measured result |
|---|---:|
| Original sources | 2: one real, one Face2Face fake |
| Completed diffusion outputs | 14 /14 |
| Demographic edits /editor no-change controls | 12 /2 |
| Geometric controls | 2 |
| Missing outputs | 0 |
| Human-approved /formal rejected /unreviewed | 0 /0 /14 |
| Mean generation seconds per output | 69.52 |
| Complete generation call, including load/import overhead | 1018.9s /16.98min |
| Peak Torch allocated /reserved memory | 39.15 /39.76GiB |
| Pilot item logical storage | 2,965,825 bytes |

Both originals were rehashed after the audit and still match their recorded bytes. All 18 original/control/candidate images decoded at matching source dimensions. Three recorded video-subject proxies occur across the two sources; actual person independence is not established. Source demographic metadata covers only two gender×race/ids8 cells (real Female-Asian/ids8=7; fake Male-Others/ids8=4). This is not broad demographic or family coverage.

## Is scaling practical?

The unchanged full plan requires 532,973 diffusion outputs. Extrapolating the measured two-source mean gives **10,292 GPU-hours, or 429 days on one continuously available GPU**. Eight hypothetical dedicated equally fast GPUs would still require about 53.6 days. Eight dedicated GPUs were not available or reserved in this audit. This estimate excludes retries, rejects, human review, scheduling and training; shared-GPU timing and only two sources limit precision.

Estimated base canonical bank storage is approximately 113 GB under the same per-source size, before filesystem overhead, regenerated versions, reviews, checkpoints and logs. High-resolution native outputs are not retained by the source. Storage is less problematic than generation and validity review. The protocol also requires at least 304,556 source-level approvals (one each of age, gender presentation, skin and no-change per original); those decisions do not exist.

The full bank is not justified by this pilot under the present resource and validity conditions. No 50-epoch pretraining, full generation, B/N/C binding or downstream training was launched. Coverage, resolution, steps, precision and losses were not silently reduced or changed.

## Quality observations from actual contact sheets

The no-change output visibly reconstructs/sharpens faces relative to the original and geometric control. On the fake source, the younger edit adds a substantial beard and changes facial contour. On the real source, the darker edit alters mouth/lip appearance as well as tone; some other edits noticeably change texture and sharpness. Older outputs show visible age-related changes, but preservation of identity and manipulation evidence is not established. These are qualitative flags, not completed human approvals or formal rejection labels.

The original authenticity label cannot automatically be transferred to a generatively edited image. Realism, visible target attainment, successful decoding or a review flag alone cannot demonstrate forensic preservation. All 14 review rows remain approved=0 with blank reviewer/evidence; no old approval from another sample bank was reused.

## Frozen in-domain reference diagnostic

A prior FF++ BCE R0 detector was fixed before scoring. It is not the new 50+50 model. The CPU diagnostic evaluated exactly 18 images, verified unchanged model/checkpoint/threshold state and no gradients, and did not use test data or choose pairs. Its sigmoid outputs are scores, not calibrated probabilities. Values below are continuous fake-class logits.

| Condition | Fake-source logit | Real-source logit |
|---|---:|---:|
| original | 39.220 | -41.348 |
| nochange | 20.266 | 19.020 |
| resize_geometry | 39.686 | -40.991 |
| younger | -9.939 | 10.310 |
| older | -16.552 | -21.391 |
| gender_feminine | 41.959 | 31.300 |
| gender_masculine | -0.973 | 12.227 |
| skin_lighter | 30.725 | 52.552 |
| skin_darker | 43.666 | -7.382 |

The real original goes from a strongly real-favoring score to a strongly fake-favoring no-change score. Several fake-source edits switch the other way. These findings show large intervention/editor sensitivity in this detector. They do not prove that a candidate is invalid: detector bias and distribution shift are alternative explanations. Do not approve/reject/select pairs using these scores. Independent forensic assessment and the no-change arm are needed to interpret demographic invariance. No population fairness metric, meaningful superiority test or training-seed conclusion can be drawn from two originals.

## Failures and recovery, preserved

1. PBS 44948 was cancelled while queued after the user explicitly requested SSH; it never executed.
2. The first SSH wrapper compared prefixed NVIDIA UUIDs with bare PyTorch UUIDs. Its failure is retained; strict normalization passed three guard regressions.
3. The aiface generation attempt failed at standalone source line 641: diffusers 0.35.1 lacks QwenImageEditPlusPipeline. It produced zero outputs.
4. The previously successful Qwen environment (llada-image: Torch 2.8.0+cu128, diffusers 0.39.0, transformers 4.57.6) was reused. Missing CPU-import packages were placed in a hash-locked per-run overlay; neither existing environment was modified. Training remains aiface/Torch 2.7.0+cu126.
5. The recovered generator initially hit physical GPU OOM when an existing reserved task loaded 40.3 GiB. Zero outputs were produced. The selector had missed the other runner’s reserved capacity.
6. The final worker respected the existing shared dispatcher registry. Three real contract checks using temporary registries verified that the legacy CES-FL reservation cannot be joined by the pool. Generation completed on gpu01 physical GPU 5 with a 55% Torch allocator cap. Existing processes continued; no unrelated job/configuration/queue was changed. The advisory reservation and project lease were released; the pool subsequently reused the card. No PBS allocation or cluster-exclusive ownership is claimed.

These setup issues added avoidable overhead. They are not evidence that the scientific method works. The useful outcome is identifying cost and editor confounds before launching the much larger study.

## Exact source and outputs

- Immutable source: `fc4e63b8811c5813afa0d298a8172ba9555b7e40`, `/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b`.
- Scientific source SHA256: `3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab`.
- Configuration SHA256: `56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0`.
- Full original byte-inventory SHA256: `5f8792a7017346b8205fa4a9b67257d31dc50113f2612a02b2f08506004d5810`.
- Cached Qwen snapshot revision: `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; 35 files / 57,720,463,453 bytes; content-inventory SHA256 `7ad7899d5af6a6f4f4550d10024ebdb07da83ba2b7b4c1775ef82066f8f790d9`.
- Reference detector checkpoint SHA256: `fbd276362ef31605b63a6dc3a14f52d57626ed48183a27bd8004ab6400d57b3a`.
- Fake contact sheet: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/pilot_inspection/source0_fake_contact.png`.
- Real contact sheet: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/pilot_inspection/source1_real_contact.png`.
- Pending human review: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/pilot_review_pending.csv`.
- Private per-image results: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/pilot_inspection/per_image_pilot_private.csv`.
- Actual commands/logs/receipts: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/checks/`, `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/logs/`; classifier preflight receipts remain in `/home/sachin.chaudhary/ces_fl_runs/cf5050_ssh_recovery_20261101T073243Z`.
- Report: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/REPORT_QWEN_PILOT.md`.

## Next action and separate FairForge-2 status

Review the two contact sheets and complete the 14-row review artifact with explicit uncertainty and evidence; do not mark uncertain forensic preservation as passed. Any bounded alternative to the full bank, faster editor/precision/step policy or revised label-preservation assumption needs a new versioned proposal. No replacement protocol is implemented here.

**FairForge-2 remains DO NOT RUN.** Its exact comparison configurations, tested Ding variants/common score adapter, valid paired controls, paired/per-group uncertainty and group/label mapping blockers remain unresolved. No FF++ configuration is substituted, and no FF2 experiment was launched. The previous CESel-MA gate remains FAIL.
