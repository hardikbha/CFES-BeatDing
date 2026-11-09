# Source-tail training: bounded frozen-encoder experiment

[GPU operator prompt](GPU_OPERATOR_PROMPT.md) · [configuration](config.json) · [updated diagram](figures/source_tail_diagram.png) · [software checks](evidence/checks.log).

This implements the proposed source-tail variant, not the full original L_H objective. It retains the complete verified FF2 L_A baseline as teacher/initial student, freezes the encoder and adapts only the existing2,049-parameter linear head. There are no demographic edits, procedural probe losses, source labels at inference, or JPEG augmentation.

The original L_H and new source-tail branch are alternatives in the diagram. The generic original framework depicts Xception/ResNet-50; this executable new runner binds the specific Xception L_A checkpoint. The diagram was edited with the built-in image-generation tool from the supplied L_H image; it does not certify empirical improvements. The original image is preserved.

## Objective and matched runs

Both arms use fixed original inverse class×ids8 BCE weights and the frozen-teacher fake-score guard. Control has lambda_source0; treatment0.1. Both use lambda_fake1,seed17,three epochs,main batch32,AdamW headLR1e-4/weight_decay1e-4,clip5 and16auxiliary reals from EACH of two sources every update. The auxiliary batches are deterministic and identical between arms. These are unoptimized fixed engineering settings, not a search against TEST.

References use the COMPLETE teacher TRAIN-real logits only:85/90/95%quantiles,teacher-real IQR scale,pooled soft exceedance targets and temperature1. All are frozen. Student starts with the teacher's existing head, not a fresh head. Scores on fake originals are discouraged from falling below their corresponding frozen-teacher scores. A soft penalty does not guarantee utility or hard-error parity, especially after validation recalibration or distribution shift.

Source IDs are canonical TRAIN-real acquisition labels FFHQ/FairFace; a complete source sidecar and provenance hashes are required. This does not infer a fake image's pixel origin, fill missing source×demographic cells, or establish source causality. Auxiliary source×joint-group exposure is recorded.

## Stages

`run.py prepare` verifies the pinned teacher checkpoint/completion/source and TRAIN/SELECT hashes, reuses the checkpoint-bound image inventory, verifies image bytes on each actual read, checks split/content overlap, and caches2048-dimensional pooled encoder features plus teacher logits. It checks feature/head versus full native forward and requires EXACT native teacher validation-metric replay. It reads TRAIN/SELECT only. A failed or partial cache has no CACHE_VERIFIED receipt and cannot train. No partial cache is overwritten.

`run.py train` checks all cache hashes, initializes the exact teacher head, and trains only that head. Both main originals and auxiliary real examples use the SAME cached encoder, so the encoder cannot drift. Mid-epoch checkpoints preserve optimizer, data cursor, auxiliary schedule, accumulated losses and exposure. The loss coefficient is the only control/treatment difference. Main originals appear once per epoch. Encoder inference is shared once across arms, substantially reducing repeated GPU work.

Each arm retains the minimum validation gender×race F_FPR among epochs satisfying AUC≥99.5%andFNR≤0.5651%, then higherAUC/earlier-epoch ties. It fits one global validation-real10%threshold per epoch; no group-specific threshold. Infeasible runs finish with COMPLETED_NO_FEASIBLE_CHECKPOINT, not an invented winner. SELECT is exposed development, and the teacher was previously selected there; this cannot provide independent confirmation. No TEST/CAL command exists.

`run.py export` reconstructs the full teacher architecture with the selected new head and requires full native validation replay to match the cached-head result before writing a deployable student. Numerical/preprocessing mismatch is an inspection blocker, not permission to relax equality or retune a threshold. The teacher and source labels are not needed at inference once the full student is exported.

## Checks and limits

Synthetic CPU checks cover objective gradient direction, detached teacher, fake guard, actual coefficient-dependent head updates, complete three-epoch control/treatment cycles, exact interrupted/resumed parameters/history, selected-score replay, original exposure, fixture exclusion, cache tampering and full-width224px Xception feature/forward identity. See CHECKS.md/evidence. These do not certify private production inputs, a real encoder-cache extraction, GPU timing, or scientific performance; those checks must execute on the node.

The script uses the exact frozen teacher-core source SHA8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733 as a local reference copy. Historical source files/results remain unchanged. The baseline checkpoint is SHA cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d. Wrong teachers, data, source mappings, incompatible heads and stale/tampered caches fail explicitly.

No new GPU grant: cap this first cohort at2aggregate GPU-hours within a verified remaining existing FF2 allowance, including cache extraction, timing, both head runs, export/replay and allocated idle time. Reconcile missing historical TEST/sanity charges before asserting that hours are available. CPU head training is supported; use consistent device/precision and retain the strict full-export replay gate. At most one GPU at a time. If cache extraction/profiling cannot fit, report the measured blocker; no automatic extensions or partial-result claims.

Publish both arms and the unchanged teacher, including source-specific and demographic errors, all regressions and infeasible outcomes. Use the previously audited full contributor/master dependency structure and declared interval-support rule for any uncertainty analysis; do not split components or infer significance from three epochs. No source-standardized race/joint claim is made where cells are absent. No claim of novelty, causal forensic sensitivity, beating Ding or60/80wins is established by this software.
