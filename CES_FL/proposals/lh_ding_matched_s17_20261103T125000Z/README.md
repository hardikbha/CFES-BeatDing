# L_H versus Ding: prospective matched training-recipe study

**Priority update:** this recipe is prepared and CPU-tested, but deferred while the current L_H result is finalized and diagnosed. See [LH_NEXT_STEPS_REVIEW.md](LH_NEXT_STEPS_REVIEW.md). Do not interpret these commands as a new automatic launch.

Prepared 3 November 2026. **No new training or TEST inference launched.** This package trains CES-FL only. It does not remove the user's earlier restriction against new Ding runs on FF++; the clarification about one corrected Ding baseline remains unanswered. A matched update budget is achievable here; a fully controlled comparison to Ding's intended fairness method is not yet established.

The hypothesis is that L_H's hierarchical error control and procedural response objectives improve demographic disparity versus the same plain linear detector with BCE under a common training recipe. This is a new prospective experiment, not an extension/relabeling of the ten-epoch L_H results. There is no guaranteed improvement and no hyperparameter search.

## Exact experiment

| Setting | New P | New L_A-recipe | New L_H-recipe |
|---|---|---|---|
| Backbone/head | Xception2048 + GAP + Linear2048→1 | Same | Same |
| Initialization | Pinned local ImageNet weights | Completed new P, all weights including classifier | Exactly the same P checkpoint |
| Epochs | 50 | 50 | 50 |
| Main batch unit | 128 images =64real+64fake | 64 images =32real+32fake | 64 images =32real+32fake |
| Sampling | Every fake once; uniform real sampling with replacement | Same | Same indices as L_A for the same epoch |
| Optimizer | SGD, lr.001, momentum.9, wd.005 | Same | Same |
| Precision | Float32; TF32 off; no clipping/AMP | Same | Same |
| Trainable | All20,809,001 parameters | Exit layers+linear:6,790,433 | Same |
| BN running statistics | Updated | Frozen | Frozen |
| Classification | Uniform BCE | Uniform BCE | Uniform BCE |
| Additional objectives | None | None | Unchanged L_H loss functions/coefficient values |
| Selection | Final epoch50 | Final epoch50 | Final epoch50 |

The main stream contains63,213fake rows and12,926real rows under the pinned manifest. Expected updates:988/epoch×50=49,400 for P;1,976/epoch×50=98,800 for each downstream arm. Each epoch draws126,426main images. Record actual counts and fail on mismatched exposure. A and H share P, so their comparison costs150 stage-epochs total, while each method path is50+50. Extra H forwards and Ding's SFD/group forwards mean equal optimizer steps are not equal GPU cost.

Training directly resizes to224×224 and normalizes by.5. Validation uses short-side224 plus center crop224, matching the archived Ding evaluation geometry. Selection remains final epoch even when earlier validation AUC is higher. Report fixed probability.5 as the main native-protocol view, and a separately labeled global validation-real empirical10%FPR view; retain all group supports, FPR/FNR/AUC, summed F_FPR, FDP, es-AUC, gaps and worst-group AUC. The calibrated threshold is never recalculated on TEST.

## What changes from current L_H

Current L_H: gated-P50 initialization, fresh linear head,10epochs,32images/update, each original once, inverse class×ids8 BCE weights, AdamW(exit1e−5/head1e−3), bf16, gradient clipping, best validation AUC, direct-resize validation.

New recipe: new plain-linear P50, retain its classifier as well as encoder,50adaptation epochs,64main images/update, fake-once/real-replacement sampling, uniform BCE, SGD.001 for every trainable parameter, float32, no clipping, final-epoch selection and archived evaluation geometry. These changes are applied to BOTH L_A and L_H. Comparing new H to the old10epoch H cannot isolate which recipe change mattered.

The scientific functions are imported from hash-pinned `src/ces_fl_head_ablation.py` without editing/monkeypatching it. H still uses real-only local-warp/local-resample probes with matched processing shams, same auxiliary group schedule, hierarchical joint-real risk control and response alignment. It does **not** generate age/gender/skin edits, remove validated real-world forgery evidence, or establish causal fairness. Age/skin metadata remain proxies where supplied as such.

## Match status and remaining blockers

- Public FF++ TRAIN/VAL manifests and six-key overlap audit stay hash-pinned. Historical records say TRAIN/VAL163 contributor overlaps; this is not identity-disjoint evidence. Fresh private byte/decode checks are required.
- Current archive code imports224px transforms; an older report says256px. The256px FLOPs example is not the training transform. See [audit](PROTOCOL_AUDIT.md); historical execution source correspondence still needs operator confirmation.
- The paper says batch64/50epochs/SGD.001; its prose does not resolve pair-versus-image units. This package targets the inspected archived50+50 implementation, rather than claiming those are unambiguously the paper's exact settings.
- Archived GDA has a detached fairness term, confirmed by saved gradient evidence. It is a **code-as-archived** baseline, not a certified realization of the paper's fairness objective. A corrected baseline would require an explicit new protocol, nonzero gradient checks and authorization. No new Ding command is included.
- Ding's head-only GDA with updating BN and code seed121 differs from CES exit tuning/frozen BN and seed17. Frozen BN is retained to make A/H loss comparisons interpretable. Architecture, trainable capacity, auxiliary exposure and seed differences must stay visible.
- Full Table1 matching needs identical locked TEST membership/annotations, duplicate handling, preprocessing, group eligibility, threshold policy and evaluator. Neither this package nor current L_H has a new TEST result. Tables2/3 cannot be inferred from this Xception validation study.
- Existing primary connected-contributor bootstrap has746/1000usable draws, below the900minimum. Do not lower that guard. Use1000paired component draws and report actual usable support; no seed-robustness claim from seed17 alone.
- No FF++ budget has been assigned to this NEW study. The72GPU-hours authorized for FairForge-2 baselines cannot be reassigned silently. Profile within an explicitly remaining allowance and record a separate total ledger before a full launch. CLI budgets are cumulative per run and checked at epoch boundaries; scheduler walltime and the operator's aggregate ledger must enforce hard caps.

## Files and readiness

`P_config.json` is an exact input-hash-bound pretraining config. `A_template.json`/`H_template.json` intentionally cannot run before new P exists; `bind-parent` verifies P's DONE/receipt/checkpoint, carries its actual path/hash, and writes new private configs without altering templates. [GPU_OPERATOR_PROMPT.md](GPU_OPERATOR_PROMPT.md) contains commands and execution conditions. [CHECKS.md](CHECKS.md) records software evidence. [request.json](request.json) records pending authorization/resource decisions without fabricating approvals.

Review/replay all three final checkpoints before reporting completion. Freeze a separate evaluation handoff before TEST. Preserve all historical results, current L_H/GPU jobs, preregistrations and public-split membership.
