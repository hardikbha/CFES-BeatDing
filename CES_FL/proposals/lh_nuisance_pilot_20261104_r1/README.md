# Bounded nuisance and procedural-sensitivity pilot

User request,4November2026: investigate why forensic sensitivity/fairness transfers poorly, run short checks, and publish runnable operator instructions. This is a prospective TRAIN/validation development pilot. It does not change completed results, use Celeb-DF TEST for tuning, or claim that augmentation disentangles features.

## Mechanism and hypothesis

Original L_H learns from classification, group-error control, synthetic-probe sensitivity, sham consistency and response alignment. These constrain output behavior. It contains no explicit factorization into demographic and forensic feature spaces. Equal synthetic responses are not proof of equal original-image error rates or valid forgery evidence.

Augmentation can make a nuisance less predictive when the SAME distribution of mild image processing is seen on both real and fake originals. For H, apply the SAME processing to original/sham/treated views so the response cannot be explained by deliberately different JPEG/resize/blur settings. This is a hypothesis about transferable behavior, not a guaranteed improvement. A transformation may erase a legitimate cue; measure and review that possibility rather than enforcing invariance blindly.

The frozen nuisance bank is identity; JPEGquality90/subsampling0; bilinear downsize to90%and back; Pillow Gaussian blur radius0.35pixels. No combined distortions or strength search. Bank case is deterministic from seed/source/epoch, independent of class/group/prediction. Main originals of both classes are transformed; A1/H1 see identical original images and cases. Every auxiliary event uses one shared case across every class/group and paired view. Processing happens AFTER the existing procedural intervention and BEFORE ResNet normalization. Validation and TRAIN-threshold calibration stay unaugmented. No new loss term or synthetic-image BCE label is added.

## Four matched arms

| Arm | Objective | TRAIN processing |
|---|---|---|
| A0 | Original weighted BCE | Original pipeline |
| H0 | Original full L_H | Original pipeline |
| A1 | Same BCE | Matched nuisance bank |
| H1 | Same full L_H | Matched nuisance bank |

All run3epochs from the same completed ResNet P50 encoder, with identical fresh seed17linear heads, batch32,layer4training,frozenBN,headLR1e-3,encoderLR1e-5,AdamWdecay1e-4,clip5,bf16TRAIN/fp32VAL. Source remains ResNet50+GAP+Linear2048→1,23,510,081total/14,966,785trainable parameters. Every TRAIN original once/epoch; H retains12,926real procedural pairs/epoch. H's five demographic axes, inverse label×ids8 original weights, auxiliary group sampling, loss coefficients and zero alignment warmup remain unchanged. Input/P50 hashes and exact configs are in `configs/`.

New controls are needed because historical ResNet A selected epoch5 of10; a three-epoch matched score-cache comparison cannot substitute that checkpoint. No new P50 or second seed is needed. Select each new checkpoint by earliest maximum clean FF++validation AUC after exactly3epochs. Keep its global validation-real10%FPR threshold and report fixed0.5 separately. Do not choose a fairness-best epoch.

Contrasts: A1−A0 measures augmentation under BCE; H1−H0 measures augmentation under H; H1−A1 tests H beyond the augmented BCE control. The descriptive interaction `(H1−A1)−(H0−A0)` tests whether the effect differs by objective package. It is not an isolated sensitivity-term ablation or causal feature-disentanglement proof. Report AUC/FNR and every demographic axis, group support, worst-group AUC and regressions. Retain900/1000usable connected-contributor draws; sparse intervals remain unsupported. No success threshold or statistical significance is invented from post-run outcomes.

## Short experiments and readiness

`scripts/diagnose_lh_nuisance.py --mode synthetic` was run locally. It shows three possible shortcuts:

1. With an unchanged feature ordering, multiplying logits by10changes the sensitivity loss from0.400166to0 for a fixed relative score gap. Sensitivity is not invariant to logit scale.
2. Two all-real groups can have identical procedural responses and nearly zero response alignment while their thresholded positive rates are0and1. The separate original-risk terms must address that difference.
3. A hand-written texture-only score satisfies the synthetic resampling sensitivity margin on four random textures, while reacting strongly to benign resizing/blur. This does not show that an actual L_H checkpoint uses that shortcut.

Actual numerical outputs are in [counterexamples_final/SYNTHETIC_COUNTEREXAMPLES.json](evidence/counterexamples_final/SYNTHETIC_COUNTEREXAMPLES.json). These are synthetic mathematical/mechanism experiments, not dataset performance results.

The checkpoint diagnostic is ready for original or pilot A/H checkpoints: it selects8TRAINexamples per joint-group/class cell, prefers distinct recorded contributors, compares all four processing cases on identical procedural pairs, measures original-logit drift, raw/transformed sham differences, response preservation and pixel-difference attenuation, and measures weighted gradient norms/cosines on one fixed clean TRAIN batch. It makes zero optimizer updates, preserves BN buffers, hashes the checkpoint/manifest/source and keeps row-level output PRIVATE. It cannot establish real forgery validity. Trained-checkpoint diagnostics and actual CUDA/input checks remain cluster work.

Focused CPU tests PASS. Full-width224px ResNet backward/BN checks and a frozen-encoder synthetic A/H lifecycle with augmentation, checkpoint reload, bitwise resume, calibration and label-flip isolation PASS. The initial full-lifecycle attempt ran out of local disk while saving synthetic checkpoints; its failure record is preserved. The retry reduces only the software fixture's optimizer scope; production configs still fine-tune layer4. See [CHECKS.md](CHECKS.md). No real-data training or new benchmark TEST inference was launched here.

## Execution and budget

Use [GPU_OPERATOR_PROMPT.md](GPU_OPERATOR_PROMPT.md) and explicit [commands.sh](commands.sh) stages. New model source `src/ces_fl_nuisance_resnet50.py` preserves the original scientific model/loss functions and adds TRAIN processing plus exposure logs. All historical sources remain unchanged. The new checkpoint format/source binding prevents silently overwriting or importing old downstream heads.

Maximum2concurrent authorized GPUs and3aggregate GPU-hours INCLUDING diagnostics, software checks, idle allocated time, training and replay, within the verified remaining study allowance. Native per-run boundary allowance is0.75hours; aggregate ledger and an external stage deadline must also be enforced. Shared GPU throughput varies, so profile and reserve before launching. CPU preflight/analysis should not hold GPUs. No automatic extensions or extra candidates; partial runs stay partial. No FF2budget transfer, new Ding run, new diffusion bank, benchmark TEST evaluation or target-domain threshold sweep belongs to this pilot.

H0/H1 require source/config-bound visual qualification. Matched processing changes H1 preview bytes; do not reuse old approvals as if unchanged. Review all emitted cases with a named reviewer, record AI versus human scope accurately, and reject or pause a dependent arm if nuisance processing erases the intended cue or creates unrelated artifacts. No source dropping, fabricated approvals or biological-identity claims. A0/A1 can proceed independently, but an incomplete factorial cohort does not establish the full interaction.
