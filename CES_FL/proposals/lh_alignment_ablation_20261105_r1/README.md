# Repeated TRAIN gradient audit and one conditional alignment ablation

User authorized preparation/implementation after the5November supervisor review. This package adds diagnostics, strict evidence binding, a single conditional run, tests and the GPU handoff. No cluster job is launched by publication.

## Question and existing evidence

The nuisance pilot's H1 improved resize robustness but did not establish improved intersectional fairness. H1−H0 validation jointF_FPR increased2.2719points, whereas H1−A1 decreased2.0220points. The primary bootstrap has746/1000usable draws, below900. One fixed TRAIN batch showed alignment/BCE cosine−0.600 in H1. Negative gradient cosine is not itself proof of harm; it motivates a controlled component test only if repeated measurements support the concern.

Reference results are pinned to `e993315a853aa278ec649010fa11daef1e601e44`. This is FF++ source-domain development after prior TEST exposure. The separate FairForge-2 intervention blocker is not cleared or modified.

## What is implemented

- `scripts/lh_alignment_followup.py bind` verifies native H0/H1/A1 completions, exact config/checkpoint hashes, validation-cache replay, initial P50/input binding and runtime consistency. It rechecks the actual existing H1 qualification and every source/sheet hash. Its preview inventory must exactly equal all352canonical source/operator/strength/nuisance cases derived from TRAIN. Empty, duplicate, incomplete, changed or rejected inventories are blocked. No new image or approval is created. This strengthens the follow-up wrapper without mutating the historical qualifier/source.
- `diagnose` audits H0 and H1 on20fixed TRAIN events covering all5axes×4operator/strength cases, under identity andresize90:80profiled batches total. Main batches use a fixed uniform permutation of all TRAIN rows,32per batch without replacement across the20events, and are identical across checkpoints/conditions. Original inverse-cell BCE weights are retained; no additional main-batch class/group balancing is introduced. Auxiliary-event diagnostics do not represent the frequency of all production steps. Actual main class/group counts are logged. It records weighted loss-gradient norms and BCE cosines for the whole model, encoder and head; raw/procedural responses, margin-pass fractions, original score magnitude, head norm, head-gain-normalized response and pixel perturbation magnitude. No optimizer update occurs and BN buffers are checked unchanged.
- The TRAIN threshold is calibrated once per checkpoint on the existing fixed2048TRAIN-real subset. Both diagnostic conditions are measured at that same threshold. The resize condition uses the same mild transformation on original/sham/treated views. Pixel retention and logit normalization are diagnostics, not evidence of valid real forgery or disentangled features.
- `cuda-check` and `train` require a bound complete diagnostic satisfying the prospective engineering trigger below. Training uses native **arm C**, while H1 used native **arm D**. The scientific model file remains SHA `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`. `C1.json` is byte-identical to H1's config. Its stored lambda_alignment remains0.1but is unused in armC; the effective alignment contribution is exactly zero. Classification, hierarchical fairness, procedural sensitivity, sham control and image processing stay identical.
- `report` verifies three complete epochs, native selection/cache replay, matched source/augmentation exposure and zero alignment loss. It compares C1 against existing H1 and A1 with the same1000draw paired connected-contributor bootstrap and900usable floor. All demographic results and detection regressions remain in the aggregate JSON. Fixed0.5 is a separate point diagnostic. No new TEST inference is exposed by this helper.

## Prospective diagnostic trigger

Both H1 conditions must have at least16valid batches. In at least one prespecified condition, at least60%of batches must have negative alignment/BCE cosine and median weighted alignment-to-BCE gradient norm ratio must be at least0.25. Both conditions and H0 are always reported. These are engineering triage thresholds chosen before the new diagnostic, not significance thresholds or proof that alignment is harmful. If unsupported, stop without training C1; do not try alternate batches or thresholds.

## Single candidate and decision

C1 uses the same P50 encoder and fresh seed17head, batch32,3epochs, original headLR1e-3/encoderLR1e-5, layer4fine-tuning, frozenBN, matched nuisance bank and original losses except response alignment. Never initialize from the trained H1 checkpoint. No extra seed, pretraining, lambda search, augmenter search or full-encoder change.

Select earliest maximum clean-validation AUC, then retain its validation-real10%FPR threshold. The primary outcome remains jointF_FPR. A numerically favorable screen requires lower jointF_FPR than H1 with AUC no lower and FNR no higher; these are point guardrails, not statistical noninferiority. All secondary axes and worst groups are reported even if they regress. If the primary interval is unsupported, the result remains inconclusive even if the point screen passes. The existing data support limitation cannot be fixed by more epochs or dropping groups.

This isolates alignment under the CURRENT augmented/original-LR setting. The earlier NO-ALIGN run used lower learning rates and no new augmentation; it cannot answer this particular interaction. A favorable C1 would still require independent confirmation and validated forensic evidence before a final-method claim.

## Resource and readiness limits

One authorized GPU, at most1aggregate GPU-hour for this follow-up including diagnostics, CUDA checks, idle lease time, training and replay, within the verified remaining original-study ledger. The native C1training cap remains0.75hours; the external allocation/ledger controller must enforce the1hourcohort cap and account for loading/idle time too. Profile before launch and reserve the complete remaining cost. Do not transfer FF2budget or auto-extend. Release the GPU before CPU bootstrap/reporting. Existing runs remain unchanged.

The helper links an operator-provided allocation/UUID receipt and checks CUDA visibility. File existence is not proof of ownership: the operator must establish a valid scheduler allocation or previously authorized lease and actual process-to-UUID mapping before launching Python. Use the same original H1 PyTorch/CUDA runtime, not a new environment.

See [CHECKS.md](CHECKS.md) for actual local tests and [GPU_OPERATOR_PROMPT.md](GPU_OPERATOR_PROMPT.md) for explicit stages. Private checkpoints/caches/88×4review cases and a live GPU are unavailable locally; production binding, repeated diagnostics and C1training have not run here.
