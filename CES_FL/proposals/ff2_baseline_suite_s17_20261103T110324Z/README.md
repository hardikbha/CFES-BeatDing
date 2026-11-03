# FairForge-2 external fairness baselines — seed17

Request: prepare and publish the remaining FairForge-2 baseline work for the GPU operator. The user explicitly authorized **up to72 additional GPU-hours**, using **at most3 concurrently allocated GPUs**. This is a new ledger, not an increase to every historical experiment. Publication does not launch anything.

## Readiness

| Method | Verified source / implementation | Current state |
|---|---|---|
| Native Xception BCE | Same architecture and recipe as the two new fairness adapters | CPU checked; real FF2/CUDA preflight pending |
| DAG-FDD | Pinned official Xception and exact objective/gradient parity | CPU checked; real FF2/CUDA preflight pending |
| DAW-FDD | Pinned official Xception and exact objective/gradient parity | CPU checked; real FF2/CUDA preflight pending |
| PG-FDD | Official source pinned and fetched | FF2 adapter, manipulation-label requirements and SAM checks pending |
| FairAdapter | Official source pinned and fetched | Paper-specific entrypoint/data/CLIP weights and adapter checks pending |
| RSEF-FDD | Paper identified; no verified official implementation found | BLOCKED_SOURCE; redundant-sample requirements unresolved |
| Ding FF2 | Existing independent run | Preserve; inspect its own branch/state, never restart here |

This package does **not** claim that all methods are runnable, that all will fit72h, or that these are new results. Source-only retrieval and a method's presence in a paper are not an FF2 implementation. See [remaining-method work orders](REMAINING_METHODS.md) and the [operator prompt](GPU_OPERATOR_PROMPT.md).

## Hypothesis and matched control

Under one fixed native Xception recipe, compare ordinary BCE with sample-tail robust risk (DAG-FDD) and nested group/sample-tail robust risk (DAW-FDD). The objective is the only difference among these three configs. This determines whether those external fairness objectives improve FF2 validation discrimination/disparity relative to their own native BCE control. It does not test a CES-FL contribution.

The source uses official `mode=original` Xception2048 features, final ReLU, global average pooling and one linear logit. It removes the upstream512-channel adapter that is constructed but **never executed** in this mode. Total/trainable parameters: **20,809,001**. Full encoder and BatchNorm buffers update. DAG CVaR alpha=0.5; DAW inner alpha=0.9 and outer alpha=0.5. Loss values and logit gradients were compared directly with the pinned official implementations.

The existing archived DAW code had a commented-out pretrained load despite a success message. This adapter loads and hashes the actual ImageNet weights and rejects missing/unexpected encoder keys. It does not execute that archived detector. The imported `CLIPFAIR/cfes/clipfair_losses.py` provides only the official-equivalent risk functions; no CLIP network or additional CLIP losses are used.

## Exact common recipe

- [BCE](configs/bce.json), [DAG](configs/dag.json), [DAW](configs/daw.json): seed17,50 full-network epochs, batch128, SGD0.001, momentum0.9, weight decay0.005, constant LR, float32, TF32 off, no AMP or extra augmentation. Same loaded encoder and identically seeded head.
- Every TRAIN original appears once per epoch in a deterministic shuffled order, including the final partial batch. No balanced resampling, group oversampling, dropped remainder or invented labels.
- TRAIN direct bilinear resize224; VAL shortest-side224 then center crop; RGB normalization `(x-0.5)/0.5`.
- Exact public TRAIN218,266 and VAL31,107 paths/hashes are pinned. Native TRAIN `intersec_label` must bijectively match canonical `gender_race`. All8 groups and both classes must exist. No TEST path is configured.
- Input preflight hashes/decodes all TRAIN/VAL files, checks byte overlap, contributor/path separation and native membership. The dataset checks the recorded image hash on each load. Cluster files are **unverified locally** until this preflight passes.
- Select the earliest strict maximum VAL AUC. Save optimizer/BN/cursor/history, independently reload the selected checkpoint and require exact VAL metric replay. Mid-epoch resume is bitwise tested on synthetic CPU data. A failed run is not a new seed attempt.
- One global threshold is calibrated from the selected checkpoint's VAL real scores at empirical FPR budget0.1. Also report fixed logit0 metrics. Never calibrate per group or on TEST.

This is a **common-recipe adaptation**, not a claim to reproduce every original paper training detail. For example, the pinned AI-Face trainers use a different original epoch/LR recipe. The current Ding FF2 implementation has a512-channel adapter/two-logit head, different real/fake sampling and a50+50 classification/GDA schedule. Do not label these unequal update budgets and architectures as fully matched. Preserve native FairAdapter/CLIP differences in its eventual table row. Existing CES-FL FF2 runs retain their own recipes and are not silently equivalent controls.

## Metrics and uncertainty

The reporter supplies overall AUC/FPR/FNR, subgroup counts and AUC/FPR/FNR, worst-group AUC, FPR/FNR gaps, `F_FPR`, `F_DP` and `es_AUC` for gender, race, gender×race, age, skin and ids8 where supported. Units are fractions in JSON and percentage points in Markdown. Unknown labels remain visible; an axis with insufficient support remains unavailable. Aggregate eligibility requires at least20 real and20 fake observations per group; disclose if this differs from another paper's eligibility rule.

`F_FPR = sum_g |FPR_g − FPR_overall|`; `F_DP = max_g P(pred=fake|g) − min_g P(pred=fake|g)`; `es_AUC = AUC_overall / (1 + sum_g |AUC_g − AUC_overall|)`. Gender×race groups are joint demographic cells, not products of marginal FPRs.

After selected-checkpoint replay, run [report_validation.py](report_validation.py) on CPU for1000 connected-recorded-contributor bootstrap draws and paired differences versus BCE, with frozen thresholds and eligibility. Include invalid-replicate counts and component counts. These intervals do not establish biological-person independence, cover model-selection optimism or measure training-seed variation. Report unavailable/poorly supported intervals honestly. No fairness-based checkpoint substitution, cherry-picked per-cell models, paper TEST comparisons using VAL, or claim of beating Ding follows from this proposal.

## Resource and execution policy

Use three independent one-GPU jobs if allocated; two or one is an acceptable fallback. Do not split CVaR minibatches across independent DDP ranks. Initial reservations are at most18h per method (54 GPU-hours total), including its CUDA smoke, profiling, training, validation and idle holding. Keep the other18h uncommitted for actual need/qualified remaining methods. All new baseline GPU activity shares one [locked ledger](budget.py); CPU preparation/reporting does not consume GPU-hours. A shared filesystem must support POSIX advisory locks.

[run_one.py](run_one.py) requires a clean detached source, explicit real allocation/binding evidence, source/config-bound input and backward receipts, and a fresh synthetic CUDA integration check. It profiles20 actual TRAIN steps, adds40% to the estimated remaining training-only cost, and defers if the projection exceeds its session reservation. The margin is an estimate, not a runtime promise. Its timeout stops **only its own child**, retaining checkpoints where possible. [one_gpu.pbs](one_gpu.pbs) requests `ngpus=1` and18h. The site/operator must establish actual device binding; low utilization is not ownership.

The ledger accounts actual held time × GPUs, checks both concurrency and cumulative reservations, and preserves failed sessions. SIGKILL/node loss leaves an ACTIVE claim for explicit reconciliation with actual scheduler start/end evidence. Never erase usage or count only optimizer time. Enforce scheduler wall limits as well; the ledger is accounting, not a GPU allocator. Further sessions need remaining budget and a distinct session ID. `--budget-hours` is a cumulative execution allowance for resuming the same fixed50-epoch run; it does not change scientific settings or authorize extra budget.

Before each session, inspect existing outputs/active processes. DONE means verify/publish, not rerun. Resume the same method/config/checkpoint explicitly. Do not pull into active source or reuse an existing output as a new run. Leave existing Ding, H, plain-head and other jobs untouched. This task launches no generation, new Ding-on-FF++ experiment, TEST inference, seed sweep or hyperparameter search.

## Evidence and dependencies

Dependencies: Python≥3.9, torch, numpy, Pillow, scipy, scikit-learn, Git; existing local ImageNet Xception weights. No package installation or model download occurs automatically. Upstream source files are explicitly fetched by commit/blob hash with [fetch_sources.py](fetch_sources.py), without executing upstream programs. The final fetch uses raw pinned files: an earlier filtered Git attempt still transferred excluded binary history, so Git cloning was replaced. The original fetch log and failed synthetic checkpoint write are retained as historical diagnostics.

CPU synthetic integration PASS: exact resumed model state, native DAW training, selected-checkpoint replay, original exposure and unchanged selected calibration. Seven focused native/loss tests and six budget-accounting tests pass. Production-size224 forward/backward and20,809,001 parameters are tested with synthetic data. These are software checks, not fairness outcomes. Fresh CUDA, real-input, timing and allocation checks remain mandatory. Source fingerprints and actual logs are in [evidence](evidence/). Do not infer current GPU liveness from GitHub.
