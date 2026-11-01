# Controlled manipulation response — new FF++ investigation

This versioned proposal implements the research direction requested after the Qwen quality pilot. It does not change any completed result, current generation job, the full-coverage 50+50 protocol, or the blocked FairForge-2 comparison. Priority means the next available authorized resources; it does not authorize stopping jobs or changing scheduler priority.

## Hypothesis and limits

An editor can alter the very evidence a detector should preserve. The new hypothesis is that a detector's response to a **known added manipulation** should remain positive and similar across matched age appearances. The hypothesis is unproven. This is not a validated causal estimator and does not establish counterfactual fairness, novelty, or superiority over Ding.

The architecture remains Xception + global/patch heads + learned sigmoid gates + positive fusion. All parameters fine-tune; downstream BN running buffers freeze. The auxiliary objective is the change. No new MMD, Group-DRO, mask-consistency objective or calibrated reliability claim is introduced.

## Data required before paired training

Each quartet starts with one original **real TRAIN** image. Construct x0 with the editor's no-change instruction and xa with an age edit, independently from that original. Apply the same recorded manipulation operator T to each, producing t0=T(x0), ta=T(xa). Match operator code, settings, donor and seed; preserve receipts for any geometry-dependent adjustments. No generic blur/JPEG/resize operation counts as the forgery operator. No-change and edit generation must share model/settings/seed apart from the instruction.

**The Qwen pilot provides only candidate x0 and xa images. It does not provide t0/ta, substantive approvals or a validated manipulation pipeline. The forgery generator is NOT implemented in this package.** The cluster operator must locate an appropriate existing pipeline, record its exact reproducible configuration and validate its outputs before creating the bank. If unavailable, paired arms are blocked; do not substitute arbitrary transforms or fabricated provenance. Assets and source/donor contributors must belong to TRAIN and exclude recorded validation contributors even when the parent benchmark permits overlap. This is stricter than the existing 32-source diagnostic; some of its sources may be ineligible.

The initial age-only bank requires one older and one younger quartet from different sources for every real TRAIN gender_race group: 16 sources if all eight groups are present. This is a bounded mechanistic pilot, not population coverage. Additional donor contributors must be recorded, and within-bank dependence must be reported. `quartet_schema_example.json` is a deliberately non-runnable schema example with approvals false. No review decision is auto-generated. Software checks verify bytes and asserted provenance, not biological identity, causal validity or the truth of human review. Detector scores must not decide which candidates are accepted.

## Objective and controls

Original images alone receive their benchmark BCE labels. x0 and xa are generatively processed and **never labeled authentic**. Their derived outputs receive no assumed benchmark binary label either. The auxiliary task is sensitivity to an additional specified operation, which may or may not transfer to genuine benchmark detection.

Let m0=asinh(f(t0)-f(x0)) and ma=asinh(f(ta)-f(xa)). The response term is mean|m0-ma| + 0.5*mean(relu(tau-m0)+relu(tau-ma)), tau=0.5. Applying asinh after the logit difference cancels a common additive offset within each pair, but does not remove general editor effects. Positive-margin penalties discourage a constant-score solution; they do not guarantee useful or demographic-independent features. Operator shortcuts must be tested with unseen operators before any generalization claim.

The fairness baseline averages absolute deviations of observed real-group mean sigmoid scores from the pooled real mean at fixed logit threshold0, temperature1. It uses TRAIN only, no validation-threshold gradients. Batches with fewer than two real groups contribute zero; their counts are logged. This is a minibatch soft surrogate, not an unbiased/full-population estimate of Ding's F_FPR, nor a guarantee about the separately calibrated threshold. Group-size/coverage sensitivity is a required diagnostic.

| Arm | Objective |
|---|---|
| bce | Original weighted BCE |
| fair | BCE + 0.1 × TRAIN soft FPR disparity |
| nochange | fair + 0.1 × mean abs(asinh(original)-asinh(x0)) |
| score | fair + 0.1 × mean abs(asinh(original)-asinh(xa)) |
| response | fair + 0.1 × proposed response term |

All numbers are unoptimized engineering choices fixed in config.json. The nochange and score arms test original-to-editor consistency; the response arm deliberately tests a different intervention. Alignments use separate forwards with frozen BN buffers, not joint original/edited BN updates. Paired arms sample the same bounded bank deterministically with up to four quartets per minibatch. They reuse sources many times; usage is logged. Original data order, classification weights, P encoder and head initialization match across arms; forward FLOPs do not. No full source-pair coverage is claimed.

## Training, selection and execution

Shared P: 50 original-only epochs through the existing tested pretrainer. Each of the five downstream arms: 50 epochs. Total if all run: 300 epochs, each path50+50. A validation-selected P encoder can come from an earlier epoch; fresh heads/gates/fusion are identical. Every original appears once per epoch with fixed inverse label×ids8 cell classification weight. Fairness groups are gender_race; those partitions are deliberately distinguished. Epoch-boundary resume requires unchanged code/config/arm and consistent history/checkpoints. Failed partial writes are preserved for explicit recovery.

The new trainer imports adjacent `ces_fl_counterfactual_5050.py` and enforces SHA256 `3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab`. Fetch both files via the pinned commit; this is not a standalone one-file distribution. Dependencies are torch/numpy/Pillow/scikit-learn. No automatic installs, model downloads, generators or cluster submissions occur.

Original-only P, BCE and fair arms do not require a quartet bank. Paired arms fail closed without the reviewed bank. `bind` checks completed P provenance and writes exact P/bank hashes. `preflight` verifies inputs; `dry-run` performs one actual backward pass without optimizer updates; `train` selects by validation AUC only and independently replays the selected checkpoint before DONE. Both source hashes are stored. No test-evaluation command is provided: the existing uncertainty/evaluation limitations must be resolved in a separately locked plan. Do not load these new-format checkpoints through an old evaluator.

The development criterion in request.json is lower intersectional validation F_FPR than fair and score controls, with no more than0.5pp AUC loss relative to fair. This is a proposed advancement criterion, not a guarantee or a significance claim. Always report other axes, FNR, F_DP, es-AUC, support and valid bootstrap counts. Existing test results have been viewed; disclose that and seek an untouched confirmation set. Three-seed confirmation, a ResNet implementation and held-out manipulation operators require separately bound configurations before final evaluation. The present package is Xception seed17 only.

## Priority handoff

Read GPU_OPERATOR_PROMPT.txt. First run the new CPU/CUDA suite on the target environment. P may then run on original TRAIN while the quartet dependency is prepared. Paired arms remain blocked until real quartet review and all input checks pass. The existing Qwen32 job is a separate diagnostic and is not automatically extended, killed or repurposed. No scientific outcome or GPU ETA is claimed by publication.

## Local software evidence

All six CPU test groups passed: independent numerical/gradient reference; additive-offset and degenerate-group/collapse checks; bank provenance/review/coverage rejection; identical P transfer and frozen BN; all five training arms with checkpoint validation replay; bitwise epoch-boundary resume. See evidence/CPU_SMOKE_RESULT.json and cpu_smoke.log for exact hashes. This uses synthetic75px fixtures with smaller heads on macOS, not real generator output, trained scientific results or a CUDA pass. The real224px target dry run and operator validity checks remain pending.
