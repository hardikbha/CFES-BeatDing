# Prospective protocol — cdf_response_20261102_r1

Freeze this version before launching its experiments. Machine-readable decisions: [PROTOCOL.json](PROTOCOL.json); exact configurations: [configs/](configs/). This study does not amend old CES-FL protocols or the failed CESel-MA evidence-floor gate. Numerical choices below are fixed initial choices, not validated optima. No sweep against TEST is permitted.

## Hypothesis and matrix

Primary: response versus score reduces gender×race F_FPR at one global validation-calibrated operating point with bounded AUC/FNR trade-off. Operation-response alignment must add value beyond group distribution parity and demographic score consistency.

| CLI arm | Objective | Required data | Interpretation |
|---|---|---|---|
| `bce` | Mean BCE | Original TRAIN | Matched supervised baseline |
| `dro` | Exponentiated label×gender_race group-DRO BCE | Original TRAIN | Standard robust-risk baseline |
| `mean_fair` | BCE +0.1 existing real mean-soft-score disparity | Original TRAIN | Existing fairness formula under the common new sampling policy |
| `cdf_fair` | BCE +0.1 real CDF parity | Original TRAIN | Objective/metric mismatch ablation |
| `nochange` | CDF objective +0.1 original/editor-no-change score consistency | Reviewed six-view bank | Editor processing control |
| `score` | CDF objective +0.1 no-change/demographic score consistency | Same bank and exposure | Demographic invariance anchor |
| `response` | Exactly `score` +0.1(response alignment + sensitivity) | Same bank and exposure | Primary incremental response hypothesis |

`score` does not also receive the original/no-change loss. That term is isolated in `nochange`; it is not silently added to another objective. The primary `response−score` contrast adds the declared response package; it does not isolate alignment from sensitivity individually. A later sensitivity-only ablation requires a new dated specification.

Initial development seeds **7,17,27**, both Xception and ResNet-50. First stage after review: Xception four non-paired arms (12 downstream runs; two new P50 runs, reusing verified seed17). ResNet uses the same four-arm matrix when its own P50 bindings/checks pass; no backbone chosen from TEST performance. Three paired arms per backbone wait for the bank. Do not launch all stages at once. Additional confirmation seeds **37,47,57,67,77** are declared now; all eight are required before a confirmatory paper claim. Three-seed development results alone remain exploratory.

Separate comparator: **Ding-CC-OT-r1**, [exact stage specification](DING_CC_OT_CONFIG.json), not archive-faithful GDA or certified paper runs. Its production stage wrapper is not implemented in this release; no comparator launch command is represented as ready. Use the same seeds (extend its concrete config to all eight before confirmation), TRAIN/VAL membership, downstream original sampling, number of optimizer steps, validation selection and calibration in a prospective common-protocol comparison. Native published/final-checkpoint rows remain separate. Optimizers and stage exposure differences must be reported; they are not attributed to CES losses.

## Mathematics and implementation

Let z be the scalar fake logit, y=0 real/y=1 fake, and g one of eight declared groups. Fixed priors pi_g are **natural TRAIN-real** group proportions, not the quota batch proportions. For detached TRAIN-real knots K={unique real logits in the batch}:

`S_g(t) = mean_(y=0,g) sigmoid((z−t)/0.25)`

`S_pool(t) = sum_g pi_g S_g(t)`

`L_CDF = max_(t in K) mean_g |S_g(t)−S_pool(t)|`.

At the population level identical real-score distributions imply equal FPR at every global threshold. The finite, smoothed batch maximum is only a surrogate and does not guarantee equal hard FPR, especially at an unseen validation-selected threshold or after domain shift. TRAIN knots are detached; no validation threshold, label, image or score enters the objective. The mean over groups differs by a factor of eight from summed F_FPR at a single threshold; lambda0.1 is fixed on this explicit scale. A constant-score model can minimize parity, so BCE, AUC and fake-image FNR guards are necessary. No clipping/post-hoc per-group threshold is used to create a fairness win.

The existing mean formula uses mean absolute group sigmoid(z) deviations from pooled TRAIN-batch real sigmoid(z), temperature1 and logit0. Under complete quotas that pooled mean is group-balanced; the historical run's natural batches and label×ids8 weights are different. This is an explicitly matched **new baseline**, not a rerun reproducing its historical numbers.

The CDF-versus-mean comparison also changes temperature and pooled group weighting as declared above; it evaluates that fixed objective package, not an isolated causal test of threshold grids alone. The equal-mean synthetic counterexample proves the old surrogate is insufficient, but does not prove it caused the historical fairness deterioration. The primary response-versus-score contrast holds the entire CDF package fixed.

For six aligned bank views, r is the original, r_resize its preprocessing control, x0 its editor no-change, xa its independent age edit, t0=T(x0), ta=T(xa). Define:

`L_nochange = mean |asinh(f(r))−asinh(f(x0))|`

`L_score = mean |asinh(f(x0))−asinh(f(xa))|`

`R0 = asinh(f(t0)−f(x0)); Ra = asinh(f(ta)−f(xa))`

`L_align = mean |R0−Ra|`

`L_sens = mean [relu(0.5−R0)+relu(0.5−Ra)]/2`.

`response` adds L_align+L_sens to `score`. These preserve the existing response mathematics; asinh of a *difference* is intentional, not difference of asinh scores. Positive T-response means stronger fake evidence after a validated operation applied to a real parent. Margin0.5 means a raw logit response at least sinh(0.5)≈0.5211 to satisfy that hinge; it is not an age or fairness threshold. Response cancellation under common group offsets is tested; CDF controls, rather than alignment alone, address group error distributions. Resize difference is **diagnostic only**, with no added control/map loss.

## Initialization, exposure and optimizer policy

Each backbone/seed has a shared original-only P50 encoder. Initialization uses the exact cached weights and input hashes in each config. P50: 50 epochs, batch16, each TRAIN original once/epoch, fixed label×ids8 inverse-frequency weighted BCE, AdamW, encoder LR1e-4, other LR1e-3, weight decay1e-4, all parameters trainable and BN updating. Only the verified **Xception seed17 P50** is currently bound; ten-epoch P checkpoints are not substitutes.

All downstream arms: same imported encoder, identical fresh prediction/head/gate/fusion seed, 50 epochs, AdamW, encoder LR1e-5, other LR1e-3, weight decay1e-4, gradient clipping5, all parameters trainable, **encoder BN buffers frozen**. No LR scheduler or augmentation was added. Direct bilinear224×224 resize and backbone-specific normalization are identical within each comparison. No architecture changes; map consistency off.

Every downstream batch has four examples from each of 2labels×8groups: batch64. Replacement sampling has the same seed/epoch schedule in all arms. `ceil(76139/64)=1190` updates/epoch, 76,160 original draws/epoch and 59,500 updates over50epochs. This is **not every original once**; log unique originals and per-cell draws. Group-DRO keeps all16cell probabilities and rejects absent cells; eta0.01 is fixed. This avoids silently varying the classification scale with absent groups.

Paired arms all perform the same additional six-view forwards for four quartets/step, same cyclic shuffled bank schedule and step count. Log both quartet and unique source/component exposure; each arm has 238,000 quartet draws and 1,428,000 paired image-forwards over50epochs. The bank size is not the number of independent training people. Non-paired arms have no paired image exposure; improvements from CDF→paired are not solely objective-controlled, while response→score is exposure-controlled. BCE applies only to originals. No validation or test gradient updates.

## Selection, calibration and reporting

For every arm choose the highest validation AUC checkpoint; ties keep the earlier epoch (`>`). No fairness-weighted selection, no test selection. Independently reload/replay the selected checkpoint before DONE. Calibration uses that checkpoint's validation **real images only**, target FPR0.1. Each model has one global threshold, common to all its groups and domains; the numerical thresholds may differ across models.

For n validation-real scores descending, k=floor(0.1n) exactly, set t=nextafter(score[k],+infinity), classify fake iff z≥t. Thus tied blocks are excluded together; achieved empirical FP≤k and may be below target. Report actual FP/n, including ties and finite-sample limitations. Freeze checkpoint and threshold before any TEST evaluation. Never recalibrate on DFDC/Celeb-DF/DFD or FF2. Supplementary fixed0.5 probability means logit0 and is reported separately. If reproducing a paper row requires another operating protocol, label it explicitly rather than mixing columns.

Report overall AUC; FPR and FNR separately; worst-group AUC/FPR/FNR; group gaps; Ding-style F_FPR=sum_g|FPR_g−FPR_all|, F_DP=max positive-rate−min positive-rate, es-AUC; gender/race/intersection/age_bin/skin_bin where defensible; forgery-family metrics; all per-class/frame and contributor counts and unknown/missing/excluded labels. Eligible aggregate groups require ≥20 real and ≥20 fake frames. Still show descriptive one-class FPR/FNR where defined; mark AUC undefined. No unknown group silently treated as ground truth.

Primary development decision is response versus score on VAL: AUC decrease≤0.005, summed F_FPR reduction≥0.02, overall and every eligible group's FNR increase≤0.02, worst-group AUC decrease≤0.01, in **each** of three seeds. Apply the same utility/fairness rule descriptively to CDF versus BCE. No eligible-group deletion to pass. Failure stops expansion or requires a new proposal; do not adjust weights on TEST.

Uncertainty uses 2,000 paired hierarchical seed/connected-contributor bootstrap draws, seed1729, shared draws across methods, fixed selected thresholds/checkpoints and fixed original support eligibility. Report individual-group and paired-delta intervals, per-group component support, all invalid replicates. With <20 overall components, <5 per class/group, or <90% valid replicates, do not make a strong inferential claim; <2 components yields no interval. Hidden donor links, frame dependence and proxy mislabels are not solved by bootstrap. Training-seed variability and sampling uncertainty are both reported; do not treat frames or replicated edits as independent people.

These intervals condition on the fitted, selected checkpoints and their calibration sample; they do not repeat checkpoint selection, re-fit a threshold or model demographic-label uncertainty. Reusing development VAL for selection, calibration and diagnosis limits inferential interpretation even with bootstrap. A future independent evaluation is needed; do not describe VAL intervals as an unbiased confirmatory test.

The actual VAL-only metadata calculation now shows209overall recorded components, but Female-Black has2perclass, Female-Others real4 and Male-Black fake4 ([receipt](evidence/VAL_COMPONENT_SUPPORT.json)). The declared confirmation support floor is therefore **not met** on this development split. Preserve the groups and report descriptive results; do not relax the criterion. More training seeds do not add independent people. A sufficiently supported, independently locked replication is needed for the stronger claim.

Seed handling seeds Python/NumPy/Torch/CUDA and disables cuDNN benchmarking with deterministic cuDNN enabled. It does not enforce every PyTorch operation's deterministic-algorithm mode; no cross-hardware bitwise reproducibility guarantee is made. Record versions and identical initial head hashes, quotas and budgets rather than substituting a generic “fully deterministic” claim.

Before future TEST release, the confirmation plan must freeze a metadata-verified independent holdout and all eight seeds. Primary confirmatory tests are response-versus-score fairness and response-versus-corrected-Ding fairness, separately by the two backbones; Holm correction covers those four tests. Pairing AUC noninferiority/FNR guards is an intersection requirement, not a license to select favorable domains. Use exact seed-level sign-flip tests, label their exchangeability assumptions and report seed deltas. With three seeds minimum two-sided p=0.25; eight seeds provide granularity but not guaranteed power.

A **jointly better detection and fairness** claim requires paired AUC difference95% lower>0 AND F_FPR difference95% upper<0 against the corrected Ding comparator, plus the FNR guards and multiplicity policy. AUC noninferiority alone does not mean better detection. A method-effect claim also requires incremental response-versus-score improvement. Already viewed historical benchmarks are descriptive evidence, not a fresh blinded confirmatory holdout. Do not promise a win or a result on every table/domain.

## Launch conditions and FairForge-2

All production training requires a real external reviewer release bound to config and source hashes, fresh immutable checkout, valid allocation/binding, input preflight and own real-data backward check. Template approval is false. Bind actual per-seed P50 checkpoints rather than inventing their hashes. Paired arms require the additional [bank gate](DATA_GATE.md). No full training or generation is authorized by this proposal-writing task.

FairForge-2 full comparison remains **HOLD/DO NOT RUN** under its existing supervisor review. User explicitly permits only the already-running original-only pretraining PID2525928 to finish. No automatic downstream, Ding comparison or TEST evaluation. Keep public218266/31107/87564 membership, internal/full-split historical builds and FF++→FF2 transfer distinct. Any FF2 release needs its own new reviewed exact proposal and resolved listed blockers; these FF++ configurations are not substitutes.
