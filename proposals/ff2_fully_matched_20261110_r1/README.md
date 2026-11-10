# Fully shared FF2 downstream comparison

The user explicitly requested a fully matched training comparison across methods, use of available GPUs, and all resulting numbers. This is the new primary FF2 suite. The preceding FF++ Ding reference runs are preserved at completed-epoch checkpoints and paused to prioritise this suite.

## Frozen matrix

Methods: BCE, DAG-FDD, DAW-FDD, Ding SFD+GDA, L_H, L_H ramp, L_H low response. Seeds:17,7,27. Exactly21ten-epoch runs, plus one TRAIN-only shared SFD-statistics pass. No best-seed/per-cell aggregation. The finite worker queue cannot add methods, tune hyperparameters or automatically retry failed scientific stages.

Every run shares FF2 public TRAIN218266/SELECT31107/TEST87564, the verified P50 encoder(transferSHA741a8eb5b60ac8efe2ca34be985b423de12facac053acac7f05803df23248a23), a freshly seeded2048→1linear head, Xception original features, exit-stage trainability, frozen BN running statistics,224RGB direct bilinear input and(x-.5)/.5. Nominal total/trainable counts are20809001/6790433. Initial named-parameter hashes are required to agree across methods within each seed. All three downstream seeds share the same seed17 pretrained encoder; these are conditional downstream repetitions, not three independent pretraining runs. The reused encoder was pretrained for50epochs; this common upstream cost must accompany the10downstream-epoch description.

Shared optimisation: AdamW headlr.001/encoderlr1e-5,weight_decay1e-4,betas(.9,.999),eps1e-8,gradientclip5,batch32,workers4,BF16forward/FP32losses and evaluation,10epochs/6821main updates per epoch, each TRAIN original once per epoch, same deterministically seeded shuffle, no LR scheduler/dropout/base-image augmentation. Base classification is unweighted BCE. CVaR's declared robust reweighting is part of DAG/DAW, not a different base sampler. The historic L_H's inverse-cell classification weights are removed in this controlled port so this choice is shared.

Every method receives the identical ordered main original/sham/treated and auxiliary original/sham/treated forward exposure. Auxiliary policy is the same eight gender×race groups for all methods,4real+4fake/group, every4main updates, deterministic draws. All procedural pixel code is unchanged padding-safePS-v2. Losses alone select which views affect the gradient. This deliberately standardises the historic L_H five-axis training auxiliaries to the shared joint axis; all six demographic diagnostic axes remain evaluated. Data-forward fingerprints on real CUDA inputs are required to agree across methods within each seed. Extra computation intrinsic to the objective is reported, never represented as exactly equal walltime.

## Objective adaptations

- BCE: common original-image BCE.
- DAG: exact pinned CVaR of per-original BCE,alpha.5.
- DAW: exact pinned inner CVaR.9 per joint group and outer CVaR.5.
- Ding: common BCE plus connected Sinkhorn fairness,lambda.005,p2,blur.0005. A uniformly chosen group's balanced reference comes from the common auxiliary batch and is detached. Symmetric two-way logits(-z/2,z/2) derive from the SAME one-logit classifier. Auxiliary frequency is compensated by4, identically to other auxiliary objectives. SFD uses the original port's channel-SNNL ranking on the shared pretrained representation, TRAIN only, every TRAIN image once,2%of2048channels(41unless cutoff ties), applied as a fixed feature mask. The parameter tensor architecture/trainability remains shared; the mask is the declared method component. This is a common-representation/common-optimiser Ding adaptation, not the native512channel/head-onlySGD or published50epoch recipe.
- L_H: common BCE plus existing hierarchical joint/parent error objective(.1) and existing sensitivity(.1),sham(.05effective)and alignment(.1), with their original relative/event weighting.
- L_H ramp: same loss, response multiplier0,0,1/3,2/3,1,1,1,1,1,1.
- L_H low: same loss, response multiplier0,0,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3.

These are explicitly controlled adaptations. Architecture-specific PG-FDD/FairAdapter/native multi-head variants are not silently represented by these rows. No claim of full counterfactual fairness, validated real-forgery interventions or superiority follows from a chosen objective.

## Data, recipe review and software evidence

All249373TRAIN/SELECTimages were freshly rehashed and decoded, using32parallel I/O checks after preserving a latency-limited serial attempt. Their full ordered inventory is byte-identical to the prior verified frozen inventory; zero cross-split image/contributor/content overlap was found under the stored metadata rules. A verified canonical row cache removes redundant per-image filesystem resolution at every job start. Pixel SHA256 remains checked on actual image use. Tests compare normalisation exactly to the historical reader. This does not establish true biological-identity or demographic-label ground truth.

The pre-existing88-casePS-v2 procedural recipe panel uses fixed panel seed17 and spans all requested joint groups. All original image/contact-sheet bytes and settings are verified unchanged, as is the complete operator-codeSHA8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733. Existing named reviewer decisions are copied verbatim and explicitly bound as a method-neutral recipe panel for every training seed. No new visual/human review or approval of every future training draw is claimed. Core qualification still checks the bound panel/review hashes. It establishes only the unchanged reviewed procedural recipe, not real-world forgery validity.

Nine focused tests passed: pinned loss/value/gradient parity, identical all-method forward exposure, cache-reader equivalence, exact lifecycle/resume, unchanged padding-safe recipe, schedules, binary-logit/BCE equivalence, SFD/parameter-count behavior, and TEST gating. Each method/seed additionally needs a full-width actual-data CUDA backward with frozen BN and common parameter/input fingerprints before training. The new suite wraps an immutable functional core; historical code/configurations/results are not edited.

## Selection, evaluation and reporting

All methods select earliest strict maximum SELECT AUC and calibrate one global threshold on SELECT reals for targetFPR10%, minimum group support20/class. Training's soft-error threshold uses the same fixed2048TRAIN-real subset and epoch refresh. No fairness-selected replacement epoch. After all10epochs and exact selected-VAL replay, source/config/checkpoint/threshold are frozen before TEST scoring. Each new checkpoint is scored once on the declared public group-valid TEST, with no TEST calibration or tuning. The previous benchmark has been exposed in prior development; these are fixed descriptive benchmark comparisons, not a claim of untouched confirmatory evaluation.

Report each seed and mean±sampleSD over the three downstream seeds, overall AUC/FPR/FNR, gender/race/joint/age/skin/ids8 F_FPR,F_DP,es-AUC,gaps,worst-groupAUC,group support and seven-family AUC. Derive other requested error-gap metrics from the same fixed group rates; never substitute validation into TEST or combine extrema from different seeds/models. Keep every regression and incomplete/unverified row explicit. Additional contributor-aware uncertainty retains its support requirements if computed separately.

## Resources and actual artefacts

Use all eligible shared-poolH100capacity on gpu01/gpu02, at most4new workers/node to preserve host throughput, one new queue worker/card,16GBreservation plus4GBadmission headroom and allocatorfraction.18. Claims use the established shared pool-v2 mutex and live reservations; no unrelated job is stopped, and exclusive physical/PBS allocation is not claimed. Only the newly owned FF++ references were stopped with saved checkpoints preserved when FF2 became the priority.

Each scientific job has a6.25hour held-card cap(including CUDA checks and frozen TEST), native epoch-boundary allowance6h. The fixed21-run matrix plus oneSFDpass is finite; maximum137.5sharedGPU-hours if every job reaches its cap, with actual costs recorded. Workers wait without CUDA allocation when no compatible card is available and terminate after the finite queue/24hour worker window. Failed jobs are preserved and receive no automatic retry. No installation changes the shared Python environment; verified isolated GeomLoss0.3.1 is reused.

Output root:/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1/. Common recipe and the21configuration-difference proof are in COMMON_RECIPE.json and MATRIX_VERIFIED.json. Actual source commit/dependency hashes/launches, queue stages, histories, validation and TEST reports remain there. Public/shared summaries contain aggregates only; faces, weights, private scores and reviewer details remain local.
