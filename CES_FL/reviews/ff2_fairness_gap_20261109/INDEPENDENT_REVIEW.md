# Independent methodological review of the FairForge-2 fairness gap

Reviewed 9 November 2026. Reviewed commit: `e7fe0ee03e6cfacca81c558bbf632d2f72e80bc0`, branch `review/ff2-fairness-gap-20261109`. The problem report's underlying result base is `fd837c9ecefc7eb15034092111ad55c22807f4f6`.

**Recommendation: complete one source-stratified frozen-score diagnostic before another training change.** The RF implementation does apply fairness gradients, but its objective does not guarantee the reported hard-FPR disparity. Processing sensitivity is measured; the fraction of demographic disparity attributable to source composition is not established. No evidence presently establishes that the original utility guards and Ding-level disparity are jointly attainable by this model.

This is a documentation-only review and proposed diagnostic handoff. No model, configuration, preregistration, approval, dataset or completed result was changed. No training, inference, image generation, calibration or cluster job was executed. See [operator handoff](READ_ONLY_OPERATOR_PROMPT.md) and [verification record](REVIEW_VERIFICATION.json).

## 1. What was verified, and what was not

Read AGENTS, README, STATUS, WORKFLOW, PROBLEM, EVIDENCE, both actual scientific Python files, all RF proposal configurations and compact histories, the 60 full RF epoch aggregates, configuration/selected-checkpoint receipts, processing probe, six-table LaTeX/provenance/snapshot, and paper claims/evidence map. Historical README/status entries were not treated as current observations.

The local SHA256 checks match the problem report:

| File | SHA256 |
|---|---|
| `src/ces_fl_realfpr.py` | `2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1` |
| `src/ces_fl_head_ablation.py` | `8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733` |
| Six-table `SNAPSHOT.json` | `20d0ae005242a0389ee2c53d1e2f18b460411be8e0617d974eab96ae58c222e9` |

The review's 68 supported validation rows and 80 TEST rows agree numerically with their corresponding six-table snapshot rows. Independently recomputing pooled FPR and sum-of-deviations from the eight saved TEST group rates gives:

| Coherent seed17 model | AUC % | FPR % | FNR % | Gender x race 100F_FPR |
|---|---:|---:|---:|---:|
| CES-FL P | 98.7838 | 12.3874 | 2.9224 | 42.2314 |
| Corrected local GDA | 93.0819 | 11.3766 | 7.8041 | 9.4734 |

This is a real fairness/utility trade-off, not an equal-utility algorithm comparison. Lower GDA AUC does not prove that its poorer detection causes its lower disparity. It remains a corrected local adaptation, not an exact published-paper reproduction.

The six-table packet explicitly combines different runs/seeds per cell. It is an exploratory oracle; its best cells are not one detector. The coherent-model paper tables are the appropriate comparison. The reserved CAL subset is documented separately, but its unused status and contributor independence were not independently established in this review.

Private images, contributor identities, row-level source histories, prediction files, checkpoint bytes and cluster execution were unavailable to this review. Receipt consistency is not independent execution replication. Freeze chronology/clock skew and contributor-aware uncertainty remain unresolved as recorded in `paper/CLAIMS_AND_GAPS.md:13-16`.

Reproducibility detail: seed17 RF-S records an earlier RF source (`5ee7c8b...`); later RF runs record `2d99295...`. The first-round configs lack three JPEG keys required by the current schema. Use each historical run's recorded source/config, not a claim that today's file can replay every archived config unchanged. This does not invalidate the saved measurements.

## 2. Ranked diagnosis

### 1. Source/processing entanglement is the strongest measured lead

**Measured:** the frozen L_A source probe reports FFHQ-real flag rate 67.8%, versus a separately reported FairFace-real 3.6%. Under the matched-geometry JPEG treatment, the 242 FFHQ reals change to 2.1% flagged, but detection on 360 FFHQ-sourced fakes falls from 99.7% to 5.8%. Geometry-only processing behaves differently. See `paper/evidence/PROCESSING_PROBE.json` and `PROBLEM.md`, section 4.

**Hypothesis:** acquisition/processing cues correlate with class and demographics, and the model relies partly on those cues. Source mixtures differ across groups and splits. Removing a cue can therefore change both disparity and real/fake discrimination.

**Unverified:** compression is the sole cause, the treatment preserves forensic evidence, the same effect holds for P/RF/GDA, or fixing preprocessing necessarily improves utility. The treatment bundles geometry/resampling/encoding effects; 360 images from 24 methods are not necessarily 360 independent source identities. Existing interval construction does not establish contributor-aware uncertainty. Repeating stronger JPEG is not a new hypothesis.

### 2. The objective controls a different quantity from the reported metric

**Verified code:** RF uses Xception + GAP + linear head; no demographic edits or procedural response losses are called. Only the pooled reference is detached in `ces_fl_realfpr.py:125-129`; the group rates retain gradients. There is no evidence here of the archived Ding detached-fairness-loss bug recurring in RF.

**Structural limitation:** one smoothed exceedance probability at a moving TRAIN threshold need not control hard FPR at the separate validation threshold, much less under source shift. See the derivation below. The present two-sided loss also directly pushes some below-reference real scores upward.

**Unverified:** how much of the measured gap is smoothing, threshold drift, score scaling, optimization conflict or generalization. Public epoch aggregates omit the logged TRAIN thresholds and group soft-risk EMA. Gradient conflicts and within-source score CDFs have not been demonstrated for these RF runs.

Replacing the absolute deviation with a one-sided penalty or adding parent-group terms is not a new proposal here: `support_upper_error` already exists at `ces_fl_head_ablation.py:732-755`, and `hierarchical_real_error` at lines676-723. The reported hierarchical/procedural L_H-PS attempt did not establish a broad fairness win. These mechanisms should not be presented as untested fixes.

### 3. The desired operating point may be harder than the comparator's

**Measured:** RF requires validation AUC >=99.5% and FNR <=0.5651%; local GDA has 99.06% and 1.07%, so GDA itself fails those guards. Strong JPEG J/SJ, full-tuning SJF, and SJ_ramp all failed to produce a feasible checkpoint. Gentle JPEG produced feasible checkpoints but did not meet the fairness target. See proposal README:39-57 and histories.

**Interpretation:** the study asks CES-FL to dominate the comparator in utility as well as disparity. This is a legitimate target, not an established feasible point. Selection guards reject checkpoints; they are not mathematical utility constraints enforced by the training gradients. Keep the historical guards fixed.

**Unverified:** an unavoidable trade-off for the dataset, or insufficient Xception capacity. A full-tuning attempt already exists; merely recommending more unfreezing/epochs repeats an attempted direction without isolating its mechanism.

### 4. Support, weighting and axis coverage can obstruct generalization

**Verified:** each RF fairness event uses only 16 real images per gender x race group. The BCE weights equalize `(label, ids8)` cells, not acquisition sources, and `ids8` should not be assumed to mean gender x race. The RF loss explicitly targets only gender x race even though inherited config fields name other axes. TRAIN calibration uses a fixed 2,048-real subset. Validation has 2,438 reals, with 194-621 per joint group.

**Hypotheses:** repeated contributors, sparse source-by-group cells and noisy tail estimates may create unstable gradients and optimistic selection. For scale only, a hard Bernoulli rate near 10% estimated from 16 independent examples has standard error about 7.5 percentage points; this is not a measured error bar for the soft loss. Absolute minibatch deviations also include sampling noise.

**Measured:** RF-S seed17 validation 100F_FPR goes from 34.12 at selected epoch2 to 38.67 at epoch10. RF-SJg seed17 reaches 23.00 at epoch10 but FNR0.7499% fails its guard. There is no simple monotonic fairness-convergence story. Changing temperature and threshold means raw loss magnitudes across epochs are not directly comparable.

Age/skin estimates and unknown values limit label-level claims. Three RF downstream seeds share one pretrained encoder; sample SD is not a confidence interval or full-training variance estimate.

## 3. What the RF surrogate can and cannot guarantee

Let `q_g(t)=P(Z>=t | Y=0,G=g)` and `q(t)=sum_g pi_g q_g(t)` when these groups cover all real images. The reported metric is

`F_FPR(t) = sum_g |q_g(t)-q(t)|`.

It is a sum across actual intersectional cells, not the product of gender and race FPRs. Its scale depends on the number of eligible groups; `100F_FPR` is not an individual probability. The executable metric uses eligible subgroup rates but the overall rate includes the whole evaluation set (`ces_fl_head_ablation.py:865-889`), so unknown/unsupported coverage must also be checked.

RF instead minimizes the mean across groups of `|s_g - stopgrad(sum pi_train*s)|`, where `s_g=E sigmoid((Z-t_train)/tau)`. It uses natural TRAIN-real group priors, not validation priors (`ces_fl_realfpr.py:125-129,151-160,181-204`). Calibration selects a finite-sample real-score quantile; it does not calibrate each group's error probability (`ces_fl_head_ablation.py:414-424`).

**A counterexample, not an experimental result:** suppose group A's smoothed scores are all 0.2; group B has 20% at 0.6 and 80% at 0.1. Both soft means are 0.2, so RF disparity is zero. Hard thresholding at smoothed score0.5 gives FPR0% versus20%; with equal group masses the pooled hard FPR is10%. Equal soft means can coexist with a large hard gap even at the same decision threshold. These scores are realizable by finite logits for any positive temperature.

A useful bound explains what additional assumptions are needed. For any margin width `delta>0`, on a fixed group distribution,

`|q_g(t)-s_g(t,tau)| <= P(|Z-t|<=delta | Y=0,G=g) + exp(-delta/tau)`.

Changing the threshold from TRAIN to validation adds the probability mass between those thresholds. Deployment/source shift, empirical estimation error, unknown-group coverage and changes in group priors add further discrepancies. Thus small soft disparity helps control hard disparity only with small smoothing error, threshold stability and distributional generalization. These conditions have not been established. Decreasing temperature is not a guarantee: it localizes gradients and can miss confident false positives.

Detaching the pooled reference does not make the loss one-sided. For a below-reference group, the direct loss derivative increases real scores. Nor does zero disparity anchor the common error level. Shared parameters can change fake scores as well; BCE alone supplies no per-group FNR guarantee.

**Group offsets:** if a procedural pair's scores both contain an additive group offset `b_g`, then `z_t-z_s` cancels it. Equal intervention responses therefore do not imply equal original-image FPR. The earlier response code uses `asinh(z_t-z_s)` (`ces_fl_head_ablation.py:779-797`). Original-image control is still necessary if responses return, but those terms are absent from RF and cannot explain RF as an active mechanism.

**Several demographic axes:** exact equal hard FPR across every gender x race cell implies equal marginal gender/race FPR at that threshold. A decrease in the aggregate sum of deviations does not imply every parent gap improves. Age and skin cut across those cells and are not controlled by the joint objective. Equal FPR, equal FNR and demographic parity also need not coincide: with fake prevalence `a_g`, positive rate is `(1-a_g)*FPR_g + a_g*TPR_g`. Different prevalences can yield different positive rates even under equalized odds. Class imbalance does not by itself mathematically require unequal FPR, which conditions on real images; it affects learning, support and other metrics.

## 4. Which direction is justified now?

1. **Source/data controls first, as a diagnostic.** Marginal disparity can arise from unequal source mixtures, within-source disparities, or both. Those require different remedies. Inspect provenance and class-conditional score distributions before altering pixels. A blanket JPEG revision is contradicted by the measured fake-detection collapse. A future data revision would require a separate version and consistent comparator treatment; historical tables cannot silently inherit it.
2. **Class-conditional CDF/tail alignment is theoretically stronger, but not yet the selected experiment.** If every group's real-score CDF is within epsilon of the pooled real CDF uniformly over the deployment threshold band, hard `F_FPR <= K*epsilon` there. A separate fake-class condition controls FNR disparity. Aligning only means, a finite sparse threshold grid, or small Wasserstein distance does not automatically provide this uniform bound without additional assumptions. Matching distributions separately within class avoids explicitly pulling real and fake targets together, but shared parameters can still erase useful evidence. Source/family composition and utility must remain controlled. This is not novel merely because it uses CDFs or alignment.
3. **Group-aware calibration can be a labelled diagnostic, not a hidden repair.** It could test how much a changed decision rule helps existing scores, using independent calibration/evaluation roles. It changes inference information and may change pooled AUC if scores are group-transformed. Good within-group AUC does not prove an unbiased representation. A single strictly increasing global calibration preserves ranking and, at the same global FPR budget, generally preserves the decisions; it cannot resolve group ordering. No calibrator is recommended for execution in this handoff, and no group thresholds belong in global-threshold rows.

## 5. Exactly one recommended study: source-standardized frozen-score audit

**Hypothesis:** a material portion of the marginal real-image demographic disparity is associated with differences in acquisition/processing-source mixture; the residual within comparable sources may require a different objective. This tests a mechanism not isolated by the completed JPEG-strength/ramp/full-tuning experiments. It is a diagnostic hypothesis, not a claim of a novel model.

**Matched objects:** use the already selected RF-S and RF-SJg checkpoints at seeds17/7/27, their existing unchanged-image SELECT score exports and stored global thresholds. Each model's unstandardized result is its paired control; RF-S is the matched algorithm control for RF-SJg. No new Ding computation. Seed17 is the primary descriptive pair, with all three paired downstream differences reported rather than selecting the best seed.

**Mathematical change:** change only the diagnostic estimand, not the network, training loss, images, scores or decisions. Let S be documented acquisition/processing provenance, not a source guessed from appearance. Estimate

`q_gs(t)=P(Z>=t | Y=0,G=g,S=s)`.

The native rate is `q_g=sum_s P(S=s|G=g,Y=0)*q_gs`. Compare it with

`q_g_std=sum_s w_s*q_gs`, with `w_s=P_TRAIN(S=s|Y=0)` fixed before accessing private score results, the same source weights for every group.

Use the same SELECT real-group proportions `pi_g` to construct native/standardized pooled rates and their sums of deviations. Primary contrast: `D_native-D_standardized` within RF-S seed17; repeat descriptively for RF-SJg and all matched seeds. Also report the paired SJg-minus-S contrasts in both views. Neither subtracting these nonlinear metrics nor observing a reduction identifies a causal percentage explained.

**Required validity and coverage:** verify saved score/checkpoint/manifest bindings, unchanged-image preprocessing, documented source/crop/encoding lineage, group annotations, duplicate lineage and all source-master/contributor links available. Source IDs used for identity overlap must not be conflated with acquisition-dataset labels. Keep age/skin proxies and unknowns explicit. Predefine provenance strata from documented TRAIN metadata, without using errors, TEST, or appearance-inferred demographic labels. All original groups/sources remain in native reporting. Empty required source-by-group cells make full standardization unidentified; do not drop them, impute counterfactuals, or extrapolate a full-population claim. Report small contributor counts and uncertainty even when a cell is nonempty.

**Utility check:** report unchanged-image native AUC/FNR and FPR, per-group FNR, and fake-family/source breakdown. Any supplementary fake standardization is class-conditional and must respect source-by-family support; otherwise report the available strata and mark the combined result unavailable. A standardized diagnostic fairness number never replaces a native model row or establishes useful detection under a new pixel treatment.

**Unchanged factors:** all existing checkpoints, their selected epochs, source hashes, fixed global thresholds, split membership, labels, image bytes, loss/optimizer settings and historical guards. No preprocessing revision, training, fresh prediction, threshold search, group calibration or TEST/CAL access. Existing TRAIN metadata and SELECT scores only. Missing cached scores or provenance are blockers for the corresponding analysis, not permission to infer missing inputs.

**Uncertainty:** use paired resampling of recorded source-master/contributor connected components, keeping derived images and paired models together. Acquisition datasets such as FFHQ are strata, not two independent bootstrap units. Preserve fixed thresholds and group eligibility. Prespecify 2,000 bootstrap replicates, analysis seed17 and >=95% valid replicates; report denominators and failures. Give a primary 95% interval for the source-standardization contrast; label additional axes exploratory with a stated multiplicity policy, or supply simultaneous intervals. Incomplete biological identity links limit these to recorded-provenance intervals. They condition on already selected checkpoints, do not include selection/pretraining uncertainty, and do not turn exposed SELECT into confirmation data.

**Decision/stop conditions:**

- Missing source history, score binding or required support: BLOCKED for the affected estimand; publish missingness, do not fill values or remove difficult sources.
- Insufficient valid bootstrap replicates or intervals spanning both directions: INCONCLUSIVE; do not declare source composition the dominant cause.
- Precisely positive native-minus-standardized disparity: evidence of a source-composition association, not causal proof or a new model improvement. Report its magnitude and residual within-source disparity.
- Substantial, well-supported residual disparity: reject a source-mixture-only explanation; a later class-conditional tail objective would need its own proposal and matched control.
- The audit stops after its report. It does not automatically authorize training. Keep AUC>=99.5% and FNR<=0.5651% as the historical native validation guards; do not lower them or choose JPEG strength/thresholds from this audit.

An independent final improvement claim needs genuinely unused confirmation data with verified provenance/independence and a declared role. The existing TEST cannot become fresh by reshuffling. A reserved CAL set is not automatically a valid untouched confirmation set just because completed selection did not use it.

## 6. Closest prior work and defensible novelty

| Primary source | Overlap and consequence for CES-FL |
|---|---|
| [Ding et al., Sec.3.3/Eq.6](https://arxiv.org/html/2511.10150v5#S3.SS3) | Already aligns demographic distributions separately within real and fake classes. Generic class-conditional alignment is not a new CES-FL contribution. Published and corrected local implementations must remain distinguished. |
| [FYC, Sec.3.2-3.3](https://arxiv.org/html/2610.37882v1#S3.SS2) | Real-only probability-disparity weight adaptation and group-specific Youden thresholds already exist. RF's threshold/temperature/reference differ, but “optimize FPR fairness” or adding calibration is not sufficient novelty. The paper's empirical successes are not a guarantee for FF2. |
| [FairReL, Sec.3.4](https://arxiv.org/html/2608.28777v1#S3.SS4) | Class-conditioned mean alignment in an SVD-isolated residual representation, alongside wavelet supervision. Representation alignment/localization has close prior art. |
| [Jiang et al., Wasserstein Fair Classification](https://proceedings.mlr.press/v115/jiang20a.html) | Distribution matching for threshold-robust fairness predates this project. Tail/CDF formulations require a sharper contribution and evidence than renaming alignment. |
| [Hardt et al., Equality of Opportunity](https://arxiv.org/abs/1610.02413) | Group-aware post-processing is established, with explicit utility trade-offs. It belongs in a separate inference protocol. |
| [Sagawa et al., Group DRO](https://arxiv.org/abs/1911.08731) | Worst-group risk optimization and the importance of generalization/regularization are established; low training group loss does not certify fairness generalization. |
| [Grommelt et al., Fake or JPEG?](https://arxiv.org/abs/2403.17608) | JPEG/size confounds in synthetic-image detection are established. Showing processing sensitivity alone is not a new general phenomenon. |

Current RF is an incremental controlled combination, not an established novel forensic-disentanglement architecture. A defensible present claim is: **on this fixed FF2 protocol, gentle processing augmentation reduces measured intersectional FPR disparity relative to the RF loss-only control, conditional on a shared pretrained encoder, while detection and some demographic axes worsen.** Mean +/- downstream SD is descriptive; statistical significance is not established.

A possible research contribution is a carefully identified link between processing provenance, demographic mixtures and thresholded error disparities, followed by a method that demonstrably addresses the isolated mechanism while retaining utility under one global threshold. That would require source-controlled evidence, reliable labels/support, class/family sensitivity checks, appropriate baselines, uncertainty and prospective confirmation. Dataset ownership, an extra loss, a diagram, or per-cell oracle wins supply none of those guarantees.
