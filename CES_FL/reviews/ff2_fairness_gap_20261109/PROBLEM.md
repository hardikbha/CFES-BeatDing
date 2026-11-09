# Supervisor review: why CES-FL still loses on FairForge-2 fairness

Prepared 9 November 2026. **Problem statement and theoretical-review request only.** No model change, new training, generation, calibration or inference is authorized or performed by this document.

The research objective is a useful detector with lower demographic error disparities. FairForge-2 is our proposed dataset, but dataset ownership is not evidence that our model should win. The present measurements do not support an overall fairness win over the corrected local Ding-style comparator.

## 1. Exact state and evidence

Reviewed result/source base: `fd837c9ecefc7eb15034092111ad55c22807f4f6`.

- [Six-table PDF and provenance](../../results/six_best_observed_tables_20261109/README.md): Tables1–3 published Ding references, Table4 public split composition, Table5 validation, Table6 TEST. These tables include user-requested retrospective per-cell bests, which are not one model.
- [Coherent-model TEST tables and limitations](../../paper/README.md): current paper uses fixed model rows / downstream seed means, not the best-cell oracle.
- [Original RF proposal, failed variants and subsequent confirmation](../../proposals/ff2_realfpr_stretch_20261107_r1/README.md).
- [Local aggregate evidence for this review](EVIDENCE.json). Derived only from saved aggregate reports; no predictions were recomputed.

Actually reported RF source: `CES_FL/src/ces_fl_realfpr.py`, SHA256 `2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1`.
Shared model/data/calibration source: `CES_FL/src/ces_fl_head_ablation.py`, SHA256 `8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733`.

## 2. The measured gap

**Validation, seed 17, stored global validation-real FPR10 operating point:**

| Model | AUC% ↑ | FNR% ↓ | Gender×race 100F_FPR ↓ |
|---|---:|---:|---:|
| Corrected local Ding-style GDA | 99.06 | 1.07 | 12.44 |
| DAW-FDD adaptation | 99.84 | 0.24 | 30.70 |
| CES-FL P | 99.81 | 0.38 | 28.39 |
| L_A plain-head BCE | 99.81 | 0.28 | 37.37 |
| RF-S | 99.80 | 0.29 | 34.12 |
| RF-SJg | 99.75 | 0.45 | 27.53 |

All have achieved validation FPR 9.9672%. They do not have the same AUC/FNR or training/selection budget.

**Existing frozen TEST evaluation, coherent seed 17 rows:**

| Model | AUC% ↑ | Achieved FPR% ↓ | FNR% ↓ | Gender×race 100F_FPR ↓ |
|---|---:|---:|---:|---:|
| Corrected local Ding-style GDA | 93.08 | 11.38 | 7.80 | 9.47 |
| DAW-FDD adaptation | 98.87 | 14.41 | 2.29 | 45.35 |
| CES-FL P | 98.78 | 12.39 | 2.92 | 42.23 |
| L_A plain-head BCE | 99.00 | 15.14 | 1.44 | 63.67 |
| RF-S | 98.97 | 14.80 | 1.67 | 59.94 |
| RF-SJg | 98.54 | 13.69 | 2.84 | 47.80 |

RF-SJg has three downstream seeds: mean TEST AUC 98.60±0.06%, primary100F_FPR 47.69±3.91, versus RF-S 98.98±0.01% / 60.95±1.92. These are sample SDs, not confidence intervals. All RF seeds share one seed 17 pretrained encoder. The primary gain over RF-S coexists with gender/age fairness regressions and poorer detection. CES-FL P also has lower TEST primary disparity than RF-SJg's mean; the latest variant is not automatically the best on every axis.

The comparison shows different utility/fairness operating outcomes. It does not prove that GDA's lower AUC causes its lower disparity. The local corrected adaptation is not identical to Ding's published original-benchmark run or the archived detached-gradient GDA.

## 3. What the executable method is doing

The RF variants currently use Xception + global average pooling + one linear head, not the full global/local/gated counterfactual CES-FL architecture. Total 20,809,001 parameters; 6,790,433 trainable under exit-block tuning, with frozen BatchNorm running buffers. All RF runs use the transferred FF2 BCE50 encoder, ten downstream epochs, batch 32, AdamW head/encoder learning rates 1e-3/1e-5.

Every original TRAIN row receives inverse class×ids8-cell-weighted BCE once per epoch. Every fourth step adds 128 unedited real images: 16 per gender×race group. With TRAIN-real priors pi_g,

    r_g = mean_i sigmoid((z_i - t_TRAIN) / tau),  for real images in group g
    b   = stop_gradient(sum_g pi_g * r_g)
    L_RF = mean_g abs(r_g - b)
    L_total = L_weighted_BCE + 4 * lambda * L_RF, on fairness steps

`lambda` ramps 0→0.1 over two epochs; temperature decreases 1→0.25. A single threshold is refreshed approximately four times per epoch on 2,048 TRAIN reals. This already targets a smoothed false-positive risk; claiming that the next idea is simply “use an FPR loss” would repeat existing work.

The loss is two-sided. Below-reference groups may be pushed toward higher soft error. Detaching the reference removes a gradient path through its calculation, not this two-sided behavior. The target is smoothed risk at a TRAIN threshold; the reported quantity is hard FPR at a separately selected validation threshold.

RF-S has no JPEG augmentation. RF-SJg augments 15% of classification images with source-scale-emulated q75/4:2:0 JPEG; the rule is shared across labels/groups. Auxiliary fairness images and evaluation images are unchanged. No generated demographic pairs, forensic-sensitivity terms, response-alignment terms or map-consistency terms are active in RF-S/RF-SJg.

The earlier full L_H-PS arm does include hierarchical real risk and procedural probe terms; its validation primary disparity is 36.23. One-sided/support-aware and hierarchical error-control code already exists in the project. A proposal using those words must explain its actual difference from previously tested configurations.

## 4. Evidence about source processing

An existing validation-only diagnostic with frozen L_A records FFHQ-real FPR 67.8% (242 images) versus FairFace-real FPR 3.6% (2,196 images). TRAIN/validation source proportions and processing profiles differ; the historical report records FairFace at about 81% of TRAIN reals and 90% of validation reals.

Matched-geometry q75/4:2:0 processing changed the flag-as-fake rate 67.8→2.1% for those reals, but also changed detection 99.7→5.8% for 360 FFHQ-sourced fakes across 24 methods. Geometry-only and other quality/subsampling controls behaved differently.

This supports strong processing sensitivity consistent with source confounding. It is not proof that compression alone explains every demographic disparity, that every model uses the same shortcut, or that this is an evidence-preserving demographic intervention. Blind compression can destroy fake sensitivity. Source-master/crop history, contributor identity, label uncertainty and fake-family composition need separate consideration.

Existing aggregate diagnostic: `CES_FL/paper/evidence/PROCESSING_PROBE.json`; original cluster narrative `/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106/REPORT.md`. The original narrative's sole-cause language is stronger than the controls establish.

## 5. Attempts already completed

- Real-only risk control, RF-S: modest validation primary reduction 37.37→34.12 against matched L_A; source gap persists.
- Strong JPEG only, combined JPEG+RF, and full-encoder combined variants: ten epochs each, no feasible checkpoint under the original detection guards. These cannot be suggested again as an untested fix.
- Strong JPEG ramp: all ten epochs infeasible under FNR guard.
- Gentle JPEG: feasible and improves primary disparity, confirmed at three downstream seeds; still misses the original target and regresses on other outcomes.
- Hierarchical-risk/procedural L_H-PS: no broad fairness win established. Full validated demographic/forensic counterfactual training remains unestablished.

RF checkpoint selection minimizes hard gender×race validation F_FPR subject to AUC≥99.5% and FNR≤0.5651%, with AUC/earlier-epoch tie-breaks. **The local Ding reference itself does not satisfy those utility guards** (validation99.06% AUC / 1.07% FNR). Thus the desired outcome is lower disparity than that reference while retaining substantially stronger utility. Feasibility has not been established. Do not relax these historical guards after seeing results; any changed study needs a separate explicit proposal.

## 6. Questions for theoretical review

1. **Where is the real bottleneck?** Given the recorded source-processing sensitivity and existing group score behavior, which mechanisms are plausible: source/confound reliance, inadequate representation, surrogate/threshold mismatch, training-group variance, or constraints that cannot all be met with this representation? What diagnostic would distinguish them without TEST-driven tuning?
2. **Objective alignment.** Does matching a single smoothed exceedance probability adequately control hard FPR at a different validation-calibrated threshold? Would class-conditional score-CDF/quantile/tail alignment across a prespecified operating band be materially different from existing RF/GDA? State assumptions, guarantees if any, failure modes, and why it would preserve fake sensitivity.
3. **Group offsets and response objectives.** A response difference f(x)-f(Tx) cancels a group-specific additive score offset. How should original-image error control and intervention-response alignment be combined if both are used, without falsely assuming response equality implies equal FPR?
4. **Several demographic axes.** How can joint-group risk be controlled without worsening parent-gender or age outcomes? Which quantities can be guaranteed jointly, and which conflict because of different class prevalences? Age/skin are estimated labels with missingness; demographic predictions must not be treated as ground truth.
5. **Data versus objective.** Is a controlled processing audit/revision necessary before another model change? How can processing be standardized for real/fake images across sources without selectively altering groups or removing forgery cues? A versioned data revision must be applied consistently to all comparators; it cannot replace a historical split silently.
6. **Calibration diagnostic.** Would a separately labelled group-aware calibrator clarify how much disparity is decision-level rather than representational? It changes the information/decision rule at inference and is not comparable to the existing single-global-threshold protocol. It must not be slipped into current Tables 5/6 or claimed as a novel CES-FL contribution.
7. **Smallest justified next study.** Recommend at most one focused hypothesis and its matched control, with explicit differences from completed attempts, a utility/fairness decision rule, uncertainty plan and conditions for stopping. Do not propose an open-ended parameter search or invent evidence-valid counterfactuals.
8. **Novelty.** What defensible CES-FL contribution remains beyond existing real-image FPR regularization, distribution alignment, group DRO and post-hoc calibration? What evidence is needed for that claim?

## 7. Closest primary sources to consider

- [Ding et al., distribution alignment / structural decoupling](https://arxiv.org/abs/2511.10150).
- [Hardt et al., fairness post-processing](https://arxiv.org/abs/1610.02413).
- [FYC, real-image FPR weight perturbation plus group-specific calibration](https://arxiv.org/abs/2610.37882). These mechanisms already appear in prior work; no novelty or transfer-success claim is inferred for our pipeline.
- [FairReL, class-conditioned representation alignment](https://arxiv.org/abs/2608.28777).

## 8. Boundaries for the review

Existing TEST results are exposed. Use TRAIN/SELECT for development and disclose exposure; an unbiased final claim needs a genuinely untouched confirmation set and an explicit plan. Do not reshuffle the exposed TEST into a nominally fresh holdout. A reserved subset can only be repurposed after checking its unused status, independence and declaring the new study role.

No new Ding-model computation, dataset edits, GPU allocation, run launch or source change is requested here. Preserve all historical protocols and outputs. The requested deliverable is a theoretical diagnosis, an evidence-based ranked explanation, and one concrete next-study recommendation—not a promise that we will beat Ding.
