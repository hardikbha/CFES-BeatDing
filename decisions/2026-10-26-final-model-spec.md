# Final model specification and primary endpoint — pre-registration, 26 Oct 2026

User: "I am agreeing with your decisions" (26 Oct, after reports/cvpr_gap_20261026/00_SUMMARY). Fixed BEFORE any
matched-CLIP baseline number, any Amendment-5 CESel arm test number, and any age/skin-tone result exists.

## 1. Which model is finalised
**CESel-MA (Counterfactual Evidence Selectivity, Multi-Axis)** is the model we finalise, built in two gated steps.
- **Step 1 gate.** The running Amendment-5 screen (gender × race floor) must show that the evidence floor works:
  Full or +floor beats B0 on the validation selection value by more than the declared spread, and no negative control
  matches it.
- **Step 2.** If the gate passes, the floor's groups are extended from gender × race to gender, age and skin tone,
  with skin tone treated continuously. This is the final model.
  - It gets one seed-7 validation check against the single-axis CESel. It must not be worse on the selection value.
  - It then runs 3 seeds.
- **Fallback.** If the gate fails, or the decision date passes, the paper's detector is **CFES-D**, presented as a
  strong matched-CLIP detector. CESel is reported as a negative result, and the cue-starvation audit stays as the
  mechanism finding.
- **Decision dates.** Gate decision when the screen report exists (expected 26 Oct). Final model frozen **2 Nov 2026**.
  No further amendment of the method after that.

## 2. CESel-MA, complete description
**Input/output.** An aligned 224×224 face crop goes in; P(fake) comes out. No demographic information is used at
test.

**Backbone.** CLIP ViT-L/14 with frozen weights and rank-1 SVD residual adapters (FairReL/Effort style; 0.20 M trainable).
It outputs patch features F (16×16×1024) and the CLS feature z (1024).

**Evidence branches.** Each branch is defined by its support. Each outputs c_k ∈ R^128.
- **Texture:** attention-pooled CLIP patch tokens of the 46 interior face cells (template ellipse cx .50, cy .48,
  ax .30, ay .38) → MLP.
- **Boundary:** the same, over the 44 ring cells at the face border.
- **Semantic:** CLS token → MLP.
- **Frequency:** a small CNN (< 1 M parameters) on the log-amplitude 2-D FFT of the face crop.
  - Remedy, fixed now, because the branch carried near-zero evidence in B0: a 1-epoch branch warm-up with its own
    auxiliary BCE (weight 0.2, removed afterwards).
  - If its validated utility is still not above 0 (bootstrap rule), it is dropped and the model is reported as
    three-cue.

**Head (additive).** s = Σ_k w_kᵀ c_k + b. Per-cue logits s_k = w_kᵀ c_k give exact per-image cue contributions.

**Counterfactual cue removal (head level, no extra backbone pass).**
- Fake with its own registered real source → that source's c_k.
- Other fakes → detached running mean of real cues of the same group.
- Reals → detached running mean of fake cues.
- Margin m(s, y) = y′·asinh(s), with y′ = +1 for fake and −1 for real.
- Utility u_ik = m(s_i, y_i) − m(s_i^(−k), y_i).

**Demographic axes used in training** (labels used in training only):

| Axis | Values | Source |
|---|---|---|
| gender | male / non-male | AI-Face cleaner labels, where available (noisy Xu labels otherwise, flagged) |
| race | Asian / White / Black / Others | same, used only for comparability with Ding |
| age | <30, 30–44, 45–59, 60+ | AI-Face predicted age (FF++/DFDC/CDF/DFD); FF2 per-image age estimate |
| skin tone | continuous ITA (individual typology angle) from lighting-normalised face-skin pixels | our estimator; validated on a stratified rater sample |

**Losses.** L = L_det + 0.5 L_sens + 1.0 L_floor-MA + 0.5 L_inv + 0.1 L_orth.
- **L_det:** balanced BCE on the main batch, source pairs and the group-stratified batch.
- **L_sens:** Σ over validated cues of mean_i relu(τ_k − u_ik). τ_k = the median validation utility, clamped to [0.1, 2.0].
- **L_floor-MA (multi-axis evidence floor).** For each validated cue k and each class c, Σ over axes a ∈ {gender,
  age, skin tone} of the per-group deficit, plus the gender × race intersection kept from CESel:
  - For a discrete axis: Σ_g relu(sg(ū_{k,c}) − ū_{g,k,c} − δ)², over groups with n ≥ 4 in the batch, with δ = 0.1.
  - For skin tone (continuous): ū_k,c(t) = Σ_i w_i(t) u_ik / Σ_i w_i(t), with Gaussian kernel weights
    w_i(t) = exp(−(ITA_i − t)²/2h²) at 8 anchor points t spanning the ITA range (bandwidth h = range/8). The penalty
    is Σ_t relu(sg(ū_{k,c}) − ū_{k,c}(t) − δ)², counted only at anchors whose effective sample size is ≥ 4.
  - Reals are weighted 2× in L_floor-MA, because real-image false positives drive the F_FPR disparity.
- **L_inv:** Huber(asinh s, asinh s′). s′ = the score after replacing the image's demographic subspace component
  (per cue, rank 2, estimated from label-conditioned group means) with that of a same-class partner from another
  group. It counts only when an online probe confirms the swap moved the group, and starts after 200 steps.
  Subspaces are computed for race × gender as in CESel. Skin tone and age are not swapped; they are handled by the
  floor.
- **L_orth:** the FairReL adapter orthogonality term.

**Training.**
- FF++ (Lin split, FF++ frames, the Ding protocol), from the CFES phase-1 checkpoint. 10 × 1,976 steps, AdamW,
  batch 176 (32 real + 32 fake + pairs + stratified 4 per group × class × family cell).
- FF2-public: the same, from a phase-1 checkpoint trained on the frozen public TRAIN only.
- Seeds 7, 17, 27.

**Checkpoint selection (validation only).** Amendment-9 value: val AUC − 0.5·(max − min group FPR), at a fitted
intercept, with eligibility pooled TPR ≥ 80% and FPR ≤ 20%. The group FPR gap is the maximum over the gender, age-bin,
skin-tone-bin and gender × race gaps, among groups with ≥ 2 validation videos.

**Inference threshold.** One global threshold chosen on validation (rule A1). Results are also reported at 0.5.

## 3. Primary endpoint (confirmatory family, Holm-corrected; declared now)
Comparisons: CESel-MA (or CFES-D on fallback) against **(i) the best matched-CLIP baseline** (by its own validation
selection value, among DAW-FDD, DAG-FDD, PG-FDD, Ding SFD+GDA (as released and GDA-connected) and FairReL) and
**(ii) the Ding reproduction**. All 3-seed means, at the A1 threshold.

| # | Hypothesis (ours < comparator) | Test set | Labels |
|---|---|---|---|
| H1 | intersectional (gender × race) F_FPR | FF2-public TEST | FF2 labels (min 405 persons/cell) |
| H2 | max-gap FPR across age bins | FF2-public TEST | FF2 age estimate |
| H3 | max-gap FPR across skin-tone bins (4 ITA bins) | FF2-public TEST | our ITA |
| H4 | intersectional F_FPR, mean over DFDC, Celeb-DF, DFD | cross-domain | AI-Face cleaner labels, min-support rule (≥ 3 real videos per group) |

- **Test.** Paired person/video-cluster bootstrap combined with seed resampling (3 seeds × clusters), 2,000 replicates.
  Holm correction over the 8 tests (4 hypotheses × 2 comparators) at family α 0.05.
- **Non-inferiority guard.** A fairness "win" counts only if pooled TPR at the same threshold is within 3 points of
  the comparator's, and AUC within 1 point.
- **Everything else is secondary/descriptive:** F_DP (with the perfect-detector oracle), es-AUC, TPR gaps,
  equalised-odds, per-family, unseen generators, noisy-label sensitivity, and the per-group cue tables.

## 4. Secondary: FairForensics/GenFace 8-IDS protocol (descriptive only; added 26 Oct ~03:15)
Written before any model was re-scored on these labels. It does not change §1–3: H1–H4, their labels, tests and
Holm family are untouched. Nothing here is used for model or checkpoint selection, or for threshold choice.
- **Why.** FairForensics (arXiv 2608.01661, Sec. IV, Table I) and GenFace report fairness over 8 intersectional
  demographic subgroups (IDS) = sex × age × race, with labels annotated automatically by the FairFace classifier.
  We add that protocol so our numbers can sit next to theirs.
- **Labels.** Official FairFace `res34_fair_align_multi_7_20190809.pt` (sha256 37f2e74c…fdb2c1). Preprocessing is
  predict.py's dlib detection, 5-point alignment, `get_face_chip(300, padding 0.25)` and a 224 resize. The
  deviations (smaller detection resolution, largest face kept, fallbacks) are documented in
  agents/DEMO_AXES/results/fairforensics_ids/README.md.
  - sex = Male if P(male) ≥ 0.5.
  - age2 = 0–29 (Youth) if the argmax 9-bin age is ≤ 20–29, else 30+ (Adult).
  - race2 = White if the argmax of the 7-race head is White, else Non-White.
  - IDS order as in Table I: 0 MYW, 1 FYW, 2 MAW, 3 FAW, 4 MAN, 5 FAN, 6 MYN, 7 FYN.
  - Video sets (FF++, DFDC, Celeb-DF, DFD): each frame takes its video's majority sex, age2 and race2. Frame-level
    labels are a sensitivity analysis.
  - FF2-public: per-image labels.
- **Metrics** (code: agents/DEMO_AXES/code/fairforensics_ids.py). Computed over the 8 IDS at two thresholds: the
  deployed rule (P(fake) ≥ 0.5, CFES-D with its folded FF++-validation intercept) and the model's existing A1
  threshold (stored or re-derived on FF++ validation with Xu labels; never re-chosen with FairFace labels). The
  metrics are F_FPR, F_DP and es-AUC (Ding's Eq. 8–10 forms), the FPR max-min gap, pooled TPR and per-IDS TPR, and,
  as FairForensics also tabulates them, F_MEO (gap form) and F_OAE.
- **Eligibility.**
  - Primary: the min-support rule of §2/H4. On video sets, a group enters the FPR terms only with ≥ 3 real videos
    and the TPR terms only with ≥ 3 fake videos. AUC/F_DP/es-AUC/F_OAE need both. On FF2, the clusters are persons.
  - Reported alongside: Ding's exact rule (every group holding both classes, no minimum).
  - A metric needs ≥ 2 eligible groups, otherwise it is NaN.
- **Models.** CFES-D s7/17/27, arm C s7/17/27, arm E (s7, each w), detection-only control s7/17/27, Ding
  reproduction (SFD+GDA s7, Ori s7/17/27), CESel B0 s7; FF2-public TEST wherever a score file exists. The 4 test
  sets plus FF2-public TEST. Report per-seed values and the seed mean.
- **Inference.** Descriptive only: point estimates, plus video/person-cluster bootstrap 95% intervals for
  transparency. No significance claim, no Holm correction, no "win" language. Group sizes per IDS
  (frames/videos/persons) are reported with every table, and with how far the labels agree with the Xu, AI-Face and
  FF2-manifest labels.

## Amendment A (26 Oct ~05:00; label-source corrections before any final-model run; coordinator decision)
- **Age bins.** The spec named AI-Face predicted age, but AI-Face releases only a 5-class age label, so the declared
  bins (<30 / 30–44 / 45–59 / 60+) cannot be formed from it. Correction (mechanical): the declared bins come from our
  InsightFace age estimate (`age4_if`, ≥ 99.9% coverage). AI-Face's 5 classes, collapsed to 4, and the FairFace
  0–29 / 30+ labels are declared sensitivity axes.
- **H4 label unit.** H4 stays as declared: AI-Face frame-level cleaner labels. Video-majority labels are added as a
  declared sensitivity analysis. DISCLOSED: a descriptive re-scoring of existing models (CFES-D vs the Ding
  reproduction, not the final model) already showed that DFD results depend strongly on the label unit (DEMO_AXES
  MULTI_AXIS_RESULTS.md, §3). That is why the primary is not switched to the more favourable unit.
- **FF++ validation labels.** The AI-Face release has no FF++ validation file, so all validation-time group
  quantities (selection value, gate) use the Xu protocol labels. Flagged in every table.
- **Skin tone (H3).** ITA validity is weak (White-vs-Black AUC 0.72–0.96; 31% of FF2 Black-labelled persons fall in
  the two lighter bins). H3 is tested only if the human skin-tone check on `labels/human_check_sample.csv` gives
  Spearman ≥ 0.6 between ITA and the rater-median Monk tone. Otherwise H3 is reported as descriptive, with no
  skin-tone claim.
- **Screen-arm test metrics.** The multi-axis re-scoring computed test metrics for the finished CESel p1 arms
  (full, sens, inv) before the gate decision. The gate is code-enforced and validation-only (`gate_check.py`). Those
  test rows are not read by the coordinator before `gate.json` exists.
