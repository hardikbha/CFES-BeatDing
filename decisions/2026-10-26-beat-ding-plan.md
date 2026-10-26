# Beat-Ding plan (priority 1), pre-registration, 26 Oct 2026 ~04:30

User (26 Oct, effort max): "first of all we need to beat Ding's numbers ... priority one; thereafter our own
dataset". This file is fixed BEFORE any run of the recipe below exists. The FF2 work moves to the end of the queue.

## 0. Target
Ding et al. CVPR'26 published Tables 1 and 2: FF++ intra-domain, and DFDC / Celeb-DF / DFD cross-domain, Xception.
That is 4 test sets × 10 columns = 40 columns (g/r/i F_FPR, F_DP, es-AUC, and AUC), with the same Lin split, the same
Xu labels, the same metric code (`cfes/metrics.py`) and Ding's numbers as printed. Table 3 (ResNet-50) needs a
user decision (CLIP-only was decided on 25 Oct).

## 1. Diagnosis (saved scores, CPU; R/agents/BEAT_DING/results/, code/)
**Ding's own operating point.** Our reproduction at 0.5 has FF++ val TPR 94.18 / FPR 13.10. Its test FPRs:
- FF++ 12.5
- DFDC 49
- Celeb-DF 70
- DFD 59

Cross-domain, Ding flags most images, which compresses F_FPR and F_DP.

**Best existing model at a matched operating point.** Arm E w3 s7, with the threshold set on FF++ val to the same
TPR (94.18), beats Ding published on **19/40** columns:

| Test set | Columns won |
|---|---|
| FF++ | 0/10 |
| DFDC | 7/10 |
| Celeb-DF | 5/10 |
| DFD | 7/10 |

Our Ding reproduction itself beats the published numbers on only 7/40.

**FF++ in-domain.**
- **AUC:** 97.06 vs 97.71.
- **Intersectional F_FPR:** 34.7 vs 20.18. The excess comes from 3–4 real non-male-Asian videos (youtube 062, 930, 985,
  616). These are TV news anchors with heavy makeup, airbrushed skin, over-exposure and broadcast compression, flagged
  on 55–93% of their frames. Male-Asian NeuralTextures fakes (235_194, 259_345) are missed by all models,
  including Ding's.
- **Noise floor.** If group labels are permuted across real videos, a model with no group effect scores FF++ i-F_FPR
  around 40 (90% range 24–63) at 12.5% FPR. Our models sit at that null (34.5, 44.5). Ding's reproduction is below its
  5th percentile (23.8), so Ding's method actively equalises groups.

**F_DP is structurally conflicted with accuracy cross-domain.** A perfect detector scores 5.1 / 43.2 / 59.6 / 33.1.
Ding's low cross-domain F_DP comes from near-random flagging. Our F_DP improves only at high TPR, where positive
rates saturate. Our F_DP losses on DFDC and DFD are reported with this oracle; no tuning against F_DP.

**No photometric augmentation exists anywhere in our training** (cfes/data.py; configs). Nothing teaches the CLIP
detector that skin smoothness, makeup, exposure or compression are not forgery evidence.

## 2. Recipe "BD" (built on Arm E = CFES-D + group-FPR-gap; code in a separate copy, R/agents/BEAT_DING)

**A. Label-preserving processing augmentation (both classes).** With p = 0.6, one op per image:
- JPEG q U[30, 90];
- bilateral "beauty" smoothing d ∈ {5, 7, 9}, σ_c, σ_s U[20, 80];
- Gaussian blur σ U[0.4, 1.6];
- down-up scale s U[0.4, 0.9] (bicubic);
- gamma U[0.6, 1.6], brightness/contrast ±25%, saturation ±30%;
- unsharp mask.

Also horizontal flip with p = 0.5. The same op and parameters apply to a (fake, own source) pair, so pairs stay
valid.

**B. Equalised score-distribution alignment (ESA), class-conditional, multi-quantile.**
- **Groups:** gender, race and intersection, each with n ≥ 4 in the batch.
- **Quantiles:** reals q ∈ {0.5, 0.75, 0.9, 0.95} (the FPR-relevant upper tail); fakes q ∈ {0.05, 0.1, 0.25, 0.5}.
- **Loss:** Σ_levels Σ_g Σ_q |Q_g(q) − sg(Q_pool(q))| on asinh(score), averaged over cells. Weight 1.0.
- **Coverage:** applied to all images in the step (clean, augmented and nuisance views).
- **Sampler:** the stratified sampler rises to 4 per (label × intersection group) cell, i.e. 64 images.

**C. Multi-layer evidence.**
- Patch tokens from CLIP blocks 8, 12, 16 and 20 each go through LN → Linear 1024→256 → GELU → Linear → patch logit.
- These are averaged into a second evidence map, fused like the existing patch map (top-8 + local norm), with its
  own gate initialised at 0.
- Adapters are unchanged.

**D. (optional arm) Group-AUC floor.** Soft pairwise AUC per group (sigmoid of fake − real, τ = 1 on asinh scores).
Loss Σ_g relu(sg(AUC_pool) − AUC_g), weight 0.5. Targets es-AUC.

**Base E weight.** Chosen among the existing Arm E runs w ∈ {1, 3, 10} (seed 7) by the validation rule in §4.
DISCLOSED: that rule is introduced after their test results were seen.

## 3. Screen (seed 7)
- **Start and budget:** from CFES phase-1 runs/clip_s7_p1/last.pt; 10 × 1976 steps; the same optimiser as Arm E.
- **Arms:**

  | Arm | Components |
  |---|---|
  | BD1 | base + A |
  | BD2 | base + A + B |
  | BD3 | base + A + B + C |
  | BD4 | base + A + B + C + D |
  | BD0 | base + B (no augmentation), isolates A |

- **References:** existing D s7, E w1/w3/w10 s7, and the Ding reproduction.
- **Order:** these runs go first in the priority queue once the CESel screen ends. CESel-MA, matched baselines, Ding
  and Arm E seeds come after. FF2 is last.

## 4. Operating point and selection (validation only)
**Operating point (all BD models, fixed now).** One global threshold per model, chosen on FF++ val so that val TPR =
94.18%, the Ding reproduction's val TPR at 0.5 ("matched sensitivity"). Our calibrated P(fake) = 0.5 sits at that
threshold. Every table also reports TPR/FPR, the native class-balanced calibration and the A1 threshold.

**Within-run checkpoint selection.** Val AUC + val intersectional es-AUC, both threshold-free. This is less noisy
than threshold metrics on tiny val groups.

**Arm comparison: the "val Ding-score".** On FF++ val at the matched threshold, count Ding's 10 columns where the arm
beats the Ding reproduction's FF++ val columns. Ties are broken by val AUC.
- The winner goes to seeds 17 and 27.
- Promotion requires the winner's val Ding-score ≥ the best existing reference's.
- If no BD arm beats the best existing reference, report that, and the paper uses the best reference.

## 5. Success (reported as is; test never chooses)
**Primary.** The 3-seed mean of the promoted recipe at the matched operating point, against Ding published, on the 40
columns. The claim "beats Ding" requires:
- ≥ 25/40 columns;
- AUC and all three es-AUC columns better on all four test sets;
- intersectional F_FPR better on ≥ 3 of the 4 test sets.

**Reporting.** Every column is reported, including losses, with seed SDs, TPR/FPR and the F_DP oracle. The FF++
F_FPR noise floor (video-permutation null) is reported next to the FF++ F_FPR.

## Amendment 1 (base weight), 26 Oct 2026 ~03:45, written before any BD run exists
**Rule** (section 2 "Base E weight", applied as in section 4). VALIDATION only. For each existing seed-7 run, one
threshold on FF++ val such that val TPR = the Ding reproduction's val TPR at 0.5. That reproduction
(DINGREPRO_LIN `full_s7_ep49_ffval.npz`, 21,294 fakes / 4,092 reals) has val TPR **94.1768%** and FPR 13.0987%.
The threshold is t = quantile of the run's val fake logits at 1 − 0.941768 (the `code/matched_op.py` rule). The
val Ding-score counts Ding's 10 columns on FF++ val where the run strictly beats the reproduction's val columns at
its 0.5. Ties go to val AUC. Code: `agents/BEAT_DING/scripts/bd_wstar.py`; output `results/wstar.json`.

Ding reproduction, FF++ val at 0.5:

| gF_FPR | gF_DP | g es-AUC | rF_FPR | rF_DP | r es-AUC | iF_FPR | iF_DP | i es-AUC | AUC |
|---|---|---|---|---|---|---|---|---|---|
| 3.94 | 1.79 | 94.77 | 24.25 | 3.48 | 86.60 | 56.37 | 12.88 | 80.91 | 97.11 |

| Run (seed 7) | val Ding-score | val AUC | val FPR at matched t | columns won |
|---|---|---|---|---|
| Arm E w1 | **6/10** | 96.883 | 13.17 | gF_FPR, gF_DP, rF_FPR, r es-AUC, iF_FPR, i es-AUC |
| Arm E w3 | 5/10 | 97.014 | 13.66 | gF_FPR, rF_FPR, r es-AUC, iF_FPR, i es-AUC |
| Arm E w10 | 6/10 | 96.561 | 17.40 | gF_FPR, gF_DP, rF_FPR, r es-AUC, iF_FPR, i es-AUC |
| CFES-D w0.3 (reference) | 4/10 | 96.067 | 21.19 | gF_DP, rF_FPR, r es-AUC, i es-AUC |

**w\* = 1.** It ties w = 10 at 6/10 and wins on val AUC (96.88 vs 96.56). BD0–BD4 therefore use
`loss.fprgap.weight = 1`.

**Promotion bar (section 4).** The best existing reference has val Ding-score 6/10 (Arm E w1 s7 and w10 s7). A BD
winner is promoted to seeds 17 and 27 only if its val Ding-score is ≥ 6.

**DISCLOSED.** This rule was introduced after the test results of these runs were seen. At the matched operating
point, total test wins against Ding published were: E w1 18/40, E w3 19/40, E w10 18/40, CFES-D s7 16/40
(`results/matched_op.json`). The validation rule does not pick w3, the run with the most test wins.

## Amendment 2 (26 Oct ~05:00, user: "Yes, run BD on ResNet-50"; before any BD run)
Table 3 (ResNet-50) is now in scope. After the CLIP winner is fixed by the seed-7 screen, the SAME winning components
(A/B/C/D as selected; C adapted to ResNet as multi-stage evidence from layer2/layer3/layer4 maps) are trained on
Ding's ResNet-50 ('adjust_channel', ImageNet init; our resnet phase-1 checkpoints runs/resnet_s{7,17,27}_p1) with the
CFES-ResNet base + Arm E group-FPR-gap at w* = 1, seeds 7/17/27, same selection rule (val AUC + val i-es-AUC), same
matched operating point. No Ding ResNet reproduction exists, so the matched TPR target stays 94.18% (the Xception
reproduction's val TPR) — disclosed proxy. Success for Table 3 is reported separately with the same column rules
(>= 25/40 etc.) against Ding's published Table 3. Queue position: right after the CLIP winner's seeds 17/27, before
CESel-MA / baselines / FF2.

## Amendment 3 (26 Oct ~15:35, user: "yes run resnet results ahead of clip seeds 17 and 27"; before the screen ends)
Queue order only: the ResNet-50 Table-3 runs (bd_resnet_s7/s17/s27 + bd_resnet_final) are moved ahead of the CLIP
winner's seeds 17/27 (bd_winner_s17/s27). Nothing else changes: the ResNet runs still start only after the CLIP seed-7
screen (bd_screen) has fixed the winning components under the section-4 rule (Amendment 2); the same CLIP winner seeds
17/27 still follow. Implemented by task order in agents/BEAT_DING/run/bd_tasks.json (the priority runner breaks ties
within an include by file order). No result of any BD arm on test exists at this point.

## Amendment 4 (26 Oct ~16:50, user: "Okay we can do so" to adding Xception-vs-Xception runs first; "CLIP's 3-seed experiment will be done later")
Purpose: apples-to-apples with Ding's Tables 1-2 (Xception) and Table 3 (ResNet-50). New queue order after the CLIP
seed-7 screen (bd_screen) fixes the winning components (section 4 rule, unchanged):
 1. Xception: the winning BD components on Ding's Xception backbone, seeds 7/17/27, compared with Ding published
    Tables 1-2 AND with our reproduction of Ding's official Xception code (DINGREPRO_LIN), at the same matched operating
    point (val TPR 94.18 %), plus 0.5 / native / A1 reported.
 2. ResNet-50 (Amendments 2-3), seeds 7/17/27, vs Ding Table 3.
 3. CLIP winner seeds 17/27 (deferred, run later).
Training recipe for the Xception arm is fixed BEFORE any Xception BD run, in an implementation note appended here as
Amendment 4b: it must match Ding's official Xception training as closely as our pipeline allows (initialisation,
input size/normalisation, optimiser and learning rate, epochs/steps, which layers train, BN handling), with every
deviation listed; the earlier CFES-Xception phase-2 failures (diverged last-block run, broken BN stats) must be
addressed explicitly. Success for Xception uses the same column rules as section 5. No test result of any Xception BD
run exists at this point.

## Amendment 4a (26 Oct ~18:05): user decisions for the Xception arm; recorded before any Xception run
The design is agents/BEAT_DING/XCEPTION_DESIGN.md, sections 3-4. The user decided ("D1 Ding phase-1 checkpoints, D2
option b, D3 yes"):
- **D1 = X0.**
  - Each Xception BD seed starts from Ding's official phase-1 checkpoint as run by us:
    agents/DINGREPRO_LIN/results/DING_base_s{7,17,27}/49.pth.
  - This reverses the 16 Oct "never load Ding's checkpoints" rule for the Xception arm only. Disclosed.
- **D2 = (b).** Ding's full method (SFD + released GDA), seeds 17 and 27, from the same DING_base checkpoints.
  - The code is Ding's released training step unchanged, except for one disclosed edit: the per-step DataLoader
    re-creation is replaced by persistent iterators. Same batch composition, same sampling distribution, same loss,
    same optimiser, same 50 epochs, same final-epoch checkpoint.
  - Purpose: the paired 3-seed reproduction reference (E2). Ding s7 stays the released, unmodified run.
  - Every comparison table marks s17/s27 as "fast-loader".
- **D3 = yes.**
  - Run the X-base control: the BD base (Arm E on Xception) without components A-D, seed 7.
  - The X-recal control is the epoch −1 (step-0) score of every run, at no extra cost.
- **Queue order.**
  - The Xception smoke and Ding s17/s27 may start immediately, because they do not depend on the CLIP screen.
  - X-BD s7/s17/s27 and X-base s7 start after bd_screen fixes the winning components.
  - Then ResNet (Amendments 2-3), then CLIP winner s17/s27 (deferred).
- The success rules (section 5) and selection (section 4) are unchanged.
- The full recipe and deviation list is written as Amendment 4b by the implementation, before any Xception BD run.
