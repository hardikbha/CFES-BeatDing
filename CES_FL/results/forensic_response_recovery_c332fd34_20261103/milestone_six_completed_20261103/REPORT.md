# Six completed and checkpoint-verified runs; FairForge-2 intervention gate blocked

Observed management UTC: 2026-11-03T04:46:38.364000+00:00. All metrics are VAL, one seed17; no TEST read/evaluation.

Production source remains immutable commit `c332fd34fb603b8919b15fee08e8d022fb9f57c8`, SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`. Actual selected checkpoint bytes, exact config/source provenance, ten epochs, DONE and native independent selected-VAL replay verified for all six runs. Full completion receipts and per-group support are adjacent JSON files.

| Dataset | Arm | Epochs | Selected | VAL AUC (%) | Gender×race F_FPR (fraction, lower) | FPR gap (pp) | Worst-group AUC (%) | FNR (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ffpp | A | 10/10 | 4 | 98.426411 | 0.505265 | 33.839133 | 93.959518 | 4.146708 |
| ffpp | B | 10/10 | 2 | 98.406797 | 0.468638 | 30.416667 | 93.967743 | 4.235935 |
| ffpp | C | 10/10 | 1 | 98.307659 | 0.480534 | 30.984848 | 93.620499 | 3.709965 |
| ffpp | D | 10/10 | 1 | 98.252564 | 0.379345 | 23.839133 | 93.851876 | 3.963558 |
| ff2 | A | 10/10 | 7 | 99.808536 | 0.290986 | 11.660852 | 99.600348 | 0.331368 |
| ff2 | B | 10/10 | 1 | 99.807457 | 0.295459 | 11.028403 | 99.586397 | 0.369737 |

A=weighted original BCE; B=A+soft error disparity; C=B+procedural sensitivity/control stability; D=C+conditional response alignment. D−C isolates the proposed alignment term. A/B versus C/D have different auxiliary forward costs. This is not the older A/ImageNet-versus-B/pretrained comparison.

FF++ D−C: primary F_FPR decreases0.10118894 (21.0576% relative), FPR gap decreases7.145716pp, AUC decreases0.055096pp, worst-group AUC increases0.231377pp, pooled fake FNR increases0.253593pp. Both checkpoints were selected at epoch1 by fixed highest-VAL-AUC selection after completing10epochs; no fairness-based epoch switch. This meets the numerical tentative pilot progression condition (primary lower, AUC loss≤0.5pp, worst-group AUC not collapsed). One seed and no confidence intervals mean this is descriptive evidence, not statistical superiority, causal validity, SOTA, or a win over Ding. Female-Black VAL has only30real/120fake, so one FP changes its FPR by3.33pp.

FF2 A/B detection AUC is nearly identical (99.808536/99.807457%). B worsens primary F_FPR0.290986→0.295459 while reducing FPR gap11.660852→11.028403pp; this mixed outcome must not be advertised as a consistent fairness improvement.

FF++ threshold calibration uses409/4092 validation reals=9.9951124% FPR. FF2 uses243/2438=9.9671862%. One global threshold per selected detector under the same rule; no guarantee of TEST FPR. Native validation reports provide all gender/race/intersection/age/skin/ids8 supports and FPR/FNR. Age and skin inherit manifest proxies. Known FF++ TRAIN/VAL contributor exceptions remain disclosed; no biological person independence certified. No bootstrap or seed-level uncertainty is claimed for this milestone.

### Cross-axis limitations at the fixed AUC-selected checkpoints

| Axis | C F_FPR | D F_FPR | C FPR gap (pp) | D FPR gap (pp) |
|---|---:|---:|---:|---:|
| gender | 0.043699 | 0.053498 | 4.369893 | 5.349798 |
| race | 0.077373 | 0.072905 | 7.296748 | 6.686992 |
| gender_race | 0.480534 | 0.379345 | 30.984848 | 23.839133 |
| age_bin | 0.276733 | 0.303616 | 24.176816 | 29.028932 |
| skin_bin | 0.073475 | 0.051027 | 5.244368 | 3.764232 |

Gender-only and age fairness worsen in D versus C. Age FPR gap rises24.176816→29.028932pp and age worst-group AUC falls97.563609→96.871904%. Skin F_FPR improves, but skin FNR gap rises1.224957→1.601167pp. The intersectional primary improvement therefore does not establish improvement across all demographic axes or error types. Do not select a different epoch to hide these outcomes. No CI or multi-seed significance is claimed.

## Actual FairForge-2 intervention review outcome

Fresh real FF2 CPU preflight decoded249373 TRAIN+VAL images and PASSed; source-bound CUDA smoke and A/B/C/D backward checks PASSed. Those software checks do not certify intervention validity.

The operator actually inspected all88 four-panel previews via22 unscaled per-source composites in this active turn, with source/image/sheet hash verification. Named reviewer is Codex AI visual reviewer, not human annotation. Outcomes:72approved,8rejected,8uncertain, across22 originals/22recorded contributor IDs. Four sources cause16unapproved cases: diagonal rotation-padding borders inside the fixed face-region support, a hat hiding upper face and dominating support, a costume mask, and a hand occluding an already-soft face. Clear border/hat failures are rejected; mask/hand suitability remains uncertain. Other approved cases still carry glasses/accessory/pose/blur shortcut caveats. The recipe is not a validated real-world forgery synthesizer.

Actual documented `qualify` command returned1 at source line840: `ValueError: Substantive named operator review incomplete/rejected; revise recipe under new proposal`. No qualification.json was created. C/D therefore remain **BLOCKED_INPUTS; DO NOT RUN**. No source was removed, coverage reduced, thresholds relaxed or approvals invented. Original pending review and completed failed review are preserved privately. The review recorder/private contact sheets/images are not published.

Next dependency: prepare/review a new versioned intervention-support/eligibility proposal addressing alignment, occlusion, padding and subgroup coverage. Preserve all original BCE rows; do not silently filter intervention sources or alter this pinned recipe. Any changed recipe/configuration needs a fresh source/config-bound preflight/preview/qualification and relevant checks before C/D. No recovery implementation or new scientific arm was launched here.

## Running work, compute and publication

The six new-study arms have finished; no new-study training is running while FF2 C/D are blocked. The separate Ding-architecture FairForge-2 baseline remains live on gpu01 with two verified rank processes, classification epoch19/50 (18 completed); its SFD/corrected-GDA stages remain separate. It was not stopped or restarted. New-study compute is10.058886/24TOTAL GPU-hours, zero outstanding reservations. Both GPU task-held times, D resume and extra idle PBS slot time are included. Existing Ding study is separate from this budget.

Observation helper version3 reports rejected/missing qualifications as BLOCKED_INPUTS instead of PENDING. Four CPU observer gate tests PASS0.001s; this is a read-only status change. Scientific code/configuration and the original coordinator were not changed. Existing observer version2/log retained; version3 keeps the original48hour deadline and10minute cadence. No training, image generation or approval occurs in a publisher.

Live next observations: ../STATUS.md and ../LIVE_STATE.json. All private data/review/weights remain on cluster.
