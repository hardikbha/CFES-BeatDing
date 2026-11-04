# FairForge-2 intersectional validation reanalysis

**Scope:** seed-17 FairForge-2 public VAL only. TEST was not read or evaluated. All aggregate calculations below use the selected-checkpoint validation logits and each model's already frozen VAL threshold; the second view uses fixed probability 0.5 (logit 0). No thresholds or checkpoints were retuned.

## Metric definitions and protocol differences

Ding reports `F_FPR = sum_g |FPR_g - pooled FPR|`, `F_DP = max group prediction rate - min group prediction rate`, and `es-AUC = overall AUC / (1 + sum_g |AUC_g - overall AUC|)`. We use those definitions and report `F_FPR` as percent (100×fraction), matching the paper's display convention. [Ding et al., metric definitions and protocol](https://arxiv.org/html/2511.10150).

The FairForensics paper reports AUC and `F_FPR`, `F_MEO`, `F_DP`, and `F_OAE`, and defines its eight IDS as sex × age (0–29 vs. 30+) × race (White vs. Non-White). It evaluates on balanced GenFace and cross-dataset FF++/DFDC/Celeb-DF settings. We compute that same **group construction** on FF2 VAL, but FF2 has different, highly imbalanced support and automatic age estimates. The paper's accessible text names a *mean* EO metric but gives no equation; therefore `F_MEO_max` below is explicitly our max convention, `max(FPR range, TPR range)`, and is not claimed as an exact FairForensics reproduction. The CVPR 2025 AI-Face benchmark separately calls this `Max Equalized Odds`. [FairForensics](https://arxiv.org/html/2608.01661) · [AI-Face metric and intersection protocol](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_AI-Face_A_Million-Scale_Demographically_Annotated_AI-Generated_Face_Dataset_and_Fairness_CVPR_2025_paper.pdf).

`F_OAE` is max-minus-min group accuracy. `FPR gap` is max-minus-min group FPR. In age×skin tables, unknown skin-tone rows remain in the overall rates but are excluded from the 16 known age-bin×skin-bin groups. Every one of those 16 cells meets the configured 20-real/20-fake support floor. Metrics are not directly comparable across different intersection counts: summed `F_FPR` naturally changes with the number of groups, so read it alongside the max FPR gap and group support.

## Age-bin × skin-bin results

Age bins are `<30`, `30–44`, `45–59`, and `60+`; skin bins are the FF2 ITA proxy bins `0–3`. These are not human-verified age labels or Fitzpatrick skin tones. The 158-real/1,773-fake unknown-skin rows are excluded from the 16 cells but remain in overall rates. Each cell is `real/fake`; the rarest cell has 51 real and 736 fake examples.

| Age bin | Skin 0 | Skin 1 | Skin 2 | Skin 3 |
|---|---:|---:|---:|---:|
| `<30` | 51/1,179 | 64/1,271 | 55/914 | 70/903 |
| `30–44` | 337/4,155 | 261/3,395 | 242/2,823 | 222/2,680 |
| `45–59` | 167/1,598 | 160/1,718 | 159/1,619 | 137/1,427 |
| `60+` | 75/781 | 99/854 | 95/843 | 86/736 |

The table compares the same three verified 50-epoch Xception baselines. `F_FPR`, `F_DP`, `es-AUC`, and `F_OAE` are percentages; `FPR gap` and `F_MEO_max` are percentage points. Higher es-AUC is better; lower other fairness values are better.

| Model | Threshold | Overall AUC % | Overall FPR % | Overall FNR % | F_FPR % | F_DP % | es-AUC % | FPR gap pp | F_MEO_max pp | F_OAE pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | VAL-real 10% | 99.8046 | 9.9672 | 0.3000 | 78.64 | 5.83 | 97.919 | 22.33 | 22.33 | 1.649 |
| DAG-FDD | VAL-real 10% | 99.8136 | 9.9672 | 0.2197 | 74.22 | 6.07 | 97.872 | 20.04 | 20.04 | 1.644 |
| DAW-FDD | VAL-real 10% | 99.8360 | 9.9672 | 0.2442 | 49.75 | 5.83 | 98.429 | 18.30 | 18.30 | 1.104 |
| BCE | Fixed 0.5 | 99.8046 | 9.024 | 0.352 | 75.48 | 5.81 | 97.919 | 22.86 | 22.86 | 1.730 |
| DAG-FDD | Fixed 0.5 | 99.8136 | 7.588 | 0.394 | 60.68 | 6.32 | 97.872 | 19.86 | 19.86 | 1.406 |
| DAW-FDD | Fixed 0.5 | 99.8360 | 8.162 | 0.345 | 51.64 | 6.15 | 98.429 | 19.86 | 19.86 | 1.186 |

DAW-FDD has the best age×skin point estimates of these three baselines on Ding-style F_FPR, es-AUC, FPR gap, F_MEO_max, and F_OAE at the VAL-calibrated threshold. It does **not** have the lowest F_DP. This intersection therefore shows meaningful residual fairness disparity even though overall VAL AUC is high. The large difference between the two threshold views shows why threshold must be fixed in advance and both views reported.

## FairForensics-style sex × age × race groups

We mapped FF2 gender to sex, age `<30` to Youth and all other known age bins to Adult, and White to White versus Asian/Black/Others to Non-White. This is an analogous FF2 VAL grouping, not the same balanced GenFace experiment. One cell—Male/Youth/White—has only **18 real / 348 fake** examples, below our 20-per-class support floor. The main table reports all eight nonempty cells to mirror the group definition; the seven-cell sensitivity view excludes that low-support cell.

| FF IDS cell | Real | Fake |
|---|---:|---:|
| Male / Youth / White | 18 | 348 |
| Female / Youth / White | 29 | 491 |
| Male / Adult / White | 238 | 2,830 |
| Female / Adult / White | 228 | 2,475 |
| Male / Youth / Non-White | 79 | 1,771 |
| Female / Youth / Non-White | 134 | 1,976 |
| Male / Adult / Non-White | 1,102 | 12,107 |
| Female / Adult / Non-White | 610 | 6,671 |

### All eight cells, no minimum-support exclusion

| Model | Threshold | F_FPR % | F_DP % | es-AUC % | F_MEO_max pp | F_OAE pp |
|---|---|---:|---:|---:|---:|---:|
| BCE | VAL-real 10% | 39.90 | 4.29 | 98.625 | 15.24 | 0.824 |
| DAG-FDD | VAL-real 10% | 40.43 | 4.19 | 98.254 | 14.16 | 0.821 |
| DAW-FDD | VAL-real 10% | 46.41 | 4.15 | 98.884 | 16.79 | 0.836 |
| BCE | Fixed 0.5 | 36.97 | 4.32 | 98.625 | 16.14 | 0.908 |
| DAG-FDD | Fixed 0.5 | 34.68 | 4.16 | 98.254 | 12.04 | 1.198 |
| DAW-FDD | Fixed 0.5 | 42.50 | 4.25 | 98.884 | 15.52 | 0.852 |

### Seven-cell sensitivity view (all cells must have ≥20 real and ≥20 fake)

| Model | Threshold | F_FPR % | F_DP % | es-AUC % | F_MEO_max pp | F_OAE pp |
|---|---|---:|---:|---:|---:|---:|
| BCE | VAL-real 10% | 27.64 | 4.29 | 98.965 | 9.47 | 0.824 |
| DAG-FDD | VAL-real 10% | 33.73 | 4.19 | 99.038 | 14.16 | 0.821 |
| DAW-FDD | VAL-real 10% | 34.15 | 4.15 | 99.303 | 16.79 | 0.836 |
| BCE | Fixed 0.5 | 23.77 | 4.31 | 98.965 | 9.59 | 0.908 |
| DAG-FDD | Fixed 0.5 | 25.60 | 4.16 | 99.038 | 11.79 | 1.198 |
| DAW-FDD | Fixed 0.5 | 33.99 | 4.25 | 99.303 | 15.52 | 0.852 |

The change from eight to seven cells is substantial for the summed F_FPR. We should publish both the support table and the fixed eligibility rule, and avoid selecting one view after seeing the scores.

All numbers here are single-seed VAL point estimates. No contributor-cluster confidence intervals have yet been computed for these new intersections; correlated frames must not be treated as independent people.

## How our current runs compare, and what “doing well” means

On the earlier five marginal/intersection axes (gender, race, gender×race, age, skin), at least one CES-FL P/A/B arm has the lowest point F_FPR among our six verified runs on **4 of 5 axes**; the exception is skin, where DAW-FDD is lower. The same 4/5 count holds for point F_DP and es-AUC. These minima come from different CES-FL arms, not one model, and the CES-FL arms ran 10 epochs versus 50 for the Xception fairness baselines. They are descriptive and are not a controlled win.

Among the 50-epoch baselines, DAW-FDD is best in age×skin on the main summed-FPR and max-gap views; it still has an 18.30 pp age×skin FPR range at the calibrated threshold. The 8-way sex×age×race view is unstable for the 18-real cell, as shown by the difference between the all-eight and seven-supported results.

The row-level validation scores for CES-FL P/A/B were not retained in their run folders. As of **2026-11-05 03:42 IST**, validation-only PBS job `45539.mgmt01` is queued with `ngpus=1`; it checks the actual CUDA binding and refuses to run if the assigned GPU has an existing compute process. It will write only private VAL score caches and aggregate outputs. It has not started, so **age×skin and sex×age×race results for CES-FL itself are still pending**. No training or TEST evaluation is attached. The shared 72 GPU-hour ledger remains at 48.808470 hours charged, zero active reservation, 23.191530 uncommitted; this queued job has not been charged.

## Why Ding and FairForensics can look better

- Ding's published framework uses Structural Fairness Decoupling plus Global Distribution Alignment to reduce group information and align group representations. Its paper uses gender, race, and gender×race, with its stated F_FPR/F_DP/es-AUC definitions, and reports FF++ and cross-dataset experiments. Our current CES-FL B is only a soft error-disparity arm; the full validated counterfactual response objective has not run on FF2. [Ding et al.](https://arxiv.org/html/2511.10150)
- FairForensics trains on balanced GenFace and uses expression-aware features, identity-aware patch perturbation, demographic-language guidance, and class-conditional prototype fairness. It evaluates eight sex×age×race groups and cross-dataset transfer. Our FF2 validation data are strongly fake-majority (2,438 real vs 28,669 fake VAL), and our age/skin annotations are proxies. Architecture, training data, group balance, and external-test protocols all differ, so paper scores cannot rank the methods on FF2. [FairForensics](https://arxiv.org/html/2608.01661)

## Major improvements required

1. **Finish the full CES-FL evaluation first.** Wait for PBS 45539 to replay P/A/B on VAL, then add the age×skin and sex×age×race tables with exact group counts, thresholds, and uncertainty. Keep the queued job validation-only.
2. **Target intersection error rates directly.** B's soft error-disparity term improved the age marginal but worsened other marginal axes relative to A. The new age×skin baseline table shows why averaging can hide a large subgroup FPR gap. A future objective should be preregistered against a fixed global VAL threshold and include a support-aware tail-risk term, with separate fake FNR and AUC constraints. Do not tune its weights on TEST.
3. **Do not over-expand intersections without power.** Age×skin has 16 well-supported cells here; sex×age×race has one cell with only18 real examples. Report all cells, mark low-support cells, and show both the full defined grouping and the fixed min-support sensitivity result.
4. **Improve demographic labels.** Age and ITA skin bins are automatic proxies. Validate a stratified human-reviewed subset, quantify agreement and missingness, and treat unknown skin labels explicitly.
5. **Make the comparison controlled.** Match CES-FL and baselines for encoder initialization, update count, backbone, preprocessing, selection, and threshold. Add at least three seeds; cluster-bootstrap by contributor/identity and report both F_FPR and max FPR gap.
6. **Complete the actual CES-FL hypothesis.** P/A/B are BCE/fairness controls. C/D require valid, independently reviewed interventions; do not infer counterfactual validity from realistic appearance and do not claim the full method from P/A/B.
7. **Keep TEST sealed** until the candidate, metrics, subgroup eligibility, and utility/fairness decision rule are frozen. Validation findings are not a win over Ding or FairForensics.

The machine-readable raw validation caches and replay logs remain private on the cluster. This report contains aggregate values only.
