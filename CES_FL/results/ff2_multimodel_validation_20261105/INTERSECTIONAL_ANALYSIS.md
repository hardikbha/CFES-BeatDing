# FairForge-2 intersectional validation reanalysis

**Scope:** seed-17 FairForge-2 public VAL only. TEST was not read or evaluated. All aggregate calculations below use the selected-checkpoint validation logits and each model's already frozen VAL threshold; the second view uses fixed probability 0.5 (logit 0). No thresholds or checkpoints were retuned.

## Metric definitions and protocol differences

Ding reports `F_FPR = sum_g |FPR_g - pooled FPR|`, `F_DP = max group prediction rate - min group prediction rate`, and `es-AUC = overall AUC / (1 + sum_g |AUC_g - overall AUC|)`. We use those definitions and report `F_FPR` as percent (100×fraction), matching the paper's display convention. [Ding et al., metric definitions and protocol](https://arxiv.org/html/2511.10150).

The FairForensics paper reports AUC and `F_FPR`, `F_MEO`, `F_DP`, and `F_OAE`, and defines its eight IDS as sex × age (0–29 vs. 30+) × race (White vs. Non-White). It evaluates on balanced GenFace and cross-dataset FF++/DFDC/Celeb-DF settings. We compute that same **group construction** on FF2 VAL, but FF2 has different, highly imbalanced support and automatic age estimates. The paper's accessible text names a *mean* EO metric but gives no equation; therefore `F_MEO_max` below is explicitly our max convention, `max(FPR range, TPR range)`, and is not claimed as an exact FairForensics reproduction. The CVPR 2025 AI-Face benchmark separately calls this `Max Equalized Odds`. [FairForensics](https://arxiv.org/html/2608.01661) · [AI-Face metric and intersection protocol](https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_AI-Face_A_Million-Scale_Demographically_Annotated_AI-Generated_Face_Dataset_and_Fairness_CVPR_2025_paper.pdf).

`F_OAE` is max-minus-min group accuracy. `FPR gap` is max-minus-min group FPR. In age×skin tables, unknown skin-tone rows remain in the overall rates but are excluded from the 16 known age-bin×skin-bin groups. Every one of those 16 cells meets the configured 20-real/20-fake support floor. Metrics are not directly comparable across different intersection counts: summed `F_FPR` naturally changes with the number of groups, so read it alongside the max FPR gap and group support.

### Metric inventory

The current aggregate uses **8 headline metric types**: 3 detection metrics (overall AUC, FPR, FNR) and 5 fairness metrics (F_FPR, F_DP, es-AUC, F_MEO_max, F_OAE). It also records 3 diagnostics (FPR gap, TPR gap, worst-group AUC) plus subgroup counts and rates. There are 7 distinct group definitions—gender, race, gender×race, age, skin, age×skin, and sex×age×race—plus a separate seven-cell minimum-support sensitivity view for the eight-cell sex×age×race partition.

Across seven verified models and two threshold views, the machine-readable table has 112 model×grouping×threshold rows: 560 fairness values plus 42 unique overall detection values (the overall values repeat by grouping), with diagnostic and group-support values alongside. All 35 recomputations for the original five axes matched their `RUN_VERIFIED` receipts. See [METRICS_LONG_7RUNS.csv](METRICS_LONG_7RUNS.csv), [GROUP_SUPPORT_7RUNS.csv](GROUP_SUPPORT_7RUNS.csv), and [SUMMARY_7RUNS.json](SUMMARY_7RUNS.json). The earlier six-model snapshot remains in `METRICS_LONG.csv` and its paired files.

## Age-bin × skin-bin results

Age bins are `<30`, `30–44`, `45–59`, and `60+`; skin bins are the FF2 ITA proxy bins `0–3`. These are not human-verified age labels or Fitzpatrick skin tones. The 158-real/1,773-fake unknown-skin rows are excluded from the 16 cells but remain in overall rates. Each cell is `real/fake`; the rarest cell has 51 real and 736 fake examples.

| Age bin | Skin 0 | Skin 1 | Skin 2 | Skin 3 |
|---|---:|---:|---:|---:|
| `<30` | 51/1,179 | 64/1,271 | 55/914 | 70/903 |
| `30–44` | 337/4,155 | 261/3,395 | 242/2,823 | 222/2,680 |
| `45–59` | 167/1,598 | 160/1,718 | 159/1,619 | 137/1,427 |
| `60+` | 75/781 | 99/854 | 95/843 | 86/736 |

The table compares all seven verified FF2 result configurations. `F_FPR`, `F_DP`, `es-AUC`, and `F_OAE` are percentages; `FPR gap` and `F_MEO_max` are percentage points. Higher es-AUC is better; lower other fairness values are better.

| Model | Threshold | Overall AUC % | Overall FPR % | Overall FNR % | F_FPR % | F_DP % | es-AUC % | FPR gap pp | F_MEO_max pp | F_OAE pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | VAL-real 10% | 99.8046 | 9.9672 | 0.3000 | 78.64 | 5.83 | 97.919 | 22.33 | 22.33 | 1.649 |
| DAG-FDD | VAL-real 10% | 99.8136 | 9.9672 | 0.2197 | 74.22 | 6.07 | 97.872 | 20.04 | 20.04 | 1.644 |
| DAW-FDD | VAL-real 10% | 99.8360 | 9.9672 | 0.2442 | 49.75 | 5.83 | 98.429 | 18.30 | 18.30 | 1.104 |
| CES-FL P | VAL-real 10% | 99.8105 | 9.9672 | 0.3802 | 75.00 | 5.71 | 98.591 | 23.75 | 23.75 | 1.919 |
| CES-FL A | VAL-real 10% | 99.8085 | 9.9672 | 0.3314 | 69.21 | 5.85 | 98.415 | 19.72 | 19.72 | 1.508 |
| CES-FL B | VAL-real 10% | 99.8075 | 9.9672 | 0.3697 | 70.98 | 5.80 | 98.481 | 22.33 | 22.33 | 1.713 |
| Ding corrected GDA | VAL-real 10% | 99.0594 | 9.9672 | 1.0743 | 47.54 | 8.83 | 91.820 | 14.19 | 14.19 | 3.243 |
| BCE | Fixed 0.5 | 99.8046 | 9.024 | 0.352 | 75.48 | 5.81 | 97.919 | 22.86 | 22.86 | 1.730 |
| DAG-FDD | Fixed 0.5 | 99.8136 | 7.588 | 0.394 | 60.68 | 6.32 | 97.872 | 19.86 | 19.86 | 1.406 |
| DAW-FDD | Fixed 0.5 | 99.8360 | 8.162 | 0.345 | 51.64 | 6.15 | 98.429 | 19.86 | 19.86 | 1.186 |
| CES-FL P | Fixed 0.5 | 99.8105 | 3.117 | 0.991 | 34.03 | 6.39 | 98.591 | 12.86 | 12.86 | 1.455 |
| CES-FL A | Fixed 0.5 | 99.8085 | 4.061 | 0.774 | 39.86 | 6.32 | 98.415 | 15.71 | 15.71 | 1.634 |
| CES-FL B | Fixed 0.5 | 99.8075 | 4.389 | 0.753 | 41.76 | 6.37 | 98.481 | 17.14 | 17.14 | 1.634 |
| Ding corrected GDA | Fixed 0.5 | 99.0594 | 0.656 | 4.060 | 11.33 | 13.69 | 91.820 | 8.57 | 8.57 | 7.659 |

DAW-FDD has the best age×skin point estimates among BCE/DAG/DAW on VAL-threshold F_FPR, es-AUC, FPR gap, F_MEO_max, and F_OAE; CES-FL A is second on F_FPR and FPR gap. Adding Ding corrected GDA changes the point winner for summed F_FPR and max gap: Ding GDA is 47.54% / 14.19 pp, versus DAW's 49.75% / 18.30 pp. But Ding GDA has lower es-AUC (91.820% vs DAW 98.429%), higher F_DP and OAE, lower overall AUC, and higher FNR. CES-FL P has the best F_DP and es-AUC point estimates in this axis. No model dominates all columns. The fixed-0.5 view changes the operating point substantially, so both threshold views must be shown.

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
| CES-FL P | VAL-real 10% | 38.23 | 4.00 | 98.909 | 14.69 | 0.994 |
| CES-FL A | VAL-real 10% | 47.38 | 4.13 | 98.843 | 20.07 | 1.102 |
| CES-FL B | VAL-real 10% | 39.56 | 4.02 | 98.816 | 14.78 | 0.678 |
| Ding corrected GDA | VAL-real 10% | **18.83** | 5.12 | 93.409 | **8.62** | 2.195 |
| BCE | Fixed 0.5 | 36.97 | 4.32 | 98.625 | 16.14 | 0.908 |
| DAG-FDD | Fixed 0.5 | 34.68 | 4.16 | 98.254 | 12.04 | 1.198 |
| DAW-FDD | Fixed 0.5 | 42.50 | 4.25 | 98.884 | 15.52 | 0.852 |
| CES-FL P | Fixed 0.5 | 25.30 | 4.12 | 98.909 | 9.14 | 1.812 |
| CES-FL A | Fixed 0.5 | 25.48 | 3.94 | 98.843 | 8.12 | 0.802 |
| CES-FL B | Fixed 0.5 | 29.47 | 3.98 | 98.816 | 13.49 | 1.089 |
| Ding corrected GDA | Fixed 0.5 | **9.22** | 7.98 | 93.409 | **5.56** | 3.852 |

### Seven-cell sensitivity view (all cells must have ≥20 real and ≥20 fake)

| Model | Threshold | F_FPR % | F_DP % | es-AUC % | F_MEO_max pp | F_OAE pp |
|---|---|---:|---:|---:|---:|---:|
| BCE | VAL-real 10% | 27.64 | 4.29 | 98.965 | 9.47 | 0.824 |
| DAG-FDD | VAL-real 10% | 33.73 | 4.19 | 99.038 | 14.16 | 0.821 |
| DAW-FDD | VAL-real 10% | 34.15 | 4.15 | 99.303 | 16.79 | 0.836 |
| CES-FL P | VAL-real 10% | 25.98 | 4.00 | 99.383 | 9.71 | 0.645 |
| CES-FL A | VAL-real 10% | 29.57 | 4.13 | 99.219 | 9.53 | 0.854 |
| CES-FL B | VAL-real 10% | 27.30 | 4.02 | 99.318 | 9.80 | 0.629 |
| Ding corrected GDA | VAL-real 10% | **14.42** | 5.12 | 95.982 | **7.85** | 1.816 |
| BCE | Fixed 0.5 | 23.77 | 4.31 | 98.965 | 9.59 | 0.908 |
| DAG-FDD | Fixed 0.5 | 25.60 | 4.16 | 99.038 | 11.79 | 1.198 |
| DAW-FDD | Fixed 0.5 | 33.99 | 4.25 | 99.303 | 15.52 | 0.852 |
| CES-FL P | Fixed 0.5 | 25.30 | 4.12 | 99.383 | 9.14 | 1.812 |
| CES-FL A | Fixed 0.5 | 25.48 | 3.94 | 99.219 | 8.12 | 0.802 |
| CES-FL B | Fixed 0.5 | 29.47 | 3.98 | 99.318 | 13.49 | 1.089 |
| Ding corrected GDA | Fixed 0.5 | **4.32** | 7.98 | 95.982 | **4.29** | 3.820 |

The change from eight to seven cells is substantial for the summed F_FPR. Ding corrected GDA has the lowest F_FPR and max EO gap in both views, while CES-FL P has the best F_DP/es-AUC point estimates and CES-FL B the best OAE. Publish both the support table and the fixed eligibility rule; do not select one view after seeing the scores.

All numbers here are single-seed VAL point estimates. No contributor-cluster confidence intervals have yet been computed for these new intersections; correlated frames must not be treated as independent people.

## What the added intersections change

- **Age×skin (16 cells):** Ding corrected GDA has the best calibrated F_FPR (47.54%) and max FPR gap (14.19 pp), narrowly ahead of DAW-FDD (49.75%, 18.30 pp), but its es-AUC (91.82%) and OAE gap (3.24 pp) are much worse. Among CES-FL runs, A is best on age×skin F_FPR/gap; P is best on F_DP/es-AUC. B is worse than A on age×skin F_FPR (70.98 vs. 69.21%) and gap (22.33 vs. 19.72 pp), despite improving the marginal age score.
- **Sex×age×race (8 cells, all included):** Ding corrected GDA has the lowest F_FPR (18.83%) and max EO gap (8.62 pp), while CES-FL P has the best F_DP (4.00%) and es-AUC (98.909%), and CES-FL B the best OAE (0.678 pp). This is not stable across support rules: the low-support Male/Youth/White cell changes F_FPR/EO values when excluded.
- **CES-FL A vs. B:** B improved some sex×age×race metrics versus A, but worsened the age×skin F_FPR/max-gap/accuracy-gap. The result depends on the intersection definition; B has not shown universal fairness improvement.
- **Threshold matters:** at fixed 0.5, the age×skin F_FPR changes materially and overall FPR/FNR shift. The VAL-real10% view is the registered operating point; fixed 0.5 is a sensitivity view, not a replacement threshold.

## How our current runs compare, and what “doing well” means

On the earlier five marginal/intersection axes (gender, race, gender×race, age, skin), Ding corrected GDA has the lowest point F_FPR on gender, race, gender×race, and age; DAW-FDD is lowest on skin. CES-FL P/A/B still have the lowest point F_DP on four of five axes and the highest es-AUC on four of five axes, but these wins are split across different CES-FL arms. They ran 10 epochs versus 50 for the baselines. These are descriptive one-seed point estimates, not a controlled win.

The new intersections show no universal winner: Ding corrected GDA is better on FPR disparity, CES-FL P on es-AUC/DP for some groupings, and DAW-FDD on age×skin accuracy gap. The 8-way sex×age×race view is sensitive to the 18-real Male/Youth/White cell; the age×skin cells are all supported.

The row-level validation scores were not retained in the CES-FL P/A/B run folders, so we re-scored their selected checkpoints on VAL only. Before inference, the 31,107 VAL image bytes were rehashed against the saved selected-BCE validation inventory; all matched. The Ding corrected-GDA `validation_best.npz` logits were checked against its VAL receipt and the five saved-axis F_FPR values before intersection aggregation. Each cache records the exact VAL manifest hash, selected checkpoint hash, frozen threshold, and `test_access=false`. No model selection or training was done.

PBS job `45539.mgmt01` was rejected before allocation because the PBS `gpu` queue running-job limit was reached; it consumed no GPU time. CES-FL P/A/B replays then ran on gpu02 under a shared-pool-v2 UUID lease (`GPU-3a2f4cf3-9b32-14c9-3deb-29e55d1123bb`), with CUDA UUID and memory verified and no exclusive ownership claimed. The shared FF2 ledger records 0.042914 additional GPU-hours for the successful and failed replay sessions combined. No TEST access or training occurred.

## Why Ding and FairForensics can look better

- Ding's published framework uses Structural Fairness Decoupling plus Global Distribution Alignment to reduce group information and align group representations. Its paper uses gender, race, and gender×race, with its stated F_FPR/F_DP/es-AUC definitions, and reports FF++ and cross-dataset experiments. Our current CES-FL B is only a soft error-disparity arm; the full validated counterfactual response objective has not run on FF2. [Ding et al.](https://arxiv.org/html/2511.10150)
- FairForensics trains on balanced GenFace and uses expression-aware features, identity-aware patch perturbation, demographic-language guidance, and class-conditional prototype fairness. It evaluates eight sex×age×race groups and cross-dataset transfer. Our FF2 validation data are strongly fake-majority (2,438 real vs 28,669 fake VAL), and our age/skin annotations are proxies. Architecture, training data, group balance, and external-test protocols all differ, so paper scores cannot rank the methods on FF2. [FairForensics](https://arxiv.org/html/2608.01661)

## Major improvements required

1. **Add uncertainty to the full CES-FL evaluation.** P/A/B VAL intersections are now computed. Run contributor-cluster bootstrap intervals on the fixed saved scores before making claims; the estimates remain single-seed point estimates.
2. **Target intersection error rates directly.** B's soft error-disparity term improved the age marginal but worsened other marginal axes relative to A. The new age×skin baseline table shows why averaging can hide a large subgroup FPR gap. A future objective should be preregistered against a fixed global VAL threshold and include a support-aware tail-risk term, with separate fake FNR and AUC constraints. Do not tune its weights on TEST.
3. **Do not over-expand intersections without power.** Age×skin has 16 well-supported cells here; sex×age×race has one cell with only18 real examples. Report all cells, mark low-support cells, and show both the full defined grouping and the fixed min-support sensitivity result.
4. **Improve demographic labels.** Age and ITA skin bins are automatic proxies. Validate a stratified human-reviewed subset, quantify agreement and missingness, and treat unknown skin labels explicitly.
5. **Make the comparison controlled.** Match CES-FL and baselines for encoder initialization, update count, backbone, preprocessing, selection, and threshold. Add at least three seeds; cluster-bootstrap by contributor/identity and report both F_FPR and max FPR gap.
6. **Complete the actual CES-FL hypothesis.** P/A/B are BCE/fairness controls. C/D require valid, independently reviewed interventions; do not infer counterfactual validity from realistic appearance and do not claim the full method from P/A/B.
7. **Keep TEST sealed** until the candidate, metrics, subgroup eligibility, and utility/fairness decision rule are frozen. Validation findings are not a win over Ding or FairForensics.

The machine-readable raw validation caches and replay logs remain private on the cluster. This report contains aggregate values only.
