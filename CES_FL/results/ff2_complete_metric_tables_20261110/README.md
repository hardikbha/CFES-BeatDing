# FF2 complete demographic and detection results

**No seed tops every metric.** This packet expands the completed 21-run comparison across every saved axis. It preserves all methods and seeds rather than selecting favorable cells.

- [Download Excel workbook](FF2_COMPLETE_RESULTS.xlsx): filtered/frozen tables, all seeds, all axes, means/SDs, metric leaders, group support and family results. Green cells mark best/tied values within each seed/axis/metric.
- [Read every metric table](ALL_METRICS.md), [full numeric CSV](ALL_METRICS_LONG.csv), [three-seed means/SDs](THREE_SEED_MEANS_ALL_METRICS.csv), [group rates/support](GROUP_RATES_AND_SUPPORT.csv).
- [Metric leaders](BEST_PER_METRIC.csv), [leader summary](LEADER_SUMMARY.csv), [family results](FAMILY_RESULTS.csv). Leader counts include correlated metrics and ties; they are not a scientific win count or statistical test.

Age and skin-tone metrics exist. Skin tone is the color-related axis; no separate color metric was computed. IDS8 represents the separate eight supplied gender×age×race cells. There is no native metric named FOV; accuracy-gap fairness (F_OAE) is included.

All 21 runs share the training/evaluation recipe, with declared objective/SFD differences. Ding is the controlled adaptation. Each threshold was frozen from SELECT-real calibration. All ramp/low selected epoch 2, before response losses activate. Three-seed SD is conditional on one shared pretrained encoder; no significance claim or independent-pretraining replication. Age/skin are automatic estimates.

The tables below show only 100F_FPR by axis (lower is better). Bold uses full precision; visually equal rounded numbers need not be ties. The workbook and full Markdown include F_DP, es-AUC, FPR/FNR/AUC gaps, MEO-max, accuracy gap, worst-group AUC and overall AUC/FPR/FNR.

## Seed 7

| Method | Gender | Race | Gender x race | Age | Skin tone | IDS8 |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 3.8201 | 35.2290 | 71.6997 | 15.1656 | 24.3755 | 48.4823 |
| DAG-FDD | 3.8524 | 32.7519 | 66.5257 | 13.3776 | 22.0298 | 44.3094 |
| DAW-FDD | 3.9755 | **31.4265** | **64.1563** | **13.0961** | **21.6577** | 45.6674 |
| Ding adaptation | **3.7603** | 32.6387 | 66.5026 | 13.5005 | 21.7124 | **43.5836** |
| Our full L_H | 4.5642 | 44.4823 | 91.1527 | 15.5344 | 28.1530 | 57.7489 |
| Our ramp | 4.0190 | 34.1189 | 69.2429 | 14.0293 | 23.0291 | 48.6503 |
| Our low response | 4.0190 | 34.1189 | 69.2429 | 14.0293 | 23.0291 | 48.6503 |

## Seed 17

| Method | Gender | Race | Gender x race | Age | Skin tone | IDS8 |
|---|---:|---:|---:|---:|---:|---:|
| BCE | 3.9433 | 33.3679 | 67.9232 | 14.1164 | 22.4947 | 45.2732 |
| DAG-FDD | 4.1130 | 32.1378 | 65.4379 | 14.2479 | 21.6486 | 45.3198 |
| DAW-FDD | 4.0066 | 31.6475 | 64.3660 | 13.7688 | 21.4946 | 45.2011 |
| Ding adaptation | **3.8982** | 32.5993 | 66.3557 | 13.6488 | 22.3461 | 47.0588 |
| Our full L_H | 6.2151 | 32.2852 | 67.4903 | **12.1312** | **21.1073** | 47.7026 |
| Our ramp | 3.9149 | **31.2738** | **63.7480** | 13.6684 | 21.3420 | **44.6184** |
| Our low response | 3.9149 | **31.2738** | **63.7480** | 13.6684 | 21.3420 | **44.6184** |

## Seed 27

| Method | Gender | Race | Gender x race | Age | Skin tone | IDS8 |
|---|---:|---:|---:|---:|---:|---:|
| BCE | **3.7918** | 32.4355 | 66.0161 | 13.9615 | 22.1180 | 45.9879 |
| DAG-FDD | 4.0827 | 32.0415 | 65.1804 | 13.7007 | 21.9629 | 45.7036 |
| DAW-FDD | 4.0225 | 31.7886 | 64.8131 | 13.6094 | 21.3504 | 45.5232 |
| Ding adaptation | 4.0520 | 31.9823 | 65.1382 | **13.3626** | 21.6473 | 45.9768 |
| Our full L_H | 5.2715 | 38.9458 | 79.7626 | 16.2555 | 25.0608 | 56.0602 |
| Our ramp | 3.9149 | **30.7358** | **62.8972** | 13.5636 | **21.3459** | **44.5186** |
| Our low response | 3.9149 | **30.7358** | **62.8972** | 13.5636 | **21.3459** | **44.5186** |


Seed 17 full L_H has the lowest age and skin FPR disparities, with lower AUC/higher miss rate. Seed 27 ramp/low has the lowest race, joint, skin and IDS8 FPR disparities; its skin advantage over DAW is only 0.00451 points. No result supports topping every metric.

[Original final-results verification](../ff2_fully_matched_final_20261110/VERIFICATION.json), [source provenance](PROVENANCE.json), [builder](build_tables.py), [artifact hashes](SHA256.json). This packet computes tables from stored aggregates, not new predictions.