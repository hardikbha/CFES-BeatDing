# Table 3 uncertainty — ResNet-50 L_A/L_H, seed 17

Intervals are percentile connected-recorded-contributor bootstrap intervals for fixed selected checkpoints and frozen thresholds. They are conditional on these single-seed models, not training-seed variability or verified biological-person independence. Fairness intervals are reportable only with at least the prespecified900/1000 valid replicates; lower counts are shown as unavailable, without relaxing the floor.

All estimates and intervals below are percentages; paired differences are percentage-point differences, L_H minus L_A. AUC and es-AUC are multiplied by100; rates and F_FPR/F_DP retain their metric definitions.

## Per-arm estimates and intervals

| Dataset | Arm | Threshold | Metric | Estimate | 95% CI | Valid / 1000 |
|---|---|---|---|---:|---|---:|
| FFPP | L_A | Source-VAL10 | overall/auc | 96.210 | [95.355, 97.087] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | overall/fpr | 11.503 | [8.721, 14.131] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | overall/fnr | 8.599 | [6.870, 10.454] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | gender/F_FPR | 0.216 | [0.091, 6.127] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | gender/F_DP | 5.024 | [1.635, 8.207] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | gender/es_AUC | 93.724 | [91.691, 95.657] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | gender/worst_group_auc | 94.868 | [93.426, 96.238] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | race/F_FPR | 16.709 | [7.018, 33.272] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | race/F_DP | 3.335 | [2.331, 13.703] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | race/es_AUC | 92.312 | [87.922, 94.149] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | race/worst_group_auc | 95.111 | [91.631, 96.525] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | age_bin/F_FPR | 19.800 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Source-VAL10 | age_bin/F_DP | 36.621 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Source-VAL10 | age_bin/es_AUC | 92.714 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Source-VAL10 | age_bin/worst_group_auc | 95.960 | Unavailable (<900) | 680/1000 |
| FFPP | L_A | Source-VAL10 | skin_bin/F_FPR | 6.309 | [3.445, 19.984] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | skin_bin/F_DP | 4.257 | [1.825, 11.027] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | skin_bin/es_AUC | 94.508 | [90.447, 95.685] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | skin_bin/worst_group_auc | 95.370 | [92.917, 96.538] | 1000/1000 |
| FFPP | L_A | Source-VAL10 | gender_race/F_FPR | 33.338 | [22.098, 76.652] | 990/1000 |
| FFPP | L_A | Source-VAL10 | gender_race/F_DP | 9.751 | [9.558, 29.930] | 990/1000 |
| FFPP | L_A | Source-VAL10 | gender_race/es_AUC | 80.394 | [73.088, 86.044] | 990/1000 |
| FFPP | L_A | Source-VAL10 | gender_race/worst_group_auc | 90.935 | [81.731, 94.736] | 990/1000 |
| FFPP | L_A | Fixed 0.5 | overall/auc | 96.210 | [95.355, 97.087] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | overall/fpr | 15.381 | [12.215, 18.693] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | overall/fnr | 6.581 | [5.119, 8.190] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | gender/F_FPR | 0.443 | [0.094, 7.248] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | gender/F_DP | 4.382 | [1.204, 7.426] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | gender/es_AUC | 93.724 | [91.691, 95.657] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | gender/worst_group_auc | 94.868 | [93.426, 96.238] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | race/F_FPR | 14.864 | [6.970, 37.051] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | race/F_DP | 3.101 | [2.045, 12.770] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | race/es_AUC | 92.312 | [87.922, 94.149] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | race/worst_group_auc | 95.111 | [91.631, 96.525] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | age_bin/F_FPR | 25.369 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Fixed 0.5 | age_bin/F_DP | 37.217 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Fixed 0.5 | age_bin/es_AUC | 92.714 | Unavailable (<900) | 675/1000 |
| FFPP | L_A | Fixed 0.5 | age_bin/worst_group_auc | 95.960 | Unavailable (<900) | 680/1000 |
| FFPP | L_A | Fixed 0.5 | skin_bin/F_FPR | 8.412 | [4.420, 23.660] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | skin_bin/F_DP | 3.797 | [1.475, 10.250] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | skin_bin/es_AUC | 94.508 | [90.447, 95.685] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | skin_bin/worst_group_auc | 95.370 | [92.917, 96.538] | 1000/1000 |
| FFPP | L_A | Fixed 0.5 | gender_race/F_FPR | 31.369 | [25.287, 89.797] | 990/1000 |
| FFPP | L_A | Fixed 0.5 | gender_race/F_DP | 9.664 | [8.989, 28.177] | 990/1000 |
| FFPP | L_A | Fixed 0.5 | gender_race/es_AUC | 80.394 | [73.088, 86.044] | 990/1000 |
| FFPP | L_A | Fixed 0.5 | gender_race/worst_group_auc | 90.935 | [81.731, 94.736] | 990/1000 |
| FFPP | L_H | Source-VAL10 | overall/auc | 96.054 | [95.191, 96.942] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | overall/fpr | 12.539 | [9.609, 15.446] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | overall/fnr | 8.321 | [6.642, 10.128] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | gender/F_FPR | 0.471 | [0.079, 6.651] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | gender/F_DP | 5.116 | [1.727, 8.496] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | gender/es_AUC | 93.710 | [91.567, 95.671] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | gender/worst_group_auc | 94.834 | [93.277, 96.273] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | race/F_FPR | 14.900 | [5.992, 31.443] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | race/F_DP | 2.489 | [1.902, 13.950] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | race/es_AUC | 92.767 | [88.106, 94.220] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | race/worst_group_auc | 95.252 | [92.142, 96.595] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | age_bin/F_FPR | 21.435 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Source-VAL10 | age_bin/F_DP | 36.901 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Source-VAL10 | age_bin/es_AUC | 92.404 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Source-VAL10 | age_bin/worst_group_auc | 96.024 | Unavailable (<900) | 680/1000 |
| FFPP | L_H | Source-VAL10 | skin_bin/F_FPR | 6.567 | [3.879, 21.869] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | skin_bin/F_DP | 3.457 | [1.533, 10.595] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | skin_bin/es_AUC | 94.168 | [89.932, 95.539] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | skin_bin/worst_group_auc | 95.058 | [92.366, 96.343] | 1000/1000 |
| FFPP | L_H | Source-VAL10 | gender_race/F_FPR | 28.629 | [21.344, 74.637] | 990/1000 |
| FFPP | L_H | Source-VAL10 | gender_race/F_DP | 9.504 | [9.302, 29.702] | 990/1000 |
| FFPP | L_H | Source-VAL10 | gender_race/es_AUC | 81.377 | [73.328, 86.418] | 990/1000 |
| FFPP | L_H | Source-VAL10 | gender_race/worst_group_auc | 91.664 | [82.372, 95.175] | 990/1000 |
| FFPP | L_H | Fixed 0.5 | overall/auc | 96.054 | [95.191, 96.942] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | overall/fpr | 16.152 | [12.722, 19.736] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | overall/fnr | 6.514 | [5.066, 8.110] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | gender/F_FPR | 0.018 | [0.138, 7.775] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | gender/F_DP | 4.260 | [1.035, 7.552] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | gender/es_AUC | 93.710 | [91.567, 95.671] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | gender/worst_group_auc | 94.834 | [93.277, 96.273] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | race/F_FPR | 15.904 | [6.269, 35.125] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | race/F_DP | 2.670 | [1.715, 13.590] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | race/es_AUC | 92.767 | [88.106, 94.220] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | race/worst_group_auc | 95.252 | [92.142, 96.595] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | age_bin/F_FPR | 26.525 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Fixed 0.5 | age_bin/F_DP | 37.309 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Fixed 0.5 | age_bin/es_AUC | 92.404 | Unavailable (<900) | 675/1000 |
| FFPP | L_H | Fixed 0.5 | age_bin/worst_group_auc | 96.024 | Unavailable (<900) | 680/1000 |
| FFPP | L_H | Fixed 0.5 | skin_bin/F_FPR | 7.628 | [4.784, 24.420] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | skin_bin/F_DP | 3.143 | [1.318, 9.775] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | skin_bin/es_AUC | 94.168 | [89.932, 95.539] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | skin_bin/worst_group_auc | 95.058 | [92.366, 96.343] | 1000/1000 |
| FFPP | L_H | Fixed 0.5 | gender_race/F_FPR | 30.434 | [25.887, 88.753] | 990/1000 |
| FFPP | L_H | Fixed 0.5 | gender_race/F_DP | 8.933 | [8.798, 27.976] | 990/1000 |
| FFPP | L_H | Fixed 0.5 | gender_race/es_AUC | 81.377 | [73.328, 86.418] | 990/1000 |
| FFPP | L_H | Fixed 0.5 | gender_race/worst_group_auc | 91.664 | [82.372, 95.175] | 990/1000 |
| DFDC | L_A | Source-VAL10 | overall/auc | 59.535 | [56.183, 63.070] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | overall/fpr | 28.941 | [26.204, 31.724] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | overall/fnr | 58.299 | [54.141, 62.350] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | gender/F_FPR | 8.875 | [3.561, 14.314] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | gender/F_DP | 9.013 | [4.526, 13.765] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | gender/es_AUC | 57.356 | [52.532, 61.567] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | gender/worst_group_auc | 57.514 | [53.002, 61.457] | 1000/1000 |
| DFDC | L_A | Source-VAL10 | race/F_FPR | 18.555 | [7.545, 33.467] | 982/1000 |
| DFDC | L_A | Source-VAL10 | race/F_DP | 5.160 | [2.980, 16.504] | 982/1000 |
| DFDC | L_A | Source-VAL10 | race/es_AUC | 49.518 | [41.110, 54.751] | 982/1000 |
| DFDC | L_A | Source-VAL10 | race/worst_group_auc | 56.182 | [47.406, 60.634] | 982/1000 |
| DFDC | L_A | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Source-VAL10 | gender_race/F_FPR | 45.316 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Source-VAL10 | gender_race/F_DP | 13.126 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Source-VAL10 | gender_race/es_AUC | 36.146 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Source-VAL10 | gender_race/worst_group_auc | 54.034 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Fixed 0.5 | overall/auc | 59.535 | [56.183, 63.070] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | overall/fpr | 35.574 | [32.569, 38.675] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | overall/fnr | 51.226 | [47.081, 55.533] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | gender/F_FPR | 8.630 | [2.714, 14.581] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | gender/F_DP | 8.707 | [3.973, 13.556] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | gender/es_AUC | 57.356 | [52.532, 61.567] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | gender/worst_group_auc | 57.514 | [53.002, 61.457] | 1000/1000 |
| DFDC | L_A | Fixed 0.5 | race/F_FPR | 18.793 | [8.435, 35.630] | 982/1000 |
| DFDC | L_A | Fixed 0.5 | race/F_DP | 4.753 | [2.964, 16.376] | 982/1000 |
| DFDC | L_A | Fixed 0.5 | race/es_AUC | 49.518 | [41.110, 54.751] | 982/1000 |
| DFDC | L_A | Fixed 0.5 | race/worst_group_auc | 56.182 | [47.406, 60.634] | 982/1000 |
| DFDC | L_A | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_A | Fixed 0.5 | gender_race/F_FPR | 46.563 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Fixed 0.5 | gender_race/F_DP | 12.361 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Fixed 0.5 | gender_race/es_AUC | 36.146 | Unavailable (<900) | 608/1000 |
| DFDC | L_A | Fixed 0.5 | gender_race/worst_group_auc | 54.034 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Source-VAL10 | overall/auc | 60.068 | [56.323, 63.552] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | overall/fpr | 30.744 | [27.907, 33.776] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | overall/fnr | 54.572 | [50.292, 58.821] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | gender/F_FPR | 9.845 | [4.136, 15.714] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | gender/F_DP | 9.365 | [4.692, 14.364] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | gender/es_AUC | 57.952 | [52.909, 62.059] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | gender/worst_group_auc | 57.953 | [53.198, 61.930] | 1000/1000 |
| DFDC | L_H | Source-VAL10 | race/F_FPR | 15.998 | [6.205, 32.129] | 982/1000 |
| DFDC | L_H | Source-VAL10 | race/F_DP | 2.452 | [2.015, 15.689] | 982/1000 |
| DFDC | L_H | Source-VAL10 | race/es_AUC | 51.183 | [40.454, 54.600] | 982/1000 |
| DFDC | L_H | Source-VAL10 | race/worst_group_auc | 55.142 | [40.146, 60.300] | 982/1000 |
| DFDC | L_H | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Source-VAL10 | gender_race/F_FPR | 46.175 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Source-VAL10 | gender_race/F_DP | 13.685 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Source-VAL10 | gender_race/es_AUC | 33.685 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Source-VAL10 | gender_race/worst_group_auc | 50.432 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Fixed 0.5 | overall/auc | 60.068 | [56.323, 63.552] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | overall/fpr | 36.792 | [33.681, 40.086] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | overall/fnr | 48.529 | [44.427, 52.924] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | gender/F_FPR | 10.086 | [4.274, 16.004] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | gender/F_DP | 9.267 | [4.479, 14.282] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | gender/es_AUC | 57.952 | [52.909, 62.059] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | gender/worst_group_auc | 57.953 | [53.198, 61.930] | 1000/1000 |
| DFDC | L_H | Fixed 0.5 | race/F_FPR | 16.356 | [6.806, 33.755] | 982/1000 |
| DFDC | L_H | Fixed 0.5 | race/F_DP | 1.596 | [2.070, 15.797] | 982/1000 |
| DFDC | L_H | Fixed 0.5 | race/es_AUC | 51.183 | [40.454, 54.600] | 982/1000 |
| DFDC | L_H | Fixed 0.5 | race/worst_group_auc | 55.142 | [40.146, 60.300] | 982/1000 |
| DFDC | L_H | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | L_H | Fixed 0.5 | gender_race/F_FPR | 48.644 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Fixed 0.5 | gender_race/F_DP | 14.664 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Fixed 0.5 | gender_race/es_AUC | 33.685 | Unavailable (<900) | 608/1000 |
| DFDC | L_H | Fixed 0.5 | gender_race/worst_group_auc | 50.432 | Unavailable (<900) | 608/1000 |
| Celeb-DF | L_A | Source-VAL10 | overall/auc | 73.497 | [71.608, 75.542] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | overall/fpr | 55.312 | [51.824, 58.769] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | overall/fnr | 16.988 | [15.969, 18.014] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender/F_FPR | 12.503 | [4.531, 20.228] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender/F_DP | 30.266 | [25.092, 35.000] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender/es_AUC | 62.410 | [58.894, 66.046] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender/worst_group_auc | 56.245 | [49.838, 62.804] | 1000/1000 |
| Celeb-DF | L_A | Source-VAL10 | race/F_FPR | 43.414 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Source-VAL10 | race/F_DP | 31.226 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Source-VAL10 | race/es_AUC | 60.013 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Source-VAL10 | race/worst_group_auc | 61.889 | Unavailable (<900) | 611/1000 |
| Celeb-DF | L_A | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender_race/F_FPR | 54.770 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender_race/F_DP | 39.998 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender_race/es_AUC | 55.898 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Source-VAL10 | gender_race/worst_group_auc | 52.444 | Unavailable (<900) | 603/1000 |
| Celeb-DF | L_A | Fixed 0.5 | overall/auc | 73.497 | [71.608, 75.542] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | overall/fpr | 62.297 | [58.894, 65.551] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | overall/fnr | 13.581 | [12.684, 14.478] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender/F_FPR | 12.193 | [4.330, 19.802] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender/F_DP | 27.694 | [22.743, 32.631] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender/es_AUC | 62.410 | [58.894, 66.046] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender/worst_group_auc | 56.245 | [49.838, 62.804] | 1000/1000 |
| Celeb-DF | L_A | Fixed 0.5 | race/F_FPR | 44.331 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Fixed 0.5 | race/F_DP | 30.750 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Fixed 0.5 | race/es_AUC | 60.013 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_A | Fixed 0.5 | race/worst_group_auc | 61.889 | Unavailable (<900) | 611/1000 |
| Celeb-DF | L_A | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender_race/F_FPR | 50.570 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender_race/F_DP | 35.473 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender_race/es_AUC | 55.898 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_A | Fixed 0.5 | gender_race/worst_group_auc | 52.444 | Unavailable (<900) | 603/1000 |
| Celeb-DF | L_H | Source-VAL10 | overall/auc | 73.811 | [71.996, 75.854] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | overall/fpr | 59.566 | [56.060, 62.924] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | overall/fnr | 14.293 | [13.402, 15.233] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender/F_FPR | 14.521 | [6.739, 22.271] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender/F_DP | 30.512 | [25.449, 35.478] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender/es_AUC | 62.541 | [59.297, 66.043] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender/worst_group_auc | 56.537 | [50.205, 63.053] | 1000/1000 |
| Celeb-DF | L_H | Source-VAL10 | race/F_FPR | 45.783 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Source-VAL10 | race/F_DP | 32.054 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Source-VAL10 | race/es_AUC | 61.958 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Source-VAL10 | race/worst_group_auc | 63.251 | Unavailable (<900) | 611/1000 |
| Celeb-DF | L_H | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender_race/F_FPR | 52.985 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender_race/F_DP | 36.928 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender_race/es_AUC | 57.854 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Source-VAL10 | gender_race/worst_group_auc | 57.074 | Unavailable (<900) | 603/1000 |
| Celeb-DF | L_H | Fixed 0.5 | overall/auc | 73.811 | [71.996, 75.854] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | overall/fpr | 65.832 | [62.618, 68.991] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | overall/fnr | 11.422 | [10.643, 12.248] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender/F_FPR | 14.282 | [7.025, 21.582] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender/F_DP | 27.915 | [23.333, 32.507] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender/es_AUC | 62.541 | [59.297, 66.043] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender/worst_group_auc | 56.537 | [50.205, 63.053] | 1000/1000 |
| Celeb-DF | L_H | Fixed 0.5 | race/F_FPR | 44.797 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Fixed 0.5 | race/F_DP | 31.430 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Fixed 0.5 | race/es_AUC | 61.958 | Unavailable (<900) | 609/1000 |
| Celeb-DF | L_H | Fixed 0.5 | race/worst_group_auc | 63.251 | Unavailable (<900) | 611/1000 |
| Celeb-DF | L_H | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender_race/F_FPR | 49.412 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender_race/F_DP | 34.486 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender_race/es_AUC | 57.854 | Unavailable (<900) | 587/1000 |
| Celeb-DF | L_H | Fixed 0.5 | gender_race/worst_group_auc | 57.074 | Unavailable (<900) | 603/1000 |
| DFD | L_A | Source-VAL10 | overall/auc | 73.655 | [69.185, 77.942] | 1000/1000 |
| DFD | L_A | Source-VAL10 | overall/fpr | 45.957 | [40.318, 52.406] | 1000/1000 |
| DFD | L_A | Source-VAL10 | overall/fnr | 22.009 | [18.341, 25.583] | 1000/1000 |
| DFD | L_A | Source-VAL10 | gender/F_FPR | 2.555 | [0.199, 14.667] | 1000/1000 |
| DFD | L_A | Source-VAL10 | gender/F_DP | 0.939 | [0.155, 8.967] | 1000/1000 |
| DFD | L_A | Source-VAL10 | gender/es_AUC | 72.317 | [65.159, 76.572] | 1000/1000 |
| DFD | L_A | Source-VAL10 | gender/worst_group_auc | 72.809 | [66.840, 77.041] | 1000/1000 |
| DFD | L_A | Source-VAL10 | race/F_FPR | 9.534 | [0.545, 23.219] | 1000/1000 |
| DFD | L_A | Source-VAL10 | race/F_DP | 3.188 | [0.212, 12.216] | 1000/1000 |
| DFD | L_A | Source-VAL10 | race/es_AUC | 71.277 | [63.320, 75.323] | 1000/1000 |
| DFD | L_A | Source-VAL10 | race/worst_group_auc | 70.778 | [60.330, 75.653] | 1000/1000 |
| DFD | L_A | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Source-VAL10 | gender_race/F_FPR | 34.454 | [17.257, 55.538] | 957/1000 |
| DFD | L_A | Source-VAL10 | gender_race/F_DP | 8.316 | [3.540, 25.987] | 957/1000 |
| DFD | L_A | Source-VAL10 | gender_race/es_AUC | 56.511 | [47.202, 65.716] | 957/1000 |
| DFD | L_A | Source-VAL10 | gender_race/worst_group_auc | 67.434 | [54.460, 72.885] | 957/1000 |
| DFD | L_A | Fixed 0.5 | overall/auc | 73.655 | [69.185, 77.942] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | overall/fpr | 52.458 | [46.950, 58.899] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | overall/fnr | 18.052 | [14.631, 21.364] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | gender/F_FPR | 2.760 | [0.271, 14.486] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | gender/F_DP | 1.314 | [0.113, 8.475] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | gender/es_AUC | 72.317 | [65.159, 76.572] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | gender/worst_group_auc | 72.809 | [66.840, 77.041] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | race/F_FPR | 8.833 | [0.457, 21.994] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | race/F_DP | 3.675 | [0.195, 12.235] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | race/es_AUC | 71.277 | [63.320, 75.323] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | race/worst_group_auc | 70.778 | [60.330, 75.653] | 1000/1000 |
| DFD | L_A | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_A | Fixed 0.5 | gender_race/F_FPR | 31.752 | [12.678, 54.146] | 957/1000 |
| DFD | L_A | Fixed 0.5 | gender_race/F_DP | 7.989 | [3.828, 23.427] | 957/1000 |
| DFD | L_A | Fixed 0.5 | gender_race/es_AUC | 56.511 | [47.202, 65.716] | 957/1000 |
| DFD | L_A | Fixed 0.5 | gender_race/worst_group_auc | 67.434 | [54.460, 72.885] | 957/1000 |
| DFD | L_H | Source-VAL10 | overall/auc | 73.472 | [68.678, 77.812] | 1000/1000 |
| DFD | L_H | Source-VAL10 | overall/fpr | 47.057 | [41.180, 53.535] | 1000/1000 |
| DFD | L_H | Source-VAL10 | overall/fnr | 21.357 | [17.726, 25.082] | 1000/1000 |
| DFD | L_H | Source-VAL10 | gender/F_FPR | 2.165 | [0.180, 15.234] | 1000/1000 |
| DFD | L_H | Source-VAL10 | gender/F_DP | 1.598 | [0.118, 9.435] | 1000/1000 |
| DFD | L_H | Source-VAL10 | gender/es_AUC | 71.704 | [64.437, 76.472] | 1000/1000 |
| DFD | L_H | Source-VAL10 | gender/worst_group_auc | 72.357 | [66.213, 76.933] | 1000/1000 |
| DFD | L_H | Source-VAL10 | race/F_FPR | 6.513 | [0.290, 21.897] | 1000/1000 |
| DFD | L_H | Source-VAL10 | race/F_DP | 0.148 | [0.148, 11.412] | 1000/1000 |
| DFD | L_H | Source-VAL10 | race/es_AUC | 70.609 | [62.316, 75.156] | 1000/1000 |
| DFD | L_H | Source-VAL10 | race/worst_group_auc | 70.013 | [58.829, 75.385] | 1000/1000 |
| DFD | L_H | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Source-VAL10 | gender_race/F_FPR | 40.819 | [22.618, 63.066] | 957/1000 |
| DFD | L_H | Source-VAL10 | gender_race/F_DP | 5.283 | [3.779, 26.727] | 957/1000 |
| DFD | L_H | Source-VAL10 | gender_race/es_AUC | 54.647 | [45.576, 64.129] | 957/1000 |
| DFD | L_H | Source-VAL10 | gender_race/worst_group_auc | 65.706 | [52.154, 72.013] | 957/1000 |
| DFD | L_H | Fixed 0.5 | overall/auc | 73.472 | [68.678, 77.812] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | overall/fpr | 52.555 | [46.812, 59.140] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | overall/fnr | 17.623 | [14.085, 21.066] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | gender/F_FPR | 2.294 | [0.259, 15.149] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | gender/F_DP | 1.767 | [0.142, 9.270] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | gender/es_AUC | 71.704 | [64.437, 76.472] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | gender/worst_group_auc | 72.357 | [66.213, 76.933] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | race/F_FPR | 6.628 | [0.433, 20.695] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | race/F_DP | 0.947 | [0.122, 10.990] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | race/es_AUC | 70.609 | [62.316, 75.156] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | race/worst_group_auc | 70.013 | [58.829, 75.385] | 1000/1000 |
| DFD | L_H | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | L_H | Fixed 0.5 | gender_race/F_FPR | 33.572 | [13.154, 57.080] | 957/1000 |
| DFD | L_H | Fixed 0.5 | gender_race/F_DP | 6.310 | [3.868, 24.645] | 957/1000 |
| DFD | L_H | Fixed 0.5 | gender_race/es_AUC | 54.647 | [45.576, 64.129] | 957/1000 |
| DFD | L_H | Fixed 0.5 | gender_race/worst_group_auc | 65.706 | [52.154, 72.013] | 957/1000 |

## Paired L_H − L_A bootstrap

The paired bootstrap reuses the same recorded-contributor resampling draws for both fixed models. A positive difference means L_H is higher on that metric; for FPR/FNR/F_FPR lower is preferable, while AUC/es-AUC higher is preferable.

| Dataset | Threshold | Metric | H−A estimate (pp) | 95% CI (pp) | Valid / 1000 |
|---|---|---|---:|---|---:|
| FFPP | Source-VAL10 | overall/auc | -0.157 | [-0.293, -0.007] | 1000/1000 |
| FFPP | Source-VAL10 | overall/fpr | 1.036 | [0.324, 1.766] | 1000/1000 |
| FFPP | Source-VAL10 | overall/fnr | -0.278 | [-0.784, 0.156] | 1000/1000 |
| FFPP | Source-VAL10 | gender/F_FPR | 0.254 | [-1.511, 2.010] | 1000/1000 |
| FFPP | Source-VAL10 | gender/F_DP | 0.092 | [-0.922, 1.036] | 1000/1000 |
| FFPP | Source-VAL10 | gender/es_AUC | -0.014 | [-0.343, 0.347] | 1000/1000 |
| FFPP | Source-VAL10 | gender/worst_group_auc | -0.034 | [-0.281, 0.260] | 1000/1000 |
| FFPP | Source-VAL10 | race/F_FPR | -1.809 | [-7.186, 3.960] | 1000/1000 |
| FFPP | Source-VAL10 | race/F_DP | -0.846 | [-1.770, 1.647] | 1000/1000 |
| FFPP | Source-VAL10 | race/es_AUC | 0.455 | [-1.285, 1.561] | 1000/1000 |
| FFPP | Source-VAL10 | race/worst_group_auc | 0.141 | [-0.346, 1.360] | 1000/1000 |
| FFPP | Source-VAL10 | age_bin/F_FPR | 1.635 | Unavailable (<900) | 675/1000 |
| FFPP | Source-VAL10 | age_bin/F_DP | 0.280 | Unavailable (<900) | 675/1000 |
| FFPP | Source-VAL10 | age_bin/es_AUC | -0.310 | Unavailable (<900) | 675/1000 |
| FFPP | Source-VAL10 | age_bin/worst_group_auc | 0.064 | Unavailable (<900) | 680/1000 |
| FFPP | Source-VAL10 | skin_bin/F_FPR | 0.258 | [-2.431, 3.758] | 1000/1000 |
| FFPP | Source-VAL10 | skin_bin/F_DP | -0.801 | [-1.438, 0.980] | 1000/1000 |
| FFPP | Source-VAL10 | skin_bin/es_AUC | -0.340 | [-0.964, 0.338] | 1000/1000 |
| FFPP | Source-VAL10 | skin_bin/worst_group_auc | -0.312 | [-0.689, 0.114] | 1000/1000 |
| FFPP | Source-VAL10 | gender_race/F_FPR | -4.709 | [-12.636, 9.389] | 990/1000 |
| FFPP | Source-VAL10 | gender_race/F_DP | -0.247 | [-1.895, 2.520] | 990/1000 |
| FFPP | Source-VAL10 | gender_race/es_AUC | 0.983 | [-0.972, 2.166] | 990/1000 |
| FFPP | Source-VAL10 | gender_race/worst_group_auc | 0.729 | [-0.379, 2.193] | 990/1000 |
| FFPP | Fixed 0.5 | overall/auc | -0.157 | [-0.293, -0.007] | 1000/1000 |
| FFPP | Fixed 0.5 | overall/fpr | 0.771 | [0.065, 1.460] | 1000/1000 |
| FFPP | Fixed 0.5 | overall/fnr | -0.067 | [-0.576, 0.317] | 1000/1000 |
| FFPP | Fixed 0.5 | gender/F_FPR | -0.425 | [-1.396, 1.675] | 1000/1000 |
| FFPP | Fixed 0.5 | gender/F_DP | -0.122 | [-1.004, 0.680] | 1000/1000 |
| FFPP | Fixed 0.5 | gender/es_AUC | -0.014 | [-0.343, 0.347] | 1000/1000 |
| FFPP | Fixed 0.5 | gender/worst_group_auc | -0.034 | [-0.281, 0.260] | 1000/1000 |
| FFPP | Fixed 0.5 | race/F_FPR | 1.040 | [-8.210, 5.829] | 1000/1000 |
| FFPP | Fixed 0.5 | race/F_DP | -0.431 | [-1.725, 1.884] | 1000/1000 |
| FFPP | Fixed 0.5 | race/es_AUC | 0.455 | [-1.285, 1.561] | 1000/1000 |
| FFPP | Fixed 0.5 | race/worst_group_auc | 0.141 | [-0.346, 1.360] | 1000/1000 |
| FFPP | Fixed 0.5 | age_bin/F_FPR | 1.156 | Unavailable (<900) | 675/1000 |
| FFPP | Fixed 0.5 | age_bin/F_DP | 0.091 | Unavailable (<900) | 675/1000 |
| FFPP | Fixed 0.5 | age_bin/es_AUC | -0.310 | Unavailable (<900) | 675/1000 |
| FFPP | Fixed 0.5 | age_bin/worst_group_auc | 0.064 | Unavailable (<900) | 680/1000 |
| FFPP | Fixed 0.5 | skin_bin/F_FPR | -0.785 | [-2.954, 3.600] | 1000/1000 |
| FFPP | Fixed 0.5 | skin_bin/F_DP | -0.654 | [-1.329, 1.016] | 1000/1000 |
| FFPP | Fixed 0.5 | skin_bin/es_AUC | -0.340 | [-0.964, 0.338] | 1000/1000 |
| FFPP | Fixed 0.5 | skin_bin/worst_group_auc | -0.312 | [-0.689, 0.114] | 1000/1000 |
| FFPP | Fixed 0.5 | gender_race/F_FPR | -0.935 | [-13.198, 9.204] | 990/1000 |
| FFPP | Fixed 0.5 | gender_race/F_DP | -0.731 | [-2.838, 2.679] | 990/1000 |
| FFPP | Fixed 0.5 | gender_race/es_AUC | 0.983 | [-0.972, 2.166] | 990/1000 |
| FFPP | Fixed 0.5 | gender_race/worst_group_auc | 0.729 | [-0.379, 2.193] | 990/1000 |
| DFDC | Source-VAL10 | overall/auc | 0.533 | [-0.291, 1.258] | 1000/1000 |
| DFDC | Source-VAL10 | overall/fpr | 1.803 | [1.125, 2.538] | 1000/1000 |
| DFDC | Source-VAL10 | overall/fnr | -3.726 | [-4.743, -2.812] | 1000/1000 |
| DFDC | Source-VAL10 | gender/F_FPR | 0.969 | [-0.567, 2.373] | 1000/1000 |
| DFDC | Source-VAL10 | gender/F_DP | 0.352 | [-0.845, 1.436] | 1000/1000 |
| DFDC | Source-VAL10 | gender/es_AUC | 0.596 | [-0.657, 1.499] | 1000/1000 |
| DFDC | Source-VAL10 | gender/worst_group_auc | 0.439 | [-0.732, 1.379] | 1000/1000 |
| DFDC | Source-VAL10 | race/F_FPR | -2.557 | [-5.552, 1.840] | 982/1000 |
| DFDC | Source-VAL10 | race/F_DP | -2.708 | [-4.180, 2.463] | 982/1000 |
| DFDC | Source-VAL10 | race/es_AUC | 1.664 | [-4.539, 3.077] | 982/1000 |
| DFDC | Source-VAL10 | race/worst_group_auc | -1.040 | [-9.572, 0.963] | 982/1000 |
| DFDC | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Source-VAL10 | gender_race/F_FPR | 0.860 | Unavailable (<900) | 608/1000 |
| DFDC | Source-VAL10 | gender_race/F_DP | 0.559 | Unavailable (<900) | 608/1000 |
| DFDC | Source-VAL10 | gender_race/es_AUC | -2.461 | Unavailable (<900) | 608/1000 |
| DFDC | Source-VAL10 | gender_race/worst_group_auc | -3.602 | Unavailable (<900) | 608/1000 |
| DFDC | Fixed 0.5 | overall/auc | 0.533 | [-0.291, 1.258] | 1000/1000 |
| DFDC | Fixed 0.5 | overall/fpr | 1.218 | [0.442, 2.034] | 1000/1000 |
| DFDC | Fixed 0.5 | overall/fnr | -2.697 | [-3.702, -1.786] | 1000/1000 |
| DFDC | Fixed 0.5 | gender/F_FPR | 1.456 | [-0.132, 3.043] | 1000/1000 |
| DFDC | Fixed 0.5 | gender/F_DP | 0.560 | [-0.616, 1.750] | 1000/1000 |
| DFDC | Fixed 0.5 | gender/es_AUC | 0.596 | [-0.657, 1.499] | 1000/1000 |
| DFDC | Fixed 0.5 | gender/worst_group_auc | 0.439 | [-0.732, 1.379] | 1000/1000 |
| DFDC | Fixed 0.5 | race/F_FPR | -2.437 | [-5.865, 1.862] | 982/1000 |
| DFDC | Fixed 0.5 | race/F_DP | -3.157 | [-3.745, 2.525] | 982/1000 |
| DFDC | Fixed 0.5 | race/es_AUC | 1.664 | [-4.539, 3.077] | 982/1000 |
| DFDC | Fixed 0.5 | race/worst_group_auc | -1.040 | [-9.572, 0.963] | 982/1000 |
| DFDC | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFDC | Fixed 0.5 | gender_race/F_FPR | 2.081 | Unavailable (<900) | 608/1000 |
| DFDC | Fixed 0.5 | gender_race/F_DP | 2.303 | Unavailable (<900) | 608/1000 |
| DFDC | Fixed 0.5 | gender_race/es_AUC | -2.461 | Unavailable (<900) | 608/1000 |
| DFDC | Fixed 0.5 | gender_race/worst_group_auc | -3.602 | Unavailable (<900) | 608/1000 |
| Celeb-DF | Source-VAL10 | overall/auc | 0.314 | [0.099, 0.553] | 1000/1000 |
| Celeb-DF | Source-VAL10 | overall/fpr | 4.253 | [3.558, 4.926] | 1000/1000 |
| Celeb-DF | Source-VAL10 | overall/fnr | -2.695 | [-2.956, -2.415] | 1000/1000 |
| Celeb-DF | Source-VAL10 | gender/F_FPR | 2.018 | [0.716, 3.329] | 1000/1000 |
| Celeb-DF | Source-VAL10 | gender/F_DP | 0.246 | [-0.647, 1.051] | 1000/1000 |
| Celeb-DF | Source-VAL10 | gender/es_AUC | 0.131 | [-0.221, 0.617] | 1000/1000 |
| Celeb-DF | Source-VAL10 | gender/worst_group_auc | 0.292 | [-0.416, 1.005] | 1000/1000 |
| Celeb-DF | Source-VAL10 | race/F_FPR | 2.370 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Source-VAL10 | race/F_DP | 0.828 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Source-VAL10 | race/es_AUC | 1.945 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Source-VAL10 | race/worst_group_auc | 1.362 | Unavailable (<900) | 611/1000 |
| Celeb-DF | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Source-VAL10 | gender_race/F_FPR | -1.784 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Source-VAL10 | gender_race/F_DP | -3.071 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Source-VAL10 | gender_race/es_AUC | 1.956 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Source-VAL10 | gender_race/worst_group_auc | 4.630 | Unavailable (<900) | 603/1000 |
| Celeb-DF | Fixed 0.5 | overall/auc | 0.314 | [0.099, 0.553] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | overall/fpr | 3.536 | [2.872, 4.199] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | overall/fnr | -2.159 | [-2.412, -1.905] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | gender/F_FPR | 2.089 | [0.731, 3.422] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | gender/F_DP | 0.221 | [-0.675, 1.111] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | gender/es_AUC | 0.131 | [-0.221, 0.617] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | gender/worst_group_auc | 0.292 | [-0.416, 1.005] | 1000/1000 |
| Celeb-DF | Fixed 0.5 | race/F_FPR | 0.466 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Fixed 0.5 | race/F_DP | 0.680 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Fixed 0.5 | race/es_AUC | 1.945 | Unavailable (<900) | 609/1000 |
| Celeb-DF | Fixed 0.5 | race/worst_group_auc | 1.362 | Unavailable (<900) | 611/1000 |
| Celeb-DF | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| Celeb-DF | Fixed 0.5 | gender_race/F_FPR | -1.157 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Fixed 0.5 | gender_race/F_DP | -0.987 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Fixed 0.5 | gender_race/es_AUC | 1.956 | Unavailable (<900) | 587/1000 |
| Celeb-DF | Fixed 0.5 | gender_race/worst_group_auc | 4.630 | Unavailable (<900) | 603/1000 |
| DFD | Source-VAL10 | overall/auc | -0.183 | [-0.857, 0.536] | 1000/1000 |
| DFD | Source-VAL10 | overall/fpr | 1.100 | [-0.451, 2.475] | 1000/1000 |
| DFD | Source-VAL10 | overall/fnr | -0.652 | [-1.413, 0.016] | 1000/1000 |
| DFD | Source-VAL10 | gender/F_FPR | -0.390 | [-2.496, 3.191] | 1000/1000 |
| DFD | Source-VAL10 | gender/F_DP | 0.658 | [-1.503, 1.867] | 1000/1000 |
| DFD | Source-VAL10 | gender/es_AUC | -0.613 | [-1.579, 0.919] | 1000/1000 |
| DFD | Source-VAL10 | gender/worst_group_auc | -0.452 | [-1.221, 0.768] | 1000/1000 |
| DFD | Source-VAL10 | race/F_FPR | -3.021 | [-5.218, 4.936] | 1000/1000 |
| DFD | Source-VAL10 | race/F_DP | -3.040 | [-4.324, 4.598] | 1000/1000 |
| DFD | Source-VAL10 | race/es_AUC | -0.668 | [-1.698, 0.833] | 1000/1000 |
| DFD | Source-VAL10 | race/worst_group_auc | -0.765 | [-2.487, 0.676] | 1000/1000 |
| DFD | Source-VAL10 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Source-VAL10 | gender_race/F_FPR | 6.365 | [-2.734, 16.081] | 957/1000 |
| DFD | Source-VAL10 | gender_race/F_DP | -3.033 | [-5.334, 7.470] | 957/1000 |
| DFD | Source-VAL10 | gender_race/es_AUC | -1.864 | [-3.433, -0.028] | 957/1000 |
| DFD | Source-VAL10 | gender_race/worst_group_auc | -1.728 | [-3.545, 0.196] | 957/1000 |
| DFD | Fixed 0.5 | overall/auc | -0.183 | [-0.857, 0.536] | 1000/1000 |
| DFD | Fixed 0.5 | overall/fpr | 0.097 | [-1.161, 1.326] | 1000/1000 |
| DFD | Fixed 0.5 | overall/fnr | -0.429 | [-1.174, 0.229] | 1000/1000 |
| DFD | Fixed 0.5 | gender/F_FPR | -0.466 | [-2.311, 2.680] | 1000/1000 |
| DFD | Fixed 0.5 | gender/F_DP | 0.453 | [-1.160, 1.659] | 1000/1000 |
| DFD | Fixed 0.5 | gender/es_AUC | -0.613 | [-1.579, 0.919] | 1000/1000 |
| DFD | Fixed 0.5 | gender/worst_group_auc | -0.452 | [-1.221, 0.768] | 1000/1000 |
| DFD | Fixed 0.5 | race/F_FPR | -2.204 | [-4.247, 3.866] | 1000/1000 |
| DFD | Fixed 0.5 | race/F_DP | -2.729 | [-3.848, 4.083] | 1000/1000 |
| DFD | Fixed 0.5 | race/es_AUC | -0.668 | [-1.698, 0.833] | 1000/1000 |
| DFD | Fixed 0.5 | race/worst_group_auc | -0.765 | [-2.487, 0.676] | 1000/1000 |
| DFD | Fixed 0.5 | age_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | age_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | age_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | age_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | skin_bin/F_FPR | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | skin_bin/F_DP | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | skin_bin/es_AUC | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | skin_bin/worst_group_auc | NA | Unavailable (no eligible support) | 0/1000 |
| DFD | Fixed 0.5 | gender_race/F_FPR | 1.820 | [-3.756, 10.362] | 957/1000 |
| DFD | Fixed 0.5 | gender_race/F_DP | -1.679 | [-3.159, 5.037] | 957/1000 |
| DFD | Fixed 0.5 | gender_race/es_AUC | -1.864 | [-3.433, -0.028] | 957/1000 |
| DFD | Fixed 0.5 | gender_race/worst_group_auc | -1.728 | [-3.545, 0.196] | 957/1000 |

Axes with no non-unknown group support are explicitly unavailable (for example age and skin tone in some cross-domain manifests). Sparse intersection intervals with fewer than900valid replicates remain unavailable. These test-set intervals describe sampling uncertainty in the fixed manifest under recorded-contributor clustering; they do not authorize selection or tuning.
