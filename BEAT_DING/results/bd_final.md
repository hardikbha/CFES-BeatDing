# Beat-Ding final (clip): BD0, 3 seed(s) at the matched operating point, vs Ding published Tables 1-2 (Xception)

Promoted BD recipe: **BD0**. Seeds: BD0_s7, BD0_s17, BD0_s27

Columns won (3-seed mean vs Ding published): **17/40**. Criteria: >= 25/40 False; AUC and all three es-AUC better on all four sets False; intersectional F_FPR better on 1/4 sets (need >= 3). **Claim 'beats Ding': False.**

| set | row | TPR | FPR | gF_FPR | gF_DP | gesAUC | rF_FPR | rF_DP | resAUC | iF_FPR | iF_DP | iesAUC | AUC | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ | Ding published | | | 0.53 | 3.61 | 96.45 | 9.29 | 4.35 | 94.86 | 20.18 | 9.47 | 86.91 | 97.71 | |
| FF++ | ours (n=3) | 93.58 | 13.41 | 1.88 ± 0.21 | 4.75 ± 0.30 | 94.88 ± 0.35 | 16.50 ± 3.19 | 5.03 ± 0.88 | 92.20 ± 0.46 | 47.44 ± 0.99 | 14.69 ± 0.78 | 85.48 ± 0.74 | 96.87 ± 0.10 | 0/10 |
| DFDC | Ding published | | | 1.76 | 3.67 | 57.94 | 17.93 | 7.58 | 52.33 | 37.37 | 11.92 | 39.03 | 61.86 | |
| DFDC | ours (n=3) | 73.63 | 33.74 | 5.29 ± 1.97 | 4.68 ± 1.82 | **70.42** ± 0.31 | **12.13** ± 6.91 | 19.36 ± 1.79 | **66.68** ± 2.04 | **35.83** ± 11.67 | 30.29 ± 6.72 | **54.11** ± 1.86 | **77.75** ± 0.75 | 6/10 |
| Celeb-DF | Ding published | | | 6.41 | 10.81 | 69.68 | 16.60 | 7.29 | 69.96 | 23.32 | 12.01 | 63.27 | 76.75 | |
| Celeb-DF | ours (n=3) | 96.27 | 61.93 | 10.39 ± 7.17 | 25.55 ± 5.69 | **77.31** ± 2.24 | 25.00 ± 13.69 | 21.77 ± 6.49 | **77.61** ± 2.31 | 26.30 ± 10.52 | 21.57 ± 7.68 | **75.71** ± 2.66 | **87.98** ± 1.53 | 4/10 |
| DFD | Ding published | | | 4.72 | 5.49 | 78.98 | 2.09 | 18.66 | 77.44 | 27.32 | 22.81 | 69.73 | 81.46 | |
| DFD | ours (n=3) | 89.99 | 33.16 | **3.14** ± 1.63 | **1.83** ± 1.98 | **89.66** ± 0.42 | 10.66 ± 3.52 | **4.42** ± 2.77 | **88.24** ± 1.06 | 34.64 ± 6.34 | 27.41 ± 2.40 | **83.59** ± 2.29 | **90.38** ± 0.29 | 7/10 |

F_DP oracle (perfect detector; g/r/i): FF++ 0.85/2.97/5.11; DFDC 12.54/31.48/43.18; Celeb-DF 42.56/64.92/59.55; DFD 2.80/17.53/33.13
FF++ i-F_FPR noise floor (BD0_s7, matched t, real FPR 13.51 %): reals-only observed 46.56, video-permutation null median 42.61 [25.40, 66.65]
FF++ i-F_FPR noise floor (BD0_s17, matched t, real FPR 12.69 %): reals-only observed 47.25, video-permutation null median 41.27 [25.61, 63.64]
FF++ i-F_FPR noise floor (BD0_s27, matched t, real FPR 14.01 %): reals-only observed 48.50, video-permutation null median 47.18 [27.79, 73.14]
