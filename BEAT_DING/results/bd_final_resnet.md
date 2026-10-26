# Beat-Ding final (resnet): R50_BD0, 3 seed(s) at the matched operating point, vs Ding published Table 3 (ResNet-50)

Promoted BD recipe: **R50_BD0**. Seeds: R50_BD0_s7, R50_BD0_s17, R50_BD0_s27

Columns won (3-seed mean vs Ding published): **2/40**. Criteria: >= 25/40 False; AUC and all three es-AUC better on all four sets False; intersectional F_FPR better on 0/4 sets (need >= 3). **Claim 'beats Ding': False.**

| set | row | TPR | FPR | gF_FPR | gF_DP | gesAUC | rF_FPR | rF_DP | resAUC | iF_FPR | iF_DP | iesAUC | AUC | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ | Ding published | | | 3.94 | 5.14 | 94.46 | 18.67 | 2.43 | 92.69 | 34.30 | 8.61 | 82.87 | 95.53 | |
| FF++ | ours (n=3) | 95.03 | 34.85 | 5.32 ± 4.72 | **3.07** ± 2.05 | 89.66 ± 1.40 | 25.98 ± 5.47 | 4.48 ± 3.00 | 86.56 ± 3.62 | 54.06 ± 8.86 | 13.45 ± 3.37 | 75.20 ± 1.55 | 92.40 ± 0.67 | 1/10 |
| DFDC | Ding published | | | 3.97 | 3.61 | 56.96 | 11.94 | 5.15 | 50.99 | 32.14 | 10.27 | 37.55 | 62.54 | |
| DFDC | ours (n=3) | 69.05 | 58.59 | 6.85 ± 5.56 | 5.58 ± 8.00 | 55.60 ± 3.64 | 20.51 ± 10.03 | 7.84 ± 2.43 | 47.68 ± 1.14 | 54.54 ± 15.08 | 17.42 ± 5.68 | 35.58 ± 5.43 | 58.45 ± 0.93 | 0/10 |
| Celeb-DF | Ding published | | | 4.53 | 12.00 | 68.98 | 6.76 | 8.16 | 71.03 | 5.56 | 9.11 | 68.25 | 77.35 | |
| Celeb-DF | ours (n=3) | 93.13 | 77.85 | 7.14 ± 6.12 | 13.76 ± 8.06 | 63.76 ± 1.81 | 15.95 ± 15.58 | 12.48 ± 7.95 | 62.11 ± 2.72 | 25.38 ± 21.07 | 15.72 ± 10.84 | 57.66 ± 2.28 | 71.13 ± 2.82 | 0/10 |
| DFD | Ding published | | | 3.12 | 1.76 | 77.68 | 6.51 | 3.03 | 70.93 | 20.99 | 4.85 | 66.19 | 78.63 | |
| DFD | ours (n=3) | 91.23 | 71.84 | 4.80 ± 3.36 | 2.85 ± 1.19 | 70.28 ± 6.07 | 6.97 ± 1.71 | **2.84** ± 3.14 | 69.15 ± 4.60 | 30.90 ± 16.26 | 9.69 ± 0.48 | 56.82 ± 1.76 | 71.37 ± 5.45 | 1/10 |

F_DP oracle (perfect detector; g/r/i): FF++ 0.85/2.97/5.11; DFDC 12.54/31.48/43.18; Celeb-DF 42.56/64.92/59.55; DFD 2.80/17.53/33.13
FF++ i-F_FPR noise floor (R50_BD0_s7, matched t, real FPR 30.56 %): reals-only observed 48.24, video-permutation null median 67.81 [38.36, 104.61]
FF++ i-F_FPR noise floor (R50_BD0_s17, matched t, real FPR 37.22 %): reals-only observed 49.67, video-permutation null median 67.77 [38.41, 105.06]
FF++ i-F_FPR noise floor (R50_BD0_s27, matched t, real FPR 36.78 %): reals-only observed 64.26, video-permutation null median 76.94 [43.60, 113.87]

Matched operating point: val TPR 94.18 %, the Xception reproduction's (no Ding ResNet reproduction exists; disclosed proxy, plan Amendment 2).
