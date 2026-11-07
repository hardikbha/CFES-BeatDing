# P4: Ding ResNet-50 SFD+GDA (released code), seed 17, GDA epoch 49 vs Ding Table 3

**Screen only, single seed, no pass/fail claim.** Start = R0_resnet50_s17/49.pth (= Ding Res50Detector_ori/49.pth stage).
Columns: 36 fairness + 4 AUC = 40. * = better than Ding published.

## op05

| set | column | Ding T3 | P4 (SFD+GDA) | R0 base |
|---|---|---|---|---|
| ff++ | g F_FPR | 3.94 | 1.28 * | 0.06 * |
| ff++ | g F_DP | 5.14 | 2.50 * | 1.59 * |
| ff++ | g es-AUC | 94.46 | 93.57 | 92.69 |
| ff++ | r F_FPR | 18.67 | 22.78 | 28.87 |
| ff++ | r F_DP | 2.43 | 4.18 | 2.83 |
| ff++ | r es-AUC | 92.69 | 93.85 * | 92.55 |
| ff++ | i F_FPR | 34.30 | 55.21 | 58.69 |
| ff++ | i F_DP | 8.61 | 7.05 * | 5.41 * |
| ff++ | i es-AUC | 82.87 | 82.79 | 80.84 |
| ff++ | AUC | 95.53 | 95.67 * | 95.11 |
| dfdc | g F_FPR | 3.97 | 9.87 | 11.94 |
| dfdc | g F_DP | 3.61 | 8.21 | 9.12 |
| dfdc | g es-AUC | 56.96 | 55.49 | 55.53 |
| dfdc | r F_FPR | 11.94 | 24.73 | 25.90 |
| dfdc | r F_DP | 5.15 | 8.54 | 9.02 |
| dfdc | r es-AUC | 50.99 | 48.77 | 48.88 |
| dfdc | i F_FPR | 32.14 | 58.38 | 62.83 |
| dfdc | i F_DP | 10.27 | 13.84 | 17.32 |
| dfdc | i es-AUC | 37.55 | 32.76 | 32.24 |
| dfdc | AUC | 62.54 | 59.89 | 59.37 |
| celebdf | g F_FPR | 4.53 | 2.05 * | 2.78 * |
| celebdf | g F_DP | 12.00 | 21.16 | 12.14 |
| celebdf | g es-AUC | 68.98 | 67.22 | 65.77 |
| celebdf | r F_FPR | 6.76 | 29.51 | 20.42 |
| celebdf | r F_DP | 8.16 | 19.52 | 12.28 |
| celebdf | r es-AUC | 71.03 | 58.52 | 56.29 |
| celebdf | i F_FPR | 5.56 | 30.73 | 19.14 |
| celebdf | i F_DP | 9.11 | 20.90 | 11.44 |
| celebdf | i es-AUC | 68.25 | 56.85 | 54.73 |
| celebdf | AUC | 77.35 | 77.22 | 76.56 |
| dfd | g F_FPR | 3.12 | 3.62 | 1.78 * |
| dfd | g F_DP | 1.76 | 1.79 | 1.42 * |
| dfd | g es-AUC | 77.68 | 72.30 | 73.41 |
| dfd | r F_FPR | 6.51 | 19.12 | 18.74 |
| dfd | r F_DP | 3.03 | 4.59 | 6.63 |
| dfd | r es-AUC | 70.93 | 66.61 | 66.28 |
| dfd | i F_FPR | 20.99 | 47.72 | 34.16 |
| dfd | i F_DP | 4.85 | 10.52 | 8.74 |
| dfd | i es-AUC | 66.19 | 59.02 | 61.82 |
| dfd | AUC | 78.63 | 73.48 | 73.87 |

P4 wins vs Ding 6/40 (fairness 5/36); R0 base 6/40 (fairness 6/36)

## valFPR10

| set | column | Ding T3 | P4 (SFD+GDA) | R0 base |
|---|---|---|---|---|
| ff++ | g F_FPR | 3.94 | 0.05 * | 0.12 * |
| ff++ | g F_DP | 5.14 | 4.47 * | 5.46 |
| ff++ | g es-AUC | 94.46 | 93.57 | 92.69 |
| ff++ | r F_FPR | 18.67 | 12.71 * | 13.93 * |
| ff++ | r F_DP | 2.43 | 4.24 | 3.29 |
| ff++ | r es-AUC | 92.69 | 93.85 * | 92.55 |
| ff++ | i F_FPR | 34.30 | 41.01 | 39.35 |
| ff++ | i F_DP | 8.61 | 8.65 | 8.70 |
| ff++ | i es-AUC | 82.87 | 82.79 | 80.84 |
| ff++ | AUC | 95.53 | 95.67 * | 95.11 |
| dfdc | g F_FPR | 3.97 | 8.17 | 8.95 |
| dfdc | g F_DP | 3.61 | 6.90 | 8.21 |
| dfdc | g es-AUC | 56.96 | 55.49 | 55.53 |
| dfdc | r F_FPR | 11.94 | 14.89 | 13.78 |
| dfdc | r F_DP | 5.15 | 2.91 * | 2.11 * |
| dfdc | r es-AUC | 50.99 | 48.77 | 48.88 |
| dfdc | i F_FPR | 32.14 | 40.49 | 39.77 |
| dfdc | i F_DP | 10.27 | 10.97 | 12.59 |
| dfdc | i es-AUC | 37.55 | 32.76 | 32.24 |
| dfdc | AUC | 62.54 | 59.89 | 59.37 |
| celebdf | g F_FPR | 4.53 | 3.38 * | 2.77 * |
| celebdf | g F_DP | 12.00 | 29.78 | 31.60 |
| celebdf | g es-AUC | 68.98 | 67.22 | 65.77 |
| celebdf | r F_FPR | 6.76 | 34.37 | 42.80 |
| celebdf | r F_DP | 8.16 | 26.36 | 30.29 |
| celebdf | r es-AUC | 71.03 | 58.52 | 56.29 |
| celebdf | i F_FPR | 5.56 | 37.55 | 43.89 |
| celebdf | i F_DP | 9.11 | 27.61 | 31.47 |
| celebdf | i es-AUC | 68.25 | 56.85 | 54.73 |
| celebdf | AUC | 77.35 | 77.22 | 76.56 |
| dfd | g F_FPR | 3.12 | 3.68 | 2.79 * |
| dfd | g F_DP | 1.76 | 0.01 * | 0.65 * |
| dfd | g es-AUC | 77.68 | 72.30 | 73.41 |
| dfd | r F_FPR | 6.51 | 15.58 | 12.41 |
| dfd | r F_DP | 3.03 | 0.20 * | 2.68 * |
| dfd | r es-AUC | 70.93 | 66.61 | 66.28 |
| dfd | i F_FPR | 20.99 | 45.93 | 35.58 |
| dfd | i F_DP | 4.85 | 11.56 | 10.93 |
| dfd | i es-AUC | 66.19 | 59.02 | 61.82 |
| dfd | AUC | 78.63 | 73.48 | 73.87 |

P4 wins vs Ding 9/40 (fairness 8/36); R0 base 7/40 (fairness 7/36)

