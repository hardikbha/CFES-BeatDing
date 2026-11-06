# Xception Tables 1-2 (Ding T12), L_A / L_H, seeds 7 / 17 / 27 (Amendment 10)

Seed 17 = original locked tables (lh_tables_s17). Seeds 7/27 = same pipeline, only seed + init changed. op05 = primary;
valFPR10 = the run's own native val-FPR10 calibration. Wins vs Ding published (lower F_FPR/F_DP, higher es-AUC/AUC).
Table 1 = FF++ (10 cols), Table 2 = DFDC + Celeb-DF + DFD (30 cols). "best seed" = selected POST HOC ON TEST.

| arm | op | seed rows: T1/10 + T2/30 = total | mean ± sd (T1, T2) | mean-row wins (T1 + T2) | best seed (post hoc on test) |
|---|---|---|---|---|---|
| L_A | op05 | s17: 8+16=24 | 24.00 ± 0.00 (8.00, 16.00) [n=1] | 8+16=24 | s17: 24 (post hoc) |
| L_A | valFPR10 | s17: 7+16=23 | 23.00 ± 0.00 (7.00, 16.00) [n=1] | 7+16=23 | s17: 23 (post hoc) |
| L_H | op05 | s17: 8+16=24 | 24.00 ± 0.00 (8.00, 16.00) [n=1] | 8+16=24 | s17: 24 (post hoc) |
| L_H | valFPR10 | s17: 6+15=21 | 21.00 ± 0.00 (6.00, 15.00) [n=1] | 6+15=21 | s17: 21 (post hoc) |

## L_A, op05: per column (seed values; mean vs Ding)

| set | column | Ding | s17 | mean |
|---|---|---|---|---|
| ffpp | g F_FPR | 0.53 | 0.47 * | 0.47 * |
| ffpp | g F_DP | 3.61 | 1.76 * | 1.76 * |
| ffpp | g es-AUC | 96.45 | 97.47 * | 97.47 * |
| ffpp | r F_FPR | 9.29 | 9.04 * | 9.04 * |
| ffpp | r F_DP | 4.35 | 3.06 * | 3.06 * |
| ffpp | r es-AUC | 94.86 | 97.06 * | 97.06 * |
| ffpp | i F_FPR | 20.18 | 22.79 | 22.79 |
| ffpp | i F_DP | 9.47 | 12.17 | 12.17 |
| ffpp | i es-AUC | 86.91 | 91.88 * | 91.88 * |
| ffpp | AUC | 97.71 | 98.47 * | 98.47 * |
| dfdc | g F_FPR | 1.76 | 0.00 * | 0.00 * |
| dfdc | g F_DP | 3.67 | 1.42 * | 1.42 * |
| dfdc | g es-AUC | 57.94 | 60.69 * | 60.69 * |
| dfdc | r F_FPR | 17.93 | 13.35 * | 13.35 * |
| dfdc | r F_DP | 7.58 | 5.39 * | 5.39 * |
| dfdc | r es-AUC | 52.33 | 53.49 * | 53.49 * |
| dfdc | i F_FPR | 37.37 | 27.84 * | 27.84 * |
| dfdc | i F_DP | 11.92 | 6.77 * | 6.77 * |
| dfdc | i es-AUC | 39.03 | 40.28 * | 40.28 * |
| dfdc | AUC | 61.86 | 61.77 | 61.77 |
| celebdf | g F_FPR | 6.41 | 6.11 * | 6.11 * |
| celebdf | g F_DP | 10.81 | 23.01 | 23.01 |
| celebdf | g es-AUC | 69.68 | 61.23 | 61.23 |
| celebdf | r F_FPR | 16.60 | 37.82 | 37.82 |
| celebdf | r F_DP | 7.29 | 29.17 | 29.17 |
| celebdf | r es-AUC | 69.96 | 50.96 | 50.96 |
| celebdf | i F_FPR | 23.32 | 40.39 | 40.39 |
| celebdf | i F_DP | 12.01 | 30.84 | 30.84 |
| celebdf | i es-AUC | 63.27 | 48.72 | 48.72 |
| celebdf | AUC | 76.75 | 70.75 | 70.75 |
| dfd | g F_FPR | 4.72 | 2.70 * | 2.70 * |
| dfd | g F_DP | 5.49 | 2.39 * | 2.39 * |
| dfd | g es-AUC | 78.98 | 80.60 * | 80.60 * |
| dfd | r F_FPR | 2.09 | 10.88 | 10.88 |
| dfd | r F_DP | 18.66 | 1.74 * | 1.74 * |
| dfd | r es-AUC | 77.44 | 79.80 * | 79.80 * |
| dfd | i F_FPR | 27.32 | 34.41 | 34.41 |
| dfd | i F_DP | 22.81 | 5.51 * | 5.51 * |
| dfd | i es-AUC | 69.73 | 68.57 | 68.57 |
| dfd | AUC | 81.46 | 81.16 | 81.16 |

## L_A, valFPR10: per column (seed values; mean vs Ding)

| set | column | Ding | s17 | mean |
|---|---|---|---|---|
| ffpp | g F_FPR | 0.53 | 0.12 * | 0.12 * |
| ffpp | g F_DP | 3.61 | 1.65 * | 1.65 * |
| ffpp | g es-AUC | 96.45 | 97.47 * | 97.47 * |
| ffpp | r F_FPR | 9.29 | 10.75 | 10.75 |
| ffpp | r F_DP | 4.35 | 2.73 * | 2.73 * |
| ffpp | r es-AUC | 94.86 | 97.06 * | 97.06 * |
| ffpp | i F_FPR | 20.18 | 24.86 | 24.86 |
| ffpp | i F_DP | 9.47 | 11.12 | 11.12 |
| ffpp | i es-AUC | 86.91 | 91.88 * | 91.88 * |
| ffpp | AUC | 97.71 | 98.47 * | 98.47 * |
| dfdc | g F_FPR | 1.76 | 0.40 * | 0.40 * |
| dfdc | g F_DP | 3.67 | 1.84 * | 1.84 * |
| dfdc | g es-AUC | 57.94 | 60.69 * | 60.69 * |
| dfdc | r F_FPR | 17.93 | 14.68 * | 14.68 * |
| dfdc | r F_DP | 7.58 | 5.67 * | 5.67 * |
| dfdc | r es-AUC | 52.33 | 53.49 * | 53.49 * |
| dfdc | i F_FPR | 37.37 | 30.44 * | 30.44 * |
| dfdc | i F_DP | 11.92 | 7.21 * | 7.21 * |
| dfdc | i es-AUC | 39.03 | 40.28 * | 40.28 * |
| dfdc | AUC | 61.86 | 61.77 | 61.77 |
| celebdf | g F_FPR | 6.41 | 6.08 * | 6.08 * |
| celebdf | g F_DP | 10.81 | 21.20 | 21.20 |
| celebdf | g es-AUC | 69.68 | 61.23 | 61.23 |
| celebdf | r F_FPR | 16.60 | 34.85 | 34.85 |
| celebdf | r F_DP | 7.29 | 26.57 | 26.57 |
| celebdf | r es-AUC | 69.96 | 50.96 | 50.96 |
| celebdf | i F_FPR | 23.32 | 37.19 | 37.19 |
| celebdf | i F_DP | 12.01 | 28.24 | 28.24 |
| celebdf | i es-AUC | 63.27 | 48.72 | 48.72 |
| celebdf | AUC | 76.75 | 70.75 | 70.75 |
| dfd | g F_FPR | 4.72 | 2.78 * | 2.78 * |
| dfd | g F_DP | 5.49 | 2.17 * | 2.17 * |
| dfd | g es-AUC | 78.98 | 80.60 * | 80.60 * |
| dfd | r F_FPR | 2.09 | 11.33 | 11.33 |
| dfd | r F_DP | 18.66 | 2.24 * | 2.24 * |
| dfd | r es-AUC | 77.44 | 79.80 * | 79.80 * |
| dfd | i F_FPR | 27.32 | 34.14 | 34.14 |
| dfd | i F_DP | 22.81 | 5.04 * | 5.04 * |
| dfd | i es-AUC | 69.73 | 68.57 | 68.57 |
| dfd | AUC | 81.46 | 81.16 | 81.16 |

## L_H, op05: per column (seed values; mean vs Ding)

| set | column | Ding | s17 | mean |
|---|---|---|---|---|
| ffpp | g F_FPR | 0.53 | 0.24 * | 0.24 * |
| ffpp | g F_DP | 3.61 | 2.41 * | 2.41 * |
| ffpp | g es-AUC | 96.45 | 97.44 * | 97.44 * |
| ffpp | r F_FPR | 9.29 | 8.84 * | 8.84 * |
| ffpp | r F_DP | 4.35 | 3.27 * | 3.27 * |
| ffpp | r es-AUC | 94.86 | 97.57 * | 97.57 * |
| ffpp | i F_FPR | 20.18 | 20.22 | 20.22 |
| ffpp | i F_DP | 9.47 | 15.16 | 15.16 |
| ffpp | i es-AUC | 86.91 | 92.07 * | 92.07 * |
| ffpp | AUC | 97.71 | 98.49 * | 98.49 * |
| dfdc | g F_FPR | 1.76 | 0.31 * | 0.31 * |
| dfdc | g F_DP | 3.67 | 1.75 * | 1.75 * |
| dfdc | g es-AUC | 57.94 | 59.73 * | 59.73 * |
| dfdc | r F_FPR | 17.93 | 6.41 * | 6.41 * |
| dfdc | r F_DP | 7.58 | 4.51 * | 4.51 * |
| dfdc | r es-AUC | 52.33 | 52.38 * | 52.38 * |
| dfdc | i F_FPR | 37.37 | 15.02 * | 15.02 * |
| dfdc | i F_DP | 11.92 | 6.63 * | 6.63 * |
| dfdc | i es-AUC | 39.03 | 40.01 * | 40.01 * |
| dfdc | AUC | 61.86 | 60.46 | 60.46 |
| celebdf | g F_FPR | 6.41 | 4.69 * | 4.69 * |
| celebdf | g F_DP | 10.81 | 28.85 | 28.85 |
| celebdf | g es-AUC | 69.68 | 61.57 | 61.57 |
| celebdf | r F_FPR | 16.60 | 41.33 | 41.33 |
| celebdf | r F_DP | 7.29 | 32.26 | 32.26 |
| celebdf | r es-AUC | 69.96 | 51.01 | 51.01 |
| celebdf | i F_FPR | 23.32 | 47.71 | 47.71 |
| celebdf | i F_DP | 12.01 | 37.54 | 37.54 |
| celebdf | i es-AUC | 63.27 | 48.49 | 48.49 |
| celebdf | AUC | 76.75 | 71.40 | 71.40 |
| dfd | g F_FPR | 4.72 | 0.85 * | 0.85 * |
| dfd | g F_DP | 5.49 | 1.86 * | 1.86 * |
| dfd | g es-AUC | 78.98 | 80.19 * | 80.19 * |
| dfd | r F_FPR | 2.09 | 13.27 | 13.27 |
| dfd | r F_DP | 18.66 | 2.54 * | 2.54 * |
| dfd | r es-AUC | 77.44 | 79.47 * | 79.47 * |
| dfd | i F_FPR | 27.32 | 40.21 | 40.21 |
| dfd | i F_DP | 22.81 | 6.46 * | 6.46 * |
| dfd | i es-AUC | 69.73 | 67.79 | 67.79 |
| dfd | AUC | 81.46 | 80.88 | 80.88 |

## L_H, valFPR10: per column (seed values; mean vs Ding)

| set | column | Ding | s17 | mean |
|---|---|---|---|---|
| ffpp | g F_FPR | 0.53 | 0.85 | 0.85 |
| ffpp | g F_DP | 3.61 | 1.67 * | 1.67 * |
| ffpp | g es-AUC | 96.45 | 97.44 * | 97.44 * |
| ffpp | r F_FPR | 9.29 | 13.74 | 13.74 |
| ffpp | r F_DP | 4.35 | 3.05 * | 3.05 * |
| ffpp | r es-AUC | 94.86 | 97.57 * | 97.57 * |
| ffpp | i F_FPR | 20.18 | 31.40 | 31.40 |
| ffpp | i F_DP | 9.47 | 11.89 | 11.89 |
| ffpp | i es-AUC | 86.91 | 92.07 * | 92.07 * |
| ffpp | AUC | 97.71 | 98.49 * | 98.49 * |
| dfdc | g F_FPR | 1.76 | 0.87 * | 0.87 * |
| dfdc | g F_DP | 3.67 | 2.20 * | 2.20 * |
| dfdc | g es-AUC | 57.94 | 59.73 * | 59.73 * |
| dfdc | r F_FPR | 17.93 | 13.54 * | 13.54 * |
| dfdc | r F_DP | 7.58 | 5.12 * | 5.12 * |
| dfdc | r es-AUC | 52.33 | 52.38 * | 52.38 * |
| dfdc | i F_FPR | 37.37 | 27.62 * | 27.62 * |
| dfdc | i F_DP | 11.92 | 7.57 * | 7.57 * |
| dfdc | i es-AUC | 39.03 | 40.01 * | 40.01 * |
| dfdc | AUC | 61.86 | 60.46 | 60.46 |
| celebdf | g F_FPR | 6.41 | 8.04 | 8.04 |
| celebdf | g F_DP | 10.81 | 23.26 | 23.26 |
| celebdf | g es-AUC | 69.68 | 61.57 | 61.57 |
| celebdf | r F_FPR | 16.60 | 35.27 | 35.27 |
| celebdf | r F_DP | 7.29 | 27.27 | 27.27 |
| celebdf | r es-AUC | 69.96 | 51.01 | 51.01 |
| celebdf | i F_FPR | 23.32 | 38.76 | 38.76 |
| celebdf | i F_DP | 12.01 | 29.49 | 29.49 |
| celebdf | i es-AUC | 63.27 | 48.49 | 48.49 |
| celebdf | AUC | 76.75 | 71.40 | 71.40 |
| dfd | g F_FPR | 4.72 | 3.66 * | 3.66 * |
| dfd | g F_DP | 5.49 | 2.48 * | 2.48 * |
| dfd | g es-AUC | 78.98 | 80.19 * | 80.19 * |
| dfd | r F_FPR | 2.09 | 11.76 | 11.76 |
| dfd | r F_DP | 18.66 | 3.39 * | 3.39 * |
| dfd | r es-AUC | 77.44 | 79.47 * | 79.47 * |
| dfd | i F_FPR | 27.32 | 33.56 | 33.56 |
| dfd | i F_DP | 22.81 | 7.63 * | 7.63 * |
| dfd | i es-AUC | 69.73 | 67.79 | 67.79 |
| dfd | AUC | 81.46 | 80.88 | 80.88 |
