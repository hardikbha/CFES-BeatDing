# Table 3 (ResNet-50) post-hoc variants (Amendment 8): SWA / SWA+TTA / SOUP / SOUP+TTA x op05 / valFPR1

op05 = Ding protocol (PRIMARY). valFPR1 = threshold at the 99th percentile of the model's own FF++ val real logits
(FROZEN secondary rule, protocol deviation). TTA = hflip logit mean. SOUP = uniform average of the arm's 3 SWA models,
BN recomputed on FF++ TRAIN. Wins vs Ding published T3 (36 fairness + 4 AUC = 40). "best seed" rows are selected
POST HOC ON TEST (Amendment 5a) and shown only next to the per-seed rows and the mean.

| arm | variant | op | per-seed totals (FF++/DFDC/CDF/DFD) | mean | mean-row wins | best seed (post hoc on test) |
|---|---|---|---|---|---|---|
| R0swa | SWA | op05 | s37 15 (5/8/0/2); s47 13 (4/7/0/2); s57 14 (5/7/0/2) | 14.00 | 15 | s37: 15 (post hoc) |
| R0swa | SWA | valFPR1 | s37 15 (4/9/0/2); s47 14 (3/8/1/2); s57 15 (4/8/1/2) | 14.67 | 16 | s37: 15 (post hoc) |
| R0swa | SWA+TTA | op05 | s37 15 (4/8/1/2); s47 17 (5/9/1/2); s57 15 (5/7/1/2) | 15.67 | 15 | s47: 17 (post hoc) |
| R0swa | SWA+TTA | valFPR1 | s37 16 (4/8/2/2); s47 16 (4/8/1/3); s57 16 (5/8/1/2) | 16.00 | 16 | s37: 16 (post hoc) |
| R0swa | SOUP | op05 | soup 14 (5/5/2/2) | 14.00 | 14 | - |
| R0swa | SOUP | valFPR1 | soup 15 (3/7/2/3) | 15.00 | 15 | - |
| R0swa | SOUP+TTA | op05 | soup 12 (4/5/2/1) | 12.00 | 12 | - |
| R0swa | SOUP+TTA | valFPR1 | soup 16 (3/8/2/3) | 16.00 | 16 | - |
| R0sbi | SWA | op05 | s37 15 (6/6/3/0); s47 10 (2/5/3/0); s57 16 (4/8/3/1) | 13.67 | 14 | s57: 16 (post hoc) |
| R0sbi | SWA | valFPR1 | s37 17 (5/8/3/1); s47 15 (3/8/3/1); s57 17 (4/10/3/0) | 16.33 | 16 | s37: 17 (post hoc) |
| R0sbi | SWA+TTA | op05 | s37 14 (5/6/3/0); s47 13 (3/7/3/0); s57 15 (3/8/3/1) | 14.00 | 13 | s57: 15 (post hoc) |
| R0sbi | SWA+TTA | valFPR1 | s37 17 (5/9/3/0); s47 14 (3/8/3/0); s57 15 (3/9/3/0) | 15.33 | 14 | s37: 17 (post hoc) |
| R0sbi | SOUP | op05 | soup 16 (7/6/3/0) | 16.00 | 16 | - |
| R0sbi | SOUP | valFPR1 | soup 18 (6/9/3/0) | 18.00 | 18 | - |
| R0sbi | SOUP+TTA | op05 | soup 13 (5/5/3/0) | 13.00 | 13 | - |
| R0sbi | SOUP+TTA | valFPR1 | soup 17 (5/9/3/0) | 17.00 | 17 | - |
| R3sbi | SWA | op05 | s37 15 (6/4/3/2); s47 17 (6/6/3/2); s57 21 (5/8/4/4) | 17.67 | 14 | s57: 21 (post hoc) |
| R3sbi | SWA | valFPR1 | s37 17 (4/8/3/2); s47 21 (6/8/3/4); s57 21 (5/8/4/4) | 19.67 | 18 | s47: 21 (post hoc) |
| R3sbi | SWA+TTA | op05 | s37 14 (5/4/3/2); s47 16 (6/4/4/2); s57 18 (5/6/3/4) | 16.00 | 15 | s57: 18 (post hoc) |
| R3sbi | SWA+TTA | valFPR1 | s37 16 (4/8/3/1); s47 19 (6/7/4/2); s57 21 (5/9/3/4) | 18.67 | 19 | s57: 21 (post hoc) |
| R3sbi | SOUP | op05 | soup 14 (5/6/3/0) | 14.00 | 14 | - |
| R3sbi | SOUP | valFPR1 | soup 15 (3/8/3/1) | 15.00 | 15 | - |
| R3sbi | SOUP+TTA | op05 | soup 16 (5/6/4/1) | 16.00 | 16 | - |
| R3sbi | SOUP+TTA | valFPR1 | soup 18 (4/8/4/2) | 18.00 | 18 | - |
| R7 | SWA | op05 | s47 16 (7/4/3/2); s57 14 (6/4/3/1) | 15.00 | 16 | s47: 16 (post hoc) |
| R7 | SWA | valFPR1 | s47 17 (6/6/3/2); s57 18 (5/8/3/2) | 17.50 | 18 | s57: 18 (post hoc) |
| R7 | SWA+TTA | op05 | s47 17 (5/5/3/4); s57 16 (6/4/3/3) | 16.50 | 15 | s47: 17 (post hoc) |
| R7 | SWA+TTA | valFPR1 | s47 19 (5/7/4/3); s57 16 (5/7/3/1) | 17.50 | 16 | s47: 19 (post hoc) |
| R0bsbi | SWA | op05 | s37 11 (3/5/3/0) | 11.00 | 11 | s37: 11 (post hoc) |
| R0bsbi | SWA | valFPR1 | s37 13 (4/6/3/0) | 13.00 | 13 | s37: 13 (post hoc) |
| R0bsbi | SWA+TTA | op05 | s37 11 (3/5/3/0) | 11.00 | 11 | s37: 11 (post hoc) |
| R0bsbi | SWA+TTA | valFPR1 | s37 11 (4/5/2/0) | 11.00 | 11 | s37: 11 (post hoc) |

## Soup sanity (FF++ val AUC; logged, not selected on)

- R0swa: soup 0.9437 vs mean single-seed 0.9504 -> OK (within 2 pt)
- R0sbi: soup 0.9669 vs mean single-seed 0.9612 -> OK (within 2 pt)
- R3sbi: soup 0.9346 vs mean single-seed 0.9580 -> DEGENERATE (> 2 pt)

## Best-by-mean variant per arm (selected post hoc on test across variants/ops; labelled) - full 40 columns

### R0swa: SWA+TTA, valFPR1 (mean 16.00/40; mean-row 16/40; best seed R0swa_s37_swa 16/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.83 * | 0.99 * |
| ffpp | g F_DP | 5.14 | 9.88 | 9.47 |
| ffpp | g es-AUC | 94.46 | 91.70 | 91.65 |
| ffpp | r F_FPR | 18.67 | 3.26 * | 2.56 * |
| ffpp | r F_DP | 2.43 | 7.33 | 6.55 |
| ffpp | r es-AUC | 92.69 | 94.35 * | 94.15 * |
| ffpp | i F_FPR | 34.30 | 8.62 * | 9.03 * |
| ffpp | i F_DP | 8.61 | 21.97 | 20.87 |
| ffpp | i es-AUC | 82.87 | 81.22 | 80.29 |
| ffpp | AUC | 95.53 | 95.37 | 95.29 |
| dfdc | g F_FPR | 3.97 | 1.35 * | 1.77 * |
| dfdc | g F_DP | 3.61 | 2.31 * | 2.67 * |
| dfdc | g es-AUC | 56.96 | 58.88 * | 59.27 * |
| dfdc | r F_FPR | 11.94 | 2.87 * | 2.49 * |
| dfdc | r F_DP | 5.15 | 3.80 * | 4.54 * |
| dfdc | r es-AUC | 50.99 | 54.23 * | 54.52 * |
| dfdc | i F_FPR | 32.14 | 20.29 * | 18.41 * |
| dfdc | i F_DP | 10.27 | 10.67 | 10.55 |
| dfdc | i es-AUC | 37.55 | 40.08 * | 40.81 * |
| dfdc | AUC | 62.54 | 61.83 | 62.02 |
| celebdf | g F_FPR | 4.53 | 3.39 * | 3.33 * |
| celebdf | g F_DP | 12.00 | 28.81 | 30.55 |
| celebdf | g es-AUC | 68.98 | 68.66 | 68.62 |
| celebdf | r F_FPR | 6.76 | 14.87 | 17.36 |
| celebdf | r F_DP | 8.16 | 23.56 | 25.05 |
| celebdf | r es-AUC | 71.03 | 62.76 | 60.94 |
| celebdf | i F_FPR | 5.56 | 13.85 | 14.07 |
| celebdf | i F_DP | 9.11 | 24.88 | 26.45 |
| celebdf | i es-AUC | 68.25 | 63.31 | 61.81 |
| celebdf | AUC | 77.35 | 76.83 | 77.42 * |
| dfd | g F_FPR | 3.12 | 0.96 * | 1.08 * |
| dfd | g F_DP | 1.76 | 5.06 | 5.18 |
| dfd | g es-AUC | 77.68 | 73.10 | 73.78 |
| dfd | r F_FPR | 6.51 | 3.47 * | 3.39 * |
| dfd | r F_DP | 3.03 | 12.43 | 12.71 |
| dfd | r es-AUC | 70.93 | 66.20 | 66.87 |
| dfd | i F_FPR | 20.99 | 20.92 * | 21.60 |
| dfd | i F_DP | 4.85 | 14.27 | 14.89 |
| dfd | i es-AUC | 66.19 | 51.99 | 52.57 |
| dfd | AUC | 78.63 | 73.97 | 74.93 |

### R0sbi: SOUP, valFPR1 (mean 18.00/40; mean-row 18/40; best seed R0sbi_soup 18/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.21 * | 0.21 * |
| ffpp | g F_DP | 5.14 | 8.77 | 8.77 |
| ffpp | g es-AUC | 94.46 | 93.59 | 93.59 |
| ffpp | r F_FPR | 18.67 | 5.02 * | 5.02 * |
| ffpp | r F_DP | 2.43 | 4.70 | 4.70 |
| ffpp | r es-AUC | 92.69 | 92.69 * | 92.69 * |
| ffpp | i F_FPR | 34.30 | 13.81 * | 13.81 * |
| ffpp | i F_DP | 8.61 | 23.12 | 23.12 |
| ffpp | i es-AUC | 82.87 | 84.08 * | 84.08 * |
| ffpp | AUC | 95.53 | 96.36 * | 96.36 * |
| dfdc | g F_FPR | 3.97 | 2.09 * | 2.09 * |
| dfdc | g F_DP | 3.61 | 4.03 | 4.03 |
| dfdc | g es-AUC | 56.96 | 61.05 * | 61.05 * |
| dfdc | r F_FPR | 11.94 | 5.30 * | 5.30 * |
| dfdc | r F_DP | 5.15 | 3.38 * | 3.38 * |
| dfdc | r es-AUC | 50.99 | 56.43 * | 56.43 * |
| dfdc | i F_FPR | 32.14 | 23.23 * | 23.23 * |
| dfdc | i F_DP | 10.27 | 8.81 * | 8.81 * |
| dfdc | i es-AUC | 37.55 | 42.36 * | 42.36 * |
| dfdc | AUC | 62.54 | 62.76 * | 62.76 * |
| celebdf | g F_FPR | 4.53 | 4.32 * | 4.32 * |
| celebdf | g F_DP | 12.00 | 33.82 | 33.82 |
| celebdf | g es-AUC | 68.98 | 73.32 * | 73.32 * |
| celebdf | r F_FPR | 6.76 | 27.33 | 27.33 |
| celebdf | r F_DP | 8.16 | 27.16 | 27.16 |
| celebdf | r es-AUC | 71.03 | 60.42 | 60.42 |
| celebdf | i F_FPR | 5.56 | 11.31 | 11.31 |
| celebdf | i F_DP | 9.11 | 30.13 | 30.13 |
| celebdf | i es-AUC | 68.25 | 65.08 | 65.08 |
| celebdf | AUC | 77.35 | 81.90 * | 81.90 * |
| dfd | g F_FPR | 3.12 | 8.17 | 8.17 |
| dfd | g F_DP | 1.76 | 4.49 | 4.49 |
| dfd | g es-AUC | 77.68 | 77.01 | 77.01 |
| dfd | r F_FPR | 6.51 | 22.85 | 22.85 |
| dfd | r F_DP | 3.03 | 3.19 | 3.19 |
| dfd | r es-AUC | 70.93 | 68.78 | 68.78 |
| dfd | i F_FPR | 20.99 | 35.64 | 35.64 |
| dfd | i F_DP | 4.85 | 20.58 | 20.58 |
| dfd | i es-AUC | 66.19 | 59.11 | 59.11 |
| dfd | AUC | 78.63 | 77.35 | 77.35 |

### R3sbi: SWA, valFPR1 (mean 19.67/40; mean-row 18/40; best seed R3sbi_s47_swa 21/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.53 * | 0.30 * |
| ffpp | g F_DP | 5.14 | 7.50 | 9.17 |
| ffpp | g es-AUC | 94.46 | 92.88 | 93.98 |
| ffpp | r F_FPR | 18.67 | 2.42 * | 2.32 * |
| ffpp | r F_DP | 2.43 | 5.71 | 6.43 |
| ffpp | r es-AUC | 92.69 | 94.77 * | 95.59 * |
| ffpp | i F_FPR | 34.30 | 6.35 * | 5.84 * |
| ffpp | i F_DP | 8.61 | 22.92 | 23.70 |
| ffpp | i es-AUC | 82.87 | 81.21 | 84.71 * |
| ffpp | AUC | 95.53 | 96.01 * | 96.86 * |
| dfdc | g F_FPR | 3.97 | 2.25 * | 3.14 * |
| dfdc | g F_DP | 3.61 | 4.39 | 5.01 |
| dfdc | g es-AUC | 56.96 | 61.31 * | 62.20 * |
| dfdc | r F_FPR | 11.94 | 6.25 * | 6.45 * |
| dfdc | r F_DP | 5.15 | 3.44 * | 4.44 * |
| dfdc | r es-AUC | 50.99 | 57.10 * | 57.69 * |
| dfdc | i F_FPR | 32.14 | 25.53 * | 28.22 * |
| dfdc | i F_DP | 10.27 | 9.19 * | 10.46 |
| dfdc | i es-AUC | 37.55 | 45.41 * | 46.13 * |
| dfdc | AUC | 62.54 | 62.23 | 63.54 * |
| celebdf | g F_FPR | 4.53 | 3.24 * | 5.94 |
| celebdf | g F_DP | 12.00 | 32.26 | 35.41 |
| celebdf | g es-AUC | 68.98 | 70.93 * | 72.10 * |
| celebdf | r F_FPR | 6.76 | 19.84 | 26.11 |
| celebdf | r F_DP | 8.16 | 26.67 | 29.69 |
| celebdf | r es-AUC | 71.03 | 64.85 | 65.57 |
| celebdf | i F_FPR | 5.56 | 11.82 | 17.76 |
| celebdf | i F_DP | 9.11 | 28.65 | 31.30 |
| celebdf | i es-AUC | 68.25 | 68.16 | 68.48 * |
| celebdf | AUC | 77.35 | 80.07 * | 80.85 * |
| dfd | g F_FPR | 3.12 | 2.19 * | 2.80 * |
| dfd | g F_DP | 1.76 | 1.10 * | 1.55 * |
| dfd | g es-AUC | 77.68 | 77.18 | 77.79 * |
| dfd | r F_FPR | 6.51 | 12.84 | 14.11 |
| dfd | r F_DP | 3.03 | 7.85 | 7.37 |
| dfd | r es-AUC | 70.93 | 67.71 | 67.75 |
| dfd | i F_FPR | 20.99 | 29.18 | 31.89 |
| dfd | i F_DP | 4.85 | 14.41 | 14.43 |
| dfd | i es-AUC | 66.19 | 59.04 | 58.97 |
| dfd | AUC | 78.63 | 77.74 | 78.69 * |

### R7: SWA, valFPR1 (mean 17.50/40; mean-row 18/40; best seed R7_s57_swa 18/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.58 * | 0.86 * |
| ffpp | g F_DP | 5.14 | 9.79 | 10.69 |
| ffpp | g es-AUC | 94.46 | 93.92 | 93.63 |
| ffpp | r F_FPR | 18.67 | 2.85 * | 3.62 * |
| ffpp | r F_DP | 2.43 | 3.70 | 4.04 |
| ffpp | r es-AUC | 92.69 | 95.64 * | 95.87 * |
| ffpp | i F_FPR | 34.30 | 7.03 * | 8.45 * |
| ffpp | i F_DP | 8.61 | 18.25 | 19.03 |
| ffpp | i es-AUC | 82.87 | 83.54 * | 82.86 |
| ffpp | AUC | 95.53 | 96.85 * | 96.77 * |
| dfdc | g F_FPR | 3.97 | 1.82 * | 2.25 * |
| dfdc | g F_DP | 3.61 | 3.94 | 3.87 |
| dfdc | g es-AUC | 56.96 | 60.47 * | 59.71 * |
| dfdc | r F_FPR | 11.94 | 8.75 * | 8.39 * |
| dfdc | r F_DP | 5.15 | 5.62 | 3.67 * |
| dfdc | r es-AUC | 50.99 | 55.17 * | 54.28 * |
| dfdc | i F_FPR | 32.14 | 25.48 * | 25.24 * |
| dfdc | i F_DP | 10.27 | 9.49 * | 8.19 * |
| dfdc | i es-AUC | 37.55 | 43.08 * | 42.86 * |
| dfdc | AUC | 62.54 | 61.71 | 61.18 |
| celebdf | g F_FPR | 4.53 | 0.05 * | 0.02 * |
| celebdf | g F_DP | 12.00 | 28.13 | 27.18 |
| celebdf | g es-AUC | 68.98 | 73.14 * | 71.94 * |
| celebdf | r F_FPR | 6.76 | 18.97 | 18.32 |
| celebdf | r F_DP | 8.16 | 20.94 | 20.54 |
| celebdf | r es-AUC | 71.03 | 63.62 | 63.16 |
| celebdf | i F_FPR | 5.56 | 11.78 | 8.99 |
| celebdf | i F_DP | 9.11 | 22.39 | 21.93 |
| celebdf | i es-AUC | 68.25 | 65.66 | 65.73 |
| celebdf | AUC | 77.35 | 80.36 * | 79.47 * |
| dfd | g F_FPR | 3.12 | 0.78 * | 0.33 * |
| dfd | g F_DP | 1.76 | 3.20 | 3.43 |
| dfd | g es-AUC | 77.68 | 76.42 | 75.50 |
| dfd | r F_FPR | 6.51 | 6.00 * | 5.52 * |
| dfd | r F_DP | 3.03 | 11.90 | 13.07 |
| dfd | r es-AUC | 70.93 | 68.75 | 67.39 |
| dfd | i F_FPR | 20.99 | 21.42 | 21.79 |
| dfd | i F_DP | 4.85 | 17.01 | 16.47 |
| dfd | i es-AUC | 66.19 | 58.42 | 57.58 |
| dfd | AUC | 78.63 | 76.99 | 75.94 |

### R0bsbi: SWA, valFPR1 (mean 13.00/40; mean-row 13/40; best seed R0bsbi_s37_swa 13/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.16 * | 0.16 * |
| ffpp | g F_DP | 5.14 | 9.48 | 9.48 |
| ffpp | g es-AUC | 94.46 | 93.30 | 93.30 |
| ffpp | r F_FPR | 18.67 | 5.49 * | 5.49 * |
| ffpp | r F_DP | 2.43 | 8.29 | 8.29 |
| ffpp | r es-AUC | 92.69 | 91.49 | 91.49 |
| ffpp | i F_FPR | 34.30 | 15.15 * | 15.15 * |
| ffpp | i F_DP | 8.61 | 26.52 | 26.52 |
| ffpp | i es-AUC | 82.87 | 80.62 | 80.62 |
| ffpp | AUC | 95.53 | 95.95 * | 95.95 * |
| dfdc | g F_FPR | 3.97 | 1.57 * | 1.57 * |
| dfdc | g F_DP | 3.61 | 3.84 | 3.84 |
| dfdc | g es-AUC | 56.96 | 60.59 * | 60.59 * |
| dfdc | r F_FPR | 11.94 | 9.51 * | 9.51 * |
| dfdc | r F_DP | 5.15 | 7.75 | 7.75 |
| dfdc | r es-AUC | 50.99 | 55.82 * | 55.82 * |
| dfdc | i F_FPR | 32.14 | 28.09 * | 28.09 * |
| dfdc | i F_DP | 10.27 | 11.73 | 11.73 |
| dfdc | i es-AUC | 37.55 | 41.14 * | 41.14 * |
| dfdc | AUC | 62.54 | 61.98 | 61.98 |
| celebdf | g F_FPR | 4.53 | 4.45 * | 4.45 * |
| celebdf | g F_DP | 12.00 | 21.94 | 21.94 |
| celebdf | g es-AUC | 68.98 | 72.99 * | 72.99 * |
| celebdf | r F_FPR | 6.76 | 30.50 | 30.50 |
| celebdf | r F_DP | 8.16 | 15.02 | 15.02 |
| celebdf | r es-AUC | 71.03 | 58.38 | 58.38 |
| celebdf | i F_FPR | 5.56 | 15.48 | 15.48 |
| celebdf | i F_DP | 9.11 | 26.79 | 26.79 |
| celebdf | i es-AUC | 68.25 | 63.22 | 63.22 |
| celebdf | AUC | 77.35 | 78.87 * | 78.87 * |
| dfd | g F_FPR | 3.12 | 7.42 | 7.42 |
| dfd | g F_DP | 1.76 | 5.27 | 5.27 |
| dfd | g es-AUC | 77.68 | 73.47 | 73.47 |
| dfd | r F_FPR | 6.51 | 27.38 | 27.38 |
| dfd | r F_DP | 3.03 | 11.06 | 11.06 |
| dfd | r es-AUC | 70.93 | 68.04 | 68.04 |
| dfd | i F_FPR | 20.99 | 50.72 | 50.72 |
| dfd | i F_DP | 4.85 | 24.39 | 24.39 |
| dfd | i es-AUC | 66.19 | 60.03 | 60.03 |
| dfd | AUC | 78.63 | 73.83 | 73.83 |

