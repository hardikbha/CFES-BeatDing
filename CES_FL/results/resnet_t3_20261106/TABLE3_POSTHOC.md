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
| R4 | SWA | op05 | s37 16 (5/5/3/3); s47 17 (6/5/3/3); s57 14 (6/4/3/1) | 15.67 | 13 | s47: 17 (post hoc) |
| R4 | SWA | valFPR1 | s37 17 (4/8/3/2); s47 17 (5/7/3/2); s57 17 (5/7/3/2) | 17.00 | 15 | s37: 17 (post hoc) |
| R4 | SWA+TTA | op05 | s37 13 (4/4/3/2); s47 14 (6/4/2/2); s57 14 (6/4/3/1) | 13.67 | 15 | s47: 14 (post hoc) |
| R4 | SWA+TTA | valFPR1 | s37 18 (4/8/3/3); s47 16 (5/7/2/2); s57 18 (5/7/4/2) | 17.33 | 17 | s37: 18 (post hoc) |
| R4 | SOUP | op05 | soup 13 (4/6/3/0) | 13.00 | 13 | - |
| R4 | SOUP | valFPR1 | soup 16 (3/8/4/1) | 16.00 | 16 | - |
| R4 | SOUP+TTA | op05 | soup 15 (4/6/2/3) | 15.00 | 15 | - |
| R4 | SOUP+TTA | valFPR1 | soup 19 (3/8/4/4) | 19.00 | 19 | - |
| R5 | SWA | op05 | s37 14 (5/3/3/3); s47 15 (6/4/4/1); s57 15 (5/4/3/3) | 14.67 | 14 | s47: 15 (post hoc) |
| R5 | SWA | valFPR1 | s37 17 (5/6/3/3); s47 19 (6/7/4/2); s57 17 (5/6/3/3) | 17.67 | 16 | s47: 19 (post hoc) |
| R5 | SWA+TTA | op05 | s37 15 (5/4/3/3); s47 13 (6/3/3/1); s57 15 (5/4/3/3) | 14.33 | 14 | s37: 15 (post hoc) |
| R5 | SWA+TTA | valFPR1 | s37 19 (5/7/4/3); s47 18 (5/7/4/2); s57 18 (5/6/3/4) | 18.33 | 18 | s37: 19 (post hoc) |
| R5 | SOUP | op05 | soup 11 (4/4/3/0) | 11.00 | 11 | - |
| R5 | SOUP | valFPR1 | soup 16 (4/7/4/1) | 16.00 | 16 | - |
| R5 | SOUP+TTA | op05 | soup 11 (4/4/2/1) | 11.00 | 11 | - |
| R5 | SOUP+TTA | valFPR1 | soup 16 (4/6/4/2) | 16.00 | 16 | - |
| R7 | SWA | op05 | s37 16 (5/5/3/3); s47 16 (7/4/3/2); s57 14 (6/4/3/1) | 15.33 | 15 | s37: 16 (post hoc) |
| R7 | SWA | valFPR1 | s37 20 (5/8/4/3); s47 17 (6/6/3/2); s57 18 (5/8/3/2) | 18.33 | 18 | s37: 20 (post hoc) |
| R7 | SWA+TTA | op05 | s37 18 (6/5/3/4); s47 17 (5/5/3/4); s57 16 (6/4/3/3) | 17.00 | 15 | s37: 18 (post hoc) |
| R7 | SWA+TTA | valFPR1 | s37 20 (5/8/3/4); s47 19 (5/7/4/3); s57 16 (5/7/3/1) | 18.33 | 16 | s37: 20 (post hoc) |
| R7 | SOUP | op05 | soup 10 (4/4/2/0) | 10.00 | 10 | - |
| R7 | SOUP | valFPR1 | soup 19 (4/9/4/2) | 19.00 | 19 | - |
| R7 | SOUP+TTA | op05 | soup 12 (4/4/2/2) | 12.00 | 12 | - |
| R7 | SOUP+TTA | valFPR1 | soup 20 (4/8/4/4) | 20.00 | 20 | - |
| R0bsbi | SWA | op05 | s37 11 (3/5/3/0) | 11.00 | 11 | s37: 11 (post hoc) |
| R0bsbi | SWA | valFPR1 | s37 13 (4/6/3/0) | 13.00 | 13 | s37: 13 (post hoc) |
| R0bsbi | SWA+TTA | op05 | s37 11 (3/5/3/0) | 11.00 | 11 | s37: 11 (post hoc) |
| R0bsbi | SWA+TTA | valFPR1 | s37 11 (4/5/2/0) | 11.00 | 11 | s37: 11 (post hoc) |
| R6 | SWA | op05 | s37 14 (5/4/3/2) | 14.00 | 14 | s37: 14 (post hoc) |
| R6 | SWA | valFPR1 | s37 17 (4/8/3/2) | 17.00 | 17 | s37: 17 (post hoc) |
| R6 | SWA+TTA | op05 | s37 14 (6/4/2/2) | 14.00 | 14 | s37: 14 (post hoc) |
| R6 | SWA+TTA | valFPR1 | s37 16 (5/8/2/1) | 16.00 | 16 | s37: 16 (post hoc) |
| R6b | SWA | op05 | s37 15 (5/5/3/2); s47 18 (7/5/3/3); s57 14 (6/5/3/0) | 15.67 | 15 | s47: 18 (post hoc) |
| R6b | SWA | valFPR1 | s37 16 (4/7/3/2); s47 19 (6/8/3/2); s57 17 (5/7/3/2) | 17.33 | 16 | s47: 19 (post hoc) |
| R6b | SWA+TTA | op05 | s37 13 (5/3/3/2); s47 18 (5/5/3/5); s57 17 (5/6/3/3) | 16.00 | 14 | s47: 18 (post hoc) |
| R6b | SWA+TTA | valFPR1 | s37 16 (4/7/3/2); s47 20 (5/9/3/3); s57 18 (5/7/3/3) | 18.00 | 16 | s47: 20 (post hoc) |
| R6b | SOUP | op05 | soup 15 (5/6/3/1) | 15.00 | 15 | - |
| R6b | SOUP | valFPR1 | soup 16 (3/8/3/2) | 16.00 | 16 | - |
| R6b | SOUP+TTA | op05 | soup 13 (4/5/3/1) | 13.00 | 13 | - |
| R6b | SOUP+TTA | valFPR1 | soup 16 (3/7/4/2) | 16.00 | 16 | - |

## Soup sanity (FF++ val AUC; logged, not selected on)

- R0swa: soup 0.9437 vs mean single-seed 0.9504 -> OK (within 2 pt)
- R0sbi: soup 0.9669 vs mean single-seed 0.9612 -> OK (within 2 pt)
- R3sbi: soup 0.9346 vs mean single-seed 0.9580 -> DEGENERATE (> 2 pt)
- R4: soup 0.9218 vs mean single-seed 0.9598 -> DEGENERATE (> 2 pt)
- R5: soup 0.9547 vs mean single-seed 0.9675 -> OK (within 2 pt)
- R7: soup 0.9422 vs mean single-seed 0.9619 -> OK (within 2 pt)
- R6b: soup 0.9411 vs mean single-seed 0.9571 -> OK (within 2 pt)

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

### R4: SOUP+TTA, valFPR1 (mean 19.00/40; mean-row 19/40; best seed R4_soup 19/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.11 * | 0.11 * |
| ffpp | g F_DP | 5.14 | 7.67 | 7.67 |
| ffpp | g es-AUC | 94.46 | 90.03 | 90.03 |
| ffpp | r F_FPR | 18.67 | 6.44 * | 6.44 * |
| ffpp | r F_DP | 2.43 | 6.91 | 6.91 |
| ffpp | r es-AUC | 92.69 | 90.81 | 90.81 |
| ffpp | i F_FPR | 34.30 | 14.39 * | 14.39 * |
| ffpp | i F_DP | 8.61 | 21.70 | 21.70 |
| ffpp | i es-AUC | 82.87 | 77.73 | 77.73 |
| ffpp | AUC | 95.53 | 93.77 | 93.77 |
| dfdc | g F_FPR | 3.97 | 1.42 * | 1.42 * |
| dfdc | g F_DP | 3.61 | 4.03 | 4.03 |
| dfdc | g es-AUC | 56.96 | 60.06 * | 60.06 * |
| dfdc | r F_FPR | 11.94 | 4.67 * | 4.67 * |
| dfdc | r F_DP | 5.15 | 4.35 * | 4.35 * |
| dfdc | r es-AUC | 50.99 | 53.37 * | 53.37 * |
| dfdc | i F_FPR | 32.14 | 20.86 * | 20.86 * |
| dfdc | i F_DP | 10.27 | 8.77 * | 8.77 * |
| dfdc | i es-AUC | 37.55 | 38.90 * | 38.90 * |
| dfdc | AUC | 62.54 | 60.55 | 60.55 |
| celebdf | g F_FPR | 4.53 | 3.73 * | 3.73 * |
| celebdf | g F_DP | 12.00 | 26.03 | 26.03 |
| celebdf | g es-AUC | 68.98 | 74.99 * | 74.99 * |
| celebdf | r F_FPR | 6.76 | 19.03 | 19.03 |
| celebdf | r F_DP | 8.16 | 22.97 | 22.97 |
| celebdf | r es-AUC | 71.03 | 61.21 | 61.21 |
| celebdf | i F_FPR | 5.56 | 5.55 * | 5.55 * |
| celebdf | i F_DP | 9.11 | 38.03 | 38.03 |
| celebdf | i es-AUC | 68.25 | 68.11 | 68.11 |
| celebdf | AUC | 77.35 | 81.33 * | 81.33 * |
| dfd | g F_FPR | 3.12 | 4.56 | 4.56 |
| dfd | g F_DP | 1.76 | 3.28 | 3.28 |
| dfd | g es-AUC | 77.68 | 77.70 * | 77.70 * |
| dfd | r F_FPR | 6.51 | 17.21 | 17.21 |
| dfd | r F_DP | 3.03 | 1.45 * | 1.45 * |
| dfd | r es-AUC | 70.93 | 71.21 * | 71.21 * |
| dfd | i F_FPR | 20.99 | 31.39 | 31.39 |
| dfd | i F_DP | 4.85 | 11.00 | 11.00 |
| dfd | i es-AUC | 66.19 | 59.65 | 59.65 |
| dfd | AUC | 78.63 | 78.75 * | 78.75 * |

### R5: SWA+TTA, valFPR1 (mean 18.33/40; mean-row 18/40; best seed R5_s37_swa 19/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.51 * | 0.28 * |
| ffpp | g F_DP | 5.14 | 7.47 | 7.80 |
| ffpp | g es-AUC | 94.46 | 93.96 | 94.23 |
| ffpp | r F_FPR | 18.67 | 3.21 * | 3.82 * |
| ffpp | r F_DP | 2.43 | 5.92 | 5.90 |
| ffpp | r es-AUC | 92.69 | 94.31 * | 93.73 * |
| ffpp | i F_FPR | 34.30 | 8.27 * | 8.46 * |
| ffpp | i F_DP | 8.61 | 19.07 | 21.24 |
| ffpp | i es-AUC | 82.87 | 80.97 | 81.15 |
| ffpp | AUC | 95.53 | 96.53 * | 96.66 * |
| dfdc | g F_FPR | 3.97 | 2.66 * | 2.80 * |
| dfdc | g F_DP | 3.61 | 4.88 | 5.56 |
| dfdc | g es-AUC | 56.96 | 60.94 * | 61.37 * |
| dfdc | r F_FPR | 11.94 | 8.06 * | 8.29 * |
| dfdc | r F_DP | 5.15 | 6.46 | 6.88 |
| dfdc | r es-AUC | 50.99 | 55.07 * | 55.26 * |
| dfdc | i F_FPR | 32.14 | 27.98 * | 29.64 * |
| dfdc | i F_DP | 10.27 | 12.28 | 12.58 |
| dfdc | i es-AUC | 37.55 | 39.81 * | 39.86 * |
| dfdc | AUC | 62.54 | 62.37 | 62.69 * |
| celebdf | g F_FPR | 4.53 | 2.02 * | 0.95 * |
| celebdf | g F_DP | 12.00 | 25.09 | 26.49 |
| celebdf | g es-AUC | 68.98 | 73.32 * | 75.20 * |
| celebdf | r F_FPR | 6.76 | 24.64 | 20.47 |
| celebdf | r F_DP | 8.16 | 17.79 | 17.88 |
| celebdf | r es-AUC | 71.03 | 62.32 | 63.15 |
| celebdf | i F_FPR | 5.56 | 12.25 | 4.57 * |
| celebdf | i F_DP | 9.11 | 23.88 | 27.33 |
| celebdf | i es-AUC | 68.25 | 65.70 | 68.05 |
| celebdf | AUC | 77.35 | 80.28 * | 81.19 * |
| dfd | g F_FPR | 3.12 | 2.29 * | 0.13 * |
| dfd | g F_DP | 1.76 | 0.82 * | 0.95 * |
| dfd | g es-AUC | 77.68 | 78.19 * | 77.85 * |
| dfd | r F_FPR | 6.51 | 11.43 | 6.96 |
| dfd | r F_DP | 3.03 | 9.16 | 11.96 |
| dfd | r es-AUC | 70.93 | 69.68 | 69.49 |
| dfd | i F_FPR | 20.99 | 34.17 | 30.15 |
| dfd | i F_DP | 4.85 | 16.64 | 22.10 |
| dfd | i es-AUC | 66.19 | 58.14 | 55.53 |
| dfd | AUC | 78.63 | 78.81 * | 78.43 |

### R7: SOUP+TTA, valFPR1 (mean 20.00/40; mean-row 20/40; best seed R7_soup 20/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.31 * | 0.31 * |
| ffpp | g F_DP | 5.14 | 9.38 | 9.38 |
| ffpp | g es-AUC | 94.46 | 90.91 | 90.91 |
| ffpp | r F_FPR | 18.67 | 4.20 * | 4.20 * |
| ffpp | r F_DP | 2.43 | 5.81 | 5.81 |
| ffpp | r es-AUC | 92.69 | 93.22 * | 93.22 * |
| ffpp | i F_FPR | 34.30 | 9.00 * | 9.00 * |
| ffpp | i F_DP | 8.61 | 18.65 | 18.65 |
| ffpp | i es-AUC | 82.87 | 80.51 | 80.51 |
| ffpp | AUC | 95.53 | 94.91 | 94.91 |
| dfdc | g F_FPR | 3.97 | 1.22 * | 1.22 * |
| dfdc | g F_DP | 3.61 | 3.80 | 3.80 |
| dfdc | g es-AUC | 56.96 | 59.94 * | 59.94 * |
| dfdc | r F_FPR | 11.94 | 2.36 * | 2.36 * |
| dfdc | r F_DP | 5.15 | 3.84 * | 3.84 * |
| dfdc | r es-AUC | 50.99 | 52.82 * | 52.82 * |
| dfdc | i F_FPR | 32.14 | 16.18 * | 16.18 * |
| dfdc | i F_DP | 10.27 | 7.57 * | 7.57 * |
| dfdc | i es-AUC | 37.55 | 39.53 * | 39.53 * |
| dfdc | AUC | 62.54 | 60.42 | 60.42 |
| celebdf | g F_FPR | 4.53 | 1.01 * | 1.01 * |
| celebdf | g F_DP | 12.00 | 27.34 | 27.34 |
| celebdf | g es-AUC | 68.98 | 75.03 * | 75.03 * |
| celebdf | r F_FPR | 6.76 | 13.09 | 13.09 |
| celebdf | r F_DP | 8.16 | 26.86 | 26.86 |
| celebdf | r es-AUC | 71.03 | 59.93 | 59.93 |
| celebdf | i F_FPR | 5.56 | 2.42 * | 2.42 * |
| celebdf | i F_DP | 9.11 | 39.17 | 39.17 |
| celebdf | i es-AUC | 68.25 | 65.47 | 65.47 |
| celebdf | AUC | 77.35 | 80.95 * | 80.95 * |
| dfd | g F_FPR | 3.12 | 0.38 * | 0.38 * |
| dfd | g F_DP | 1.76 | 0.96 * | 0.96 * |
| dfd | g es-AUC | 77.68 | 77.85 * | 77.85 * |
| dfd | r F_FPR | 6.51 | 10.44 | 10.44 |
| dfd | r F_DP | 3.03 | 4.11 | 4.11 |
| dfd | r es-AUC | 70.93 | 71.86 * | 71.86 * |
| dfd | i F_FPR | 20.99 | 25.70 | 25.70 |
| dfd | i F_DP | 4.85 | 15.26 | 15.26 |
| dfd | i es-AUC | 66.19 | 59.19 | 59.19 |
| dfd | AUC | 78.63 | 78.33 | 78.33 |

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

### R6: SWA, valFPR1 (mean 17.00/40; mean-row 17/40; best seed R6_s37_swa 17/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 1.00 * | 1.00 * |
| ffpp | g F_DP | 5.14 | 6.75 | 6.75 |
| ffpp | g es-AUC | 94.46 | 92.08 | 92.08 |
| ffpp | r F_FPR | 18.67 | 2.16 * | 2.16 * |
| ffpp | r F_DP | 2.43 | 4.98 | 4.98 |
| ffpp | r es-AUC | 92.69 | 94.49 * | 94.49 * |
| ffpp | i F_FPR | 34.30 | 5.80 * | 5.80 * |
| ffpp | i F_DP | 8.61 | 21.60 | 21.60 |
| ffpp | i es-AUC | 82.87 | 80.15 | 80.15 |
| ffpp | AUC | 95.53 | 95.31 | 95.31 |
| dfdc | g F_FPR | 3.97 | 0.77 * | 0.77 * |
| dfdc | g F_DP | 3.61 | 2.78 * | 2.78 * |
| dfdc | g es-AUC | 56.96 | 59.13 * | 59.13 * |
| dfdc | r F_FPR | 11.94 | 7.97 * | 7.97 * |
| dfdc | r F_DP | 5.15 | 6.61 | 6.61 |
| dfdc | r es-AUC | 50.99 | 54.46 * | 54.46 * |
| dfdc | i F_FPR | 32.14 | 22.19 * | 22.19 * |
| dfdc | i F_DP | 10.27 | 9.79 * | 9.79 * |
| dfdc | i es-AUC | 37.55 | 40.97 * | 40.97 * |
| dfdc | AUC | 62.54 | 60.66 | 60.66 |
| celebdf | g F_FPR | 4.53 | 4.09 * | 4.09 * |
| celebdf | g F_DP | 12.00 | 20.60 | 20.60 |
| celebdf | g es-AUC | 68.98 | 72.33 * | 72.33 * |
| celebdf | r F_FPR | 6.76 | 19.84 | 19.84 |
| celebdf | r F_DP | 8.16 | 16.12 | 16.12 |
| celebdf | r es-AUC | 71.03 | 63.99 | 63.99 |
| celebdf | i F_FPR | 5.56 | 11.82 | 11.82 |
| celebdf | i F_DP | 9.11 | 21.81 | 21.81 |
| celebdf | i es-AUC | 68.25 | 67.06 | 67.06 |
| celebdf | AUC | 77.35 | 78.18 * | 78.18 * |
| dfd | g F_FPR | 3.12 | 1.63 * | 1.63 * |
| dfd | g F_DP | 1.76 | 3.72 | 3.72 |
| dfd | g es-AUC | 77.68 | 76.41 | 76.41 |
| dfd | r F_FPR | 6.51 | 5.13 * | 5.13 * |
| dfd | r F_DP | 3.03 | 11.41 | 11.41 |
| dfd | r es-AUC | 70.93 | 70.39 | 70.39 |
| dfd | i F_FPR | 20.99 | 24.76 | 24.76 |
| dfd | i F_DP | 4.85 | 17.38 | 17.38 |
| dfd | i es-AUC | 66.19 | 55.75 | 55.75 |
| dfd | AUC | 78.63 | 77.15 | 77.15 |

### R6b: SWA+TTA, valFPR1 (mean 18.00/40; mean-row 16/40; best seed R6b_s47_swa 20/40, post hoc)

| set | column | Ding | mean | best seed (post hoc) |
|---|---|---|---|---|
| ffpp | g F_FPR | 3.94 | 0.12 * | 0.18 * |
| ffpp | g F_DP | 5.14 | 9.41 | 9.50 |
| ffpp | g es-AUC | 94.46 | 93.10 | 94.14 |
| ffpp | r F_FPR | 18.67 | 3.74 * | 3.56 * |
| ffpp | r F_DP | 2.43 | 5.97 | 5.98 |
| ffpp | r es-AUC | 92.69 | 93.84 * | 93.96 * |
| ffpp | i F_FPR | 34.30 | 8.80 * | 8.13 * |
| ffpp | i F_DP | 8.61 | 22.44 | 22.34 |
| ffpp | i es-AUC | 82.87 | 80.72 | 82.50 |
| ffpp | AUC | 95.53 | 96.06 * | 96.72 * |
| dfdc | g F_FPR | 3.97 | 1.59 * | 1.34 * |
| dfdc | g F_DP | 3.61 | 4.23 | 4.42 |
| dfdc | g es-AUC | 56.96 | 61.00 * | 62.04 * |
| dfdc | r F_FPR | 11.94 | 5.59 * | 6.25 * |
| dfdc | r F_DP | 5.15 | 2.78 * | 2.81 * |
| dfdc | r es-AUC | 50.99 | 55.55 * | 55.94 * |
| dfdc | i F_FPR | 32.14 | 23.77 * | 20.74 * |
| dfdc | i F_DP | 10.27 | 11.47 | 8.93 * |
| dfdc | i es-AUC | 37.55 | 42.70 * | 42.16 * |
| dfdc | AUC | 62.54 | 62.10 | 62.77 * |
| celebdf | g F_FPR | 4.53 | 1.57 * | 2.65 * |
| celebdf | g F_DP | 12.00 | 30.86 | 32.86 |
| celebdf | g es-AUC | 68.98 | 71.44 * | 72.77 * |
| celebdf | r F_FPR | 6.76 | 22.67 | 25.15 |
| celebdf | r F_DP | 8.16 | 25.22 | 26.82 |
| celebdf | r es-AUC | 71.03 | 62.69 | 63.43 |
| celebdf | i F_FPR | 5.56 | 12.01 | 16.44 |
| celebdf | i F_DP | 9.11 | 27.36 | 28.37 |
| celebdf | i es-AUC | 68.25 | 66.18 | 66.70 |
| celebdf | AUC | 77.35 | 80.21 * | 81.57 * |
| dfd | g F_FPR | 3.12 | 1.56 * | 1.54 * |
| dfd | g F_DP | 1.76 | 2.12 | 1.77 |
| dfd | g es-AUC | 77.68 | 77.36 | 78.30 * |
| dfd | r F_FPR | 6.51 | 7.32 | 10.07 |
| dfd | r F_DP | 3.03 | 11.16 | 9.36 |
| dfd | r es-AUC | 70.93 | 69.78 | 70.74 |
| dfd | i F_FPR | 20.99 | 28.41 | 31.25 |
| dfd | i F_DP | 4.85 | 17.88 | 15.91 |
| dfd | i es-AUC | 66.19 | 57.09 | 58.58 |
| dfd | AUC | 78.63 | 78.16 | 79.13 * |

