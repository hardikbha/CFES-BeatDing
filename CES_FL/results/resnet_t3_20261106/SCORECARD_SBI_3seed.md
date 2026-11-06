# SBI arms (Amendment 5): R0-SWA vs R0-SBI-SWA vs R3-SBI-SWA, ResNet-50, seeds 37/47/57, vs Ding Table 3

Primary: SWA model at op05 (prob >= 0.5). Secondary: valFPR10 (one global FF++-val-real 90th-percentile threshold)
and raw epoch 49 (labelled). 36 fairness + 4 AUC = 40 columns. No test-based selection of seeds, epochs or thresholds.

## Run status and kill rule (val-AUC rule only; aux>CE rule dropped, aux/CE logged)

| run | status | val AUC ep10 | ep15 | aux>CE (diagnostic) |
|---|---|---|---|---|
| R0swa_s37 | DONE | 0.944050406674476 | 0.944978236541512 | 0/0 |
| R0swa_s47 | DONE | 0.9438000653881549 | 0.9537041512847966 | 0/0 |
| R0swa_s57 | DONE | 0.9429914240708285 | 0.9374411545627427 | 0/0 |
| R0sbi_s37 | DONE | 0.9332845033837589 | 0.953455101097781 | 0/0 |
| R0sbi_s47 | DONE | 0.9468255012609852 | 0.9313660216265676 | 0/0 |
| R0sbi_s57 | DONE | 0.9435969496453367 | 0.9431966342636318 | 0/0 |
| R3sbi_s37 | DONE | 0.9417816238535841 | 0.9138362097419169 | 1934/12350 |
| R3sbi_s47 | DONE | 0.9486648701909248 | 0.9601593781184352 | 1341/12350 |
| R3sbi_s57 | running (epoch 50/50) | 0.9432143481461099 | 0.8978277718972508 | 1398/12350 |
- R3sbi_s37 val-AUC kill rule vs R0sbi_s37: not triggered
- R3sbi_s47 val-AUC kill rule vs R0sbi_s47: not triggered
- R3sbi_s57 val-AUC kill rule vs R0sbi_s57: not triggered

## SWA (primary), op05 (primary)

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD | Celeb-DF wins/10 |
|---|---|---|---|---|---|---|---|
| R0swa_s37 | 15 | 15 | 95.01 | 60.64 | 76.45 | 73.84 | 0 |
| R0swa_s47 | 13 | 13 | 95.08 | 60.86 | 75.61 | 72.63 | 0 |
| R0swa_s57 | 14 | 14 | 95.44 | 60.71 | 75.09 | 71.93 | 0 |
| R0sbi_s37 | 15 | 12 | 95.97 | 62.86 | 79.65 | 73.78 | 3 |
| R0sbi_s47 | 10 | 9 | 95.10 | 62.49 | 78.71 | 71.06 | 3 |
| R0sbi_s57 | 16 | 14 | 95.45 | 62.64 | 77.99 | 72.81 | 3 |
| R3sbi_s37 | 15 | 14 | 94.63 | 60.88 | 79.49 | 75.76 | 3 |
| R3sbi_s47 | 17 | 13 | 96.86 | 63.54 | 80.85 | 78.69 | 3 |
| R3sbi_s57 | 21 | 18 | 96.55 | 62.28 | 79.87 | 78.78 | 4 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 | - |
| **R0swa mean (n=3)** | 14.00 ± 1.00 | - | 95.18 | 60.74 | 75.72 | 72.80 | 0.00 |  (wins of the mean-metric row: 15/40)
| **R0sbi mean (n=3)** | 13.67 ± 3.21 | - | 95.51 | 62.66 | 78.79 | 72.55 | 3.00 |  (wins of the mean-metric row: 14/40)
| **R3sbi mean (n=3)** | 17.67 ± 3.06 | - | 96.01 | 62.23 | 80.07 | 77.74 | 3.33 |  (wins of the mean-metric row: 14/40)
- SBI effect (R0sbi - R0swa), seed 37: wins +0; AUC +0.96, +2.22, +3.20, -0.06
- SBI effect (R0sbi - R0swa), seed 47: wins -3; AUC +0.01, +1.63, +3.10, -1.56
- SBI effect (R0sbi - R0swa), seed 57: wins +2; AUC +0.01, +1.92, +2.91, +0.88
- fairness term (R3sbi - R0sbi), seed 37: wins +0; AUC -1.34, -1.99, -0.16, +1.98
- fairness term (R3sbi - R0sbi), seed 47: wins +7; AUC +1.76, +1.05, +2.14, +7.63
- fairness term (R3sbi - R0sbi), seed 57: wins +5; AUC +1.10, -0.35, +1.88, +5.97
- **Amendment-4/5 criteria (op05)**: target_R0sbi: NOT MET; target_R3sbi: NOT MET; method_evidence_R3sbi_vs_R0sbi: MET

### Celeb-DF per column (SWA, op05): seed values and mean vs Ding

| column | Ding | R0swa (s37 / s47 / s57) | R0swa mean | R0sbi (s37 / s47 / s57) | R0sbi mean | R3sbi (s37 / s47 / s57) | R3sbi mean |
|---|---|---|---|---|---|---|---|
| g F_FPR | 4.53 | 7.66 / 6.89 / 4.74 | 6.43 | 4.02 / 3.36 / 2.90 | 3.43 * | 1.89 / 5.82 / 2.61 | 3.44 * |
| g F_DP | 12.00 | 32.59 / 31.38 / 30.28 | 31.41 | 26.44 / 24.13 / 23.69 | 24.75 | 29.35 / 31.08 / 32.06 | 30.83 |
| g es-AUC | 68.98 | 67.81 / 68.57 / 66.93 | 67.77 | 72.28 / 71.08 / 70.43 | 71.26 * | 70.86 / 72.10 / 69.82 | 70.93 * |
| r F_FPR | 6.76 | 33.13 / 26.34 / 38.55 | 32.68 | 30.56 / 29.49 / 34.09 | 31.38 | 26.95 / 35.17 / 25.36 | 29.16 |
| r F_DP | 8.16 | 27.19 / 26.97 / 26.54 | 26.90 | 20.23 / 18.43 / 18.17 | 18.94 | 23.98 / 25.76 / 27.13 | 25.63 |
| r es-AUC | 71.03 | 59.45 / 67.69 / 61.92 | 63.02 | 64.40 / 59.58 / 59.50 | 61.16 | 64.42 / 65.57 / 64.56 | 64.85 |
| i F_FPR | 5.56 | 34.72 / 28.18 / 42.27 | 35.06 | 20.31 / 21.35 / 29.28 | 23.65 | 20.20 / 31.52 / 15.33 | 22.35 |
| i F_DP | 9.11 | 28.67 / 28.32 / 27.82 | 28.27 | 21.59 / 19.68 / 19.38 | 20.21 | 25.36 / 27.20 / 28.58 | 27.05 |
| i es-AUC | 68.25 | 59.51 / 66.06 / 60.94 | 62.17 | 65.90 / 63.57 / 62.50 | 63.99 | 66.86 / 68.48 / 69.14 | 68.16 |
| AUC | 77.35 | 76.45 / 75.61 / 75.09 | 75.72 | 79.65 / 78.71 / 77.99 | 78.79 * | 79.49 / 80.85 / 79.87 | 80.07 * |

## SWA (primary), valFPR10

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD | Celeb-DF wins/10 |
|---|---|---|---|---|---|---|---|
| R0swa_s37 | 15 | 15 | 95.01 | 60.64 | 76.45 | 73.84 | 0 |
| R0swa_s47 | 12 | 12 | 95.08 | 60.86 | 75.61 | 72.63 | 0 |
| R0swa_s57 | 14 | 14 | 95.44 | 60.71 | 75.09 | 71.93 | 0 |
| R0sbi_s37 | 11 | 8 | 95.97 | 62.86 | 79.65 | 73.78 | 2 |
| R0sbi_s47 | 8 | 7 | 95.10 | 62.49 | 78.71 | 71.06 | 3 |
| R0sbi_s57 | 11 | 9 | 95.45 | 62.64 | 77.99 | 72.81 | 2 |
| R3sbi_s37 | 13 | 12 | 94.63 | 60.88 | 79.49 | 75.76 | 3 |
| R3sbi_s47 | 17 | 13 | 96.86 | 63.54 | 80.85 | 78.69 | 3 |
| R3sbi_s57 | 14 | 11 | 96.55 | 62.28 | 79.87 | 78.78 | 3 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 | - |
| **R0swa mean (n=3)** | 13.67 ± 1.53 | - | 95.18 | 60.74 | 75.72 | 72.80 | 0.00 |  (wins of the mean-metric row: 15/40)
| **R0sbi mean (n=3)** | 10.00 ± 1.73 | - | 95.51 | 62.66 | 78.79 | 72.55 | 2.33 |  (wins of the mean-metric row: 12/40)
| **R3sbi mean (n=3)** | 14.67 ± 2.08 | - | 96.01 | 62.23 | 80.07 | 77.74 | 3.00 |  (wins of the mean-metric row: 13/40)
- SBI effect (R0sbi - R0swa), seed 37: wins -4; AUC +0.96, +2.22, +3.20, -0.06
- SBI effect (R0sbi - R0swa), seed 47: wins -4; AUC +0.01, +1.63, +3.10, -1.56
- SBI effect (R0sbi - R0swa), seed 57: wins -3; AUC +0.01, +1.92, +2.91, +0.88
- fairness term (R3sbi - R0sbi), seed 37: wins +2; AUC -1.34, -1.99, -0.16, +1.98
- fairness term (R3sbi - R0sbi), seed 47: wins +9; AUC +1.76, +1.05, +2.14, +7.63
- fairness term (R3sbi - R0sbi), seed 57: wins +3; AUC +1.10, -0.35, +1.88, +5.97
- **Amendment-4/5 criteria (valFPR10)**: target_R0sbi: NOT MET; target_R3sbi: NOT MET; method_evidence_R3sbi_vs_R0sbi: MET

## raw epoch 49 (secondary), op05 (primary)

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD | Celeb-DF wins/10 |
|---|---|---|---|---|---|---|---|
| R0swa_s37 | 8 | 8 | 94.44 | 59.20 | 74.35 | 70.39 | 0 |
| R0swa_s47 | 9 | 9 | 94.72 | 58.69 | 70.38 | 69.76 | 4 |
| R0swa_s57 | 12 | 12 | 95.39 | 59.00 | 71.85 | 68.78 | 5 |
| R0sbi_s37 | 11 | 11 | 93.58 | 59.45 | 68.25 | 67.54 | 3 |
| R0sbi_s47 | 8 | 8 | 92.01 | 58.08 | 70.82 | 74.63 | 4 |
| R0sbi_s57 | 9 | 9 | 92.95 | 61.21 | 70.80 | 74.65 | 3 |
| R3sbi_s37 | 8 | 8 | 92.06 | 60.19 | 74.64 | 72.10 | 2 |
| R3sbi_s47 | 6 | 5 | 92.44 | 62.75 | 76.59 | 77.24 | 0 |
| R3sbi_s57 | 7 | 5 | 95.74 | 59.71 | 79.48 | 72.74 | 1 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 | - |
| **R0swa mean (n=3)** | 9.67 ± 2.08 | - | 94.85 | 58.96 | 72.20 | 69.64 | 3.00 |  (wins of the mean-metric row: 11/40)
| **R0sbi mean (n=3)** | 9.33 ± 1.53 | - | 92.85 | 59.58 | 69.95 | 72.27 | 3.33 |  (wins of the mean-metric row: 8/40)
| **R3sbi mean (n=3)** | 7.00 ± 1.00 | - | 93.41 | 60.88 | 76.90 | 74.03 | 1.00 |  (wins of the mean-metric row: 2/40)
- SBI effect (R0sbi - R0swa), seed 37: wins +3; AUC -0.86, +0.25, -6.11, -2.84
- SBI effect (R0sbi - R0swa), seed 47: wins -1; AUC -2.70, -0.61, +0.43, +4.88
- SBI effect (R0sbi - R0swa), seed 57: wins -3; AUC -2.44, +2.21, -1.05, +5.87
- fairness term (R3sbi - R0sbi), seed 37: wins -3; AUC -1.51, +0.74, +6.39, +4.55
- fairness term (R3sbi - R0sbi), seed 47: wins -2; AUC +0.42, +4.66, +5.78, +2.61
- fairness term (R3sbi - R0sbi), seed 57: wins -2; AUC +2.79, -1.50, +8.68, -1.90

## raw epoch 49 (secondary), valFPR10

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD | Celeb-DF wins/10 |
|---|---|---|---|---|---|---|---|
| R0swa_s37 | 8 | 8 | 94.44 | 59.20 | 74.35 | 70.39 | 0 |
| R0swa_s47 | 10 | 10 | 94.72 | 58.69 | 70.38 | 69.76 | 2 |
| R0swa_s57 | 12 | 12 | 95.39 | 59.00 | 71.85 | 68.78 | 5 |
| R0sbi_s37 | 4 | 4 | 93.58 | 59.45 | 68.25 | 67.54 | 0 |
| R0sbi_s47 | 4 | 4 | 92.01 | 58.08 | 70.82 | 74.63 | 0 |
| R0sbi_s57 | 6 | 6 | 92.95 | 61.21 | 70.80 | 74.65 | 1 |
| R3sbi_s37 | 5 | 5 | 92.06 | 60.19 | 74.64 | 72.10 | 0 |
| R3sbi_s47 | 10 | 9 | 92.44 | 62.75 | 76.59 | 77.24 | 0 |
| R3sbi_s57 | 7 | 5 | 95.74 | 59.71 | 79.48 | 72.74 | 1 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 | - |
| **R0swa mean (n=3)** | 10.00 ± 2.00 | - | 94.85 | 58.96 | 72.20 | 69.64 | 2.33 |  (wins of the mean-metric row: 8/40)
| **R0sbi mean (n=3)** | 4.67 ± 1.15 | - | 92.85 | 59.58 | 69.95 | 72.27 | 0.33 |  (wins of the mean-metric row: 2/40)
| **R3sbi mean (n=3)** | 7.33 ± 2.52 | - | 93.41 | 60.88 | 76.90 | 74.03 | 0.33 |  (wins of the mean-metric row: 3/40)
- SBI effect (R0sbi - R0swa), seed 37: wins -4; AUC -0.86, +0.25, -6.11, -2.84
- SBI effect (R0sbi - R0swa), seed 47: wins -6; AUC -2.70, -0.61, +0.43, +4.88
- SBI effect (R0sbi - R0swa), seed 57: wins -6; AUC -2.44, +2.21, -1.05, +5.87
- fairness term (R3sbi - R0sbi), seed 37: wins +1; AUC -1.51, +0.74, +6.39, +4.55
- fairness term (R3sbi - R0sbi), seed 47: wins +6; AUC +0.42, +4.66, +5.78, +2.61
- fairness term (R3sbi - R0sbi), seed 57: wins +1; AUC +2.79, -1.50, +8.68, -1.90
