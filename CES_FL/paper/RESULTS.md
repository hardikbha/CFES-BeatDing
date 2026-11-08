# Frozen FairForge-2 TEST summary

All rates/AUC are multiplied by 100. F_FPR is a sum of absolute group deviations, not a single error rate.
± denotes sample SD over downstream seeds 7/17/27 sharing one seed17 pretrained encoder; it is not a CI.

| Method | Seeds | AUC | Achieved FPR | FNR | Gender×race 100F_FPR | Worst-group AUC |
|---|---:|---:|---:|---:|---:|---:|
| Ding corrected GDA | 1 | 93.08 | 11.38 | 7.80 | 9.47 | 89.03 |
| DAW-FDD | 1 | 98.87 | 14.41 | 2.29 | 45.35 | 97.22 |
| CES-FL P | 1 | 98.78 | 12.39 | 2.92 | 42.23 | 96.85 |
| L_A (plain-head BCE) | 1 | 99.00 | 15.14 | 1.44 | 63.67 | 97.33 |
| RF-S (ours) | 3 | 98.98 ± 0.01 | 14.92 ± 0.16 | 1.60 ± 0.13 | 60.95 ± 1.92 | 97.28 ± 0.02 |
| RF-SJg (ours) | 3 | 98.60 ± 0.06 | 13.69 ± 0.28 | 2.84 ± 0.26 | 47.69 ± 3.91 | 96.43 ± 0.16 |

## Supported axes: 100F_FPR (lower is better)

| Axis | Corrected GDA | DAW-FDD | CES-FL P | L_A | RF-S | RF-SJg |
|---|---:|---:|---:|---:|---:|---:|
| Gender | 0.36 | 2.72 | 2.50 | 4.02 | 4.16 ± 0.22 | 4.36 ± 0.13 |
| Race | 3.29 | 22.24 | 20.42 | 31.15 | 29.70 ± 1.05 | 22.91 ± 1.99 |
| Gender × race | 9.47 | 45.35 | 42.23 | 63.67 | 60.95 ± 1.92 | 47.69 ± 3.91 |
| Age (estimated) | 4.37 | 12.53 | 8.17 | 13.25 | 12.71 ± 0.79 | 13.28 ± 0.73 |
| Skin (estimated) | 3.44 | 16.86 | 16.55 | 21.80 | 20.72 ± 1.08 | 14.83 ± 0.33 |
| Age × skin | 31.97 | 91.00 | 80.49 | 102.75 | 98.58 ± 3.98 | 86.00 ± 2.18 |
| Sex × age × race | 12.86 | 62.90 | 33.12 | 57.26 | 54.78 ± 3.13 | 53.74 ± 2.79 |
