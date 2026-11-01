# Table 1. FF++ intra-domain fairness and detection

Our rows: seed 17 TEST, stored validation-real FPR10% thresholds; no test recalibration. Published rows retain Ding's reported numbers. Shared format does not establish identical training/evaluation protocols.

All values are percentages. ↓ lower; ↑ higher. Intersection = gender × race. Published baselines appear first in each block; our single-seed rows follow. [Arm definitions and limitations](README.md) · [Combined PDF](TABLES_1_2_3.pdf)

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---|---|---|---|
| Gender | Ori (published) | Xception | 4.10 | 5.72 | 91.93 | 92.69 |
| Gender | DAG-FDD WACV’24 [20] | Xception | 1.82 | 4.65 | 94.87 | 96.72 |
| Gender | DAW-FDD WACV’24 [20] | Xception | 0.78 | 9.52 | 95.76 | 97.46 |
| Gender | PG-FDD CVPR’24 [24] | Xception | 0.62 | 4.74 | 96.32 | 97.66 |
| Gender | Fairadapter ICASSP’25 [10] | ViT-L/14 | 4.16 | 12.21 | 67.85 | 71.50 |
| Gender | RSEF-FDD arxiv’25 [18] | Xception | 0.57 | 8.55 | 94.91 | 97.09 |
| Gender | Ding et al. (published) | Xception | 0.53 | 3.61 | 96.45 | 97.71 |
| Gender | CES-FL A: ImageNet frozen [s17] | Xception | 2.89 | 6.88 | 65.06 | 67.48 |
| Gender | CES-FL B: detection frozen + BCE [s17] | Xception | 2.02 | 1.74 | 96.94 | 98.35 |
| Gender | CES-FL C: detection frozen + invariance [s17] | Xception | 1.99 | 1.82 | 96.73 | 98.22 |
| Gender | CES-FL upgrade R0: frozen BCE [s17] | Xception | 2.02 | 1.74 | 96.94 | 98.35 |
| Gender | CES-FL upgrade R1: partial BCE [s17] | Xception | 0.17 | 2.78 | 96.99 | 98.40 |
| Gender | CES-FL upgrade R2: partial DRO [s17] | Xception | 0.88 | 3.85 | 97.02 | 98.40 |
| Gender | CES-FL upgrade RN: no-change diagnostic [s17] | Xception | 0.26 | 4.04 | 96.67 | 97.85 |
| Gender | CES-FL upgrade RD: age diagnostic [s17] | Xception | 0.75 | 3.82 | 96.73 | 98.20 |
| Race | Ori (published) | Xception | 19.76 | 4.74 | 82.85 | 92.69 |
| Race | DAG-FDD WACV’24 [20] | Xception | 5.48 | 9.20 | 93.43 | 96.72 |
| Race | DAW-FDD WACV’24 [20] | Xception | 5.43 | 14.69 | 94.15 | 97.46 |
| Race | PG-FDD CVPR’24 [24] | Xception | 11.13 | 4.78 | 94.52 | 97.66 |
| Race | Fairadapter ICASSP’25 [10] | ViT-L/14 | 43.22 | 20.39 | 56.03 | 71.50 |
| Race | RSEF-FDD arxiv’25 [18] | Xception | 8.39 | 5.28 | 93.60 | 97.09 |
| Race | Ding et al. (published) | Xception | 9.29 | 4.35 | 94.86 | 97.71 |
| Race | CES-FL A: ImageNet frozen [s17] | Xception | 31.07 | 14.16 | 60.79 | 67.48 |
| Race | CES-FL B: detection frozen + BCE [s17] | Xception | 11.41 | 2.37 | 96.04 | 98.35 |
| Race | CES-FL C: detection frozen + invariance [s17] | Xception | 11.36 | 2.46 | 95.72 | 98.22 |
| Race | CES-FL upgrade R0: frozen BCE [s17] | Xception | 11.41 | 2.37 | 96.04 | 98.35 |
| Race | CES-FL upgrade R1: partial BCE [s17] | Xception | 11.51 | 2.95 | 95.98 | 98.40 |
| Race | CES-FL upgrade R2: partial DRO [s17] | Xception | 15.63 | 3.88 | 96.07 | 98.40 |
| Race | CES-FL upgrade RN: no-change diagnostic [s17] | Xception | 8.55 | 3.61 | 95.06 | 97.85 |
| Race | CES-FL upgrade RD: age diagnostic [s17] | Xception | 11.83 | 3.33 | 95.77 | 98.20 |
| Intersection | Ori (published) | Xception | 36.03 | 14.64 | 74.43 | 92.69 |
| Intersection | DAG-FDD WACV’24 [20] | Xception | 24.08 | 26.25 | 85.80 | 96.72 |
| Intersection | DAW-FDD WACV’24 [20] | Xception | 14.36 | 26.13 | 86.74 | 97.46 |
| Intersection | PG-FDD CVPR’24 [24] | Xception | 9.19 | 13.39 | 86.83 | 97.66 |
| Intersection | Fairadapter ICASSP’25 [10] | ViT-L/14 | 86.91 | 42.44 | 45.98 | 71.50 |
| Intersection | RSEF-FDD arxiv’25 [18] | Xception | 23.64 | 21.74 | 85.77 | 97.09 |
| Intersection | Ding et al. (published) | Xception | 20.18 | 9.47 | 86.91 | 97.71 |
| Intersection | CES-FL A: ImageNet frozen [s17] | Xception | 61.82 | 25.20 | 53.13 | 67.48 |
| Intersection | CES-FL B: detection frozen + BCE [s17] | Xception | 37.89 | 11.84 | 90.90 | 98.35 |
| Intersection | CES-FL C: detection frozen + invariance [s17] | Xception | 37.48 | 12.10 | 90.30 | 98.22 |
| Intersection | CES-FL upgrade R0: frozen BCE [s17] | Xception | 37.89 | 11.84 | 90.90 | 98.35 |
| Intersection | CES-FL upgrade R1: partial BCE [s17] | Xception | 29.92 | 12.85 | 90.50 | 98.40 |
| Intersection | CES-FL upgrade R2: partial DRO [s17] | Xception | 30.23 | 12.26 | 90.57 | 98.40 |
| Intersection | CES-FL upgrade RN: no-change diagnostic [s17] | Xception | 22.93 | 13.03 | 90.55 | 97.85 |
| Intersection | CES-FL upgrade RD: age diagnostic [s17] | Xception | 26.27 | 12.94 | 90.09 | 98.20 |

Latest source `58773bb` 50+50 runs have VAL results only and are excluded. [Latest validation-only results](LATEST_VALIDATION_ONLY.md).
