# Table 1: FF++ intra-domain — FIXED05

All displayed values are100 times native fractions. F_FPR is summed disparity, not a probability. Published baseline rows are reference results transcribed from Ding et al., arXiv2511.10150v5, Tables1/2. Our Xception rows are seed17, one original validation-selected checkpoint per arm, ten downstream epochs; L_A selected epoch3, L_H epoch1. Stage budgets, preprocessing and checkpoint/support policies differ from the paper. No new Ding computation, inference, calibration or model selection was performed to create these tables. Same layout does not establish protocol equivalence. No bold best-cell ranking or pooled win count is used. Paper: https://arxiv.org/html/2511.10150v5. [Uncertainty and support](../UNCERTAINTY.md).

Our threshold is fixed probability0.5 (logit0); Ding values retain the paper operating protocol.

| Attribute | Method | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---|---|---|---|
| Gender | Ori (published) | Xception | 4.100 | 5.720 | 91.930 | 92.690 |
| Gender | DAG-FDD WACV’24 [20] | Xception | 1.820 | 4.650 | 94.870 | 96.720 |
| Gender | DAW-FDD WACV’24 [20] | Xception | 0.780 | 9.520 | 95.760 | 97.460 |
| Gender | PG-FDD CVPR’24 [24] | Xception | 0.620 | 4.740 | 96.320 | 97.660 |
| Gender | Fairadapter ICASSP’25 [10] | ViT-L/14 | 4.160 | 12.210 | 67.850 | 71.500 |
| Gender | RSEF-FDD arxiv’25 [18] | Xception | 0.570 | 8.550 | 94.910 | 97.090 |
| Gender | Ding et al. (published) | Xception | 0.530 | 3.610 | 96.450 | 97.710 |
| Gender | CES-FL L_A [seed17] | Xception | 0.472 | 1.763 | 97.473 | 98.471 |
| Gender | CES-FL L_H [seed17] | Xception | 0.239 | 2.413 | 97.440 | 98.489 |
| Race | Ori (published) | Xception | 19.760 | 4.740 | 82.850 | 92.690 |
| Race | DAG-FDD WACV’24 [20] | Xception | 5.480 | 9.200 | 93.430 | 96.720 |
| Race | DAW-FDD WACV’24 [20] | Xception | 5.430 | 14.690 | 94.150 | 97.460 |
| Race | PG-FDD CVPR’24 [24] | Xception | 11.130 | 4.780 | 94.520 | 97.660 |
| Race | Fairadapter ICASSP’25 [10] | ViT-L/14 | 43.220 | 20.390 | 56.030 | 71.500 |
| Race | RSEF-FDD arxiv’25 [18] | Xception | 8.390 | 5.280 | 93.600 | 97.090 |
| Race | Ding et al. (published) | Xception | 9.290 | 4.350 | 94.860 | 97.710 |
| Race | CES-FL L_A [seed17] | Xception | 9.040 | 3.059 | 97.061 | 98.471 |
| Race | CES-FL L_H [seed17] | Xception | 8.843 | 3.271 | 97.566 | 98.489 |
| Intersection | Ori (published) | Xception | 36.030 | 14.640 | 74.430 | 92.690 |
| Intersection | DAG-FDD WACV’24 [20] | Xception | 24.080 | 26.250 | 85.800 | 96.720 |
| Intersection | DAW-FDD WACV’24 [20] | Xception | 14.360 | 26.130 | 86.740 | 97.460 |
| Intersection | PG-FDD CVPR’24 [24] | Xception | 9.190 | 13.390 | 86.830 | 97.660 |
| Intersection | Fairadapter ICASSP’25 [10] | ViT-L/14 | 86.910 | 42.440 | 45.980 | 71.500 |
| Intersection | RSEF-FDD arxiv’25 [18] | Xception | 23.640 | 21.740 | 85.770 | 97.090 |
| Intersection | Ding et al. (published) | Xception | 20.180 | 9.470 | 86.910 | 97.710 |
| Intersection | CES-FL L_A [seed17] | Xception | 22.793 | 12.166 | 91.884 | 98.471 |
| Intersection | CES-FL L_H [seed17] | Xception | 20.217 | 15.159 | 92.066 | 98.489 |
