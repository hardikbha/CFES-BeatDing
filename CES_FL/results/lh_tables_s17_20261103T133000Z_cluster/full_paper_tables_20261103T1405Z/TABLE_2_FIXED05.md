# Table 2: Cross-domain — FIXED05

All displayed values are100 times native fractions. F_FPR is summed disparity, not a probability. Published baseline rows are reference results transcribed from Ding et al., arXiv2511.10150v5, Tables1/2. Our Xception rows are seed17, one original validation-selected checkpoint per arm, ten downstream epochs; L_A selected epoch3, L_H epoch1. Stage budgets, preprocessing and checkpoint/support policies differ from the paper. No new Ding computation, inference, calibration or model selection was performed to create these tables. Same layout does not establish protocol equivalence. No bold best-cell ranking or pooled win count is used. Paper: https://arxiv.org/html/2511.10150v5. [Uncertainty and support](../UNCERTAINTY.md).

Our threshold is fixed probability0.5 (logit0); Ding values retain the paper operating protocol.

| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | AUC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFDC | Ori (published) | Xception | 8.670 | 6.700 | 54.560 | 19.410 | 7.990 | 47.020 | 53.420 | 17.340 | 36.960 | 56.130 |
| DFDC | DAG-FDD WACV’24 [20] | Xception | 5.290 | 6.680 | 57.410 | 9.260 | 12.510 | 45.910 | 45.300 | 12.920 | 36.420 | 59.130 |
| DFDC | DAW-FDD WACV’24 [20] | Xception | 3.600 | 3.670 | 55.880 | 22.360 | 9.720 | 47.770 | 46.800 | 12.690 | 36.630 | 56.900 |
| DFDC | PG-FDD CVPR’24 [24] | Xception | 2.350 | 2.570 | 57.710 | 10.100 | 11.490 | 50.530 | 32.160 | 17.710 | 37.420 | 59.640 |
| DFDC | Fairadapter ICASSP’25 [10] | ViT-L/14 | 2.390 | 4.570 | 55.880 | 32.490 | 26.770 | 52.270 | 74.100 | 31.530 | 31.390 | 59.250 |
| DFDC | RSEF-FDD arxiv’25 [18] | Xception | 2.090 | 3.830 | 57.900 | 12.240 | 7.740 | 50.270 | 38.890 | 12.180 | 37.720 | 59.080 |
| DFDC | Ding et al. (published) | Xception | 1.760 | 3.670 | 57.940 | 17.930 | 7.580 | 52.330 | 37.370 | 11.920 | 39.030 | 61.860 |
| DFDC | CES-FL L_A [seed17] | Xception | 0.004 | 1.416 | 60.692 | 13.351 | 5.391 | 53.488 | 27.840 | 6.773 | 40.283 | 61.769 |
| DFDC | CES-FL L_H [seed17] | Xception | 0.307 | 1.754 | 59.732 | 6.406 | 4.514 | 52.376 | 15.022 | 6.629 | 40.014 | 60.456 |
| Celeb-DF | Ori (published) | Xception | 6.930 | 20.230 | 60.940 | 23.440 | 17.890 | 61.210 | 27.380 | 18.720 | 61.220 | 69.180 |
| Celeb-DF | DAG-FDD WACV’24 [20] | Xception | 8.700 | 14.250 | 66.230 | 19.440 | 13.260 | 62.610 | 23.600 | 13.800 | 63.080 | 72.290 |
| Celeb-DF | DAW-FDD WACV’24 [20] | Xception | 8.710 | 12.730 | 68.310 | 14.760 | 7.800 | 66.470 | 24.750 | 13.780 | 59.240 | 74.090 |
| Celeb-DF | PG-FDD CVPR’24 [24] | Xception | 7.950 | 16.290 | 69.470 | 23.990 | 12.470 | 67.780 | 23.710 | 13.300 | 62.630 | 74.240 |
| Celeb-DF | Fairadapter ICASSP’25 [10] | ViT-L/14 | 8.590 | 26.840 | 56.840 | 46.140 | 36.350 | 58.340 | 47.030 | 35.690 | 54.770 | 64.940 |
| Celeb-DF | RSEF-FDD arxiv’25 [18] | Xception | 1.940 | 35.630 | 68.920 | 17.400 | 27.480 | 67.800 | 24.170 | 27.730 | 62.560 | 74.360 |
| Celeb-DF | Ding et al. (published) | Xception | 6.410 | 10.810 | 69.680 | 16.600 | 7.290 | 69.960 | 23.320 | 12.010 | 63.270 | 76.750 |
| Celeb-DF | CES-FL L_A [seed17] | Xception | 6.113 | 23.008 | 61.235 | 37.821 | 29.174 | 50.957 | 40.388 | 30.841 | 48.722 | 70.747 |
| Celeb-DF | CES-FL L_H [seed17] | Xception | 4.690 | 28.850 | 61.570 | 41.327 | 32.262 | 51.011 | 47.712 | 37.540 | 48.492 | 71.404 |
| DFD | Ori (published) | Xception | 9.450 | 6.630 | 72.680 | 7.750 | 22.310 | 69.070 | 35.060 | 25.530 | 59.200 | 74.320 |
| DFD | DAG-FDD WACV’24 [20] | Xception | 5.410 | 5.600 | 75.870 | 2.560 | 21.340 | 70.140 | 33.000 | 23.020 | 61.090 | 76.290 |
| DFD | DAW-FDD WACV’24 [20] | Xception | 5.300 | 10.630 | 71.680 | 4.320 | 20.160 | 63.240 | 33.810 | 24.720 | 56.010 | 73.710 |
| DFD | PG-FDD CVPR’24 [24] | Xception | 5.050 | 1.860 | 76.370 | 5.790 | 19.310 | 74.810 | 28.000 | 23.930 | 66.320 | 80.700 |
| DFD | Fairadapter ICASSP’25 [10] | ViT-L/14 | 6.320 | 7.550 | 63.280 | 11.290 | 12.280 | 56.280 | 29.560 | 33.110 | 48.630 | 68.120 |
| DFD | RSEF-FDD arxiv’25 [18] | Xception | 18.500 | 6.110 | 75.970 | 19.450 | 19.400 | 72.650 | 35.340 | 10.970 | 67.240 | 80.540 |
| DFD | Ding et al. (published) | Xception | 4.720 | 5.490 | 78.980 | 2.090 | 18.660 | 77.440 | 27.320 | 22.810 | 69.730 | 81.460 |
| DFD | CES-FL L_A [seed17] | Xception | 2.700 | 2.388 | 80.596 | 10.876 | 1.736 | 79.797 | 34.406 | 5.508 | 68.570 | 81.156 |
| DFD | CES-FL L_H [seed17] | Xception | 0.845 | 1.858 | 80.186 | 13.265 | 2.538 | 79.472 | 40.206 | 6.462 | 67.793 | 80.879 |
