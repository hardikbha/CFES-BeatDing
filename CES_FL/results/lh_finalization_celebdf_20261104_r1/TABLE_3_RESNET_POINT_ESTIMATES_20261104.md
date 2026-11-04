# Table 3 — ResNet-50 current L_A/L_H results

Locked plan SHA256: `a234331c21c5539fb350f011737c993dd9bf84feb3d1ea65132a4fc36e98a788`. This is an aggregate-only rendering from eight one-pass TEST receipts; no threshold tuning or checkpoint selection used TEST.

Ding rows are published ResNet-50 references under the paper protocol. CES-FL rows are the new seed-17 ResNet-50 runs and use their frozen FF++ validation-real 10% FPR threshold on all four datasets. The protocols differ; these rows do not establish a protocol-matched win. Connected-contributor confidence intervals and paired L_H−L_A differences are available in [the full Table 3 uncertainty report](TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md).

## Source-validation-real 10% FPR threshold

| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFPP | Ding et al. (published reference) | ResNet-50 | 3.940 | 5.140 | 94.460 | 18.670 | 2.430 | 92.690 | 34.300 | 8.610 | 82.870 | 95.530 |
| FFPP | CES-FL ResNet L_A [seed17] | ResNet-50 | 0.216 | 5.024 | 93.724 | 16.709 | 3.335 | 92.312 | 33.338 | 9.751 | 80.394 | 96.210 |
| FFPP | CES-FL ResNet L_H [seed17] | ResNet-50 | 0.471 | 5.116 | 93.710 | 14.900 | 2.489 | 92.767 | 28.629 | 9.504 | 81.377 | 96.054 |
| DFDC | Ding et al. (published reference) | ResNet-50 | 3.970 | 3.610 | 56.960 | 11.940 | 5.150 | 50.990 | 32.140 | 10.270 | 37.550 | 62.540 |
| DFDC | CES-FL ResNet L_A [seed17] | ResNet-50 | 8.875 | 9.013 | 57.356 | 18.555 | 5.160 | 49.518 | 45.316 | 13.126 | 36.146 | 59.535 |
| DFDC | CES-FL ResNet L_H [seed17] | ResNet-50 | 9.845 | 9.365 | 57.952 | 15.998 | 2.452 | 51.183 | 46.175 | 13.685 | 33.685 | 60.068 |
| Celeb-DF | Ding et al. (published reference) | ResNet-50 | 4.530 | 12.000 | 68.980 | 6.760 | 8.160 | 71.030 | 5.560 | 9.110 | 68.250 | 77.350 |
| Celeb-DF | CES-FL ResNet L_A [seed17] | ResNet-50 | 12.503 | 30.266 | 62.410 | 43.414 | 31.226 | 60.013 | 54.770 | 39.998 | 55.898 | 73.497 |
| Celeb-DF | CES-FL ResNet L_H [seed17] | ResNet-50 | 14.521 | 30.512 | 62.541 | 45.783 | 32.054 | 61.958 | 52.985 | 36.928 | 57.854 | 73.811 |
| DFD | Ding et al. (published reference) | ResNet-50 | 3.120 | 1.760 | 77.680 | 6.510 | 3.030 | 70.930 | 20.990 | 4.850 | 66.190 | 78.630 |
| DFD | CES-FL ResNet L_A [seed17] | ResNet-50 | 2.555 | 0.939 | 72.317 | 9.534 | 3.188 | 71.277 | 34.454 | 8.316 | 56.511 | 73.655 |
| DFD | CES-FL ResNet L_H [seed17] | ResNet-50 | 2.165 | 1.598 | 71.704 | 6.513 | 0.148 | 70.609 | 40.819 | 5.283 | 54.647 | 73.472 |

## Fixed probability 0.5

| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | AUC ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FFPP | CES-FL ResNet L_A [seed17] | ResNet-50 | 0.443 | 4.382 | 93.724 | 14.864 | 3.101 | 92.312 | 31.369 | 9.664 | 80.394 | 96.210 |
| FFPP | CES-FL ResNet L_H [seed17] | ResNet-50 | 0.018 | 4.260 | 93.710 | 15.904 | 2.670 | 92.767 | 30.434 | 8.933 | 81.377 | 96.054 |
| DFDC | CES-FL ResNet L_A [seed17] | ResNet-50 | 8.630 | 8.707 | 57.356 | 18.793 | 4.753 | 49.518 | 46.563 | 12.361 | 36.146 | 59.535 |
| DFDC | CES-FL ResNet L_H [seed17] | ResNet-50 | 10.086 | 9.267 | 57.952 | 16.356 | 1.596 | 51.183 | 48.644 | 14.664 | 33.685 | 60.068 |
| Celeb-DF | CES-FL ResNet L_A [seed17] | ResNet-50 | 12.193 | 27.694 | 62.410 | 44.331 | 30.750 | 60.013 | 50.570 | 35.473 | 55.898 | 73.497 |
| Celeb-DF | CES-FL ResNet L_H [seed17] | ResNet-50 | 14.282 | 27.915 | 62.541 | 44.797 | 31.430 | 61.958 | 49.412 | 34.486 | 57.854 | 73.811 |
| DFD | CES-FL ResNet L_A [seed17] | ResNet-50 | 2.760 | 1.314 | 72.317 | 8.833 | 3.675 | 71.277 | 31.752 | 7.989 | 56.511 | 73.655 |
| DFD | CES-FL ResNet L_H [seed17] | ResNet-50 | 2.294 | 1.767 | 71.704 | 6.628 | 0.947 | 70.609 | 33.572 | 6.310 | 54.647 | 73.472 |

## Overall operating points and support

| Dataset | Arm | Threshold view | N | Real N | Fake N | AUC | FPR | FNR | Worst gender×race AUC | Eligible gender×race groups |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FFPP | L_A | Source-VAL10 | 25401 | 4538 | 20863 | 96.210 | 11.503 | 8.599 | 90.935 | 8 |
| FFPP | L_A | Fixed 0.5 | 25401 | 4538 | 20863 | 96.210 | 15.381 | 6.581 | 90.935 | 8 |
| FFPP | L_H | Source-VAL10 | 25401 | 4538 | 20863 | 96.054 | 12.539 | 8.321 | 91.664 | 8 |
| FFPP | L_H | Fixed 0.5 | 25401 | 4538 | 20863 | 96.054 | 16.152 | 6.514 | 91.664 | 8 |
| DFDC | L_A | Source-VAL10 | 22857 | 14699 | 8158 | 59.535 | 28.941 | 58.299 | 54.034 | 8 |
| DFDC | L_A | Fixed 0.5 | 22857 | 14699 | 8158 | 59.535 | 35.574 | 51.226 | 54.034 | 8 |
| DFDC | L_H | Source-VAL10 | 22857 | 14699 | 8158 | 60.068 | 30.744 | 54.572 | 50.432 | 8 |
| DFDC | L_H | Fixed 0.5 | 22857 | 14699 | 8158 | 60.068 | 36.792 | 48.529 | 50.432 | 8 |
| Celeb-DF | L_A | Source-VAL10 | 28458 | 5713 | 22745 | 73.497 | 55.312 | 16.988 | 52.444 | 3 |
| Celeb-DF | L_A | Fixed 0.5 | 28458 | 5713 | 22745 | 73.497 | 62.297 | 13.581 | 52.444 | 3 |
| Celeb-DF | L_H | Source-VAL10 | 28458 | 5713 | 22745 | 73.811 | 59.566 | 14.293 | 57.074 | 3 |
| Celeb-DF | L_H | Fixed 0.5 | 28458 | 5713 | 22745 | 73.811 | 65.832 | 11.422 | 57.074 | 3 |
| DFD | L_A | Source-VAL10 | 9385 | 3092 | 6293 | 73.655 | 45.957 | 22.009 | 67.434 | 4 |
| DFD | L_A | Fixed 0.5 | 9385 | 3092 | 6293 | 73.655 | 52.458 | 18.052 | 67.434 | 4 |
| DFD | L_H | Source-VAL10 | 9385 | 3092 | 6293 | 73.472 | 47.057 | 21.357 | 65.706 | 4 |
| DFD | L_H | Fixed 0.5 | 9385 | 3092 | 6293 | 73.472 | 52.555 | 17.623 | 65.706 | 4 |

## Provenance

- L_A checkpoint: `0571a220faa63ad61c3b2d5815d73f25a077ecf59a1a8e30c379a620e31c0555` (selected epoch 5).
- L_H checkpoint: `b08c2b5180d999f81e044872a8228f7e876d0250a6cb29034d59b11b1a897f10` (selected epoch 1).
- The TEST scores are held privately on the cluster; only aggregate metrics are rendered here.
- Confidence intervals are reported separately in [TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md](TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md), with actual valid-replicate counts and the prespecified 900/1000 interval floor.
- AUC is threshold-independent. FPR/FNR and fairness rows are measured separately at each frozen threshold.
- No direct Ding model was run for this Table 3 extension; Ding values are published references.
