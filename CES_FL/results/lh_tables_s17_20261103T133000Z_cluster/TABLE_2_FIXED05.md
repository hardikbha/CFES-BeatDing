# Table 2: Cross-domain — FIXED05

Published rows retain their archived paper values and own protocol. Historical rows are fixed0.5 and are included only in FIXED05. New rows preserve one selected model per arm; no per-cell selection. All values are100×native fractions; F_FPR is a sum, not a probability. New rows use fixed probability0.5.

| Dataset | Method / protocol | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | Overall AUC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFDC | Ding published reference [archived v5; own protocol] | Xception | 1.760 | 3.670 | 57.940 | 17.930 | 7.580 | 52.330 | 37.370 | 11.920 | 39.030 | 61.860 |
| DFDC | Ding classification historical, s17 | Xception | 3.196 | 4.280 | 55.041 | 28.904 | 12.173 | 44.317 | 57.590 | 15.802 | 33.696 | 55.599 |
| DFDC | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 3.063 | 3.835 | 55.773 | 20.755 | 10.804 | 47.435 | 42.557 | 13.860 | 34.862 | 56.047 |
| DFDC | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFDC | L_A, s17, selected ep3 | Xception | 0.004 | 1.416 | 60.692 | 13.351 | 5.391 | 53.488 | 27.840 | 6.773 | 40.283 | 61.769 |
| DFDC | L_H, s17, selected ep1 | Xception | 0.307 | 1.754 | 59.732 | 6.406 | 4.514 | 52.376 | 15.022 | 6.629 | 40.014 | 60.456 |
| Celeb-DF | Ding published reference [archived v5; own protocol] | Xception | 6.410 | 10.810 | 69.680 | 16.600 | 7.290 | 69.960 | 23.320 | 12.010 | 63.270 | 76.750 |
| Celeb-DF | Ding classification historical, s17 | Xception | 7.870 | 18.032 | 63.473 | 34.132 | 21.461 | 48.829 | 39.406 | 23.405 | 45.994 | 69.159 |
| Celeb-DF | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 6.961 | 19.686 | 59.782 | 45.467 | 29.599 | 58.071 | 47.594 | 32.099 | 53.384 | 71.220 |
| Celeb-DF | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Celeb-DF | L_A, s17, selected ep3 | Xception | 6.113 | 23.008 | 61.235 | 37.821 | 29.174 | 50.957 | 40.388 | 30.841 | 48.722 | 70.747 |
| Celeb-DF | L_H, s17, selected ep1 | Xception | 4.690 | 28.850 | 61.570 | 41.327 | 32.262 | 51.011 | 47.712 | 37.540 | 48.492 | 71.404 |
| DFD | Ding published reference [archived v5; own protocol] | Xception | 4.720 | 5.490 | 78.980 | 2.090 | 18.660 | 77.440 | 27.320 | 22.810 | 69.730 | 81.460 |
| DFD | Ding classification historical, s17 | Xception | 0.485 | 0.930 | 67.713 | 22.085 | 10.363 | 61.840 | 36.137 | 12.737 | 55.940 | 68.008 |
| DFD | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 3.273 | 2.989 | 72.497 | 12.023 | 0.560 | 66.022 | 49.820 | 14.332 | 59.525 | 72.537 |
| DFD | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFD | L_A, s17, selected ep3 | Xception | 2.700 | 2.388 | 80.596 | 10.876 | 1.736 | 79.797 | 34.406 | 5.508 | 68.570 | 81.156 |
| DFD | L_H, s17, selected ep1 | Xception | 0.845 | 1.858 | 80.186 | 13.265 | 2.538 | 79.472 | 40.206 | 6.462 | 67.793 | 80.879 |
