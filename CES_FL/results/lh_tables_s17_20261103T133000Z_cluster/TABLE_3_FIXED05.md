# Table 3: ResNet-50 — FIXED05

Published rows retain their archived paper values and own protocol. Historical rows are fixed0.5 and are included only in FIXED05. New rows preserve one selected model per arm; no per-cell selection. All values are100×native fractions; F_FPR is a sum, not a probability. New rows use fixed probability0.5.

| Dataset | Method / protocol | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | Overall AUC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ | Ding published reference [archived v5; own protocol] | ResNet-50 | 3.940 | 5.140 | 94.460 | 18.670 | 2.430 | 92.690 | 34.300 | 8.610 | 82.870 | 95.530 |
| FF++ | Current L_H ResNet-50 [implementation/P50/A/H training pending] | ResNet-50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFDC | Ding published reference [archived v5; own protocol] | ResNet-50 | 3.970 | 3.610 | 56.960 | 11.940 | 5.150 | 50.990 | 32.140 | 10.270 | 37.550 | 62.540 |
| DFDC | Current L_H ResNet-50 [implementation/P50/A/H training pending] | ResNet-50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Celeb-DF | Ding published reference [archived v5; own protocol] | ResNet-50 | 4.530 | 12.000 | 68.980 | 6.760 | 8.160 | 71.030 | 5.560 | 9.110 | 68.250 | 77.350 |
| Celeb-DF | Current L_H ResNet-50 [implementation/P50/A/H training pending] | ResNet-50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFD | Ding published reference [archived v5; own protocol] | ResNet-50 | 3.120 | 1.760 | 77.680 | 6.510 | 3.030 | 70.930 | 20.990 | 4.850 | 66.190 | 78.630 |
| DFD | Current L_H ResNet-50 [implementation/P50/A/H training pending] | ResNet-50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

No new ResNet L_H computation has completed. Historical ResNet A/B/C cannot fill this row.
