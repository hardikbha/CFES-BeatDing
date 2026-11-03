# Table 3: ResNet-50 — SOURCE_VAL10

Published rows are reference values under their own protocol. New rows use each selected checkpoint's global source FF++ validation10%FPR threshold on every target dataset. Target TEST scores/labels never recalibrate it. Historical fixed0.5 rows are omitted here. All displayed values are100×native fractions; no protocol-equivalent win is inferred.

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
