# Table 2: Cross-domain — SOURCE_VAL10

Published rows are reference values under their own protocol. New rows use each selected checkpoint's global source FF++ validation10%FPR threshold on every target dataset. Target TEST scores/labels never recalibrate it. Historical fixed0.5 rows are omitted here. All displayed values are100×native fractions; no protocol-equivalent win is inferred.

| Dataset | Method / protocol | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | Overall AUC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFDC | Ding published reference [archived v5; own protocol] | Xception | 1.760 | 3.670 | 57.940 | 17.930 | 7.580 | 52.330 | 37.370 | 11.920 | 39.030 | 61.860 |
| DFDC | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFDC | L_A, s17, selected ep3 | Xception | 0.404 | 1.838 | 60.692 | 14.683 | 5.666 | 53.488 | 30.442 | 7.212 | 40.283 | 61.769 |
| DFDC | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Celeb-DF | Ding published reference [archived v5; own protocol] | Xception | 6.410 | 10.810 | 69.680 | 16.600 | 7.290 | 69.960 | 23.320 | 12.010 | 63.270 | 76.750 |
| Celeb-DF | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| Celeb-DF | L_A, s17, selected ep3 | Xception | 6.077 | 21.197 | 61.235 | 34.847 | 26.575 | 50.957 | 37.192 | 28.241 | 48.722 | 70.747 |
| Celeb-DF | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFD | Ding published reference [archived v5; own protocol] | Xception | 4.720 | 5.490 | 78.980 | 2.090 | 18.660 | 77.440 | 27.320 | 22.810 | 69.730 | 81.460 |
| DFD | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFD | L_A, s17, selected ep3 | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| DFD | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
