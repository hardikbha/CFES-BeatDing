# Table 1: FF++ — SOURCE_VAL10

Published rows are reference values under their own protocol. New rows use each selected checkpoint's global source FF++ validation10%FPR threshold on every target dataset. Target TEST scores/labels never recalibrate it. Historical fixed0.5 rows are omitted here. All displayed values are100×native fractions; no protocol-equivalent win is inferred.

| Attribute | Method / protocol | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---|---|---|---|
| Gender | Ding published reference [archived v5; own protocol] | Xception | 0.530 | 3.610 | 96.450 | 97.710 |
| Gender | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Gender | L_A, s17, selected ep3 | Xception | 0.124 | 1.654 | 97.473 | 98.471 |
| Gender | L_H, s17, selected ep1 | Xception | 0.848 | 1.674 | 97.440 | 98.489 |
| Race | Ding published reference [archived v5; own protocol] | Xception | 9.290 | 4.350 | 94.860 | 97.710 |
| Race | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Race | L_A, s17, selected ep3 | Xception | 10.750 | 2.734 | 97.061 | 98.471 |
| Race | L_H, s17, selected ep1 | Xception | 13.736 | 3.049 | 97.566 | 98.489 |
| Intersection | Ding published reference [archived v5; own protocol] | Xception | 20.180 | 9.470 | 86.910 | 97.710 |
| Intersection | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Intersection | L_A, s17, selected ep3 | Xception | 24.859 | 11.122 | 91.884 | 98.471 |
| Intersection | L_H, s17, selected ep1 | Xception | 31.402 | 11.888 | 92.066 | 98.489 |
