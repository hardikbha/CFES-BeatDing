# Table 1: FF++ — FIXED05

Published rows retain their archived paper values and own protocol. Historical rows are fixed0.5 and are included only in FIXED05. New rows preserve one selected model per arm; no per-cell selection. All values are100×native fractions; F_FPR is a sum, not a probability. New rows use fixed probability0.5.

| Attribute | Method / protocol | Backbone | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | AUC ↑ |
|---|---|---|---|---|---|---|
| Gender | Ding published reference [archived v5; own protocol] | Xception | 0.530 | 3.610 | 96.450 | 97.710 |
| Gender | Ding classification historical, s17 | Xception | 0.188 | 4.480 | 94.670 | 96.321 |
| Gender | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 1.479 | 1.702 | 96.229 | 97.444 |
| Gender | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Gender | L_A, s17, selected ep3 | Xception | 0.472 | 1.763 | 97.473 | 98.471 |
| Gender | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING |
| Race | Ding published reference [archived v5; own protocol] | Xception | 9.290 | 4.350 | 94.860 | 97.710 |
| Race | Ding classification historical, s17 | Xception | 10.192 | 4.388 | 94.616 | 96.321 |
| Race | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 6.289 | 6.737 | 94.085 | 97.444 |
| Race | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Race | L_A, s17, selected ep3 | Xception | 9.040 | 3.059 | 97.061 | 98.471 |
| Race | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING |
| Intersection | Ding published reference [archived v5; own protocol] | Xception | 20.180 | 9.470 | 86.910 | 97.710 |
| Intersection | Ding classification historical, s17 | Xception | 26.648 | 12.814 | 83.173 | 96.321 |
| Intersection | Ding SFD+GDA historical, s7 [detached fairness] | Xception | 24.238 | 12.904 | 88.155 | 97.444 |
| Intersection | Ding connected-gradient 10epoch adaptation [not implemented] | Xception | PENDING | PENDING | PENDING | PENDING |
| Intersection | L_A, s17, selected ep3 | Xception | 22.793 | 12.166 | 91.884 | 98.471 |
| Intersection | L_H, s17, selected ep1 | Xception | PENDING | PENDING | PENDING | PENDING |
