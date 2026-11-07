# R3-SBI-SWA (op05) vs Ding reproduced (P4, released code) and Ding published (T3)

| seed | vs reproduced: FF++ | DFDC | Celeb-DF | DFD | **total /40** | vs published total /40 |
|---|---|---|---|---|---|---|
| s37 | 3 | 10 | 7 | 7 | **27** | 15 |
| s47 | 8 | 8 | 4 | 6 | **26** | 17 |
| s57 | 6 | 10 | 6 | 10 | **32** | 21 |
| s67 | 5 | 10 | 4 | 4 | **23** | 14 |
| s77 | 5 | 10 | 5 | 6 | **26** | 18 |
| s87 | 5 | 10 | 7 | 8 | **30** | 15 |
| s97 | 6 | 10 | 5 | 6 | **27** | 14 |
| mean ± sd | | | | | **27.29 ± 2.93** | 16.29 ± 2.56 |

## Celeb-DF per column: Ding published | Ding reproduced (P4) | R3-SBI mean of 7 seeds
| column | published | reproduced | ours (mean) | ours beats reproduced (seeds of 7) |
|---|---|---|---|---|
| g F_FPR | 4.53 | 2.05 | 2.55 | 3/7 |
| g F_DP | 12.00 | 21.16 | 29.00 | 0/7 |
| g es-AUC | 68.98 | 67.22 | 70.89 | 7/7 |
| r F_FPR | 6.76 | 29.51 | 33.34 | 2/7 |
| r F_DP | 8.16 | 19.52 | 23.49 | 1/7 |
| r es-AUC | 71.03 | 58.52 | 62.52 | 7/7 |
| i F_FPR | 5.56 | 30.73 | 29.87 | 3/7 |
| i F_DP | 9.11 | 20.90 | 24.89 | 1/7 |
| i es-AUC | 68.25 | 56.85 | 64.59 | 7/7 |
| AUC | 77.35 | 77.22 | 79.66 | 7/7 |
