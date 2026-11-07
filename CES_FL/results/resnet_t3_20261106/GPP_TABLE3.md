# GPP (Amendment 12): per-group valFPR1 thresholds, demographic labels at inference (deviation from Ding)

## R3-SBI
| seed | FF++ | DFDC | Celeb-DF | DFD | total /40 | Celeb-DF wins |
|---|---|---|---|---|---|---|
| s37 | 4 | 4 | 2 | 1 | **11** | g es-AUC, AUC |
| s47 | 6 | 5 | 3 | 3 | **17** | g es-AUC, i es-AUC, AUC |
| s57 | 5 | 4 | 3 | 3 | **15** | g es-AUC, i es-AUC, AUC |
| s67 | 5 | 4 | 2 | 1 | **12** | g es-AUC, AUC |
| s77 | 5 | 3 | 2 | 1 | **11** | g es-AUC, AUC |
| s87 | 5 | 3 | 2 | 1 | **11** | g es-AUC, AUC |
| s97 | 5 | 3 | 2 | 1 | **11** | g es-AUC, AUC |
| mean ± sd | | | | | **12.57 ± 2.44** | |
| mean-metric row | 5 | 4 | 2 | 0 | 11 | |

Celeb-DF per column (mean over seeds vs Ding): g F_FPR 22.40/4.53; g F_DP 45.01/12.0; g es-AUC 70.89/68.98; r F_FPR 41.63/6.76; r F_DP 33.96/8.16; r es-AUC 62.52/71.03; i F_FPR 60.47/5.56; i F_DP 41.74/9.11; i es-AUC 64.59/68.25; AUC 79.66/77.35

## R0-SBI
| seed | FF++ | DFDC | Celeb-DF | DFD | total /40 | Celeb-DF wins |
|---|---|---|---|---|---|---|
| s37 | 5 | 4 | 2 | 0 | **11** | g es-AUC, AUC |
| s47 | 3 | 3 | 2 | 1 | **9** | g es-AUC, AUC |
| s57 | 3 | 4 | 2 | 1 | **10** | g es-AUC, AUC |
| mean ± sd | | | | | **10.00 ± 1.00** | |
| mean-metric row | 4 | 4 | 2 | 0 | 10 | |

Celeb-DF per column (mean over seeds vs Ding): g F_FPR 19.10/4.53; g F_DP 39.77/12.0; g es-AUC 71.26/68.98; r F_FPR 46.70/6.76; r F_DP 28.53/8.16; r es-AUC 61.16/71.03; i F_FPR 50.56/5.56; i F_DP 30.88/9.11; i es-AUC 63.99/68.25; AUC 78.79/77.35
