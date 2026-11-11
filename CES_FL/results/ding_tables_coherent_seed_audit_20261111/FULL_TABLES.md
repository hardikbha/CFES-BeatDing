# Coherent full-L_H rows for Ding Tables1-3

All ours rows use seed17. Each backbone uses one checkpoint across every dataset. All saved fixed-probability0.5 reports are used consistently; no threshold search or new inference. Rates/AUC are multiplied by100; F_FPR is summed disparity, not overall FPR. Numerical comparisons are descriptive.

## Table 1

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ | Ding published reference | 0.5300 | 3.6100 | 96.4500 | 9.2900 | 4.3500 | 94.8600 | 20.1800 | 9.4700 | 86.9100 | 97.7100 |
| FF++ | Full L_H | 0.2394 | 2.4134 | 97.4401 | 8.8427 | 3.2709 | 97.5657 | 20.2166 | 15.1586 | 92.0656 | 98.4889 |

## Table 2

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFDC | Ding published reference | 1.7600 | 3.6700 | 57.9400 | 17.9300 | 7.5800 | 52.3300 | 37.3700 | 11.9200 | 39.0300 | 61.8600 |
| DFDC | Full L_H | 0.3069 | 1.7536 | 59.7320 | 6.4064 | 4.5141 | 52.3763 | 15.0223 | 6.6291 | 40.0141 | 60.4556 |
| Celeb-DF | Ding published reference | 6.4100 | 10.8100 | 69.6800 | 16.6000 | 7.2900 | 69.9600 | 23.3200 | 12.0100 | 63.2700 | 76.7500 |
| Celeb-DF | Full L_H | 4.6904 | 28.8503 | 61.5696 | 41.3266 | 32.2621 | 51.0108 | 47.7116 | 37.5399 | 48.4917 | 71.4039 |
| DFD | Ding published reference | 4.7200 | 5.4900 | 78.9800 | 2.0900 | 18.6600 | 77.4400 | 27.3200 | 22.8100 | 69.7300 | 81.4600 |
| DFD | Full L_H | 0.8452 | 1.8578 | 80.1861 | 13.2652 | 2.5381 | 79.4719 | 40.2055 | 6.4617 | 67.7926 | 80.8788 |

## Table 3

| Dataset | Method | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Joint F_FPR ↓ | Joint F_DP ↓ | Joint es-AUC ↑ | AUC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ | Ding published reference | 3.9400 | 5.1400 | 94.4600 | 18.6700 | 2.4300 | 92.6900 | 34.3000 | 8.6100 | 82.8700 | 95.5300 |
| FF++ | Full L_H | 0.0179 | 4.2599 | 93.7101 | 15.9039 | 2.6700 | 92.7671 | 30.4337 | 8.9332 | 81.3775 | 96.0537 |
| DFDC | Ding published reference | 3.9700 | 3.6100 | 56.9600 | 11.9400 | 5.1500 | 50.9900 | 32.1400 | 10.2700 | 37.5500 | 62.5400 |
| DFDC | Full L_H | 10.0855 | 9.2670 | 57.9516 | 16.3564 | 1.5962 | 51.1826 | 48.6436 | 14.6642 | 33.6852 | 60.0680 |
| Celeb-DF | Ding published reference | 4.5300 | 12.0000 | 68.9800 | 6.7600 | 8.1600 | 71.0300 | 5.5600 | 9.1100 | 68.2500 | 77.3500 |
| Celeb-DF | Full L_H | 14.2821 | 27.9151 | 62.5406 | 44.7970 | 31.4302 | 61.9582 | 49.4121 | 34.4858 | 57.8543 | 73.8106 |
| DFD | Ding published reference | 3.1200 | 1.7600 | 77.6800 | 6.5100 | 3.0300 | 70.9300 | 20.9900 | 4.8500 | 66.1900 | 78.6300 |
| DFD | Full L_H | 2.2945 | 1.7669 | 71.7039 | 6.6281 | 0.9466 | 70.6093 | 33.5722 | 6.3102 | 54.6468 | 73.4718 |
