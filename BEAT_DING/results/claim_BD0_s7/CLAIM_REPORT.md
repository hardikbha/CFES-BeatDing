# Claim report: detection, fairness at the matched operating point, metric fragility

Runs: runs/BD0_s7. Ding repro = official code, Xception, seed 7 (0.5 rule). Published = Ding CVPR'26 Tables 1-2.

| Set | AUC ours | AUC Ding repro | AUC Ding pub | dAUC 95% CI (vs repro) | wins /10 | iF_FPR ours [boot95] (null p) | iF_FPR Ding repro [boot95] (null p) | videos to halve ours | iF_FPR cleaner labels ours / Ding |
|---|---|---|---|---|---|---|---|---|---|
| FF++ | 96.78 | 97.44 | 97.71 | [-1.31, 0.04] | 0 | 46.56 [37.04, 75.51] (0.353) | 24.24 [22.15, 81.8] (0.953) | 8 | 43.07 / 30.15 |
| DFDC | 77.41 | 56.05 | 61.86 | [17.81, 25.03] | 7 | 32.23 [26.22, 75.19] (0.59) | 42.56 [35.23, 89.91] (0.343) | 9 | 63.58 / 54.1 |
| Celeb-DF | 87.32 | 71.22 | 76.75 | [13.65, 18.71] | 4 | 32.48 [13.05, 47.2] (0.163) | 47.59 [10.72, 57.48] (0.007) | 8 | 41.75 / 41.07 |
| DFD | 90.21 | 72.54 | 81.46 | [13.46, 22.11] | 7 | 29.05 [9.22, 79.53] (0.323) | 49.82 [19.4, 73.01] (0.033) | 2 | 109.68 / 41.22 |

Total wins vs Ding published: 18/40.

