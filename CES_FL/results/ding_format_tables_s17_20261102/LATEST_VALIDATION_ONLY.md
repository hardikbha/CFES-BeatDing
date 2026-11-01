# Latest 50+50 experiment: VALIDATION ONLY

Source `58773bbef9d9fe79dd313610f694ddeea12c6037`. Both arms completed 50 downstream epochs after shared 50-epoch original-only pretraining. These numbers are excluded from Ding-format TEST Tables 1–3. The manipulation-response arm remains blocked by qualified reviewed quartets.

| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | Overall AUC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ VAL | Latest bce [s17; validation only] | Xception | 6.23 | 0.79 | 96.35 | 14.99 | 3.76 | 92.86 | 65.78 | 15.25 | 88.91 | 98.45 |
| FF++ VAL | Latest fair [s17; validation only] | Xception | 4.76 | 1.29 | 96.47 | 10.34 | 2.97 | 93.60 | 68.42 | 14.88 | 89.55 | 98.49 |

Selection was validation AUC only; thresholds use validation real images. Both achieved validation FPR 409/4092 = 9.995112%. These are original-only baselines, not a completed counterfactual response comparison. Fair has slightly higher AUC but worse gender×race F_FPR and worst-group AUC; no primary fairness win is established.

[Complete completion/subgroup report](../manip_response_originals_s17_20261102/REPORT.md). No test evaluation was run for this reporting task.
