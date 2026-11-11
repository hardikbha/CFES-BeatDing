# Seed17 response-strength screen

Observed management UTC: 2026-11-11T05:14:59.265801+00:00

| Candidate | State | Stage | Node/GPU | Epochs | Selected epoch | VAL AUC % | VAL FNR % | Joint100F_FPR |
|---|---|---|---|---:|---:|---:|---:|---:|
| fair_only | COMPLETED_VERIFIED | DONE | gpu01/6 | 10/10 | 9 | 99.8229 | 0.2790 | 40.9199 |
| smooth_005 | COMPLETED_VERIFIED | DONE | gpu02/7 | 10/10 | 5 | 99.8226 | 0.2965 | 43.1332 |
| smooth_010 | COMPLETED_VERIFIED | DONE | gpu01/4 | 10/10 | 4 | 99.8221 | 0.2930 | 40.1410 |
| smooth_020 | COMPLETED_VERIFIED | DONE | gpu02/4 | 10/10 | 2 | 99.8182 | 0.2965 | 38.4295 |

SELECT/validation only. No new TEST or CAL inference. A pre-activation selected checkpoint does not demonstrate response-learning benefit. All rows remain exploratory after historical TEST exposure.
