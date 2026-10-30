# Controlled seed17 results

Validation AUC only; no real TEST scoring. C tests demographic invariance only.

| Arm | Selected epoch | Validation AUC (%) |
|---|---:|---:|
| P_pretrain | 9 | 98.1711 |
| A_imagenet_frozen | 8 | 67.9134 |
| B_detection_frozen | 7 | 98.1949 |
| C_detection_invariance | 3 | 98.0661 |

A→B detection pretraining effect: +30.2815 percentage points.

B→C invariance effect: -0.1288 percentage points.

Interpret with the full validation group metrics, sparse support (six originals), single-seed uncertainty, and recorded control drift. AUC alone is not a fairness conclusion.

Failures: {}
