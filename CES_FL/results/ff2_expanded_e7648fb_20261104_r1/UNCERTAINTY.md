# Existing FairForge-2 BCE/DAG/DAW validation uncertainty

This is a 1,000-draw, seed-17 connected-contributor bootstrap over the existing selected validation predictions. It conditions on one trained seed and the original validation-selected checkpoints. It does not measure training-seed variability or TEST performance. The resampling found45 recorded contributor components; those are not independently verified people.

| Method | Overall AUC (%) | AUC 95% interval (%) | Gender×race F_FPR (fraction sum) | 95% interval | Usable draws |
|---|---:|---:|---:|---:|---:|
| BCE | 99.8046 | [99.5169, 99.9107] | 0.4052 | [0.3536, 2.3191] | 490/1000 |
| DAG-FDD | 99.8136 | [99.5527, 99.9231] | 0.3789 | [0.2888, 1.9605] | 490/1000 |
| DAW-FDD | 99.8360 | [99.6231, 99.9218] | 0.3070 | [0.1974, 1.5808] | 490/1000 |

| Method | Fixed logit-zero gender×race F_FPR | 95% interval | Usable draws |
|---|---:|---:|---:|
| BCE | 0.4015 | [0.3410, 2.2351] | 490/1000 |
| DAG-FDD | 0.3221 | [0.2055, 1.5592] | 490/1000 |
| DAW-FDD | 0.3238 | [0.1837, 1.5031] | 490/1000 |

The retained rule requires at least900usable draws. Both threshold views' intersection F_FPR intervals therefore fail the support gate and are descriptive only. The other threshold view is the unchanged validation-real empirical10%FPR threshold. The AUC intervals have1000usable draws each. Several single-axis fairness metrics have972 usable draws; the full supported group rows are in `BASELINE_GROUP_METRICS.csv`.

These intervals do not establish fairness equivalence or superiority. Use group counts and label provenance with them; age and skin bins remain automatic proxy labels. The calculations use only saved validation scores, read no TEST images/labels, and do not change model, checkpoint or threshold selection.
