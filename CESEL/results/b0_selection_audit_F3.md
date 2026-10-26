# B0 selection audit (Codex F3) -- runs/ffpp_B0_s7

Per-epoch validation logits saved: NO (only best.pt is re-scorable); best epoch 2 (recorded selection value 85.624, probability-based AUC).

| epoch | val AUC (prob, recorded) | selection (recorded) | fake logit mean | logit SD |
|---|---|---|---|---|
| -1 | 56.759 | -1000000000.000 | None | None |
| 0 | 94.351 | 84.345 | 10.96 | 8.16 |
| 1 | 94.234 | 84.790 | 13.27 | 11.41 |
| 2 | 94.201 | 85.624 | 16.24 | 14.01 |
| 3 | 94.393 | 82.030 | 22.94 | 17.21 |
| 4 | 94.437 | 84.437 | 24.63 | 21.18 |
| 5 | 94.035 | 84.739 | 29.49 | 22.27 |
| 6 | 93.673 | 85.051 | 32.9 | 25.14 |
| 7 | 93.670 | 82.558 | 34.42 | 26.53 |
| 8 | 93.445 | 77.334 | 35.98 | 27.18 |
| 9 | 93.450 | 79.005 | 35.82 | 27.5 |

best.pt re-scored from eval/val_scores.npz: {"val_auc_logit": 94.20117723467601, "val_auc_prob_recomputed": 94.20117723467601, "inter_es_auc_logit": 77.64900612998177, "inter_es_auc_prob": 77.64900612998174, "select_value_fixed_rule": 85.62377137693542, "select_value_recorded": 85.62377137693542, "n_real_saturated": 0, "n_fake_saturated": 0, "n_real": 4092, "n_fake": 21294, "cross_class_sigmoid_tie_pairs": 0}
