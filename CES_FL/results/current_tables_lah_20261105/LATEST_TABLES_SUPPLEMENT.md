# Latest L_A/L_H rows for Ding-style Tables 1–3

Prepared 5 November 2026 from the completed seed-17 locked evaluations under `/home/sachin.chaudhary/ces_fl_runs/lh_tables_s17_20261103T133000Z/`. No inference or threshold tuning was performed for this summary. These L_A/L_H rows were not yet merged into the older full table PDF.

## Table 1: FF++ and Table 2: cross-domain results

All values are percentages. F_FPR, F_DP and es-AUC are shown in Gender / Race / Gender×Race order; AUC is the overall test AUC. L_A and L_H use the frozen FF++ validation-real 10% FPR operating point; the test FPR is therefore measured and may differ from 10%. One seed is reported.

| Dataset | Ding published AUC | L_A AUC | L_H AUC | Ding G×R F_FPR | L_A G×R F_FPR | L_H G×R F_FPR |
|---|---:|---:|---:|---:|---:|---:|
| FF++ | 97.710 | 98.471 | 98.489 | 20.180 | 24.859 | 31.402 |
| DFDC | 61.860 | 61.769 | 60.456 | 37.370 | 30.442 | 27.623 |
| Celeb-DF | 76.750 | 70.747 | 71.404 | 23.320 | 37.192 | 38.765 |
| DFD | 81.460 | 81.156 | 80.879 | 27.320 | 34.135 | 33.561 |

### Full L_A/L_H metrics

| Dataset | Method | Gender F_FPR / F_DP / es-AUC | Race F_FPR / F_DP / es-AUC | Gender×Race F_FPR / F_DP / es-AUC | Overall AUC | Test FPR / FNR |
|---|---|---|---|---|---:|---|
| FF++ | L_A | 0.124 / 1.654 / 97.473 | 10.750 / 2.734 / 97.061 | 24.859 / 11.122 / 91.884 | 98.471 | 11.305 / 3.269 |
| FF++ | L_H | 0.848 / 1.674 / 97.440 | 13.736 / 3.049 / 97.566 | 31.402 / 11.888 / 92.066 | 98.489 | 10.688 / 3.667 |
| DFDC | L_A | 0.404 / 1.838 / 60.692 | 14.683 / 5.666 / 53.488 | 30.442 / 7.212 / 40.283 | 61.769 | 34.655 / 50.858 |
| DFDC | L_H | 0.868 / 2.205 / 59.732 | 13.538 / 5.122 / 52.376 | 27.623 / 7.573 / 40.014 | 60.456 | 37.642 / 48.590 |
| Celeb-DF | L_A | 6.077 / 21.197 / 61.235 | 34.847 / 26.575 / 50.957 | 37.192 / 28.241 / 48.722 | 70.747 | 65.255 / 13.528 |
| Celeb-DF | L_H | 8.037 / 23.258 / 61.570 | 35.272 / 27.268 / 51.011 | 38.765 / 29.490 / 48.492 | 71.404 | 64.922 / 13.111 |
| DFD | L_A | 2.783 / 2.167 / 80.596 | 11.328 / 2.236 / 79.797 | 34.135 / 5.036 / 68.570 | 81.156 | 50.065 / 12.887 |
| DFD | L_H | 3.662 / 2.482 / 80.186 | 11.756 / 3.392 / 79.472 | 33.561 / 7.634 / 67.793 | 80.879 | 49.224 / 13.523 |

These results are mixed. L_H has slightly higher FF++ and Celeb-DF AUC than L_A, but lower DFDC and DFD AUC. Its G×R F_FPR is worse than L_A on FF++ and Celeb-DF, and lower on DFDC and DFD. They do not establish a general fairness or detection win over Ding.

## Table 3: ResNet-50 status

The currently available combined [Tables 1–3 PDF](../ding_format_tables_s17_20261102/TABLES_1_2_3.pdf) contains the older measured ResNet-50 A/B/C rows. It does **not** contain a current ResNet-50 L_H result. The ResNet-50 P50 pretraining checkpoint exists, but its matched downstream L_A/L_H Table 3 experiment was not completed in this result set. The 5 November B0/C0 runs are three-epoch validation diagnostics and cannot fill a TEST table cell.

## Architecture and provenance

The L_A/L_H rows above use the Xception plain-head ablation: one Xception encoder followed by global average pooling and one `Linear(2048, 1)` binary logit head. Total parameters: **20,809,001**; trainable under the `exit` policy: **6,790,433**. Encoder BN running statistics remain frozen during downstream training. L_A is the BCE-only arm; L_H is the hierarchical-real fairness arm with the configured procedural sensitivity/control and response-alignment terms. Both completed ten downstream epochs; validation-selected epochs were L_A epoch 3 and L_H epoch 1.

This plain-head L_H is distinct from the full CES-FL DiagramCES architecture, which uses one encoder, global and patch heads, image and patch gates, learned fusion weights and a bias. The ResNet-50 P50 DiagramCES checkpoint has **24,295,239** parameters. The November 5 B0/C0 attribution runs used a ResNet-50 trunk plus one linear head, with layer4 and that head trainable; they are not current L_H Table 3 results.

The saved L_A/L_H evaluation source commit is `ee73f2fdf5218e440852917e90281818bbe711bb`. Checkpoints: L_A `aee9670ca011937d457846fbf13717445c5607715064dd6a54da5a82bda580ae` (selected epoch 3); L_H `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0` (selected epoch 1). Tests used the pinned Ding splits and stored source-domain validation thresholds; they do not establish identical training protocols to Ding’s published runs.
