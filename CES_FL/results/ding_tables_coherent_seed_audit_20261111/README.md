# Ding Tables 1–3: coherent model/seed audit

Observed management UTC: 2026-11-11T05:53:59.660916+00:00. This audits existing saved TEST aggregates. No model execution, new TEST/CAL evaluation, threshold fitting or training was performed.

**For full L_H, seed 17 is the verified available seed for both Xception and ResNet-50. It is not established as the best seed across repeated full-L_H runs, because the checked full-L_H records contain only this seed.** The higher archived Xception FF++ score 98.5412% belongs to L_A seed 7, the BCE control, not full L_H.

By Ding's benchmarks we mean FF++, DFDC, Celeb-DF and DFD used in its three tables. Keep FF2 seed 17 separate from these experiments; the checkpoints/training populations differ.

| Table | Backbone / TEST set | Full L_H seed | Our AUC % | Ding published AUC % | Our joint 100F_FPR | Ding joint 100F_FPR |
|---|---|---:|---:|---:|---:|---:|
| 1 | Xception / FF++ | 17 | 98.4889 | 97.71 | 20.2166 | 20.18 |
| 2 | Xception / DFDC | 17 | 60.4556 | 61.86 | 15.0223 | 37.37 |
| 2 | Xception / Celeb-DF | 17 | 71.4039 | 76.75 | 47.7116 | 23.32 |
| 2 | Xception / DFD | 17 | 80.8788 | 81.46 | 40.2055 | 27.32 |
| 3 | ResNet-50 / FF++ | 17 | 96.0537 | 95.53 | 30.4337 | 34.30 |
| 3 | ResNet-50 / DFDC | 17 | 60.0680 | 62.54 | 48.6436 | 32.14 |
| 3 | ResNet-50 / Celeb-DF | 17 | 73.8106 | 77.35 | 49.4121 | 5.56 |
| 3 | ResNet-50 / DFD | 17 | 73.4718 | 78.63 | 33.5722 | 20.99 |

All fairness entries above use the existing fixed probability 0.5 analysis, consistently across methods/tables. The older Overleaf coherent-model tables use our SELECT-real 10%-FPR-calibrated thresholds, which produce different F_FPR/F_DP values while leaving AUC/es-AUC unchanged. Do not mix those operating points or choose them per metric. The original retrospective six-table packet also used 0.5 and is kept separate from this coherent-row audit.

The full Xception L_H seed 17 row has higher FF++ AUC than the published reference (98.4889 versus 97.71; +0.7789pp) and higher joint es-AUC (92.0656 versus 86.91; +5.1556pp). At fixed 0.5 it is numerically better on 8/10 FF++ columns, but joint F_FPR (20.2166 versus 20.18) and joint F_DP (15.1586 versus 9.47) are worse. Its DFDC disparity improves substantially, with lower AUC; Celeb-DF and DFD still have material weaknesses. These are specific observed comparisons, not uniform superiority or statistical significance.

The full ResNet L_H seed 17 run is actually completed: native arm D, ten epochs, selected epoch 1, checkpoint `b08c2b5180d999f81e044872a8228f7e876d0250a6cb29034d59b11b1a897f10`. On FF++ it has higher AUC 96.0537 versus 95.53 and lower gender/race/joint F_FPR 0.0179/15.9039/30.4337 versus 3.94/18.67/34.30. It does not lead every metric or external dataset. [Native completion evidence](RESNET_COMPLETION_RECEIPTS.json).

**Manuscript inconsistency found:** the current Overleaf text says no ResNet L_H was trained, but the completed receipts and saved TEST reports contradict that statement. Its current Table 3 instead displays a separate SGD/SWA/SBI experiment family as means over seeds 37/47/57. Those 96.01% FF++ and 80.07% Celeb-DF numbers belong to R3-SBI-SWA means, not a single seed or full L_H. They must not be relabeled as full L_H or used to infer its best seed. Overleaf has not been edited by this audit.

## Highest archived AUC values are separate model rows

The following are retrospective maxima by AUC within the 68 dataset-report candidate pool of the 9 November packet. That pool includes earlier encoder/upgrade controls and L_A/L_H; it excludes the separate native-recipe SWA/SBI family. These are not an exhaustive claim over every project run and not one model achieving every row.

| Backbone | Dataset | Archived AUC-leading model | Seed | AUC % |
|---|---|---|---:|---:|
| Xception | FF++ | X-LA-7 (L_A) | 7 | 98.5412 |
| Xception | DFDC | X-LA-17 (L_A) | 17 | 61.7690 |
| Xception | Celeb-DF | X-R2-17 (R2_partial_dro) | 17 | 74.6706 |
| Xception | DFD | X-LA-7 (L_A) | 7 | 84.2339 |
| ResNet-50 | FF++ | R-LA-17 (L_A) | 17 | 96.2103 |
| ResNet-50 | DFDC | R-LH-17 (L_H) | 17 | 60.0680 |
| ResNet-50 | Celeb-DF | R-LH-17 (L_H) | 17 | 73.8106 |
| ResNet-50 | DFD | R-C-17 (C_resnet50) | 17 | 74.0159 |

In particular, Table 1's highest Xception AUC is L_A seed 7 (98.5412%). For Table 2 there is no one highest-AUC seed/model across all datasets: L_A seed 17 leads DFDC within the audited pool, R2 partial-DRO seed 17leads Celeb-DF, and L_A seed 7 leads DFD. Switching between these models inside a single claimed detector row would misrepresent the evidence.

If a coherent main-method presentation is wanted, use full L_H seed 17 for Tables1/2 and its ResNet-50 counterpart seed 17 for Table 3, and include L_A controls explicitly. This is an available full-model presentation choice, not a claim that seed 17 is universally optimal or that full L_H beats all controls. [Complete ten-metric tables](FULL_TABLES.md), [numeric full-L_H rows](FULL_LH_TABLES_1_2_3.csv), [coherent alternative index](COHERENT_CANDIDATE_INDEX.json), [retrospective AUC index](RETROSPECTIVE_AUC_LEADERS.csv).

## Protocol limits and verification

Training budgets, optimizer, initialization, head/trainability and eligibility differ from the published runs. The ResNet cohort records 231 shared fit/test identity/source values; biological-person independence is unverified. A shared backbone or evaluator is not a fully matched native training reproduction. The seed 17 point results have no established multi-seed uncertainty or overall superiority guarantee.

Every selected native aggregate file was SHA256-checked against the frozen 9 November source index; fixed 0.5 values and checkpoint hashes match. Each backbone's full-L_H checkpoint is identical across its four dataset reports. Reference numbers come from the frozen official arXivv5Table 1-3 extraction, whose original packet verified 494 published values. No per-cell oracle values are presented as one model. [Native fixed-point reports](NATIVE_FIXED_POINT_REPORTS.json), [audit verification](VERIFICATION.json), [artifact hashes](SHA256.json).

[Official reference paper](https://arxiv.org/html/2511.10150v5). The original fixed snapshot remains preserved. This audit changes no checkpoints, old result files, paper tables or execution configurations; it adds only a dated review artifact.
