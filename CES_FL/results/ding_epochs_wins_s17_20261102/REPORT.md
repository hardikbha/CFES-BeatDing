# CES-FL: training epochs and numerical comparisons with Ding

Prepared 2 November 2026. These are the existing seed17 TEST results in [Tables 1–3](../ding_format_tables_s17_20261102/README.md). No training, inference, calibration or test-based selection was performed for this report.

**Epochs:** the reported detection-pretrained models completed 10 original-only pretraining epochs and 10 downstream epochs per arm. Pretraining is shared by the arms, not rerun for each one. ImageNet initialization predates these stages. A uses no deepfake pretraining.

**We cannot identify the first epoch at which a model exceeded Ding.** Only validation-selected frozen checkpoints were tested; a selected epoch is not a measured crossover epoch. Do not choose a different checkpoint by comparing epochs on TEST.

## Verified pretraining

| Backbone | Completed epochs | Validation-selected epoch | Checkpoint SHA256 |
|---|---:|---:|---|
| Xception | 10 | 9 | `484f09d6f826aa4baf1e4d6413813aa8608e320349082be377617c9f943e9dc8` |
| ResNet-50 | 10 | 7 | `f67d7314f27f8df0940ea4c0b5217b96abcc0e10a76129f097b0674c675ea765` |

## Completed downstream arms and numerical wins

Counts compare unrounded stored measurements with Ding's displayed, rounded published numbers for the same backbone. They are descriptive point comparisons, not statistical wins or proof of protocol equivalence. Main rows use the previously stored validation-calibrated thresholds.

| Arm | Backbone | Completed downstream epochs | Validation-selected epoch | Wins / 40 | Ties / 40 | Fairness wins / 36 | AUC wins / 4 | F_FPR wins / 12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_imagenet_frozen | Xception | 10 | 8 | 10 | 0 | 10 | 0 | 5 |
| B_detection_frozen | Xception | 10 | 7 | 17 | 0 | 15 | 2 | 1 |
| C_detection_invariance | Xception | 10 | 3 | 17 | 0 | 16 | 1 | 1 |
| A_resnet50 | ResNet-50 | 10 | 9 | 7 | 0 | 7 | 0 | 3 |
| B_resnet50 | ResNet-50 | 10 | 3 | 5 | 0 | 4 | 1 | 1 |
| C_resnet50 | ResNet-50 | 10 | 3 | 5 | 0 | 4 | 1 | 1 |
| R0_frozen_bce | Xception | 10 | 7 | 17 | 0 | 15 | 2 | 1 |
| R1_partial_bce | Xception | 10 | 1 | 15 | 0 | 13 | 2 | 1 |
| R2_partial_dro | Xception | 10 | 5 | 14 | 0 | 12 | 2 | 0 |
| RN_partial_dro_nochange_pilot | Xception | 10 | 3 | 17 | 0 | 15 | 2 | 3 |
| RD_partial_dro_age_pilot | Xception | 10 | 5 | 15 | 0 | 13 | 2 | 0 |

## Wins by dataset

| Arm | Backbone | FF++ / 10 | DFDC / 10 | Celeb-DF / 10 | DFD / 10 |
|---|---|---:|---:|---:|---:|
| A_imagenet_frozen | Xception | 0 | 3 | 2 | 5 |
| B_detection_frozen | Xception | 6 | 4 | 0 | 7 |
| C_detection_invariance | Xception | 6 | 5 | 0 | 6 |
| A_resnet50 | ResNet-50 | 2 | 2 | 1 | 2 |
| B_resnet50 | ResNet-50 | 2 | 1 | 0 | 2 |
| C_resnet50 | ResNet-50 | 2 | 1 | 0 | 2 |
| R0_frozen_bce | Xception | 6 | 4 | 0 | 7 |
| R1_partial_bce | Xception | 7 | 1 | 0 | 7 |
| R2_partial_dro | Xception | 5 | 2 | 0 | 7 |
| RN_partial_dro_nochange_pilot | Xception | 7 | 2 | 0 | 8 |
| RD_partial_dro_age_pilot | Xception | 5 | 3 | 0 | 7 |

## Counting and interpretation

For Xception, Table1 supplies FF++ and Table2 supplies the other three datasets. For ResNet-50, Table3 supplies all four datasets. Each dataset contributes nine fairness cells plus one overall AUC: 40 unique comparisons. Table1 repeats overall AUC in the Gender, Race and Intersection blocks; counting all 12 printed cells would double-count AUC. The JSON records that raw Table1 count separately.

A strict win means lower F_FPR/F_DP or higher es-AUC/AUC. Absolute differences up to 1e-12 percentage points are numerical ties. Paper rounding and experimental uncertainty are not resolved by this tolerance. No checkpoint, threshold or preferred arm was changed after counting.

**Fairness remains mixed.** Xception B and C both have 17/40 total numerical wins. C has 16/36 fairness wins versus B's 15/36, but only one of four AUC wins versus B's two. Both exceed Ding on just 1/12 F_FPR cells. RD has 15/40 total wins and 0/12 F_FPR wins. These counts do not support an overall fairness-superiority claim.

B and upgrade R0 repeat the same measured scores and are not independent seeds. C is invariance-only; RN/RD are six-source diagnostics. The current source `58773bb` 50+50 study is validation-only and excluded. Counts here are not the separate historical BD0 preregistered endpoint.

Our operating points, training policies, support filtering and architecture differ from the published report; the same table layout alone does not establish a matched-protocol comparison. [Full protocol caveats](../ding_format_tables_s17_20261102/README.md).

[Exact per-metric comparisons](METRIC_COMPARISONS.csv) include both saved calibrated and fixed0.5 reports without selecting between them. [Epoch/config/history hashes and counts](comparison.json). Training histories, DONE and native verification receipts were read; no large checkpoint was reloaded or rehashed.

## Actual command and checks

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python CES_FL/results/ding_epochs_wins_s17_20261102/build_report.py --snapshot CES_FL/results/ding_format_tables_s17_20261102/source_snapshot.json --out CES_FL/results/ding_epochs_wins_s17_20261102
```

Actual successful output: `PASS: 13 training histories; 11 arms; 880 metric comparisons (two stored operating points); 40 unique cells per arm/report; no training/inference.` Histories contain epochs 1–10, all completion receipts pass, and selected epochs match earliest highest validation AUC. Existing output files are never overwritten.
