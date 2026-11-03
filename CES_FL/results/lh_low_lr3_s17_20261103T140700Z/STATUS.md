# L_H lower LR, seed17, three epochs

2026-11-03T15:11:44.054726+00:00

State: COMPLETED_VERIFIED; 3/3epochs recorded. Scheduler observations are in STATUS.json.

Only head_lr1e-4,encoder_lr1e-6,epochs3 differ from original L_H. Same P50/fresh seed17 head, losses, preprocessing, frozen BN, sampling and validation AUC selection. No TEST access.

[Frozen proposal](../../proposals/lh_low_lr3_s17_20261103T140700Z/README.md) · [Fixed TRAIN drift diagnosis](../../proposals/lh_low_lr3_s17_20261103T140700Z/TRAIN_DIAGNOSIS.md)

|Epoch|AUC %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|97.874579|5.466329|0.042719|0.101704|0.413609|0.118314|0.034494|
|2|98.096682|4.775993|0.040759|0.091281|0.407493|0.140089|0.037547|
|3|98.185660|4.611628|0.043699|0.094555|0.418939|0.192721|0.036871|

All axis metrics, group support and exposure remain in the complete saved histories. Interim epochs are not the final selected model. No automatic extension or additional candidate.

Original ledger: 15.936613/24GPUh used. This allocation cap1.5GPUh; CPU checks/reports do not hold a GPU.

[Final VAL comparison](REPORT.md) · [Every fairness metric](ALL_FAIRNESS_METRICS.csv) · [Group counts/FPR/FNR/AUC](GROUP_COUNTS_AND_RATES.csv)
