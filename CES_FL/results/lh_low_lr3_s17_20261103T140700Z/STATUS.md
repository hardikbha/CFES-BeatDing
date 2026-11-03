# L_H lower LR, seed17, three epochs

2026-11-03T15:01:37.026458+00:00

State: RECORDED_PROGRESS; 2/3epochs recorded. Scheduler observations are in STATUS.json.

Only head_lr1e-4,encoder_lr1e-6,epochs3 differ from original L_H. Same P50/fresh seed17 head, losses, preprocessing, frozen BN, sampling and validation AUC selection. No TEST access.

[Frozen proposal](../../proposals/lh_low_lr3_s17_20261103T140700Z/README.md) · [Fixed TRAIN drift diagnosis](../../proposals/lh_low_lr3_s17_20261103T140700Z/TRAIN_DIAGNOSIS.md)

|Epoch|AUC %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|1|97.874579|5.466329|0.042719|0.101704|0.413609|0.118314|0.034494|
|2|98.096682|4.775993|0.040759|0.091281|0.407493|0.140089|0.037547|

All axis metrics, group support and exposure remain in the complete saved histories. Interim epochs are not the final selected model. No automatic extension or additional candidate.

Original ledger: 15.810575/24GPUh used. This allocation cap1.5GPUh; CPU checks/reports do not hold a GPU.
