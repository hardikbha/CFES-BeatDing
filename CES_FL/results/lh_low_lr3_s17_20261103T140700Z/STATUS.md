# L_H lower LR, seed17, three epochs

2026-11-03T14:44:36.813299+00:00

State: CHECKS_OR_QUALIFICATION_PENDING; 0/3epochs recorded. Scheduler observations are in STATUS.json.

Only head_lr1e-4,encoder_lr1e-6,epochs3 differ from original L_H. Same P50/fresh seed17 head, losses, preprocessing, frozen BN, sampling and validation AUC selection. No TEST access.

[Frozen proposal](../../proposals/lh_low_lr3_s17_20261103T140700Z/README.md) · [Fixed TRAIN drift diagnosis](../../proposals/lh_low_lr3_s17_20261103T140700Z/TRAIN_DIAGNOSIS.md)

|Epoch|AUC %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|

All axis metrics, group support and exposure remain in the complete saved histories. Interim epochs are not the final selected model. No automatic extension or additional candidate.

Original ledger: 15.526991/24GPUh used. This allocation cap1.5GPUh; CPU checks/reports do not hold a GPU.
