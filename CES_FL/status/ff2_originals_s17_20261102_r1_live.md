# FairForge-2 original-only CES-FL pretraining — seed17

Observed management UTC: 2026-11-02T06:53:56.236385+00:00.

Status: **COMPLETED**. Completed epochs: **10/10**.

Direct SSH host **gpu01**, physical GPU **4**, H10080GB; trainer PID **2525928**, worker PID **2525919**. GPU UUID `GPU-9b669a06-9156-a98e-2194-36f2fc11d7ca`. No PBS allocation is claimed; cooperative locks and20% VRAM cap. Existing jobs preserved.

Actual child process observation: ``.

TRAIN218,266; VAL31,107. TEST87,564 is not evaluated. Original-image BCE; all21,594,159parameters trainable, BNupdates. Validation AUC selects checkpoints; calibration uses validation reals only.

Input hash/reader/path/provenance checks PASS;64TRAIN+64VALimages decoded by stratum. Native CUDA suite and two real TRAIN-only optimizer steps PASS with losses0.71599847,0.70625442. No synthetic edit is used.

| Epoch | Validation AUC (%) | Classification loss |
|---|---:|---:|
| 1 | 99.562130 | 0.15681399 |
| 2 | 99.652957 | 0.06285439 |
| 3 | 99.686517 | 0.04223681 |
| 4 | 99.710703 | 0.03132660 |
| 5 | 99.735900 | 0.02618677 |
| 6 | 99.728192 | 0.02164583 |
| 7 | 99.797400 | 0.01867947 |
| 8 | 99.771616 | 0.01624456 |
| 9 | 99.810536 | 0.01495648 |
| 10 | 99.794726 | 0.01348887 |

Current validation-selected epoch: 9. Achieved VALFPR: 9.967186% (243/2438); no TESTFPR guarantee.

| Attribute | F_FPR (%) | F_DP (%) | es-AUC (%) | Worst eligible AUC (%) |
|---|---:|---:|---:|---:|
| gender | 3.598003 | 0.243770 | 99.802378 | 99.808019 |
| race | 12.156468 | 1.057359 | 99.437360 | 99.683965 |
| gender_race | 28.385680 | 1.530781 | 99.075194 | 99.582704 |
| age_bin | 9.232910 | 4.331311 | 99.631144 | 99.698245 |
| skin_bin | 11.682691 | 0.404731 | 99.621293 | 99.697721 |

Selected checkpoint SHA256: `43152de1b98dce48801114efb2454cb0bc6aa7400445dd5f139759022229895a`. Native independent selected-checkpoint VAL replay and strict reload PASS.


Model source commit `875bce025084871438ef799ca9086985c4375672`; exact fixed FF2P10config SHA256 `db0c886527635295f0394dddbca9da1a8a3c40807609ab78f324b75b382fbee8`. No architecture/loss edits or FF++configuration substitution.

Run root: `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1`; actual training log: `logs/train.log`; history/checkpoints: `training/`; checks: `checks/`.

[Exact request, configuration, fresh checks and scope](../proposals/ff2_originals_s17_20261102_r1/README.md). Full matched Ding comparison, paired arms and newTEST evaluation remain blocked/unlaunched. Current run is original-only P, not a completed counterfactual comparison.

This finite read-only observer publishes state/epoch milestones and stops after completion, failure, process ambiguity or12h. It cannot launch training, generate images, change selection/calibration or kill jobs. Timestamps remain observations, not a monitoring guarantee.
