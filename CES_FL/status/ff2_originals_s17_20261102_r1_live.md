# FairForge-2 original-only CES-FL pretraining — seed17

Observed management UTC: 2026-11-02T06:14:00.332107+00:00.

Status: **RUNNING**. Completed epochs: **5/10**.

Direct SSH host **gpu01**, physical GPU **4**, H10080GB; trainer PID **2525928**, worker PID **2525919**. GPU UUID `GPU-9b669a06-9156-a98e-2194-36f2fc11d7ca`. No PBS allocation is claimed; cooperative locks and20% VRAM cap. Existing jobs preserved.

Actual child process observation: `2525928       42:44 /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/runtime/stage_program.py --stage train`.

TRAIN218,266; VAL31,107. TEST87,564 is not evaluated. Original-image BCE; all21,594,159parameters trainable, BNupdates. Validation AUC selects checkpoints; calibration uses validation reals only.

Input hash/reader/path/provenance checks PASS;64TRAIN+64VALimages decoded by stratum. Native CUDA suite and two real TRAIN-only optimizer steps PASS with losses0.71599847,0.70625442. No synthetic edit is used.

| Epoch | Validation AUC (%) | Classification loss |
|---|---:|---:|
| 1 | 99.562130 | 0.15681399 |
| 2 | 99.652957 | 0.06285439 |
| 3 | 99.686517 | 0.04223681 |
| 4 | 99.710703 | 0.03132660 |
| 5 | 99.735900 | 0.02618677 |

Current validation-selected epoch: 5. Achieved VALFPR: 9.967186% (243/2438); no TESTFPR guarantee.

| Attribute | F_FPR (%) | F_DP (%) | es-AUC (%) | Worst eligible AUC (%) |
|---|---:|---:|---:|---:|
| gender | 3.089534 | 0.092091 | 99.711345 | 99.721582 |
| race | 10.699709 | 0.904330 | 99.248797 | 99.583948 |
| gender_race | 27.718613 | 1.166677 | 98.723102 | 99.508128 |
| age_bin | 5.825345 | 4.422205 | 99.504496 | 99.641719 |
| skin_bin | 5.565458 | 1.101464 | 99.423286 | 99.560860 |

Model source commit `875bce025084871438ef799ca9086985c4375672`; exact fixed FF2P10config SHA256 `db0c886527635295f0394dddbca9da1a8a3c40807609ab78f324b75b382fbee8`. No architecture/loss edits or FF++configuration substitution.

Run root: `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1`; actual training log: `logs/train.log`; history/checkpoints: `training/`; checks: `checks/`.

[Exact request, configuration, fresh checks and scope](../proposals/ff2_originals_s17_20261102_r1/README.md). Full matched Ding comparison, paired arms and newTEST evaluation remain blocked/unlaunched. Current run is original-only P, not a completed counterfactual comparison.

This finite read-only observer publishes state/epoch milestones and stops after completion, failure, process ambiguity or12h. It cannot launch training, generate images, change selection/calibration or kill jobs. Timestamps remain observations, not a monitoring guarantee.
