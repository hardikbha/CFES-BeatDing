# SHAM-FIX and NO-ALIGN: observed status

2026-11-03T18:37:07.334330+00:00

Source execution commit: `f4e1f72723afd7270b0f45db7b6142433215d500`. Independent three-epoch ablations against completed lower-LR L_H; seed17, FF++ TRAIN/VAL only. No TEST.

|Arm|PBS job|Observed scheduler state|Completed epochs|Last recorded VAL AUC %|
|---|---|---|---:|---:|
|SHAM_FIX|45352.mgmt01|H|0/3|—|
|NO_ALIGN|45352.mgmt01|H|0/3|—|

Completed epochs are recorded history, not proof of a live process. H=held, Q=queued, R=running allocation; GPU checks precede training. Qualification is reused only on exact matching bytes and retains the original AI procedural-review scope. Both runs start from P50 plus the identical fresh head; neither starts from a trained L_H checkpoint.

Budget: {"initial_total_gpu_hours": 24, "consumed_gpu_hours": 15.936613496733413, "reserved_gpu_hours": 3.0}. A maximum 3 GPU-hours is reserved across the two arms.

All fairness values are fractions. Keep regressions visible. Single-seed validation findings are not a Ding TEST comparison or proof of generalization. This helper publishes only aggregate evidence and does not control execution.
