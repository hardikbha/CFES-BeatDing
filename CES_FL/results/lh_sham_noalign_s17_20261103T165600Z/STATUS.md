# SHAM-FIX and NO-ALIGN: observed status

2026-11-03T19:06:55.895407+00:00

Source execution commit: `ab7f36798cfa06226e17e31c2171e0c02404a216`. Independent three-epoch ablations against completed lower-LR L_H; seed17, FF++ TRAIN/VAL only. No TEST.

|Arm|PBS job|Observed scheduler state|Completed epochs|Last recorded VAL AUC %|
|---|---|---|---:|---:|
|SHAM_FIX|45361.mgmt01|R|0/3|—|
|NO_ALIGN|45361.mgmt01|R|1/3|97.873508|

### NO_ALIGN recorded epochs

|Epoch|AUC %|FNR %|Gender F_FPR|Race|Gender×race|Age|Skin|ids8|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|97.873508|5.475721|0.042719|0.101704|0.413609|0.119534|0.032643|0.336776|

Completed epochs are recorded history, not proof of a live process. H=held, Q=queued, R=running allocation; GPU checks precede training. Qualification is reused only on exact matching bytes and retains the original AI procedural-review scope. Both runs start from P50 plus the identical fresh head; neither starts from a trained L_H checkpoint.

Budget: {"initial_total_gpu_hours": 24, "consumed_gpu_hours": 16.320081049573297, "reserved_gpu_hours": 2.616532447160118}. A maximum 3 GPU-hours is reserved across the two arms.

All fairness values are fractions. Keep regressions visible. Single-seed validation findings are not a Ding TEST comparison or proof of generalization. This helper publishes only aggregate evidence and does not control execution.
