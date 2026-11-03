# SHAM-FIX and NO-ALIGN: observed status

2026-11-03T19:36:35.251120+00:00

Source execution commit: `ab7f36798cfa06226e17e31c2171e0c02404a216`. Independent three-epoch ablations against completed lower-LR L_H; seed17, FF++ TRAIN/VAL only. No TEST.

|Arm|PBS job|Observed scheduler state|Completed epochs|Last recorded VAL AUC %|
|---|---|---|---:|---:|
|SHAM_FIX|45361.mgmt01|F|3/3|98.162281|

### SHAM_FIX recorded epochs

|Epoch|AUC %|FNR %|Gender F_FPR|Race|Gender×race|Age|Skin|ids8|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|97.892568|5.442848|0.036840|0.105853|0.447757|0.117667|0.025796|0.339438|
|2|98.097466|4.818259|0.034880|0.088648|0.412280|0.139442|0.021834|0.331834|
|3|98.162281|4.494224|0.038799|0.070286|0.430919|0.144968|0.028359|0.377597|
|NO_ALIGN|45361.mgmt01|F|3/3|98.208318|

### NO_ALIGN recorded epochs

|Epoch|AUC %|FNR %|Gender F_FPR|Race|Gender×race|Age|Skin|ids8|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|97.873508|5.475721|0.042719|0.101704|0.413609|0.119534|0.032643|0.336776|
|2|98.108690|4.790082|0.040759|0.088564|0.404100|0.134924|0.033656|0.346656|
|3|98.208318|4.461351|0.044679|0.094555|0.420300|0.207825|0.035019|0.336647|

Completed epochs are recorded history, not proof of a live process. H=held, Q=queued, R=running allocation; GPU checks precede training. Qualification is reused only on exact matching bytes and retains the original AI procedural-review scope. Both runs start from P50 plus the identical fresh head; neither starts from a trained L_H checkpoint.

Budget: {"initial_total_gpu_hours": 24, "consumed_gpu_hours": 17.286613496733413, "reserved_gpu_hours": 0}. A maximum 3 GPU-hours is reserved across the two arms.

All fairness values are fractions. Keep regressions visible. Single-seed validation findings are not a Ding TEST comparison or proof of generalization. This helper publishes only aggregate evidence and does not control execution.
