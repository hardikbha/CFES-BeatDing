# Actual execution status

Observed management UTC: **2026-11-02T15:47:15.711404+00:00**.

State: **RUNNING**. Source commit `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b`.

Host `gpu01`, physical GPUs0/2 in shared pool-v2 mode; no PBS allocation or exclusive CPU/GPU ownership claimed.
Controller PID `2754578`, training torchrun PID `2754977`.

Latest recorded event: `TRAIN_STEP`, stage `classification`, epoch `3`, batch `2150/3149`, loss `0.014044276438653469`.

Eight CPU tests PASS; real-TRAIN full-native-batch two-GPU CUDA smoke PASS. 128classification images/update; 64main+64auxiliary GDA images/update; 224px; seed17;50classification→TRAIN-SFD→50corrected-GDAepochs.
Corrected differentiable class-conditioned GDA with frozen encoder BN is explicitly distinguished from the detached archived GDA and paper results. No new validation score exists until an epoch finishes. No TEST inference is attached.

Continuous private log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.

Initial launch failed before GPU claim because compute nodes lack Git; a private byte-identical copy of management /usr/bin/git fixed source verification. Its hash and command are in execution_receipt.json. Model code/settings were unchanged.

Compute-node wall clock is approximately649s behind management at launch; preserve both clocks. Elapsed timings are node-local.

Finite observation/publication only, every15min for at most72h; no training retries or new jobs.

See live_status.json for verified process observation, aggregate validation/group metrics, checkpoint hashes and completion/failure state. Review historical PROJECT_BLUEPRINT.md as a dated snapshot; this page is the execution update.
