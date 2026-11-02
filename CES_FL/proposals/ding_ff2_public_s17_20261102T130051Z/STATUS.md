# Actual execution status

Observed management UTC: **2026-11-02T20:01:06.719694+00:00**.

State: **RUNNING**. Source commit `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b`.

Host `gpu01`, physical GPUs0/2 in shared pool-v2 mode; no PBS allocation or exclusive CPU/GPU ownership claimed.
Controller PID `2754578`, training torchrun PID `2754977`.

Latest recorded event: `TRAIN_STEP`, stage `classification`, epoch `8`, batch `1950/3149`, loss `0.011451894417405128`.

Eight CPU tests PASS; real-TRAIN full-native-batch two-GPU CUDA smoke PASS. 128classification images/update; 64main+64auxiliary GDA images/update; 224px; seed17;50classification→TRAIN-SFD→50corrected-GDAepochs.
Corrected differentiable class-conditioned GDA with frozen encoder BN is explicitly distinguished from the detached archived GDA and paper results. Values below are completed validation epochs, not TEST scores. No TEST inference is attached.

Continuous private log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.

Initial launch failed before GPU claim because compute nodes lack Git; a private byte-identical copy of management /usr/bin/git fixed source verification. Its hash and command are in execution_receipt.json. Model code/settings were unchanged.

Compute-node wall clock is approximately649s behind management at launch; preserve both clocks. Elapsed timings are node-local.

Finite observation/publication only, every15min for at most72h; no training retries or new jobs.

See live_status.json for verified process observation, aggregate validation/group metrics, checkpoint hashes and completion/failure state. Review historical PROJECT_BLUEPRINT.md as a dated snapshot; this page is the execution update.

## classification: 7 completed validation epochs

Best validation AUC: 98.4244% at epoch6 (fixed AUC-only selection).

| Epoch | Train CE | VAL AUC (%) | Worst gender×race AUC (%) | FPR gap (pp) | F_FPR (Ding scale) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.260279 | 97.0067 | 96.4288 | 9.3780 | 21.1840 |
| 2 | 0.121754 | 97.8417 | 96.8216 | 8.6683 | 23.6116 |
| 3 | 0.064228 | 97.9288 | 96.5497 | 10.8264 | 32.6147 |
| 4 | 0.035999 | 98.1493 | 97.0077 | 11.2456 | 24.6681 |
| 5 | 0.022846 | 97.7694 | 95.7756 | 10.5395 | 27.6514 |
| 6 | 0.016415 | 98.4244 | 97.2242 | 9.7682 | 25.2413 |
| 7 | 0.013938 | 97.5974 | 95.6088 | 8.3786 | 21.4778 |

## gda_corrected: 0 completed validation epochs

No completed validation epoch in this stage yet.
